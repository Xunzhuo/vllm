use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;
use axum::http::{HeaderName, HeaderValue, Method};
use serde::Serialize;
use serde_json::Value;
use vllm_chat::{ChatTemplateContentFormatOption, ParserSelection, RendererSelection};
use vllm_engine_core_client::{CoordinatorMode as EngineCoreCoordinatorMode, TransportMode};

/// How the HTTP server obtains its listening socket.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum HttpListenerMode {
    /// Bind a fresh TCP listener on the given host/port.
    BindTcp { host: String, port: u16 },
    /// Bind a fresh Unix domain listener on the given filesystem path.
    BindUnix { path: String },
    /// Adopt an already-open listening socket inherited from a supervisor
    /// process.
    InheritedFd { fd: i32 },
}

/// Which coordinator implementation should be active when one is present for a
/// frontend client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum CoordinatorMode {
    /// Do not run a coordinator at all.
    None,
    /// Run the Rust in-process coordinator for managed `serve` deployments, if
    /// there are multiple engines and the model is MoE.
    MaybeInProc,
    /// Connect to an external coordinator owned by another process.
    External { address: String },
}

/// HTTP CORS behavior for the OpenAI-compatible server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CorsConfig {
    /// Whether `Access-Control-Allow-Credentials` should be set to `true`.
    pub allow_credentials: bool,
    /// Allowed origins. `["*"]` means any origin.
    pub allowed_origins: Vec<String>,
    /// Allowed HTTP methods. `["*"]` means any method.
    pub allowed_methods: Vec<String>,
    /// Allowed request headers. `["*"]` means any header.
    pub allowed_headers: Vec<String>,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            allow_credentials: false,
            allowed_origins: vec!["*".to_string()],
            allowed_methods: vec!["*".to_string()],
            allowed_headers: vec!["*".to_string()],
        }
    }
}

impl CorsConfig {
    /// Validate the configured CORS values without constructing middleware.
    pub fn validate(&self) -> Result<()> {
        self.allowed_origin_values()?;
        self.allowed_method_values()?;
        self.allowed_header_values()?;
        Ok(())
    }

    pub(crate) fn allows_any_origin(&self) -> bool {
        is_wildcard_list(&self.allowed_origins)
    }

    pub(crate) fn allows_any_method(&self) -> bool {
        is_wildcard_list(&self.allowed_methods)
    }

    pub(crate) fn allows_any_header(&self) -> bool {
        is_wildcard_list(&self.allowed_headers)
    }

    pub(crate) fn allowed_origin_values(&self) -> Result<Vec<HeaderValue>> {
        if self.allows_any_origin() {
            return Ok(Vec::new());
        }

        self.allowed_origins
            .iter()
            .map(|origin| {
                HeaderValue::from_str(origin)
                    .map_err(|error| anyhow::anyhow!("invalid CORS origin `{origin}`: {error}"))
            })
            .collect()
    }

    pub(crate) fn allowed_method_values(&self) -> Result<Vec<Method>> {
        if self.allows_any_method() {
            return Ok(Vec::new());
        }

        self.allowed_methods
            .iter()
            .map(|method| {
                Method::from_bytes(method.as_bytes())
                    .map_err(|error| anyhow::anyhow!("invalid CORS method `{method}`: {error}"))
            })
            .collect()
    }

    pub(crate) fn allowed_header_values(&self) -> Result<Vec<HeaderName>> {
        if self.allows_any_header() {
            return Ok(Vec::new());
        }

        self.allowed_headers
            .iter()
            .map(|header| {
                HeaderName::from_bytes(header.as_bytes())
                    .map_err(|error| anyhow::anyhow!("invalid CORS header `{header}`: {error}"))
            })
            .collect()
    }
}

fn is_wildcard_list(values: &[String]) -> bool {
    values.len() == 1 && values.first().is_some_and(|value| value == "*")
}

/// Normalized runtime configuration for the minimal OpenAI-compatible server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Config {
    /// Frontend-to-engine transport setup.
    pub transport_mode: TransportMode,
    /// Requested frontend-side coordinator behavior.
    pub coordinator_mode: CoordinatorMode,
    /// Backend model identifier used for engine-core loading.
    pub model: String,
    /// Model name(s) exposed to clients via the OpenAI API. When non-empty,
    /// the first entry is used as the primary ID in responses and all entries
    /// are accepted in requests. When empty, falls back to `model`.
    pub served_model_name: Vec<String>,
    /// HTTP listener setup.
    pub listener_mode: HttpListenerMode,
    /// Tool-call parser selection.
    pub tool_call_parser: ParserSelection,
    /// Reasoning parser selection.
    pub reasoning_parser: ParserSelection,
    /// Chat renderer selection.
    pub renderer: RendererSelection,
    /// Server-default chat template override, as a file path or inline
    /// template.
    pub chat_template: Option<String>,
    /// Server-default keyword arguments merged into every chat-template render.
    pub default_chat_template_kwargs: Option<HashMap<String, Value>>,
    /// How to serialize `message.content` for chat-template rendering.
    pub chat_template_content_format: ChatTemplateContentFormatOption,
    /// Log a summary line for each completed request.
    pub enable_log_requests: bool,
    /// When `true`, set `X-Request-Id` on every HTTP response.
    pub enable_request_id_headers: bool,
    /// When `true`, suppress periodic stats logging (throughput, queue depth,
    /// cache usage).
    pub disable_log_stats: bool,
    /// HTTP CORS behavior.
    pub cors: CorsConfig,
    /// TCP port for the gRPC Generate service. When `None`, no gRPC server is
    /// started.
    pub grpc_port: Option<u16>,
    /// Maximum time to wait for active HTTP/gRPC requests to drain on shutdown.
    pub shutdown_timeout: Duration,
}

impl Config {
    /// Validate frontend configuration that can be checked before engine
    /// startup.
    pub fn validate(&self) -> Result<()> {
        vllm_chat::validate_parser_overrides(&self.tool_call_parser, &self.reasoning_parser)?;
        self.cors.validate()?;

        Ok(())
    }

    /// Return the number of engines implied by the configured transport mode.
    pub fn engine_count(&self) -> usize {
        match &self.transport_mode {
            TransportMode::HandshakeOwner { engine_count, .. }
            | TransportMode::Bootstrapped { engine_count, .. } => *engine_count,
        }
    }

    /// Resolve the effective coordinator mode.
    pub fn effective_coordinator_mode(
        &self,
        model_is_moe: bool,
    ) -> Option<EngineCoreCoordinatorMode> {
        match &self.coordinator_mode {
            CoordinatorMode::None => None,
            CoordinatorMode::MaybeInProc => {
                if model_is_moe && self.engine_count() > 1 {
                    Some(EngineCoreCoordinatorMode::InProc)
                } else {
                    None
                }
            }
            CoordinatorMode::External { address } => Some(EngineCoreCoordinatorMode::External {
                address: address.clone(),
            }),
        }
    }
}
