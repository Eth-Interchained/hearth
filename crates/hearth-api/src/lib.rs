//! One OpenAI-compatible port in front of a hearth fleet.
//!
//! The proxying is the boring part. What makes this worth having is that when
//! hearth *cannot* serve a request, the HTTP response says which of four very
//! different things happened — still loading, evicted, GPU reclaimed, or will
//! never fit — and whether retrying here could ever work. See [`gateway`].

pub mod gateway;
pub mod http;

pub use gateway::{decide, Decision, Endpoint, Response};
pub use http::{Request, Server};
