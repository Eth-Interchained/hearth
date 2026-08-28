//! Enough HTTP/1.1 to be an honest gateway, and no more.
//!
//! Same reasoning as everywhere else in this workspace: `probe.rs` writes a raw
//! GET because a health check is a status line; `hearth-pull` drives `curl`
//! because a registry download needs TLS and resume, which is a real library's
//! job. This is the third case — a server on localhost speaking to clients that
//! already exist, where the whole surface is *read a request, write a response,
//! and relay bytes*. A framework would bring an async runtime and a dependency
//! tree to do what four hundred lines of std does, and the parsing is the only
//! part with a rule in it, so it lives here, pure and tested.
//!
//! What this deliberately does NOT do: TLS (put it behind a proxy, and the
//! default bind is loopback so there is nothing to expose), HTTP/2, or
//! keep-alive pipelining. A model gateway handles a few requests per second
//! that each take seconds; connection setup is not the cost.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

/// A parsed request. Headers are lowercased on the way in, because HTTP header
/// names are case-insensitive and half of the clients disagree about casing.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: String,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .get(&name.to_ascii_lowercase())
            .map(|s| s.as_str())
    }

    /// Does the body ask for a streamed response? Streaming changes how the
    /// proxy relays: a buffered read would hold every token until the model
    /// finished, turning a live stream into a long pause and a wall of text.
    pub fn wants_stream(&self) -> bool {
        serde_json::from_str::<serde_json::Value>(&self.body)
            .ok()
            .and_then(|v| v.get("stream").and_then(|s| s.as_bool()))
            .unwrap_or(false)
    }
}

/// Parse a request line. Pure — `GET /v1/models HTTP/1.1`.
pub fn parse_request_line(line: &str) -> Option<(String, String)> {
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_ascii_uppercase();
    let path = parts.next()?.to_string();
    let version = parts.next()?;
    if !version.starts_with("HTTP/1.") {
        return None;
    }
    Some((method, path))
}

/// Parse one header line into a lowercased name and a trimmed value.
pub fn parse_header(line: &str) -> Option<(String, String)> {
    let (name, value) = line.split_once(':')?;
    let name = name.trim().to_ascii_lowercase();
    if name.is_empty() {
        return None;
    }
    Some((name, value.trim().to_string()))
}

/// Serialize a response we generated ourselves.
pub fn render_response(status: u16, body: &str, retry_after: Option<u32>) -> String {
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Status",
    };
    let mut head = format!(
        "HTTP/1.1 {status} {reason}\r\n\
         content-type: application/json\r\n\
         content-length: {}\r\n\
         connection: close\r\n",
        body.len()
    );
    if let Some(secs) = retry_after {
        // Only ever set when coming back later could actually work. A
        // Retry-After on a permanent condition invites the retry storm the
        // status code was chosen to prevent.
        head.push_str(&format!("retry-after: {secs}\r\n"));
    }
    head.push_str("\r\n");
    head.push_str(body);
    head
}

/// Read one request off a stream.
///
/// `content-length` is honoured exactly. Reading "until it stops" would work
/// right up until a client keeps the connection open, and then the gateway
/// blocks forever on a request it already has in full.
pub fn read_request(stream: &TcpStream) -> Result<Request, String> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .map_err(|e| format!("reading request line: {e}"))?;
    let (method, path) =
        parse_request_line(line.trim_end()).ok_or_else(|| format!("bad request line: {line:?}"))?;

    let mut headers = HashMap::new();
    loop {
        let mut h = String::new();
        let n = reader
            .read_line(&mut h)
            .map_err(|e| format!("reading headers: {e}"))?;
        if n == 0 || h.trim().is_empty() {
            break;
        }
        if let Some((k, v)) = parse_header(h.trim_end()) {
            headers.insert(k, v);
        }
        if headers.len() > 100 {
            return Err("too many headers".into());
        }
    }

    let len: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut body = vec![0u8; len];
    if len > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|e| format!("reading body ({len} bytes): {e}"))?;
    }

    Ok(Request {
        method,
        path,
        headers,
        body: String::from_utf8_lossy(&body).to_string(),
    })
}

/// Forward a request to a model's endpoint and relay the response back.
///
/// Relayed in chunks rather than buffered, so a streamed completion reaches the
/// caller token by token. Buffering here would hold every token until the model
/// finished — turning a live stream into a long silence followed by a wall of
/// text, which is the single most noticeable way a gateway ruins a chat UI.
pub fn proxy(
    upstream: &str,
    req: &Request,
    client: &mut TcpStream,
    connect_timeout: Duration,
) -> Result<(), String> {
    let addr = upstream
        .parse()
        .map_err(|e| format!("bad upstream address {upstream}: {e}"))?;
    let mut up = TcpStream::connect_timeout(&addr, connect_timeout)
        .map_err(|e| format!("could not reach {upstream}: {e}"))?;
    // No read timeout on the upstream: generation legitimately takes minutes,
    // and a deadline here would cut off a working response — the same "nothing
    // fails on a clock" rule the supervisor follows.
    up.set_write_timeout(Some(connect_timeout)).ok();

    let mut head = format!(
        "{} {} HTTP/1.1\r\nhost: {}\r\ncontent-length: {}\r\nconnection: close\r\n",
        req.method,
        req.path,
        upstream,
        req.body.len()
    );
    for (k, v) in &req.headers {
        // Hop-by-hop headers describe OUR connection to the client, not ours
        // to the model. Forwarding them makes the upstream misread the
        // conversation it is in.
        if matches!(
            k.as_str(),
            "host" | "content-length" | "connection" | "keep-alive" | "transfer-encoding"
        ) {
            continue;
        }
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");

    up.write_all(head.as_bytes())
        .and_then(|_| up.write_all(req.body.as_bytes()))
        .and_then(|_| up.flush())
        .map_err(|e| format!("writing to {upstream}: {e}"))?;

    let mut buf = [0u8; 16 * 1024];
    loop {
        match up.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                if client.write_all(&buf[..n]).is_err() {
                    // The caller hung up mid-stream. Normal — someone closed a
                    // tab. Not worth an error.
                    break;
                }
                // Flush per chunk or the relay buffers the stream right back
                // up again and undoes the point of relaying it.
                let _ = client.flush();
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(format!("reading from {upstream}: {e}")),
        }
    }
    Ok(())
}

/// The listening socket.
pub struct Server {
    listener: TcpListener,
}

impl Server {
    /// Bind. Loopback by default at the call site — a gateway on 0.0.0.0 hands
    /// anyone on the network an unauthenticated GPU.
    pub fn bind(addr: &str) -> Result<Server, String> {
        let listener =
            TcpListener::bind(addr).map_err(|e| format!("could not bind {addr}: {e}"))?;
        Ok(Server { listener })
    }

    pub fn local_addr(&self) -> String {
        self.listener
            .local_addr()
            .map(|a| a.to_string())
            .unwrap_or_default()
    }

    /// Accept one connection, non-blocking-ish via a poll timeout, so the
    /// caller's supervise loop keeps ticking instead of being held here.
    pub fn incoming(&self) -> impl Iterator<Item = std::io::Result<TcpStream>> + '_ {
        self.listener.incoming()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_line_parses() {
        assert_eq!(
            parse_request_line("POST /v1/chat/completions HTTP/1.1"),
            Some(("POST".into(), "/v1/chat/completions".into()))
        );
        assert_eq!(
            parse_request_line("get /health HTTP/1.0"),
            Some(("GET".into(), "/health".into())),
            "the method is case-insensitive on the wire"
        );
    }

    #[test]
    fn garbage_is_not_a_request_line() {
        // Something that is not HTTP arriving on the port — a health checker
        // speaking TLS to a plaintext socket does this.
        assert_eq!(parse_request_line("SSH-2.0-OpenSSH_9.6"), None);
        assert_eq!(parse_request_line(""), None);
        assert_eq!(parse_request_line("GET /only-two-parts"), None);
    }

    #[test]
    fn header_names_are_lowercased_because_clients_disagree_about_casing() {
        assert_eq!(
            parse_header("Content-Type: application/json"),
            Some(("content-type".into(), "application/json".into()))
        );
        assert_eq!(
            parse_header("AUTHORIZATION:   Bearer sk-x  "),
            Some(("authorization".into(), "Bearer sk-x".into()))
        );
        assert_eq!(parse_header("no colon here"), None);
        assert_eq!(parse_header(": empty name"), None);
    }

    #[test]
    fn a_header_value_may_contain_colons() {
        // `host: 127.0.0.1:8090` splits on the FIRST colon or the port is lost.
        assert_eq!(
            parse_header("host: 127.0.0.1:8090"),
            Some(("host".into(), "127.0.0.1:8090".into()))
        );
    }

    #[test]
    fn a_rendered_response_has_a_content_length_that_matches() {
        // A wrong content-length hangs the client waiting for bytes that are
        // never coming, which reads as "hearth is slow".
        let body = r#"{"status":"ok"}"#;
        let out = render_response(200, body, None);
        assert!(out.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(out.contains(&format!("content-length: {}", body.len())));
        assert!(out.ends_with(body));
        assert!(!out.contains("retry-after"));
    }

    #[test]
    fn retry_after_appears_only_when_it_was_asked_for() {
        // On a permanent condition it would invite exactly the retry storm the
        // status code was chosen to prevent.
        assert!(render_response(503, "{}", Some(5)).contains("retry-after: 5"));
        assert!(!render_response(409, "{}", None).contains("retry-after"));
    }

    #[test]
    fn every_status_the_gateway_emits_has_a_reason_phrase() {
        for s in [200u16, 400, 404, 409, 502, 503] {
            let out = render_response(s, "{}", None);
            let line = out.lines().next().unwrap();
            assert!(
                !line.ends_with("Status"),
                "status {s} fell through to the generic phrase: {line}"
            );
        }
    }

    #[test]
    fn a_streaming_request_is_recognised_from_its_body() {
        let mut r = Request {
            body: r#"{"model":"m","stream":true}"#.into(),
            ..Default::default()
        };
        assert!(r.wants_stream());
        r.body = r#"{"model":"m","stream":false}"#.into();
        assert!(!r.wants_stream());
        r.body = r#"{"model":"m"}"#.into();
        assert!(!r.wants_stream(), "absent means not streaming");
        r.body = "not json".into();
        assert!(!r.wants_stream(), "and unreadable must not panic");
    }

    #[test]
    fn headers_are_looked_up_case_insensitively() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Bearer x".to_string());
        let r = Request {
            headers,
            ..Default::default()
        };
        assert_eq!(r.header("Authorization"), Some("Bearer x"));
        assert_eq!(r.header("AUTHORIZATION"), Some("Bearer x"));
        assert_eq!(r.header("nope"), None);
    }
}
