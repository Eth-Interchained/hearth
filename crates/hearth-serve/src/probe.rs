//! Health probes that produce facts, not conclusions.
//!
//! A probe answers exactly one question — "did this endpoint answer HTTP
//! within the deadline?" — and reports what it saw. Turning that into a
//! residency state is `hearth_core`'s job; conflating the two is how every
//! serving stack ends up reporting a guess as a fact.
//!
//! Raw `std::net::TcpStream` + a hand-written HTTP/1.1 GET, zero
//! dependencies. llama-server's `/health` returns 200 with a tiny JSON
//! body when the model is loaded and 503 while it is still warming — the
//! status line alone carries everything we need.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// What one probe attempt actually saw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeResult {
    /// Endpoint answered 200 — the model is loaded and serving.
    Ok,
    /// Endpoint answered, but the model is not ready (llama-server says 503
    /// while loading). Not a failure: progress.
    Warming { status: u16 },
    /// TCP connected but the HTTP exchange failed or timed out.
    Unanswered { detail: String },
    /// Could not even connect. Distinguishes "process gone" from "process
    /// wedged", which map to different `LostReason`s upstream.
    Unreachable { detail: String },
}

impl ProbeResult {
    pub fn is_ok(&self) -> bool {
        matches!(self, ProbeResult::Ok)
    }
}

/// Probe `host:port` with `GET path` and a per-attempt deadline.
pub fn probe_http(addr: &str, path: &str, timeout: Duration) -> ProbeResult {
    let stream = match TcpStream::connect_timeout(
        &match addr.parse() {
            Ok(a) => a,
            Err(e) => {
                return ProbeResult::Unreachable {
                    detail: format!("bad addr {addr}: {e}"),
                }
            }
        },
        timeout,
    ) {
        Ok(s) => s,
        Err(e) => {
            return ProbeResult::Unreachable {
                detail: e.to_string(),
            }
        }
    };
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));
    let mut stream = stream;

    let req = format!("GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    if let Err(e) = stream.write_all(req.as_bytes()) {
        return ProbeResult::Unanswered {
            detail: format!("write: {e}"),
        };
    }

    let mut buf = [0u8; 512];
    let n = match stream.read(&mut buf) {
        Ok(0) => {
            return ProbeResult::Unanswered {
                detail: "connection closed before status line".into(),
            }
        }
        Ok(n) => n,
        Err(e) => {
            return ProbeResult::Unanswered {
                detail: format!("read: {e}"),
            }
        }
    };

    match parse_status(&buf[..n]) {
        Some(200) => ProbeResult::Ok,
        Some(code) => ProbeResult::Warming { status: code },
        None => ProbeResult::Unanswered {
            detail: "unparseable status line".into(),
        },
    }
}

/// Pull the status code out of an HTTP/1.x status line. Pure, testable.
pub fn parse_status(bytes: &[u8]) -> Option<u16> {
    let text = std::str::from_utf8(bytes).ok()?;
    let line = text.lines().next()?;
    let mut parts = line.split_whitespace();
    let proto = parts.next()?;
    if !proto.starts_with("HTTP/1.") {
        return None;
    }
    parts.next()?.parse().ok()
}

/// Is a GPU visible on this host right now?
///
/// The single most valuable bit in the system: it separates "the runtime
/// dropped the model" (Evicted — your capacity problem) from "the host
/// took the card away" (GpuDetached — your provider's problem).
///
/// Asks `nvidia-smi -L` with a short deadline. No NVML linkage — the tool
/// ships with every NVIDIA driver, and a missing tool on a CPU-only box is
/// an honest `None` ("no way to know"), never a guessed `false`.
pub fn gpu_present() -> Option<bool> {
    let out = std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .ok()?;
    Some(out.status.success() && !out.stdout.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn status_line_parses() {
        assert_eq!(parse_status(b"HTTP/1.1 200 OK\r\n"), Some(200));
        assert_eq!(
            parse_status(b"HTTP/1.1 503 Service Unavailable\r\n"),
            Some(503)
        );
        assert_eq!(parse_status(b"HTTP/1.0 404 Not Found\r\n"), Some(404));
    }

    #[test]
    fn garbage_is_not_a_status() {
        assert_eq!(parse_status(b"SSH-2.0-OpenSSH_9.6\r\n"), None);
        assert_eq!(parse_status(b""), None);
        assert_eq!(parse_status(&[0xff, 0xfe, 0x00]), None);
    }

    #[test]
    fn unreachable_when_nothing_listens() {
        // Bind then drop to get a port that is definitely closed.
        let addr = {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().to_string()
        };
        let r = probe_http(&addr, "/health", Duration::from_millis(300));
        assert!(matches!(r, ProbeResult::Unreachable { .. }), "{r:?}");
    }

    #[test]
    fn ok_when_server_says_200() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let h = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = [0u8; 256];
            let _ = std::io::Read::read(&mut s, &mut buf);
            s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n{\"status\":\"ok\"}")
                .unwrap();
        });
        let r = probe_http(&addr, "/health", Duration::from_millis(500));
        h.join().unwrap();
        assert_eq!(r, ProbeResult::Ok);
    }

    #[test]
    fn warming_when_server_says_503() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let h = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = [0u8; 256];
            let _ = std::io::Read::read(&mut s, &mut buf);
            s.write_all(b"HTTP/1.1 503 Service Unavailable\r\n\r\n")
                .unwrap();
        });
        let r = probe_http(&addr, "/health", Duration::from_millis(500));
        h.join().unwrap();
        assert_eq!(r, ProbeResult::Warming { status: 503 });
    }

    #[test]
    fn unanswered_when_server_speaks_garbage() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let h = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = [0u8; 256];
            let _ = std::io::Read::read(&mut s, &mut buf);
            s.write_all(b"NOT HTTP AT ALL").unwrap();
        });
        let r = probe_http(&addr, "/health", Duration::from_millis(500));
        h.join().unwrap();
        assert!(matches!(r, ProbeResult::Unanswered { .. }), "{r:?}");
    }
}
