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
/// Asks `nvidia-smi -L`. No NVML linkage — the tool ships with every
/// NVIDIA driver, and a missing tool on a CPU-only box is an honest `None`
/// ("no way to know"), never a guessed `false`.
///
/// The answer is read from the OUTPUT, not from the exit status, because
/// `nvidia-smi -L` is a listing command: on a host whose card has been
/// reclaimed it prints `No devices were found` and can still exit 0. Judging
/// by `success() && !stdout.is_empty()` calls that a present GPU — which
/// turns a provider's detach into `Evicted`, and `Evicted` is
/// `is_operator_fault() == true`. That is the exact misattribution this
/// crate exists to prevent, so it is decided by `read_gpu_listing` and
/// tested there.
pub fn gpu_present() -> Option<bool> {
    let out = std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .ok()?;
    read_gpu_listing(
        &String::from_utf8_lossy(&out.stdout),
        &String::from_utf8_lossy(&out.stderr),
    )
}

/// Decide presence from what `nvidia-smi -L` actually said. Pure, because
/// this is the highest-stakes branch in the system and it must be assertable
/// without a GPU.
///
/// Three answers, and the third is the point:
///
///   * `Some(true)`  — a device was listed. Observed presence.
///   * `Some(false)` — the driver said, in words, that there are none.
///     Observed ABSENCE. Only this exonerates an operator.
///   * `None`        — we could not tell. The caller assumes present, so an
///     unreadable probe counts AGAINST us rather than clearing a fault that
///     may have been real. Same rule as an unreadable loss reason defaulting
///     to `Evicted` in the spine: never let a parse failure hand out an
///     alibi.
pub fn read_gpu_listing(stdout: &str, stderr: &str) -> Option<bool> {
    let said = format!("{stdout}\n{stderr}").to_ascii_lowercase();

    // Positively observed absence, whatever the exit status claims.
    if said.contains("no devices were found") || said.contains("no devices found") {
        return Some(false);
    }

    // A driver that cannot talk to the card at all. The card being
    // unreachable is not the operator over-committing VRAM, so absence is
    // the honest reading — but only when the driver says WHY, rather than
    // when we simply failed to parse something.
    if said.contains("couldn't communicate with the nvidia driver")
        || said.contains("driver/library version mismatch")
        || said.contains("nvidia-smi has failed because it couldn't")
    {
        return Some(false);
    }

    // A listing looks like `GPU 0: NVIDIA RTX A6000 (UUID: GPU-…)`.
    if stdout
        .lines()
        .any(|l| l.trim_start().to_ascii_uppercase().starts_with("GPU "))
    {
        return Some(true);
    }

    // Said nothing we recognise. On a `-L` that probably means nothing was
    // listed — but "probably" is not an observation, and this is the branch
    // that would hand out an unearned alibi. Unknown, deliberately.
    //
    // The exit status is not consulted anywhere above, and that is the fix:
    // `-L` is a listing command whose status says whether the TOOL ran, not
    // whether a CARD is there. Reading presence off `success()` is what
    // billed a provider's detach to the operator.
    None
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

    // -- the one boolean -------------------------------------------------
    //
    // These are the highest-stakes assertions in the repo. `gpu_present`
    // decides whether a cold model is `Evicted` (the operator over-committed)
    // or `GpuDetached` (the host took the card), and `LostReason::
    // is_operator_fault` reads that straight out. Getting it backwards scores
    // an operator down for someone else's decision, which is the thing hearth
    // was built to stop.

    #[test]
    fn a_listed_device_is_observed_presence() {
        let out = "GPU 0: NVIDIA RTX A6000 (UUID: GPU-1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d)\n";
        assert_eq!(read_gpu_listing(out, ""), Some(true));
    }

    #[test]
    fn several_devices_still_read_as_present() {
        let out =
            "GPU 0: NVIDIA RTX A6000 (UUID: GPU-aaa)\nGPU 1: NVIDIA RTX A6000 (UUID: GPU-bbb)\n";
        assert_eq!(read_gpu_listing(out, ""), Some(true));
    }

    #[test]
    fn no_devices_were_found_is_observed_absence_even_on_a_zero_exit() {
        // THE regression. `nvidia-smi -L` is a listing command: when the host
        // has reclaimed the card it prints this and can still exit 0. The old
        // rule was `status.success() && !stdout.is_empty()`, which is
        // `true && true` here — a detached GPU reported as present, so the
        // loss lands as Evicted and the operator wears a fault that was the
        // provider's. Content decides, not the exit code.
        assert_eq!(read_gpu_listing("No devices were found\n", ""), Some(false));
    }

    #[test]
    fn the_old_formula_would_have_failed_this_and_that_is_the_point() {
        let stdout = "No devices were found\n";
        let old_answer = /* status.success() && */ !stdout.is_empty();
        assert!(old_answer, "the old rule said: GPU present");
        assert_eq!(
            read_gpu_listing(stdout, ""),
            Some(false),
            "the new rule says: the card is gone, and that is what happened"
        );
    }

    #[test]
    fn a_driver_that_cannot_reach_the_card_is_not_the_operators_fault() {
        let stderr = "NVIDIA-SMI has failed because it couldn't communicate with \
                      the NVIDIA driver. Make sure that the latest NVIDIA driver \
                      is installed and running.\n";
        assert_eq!(read_gpu_listing("", stderr), Some(false));
    }

    #[test]
    fn a_version_mismatch_is_the_hosts_problem_too() {
        assert_eq!(
            read_gpu_listing(
                "",
                "Failed to initialize NVML: Driver/library version mismatch\n"
            ),
            Some(false)
        );
    }

    #[test]
    fn output_we_cannot_read_is_unknown_and_unknown_never_exonerates() {
        // The caller does `gpu_present().unwrap_or(true)`, so None means
        // "assume the card was there" — the loss counts against us. That is
        // deliberate: absence has to be positively observed, or a flaky parse
        // hands out an alibi. Same rule as an unreadable loss reason
        // defaulting to Evicted in the spine.
        assert_eq!(read_gpu_listing("something unexpected\n", ""), None);
        assert_eq!(read_gpu_listing("", ""), None);
        assert_eq!(read_gpu_listing("\n\n", ""), None);
    }

    #[test]
    fn the_unknown_answer_resolves_to_operator_fault_at_the_call_site() {
        // Asserting the composition, not just the parse — the guarantee only
        // holds if the caller's default agrees with the parser's intent.
        use hearth_core::LostReason;
        let assumed_present = read_gpu_listing("garbage", "").unwrap_or(true);
        assert!(assumed_present);
        assert!(
            LostReason::Evicted.is_operator_fault(),
            "and present-with-a-cold-model is ours to answer for"
        );
        assert!(
            !LostReason::GpuDetached.is_operator_fault(),
            "while a detach never is",
        );
    }

    #[test]
    fn case_and_whitespace_do_not_change_the_verdict() {
        assert_eq!(
            read_gpu_listing("  no devices were found  ", ""),
            Some(false)
        );
        assert_eq!(read_gpu_listing("NO DEVICES WERE FOUND", ""), Some(false));
        assert_eq!(
            read_gpu_listing("   gpu 0: NVIDIA RTX A6000 (UUID: GPU-x)", ""),
            Some(true)
        );
    }

    #[test]
    fn absence_beats_a_stray_gpu_word_elsewhere_in_the_output() {
        // If the driver says there are none, a line that merely mentions GPUs
        // must not talk us back into presence.
        let out = "No devices were found\nGPU support: compiled\n";
        assert_eq!(read_gpu_listing(out, ""), Some(false));
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
