//! HTTP by driving `curl`, not by linking one.
//!
//! This is the same call the rest of the workspace makes about dependencies,
//! applied to a harder case. `probe.rs` writes a raw HTTP GET over a TcpStream
//! because a localhost health check is a status line and nothing else. A
//! registry download is not that: it needs TLS, redirects to a CDN,
//! resume-after-interrupt, and retry with backoff. Writing those is not
//! declining a dependency — it is becoming one, badly.
//!
//! So: `curl`, which is already on every machine that can run a model, is
//! maintained by people who do this full time, and gets all four of those
//! right. We already shell out to `nvidia-smi` and to `llama-server`; this is
//! the same bargain with a much better-tested binary.
//!
//! What lives here is the argv and the error reading — the parts with a rule
//! in them — pure and tested without touching the network.

use std::path::{Path, PathBuf};
use std::process::Command;

/// A curl invocation we are about to make.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub url: String,
    pub headers: Vec<(String, String)>,
    /// Where the body goes. `None` means capture it as a string.
    pub dest: Option<PathBuf>,
    /// Resume a partial file rather than starting over. Only meaningful with
    /// `dest`, and the reason a 20 GiB download that died at 19 GiB is not a
    /// reason to start again.
    pub resume: bool,
    pub connect_timeout_secs: u32,
    /// Retries for transient failures. Not a deadline on the transfer: a 40 GiB
    /// model over a slow link is slow, not broken, and the supervisor's whole
    /// argument is that nothing should fail on a clock.
    pub retries: u32,
}

impl Request {
    pub fn get(url: impl Into<String>) -> Request {
        Request {
            url: url.into(),
            headers: Vec::new(),
            dest: None,
            resume: false,
            connect_timeout_secs: 20,
            retries: 3,
        }
    }

    pub fn header(mut self, k: impl Into<String>, v: impl Into<String>) -> Request {
        self.headers.push((k.into(), v.into()));
        self
    }

    pub fn to_file(mut self, dest: impl Into<PathBuf>) -> Request {
        self.dest = Some(dest.into());
        self
    }

    pub fn resuming(mut self) -> Request {
        self.resume = true;
        self
    }

    /// The exact arguments. Pure, so the command can be asserted on without a
    /// network — a wrong flag here fails in a way that looks like a bad model.
    pub fn argv(&self) -> Vec<String> {
        let mut v: Vec<String> = vec![
            // Fail on 4xx/5xx instead of writing the error page to disk as if
            // it were a model. Without this a 404 becomes a corrupt GGUF.
            "--fail".into(),
            // Registries redirect to a CDN. HuggingFace always does.
            "--location".into(),
            "--silent".into(),
            "--show-error".into(),
            "--connect-timeout".into(),
            self.connect_timeout_secs.to_string(),
            // Exponential backoff, and it honours Retry-After. A hand-rolled
            // sleep loop guesses at a delay the server already told us.
            "--retry".into(),
            self.retries.to_string(),
            "--retry-connrefused".into(),
        ];
        for (k, val) in &self.headers {
            v.push("--header".into());
            v.push(format!("{k}: {val}"));
        }
        if let Some(dest) = &self.dest {
            if self.resume {
                // `-C -` asks curl to work out the offset itself.
                v.push("--continue-at".into());
                v.push("-".into());
            }
            v.push("--output".into());
            v.push(dest.display().to_string());
        }
        // `--` so a URL that begins with a dash is a URL and not a flag.
        v.push("--".into());
        v.push(self.url.clone());
        v
    }
}

#[derive(Debug)]
pub struct CurlError(pub String);

impl std::fmt::Display for CurlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for CurlError {}

/// Is curl usable here? Checked once, so a missing binary is one clear error
/// rather than one confusing error per blob.
pub fn curl_available() -> bool {
    Command::new("curl")
        .arg("--version")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Turn curl's exit code and stderr into something a person can act on.
///
/// Pure and tested, because "it failed" is the least useful thing a downloader
/// can say. The mapping matters: 22 is the server refusing us and 6 is DNS,
/// and those want completely different responses from the operator.
pub fn explain_failure(code: Option<i32>, stderr: &str) -> String {
    let said = stderr.trim();
    let known = match code {
        Some(6) => Some("could not resolve the host — DNS, or no network from this box"),
        Some(7) => Some("could not connect — the host is reachable but refused us"),
        Some(22) => Some("the server returned an error status (see below)"),
        Some(28) => Some("timed out before the transfer began"),
        Some(23) => Some("could not write the file — check the disk and permissions"),
        Some(35) | Some(60) => Some("TLS failed — a proxy or a clock skew will do this"),
        Some(33) => Some("the server refused a resumed range; retry without resume"),
        _ => None,
    };
    match (known, said.is_empty()) {
        (Some(k), true) => format!("curl: {k}"),
        (Some(k), false) => format!("curl: {k}\n  {said}"),
        (None, true) => format!("curl exited {code:?} and said nothing"),
        (None, false) => format!("curl exited {code:?}: {said}"),
    }
}

/// Run it, capturing the body as a string. For manifests and file listings.
pub fn fetch_string(req: &Request) -> Result<String, CurlError> {
    if !curl_available() {
        return Err(CurlError(
            "curl is not on PATH — hearth uses it for registry downloads (TLS, \
             redirects, resume). Install curl, or pre-place the blobs yourself."
                .into(),
        ));
    }
    let out = Command::new("curl")
        .args(req.argv())
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|e| CurlError(format!("could not run curl: {e}")))?;
    if !out.status.success() {
        return Err(CurlError(format!(
            "{}\n  url: {}",
            explain_failure(out.status.code(), &String::from_utf8_lossy(&out.stderr)),
            req.url
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// Run it, writing the body to `dest`. For blobs.
///
/// Progress goes to the inherited stderr rather than being captured, so a
/// multi-gigabyte download shows the operator that it is moving. A silent
/// twenty-minute pause is indistinguishable from a hang, and a downloader that
/// looks hung gets killed halfway.
pub fn fetch_file(req: &Request, show_progress: bool) -> Result<u64, CurlError> {
    let dest = req
        .dest
        .clone()
        .ok_or_else(|| CurlError("fetch_file needs a destination".into()))?;
    if !curl_available() {
        return Err(CurlError(
            "curl is not on PATH — hearth uses it for registry downloads (TLS, \
             redirects, resume). Install curl, or pre-place the blobs yourself."
                .into(),
        ));
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| CurlError(format!("could not create {}: {e}", parent.display())))?;
    }

    let mut argv = req.argv();
    if show_progress {
        // Replace --silent so curl draws its progress meter.
        if let Some(i) = argv.iter().position(|a| a == "--silent") {
            argv[i] = "--progress-bar".into();
        }
    }

    let status = Command::new("curl")
        .args(&argv)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::inherit())
        .status()
        .map_err(|e| CurlError(format!("could not run curl: {e}")))?;

    if !status.success() {
        return Err(CurlError(format!(
            "{}\n  url: {}",
            explain_failure(status.code(), ""),
            req.url
        )));
    }
    file_len(&dest)
        .map_err(|e| CurlError(format!("{} vanished after download: {e}", dest.display())))
}

fn file_len(p: &Path) -> std::io::Result<u64> {
    Ok(std::fs::metadata(p)?.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_404_must_not_be_written_to_disk_as_a_model() {
        // Without --fail, curl writes the error page to the output file and
        // exits 0. That is a 200-byte "not found" page named muse.gguf, which
        // fails much later as a corrupt tokenizer.
        let argv = Request::get("https://example.invalid/blob")
            .to_file("/tmp/x")
            .argv();
        assert!(argv.contains(&"--fail".to_string()));
    }

    #[test]
    fn redirects_are_followed_because_huggingface_always_redirects() {
        let argv = Request::get("https://huggingface.co/x").argv();
        assert!(argv.contains(&"--location".to_string()));
    }

    #[test]
    fn headers_are_passed_as_one_argument_each() {
        let argv = Request::get("https://r/x")
            .header(
                "Accept",
                "application/vnd.docker.distribution.manifest.v2+json",
            )
            .argv();
        let i = argv.iter().position(|a| a == "--header").expect("--header");
        assert_eq!(
            argv[i + 1],
            "Accept: application/vnd.docker.distribution.manifest.v2+json"
        );
    }

    #[test]
    fn resume_is_only_added_when_asked_and_only_with_a_destination() {
        let plain = Request::get("https://r/x").to_file("/tmp/x").argv();
        assert!(!plain.contains(&"--continue-at".to_string()));

        let resumed = Request::get("https://r/x")
            .to_file("/tmp/x")
            .resuming()
            .argv();
        let i = resumed.iter().position(|a| a == "--continue-at").unwrap();
        assert_eq!(resumed[i + 1], "-");

        // Resume without a destination is meaningless; it must not appear.
        let no_dest = Request {
            resume: true,
            ..Request::get("https://r/x")
        };
        assert!(!no_dest.argv().contains(&"--continue-at".to_string()));
    }

    #[test]
    fn the_url_is_after_a_double_dash_so_a_leading_dash_is_not_a_flag() {
        let argv = Request::get("-oops").argv();
        let i = argv.iter().position(|a| a == "--").expect("--");
        assert_eq!(argv[i + 1], "-oops");
        assert_eq!(i + 2, argv.len(), "the url is last");
    }

    #[test]
    fn retry_is_configured_but_no_max_time_is() {
        // Deliberate: retries handle a flaky link, but a total deadline would
        // kill a legitimately slow 40 GiB transfer and report it as failure.
        // Same rule as the supervisor: nothing fails on a clock.
        let argv = Request::get("https://r/x").argv();
        assert!(argv.contains(&"--retry".to_string()));
        assert!(argv.contains(&"--retry-connrefused".to_string()));
        assert!(
            !argv.iter().any(|a| a == "--max-time"),
            "a slow model is slow, not broken"
        );
    }

    #[test]
    fn failures_are_explained_by_cause_not_just_by_number() {
        assert!(explain_failure(Some(6), "").contains("resolve"));
        assert!(explain_failure(Some(22), "404").contains("error status"));
        assert!(explain_failure(Some(22), "404").contains("404"));
        assert!(explain_failure(Some(23), "").contains("disk"));
        assert!(explain_failure(Some(33), "").contains("resume"));
        // An unmapped code still says what it saw rather than swallowing it.
        let odd = explain_failure(Some(99), "something odd");
        assert!(odd.contains("99") && odd.contains("something odd"));
        assert!(explain_failure(None, "").contains("said nothing"));
    }
}
