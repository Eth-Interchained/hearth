//! The llama-server child: spawn, watch, stop.
//!
//! hearth does not do inference. llama.cpp has spent years on kernels and
//! samplers; the supervisor's job is to keep that work running and tell
//! the truth about whether it is. One child per model, one port per child,
//! stdout/stderr captured to files so a crash leaves evidence.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

/// Everything needed to start one model's serving child.
#[derive(Debug, Clone)]
pub struct ServerSpec {
    /// Model name as declared ("muse", "llama3:latest").
    pub model: String,
    /// Path to the GGUF on disk.
    pub gguf: PathBuf,
    /// Port to serve on.
    pub port: u16,
    /// The llama-server binary. Default: "llama-server" on PATH.
    pub binary: PathBuf,
    /// Context size. 0 = llama-server's default.
    pub ctx: u32,
    /// Extra args passed through verbatim, after ours.
    pub extra_args: Vec<String>,
    /// Directory for stdout/stderr capture files.
    pub log_dir: PathBuf,
}

impl ServerSpec {
    pub fn new(model: impl Into<String>, gguf: impl Into<PathBuf>, port: u16) -> ServerSpec {
        ServerSpec {
            model: model.into(),
            gguf: gguf.into(),
            port,
            binary: PathBuf::from("llama-server"),
            ctx: 0,
            extra_args: Vec::new(),
            log_dir: std::env::temp_dir(),
        }
    }

    pub fn endpoint(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    /// The argv we will actually run — pure, so tests can assert it without
    /// spawning anything.
    pub fn argv(&self) -> Vec<String> {
        let mut args = vec![
            "-m".to_string(),
            self.gguf.display().to_string(),
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            self.port.to_string(),
        ];
        if self.ctx > 0 {
            args.push("-c".to_string());
            args.push(self.ctx.to_string());
        }
        args.extend(self.extra_args.iter().cloned());
        args
    }

    fn log_path(&self, stream: &str) -> PathBuf {
        // Model names can carry ':' and '/'; keep filenames boring.
        let safe: String = self
            .model
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '-' || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        self.log_dir.join(format!("hearth-{safe}-{stream}.log"))
    }
}

/// A running (or exited) serving child.
pub struct ServerChild {
    pub spec: ServerSpec,
    child: Child,
}

impl ServerChild {
    /// Spawn the child. Fails fast if the GGUF is missing — a child that
    /// starts and then dies on a bad path wastes a whole probe cycle to
    /// learn what a stat call knew for free.
    pub fn spawn(spec: ServerSpec) -> Result<ServerChild, String> {
        if !spec.gguf.exists() {
            return Err(format!("gguf not found: {}", spec.gguf.display()));
        }
        let stdout =
            std::fs::File::create(spec.log_path("out")).map_err(|e| format!("log file: {e}"))?;
        let stderr =
            std::fs::File::create(spec.log_path("err")).map_err(|e| format!("log file: {e}"))?;
        let child = Command::new(&spec.binary)
            .args(spec.argv())
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(|e| format!("spawn {}: {e}", spec.binary.display()))?;
        Ok(ServerChild { spec, child })
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Has the process exited? `None` = still running.
    pub fn exit_code(&mut self) -> Option<Option<i32>> {
        match self.child.try_wait() {
            Ok(Some(status)) => Some(status.code()),
            Ok(None) => None,
            Err(_) => Some(None),
        }
    }

    /// Stop deliberately. SIGKILL is honest here: llama-server holds no
    /// state worth a graceful window, and a supervisor that waits politely
    /// on a wedged child is a supervisor that hangs.
    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    /// Where the child's stderr went, for post-mortems.
    pub fn stderr_log(&self) -> PathBuf {
        self.spec.log_path("err")
    }
}

impl Drop for ServerChild {
    fn drop(&mut self) {
        // Never leak a serving child past its supervisor.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Find a free port by asking the OS for one and releasing it. Racy in
/// principle, fine in practice for a single supervisor per host.
pub fn free_port() -> Result<u16, String> {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .map_err(|e| e.to_string())
}

/// Does `binary` exist and run? Checked once at startup so a missing
/// llama-server is one clear error, not one confusing error per model.
pub fn runtime_available(binary: &Path) -> bool {
    Command::new(binary)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argv_is_exactly_what_we_run() {
        let mut spec = ServerSpec::new("muse", "/models/muse.gguf", 8080);
        spec.ctx = 4096;
        spec.extra_args = vec!["-t".into(), "2".into()];
        assert_eq!(
            spec.argv(),
            vec![
                "-m",
                "/models/muse.gguf",
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "-c",
                "4096",
                "-t",
                "2"
            ]
        );
    }

    #[test]
    fn zero_ctx_means_no_flag() {
        let spec = ServerSpec::new("muse", "/models/muse.gguf", 8080);
        assert!(!spec.argv().contains(&"-c".to_string()));
    }

    #[test]
    fn log_names_are_boring() {
        let spec = ServerSpec::new("hf:owner/repo:Q4_K_M", "/m.gguf", 1);
        let p = spec.log_path("err");
        let name = p.file_name().unwrap().to_str().unwrap();
        assert_eq!(name, "hearth-hf_owner_repo_Q4_K_M-err.log");
    }

    #[test]
    fn missing_gguf_fails_before_spawn() {
        let spec = ServerSpec::new("ghost", "/definitely/not/here.gguf", 1);
        let err = match ServerChild::spawn(spec) {
            Err(e) => e,
            Ok(_) => panic!("spawn succeeded on a missing gguf"),
        };
        assert!(err.contains("gguf not found"), "{err}");
    }

    #[test]
    fn missing_binary_is_one_clear_answer() {
        assert!(!runtime_available(Path::new(
            "/definitely/not/llama-server"
        )));
    }

    #[test]
    fn a_real_child_runs_exits_and_reports() {
        // Use /bin/sh as a stand-in runtime: spawns, sleeps briefly, exits 7.
        let dir = tempfile::tempdir().unwrap();
        let gguf = dir.path().join("fake.gguf");
        std::fs::write(&gguf, b"not a real gguf").unwrap();
        let mut spec = ServerSpec::new("fake", &gguf, 1);
        spec.binary = PathBuf::from("/bin/sh");
        spec.extra_args = vec!["-c".into(), "exit 7".into()];
        // /bin/sh ignores our -m/--host args? No — it errors on them, which
        // is fine: we only assert the child lifecycle, not the exit value.
        spec.log_dir = dir.path().to_path_buf();
        let mut child = ServerChild::spawn(spec).unwrap();
        assert!(child.pid() > 0);
        // Wait for exit (bounded).
        let mut code = None;
        for _ in 0..50 {
            if let Some(c) = child.exit_code() {
                code = Some(c);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(code.is_some(), "child never exited");
    }
}
