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
    ///
    /// Note this is the TOTAL context divided across slots by llama-server, so
    /// `--ctx 8192 --parallel 4` gives each concurrent request 2048 — not 8192
    /// each. Sizing it as if every slot got the full window is the most common
    /// way a "parallel" server starts truncating prompts.
    pub ctx: u32,
    /// Concurrent request slots. This is what makes a server able to answer two
    /// callers at once; llama-server defaults to ONE, which means every request
    /// after the first waits for the one in front of it.
    ///
    /// hearth defaults to 8 because a production node fronting a router is
    /// answering more than one caller, and a serving stack that silently
    /// serializes looks exactly like a slow model. Each slot costs its own KV
    /// cache, which is why the budget wants to know.
    pub parallel: u32,
    /// Layers to offload to the GPU. `None` means all of them (`-1`).
    ///
    /// llama.cpp's own default is 0 — CPU. A model runner that quietly runs a
    /// 20 GiB model on the CPU is not slow by a little; it is two orders of
    /// magnitude off, and it looks like a hardware problem rather than a
    /// missing flag. If the model did not fit the planner would already have
    /// refused it, so "all of them" is the honest default here.
    pub gpu_layers: Option<i32>,
    /// Lock the model's pages into RAM (`--mlock`). OFF by default, and the
    /// default matters: with all layers offloaded to the GPU, the host copy of
    /// the weights is cold after load — mlock pins it in RAM anyway, turning
    /// droppable page cache into unreclaimable locked memory. On a box whose
    /// RAM is the same size as its card (a 48 GiB A6000 host with 48 GiB RAM,
    /// say), locking two 20 GiB models is a system-OOM assist, and it landed
    /// on exactly that box within hours of shipping as a default.
    ///
    /// Turn it on for CPU inference, where a paged-out weight page really does
    /// turn the first token after an idle period into a disk read.
    pub mlock: bool,
    /// Continuous batching. Only meaningful with more than one slot, and the
    /// reason parallel slots actually help instead of just interleaving.
    pub cont_batching: bool,
    /// Which GPUs THIS child may see, as `CUDA_VISIBLE_DEVICES` (e.g. `"0,1,2"`
    /// or `"3"`). `None` leaves the environment alone — every child sees every
    /// card, which is llama.cpp's own behaviour.
    ///
    /// This exists because "all the cards" is the wrong answer for a MIXED
    /// fleet. llama-server spreads a model proportionally across everything it
    /// can see, so on a 4×V100 box an 8B that fits on one card was found
    /// holding ~2.8 GiB on each of three of them — paying a PCIe host-bridge
    /// hop per layer boundary for nothing, and taking 8.5 GiB of headroom away
    /// from the 32B it was sharing them with. Neither `--parallel` nor
    /// `--ctx` nor any amount of `extra` could express "put the small one over
    /// there"; placement is not a llama-server flag, it is the environment the
    /// child is spawned into.
    ///
    /// Pinning is also how a card gets RESERVED: a device named by no model is
    /// a device the fleet will not touch, which is the only way to leave room
    /// for a non-hearth tenant (a TTS process, another runtime) on the same
    /// host.
    pub devices: Option<String>,
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
            parallel: 8,
            gpu_layers: None,
            mlock: false,
            cont_batching: true,
            devices: None,
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
        // All layers on the GPU unless told otherwise. llama.cpp defaults to 0
        // — CPU — and a 20 GiB model quietly running on CPU reads as broken
        // hardware, not as a missing flag.
        args.push("--n-gpu-layers".to_string());
        args.push(match self.gpu_layers {
            Some(n) => n.to_string(),
            None => "-1".to_string(),
        });
        if self.parallel > 0 {
            args.push("--parallel".to_string());
            args.push(self.parallel.to_string());
        }
        // Continuous batching is what makes extra slots serve concurrent
        // callers rather than just queue them differently.
        if self.cont_batching && self.parallel > 1 {
            args.push("--cont-batching".to_string());
        }
        if self.mlock {
            args.push("--mlock".to_string());
        }
        // Caller's args go LAST so an operator can override any default above.
        args.extend(self.extra_args.iter().cloned());
        args
    }

    /// The environment we will actually set on the child — pure, for the same
    /// reason `argv()` is: placement that only exists inside a spawn call is
    /// placement no test can assert. Returns only the variables hearth SETS;
    /// everything else is inherited.
    ///
    /// `ld_library_path` is the inherited value, passed in rather than read
    /// here so this stays a function of its inputs.
    pub fn child_env(&self, ld_library_path: &str) -> Vec<(String, String)> {
        let mut env = Vec::new();
        // A fetched runtime keeps its shared libraries next to the binary;
        // without this, a prebuilt llama-server dies on ggml .so lookups and
        // reads as "started, then died".
        if let Some(dir) = self.binary.parent() {
            if dir.components().count() > 0 {
                let joined = if ld_library_path.is_empty() {
                    dir.display().to_string()
                } else {
                    format!("{}:{ld_library_path}", dir.display())
                };
                env.push(("LD_LIBRARY_PATH".to_string(), joined));
            }
        }
        // Placement. Set only when asked: an unconditional
        // CUDA_VISIBLE_DEVICES would override an operator's own export for
        // every fleet that never mentions devices at all.
        if let Some(d) = &self.devices {
            env.push(("CUDA_VISIBLE_DEVICES".to_string(), d.clone()));
        }
        env
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
        let mut cmd = Command::new(&spec.binary);
        let inherited = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        for (k, v) in spec.child_env(&inherited) {
            cmd.env(k, v);
        }
        let child = cmd
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

    /// The last few lines the child said before dying.
    ///
    /// This method exists because `stderr_log()` did not have one caller.
    /// hearth faithfully captured llama-server's stderr to a file and then
    /// reported only "the serving process exited" — so the operator was told
    /// their model died, was not told why, and was not even told the file
    /// existed. The cause of a 271.79 GiB model failing to load was
    /// `cudaMalloc failed: out of memory` sitting on disk, unmentioned,
    /// through several rounds of guessing.
    ///
    /// Reads the tail only: a llama-server log is thousands of lines of
    /// tensor metadata and the interesting part is always at the end.
    pub fn last_words(&self, lines: usize) -> Option<String> {
        tail_of(&self.stderr_log(), lines)
    }
}

/// Last `lines` non-empty lines of a file, or None when there is nothing
/// useful to show. Reads the whole file: a crash log is small, and doing this
/// properly with seeks would be more code than the problem deserves.
fn tail_of(path: &Path, lines: usize) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    let kept: Vec<&str> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .rev()
        .take(lines)
        .collect();
    if kept.is_empty() {
        return None;
    }
    Some(kept.into_iter().rev().collect::<Vec<_>>().join("\n"))
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
        // Exact, deliberately. A loose assertion here would not have noticed
        // that hearth shipped WITHOUT --n-gpu-layers or --parallel, which meant
        // llama.cpp's own defaults applied: layers on the CPU and one slot.
        // This test failing is the correct response to changing that contract.
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
                // Production defaults, in the order argv() emits them.
                "--n-gpu-layers",
                "-1",
                "--parallel",
                "8",
                "--cont-batching",
                // The operator's own arguments last, so they win.
                "-t",
                "2"
            ]
        );
    }

    #[test]
    fn placement_is_environment_not_argv() {
        // The whole point: CUDA_VISIBLE_DEVICES is not a llama-server flag.
        // If this ever leaks into argv, llama-server dies at spawn on an
        // unknown argument — and it would read as "the runtime is broken".
        let mut spec = ServerSpec::new("glm", "/models/glm.gguf", 8080);
        spec.devices = Some("0,1,2".into());
        assert!(!spec.argv().iter().any(|a| a.contains("CUDA")));
        assert!(!spec.argv().iter().any(|a| a == "--devices"));
        assert_eq!(
            spec.child_env(""),
            vec![("CUDA_VISIBLE_DEVICES".to_string(), "0,1,2".to_string())]
        );
    }

    #[test]
    fn no_devices_means_the_variable_is_left_alone() {
        // NOT `CUDA_VISIBLE_DEVICES=""` — that means "no GPU" and would put
        // every unpinned fleet on the CPU. And not an unconditional set of
        // "all cards" either: an operator who exports the variable themselves
        // for a fleet that never mentions devices must keep winning.
        let spec = ServerSpec::new("glm", "/models/glm.gguf", 8080);
        assert!(spec
            .child_env("")
            .iter()
            .all(|(k, _)| k != "CUDA_VISIBLE_DEVICES"));
    }

    #[test]
    fn the_runtimes_own_libraries_still_come_first() {
        // Placement must not have displaced the LD_LIBRARY_PATH fix: a
        // prebuilt llama-server dies on ggml .so lookups without it, and that
        // failure reads as "started, then died".
        let spec = ServerSpec::new("glm", "/models/glm.gguf", 8080);
        assert!(
            spec.child_env("").is_empty(),
            "a bare binary name has no parent dir to add"
        );

        let mut spec = ServerSpec::new("glm", "/models/glm.gguf", 8080);
        spec.binary = "/opt/llama/bin/llama-server".into();
        spec.devices = Some("3".into());
        assert_eq!(
            spec.child_env("/usr/lib"),
            vec![
                (
                    "LD_LIBRARY_PATH".to_string(),
                    "/opt/llama/bin:/usr/lib".to_string()
                ),
                ("CUDA_VISIBLE_DEVICES".to_string(), "3".to_string()),
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

#[cfg(test)]
mod production_defaults {
    use super::*;

    fn argv_of(spec: &ServerSpec) -> Vec<String> {
        spec.argv()
    }

    fn pair(argv: &[String], flag: &str) -> Option<String> {
        argv.iter()
            .position(|a| a == flag)
            .map(|i| argv[i + 1].clone())
    }

    #[test]
    fn all_layers_go_to_the_gpu_by_default() {
        // llama.cpp's own default is 0 — CPU. hearth shipped without this flag
        // at all, so a 20 GiB model ran wherever the build happened to default,
        // and CPU inference on a serving node reads as broken hardware rather
        // than as a missing argument.
        let argv = argv_of(&ServerSpec::new("m", "/m.gguf", 8080));
        assert_eq!(pair(&argv, "--n-gpu-layers").as_deref(), Some("-1"));
    }

    #[test]
    fn a_node_answers_more_than_one_caller_by_default() {
        // llama-server defaults to ONE slot: every request after the first
        // waits for the one in front of it, which looks exactly like a slow
        // model rather than a serialized one.
        let argv = argv_of(&ServerSpec::new("m", "/m.gguf", 8080));
        assert_eq!(pair(&argv, "--parallel").as_deref(), Some("8"));
        assert!(
            argv.iter().any(|a| a == "--cont-batching"),
            "extra slots without continuous batching just queue differently"
        );
    }

    #[test]
    fn one_slot_means_no_continuous_batching() {
        // With a single slot the flag is noise, and noise in an argv is how a
        // reader stops trusting the rest of it.
        let spec = ServerSpec {
            parallel: 1,
            ..ServerSpec::new("m", "/m.gguf", 8080)
        };
        assert!(!spec.argv().iter().any(|a| a == "--cont-batching"));
    }

    #[test]
    fn the_operators_own_arguments_win() {
        // Defaults are a starting point, not a policy. Ours go first so the
        // caller's --n-gpu-layers overrides rather than conflicts.
        let spec = ServerSpec {
            extra_args: vec!["--n-gpu-layers".into(), "20".into()],
            ..ServerSpec::new("m", "/m.gguf", 8080)
        };
        let argv = spec.argv();
        let ours = argv.iter().position(|a| a == "--n-gpu-layers").unwrap();
        let theirs = argv.iter().rposition(|a| a == "--n-gpu-layers").unwrap();
        assert!(theirs > ours, "the caller's copy must come last to win");
        assert_eq!(argv[theirs + 1], "20");
    }

    #[test]
    fn gpu_layers_can_be_pinned_including_to_cpu() {
        // A CPU-only box is a real deployment; it just must be chosen, not
        // arrived at by accident.
        let spec = ServerSpec {
            gpu_layers: Some(0),
            ..ServerSpec::new("m", "/m.gguf", 8080)
        };
        assert_eq!(pair(&spec.argv(), "--n-gpu-layers").as_deref(), Some("0"));
    }

    #[test]
    fn mlock_is_opt_in_because_it_pins_host_ram_the_gpu_path_does_not_need() {
        // Shipped as an always-on default first, and it hit a real box within
        // hours: 48 GiB A6000 host with 48 GiB RAM, two 20 GiB models — mlock
        // pinned the cold host copies and helped the system toward OOM. With
        // full GPU offload the host copy is droppable page cache; locking it
        // buys nothing and costs the model's size in RAM.
        assert!(!ServerSpec::new("m", "/m.gguf", 8080)
            .argv()
            .iter()
            .any(|a| a == "--mlock"));
        // Still there for CPU inference, where it earns its keep.
        let spec = ServerSpec {
            mlock: true,
            ..ServerSpec::new("m", "/m.gguf", 8080)
        };
        assert!(spec.argv().iter().any(|a| a == "--mlock"));
    }

    #[test]
    fn the_context_is_still_bound_to_the_port_and_model() {
        // The defaults must not have displaced the arguments that identify
        // WHICH model on WHICH port — an argv bug here looks like the model
        // being broken.
        let spec = ServerSpec {
            ctx: 8192,
            ..ServerSpec::new("muse", "/models/muse.gguf", 9001)
        };
        let argv = spec.argv();
        assert_eq!(pair(&argv, "-m").as_deref(), Some("/models/muse.gguf"));
        assert_eq!(pair(&argv, "--port").as_deref(), Some("9001"));
        assert_eq!(pair(&argv, "--host").as_deref(), Some("127.0.0.1"));
        assert_eq!(pair(&argv, "-c").as_deref(), Some("8192"));
    }
    #[test]
    fn a_dead_childs_last_words_are_the_tail_not_the_whole_log() {
        // The real log that cost three rounds of misdiagnosis: thousands of
        // lines of tensor metadata, and the answer in the last few. hearth
        // captured all of it and surfaced none of it.
        let dir = std::env::temp_dir().join(format!("hearth-lastwords-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("err.log");

        let mut body = String::new();
        for i in 0..3000 {
            body.push_str(&format!("I load_tensors: layer {i} noise\n"));
        }
        body.push('\n'); // blank lines must not eat a slot in the tail
        body.push_str(
            "E llama_model_load: error loading model: unknown model architecture: 'glm5next'\n",
        );
        body.push_str("E srv  llama_server: exiting due to model loading error\n");
        std::fs::write(&log, &body).unwrap();

        let tail = tail_of(&log, 12).expect("a non-empty log must yield a tail");
        // The cause must be present — that is the entire point.
        assert!(
            tail.contains("unknown model architecture: 'glm5next'"),
            "the tail dropped the actual error: {tail}"
        );
        assert!(
            tail.contains("exiting due to model loading error"),
            "{tail}"
        );
        // And the 3000 lines of metadata must NOT be, or the operator is back
        // to not reading it.
        assert!(
            tail.lines().count() <= 12,
            "tail too long: {}",
            tail.lines().count()
        );
        assert!(
            !tail.contains("layer 0 noise"),
            "tail reached the top of the log"
        );
        // Blank lines are skipped rather than consuming budget.
        assert!(!tail.lines().any(|l| l.trim().is_empty()), "{tail}");

        // A missing or empty log reports nothing rather than pretending. The
        // caller distinguishes "no evidence" from "here is the evidence", and
        // an empty string masquerading as a tail would be a quiet lie.
        assert!(tail_of(&dir.join("nope.log"), 12).is_none());
        std::fs::write(dir.join("empty.log"), "\n\n  \n").unwrap();
        assert!(tail_of(&dir.join("empty.log"), 12).is_none());

        std::fs::remove_dir_all(&dir).ok();
    }
}
