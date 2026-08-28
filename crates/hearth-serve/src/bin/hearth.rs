//! hearth — keep declared models warm, and tell the truth about which are.
//!
//!   hearth serve --model NAME --gguf PATH [--port N] [--vram-gib N]
//!                [--ctx N] [--binary PATH] [--total-gib N] [--once]
//!   hearth status
//!   hearth why MODEL
//!   hearth as-of MODEL SEQ
//!   hearth verify
//!
//! `status`, `why`, `as-of` and `verify` read the on-disk spine directly —
//! they answer from the recorded history whether or not a supervisor is
//! running, because the history is the database, not a process's memory.

use std::io::Write;
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// SIGINT/SIGTERM latch. The handler only flips a bool — all real work
/// (stop children, record `unloaded`, flush the spine) happens on the
/// supervise loop, because a signal handler that touches a database is a
/// signal handler that corrupts one. Found live: without this, killing
/// the supervisor leaked the llama-server child and lost the final events.
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

extern "C" fn on_signal(_sig: i32) {
    SHUTDOWN.store(true, Ordering::SeqCst);
}

fn install_signal_handlers() {
    // SAFETY: on_signal is async-signal-safe — it writes one atomic bool.
    unsafe {
        extern "C" {
            fn signal(signum: i32, handler: extern "C" fn(i32)) -> usize;
        }
        const SIGINT: i32 = 2;
        const SIGTERM: i32 = 15;
        signal(SIGINT, on_signal);
        signal(SIGTERM, on_signal);
    }
}

use hearth_core::{Budget, Declared, GIB};
use hearth_serve::server::{free_port, runtime_available, ServerSpec};
use hearth_serve::{hearth_home, Supervisor};
use hearth_store::Spine;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("serve") => cmd_serve(&args[1..]),
        Some("status") => cmd_status(),
        Some("why") => cmd_why(&args[1..]),
        Some("as-of") => cmd_as_of(&args[1..]),
        Some("verify") => cmd_verify(),
        Some("pull") => cmd_pull(&args[1..]),
        Some("up") => cmd_up(&args[1..]),
        _ => {
            eprintln!("usage: hearth up|pull|serve|status|why|as-of|verify (see crate docs)");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("hearth: {e}");
            ExitCode::FAILURE
        }
    }
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn open_spine() -> Result<Spine, String> {
    let dir = hearth_home().join("spine");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    Spine::open(&dir)
}

/// `hearth pull <reference>` — fetch weights, verify the digest, record it.
///
/// The recording is the part no other tool does. Months later, "where did this
/// file come from" is `hearth why <model>` rather than an archaeology project.
/// A tunable: command-line flag wins, then the environment, then the default.
///
/// Flags beat env deliberately — an operator typing a flag is expressing intent
/// NOW, while an env var is standing configuration. The reverse order makes a
/// debugging session fight the deployment.
fn tunable<T: std::str::FromStr>(args: &[String], flag_name: &str, env: &str, default: T) -> T {
    if let Some(v) = flag(args, flag_name) {
        if let Ok(t) = v.parse() {
            return t;
        }
        eprintln!("hearth: {flag_name} {v:?} is not a valid value — using the default");
    }
    if let Ok(v) = std::env::var(env) {
        if let Ok(t) = v.parse() {
            return t;
        }
        eprintln!("hearth: {env}={v:?} is not a valid value — using the default");
    }
    default
}

fn cmd_pull(args: &[String]) -> Result<(), String> {
    let reference = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .ok_or("usage: hearth pull REFERENCE   (e.g. tinyllama, ollama:library/llama3:latest, file:./muse.gguf)")?;

    let cfg = hearth_pull::PullConfig {
        blobs_dir: match flag(args, "--blobs") {
            Some(d) => d.into(),
            None => hearth_home().join("blobs"),
        },
        // A silent multi-gigabyte pause is indistinguishable from a hang.
        progress: !args.iter().any(|a| a == "--quiet"),
        verify_existing: args.iter().any(|a| a == "--verify-existing"),
    };

    let spine = open_spine()?;
    eprintln!("hearth: pulling {reference} …");
    let out = hearth_pull::pull(reference, &cfg, &spine)?;
    spine.flush();

    let gib = out.bytes as f64 / GIB as f64;
    if out.already_had_it {
        println!("{} already here — {:.2} GiB", out.model, gib);
    } else {
        println!("{} pulled and verified — {:.2} GiB", out.model, gib);
    }
    println!("  from  {}", out.source);
    println!("  at    {}", out.weights_path.display());
    println!();
    println!(
        "  hearth serve --model {} --gguf {}",
        out.model,
        out.weights_path.display()
    );
    Ok(())
}

/// `hearth up` — a whole fleet behind one OpenAI-compatible port.
///
/// This is the shape that replaces a model runner rather than supplementing
/// one: declare a roster, hearth refuses what will not fit, brings up the rest,
/// and answers on a single endpoint — routing by the `model` field the way
/// every OpenAI client already sends it.
///
/// The part nothing else does is what happens when it CANNOT serve you. A
/// model still loading is a 503 with Retry-After; a model whose GPU the host
/// reclaimed is a 503 that says so in the body; a model that will never fit on
/// this card is a 409, because a retryable status there is a router hammering a
/// box that is arithmetically incapable of answering.
fn cmd_up(args: &[String]) -> Result<(), String> {
    let specs = collect_models(args)?;
    if specs.is_empty() {
        return Err(
            "usage: hearth up --model NAME=/path/to.gguf[:GIB] [--model …] \
                    [--port 11434] [--total-gib 48]"
                .into(),
        );
    }
    let api_port: u16 = tunable(args, "--port", "HEARTH_PORT", 11434);
    let total_gib: u64 = flag(args, "--total-gib")
        .map(|v| v.parse().map_err(|e| format!("--total-gib: {e}")))
        .transpose()?
        .unwrap_or(24);
    // Every knob: flag beats env beats default, so a deployment sets
    // HEARTH_* once and a debugging session overrides per-run.
    let ctx: u32 = tunable(args, "--ctx", "HEARTH_CTX", 0);
    // Concurrent slots per model. llama-server's own default is 1, which
    // serializes every caller behind the one in front of them. 8 because a
    // production node fronting a router is answering more than one caller,
    // and each slot's KV cache is already inside the declared budget.
    let parallel: u32 = tunable(args, "--parallel", "HEARTH_PARALLEL", 8);
    let gpu_layers: Option<i32> = {
        let v: i32 = tunable(args, "--gpu-layers", "HEARTH_GPU_LAYERS", -1);
        if v < 0 {
            None
        } else {
            Some(v)
        }
    };

    let budget = Budget {
        total_bytes: total_gib * GIB,
        reserve_bytes: 2 * GIB,
    };
    // Declaration order IS priority order. The operator listed these in the
    // order that matters to them; first fit, never best fit.
    let declared: Vec<Declared> = specs
        .iter()
        .map(|(name, _, gib)| Declared {
            model: name.clone(),
            weights_bytes: gib * GIB,
            kv_bytes: 0,
        })
        .collect();

    install_signal_handlers();
    let sup = Arc::new(Mutex::new(Supervisor::new(open_spine()?, budget, declared)));

    // Start only what the budget admitted. Anything refused stays declared and
    // visible — /residency reports it with the exact shortfall, rather than it
    // silently not existing.
    let log_dir = hearth_home().join("logs");
    std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
    for (name, gguf, _) in &specs {
        let port = free_port()?;
        let mut spec = ServerSpec::new(name, gguf, port);
        spec.ctx = ctx;
        spec.parallel = parallel;
        spec.gpu_layers = gpu_layers;
        spec.log_dir = log_dir.clone();
        if let Some(b) = flag(args, "--binary") {
            spec.binary = b.into();
        }
        if !runtime_available(&spec.binary) {
            return Err(format!(
                "runtime not available: {} — install llama.cpp or pass --binary",
                spec.binary.display()
            ));
        }
        match sup.lock().unwrap().start(spec) {
            Ok(()) => eprintln!("hearth: {name} loading on 127.0.0.1:{port} …"),
            // A refusal is not a crash. Say it and keep going: the rest of the
            // fleet is still worth serving.
            Err(e) => eprintln!("hearth: {name} not started — {e}"),
        }
    }

    let addr = format!("127.0.0.1:{api_port}");
    let server = hearth_api::Server::bind(&addr)?;

    // SIGTERM latches SHUTDOWN, but the accept loop below is BLOCKED inside
    // accept(2), which only returns when a connection arrives — so a TERM was
    // ignored until the next request happened to come in, and every clean stop
    // ended in somebody's kill -9, which loses the final unloaded events from
    // the spine. This watcher unblocks accept the moment the latch flips, by
    // connecting to our own port once. Found by `start.sh down` timing out at
    // 10 seconds of TERM on a healthy gateway.
    {
        let addr = addr.clone();
        std::thread::spawn(move || {
            while !SHUTDOWN.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(250));
            }
            let _ = std::net::TcpStream::connect(&addr);
        });
    }
    println!("hearth up on http://{addr}");
    // Print what was actually chosen. A server whose concurrency you have to
    // infer from a config file is a server nobody tunes.
    println!(
        "  {} slot(s)/model · {} · ctx {}",
        parallel,
        match gpu_layers {
            Some(n) => format!("{n} GPU layer(s)"),
            None => "all layers on GPU".to_string(),
        },
        if ctx == 0 {
            "runtime default".to_string()
        } else {
            format!("{ctx} shared across slots")
        },
    );
    println!("  POST /v1/chat/completions   routed by the \"model\" field");
    println!("  GET  /v1/models             what is declared, and what is ready");
    println!("  GET  /residency             the truth the OpenAI shape cannot carry");

    // The supervisor ticks on its own thread so a long generation never stalls
    // residency tracking, and so the accept loop is never the thing holding
    // the fleet lock.
    let ticker = Arc::clone(&sup);
    std::thread::spawn(move || {
        while !SHUTDOWN.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(500));
            let mut s = match ticker.lock() {
                Ok(s) => s,
                Err(_) => break,
            };
            let n = s.tick();
            if n > 0 {
                eprintln!("hearth: {n} transition(s) recorded");
                eprint!("{}", s.report());
            }
        }
    });

    // One thread per connection, bounded. The accept loop used to handle
    // requests inline, which meant a single generation blocked every other
    // caller — parallel slots inside llama-server buy nothing if the gateway
    // in front of them is a queue of one.
    //
    // Bounded rather than unbounded because an unbounded spawn is a way to
    // convert a traffic spike into an OOM. Over the cap we answer 503 with a
    // Retry-After, which is the honest thing: we are busy, come back.
    let inflight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let max_inflight: usize = tunable(args, "--max-inflight", "HEARTH_MAX_INFLIGHT", 64);

    for conn in server.incoming() {
        if SHUTDOWN.load(Ordering::SeqCst) {
            break;
        }
        let mut stream = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("hearth: accept failed: {e}");
                continue;
            }
        };

        if inflight.load(Ordering::SeqCst) >= max_inflight {
            let body = serde_json::json!({
                "error": {
                    "message": format!(
                        "hearth is at its in-flight limit ({max_inflight}) — this is \
                         backpressure, not a failure"),
                    "type": "server_busy",
                    "retryable": true,
                    "operator_fault": false,
                }
            })
            .to_string();
            let _ =
                stream.write_all(hearth_api::http::render_response(503, &body, Some(1)).as_bytes());
            continue;
        }

        let sup = Arc::clone(&sup);
        let inflight = Arc::clone(&inflight);
        inflight.fetch_add(1, Ordering::SeqCst);
        std::thread::spawn(move || {
            // Decrement no matter how this thread leaves, or the limit ratchets
            // down until the gateway refuses everything.
            struct Guard(Arc<std::sync::atomic::AtomicUsize>);
            impl Drop for Guard {
                fn drop(&mut self) {
                    self.0.fetch_sub(1, Ordering::SeqCst);
                }
            }
            let _guard = Guard(inflight);

            let req = match hearth_api::http::read_request(&stream) {
                Ok(r) => r,
                Err(e) => {
                    let body = serde_json::json!({ "error": { "message": e } }).to_string();
                    let _ = stream
                        .write_all(hearth_api::http::render_response(400, &body, None).as_bytes());
                    return;
                }
            };

            // Decide under the lock; act OUTSIDE it. A generation can take
            // minutes, and holding the fleet lock across one would stall every
            // other request and the supervisor with it.
            let decision = {
                let s = match sup.lock() {
                    Ok(s) => s,
                    Err(_) => return,
                };
                hearth_api::decide(&req.path, &req.body, s.fleet(), hearth_serve::now_ms())
            };

            match decision {
                hearth_api::Decision::Proxy { model, endpoint } => {
                    if let Err(e) = hearth_api::http::proxy(
                        &endpoint,
                        &req,
                        &mut stream,
                        Duration::from_secs(5),
                    ) {
                        eprintln!("hearth: proxy to {model} at {endpoint} failed: {e}");
                        let body = serde_json::json!({
                            "error": { "message": e, "type": "upstream_unreachable" }
                        })
                        .to_string();
                        let _ = stream.write_all(
                            hearth_api::http::render_response(502, &body, None).as_bytes(),
                        );
                    }
                }
                hearth_api::Decision::Answer(r) => {
                    let _ = stream.write_all(
                        hearth_api::http::render_response(r.status, &r.body, r.retry_after)
                            .as_bytes(),
                    );
                }
            }
        });
    }

    eprintln!("hearth: shutting down — recording unloaded, reaping children");
    sup.lock().unwrap().stop_all();
    Ok(())
}

/// `--model NAME=/path/to.gguf` or `--model NAME=/path/to.gguf:20` (GiB).
///
/// The size is what the budget plans against. Without it we guess 4 GiB, which
/// is deliberately conservative: over-guessing refuses models that would fit,
/// and that is a worse failure than admitting one that is slightly tight.
fn collect_models(args: &[String]) -> Result<Vec<(String, String, u64)>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--model" {
            let spec = args
                .get(i + 1)
                .ok_or("--model needs NAME=/path/to.gguf[:GIB]")?;
            let (name, rest) = spec
                .split_once('=')
                .ok_or_else(|| format!("--model {spec}: expected NAME=/path/to.gguf[:GIB]"))?;
            // Split the size off the RIGHT, so a Windows path's drive colon is
            // not mistaken for a size separator.
            let (path, gib) = match rest.rsplit_once(':') {
                Some((p, g)) if g.chars().all(|c| c.is_ascii_digit()) && !g.is_empty() => {
                    (p, g.parse().unwrap_or(4))
                }
                _ => (rest, 4),
            };
            if name.is_empty() || path.is_empty() {
                return Err(format!("--model {spec}: name and path are both required"));
            }
            out.push((name.to_string(), path.to_string(), gib));
            i += 2;
        } else {
            i += 1;
        }
    }
    Ok(out)
}

fn cmd_serve(args: &[String]) -> Result<(), String> {
    let model = flag(args, "--model").ok_or("--model is required")?;
    let gguf = flag(args, "--gguf").ok_or("--gguf is required")?;
    let port: u16 = match flag(args, "--port") {
        Some(p) => p.parse().map_err(|e| format!("--port: {e}"))?,
        None => free_port()?,
    };
    let vram_gib: u64 = flag(args, "--vram-gib")
        .map(|v| v.parse().map_err(|e| format!("--vram-gib: {e}")))
        .transpose()?
        .unwrap_or(4);
    let total_gib: u64 = flag(args, "--total-gib")
        .map(|v| v.parse().map_err(|e| format!("--total-gib: {e}")))
        .transpose()?
        .unwrap_or(24);
    let ctx: u32 = flag(args, "--ctx")
        .map(|v| v.parse().map_err(|e| format!("--ctx: {e}")))
        .transpose()?
        .unwrap_or(0);
    let once = args.iter().any(|a| a == "--once");

    let mut spec = ServerSpec::new(&model, &gguf, port);
    spec.ctx = ctx;
    if let Some(b) = flag(args, "--binary") {
        spec.binary = b.into();
    }
    spec.log_dir = hearth_home().join("logs");
    std::fs::create_dir_all(&spec.log_dir).map_err(|e| e.to_string())?;

    if !runtime_available(&spec.binary) {
        return Err(format!(
            "runtime not available: {} — install llama.cpp or pass --binary",
            spec.binary.display()
        ));
    }

    let budget = Budget {
        total_bytes: total_gib * GIB,
        reserve_bytes: 2 * GIB,
    };
    let declared = vec![Declared {
        model: model.clone(),
        weights_bytes: vram_gib * GIB,
        kv_bytes: 0,
    }];

    install_signal_handlers();
    let mut sup = Supervisor::new(open_spine()?, budget, declared);
    sup.start(spec)?;
    eprintln!("hearth: {model} loading on 127.0.0.1:{port} …");

    let endpoint = sup.wait_ready(&model, Duration::from_secs(600))?;
    println!("resident {model} http://{endpoint} (OpenAI-compatible at /v1)");

    if once {
        // CI's mode: supervise until resident, report, stop cleanly, exit.
        sup.stop_all();
        return Ok(());
    }

    // Supervise until told to stop. Every transition lands in the spine as
    // it happens; a SIGINT/SIGTERM records `unloaded`, reaps the child, and
    // flushes — a kill is an event in the history, not the end of it.
    while !SHUTDOWN.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(500));
        let n = sup.tick();
        if n > 0 {
            eprintln!("hearth: {n} transition(s) recorded");
            eprint!("{}", sup.report());
        }
    }
    eprintln!("hearth: shutting down — recording unloaded, reaping children");
    sup.stop_all();
    Ok(())
}

fn cmd_status() -> Result<(), String> {
    let spine = open_spine()?;
    let events = spine.all_latest();
    if events.is_empty() {
        println!("no models in the spine yet — run `hearth serve`");
        return Ok(());
    }
    println!("{:<24} {:<14} {:>8}  facts", "MODEL", "STATE", "SEQ");
    for ev in events {
        println!(
            "{:<24} {:<14} {:>8}  {}",
            ev.model, ev.kind, ev.seq, ev.facts
        );
    }
    println!("\nhead {}", spine.head());
    Ok(())
}

fn cmd_why(args: &[String]) -> Result<(), String> {
    let model = args.first().ok_or("usage: hearth why MODEL")?;
    let spine = open_spine()?;
    let chain = spine.why(model);
    // Where the bytes came from is a different chain from why it is (not)
    // resident, and both are part of the answer. Pull events live in their own
    // collection; for one commit nothing read them, so this command claimed
    // "no history" about a model it had just downloaded and verified.
    let provenance = spine.provenance(model);

    if chain.is_empty() && provenance.is_empty() {
        return Err(format!("{model} has no history in the spine"));
    }

    if !provenance.is_empty() {
        println!("where {model} came from:\n");
        for (i, ev) in provenance.iter().enumerate() {
            let arrow = if i == 0 { "●" } else { "└─" };
            println!("{arrow} seq {:>5}  {:<14} {}", ev.seq, ev.kind, ev.facts);
        }
        if !chain.is_empty() {
            println!();
        }
    }

    if chain.is_empty() {
        println!("(never served — the bytes are here, nothing has loaded them)");
        return Ok(());
    }

    println!("why {model} — causal chain, newest first:\n");
    for (i, ev) in chain.iter().enumerate() {
        let arrow = if i == 0 { "●" } else { "└─" };
        println!("{arrow} seq {:>5}  {:<14} {}", ev.seq, ev.kind, ev.facts);
    }
    Ok(())
}

fn cmd_as_of(args: &[String]) -> Result<(), String> {
    let model = args.first().ok_or("usage: hearth as-of MODEL SEQ")?;
    let seq: u64 = args
        .get(1)
        .ok_or("usage: hearth as-of MODEL SEQ")?
        .parse()
        .map_err(|e| format!("SEQ: {e}"))?;
    let spine = open_spine()?;
    match spine.state_as_of(model, seq) {
        Some(ev) => {
            println!(
                "as of seq {seq}: {} was `{}` {}",
                ev.model, ev.kind, ev.facts
            );
            Ok(())
        }
        None => Err(format!("{model} had no state at seq {seq}")),
    }
}

fn cmd_verify() -> Result<(), String> {
    let spine = open_spine()?;
    let (checked, failures) = spine.verify();
    if failures.is_empty() {
        println!("verify ok — {checked} nodes checked, history intact");
        Ok(())
    } else {
        Err(format!(
            "verify FAILED — {} of {checked} nodes corrupt: {failures:?}",
            failures.len()
        ))
    }
}
