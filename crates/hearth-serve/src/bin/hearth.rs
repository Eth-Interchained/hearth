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

use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, Ordering};
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
        _ => {
            eprintln!("usage: hearth pull|serve|status|why|as-of|verify (see crate docs)");
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
