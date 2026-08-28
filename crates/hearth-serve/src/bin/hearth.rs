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
use std::time::Duration;

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
        _ => {
            eprintln!("usage: hearth serve|status|why|as-of|verify (see crate docs)");
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

    // Supervise forever. Every transition lands in the spine as it happens.
    loop {
        std::thread::sleep(Duration::from_secs(2));
        let n = sup.tick();
        if n > 0 {
            eprintln!("hearth: {n} transition(s) recorded");
            eprint!("{}", sup.report());
        }
    }
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
    if chain.is_empty() {
        return Err(format!("{model} has no history in the spine"));
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
