//! The README, compiled.
//!
//! Every snippet in `crates/hearth-core/README.md` is this file. A README is
//! documentation that no compiler ever reads, so it rots silently and then
//! wastes the first ten minutes of every newcomer's day. Keeping the examples
//! here and quoting them there means `cargo test --workspace` fails when the
//! docs start lying.
//!
//!     cargo run -p hearth-core --example quickstart

use hearth_core::budget::{gib, plan, Budget, Declared, GIB};
use hearth_core::fleet::{Fleet, Route};
use hearth_core::residency::{LostReason, Observation, Residency};

fn main() {
    residency_is_a_named_state();
    the_card_is_arithmetic();
    routers_get_an_answer();
}

/// Residency is a named state with a named reason.
fn residency_is_a_named_state() {
    println!("== residency ==");

    let s = Residency::Unknown
        .observe(&Observation::LoadStarted, 1_000)
        .observe(
            &Observation::ProbeOk {
                vram_bytes: 21 * GIB,
            },
            41_000,
        );

    assert!(s.is_ready());
    println!("  {}", s.explain(60_000)); // "resident for 19s"

    // When it goes wrong, the reason survives.
    let lost = s.observe(
        &Observation::ProbeFailed {
            gpu_present: false,
            detail: "no CUDA device".into(),
        },
        3_600_000,
    );

    match lost {
        Residency::Lost { reason, .. } => {
            assert_eq!(reason, LostReason::GpuDetached);
            // The bit that matters, and it is one boolean.
            assert!(!reason.is_operator_fault());
            println!(
                "  lost: {reason:?}, operator at fault: {}",
                reason.is_operator_fault()
            );
        }
        other => unreachable!("expected a loss, got {other:?}"),
    }
}

/// The card's size is arithmetic, checked before anything loads.
fn the_card_is_arithmetic() {
    println!("== budget ==");

    let budget = Budget::with_reserve_pct(48 * GIB, 8);
    let roster = vec![
        Declared {
            model: "muse-local:latest".into(),
            weights_bytes: 20 * GIB,
            kv_bytes: GIB,
        },
        Declared {
            model: "deepseek-r1:32b".into(),
            weights_bytes: 20 * GIB,
            kv_bytes: GIB,
        },
        Declared {
            model: "gemma4:26b".into(),
            weights_bytes: 16 * GIB,
            kv_bytes: GIB,
        },
    ];

    let p = plan(budget, &roster);
    println!("  {}", p.explain());
    for r in &p.rejected {
        println!(
            "  {} will never fit here — short by {:.1} GiB",
            r.model,
            gib(r.short_bytes())
        );
    }

    // Two admitted, one refused: 21 + 21 = 42 GiB against 44.2 GiB usable,
    // so the 17 GiB third model is short. Refused at declare time, not
    // discovered at 3am as "the models got slow".
    assert_eq!(p.admitted.len(), 2);
    assert_eq!(p.rejected.len(), 1);
}

/// Routers get an answer they can act on.
fn routers_get_an_answer() {
    println!("== routing ==");

    let budget = Budget::with_reserve_pct(48 * GIB, 8);
    let roster = vec![Declared {
        model: "muse-local:latest".into(),
        weights_bytes: 20 * GIB,
        kv_bytes: GIB,
    }];

    let now = 1_000;
    let mut fleet = Fleet::declare(budget, roster);
    fleet.set_endpoint("muse-local:latest", "127.0.0.1:8090");
    fleet.observe("muse-local:latest", &Observation::LoadStarted, now);

    // Every variant carries `model`, so `..` is not optional in these arms.
    match fleet.route("muse-local:latest", now + 5_000) {
        Route::Ready { endpoint, .. } => println!("  ready — send it to {endpoint}"),
        // The one every stack gets wrong: routing to a model that is still
        // coming up, then calling the inevitable timeout an error.
        Route::Warming { for_ms, .. } => {
            println!("  warming for {for_ms}ms — wait or try elsewhere, but do NOT fault this node")
        }
        Route::Lost {
            operator_fault,
            reason,
            ..
        } => {
            println!("  lost ({reason:?}) — try elsewhere; score only if {operator_fault}")
        }
        Route::NotAdmitted { short_bytes, .. } => {
            println!(
                "  never going to fit — short {:.1} GiB, stop asking",
                gib(short_bytes)
            )
        }
        Route::Failed { reason, .. } => println!("  failed: {reason}"),
        Route::NotDeclared { .. } => println!("  not declared here"),
        Route::Unknown { .. } => println!("  declared, never observed — we say so"),
    }

    println!("  {}", fleet.report(now + 5_000).trim_end());
}
