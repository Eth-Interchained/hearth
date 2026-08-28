//! The scenarios these tests encode all happened, on 2026-08-27, on one
//! RTX A6000 rented from a GPU-virtualization host, serving a PIN operator.
//!
//! Symptom: requests hung, then failed. Four pull requests were merged that
//! night against the streaming path, and none of them were the cause, because
//! the cause was never visible — every distinct failure arrived as the same
//! timeout.

use hearth_core::budget::{gib, GIB};
use hearth_core::{plan, Budget, Declared, LostReason, Observation, Residency};

const T0: u64 = 1_000_000;
const SEC: u64 = 1_000;

fn resident(vram: u64) -> Residency {
    Residency::Unknown
        .observe(&Observation::LoadStarted, T0)
        .observe(&Observation::ProbeOk { vram_bytes: vram }, T0 + 30 * SEC)
}

// ---------------------------------------------------------------------------
// The two states that do not exist anywhere else.
// ---------------------------------------------------------------------------

#[test]
fn a_detached_gpu_is_not_the_operators_fault() {
    // The host reclaimed the card while the process kept running. This is the
    // whole reason the crate exists: the operator did nothing wrong, and
    // scoring them down for it poisons the only signal the network has.
    let lost = resident(20 * GIB).observe(
        &Observation::ProbeFailed {
            gpu_present: false,
            detail: "no CUDA device".into(),
        },
        T0 + 300 * SEC,
    );

    match &lost {
        Residency::Lost { reason, .. } => {
            assert_eq!(*reason, LostReason::GpuDetached);
            assert!(!reason.is_operator_fault(), "must never count against them");
            assert!(reason.worth_retrying_here(), "the card may come right back");
        }
        other => panic!("expected Lost, got {other:?}"),
    }
    assert!(lost
        .explain(T0 + 300 * SEC)
        .contains("detached by the host"));
}

#[test]
fn an_eviction_is_the_operators_problem_and_says_so() {
    // GPU still present, model gone: the runtime dropped it to free VRAM.
    // That IS the operator's to fix — they declared more than the card holds.
    let lost = resident(20 * GIB).observe(
        &Observation::ProbeFailed {
            gpu_present: true,
            detail: "model not loaded".into(),
        },
        T0 + 300 * SEC,
    );

    match &lost {
        Residency::Lost { reason, .. } => {
            assert_eq!(*reason, LostReason::Evicted);
            assert!(reason.is_operator_fault());
            // Re-asking the same over-committed box just evicts something else.
            assert!(!reason.worth_retrying_here());
        }
        other => panic!("expected Lost, got {other:?}"),
    }
}

#[test]
fn the_two_losses_are_distinguishable_from_identical_symptoms() {
    // Same prior state, same failed probe, one bit of difference — and that
    // bit is the entire diagnosis. Without it both are "timeout".
    let detached = resident(20 * GIB).observe(
        &Observation::ProbeFailed {
            gpu_present: false,
            detail: "x".into(),
        },
        T0 + SEC,
    );
    let evicted = resident(20 * GIB).observe(
        &Observation::ProbeFailed {
            gpu_present: true,
            detail: "x".into(),
        },
        T0 + SEC,
    );
    assert_ne!(detached, evicted);
    assert_ne!(
        detached.explain(T0 + SEC),
        evicted.explain(T0 + SEC),
        "an operator must be able to tell these apart by reading one line",
    );
}

// ---------------------------------------------------------------------------
// Loading. The state everything else lies about.
// ---------------------------------------------------------------------------

#[test]
fn loading_is_not_ready_and_routing_must_not_pretend_otherwise() {
    // The most common way a serving stack lies: route to something still
    // coming up, then call the inevitable timeout an error.
    let loading = Residency::Unknown.observe(&Observation::LoadStarted, T0);
    assert!(!loading.is_ready());
    assert!(
        loading.is_coming(),
        "a caller can wait instead of failing over"
    );
}

#[test]
fn a_slow_load_is_never_a_failure_however_long_it_takes() {
    // A 32B materializing over a network fabric can legitimately take minutes.
    // Killing it on a deadline converts a slow success into a fast failure.
    let loading = Residency::Unknown.observe(&Observation::LoadStarted, T0);
    let after_ten_minutes = T0 + 600 * SEC;
    assert!(loading.is_coming());
    assert_eq!(loading.loading_for(after_ten_minutes), Some(600 * SEC));
    assert!(loading.explain(after_ten_minutes).contains("600s"));
}

#[test]
fn a_failed_probe_while_still_loading_is_expected_not_a_loss() {
    // The server simply is not up yet. Demoting here would make every cold
    // start look like a fault.
    let loading = Residency::Unknown.observe(&Observation::LoadStarted, T0);
    let still = loading.observe(
        &Observation::ProbeFailed {
            gpu_present: true,
            detail: "conn refused".into(),
        },
        T0 + 5 * SEC,
    );
    assert_eq!(still, loading, "still loading, nothing has gone wrong");
}

#[test]
fn but_a_vanishing_gpu_is_news_even_mid_load() {
    let loading = Residency::Unknown.observe(&Observation::LoadStarted, T0);
    let lost = loading.observe(
        &Observation::ProbeFailed {
            gpu_present: false,
            detail: "no device".into(),
        },
        T0 + 5 * SEC,
    );
    assert!(matches!(
        lost,
        Residency::Lost {
            reason: LostReason::GpuDetached,
            ..
        }
    ));
}

// ---------------------------------------------------------------------------
// Bookkeeping that has to survive contact with reality.
// ---------------------------------------------------------------------------

#[test]
fn resident_since_survives_thousands_of_probes() {
    // "Resident for four hours" must stay true no matter how often we check.
    let mut r = resident(20 * GIB);
    let start = match r {
        Residency::Resident { since, .. } => since,
        _ => unreachable!(),
    };
    for i in 1..2000u64 {
        r = r.observe(
            &Observation::ProbeOk {
                vram_bytes: 20 * GIB,
            },
            T0 + i * SEC,
        );
    }
    assert!(matches!(r, Residency::Resident { since, .. } if since == start));
}

#[test]
fn a_deliberate_stop_is_nobodys_fault_and_its_exit_is_not_either() {
    // Without this, every clean shutdown files a fault against the operator.
    let stopped = resident(20 * GIB).observe(&Observation::StopRequested, T0 + SEC);
    assert!(matches!(stopped, Residency::Stopped { .. }));

    let after_exit = stopped.observe(&Observation::ProcessExited { code: Some(0) }, T0 + 2 * SEC);
    assert!(
        matches!(after_exit, Residency::Stopped { .. }),
        "the exit is the epilogue of the stop, not a new loss",
    );
}

#[test]
fn a_spurious_load_start_does_not_knock_a_healthy_model_offline() {
    let r = resident(20 * GIB);
    let again = r.observe(&Observation::LoadStarted, T0 + 60 * SEC);
    assert_eq!(again, r, "a healthy model must not be reported unavailable");
}

#[test]
fn only_resident_models_count_against_the_vram_budget() {
    // A loading model has not claimed its memory yet. Counting it would make
    // the planner refuse admissions it should allow.
    let loading = Residency::Unknown.observe(&Observation::LoadStarted, T0);
    assert_eq!(loading.vram_bytes(), 0);
    assert_eq!(resident(20 * GIB).vram_bytes(), 20 * GIB);
}

#[test]
fn every_state_accepts_every_observation_without_panicking() {
    // The real world delivers these in orders nobody planned for. A supervisor
    // that panics on a surprising sequence is worse than one that records
    // something slightly odd.
    let states = [
        Residency::Unknown,
        Residency::Loading { since: T0 },
        Residency::Resident {
            since: T0,
            vram_bytes: GIB,
        },
        Residency::Lost {
            at: T0,
            reason: LostReason::Evicted,
        },
        Residency::Failed {
            at: T0,
            reason: "oom".into(),
        },
        Residency::Stopped { at: T0 },
    ];
    let obs = [
        Observation::LoadStarted,
        Observation::ProbeOk { vram_bytes: GIB },
        Observation::ProbeFailed {
            gpu_present: true,
            detail: "x".into(),
        },
        Observation::ProbeFailed {
            gpu_present: false,
            detail: "x".into(),
        },
        Observation::ProcessExited { code: Some(1) },
        Observation::LoadFailed {
            detail: "no such model".into(),
        },
        Observation::StopRequested,
    ];
    for s in &states {
        for o in &obs {
            let next = s.observe(o, T0 + SEC);
            // And it always has something honest to say about itself.
            assert!(!next.explain(T0 + SEC).is_empty());
        }
    }
}

// ---------------------------------------------------------------------------
// The budget. This is the check that would have caught it on day one.
// ---------------------------------------------------------------------------

#[test]
fn one_a6000_cannot_hold_the_roster_that_was_declared_on_it() {
    // 48 GiB card. These are the models that were registered when PIN stopped
    // working, with rough Q4 weights plus a modest KV allowance. Nothing
    // errored at the time — the runtime simply loaded and evicted in a loop
    // forever, and it presented as "the models got slow".
    let card = Budget::with_reserve_pct(48 * GIB, 8);
    let declared = vec![
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
        Declared {
            model: "qwen3.6:27b".into(),
            weights_bytes: 17 * GIB,
            kv_bytes: GIB,
        },
    ];

    let p = plan(card, &declared);

    assert!(!p.fits(), "this roster has never fit on this card");
    assert_eq!(p.admitted, vec!["muse-local:latest", "deepseek-r1:32b"]);
    assert_eq!(p.rejected.len(), 2);

    // The rejection has to carry the number, because "you are short by N GiB"
    // is actionable and "invalid configuration" is not.
    let first = &p.rejected[0];
    assert_eq!(first.model, "gemma4:26b");
    assert!(first.short_bytes() > 0);
    assert!(p.explain().contains("REJECTED gemma4:26b"));
    assert!(p.explain().contains("short by"));
}

#[test]
fn declaration_order_is_priority_order() {
    // First fit, never best fit. Reordering to squeeze in one more model would
    // silently demote the model the operator listed first — and on a serving
    // box, first means most important.
    let card = Budget::with_reserve_pct(48 * GIB, 8);
    let big_first = vec![
        Declared {
            model: "big".into(),
            weights_bytes: 40 * GIB,
            kv_bytes: 0,
        },
        Declared {
            model: "small".into(),
            weights_bytes: 2 * GIB,
            kv_bytes: 0,
        },
        Declared {
            model: "tiny".into(),
            weights_bytes: GIB,
            kv_bytes: 0,
        },
    ];
    let p = plan(card, &big_first);
    assert_eq!(p.admitted[0], "big", "the operator asked for big first");
}

#[test]
fn the_reserve_is_never_planned_into() {
    let card = Budget::with_reserve_pct(48 * GIB, 8);
    assert!(card.usable_bytes() < card.total_bytes);
    let all = vec![Declared {
        model: "greedy".into(),
        weights_bytes: 48 * GIB,
        kv_bytes: 0,
    }];
    let p = plan(card, &all);
    assert!(
        !p.fits(),
        "a model sized to the whole card must not be admitted — KV cache, \
         CUDA context and fragmentation all still have to fit somewhere",
    );
}

#[test]
fn a_roster_that_fits_reports_its_headroom() {
    let card = Budget::with_reserve_pct(48 * GIB, 8);
    let declared = vec![
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
    ];
    let p = plan(card, &declared);
    assert!(p.fits());
    assert_eq!(p.rejected.len(), 0);
    assert!(gib(p.headroom_bytes()) >= 0.0);
    assert!(p.explain().contains("2 of 2 admitted"));
}

#[test]
fn a_tiny_card_still_gets_a_real_reserve() {
    // 8% of a small card is not enough for a CUDA context, so the reserve has
    // a floor. Otherwise the planner cheerfully fills a 4 GiB card to 3.7.
    let small = Budget::with_reserve_pct(4 * GIB, 8);
    assert!(small.reserve_bytes >= GIB);
}

// ---------------------------------------------------------------------------
// Reading a roster out of JSON.
//
// These exist because the Node binding shipped a version that dropped every
// model on the floor and then reported `fits: true`. JavaScript has no
// integers, every size arrived as an f64, `as_u64()` said None to all of them,
// and a `filter_map` quietly discarded the entire roster. An empty roster
// trivially fits — so the answer was confident, instant, and about a question
// nobody asked.
//
// Caught by running the addon, not by reading it.
// ---------------------------------------------------------------------------

use hearth_core::budget::declared_from_json;
use serde_json::json;

#[test]
fn a_javascript_number_is_a_valid_byte_count() {
    // THE BUG. JS sends 21474836480 as an f64 and there is nothing wrong with
    // that — a language without integers is not making a mistake.
    let v = json!([{ "model": "muse", "weightsBytes": 21474836480.0_f64, "kvBytes": 1.0e9 }]);
    let got = declared_from_json(&v).expect("an f64 byte count must be accepted");
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].weights_bytes, 21474836480);
    assert_eq!(got[0].kv_bytes, 1_000_000_000);
}

#[test]
fn a_roster_never_ever_silently_shrinks() {
    // The real sin was not the f64 — it was skipping what could not be read.
    // A plan computed over a roster that quietly lost half its entries is
    // right about the wrong question, which is worse than an error.
    let v = json!([
        { "model": "good", "weightsBytes": 1000 },
        { "model": "bad" },
        { "weightsBytes": 1000 },
    ]);
    let errs = declared_from_json(&v).expect_err("must refuse, not shrink");
    assert_eq!(errs.len(), 2, "every unreadable entry is named: {errs:?}");
    assert!(errs.iter().any(|e| e.contains("bad")), "{errs:?}");
    assert!(errs.iter().any(|e| e.contains("model")), "{errs:?}");
}

#[test]
fn both_spellings_work_because_two_languages_call_it_two_things() {
    let camel = json!([{ "model": "m", "weightsBytes": 10, "kvBytes": 5 }]);
    let snake = json!([{ "model": "m", "weights_bytes": 10, "kv_bytes": 5 }]);
    assert_eq!(
        declared_from_json(&camel).unwrap(),
        declared_from_json(&snake).unwrap()
    );
}

#[test]
fn kv_is_optional_but_a_malformed_kv_is_not_ignored() {
    let absent = json!([{ "model": "m", "weightsBytes": 10 }]);
    assert_eq!(declared_from_json(&absent).unwrap()[0].kv_bytes, 0);

    let null = json!([{ "model": "m", "weightsBytes": 10, "kvBytes": null }]);
    assert_eq!(declared_from_json(&null).unwrap()[0].kv_bytes, 0);

    // Present and wrong is a different thing from absent, and must be said.
    let junk = json!([{ "model": "m", "weightsBytes": 10, "kvBytes": "lots" }]);
    assert!(declared_from_json(&junk).is_err());
}

#[test]
fn a_fractional_or_negative_size_is_a_bug_worth_surfacing() {
    // Truncating 1.5 bytes would be inventing data; clamping -1 would be
    // hiding a caller's arithmetic error.
    assert!(declared_from_json(&json!([{ "model": "m", "weightsBytes": 1.5 }])).is_err());
    assert!(declared_from_json(&json!([{ "model": "m", "weightsBytes": -1 }])).is_err());
}

#[test]
fn an_empty_model_name_is_refused() {
    // It would produce a slot nothing can ever route to.
    assert!(declared_from_json(&json!([{ "model": "", "weightsBytes": 10 }])).is_err());
}

#[test]
fn a_genuinely_empty_roster_is_fine_but_a_non_array_is_not() {
    assert_eq!(declared_from_json(&json!([])).unwrap().len(), 0);
    assert!(declared_from_json(&json!({ "model": "m" })).is_err());
}
