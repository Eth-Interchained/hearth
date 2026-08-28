//! What a router is told, and what it should do about it.
//!
//! The bug these exist to prevent: PIN's `send_inference_request` returns
//! `None` for "the operator is not connected" AND for "the operator is still
//! working." Two opposite conditions, one value — so a slow node gets retried
//! as though it were a broken one, and an operator whose host reclaimed a GPU
//! gets scored down for it.

use hearth_core::budget::GIB;
use hearth_core::fleet::{Fleet, Route};
use hearth_core::{Budget, Declared, LostReason, Observation};

const T0: u64 = 1_000_000;
const SEC: u64 = 1_000;

fn card() -> Budget {
    Budget::with_reserve_pct(48 * GIB, 8)
}

fn roster() -> Vec<Declared> {
    vec![
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
    ]
}

fn warm(fleet: &mut Fleet, model: &str, port: u16) {
    fleet.observe(model, &Observation::LoadStarted, T0);
    fleet.observe(
        model,
        &Observation::ProbeOk {
            vram_bytes: 21 * GIB,
        },
        T0 + 30 * SEC,
    );
    fleet.set_endpoint(model, format!("http://127.0.0.1:{port}"));
}

// ---------------------------------------------------------------------------
// The six answers.
// ---------------------------------------------------------------------------

#[test]
fn a_warm_model_routes_to_its_endpoint() {
    let mut f = Fleet::declare(card(), roster());
    warm(&mut f, "muse-local:latest", 8081);
    let r = f.route("muse-local:latest", T0 + 60 * SEC);
    assert!(r.is_ready());
    assert!(!r.should_try_elsewhere());
    assert!(matches!(r, Route::Ready { endpoint, .. } if endpoint.ends_with(":8081")));
}

#[test]
fn a_warming_model_says_how_long_and_is_nobodys_fault() {
    // The critical one. Today this is a socket that hangs and then a fault
    // filed against a node that was doing exactly what it should.
    let mut f = Fleet::declare(card(), roster());
    f.observe("deepseek-r1:32b", &Observation::LoadStarted, T0);

    let r = f.route("deepseek-r1:32b", T0 + 43 * SEC);
    match &r {
        Route::Warming { for_ms, .. } => assert_eq!(*for_ms, 43 * SEC),
        other => panic!("expected Warming, got {other:?}"),
    }
    assert!(!r.is_ready(), "must never route traffic to a loading model");
    assert!(!r.should_try_elsewhere(), "it is coming — waiting is valid");
    assert!(!r.operator_fault());
}

#[test]
fn a_detached_gpu_routes_elsewhere_without_blaming_anyone() {
    let mut f = Fleet::declare(card(), roster());
    warm(&mut f, "muse-local:latest", 8081);
    f.observe(
        "muse-local:latest",
        &Observation::ProbeFailed {
            gpu_present: false,
            detail: "no device".into(),
        },
        T0 + 300 * SEC,
    );

    let r = f.route("muse-local:latest", T0 + 300 * SEC);
    assert!(r.should_try_elsewhere());
    assert!(
        !r.operator_fault(),
        "the host took the card — scoring the operator down for that is the \
         bug that quietly deletes honest operators from a marketplace",
    );
    assert!(matches!(
        r,
        Route::Lost {
            reason: LostReason::GpuDetached,
            ..
        }
    ));
}

#[test]
fn an_eviction_does_count_against_the_operator() {
    let mut f = Fleet::declare(card(), roster());
    warm(&mut f, "muse-local:latest", 8081);
    f.observe(
        "muse-local:latest",
        &Observation::ProbeFailed {
            gpu_present: true,
            detail: "not loaded".into(),
        },
        T0 + 300 * SEC,
    );
    let r = f.route("muse-local:latest", T0 + 300 * SEC);
    assert!(
        r.operator_fault(),
        "over-committing the card IS theirs to fix"
    );
}

#[test]
fn a_model_that_does_not_fit_says_so_permanently() {
    // Not an outage. A configuration fact, and the caller should stop asking
    // rather than retrying forever against a card that will never hold it.
    let f = Fleet::declare(card(), roster());
    let r = f.route("gemma4:26b", T0);
    match r {
        Route::NotAdmitted { short_bytes, .. } => assert!(short_bytes > 0),
        other => panic!("expected NotAdmitted, got {other:?}"),
    }
}

#[test]
fn an_undeclared_model_is_distinguishable_from_a_broken_one() {
    let f = Fleet::declare(card(), roster());
    assert!(matches!(
        f.route("llama9:700b", T0),
        Route::NotDeclared { .. }
    ));
}

#[test]
fn resident_with_nowhere_to_send_traffic_is_not_ready() {
    // A model can be loaded and holding VRAM while its server has not bound a
    // port yet. Reporting Ready here hands the caller a hole to fall into.
    let mut f = Fleet::declare(card(), roster());
    f.observe("muse-local:latest", &Observation::LoadStarted, T0);
    f.observe(
        "muse-local:latest",
        &Observation::ProbeOk {
            vram_bytes: 21 * GIB,
        },
        T0 + SEC,
    );
    assert!(matches!(
        f.route("muse-local:latest", T0 + SEC),
        Route::Unknown { .. }
    ));
}

// ---------------------------------------------------------------------------
// Holding the line on VRAM.
// ---------------------------------------------------------------------------

#[test]
fn committed_vram_is_measured_not_estimated() {
    let mut f = Fleet::declare(card(), roster());
    assert_eq!(f.live_committed_bytes(), 0, "nothing loaded, nothing held");

    warm(&mut f, "muse-local:latest", 8081);
    assert_eq!(f.live_committed_bytes(), 21 * GIB);

    // Losing it frees the accounting too, or the fleet slowly convinces itself
    // the card is full of models that are not there.
    f.observe(
        "muse-local:latest",
        &Observation::ProbeFailed {
            gpu_present: false,
            detail: "gone".into(),
        },
        T0 + 100 * SEC,
    );
    assert_eq!(f.live_committed_bytes(), 0);
}

#[test]
fn the_loader_never_evicts_to_make_room() {
    // The entire point. If a model does not fit, the answer is that it does
    // not fit — not "evict the thing someone is using".
    let mut f = Fleet::declare(card(), roster());
    warm(&mut f, "muse-local:latest", 8081);
    warm(&mut f, "deepseek-r1:32b", 8082);

    assert_eq!(f.live_committed_bytes(), 42 * GIB);
    assert!(
        f.next_to_load().is_none(),
        "gemma4 does not fit and must NOT be brought up by evicting a peer",
    );
}

#[test]
fn loading_follows_declaration_priority() {
    let f = Fleet::declare(card(), roster());
    assert_eq!(
        f.next_to_load().map(|s| s.declared.model.as_str()),
        Some("muse-local:latest"),
        "first declared is first loaded",
    );
}

#[test]
fn a_model_already_coming_up_is_not_started_twice() {
    let mut f = Fleet::declare(card(), roster());
    f.observe("muse-local:latest", &Observation::LoadStarted, T0);
    assert_eq!(
        f.next_to_load().map(|s| s.declared.model.as_str()),
        Some("deepseek-r1:32b"),
        "move on to the next one rather than double-starting",
    );
}

#[test]
fn a_lost_model_is_eligible_to_come_back() {
    // A detached GPU may return. Steady state is: whatever can be warm, is.
    let mut f = Fleet::declare(card(), roster());
    warm(&mut f, "muse-local:latest", 8081);
    f.observe(
        "muse-local:latest",
        &Observation::ProbeFailed {
            gpu_present: false,
            detail: "gone".into(),
        },
        T0 + 100 * SEC,
    );
    assert_eq!(
        f.next_to_load().map(|s| s.declared.model.as_str()),
        Some("muse-local:latest"),
        "it freed its VRAM when it went, so it can be reloaded",
    );
}

#[test]
fn a_deliberately_stopped_model_stays_stopped() {
    // Otherwise the supervisor immediately restarts what the operator just
    // asked it to shut down, which is maddening and looks like a bug.
    let mut f = Fleet::declare(card(), roster());
    warm(&mut f, "muse-local:latest", 8081);
    f.observe(
        "muse-local:latest",
        &Observation::StopRequested,
        T0 + 50 * SEC,
    );
    assert_eq!(
        f.next_to_load().map(|s| s.declared.model.as_str()),
        Some("deepseek-r1:32b"),
        "never resurrect what was stopped on purpose",
    );
}

#[test]
fn nothing_to_do_is_the_normal_answer() {
    let mut f = Fleet::declare(card(), roster());
    warm(&mut f, "muse-local:latest", 8081);
    warm(&mut f, "deepseek-r1:32b", 8082);
    assert!(f.next_to_load().is_none(), "steady state should be boring");
}

#[test]
fn a_stray_observation_for_an_unserved_model_is_ignored_not_fatal() {
    let mut f = Fleet::declare(card(), roster());
    f.observe("something-else", &Observation::LoadStarted, T0);
    assert!(f.slot("something-else").is_none());
}

#[test]
fn the_report_shows_every_model_and_why() {
    let mut f = Fleet::declare(card(), roster());
    warm(&mut f, "muse-local:latest", 8081);
    f.observe("deepseek-r1:32b", &Observation::LoadStarted, T0);

    let r = f.report(T0 + 30 * SEC);
    assert!(r.contains("muse-local:latest"));
    assert!(r.contains("resident for"));
    assert!(r.contains("loading for"));
    assert!(
        r.contains("not admitted"),
        "gemma4 must appear WITH its reason"
    );
    assert!(r.contains("3 declared, 2 admitted"));
}
