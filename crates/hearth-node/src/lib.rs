//! Node bindings.
//!
//! Deliberately thin. Every rule about residency and VRAM lives in
//! `hearth-core` and is tested there once, in Rust — a binding that
//! reimplements any of it is a second source of truth waiting to disagree
//! with the first. What crosses the boundary is data.
//!
//! JSON in, JSON out, converted to real JS objects by napi's serde support.
//! Structural rather than idiomatic on purpose: three languages describing the
//! same state machine will drift, and JSON is the one shape all three already
//! agree on.

use hearth_core::budget::{plan as core_plan, Budget, Declared};
use hearth_core::fleet::Fleet;
use hearth_core::residency::{Millis, Observation};
use napi_derive::napi;
use serde_json::{json, Value};

fn declared_from(v: &Value) -> Option<Declared> {
    Some(Declared {
        model: v.get("model")?.as_str()?.to_string(),
        weights_bytes: v
            .get("weightsBytes")
            .or_else(|| v.get("weights_bytes"))?
            .as_u64()?,
        kv_bytes: v
            .get("kvBytes")
            .or_else(|| v.get("kv_bytes"))
            .and_then(|k| k.as_u64())
            .unwrap_or(0),
    })
}

fn roster_from(v: &Value) -> Vec<Declared> {
    v.as_array()
        .map(|a| a.iter().filter_map(declared_from).collect())
        .unwrap_or_default()
}

/// Can this card hold this roster? Answers before anything loads.
#[napi]
pub fn plan(total_bytes: i64, reserve_pct: u32, declared: Value) -> Value {
    let budget = Budget::with_reserve_pct(total_bytes.max(0) as u64, reserve_pct.min(100) as u8);
    let roster = roster_from(&declared);
    let p = core_plan(budget, &roster);
    json!({
        "fits": p.fits(),
        "admitted": p.admitted,
        "rejected": p.rejected.iter().map(|r| json!({
            "model": r.model,
            "neededBytes": r.needed_bytes,
            "freeBytes": r.free_bytes,
            "shortBytes": r.short_bytes(),
        })).collect::<Vec<_>>(),
        "committedBytes": p.committed_bytes,
        "usableBytes": p.usable_bytes,
        "headroomBytes": p.headroom_bytes(),
        "explain": p.explain(),
    })
}

/// A live fleet. Declare once, feed observations, ask for routes.
#[napi]
pub struct HearthFleet {
    inner: Fleet,
}

#[napi]
impl HearthFleet {
    #[napi(constructor)]
    pub fn new(total_bytes: i64, reserve_pct: u32, declared: Value) -> Self {
        let budget =
            Budget::with_reserve_pct(total_bytes.max(0) as u64, reserve_pct.min(100) as u8);
        HearthFleet {
            inner: Fleet::declare(budget, roster_from(&declared)),
        }
    }

    #[napi]
    pub fn set_endpoint(&mut self, model: String, endpoint: String) {
        self.inner.set_endpoint(&model, endpoint);
    }

    /// Record something observed. `kind` is one of:
    /// load_started · probe_ok · probe_failed · process_exited · load_failed · stop
    ///
    /// `gpuPresent` on a probe_failed is the single most important field that
    /// crosses this boundary: it is what separates the runtime dropping a model
    /// from the host taking the card away.
    #[napi]
    pub fn observe(&mut self, model: String, kind: String, detail: Value, now: i64) {
        let now = now.max(0) as Millis;
        let obs = match kind.as_str() {
            "load_started" => Observation::LoadStarted,
            "probe_ok" => Observation::ProbeOk {
                vram_bytes: detail
                    .get("vramBytes")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
            },
            "probe_failed" => Observation::ProbeFailed {
                // Absent means "we could not tell", and the safe reading of
                // "could not tell" is that the GPU is still there — that keeps
                // a missing field from silently exonerating an operator.
                gpu_present: detail
                    .get("gpuPresent")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
                detail: detail
                    .get("detail")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            },
            "process_exited" => Observation::ProcessExited {
                code: detail
                    .get("code")
                    .and_then(|v| v.as_i64())
                    .map(|c| c as i32),
            },
            "load_failed" => Observation::LoadFailed {
                detail: detail
                    .get("detail")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
            },
            "stop" => Observation::StopRequested,
            _ => return,
        };
        self.inner.observe(&model, &obs, now);
    }

    /// What should a router do with a request for this model?
    #[napi]
    pub fn route(&self, model: String, now: i64) -> Value {
        let r = self.inner.route(&model, now.max(0) as Millis);
        let mut v = serde_json::to_value(&r).unwrap_or(Value::Null);
        if let Value::Object(ref mut m) = v {
            m.insert("ready".into(), json!(r.is_ready()));
            m.insert("tryElsewhere".into(), json!(r.should_try_elsewhere()));
            m.insert("operatorFault".into(), json!(r.operator_fault()));
        }
        v
    }

    /// The next model worth bringing up, or null when everything that can be
    /// warm already is.
    #[napi]
    pub fn next_to_load(&self) -> Option<String> {
        self.inner.next_to_load().map(|s| s.declared.model.clone())
    }

    #[napi]
    pub fn committed_bytes(&self) -> i64 {
        self.inner.live_committed_bytes() as i64
    }

    #[napi]
    pub fn free_bytes(&self) -> i64 {
        self.inner.live_free_bytes() as i64
    }

    /// Everything, as one human-readable block. Same text `hearth status`
    /// prints, because one truth in two places is how they stop matching.
    #[napi]
    pub fn report(&self, now: i64) -> String {
        self.inner.report(now.max(0) as Millis)
    }
}
