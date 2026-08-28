//! Python bindings.
//!
//! Same discipline as the Node side: every rule lives in `hearth-core` and is
//! tested there once. This layer moves data and nothing else.
//!
//! Values cross as JSON strings and the thin `hearth` Python package turns
//! them into dicts. That keeps this file free of any conversion logic worth
//! getting wrong, which matters more than idiomatic signatures on a v0.1 —
//! three languages describing one state machine will drift, and the way you
//! prevent it is by having only one of them contain rules.

use hearth_core::budget::{plan as core_plan, Budget, Declared};
use hearth_core::fleet::Fleet;
use hearth_core::residency::{Millis, Observation};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{json, Value};

fn parse(s: &str) -> PyResult<Value> {
    serde_json::from_str(s).map_err(|e| PyValueError::new_err(format!("bad json: {e}")))
}

fn roster_from(v: &Value) -> Vec<Declared> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|d| {
                    Some(Declared {
                        model: d.get("model")?.as_str()?.to_string(),
                        weights_bytes: d.get("weights_bytes")?.as_u64()?,
                        kv_bytes: d.get("kv_bytes").and_then(|k| k.as_u64()).unwrap_or(0),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Can this card hold this roster? Returns a JSON string.
#[pyfunction]
#[pyo3(signature = (total_bytes, reserve_pct, declared_json))]
fn plan(total_bytes: u64, reserve_pct: u8, declared_json: &str) -> PyResult<String> {
    let budget = Budget::with_reserve_pct(total_bytes, reserve_pct.min(100));
    let p = core_plan(budget, &roster_from(&parse(declared_json)?));
    Ok(json!({
        "fits": p.fits(),
        "admitted": p.admitted,
        "rejected": p.rejected.iter().map(|r| json!({
            "model": r.model,
            "needed_bytes": r.needed_bytes,
            "free_bytes": r.free_bytes,
            "short_bytes": r.short_bytes(),
        })).collect::<Vec<_>>(),
        "committed_bytes": p.committed_bytes,
        "usable_bytes": p.usable_bytes,
        "headroom_bytes": p.headroom_bytes(),
        "explain": p.explain(),
    })
    .to_string())
}

#[pyclass(name = "Fleet")]
struct PyFleet {
    inner: Fleet,
}

#[pymethods]
impl PyFleet {
    #[new]
    #[pyo3(signature = (total_bytes, reserve_pct, declared_json))]
    fn new(total_bytes: u64, reserve_pct: u8, declared_json: &str) -> PyResult<Self> {
        let budget = Budget::with_reserve_pct(total_bytes, reserve_pct.min(100));
        Ok(PyFleet {
            inner: Fleet::declare(budget, roster_from(&parse(declared_json)?)),
        })
    }

    fn set_endpoint(&mut self, model: &str, endpoint: &str) {
        self.inner.set_endpoint(model, endpoint);
    }

    /// kind: load_started · probe_ok · probe_failed · process_exited ·
    ///       load_failed · stop
    #[pyo3(signature = (model, kind, detail_json, now))]
    fn observe(&mut self, model: &str, kind: &str, detail_json: &str, now: u64) -> PyResult<()> {
        let d = parse(detail_json)?;
        let obs = match kind {
            "load_started" => Observation::LoadStarted,
            "probe_ok" => Observation::ProbeOk {
                vram_bytes: d.get("vram_bytes").and_then(|v| v.as_u64()).unwrap_or(0),
            },
            "probe_failed" => Observation::ProbeFailed {
                // Missing means "could not tell", and the safe reading of that
                // is the GPU is still present — a missing field must never
                // silently exonerate an operator.
                gpu_present: d
                    .get("gpu_present")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true),
                detail: d
                    .get("detail")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
            },
            "process_exited" => Observation::ProcessExited {
                code: d.get("code").and_then(|v| v.as_i64()).map(|c| c as i32),
            },
            "load_failed" => Observation::LoadFailed {
                detail: d
                    .get("detail")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string(),
            },
            "stop" => Observation::StopRequested,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown observation: {other}"
                )))
            }
        };
        self.inner.observe(model, &obs, now as Millis);
        Ok(())
    }

    fn route(&self, model: &str, now: u64) -> PyResult<String> {
        let r = self.inner.route(model, now as Millis);
        let mut v = serde_json::to_value(&r)
            .map_err(|e| PyValueError::new_err(format!("serialize: {e}")))?;
        if let Value::Object(ref mut m) = v {
            m.insert("ready".into(), json!(r.is_ready()));
            m.insert("try_elsewhere".into(), json!(r.should_try_elsewhere()));
            m.insert("operator_fault".into(), json!(r.operator_fault()));
        }
        Ok(v.to_string())
    }

    fn next_to_load(&self) -> Option<String> {
        self.inner.next_to_load().map(|s| s.declared.model.clone())
    }

    fn committed_bytes(&self) -> u64 {
        self.inner.live_committed_bytes()
    }

    fn free_bytes(&self) -> u64 {
        self.inner.live_free_bytes()
    }

    fn report(&self, now: u64) -> String {
        self.inner.report(now as Millis)
    }
}

#[pymodule]
fn _hearth(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    m.add_class::<PyFleet>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
