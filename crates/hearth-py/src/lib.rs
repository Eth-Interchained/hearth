//! Python bindings.
//!
// pyo3's #[pyfunction] / #[pymethods] macros expand to a PyErr -> PyErr
// conversion that clippy flags as useless. It is useless, and it is also not
// ours — the alternative is contorting real signatures to satisfy a lint about
// generated code. Allowed here, narrowly, with the reason attached.
#![allow(clippy::useless_conversion)]
//!
//! Same discipline as the Node side: every rule lives in `hearth-core` and is
//! tested there once. This layer moves data and nothing else.
//!
//! Values cross as JSON strings and the thin `hearth` Python package turns
//! them into dicts. That keeps this file free of any conversion logic worth
//! getting wrong, which matters more than idiomatic signatures on a v0.1 —
//! three languages describing one state machine will drift, and the way you
//! prevent it is by having only one of them contain rules.

use hearth_core::budget::{declared_from_json, plan as core_plan, Budget, Declared};
use hearth_core::fleet::Fleet;
use hearth_core::residency::{observation_from_json, Millis};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{json, Value};

fn parse(s: &str) -> PyResult<Value> {
    serde_json::from_str(s).map_err(|e| PyValueError::new_err(format!("bad json: {e}")))
}

/// Parse a roster, or raise. The core owns the rules; this only turns a
/// failure into a Python exception. Never skips an entry it could not read —
/// a roster that silently shrinks produces a plan about the wrong question.
fn roster_from(v: &Value) -> PyResult<Vec<Declared>> {
    declared_from_json(v).map_err(|errs| {
        PyValueError::new_err(format!("could not read the roster: {}", errs.join("; ")))
    })
}

/// Can this card hold this roster? Returns a JSON string.
#[pyfunction]
#[pyo3(signature = (total_bytes, reserve_pct, declared_json))]
fn plan(total_bytes: u64, reserve_pct: u8, declared_json: &str) -> PyResult<String> {
    let budget = Budget::with_reserve_pct(total_bytes, reserve_pct.min(100));
    let roster = roster_from(&parse(declared_json)?)?;
    let p = core_plan(budget, &roster);
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
        "declared": roster.len(),
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
            inner: Fleet::declare(budget, roster_from(&parse(declared_json)?)?),
        })
    }

    fn set_endpoint(&mut self, model: &str, endpoint: &str) {
        self.inner.set_endpoint(model, endpoint);
    }

    /// kind: load_started · probe_ok · probe_failed · process_exited ·
    ///       load_failed · stop
    #[pyo3(signature = (model, kind, detail_json, now))]
    /// Record something observed. `kind` is one of:
    /// load_started · probe_ok · probe_failed · process_exited · load_failed · stop
    ///
    /// Either key spelling is accepted (`gpu_present` or `gpuPresent`,
    /// `vram_bytes` or `vramBytes`) because the mapping lives in the core now,
    /// once. It used to live here AND in the Node binding with different
    /// spellings, so a dict written the other language's way silently lost its
    /// most important field.
    ///
    /// Raises on an unrecognised `kind`; it used to return silently, which made
    /// a typo indistinguishable from a model that never changed state.
    fn observe(&mut self, model: &str, kind: &str, detail_json: &str, now: u64) -> PyResult<()> {
        let d = parse(detail_json)?;
        let obs = observation_from_json(kind, &d).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown observation kind {kind:?} — expected one of: \
                 load_started, probe_ok, probe_failed, process_exited, \
                 load_failed, stop"
            ))
        })?;
        self.inner.observe(model, &obs, now);
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
