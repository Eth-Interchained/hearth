//! VRAM accounting. Refuse to overcommit instead of thrashing.
//!
//! The failure this prevents is quiet and expensive. Declare four 20GB models
//! on a 48GB card and nothing errors — the runtime simply loads, evicts,
//! loads, evicts, forever. Every request pays a cold start, and from the
//! outside it looks like the models got slow. Nobody sees an out-of-memory,
//! because technically nothing ever ran out of memory: it just never kept
//! anything.
//!
//! A card has a fixed size. That is arithmetic, and arithmetic can be checked
//! at declare time, before a single weight is read from disk. So hearth
//! refuses the fifth model and says exactly how much it was over by, which is
//! a sentence an operator can act on. "The models got slow" is not.
//!
//! ## Headroom is not pessimism
//!
//! Weights are not the whole cost. KV cache scales with context length and
//! parallel slots, the CUDA context itself costs hundreds of megabytes, and
//! fragmentation is real on a card that has been up for weeks. Planning to
//! 100% of VRAM means the last model loads and then the first long context
//! kills it. The reserve exists so a working fleet stays working.

use serde::{Deserialize, Serialize};

/// A model we intend to keep resident, and what we expect it to cost.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Declared {
    pub model: String,
    /// Weights, in bytes. Measured if we have ever loaded it, estimated from
    /// the file otherwise.
    pub weights_bytes: u64,
    /// Expected KV cache at the configured context and parallelism. Separate
    /// from weights because it is the part that grows with how you USE the
    /// model, and the part people forget.
    pub kv_bytes: u64,
}

impl Declared {
    pub fn total_bytes(&self) -> u64 {
        self.weights_bytes.saturating_add(self.kv_bytes)
    }
}

/// The VRAM available and what we hold back.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    pub total_bytes: u64,
    /// Never planned into. Covers CUDA context, fragmentation, and the honest
    /// error in every size estimate here.
    pub reserve_bytes: u64,
}

pub const GIB: u64 = 1024 * 1024 * 1024;

impl Budget {
    /// A budget with a proportional reserve, floored so small cards are not
    /// left with a reserve too thin to cover a CUDA context.
    pub fn with_reserve_pct(total_bytes: u64, pct: u8) -> Budget {
        let pct = pct.min(100) as u64;
        let proportional = total_bytes / 100 * pct;
        Budget {
            total_bytes,
            reserve_bytes: proportional.max(GIB).min(total_bytes),
        }
    }

    /// What is actually available to models.
    pub fn usable_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.reserve_bytes)
    }
}

/// The verdict on a declared set. Never a bare bool — the numbers are the
/// point, because "you are 14.2 GiB over" is actionable and "invalid
/// configuration" is not.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    /// Models that fit, in the order given. Declaration order is priority
    /// order: the operator says what matters most, not a heuristic.
    pub admitted: Vec<String>,
    /// Models that do not fit, each with what it needed.
    pub rejected: Vec<Rejection>,
    pub committed_bytes: u64,
    pub usable_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rejection {
    pub model: String,
    pub needed_bytes: u64,
    /// How much room was left when we got to it.
    pub free_bytes: u64,
}

impl Rejection {
    pub fn short_bytes(&self) -> u64 {
        self.needed_bytes.saturating_sub(self.free_bytes)
    }
}

impl Plan {
    pub fn fits(&self) -> bool {
        self.rejected.is_empty()
    }

    pub fn headroom_bytes(&self) -> u64 {
        self.usable_bytes.saturating_sub(self.committed_bytes)
    }

    /// A report an operator can act on without reading the source.
    pub fn explain(&self) -> String {
        let mut out = format!(
            "{} of {} admitted, {:.1} GiB committed of {:.1} GiB usable",
            self.admitted.len(),
            self.admitted.len() + self.rejected.len(),
            gib(self.committed_bytes),
            gib(self.usable_bytes),
        );
        for r in &self.rejected {
            out.push_str(&format!(
                "\n  REJECTED {} — needs {:.1} GiB, {:.1} GiB free, short by {:.1} GiB",
                r.model,
                gib(r.needed_bytes),
                gib(r.free_bytes),
                gib(r.short_bytes()),
            ));
        }
        out
    }
}

pub fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

/// The total KV-cache context a running server will actually hold, given
/// hearth's own `--ctx` flag and the GGUF's native context length.
///
/// The two cases are NOT the same shape, and getting them backwards is how
/// this went uncaught: hearth's `--ctx` is documented — correctly — as the
/// TOTAL context, pre-divided across `--parallel` slots by llama-server
/// itself (`--ctx 8192 --parallel 4` gives each caller 2048, not 8192). But
/// when `--ctx` is left at 0, llama-server does not divide anything; each of
/// the `parallel` slots gets its own full copy of the model's native
/// `context_length`. The total footprint in that case is the native length
/// MULTIPLIED by slot count, not left alone.
///
///   ctx explicit (> 0): total = ctx                (already pre-divided)
///   ctx unset (== 0):   total = native_ctx * parallel   (each slot gets one)
///
/// Measured against the actual `n_ctx_slot = 4096` server log line at
/// `--parallel 8` with no `--ctx` passed: that is a per-slot number that did
/// not shrink as slots were added, which only makes sense if the total grew
/// with them.
pub fn total_ctx_tokens(ctx_flag: u32, native_context_length: u64, parallel: u32) -> u64 {
    if ctx_flag > 0 {
        ctx_flag as u64
    } else {
        native_context_length.saturating_mul(parallel.max(1) as u64)
    }
}

/// Decide what can be held resident, in declaration order.
///
/// First fit, not best fit, and deliberately so. Reordering to squeeze in one
/// more model would silently demote the model the operator listed first, and
/// on a serving box the first model is first because it matters most. A
/// planner that outsmarts the operator is a planner that surprises them at
/// 3am.
pub fn plan(budget: Budget, declared: &[Declared]) -> Plan {
    let usable = budget.usable_bytes();
    let mut committed: u64 = 0;
    let mut admitted = Vec::new();
    let mut rejected = Vec::new();

    for d in declared {
        let need = d.total_bytes();
        let free = usable.saturating_sub(committed);
        if need <= free {
            committed += need;
            admitted.push(d.model.clone());
        } else {
            rejected.push(Rejection {
                model: d.model.clone(),
                needed_bytes: need,
                free_bytes: free,
            });
        }
    }

    Plan {
        admitted,
        rejected,
        committed_bytes: committed,
        usable_bytes: usable,
    }
}

/// Read a declared roster out of JSON.
///
/// This lives in the core, not in the bindings, because parsing is a RULE and
/// rules belong where they can be tested. It got moved here after the Node
/// binding silently dropped an entire roster and then cheerfully reported
/// `fits: true` — JavaScript has no integers, so every size arrived as an f64
/// and `as_u64()` returned None for all of them. A `filter_map` swallowed the
/// lot. An empty roster trivially fits, so the answer was confident, instant
/// and completely wrong.
///
/// Two rules came out of that, and both are load-bearing:
///
///   1. Accept a float that is exactly an integer. A caller in a language
///      without integers is not making a mistake by sending 2.147e10.
///   2. NEVER skip an entry you could not read. Every failure is named and
///      returned, because a roster that silently shrinks produces a plan that
///      is right about the wrong question.
pub fn declared_from_json(v: &serde_json::Value) -> Result<Vec<Declared>, Vec<String>> {
    let Some(items) = v.as_array() else {
        return Err(vec!["expected an array of declared models".into()]);
    };

    let mut out = Vec::with_capacity(items.len());
    let mut errors = Vec::new();

    for (i, item) in items.iter().enumerate() {
        match one_declared(item) {
            Ok(d) => out.push(d),
            Err(why) => errors.push(format!("entry {i}: {why}")),
        }
    }

    if errors.is_empty() {
        Ok(out)
    } else {
        Err(errors)
    }
}

/// A size in bytes, from either an integer or an integer-valued float.
///
/// Public because the bindings need exactly this leniency for every byte count
/// that crosses a language boundary, not just roster entries. JS has no
/// integers: every number arrives as `f64`, so a bare `as_u64()` returns `None`
/// for a perfectly good size and the value silently becomes zero. That bug ate
/// whole rosters in 0.1.0 and was still eating `vramBytes` on `probe_ok` after.
pub fn whole_bytes(v: &serde_json::Value) -> Option<u64> {
    size_of(Some(v))
}

/// A size in bytes, from either an integer or an integer-valued float.
fn size_of(v: Option<&serde_json::Value>) -> Option<u64> {
    let v = v?;
    if let Some(n) = v.as_u64() {
        return Some(n);
    }
    // JS numbers are f64. Accept one only when it is genuinely a whole
    // non-negative count — silently truncating 1.5 bytes would be inventing
    // data, and a negative size is a bug worth surfacing rather than clamping.
    let f = v.as_f64()?;
    if f.is_finite() && f >= 0.0 && f.fract() == 0.0 {
        Some(f as u64)
    } else {
        None
    }
}

fn one_declared(v: &serde_json::Value) -> Result<Declared, String> {
    let model = v
        .get("model")
        .and_then(|m| m.as_str())
        .ok_or("missing \"model\"")?;
    if model.is_empty() {
        return Err("\"model\" is empty".into());
    }
    // Both spellings, because the same core serves a camelCase language and a
    // snake_case one and neither should have to translate.
    let weights = size_of(v.get("weightsBytes").or_else(|| v.get("weights_bytes")))
        .ok_or_else(|| format!("{model}: \"weightsBytes\" missing or not a whole byte count"))?;
    let kv = match v.get("kvBytes").or_else(|| v.get("kv_bytes")) {
        None | Some(serde_json::Value::Null) => 0,
        some => size_of(some)
            .ok_or_else(|| format!("{model}: \"kvBytes\" is not a whole byte count"))?,
    };
    Ok(Declared {
        model: model.to_string(),
        weights_bytes: weights,
        kv_bytes: kv,
    })
}

#[cfg(test)]
mod ctx_tests {
    use super::total_ctx_tokens;

    #[test]
    fn explicit_ctx_is_already_the_total() {
        // hearth's own documented contract for --ctx: pre-divided across
        // slots by llama-server. The total does not grow with parallel.
        assert_eq!(total_ctx_tokens(8192, 32_768, 4), 8192);
        assert_eq!(total_ctx_tokens(8192, 32_768, 1), 8192);
    }

    #[test]
    fn unset_ctx_multiplies_native_length_by_parallel() {
        // The behaviour that produced the incident: each slot gets its own
        // full copy of the model's native window when nothing was passed.
        assert_eq!(total_ctx_tokens(0, 4096, 8), 32_768);
    }

    #[test]
    fn zero_parallel_is_treated_as_one_slot_not_zero_context() {
        // A caller that forgot to set parallel should not silently zero out
        // the whole KV budget and let everything past it look free.
        assert_eq!(total_ctx_tokens(0, 4096, 0), 4096);
    }

    #[test]
    fn explicit_ctx_ignores_parallel_entirely() {
        assert_eq!(total_ctx_tokens(1024, 999_999, 64), 1024);
    }
}
