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
