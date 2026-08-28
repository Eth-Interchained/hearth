//! The fleet: every declared model, what state it is in, and what to tell a
//! caller who wants one.
//!
//! This is where the state machine stops being a curiosity and starts being
//! worth something. A router asks `route("deepseek-r1:32b")` and gets back a
//! NAMED answer — ready, warming for 43 seconds, the host took the GPU, it
//! never fit on this card — instead of being handed a socket that will hang
//! and calling the result a timeout.
//!
//! Every one of those answers implies a different action upstream:
//!
//! | answer          | what a router should do                              |
//! |-----------------|------------------------------------------------------|
//! | `Ready`         | send it                                              |
//! | `Warming`       | wait, or try another node — but do NOT fault this one |
//! | `Lost{Detached}`| try elsewhere, and do NOT score this operator down    |
//! | `Lost{Evicted}` | try elsewhere; this box is over-committed            |
//! | `NotAdmitted`   | never ask again — it does not fit here                |
//! | `Unknown`       | we genuinely do not know yet, and say so              |
//!
//! Today all six of those are one behaviour: connect, wait, fail, blame.

use serde::{Deserialize, Serialize};

use crate::budget::{plan, Budget, Declared, Plan};
use crate::residency::{LostReason, Millis, Observation, Residency};

/// One declared model and everything we know about it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Slot {
    pub declared: Declared,
    pub state: Residency,
    /// Where its server listens, once it has one.
    pub endpoint: Option<String>,
    /// Did it survive the budget plan? A declared model that does not fit is
    /// not a failure and not an outage — it is a configuration fact, and it
    /// deserves to be reported as one rather than as a permanent error.
    pub admitted: bool,
    /// How short it was, when it did not fit.
    pub short_bytes: u64,
}

/// The answer to "can I use this model right now?".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "route", rename_all = "snake_case")]
pub enum Route {
    Ready {
        model: String,
        endpoint: String,
    },
    /// Coming up. Carries elapsed so a caller can decide for itself whether to
    /// keep waiting — hearth never makes that call on someone else's behalf.
    Warming {
        model: String,
        for_ms: Millis,
    },
    Lost {
        model: String,
        reason: LostReason,
        /// False for a detached GPU. Routers must not score an operator down
        /// for a card their host reclaimed.
        operator_fault: bool,
    },
    Failed {
        model: String,
        reason: String,
    },
    /// Declared, but it does not fit on this card. Permanent until the
    /// declaration or the hardware changes — so a caller should stop asking.
    NotAdmitted {
        model: String,
        short_bytes: u64,
    },
    /// Not declared here at all.
    NotDeclared {
        model: String,
    },
    /// Declared, admitted, never observed. An honest "I don't know yet" beats
    /// a confident wrong answer in either direction.
    Unknown {
        model: String,
    },
}

impl Route {
    pub fn is_ready(&self) -> bool {
        matches!(self, Route::Ready { .. })
    }

    /// Should a router try a different operator for this?
    pub fn should_try_elsewhere(&self) -> bool {
        !matches!(self, Route::Ready { .. } | Route::Warming { .. })
    }

    /// Does this reflect badly on the operator? The default is NO — most ways
    /// a model is unavailable are nobody's fault, and a reputation system that
    /// assumes otherwise slowly deletes its own honest operators.
    pub fn operator_fault(&self) -> bool {
        match self {
            Route::Lost { operator_fault, .. } => *operator_fault,
            Route::Failed { .. } => true,
            _ => false,
        }
    }
}

/// Every declared model on one host, in priority order.
#[derive(Debug, Clone)]
pub struct Fleet {
    budget: Budget,
    slots: Vec<Slot>,
}

impl Fleet {
    /// Declare a roster. Order is priority order — the budget admits from the
    /// front, so the operator decides what matters, not a packing heuristic.
    pub fn declare(budget: Budget, declared: Vec<Declared>) -> Fleet {
        let p: Plan = plan(budget, &declared);
        let slots = declared
            .into_iter()
            .map(|d| {
                let rejection = p.rejected.iter().find(|r| r.model == d.model);
                Slot {
                    admitted: rejection.is_none(),
                    short_bytes: rejection.map(|r| r.short_bytes()).unwrap_or(0),
                    declared: d,
                    state: Residency::Unknown,
                    endpoint: None,
                }
            })
            .collect();
        Fleet { budget, slots }
    }

    pub fn budget(&self) -> Budget {
        self.budget
    }

    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    pub fn slot(&self, model: &str) -> Option<&Slot> {
        self.slots.iter().find(|s| s.declared.model == model)
    }

    /// VRAM actually held right now, measured from resident models rather than
    /// estimated from the declaration. The plan is what we intended; this is
    /// what is true.
    pub fn live_committed_bytes(&self) -> u64 {
        self.slots.iter().map(|s| s.state.vram_bytes()).sum()
    }

    pub fn live_free_bytes(&self) -> u64 {
        self.budget
            .usable_bytes()
            .saturating_sub(self.live_committed_bytes())
    }

    /// Record where a model's server is listening.
    pub fn set_endpoint(&mut self, model: &str, endpoint: impl Into<String>) {
        if let Some(s) = self.slots.iter_mut().find(|s| s.declared.model == model) {
            s.endpoint = Some(endpoint.into());
        }
    }

    /// Feed in something we observed. Unknown models are ignored rather than
    /// erroring: a stray probe for something we do not serve is noise, not a
    /// crash.
    pub fn observe(&mut self, model: &str, obs: &Observation, now: Millis) {
        if let Some(s) = self.slots.iter_mut().find(|s| s.declared.model == model) {
            s.state = s.state.observe(obs, now);
        }
    }

    /// The answer a router needs.
    pub fn route(&self, model: &str, now: Millis) -> Route {
        let Some(slot) = self.slot(model) else {
            return Route::NotDeclared {
                model: model.to_string(),
            };
        };
        if !slot.admitted {
            return Route::NotAdmitted {
                model: model.to_string(),
                short_bytes: slot.short_bytes,
            };
        }
        match &slot.state {
            // Resident but with nowhere to send traffic is not ready. Reporting
            // otherwise hands the caller a hole to fall into.
            Residency::Resident { .. } => match &slot.endpoint {
                Some(endpoint) => Route::Ready {
                    model: model.to_string(),
                    endpoint: endpoint.clone(),
                },
                None => Route::Unknown {
                    model: model.to_string(),
                },
            },
            Residency::Loading { .. } => Route::Warming {
                model: model.to_string(),
                for_ms: slot.state.loading_for(now).unwrap_or(0),
            },
            Residency::Lost { reason, .. } => Route::Lost {
                model: model.to_string(),
                reason: *reason,
                operator_fault: reason.is_operator_fault(),
            },
            Residency::Failed { reason, .. } => Route::Failed {
                model: model.to_string(),
                reason: reason.clone(),
            },
            Residency::Stopped { .. } | Residency::Unknown => Route::Unknown {
                model: model.to_string(),
            },
        }
    }

    /// Which model should we bring up next?
    ///
    /// Highest priority admitted model that is not already up or coming up, and
    /// that fits in the VRAM actually free right now. Returns `None` when
    /// everything that can be warm is warm — which is the steady state, and
    /// should be the boring answer almost always.
    ///
    /// Deliberately refuses to evict anything to make room. Eviction to satisfy
    /// a load is precisely the behaviour hearth exists to eliminate; if a model
    /// does not fit, the honest answer is that it does not fit.
    pub fn next_to_load(&self) -> Option<&Slot> {
        let free = self.live_free_bytes();
        self.slots.iter().find(|s| {
            s.admitted
                && !matches!(
                    s.state,
                    Residency::Resident { .. } | Residency::Loading { .. } | Residency::Stopped { .. }
                )
                && s.declared.total_bytes() <= free
        })
    }

    /// One line per model. This is what `hearth status` prints and what
    /// `/residency` serves — the same truth in both places, by construction.
    pub fn report(&self, now: Millis) -> String {
        let mut out = format!(
            "{:.1} / {:.1} GiB held  ({} declared, {} admitted)\n",
            crate::budget::gib(self.live_committed_bytes()),
            crate::budget::gib(self.budget.usable_bytes()),
            self.slots.len(),
            self.slots.iter().filter(|s| s.admitted).count(),
        );
        for s in &self.slots {
            let what = if s.admitted {
                s.state.explain(now)
            } else {
                format!(
                    "not admitted — short by {:.1} GiB",
                    crate::budget::gib(s.short_bytes)
                )
            };
            out.push_str(&format!("  {:<28} {}\n", s.declared.model, what));
        }
        out
    }
}
