//! What state a model is actually in, and how it got there.
//!
//! This is the whole reason hearth exists. Every serving stack can tell you
//! "the request failed." None of them can tell you WHY a model that was warm
//! thirty seconds ago isn't anymore — and the two most common causes are
//! completely different problems with completely different fixes:
//!
//!   * The runtime evicted it. An LRU cache decided something else was more
//!     deserving of the VRAM. Your fix is capacity or configuration.
//!   * The GPU was taken away underneath the process. On a virtualized GPU
//!     host the card detaches when your workload looks idle and gets handed
//!     to another tenant. Your fix is with your provider, and no amount of
//!     configuration on your side will touch it.
//!
//! Today both of those surface as a timeout. A timeout is not a diagnosis, it
//! is the absence of one — and an operator marked unreliable for a cache
//! policy it never chose is a lie recorded as data.
//!
//! So residency is a state machine with named states and named reasons, and
//! the reason survives all the way out to the caller.
//!
//! ## On clocks
//!
//! Nothing here fails a model for taking too long. A 32B model materializing
//! over a network fabric can legitimately spend minutes before its first
//! token, and killing it at an arbitrary deadline turns a slow success into a
//! fast failure. `Loading` carries how long it has been loading; deciding what
//! to do about that belongs to the caller, who knows whether a human is
//! waiting. Progress is reported. Patience is a policy, not a constant.

use serde::{Deserialize, Serialize};

/// Milliseconds since the unix epoch. Passed in explicitly, never read from a
/// global clock, so every transition in this file is a pure function and the
/// tests do not sleep.
pub type Millis = u64;

/// Why a model stopped being resident. The distinction IS the product.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LostReason {
    /// The runtime unloaded it to make room for something else. Ours to fix:
    /// too many models declared for the VRAM available.
    Evicted,
    /// The GPU itself went away while the process kept running. Not ours to
    /// fix — this is the host reclaiming a card. Reporting it honestly is the
    /// only correct behaviour.
    GpuDetached,
    /// The serving process died.
    ProcessExited,
    /// It answered, but wrongly or not at all, while the GPU was still present
    /// and the process still alive. Something is broken rather than absent.
    Unhealthy,
}

impl LostReason {
    /// Whether an operator should be held responsible for this in reputation.
    ///
    /// `GpuDetached` must never count against anyone. It is the defining case:
    /// the host took the card, the operator did nothing wrong, and scoring them
    /// down for it corrupts the only signal the marketplace has.
    pub fn is_operator_fault(self) -> bool {
        !matches!(self, LostReason::GpuDetached)
    }

    /// Whether re-asking the same node immediately could plausibly work.
    /// A detached GPU may come straight back; an over-committed box will just
    /// evict something else, so retrying there makes the fleet worse.
    pub fn worth_retrying_here(self) -> bool {
        matches!(self, LostReason::GpuDetached | LostReason::ProcessExited)
    }
}

/// Where a model stands right now.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Residency {
    /// Declared but never observed. Not a failure — an absence of information,
    /// and worth its own state so nobody reports a guess as a fact.
    Unknown,
    /// Weights are materializing. Never fatal, however long it takes.
    Loading { since: Millis },
    /// Loaded, and it answered a probe. Both halves are required: a process
    /// holding VRAM that cannot answer is not serving anyone.
    Resident { since: Millis, vram_bytes: u64 },
    /// It was resident and now is not, and we did not ask for that.
    Lost { at: Millis, reason: LostReason },
    /// It will not load, and here is what the runtime said.
    Failed { at: Millis, reason: String },
    /// We unloaded it deliberately. The only state that is nobody's problem.
    Stopped { at: Millis },
}

/// Something we observed about a model. Facts, not conclusions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Observation {
    /// A serving process was started for this model.
    LoadStarted,
    /// It answered a health probe, holding this much VRAM.
    ProbeOk { vram_bytes: u64 },
    /// It failed a probe. `gpu_present` is what separates "the runtime dropped
    /// the model" from "the card is gone", and it is the single most valuable
    /// bit in this entire file.
    ProbeFailed { gpu_present: bool, detail: String },
    /// The serving process exited.
    ProcessExited { code: Option<i32> },
    /// The runtime refused to load it at all.
    LoadFailed { detail: String },
    /// We asked for it to be unloaded.
    StopRequested,
}

impl Default for Residency {
    fn default() -> Self {
        Residency::Unknown
    }
}

impl Residency {
    /// Is this model able to serve a request right now?
    ///
    /// Deliberately narrow. `Loading` is NOT ready — the most common way a
    /// serving stack lies is by routing to something that is still coming up
    /// and calling the resulting timeout an error.
    pub fn is_ready(&self) -> bool {
        matches!(self, Residency::Resident { .. })
    }

    /// Is something actively happening that will plausibly end in readiness?
    /// A caller can wait on this instead of retrying elsewhere.
    pub fn is_coming(&self) -> bool {
        matches!(self, Residency::Loading { .. })
    }

    /// How long it has been loading, if it is. The caller decides what is too
    /// long — this only reports.
    pub fn loading_for(&self, now: Millis) -> Option<Millis> {
        match self {
            Residency::Loading { since } => Some(now.saturating_sub(*since)),
            _ => None,
        }
    }

    /// VRAM currently attributable to this model. Only a resident model holds
    /// any; a loading one has not finished claiming it and must not be counted
    /// against the budget as though it had.
    pub fn vram_bytes(&self) -> u64 {
        match self {
            Residency::Resident { vram_bytes, .. } => *vram_bytes,
            _ => 0,
        }
    }

    /// A short, honest phrase for a human or an API. Never "timeout".
    pub fn explain(&self, now: Millis) -> String {
        match self {
            Residency::Unknown => "never probed".into(),
            Residency::Loading { .. } => {
                let secs = self.loading_for(now).unwrap_or(0) / 1000;
                format!("loading for {secs}s")
            }
            Residency::Resident { since, .. } => {
                format!("resident for {}s", now.saturating_sub(*since) / 1000)
            }
            Residency::Lost { reason, .. } => match reason {
                LostReason::Evicted => "evicted by the runtime to free VRAM".into(),
                LostReason::GpuDetached => "the GPU was detached by the host".into(),
                LostReason::ProcessExited => "the serving process exited".into(),
                LostReason::Unhealthy => "loaded but not answering".into(),
            },
            Residency::Failed { reason, .. } => format!("failed to load: {reason}"),
            Residency::Stopped { .. } => "stopped on request".into(),
        }
    }

    /// Apply an observation. Returns the new state.
    ///
    /// Total and deterministic: every state accepts every observation, because
    /// the real world will deliver them in orders nobody planned for, and a
    /// supervisor that panics on a surprising sequence is worse than one that
    /// records something slightly odd.
    pub fn observe(&self, obs: &Observation, now: Millis) -> Residency {
        match obs {
            // An explicit stop always wins. It is the one transition the
            // operator asked for, so it can never be a failure.
            Observation::StopRequested => Residency::Stopped { at: now },

            Observation::LoadStarted => match self {
                // Already up and answering: a spurious start changes nothing.
                // Re-entering Loading here would make a healthy model look
                // unavailable for no reason.
                Residency::Resident { .. } => self.clone(),
                Residency::Loading { .. } => self.clone(),
                _ => Residency::Loading { since: now },
            },

            Observation::ProbeOk { vram_bytes } => match self {
                // Keep the original `since` so "resident for 4 hours" stays
                // true across thousands of successful probes.
                Residency::Resident { since, .. } => Residency::Resident {
                    since: *since,
                    vram_bytes: *vram_bytes,
                },
                _ => Residency::Resident {
                    since: now,
                    vram_bytes: *vram_bytes,
                },
            },

            Observation::ProbeFailed {
                gpu_present,
                detail: _,
            } => {
                let reason = if !*gpu_present {
                    // The card is gone. Nothing on this box did anything wrong.
                    LostReason::GpuDetached
                } else if self.is_ready() {
                    // It was resident, the GPU is still here, and now it is not
                    // answering: the runtime dropped it underneath us.
                    LostReason::Evicted
                } else {
                    LostReason::Unhealthy
                };
                match self {
                    // A probe that fails while still loading is expected — the
                    // server is not up yet. Only demote if the GPU vanished,
                    // which is real news at any point in the lifecycle.
                    Residency::Loading { .. } if *gpu_present => self.clone(),
                    _ => Residency::Lost { at: now, reason },
                }
            }

            Observation::ProcessExited { .. } => match self {
                // Exiting after a deliberate stop is the expected epilogue, not
                // a loss. Without this, every clean shutdown files a fault.
                Residency::Stopped { .. } => self.clone(),
                _ => Residency::Lost {
                    at: now,
                    reason: LostReason::ProcessExited,
                },
            },

            Observation::LoadFailed { detail } => Residency::Failed {
                at: now,
                reason: detail.clone(),
            },
        }
    }
}
