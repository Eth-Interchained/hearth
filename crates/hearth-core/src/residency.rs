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
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Residency {
    /// Declared but never observed. Not a failure — an absence of information,
    /// and worth its own state so nobody reports a guess as a fact.
    ///
    /// The default, deliberately: a fresh slot knows nothing, and "nothing
    /// known" must never be confused with "known to be fine".
    #[default]
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

/// Read an `Observation` out of a `kind` plus a JSON detail object.
///
/// Lives here, in the core, because both bindings previously carried their own
/// copy of this mapping — and the copies disagreed. Node read `gpuPresent`,
/// Python read `gpu_present`, and each treated the *other* language's spelling
/// as an absent field. Since an absent `gpu_present` deliberately reads as
/// `true` (an unreadable fact must never exonerate an operator), a Python
/// caller who wrote `{"gpuPresent": false}` got `Evicted` with
/// `is_operator_fault() == true` — the opposite of what they meant, and no
/// error. One character of casing decided who took the blame for a card their
/// host reclaimed.
///
/// So: both spellings, from one function tested once. `budget::declared_from_json`
/// already worked this way for exactly this reason; this brings observations
/// in line with the roster.
///
/// Returns `None` for an unrecognised `kind` so a caller can say so out loud,
/// rather than silently doing nothing — which is what both bindings did.
pub fn observation_from_json(kind: &str, detail: &serde_json::Value) -> Option<Observation> {
    fn field<'a>(
        d: &'a serde_json::Value,
        camel: &str,
        snake: &str,
    ) -> Option<&'a serde_json::Value> {
        d.get(camel).or_else(|| d.get(snake))
    }

    fn detail_str(d: &serde_json::Value, fallback: &str) -> String {
        d.get("detail")
            .and_then(|v| v.as_str())
            .unwrap_or(fallback)
            .to_string()
    }

    Some(match kind {
        "load_started" | "loadStarted" => Observation::LoadStarted,
        "probe_ok" | "probeOk" => Observation::ProbeOk {
            vram_bytes: field(detail, "vramBytes", "vram_bytes")
                .and_then(crate::budget::whole_bytes)
                .unwrap_or(0),
        },
        "probe_failed" | "probeFailed" => Observation::ProbeFailed {
            // Absent — or unreadable — means "we could not tell", and the safe
            // reading of that is that the card was still there. A missing field
            // must never hand out an alibi.
            gpu_present: field(detail, "gpuPresent", "gpu_present")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            detail: detail_str(detail, ""),
        },
        "process_exited" | "processExited" => Observation::ProcessExited {
            code: detail
                .get("code")
                .and_then(|v| v.as_i64())
                .map(|c| c as i32),
        },
        "load_failed" | "loadFailed" => Observation::LoadFailed {
            detail: detail_str(detail, "unknown"),
        },
        "stop" | "stop_requested" | "stopRequested" => Observation::StopRequested,
        _ => return None,
    })
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

#[cfg(test)]
mod observation_json_tests {
    use super::*;
    use serde_json::json;

    // These are the highest-stakes assertions in the crate. `gpu_present`
    // decides Evicted vs GpuDetached, and `is_operator_fault` reads it
    // straight out — so a field lost in translation bills a provider's
    // decision to the operator.

    #[test]
    fn both_spellings_of_gpu_present_are_read() {
        for detail in [json!({"gpu_present": false}), json!({"gpuPresent": false})] {
            let obs = observation_from_json("probe_failed", &detail).expect("known kind");
            assert_eq!(
                obs,
                Observation::ProbeFailed {
                    gpu_present: false,
                    detail: String::new()
                },
                "spelling must not change the verdict: {detail}"
            );
        }
    }

    #[test]
    fn the_bug_this_function_exists_to_delete() {
        // Before this lived in the core, the Python binding read only
        // `gpu_present` and the Node binding read only `gpuPresent`. Each
        // treated the other's spelling as absent, and absent means "assume the
        // card was there" — so `{"gpuPresent": false}` in Python produced
        // Evicted, which IS the operator's fault. The opposite of the truth,
        // silently.
        let detail = json!({"gpuPresent": false, "detail": "no CUDA device"});
        let obs = observation_from_json("probe_failed", &detail).unwrap();

        let state = Residency::Unknown
            .observe(&Observation::LoadStarted, 0)
            .observe(&Observation::ProbeOk { vram_bytes: 1 }, 1_000)
            .observe(&obs, 2_000);

        match state {
            Residency::Lost { reason, .. } => {
                assert_eq!(reason, LostReason::GpuDetached);
                assert!(
                    !reason.is_operator_fault(),
                    "the host took the card — this must never count against them"
                );
            }
            other => panic!("expected a loss, got {other:?}"),
        }
    }

    #[test]
    fn a_missing_gpu_present_still_counts_against_us() {
        // Unchanged and deliberate: unreadable must never exonerate.
        let obs = observation_from_json("probe_failed", &json!({"detail": "?"})).unwrap();
        assert_eq!(
            obs,
            Observation::ProbeFailed {
                gpu_present: true,
                detail: "?".into()
            }
        );
    }

    #[test]
    fn an_unreadable_gpu_present_is_treated_as_missing_not_as_absence() {
        let obs = observation_from_json("probe_failed", &json!({"gpu_present": "nope"})).unwrap();
        assert_eq!(
            obs,
            Observation::ProbeFailed {
                gpu_present: true,
                detail: String::new()
            }
        );
    }

    #[test]
    fn a_js_float_vram_is_not_silently_zero() {
        // JS has no integers: 21 GiB arrives as an f64. A bare `as_u64()`
        // returns None for it, so the Node binding recorded a resident model
        // holding ZERO bytes — the budget then believed the card was empty and
        // would start models that could not fit. Same class of bug as the
        // 0.1.0 roster loss, in a different field.
        let gib = 1024u64 * 1024 * 1024;
        let as_float = json!({"vramBytes": (21 * gib) as f64});
        assert_eq!(
            observation_from_json("probe_ok", &as_float).unwrap(),
            Observation::ProbeOk {
                vram_bytes: 21 * gib
            },
        );
        // And the snake_case integer form, which always worked.
        assert_eq!(
            observation_from_json("probe_ok", &json!({"vram_bytes": 21 * gib})).unwrap(),
            Observation::ProbeOk {
                vram_bytes: 21 * gib
            },
        );
    }

    #[test]
    fn a_fractional_vram_is_refused_rather_than_truncated() {
        // 1.5 bytes is not a byte count. Truncating it would be inventing data.
        assert_eq!(
            observation_from_json("probe_ok", &json!({"vramBytes": 1.5})).unwrap(),
            Observation::ProbeOk { vram_bytes: 0 },
        );
    }

    #[test]
    fn every_kind_round_trips() {
        assert_eq!(
            observation_from_json("load_started", &json!({})),
            Some(Observation::LoadStarted)
        );
        assert_eq!(
            observation_from_json("process_exited", &json!({"code": 1})),
            Some(Observation::ProcessExited { code: Some(1) })
        );
        assert_eq!(
            observation_from_json("process_exited", &json!({})),
            Some(Observation::ProcessExited { code: None })
        );
        assert_eq!(
            observation_from_json("load_failed", &json!({"detail": "no such file"})),
            Some(Observation::LoadFailed {
                detail: "no such file".into()
            })
        );
        // A load failure with nothing said is still a load failure, and
        // "unknown" is a more honest detail than an empty string.
        assert_eq!(
            observation_from_json("load_failed", &json!({})),
            Some(Observation::LoadFailed {
                detail: "unknown".into()
            })
        );
        assert_eq!(
            observation_from_json("stop", &json!({})),
            Some(Observation::StopRequested)
        );
    }

    #[test]
    fn camelcase_kinds_work_too_because_one_language_writes_them_that_way() {
        assert_eq!(
            observation_from_json("loadStarted", &json!({})),
            Some(Observation::LoadStarted)
        );
        assert_eq!(
            observation_from_json("probeOk", &json!({"vramBytes": 8})),
            Some(Observation::ProbeOk { vram_bytes: 8 })
        );
    }

    #[test]
    fn an_unknown_kind_is_none_so_a_caller_can_say_so_out_loud() {
        // Both bindings used to `return` silently here, which made a typo
        // indistinguishable from a model that never changed state.
        assert_eq!(observation_from_json("probe_okay", &json!({})), None);
        assert_eq!(observation_from_json("", &json!({})), None);
    }

    #[test]
    fn a_non_object_detail_does_not_panic() {
        assert_eq!(
            observation_from_json("probe_failed", &json!(null)).unwrap(),
            Observation::ProbeFailed {
                gpu_present: true,
                detail: String::new()
            }
        );
    }
}
