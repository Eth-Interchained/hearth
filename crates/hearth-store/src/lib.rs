//! hearth-store — the NEDB spine.
//!
//! Every residency transition is a bi-temporal, causally-linked,
//! tamper-evident event in an embedded NEDB database. This is not a log
//! bolted onto the side of the supervisor: it IS the supervisor's state.
//!
//! Three properties no other serving stack has, inherited for free:
//!
//!   * `state_as_of(model, seq)` — what was resident *then*, not now.
//!   * `why(model)` — a causal chain from the current state back through
//!     every transition that produced it, not a guess from grep.
//!   * `verify()` — cryptographic proof the history wasn't rewritten.
//!
//! Layout: one versioned document per model in the `residency` collection
//! (id = model name). Each transition is a new version of that document,
//! with `caused_by` linking to the event that produced it — a load links
//! to the pull that fetched the bytes, a loss links to the load it undid.
//! NEDB's version chain gives history; its DAG gives causality; its
//! Merkle head gives tamper-evidence.

use std::path::Path;

use hearth_core::{LostReason, Residency};
use nedb_engine::db::Db;
use nedb_engine::store::Node;
use serde_json::{json, Value};

/// Collection holding one versioned doc per model.
const RESIDENCY: &str = "residency";
/// Collection holding one versioned doc per model for pulls.
const PULLS: &str = "pulls";

/// A recorded event: the node hash (for causal linking) and its seq
/// (for AS OF reads). Everything a caller needs to chain the next event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventRef {
    pub hash: String,
    pub seq: u64,
}

impl EventRef {
    fn of(node: &Node) -> Self {
        EventRef {
            hash: node.hash.clone(),
            seq: node.seq,
        }
    }
}

/// What kind of transition is being recorded.
///
/// These mirror `hearth_core::Residency` but carry the extra facts the
/// state machine doesn't hold (endpoint, pid, timing) — the store is the
/// system of record, the state machine is the decision procedure.
#[derive(Debug, Clone, PartialEq)]
pub enum Transition {
    /// Model declared into the fleet. `admitted: false` means the budget
    /// refused it — recorded anyway, because a refusal is a fact too.
    Declared { vram_bytes: u64, admitted: bool },
    /// Bytes are being fetched. Source is the resolved origin
    /// ("ollama:library/llama3:latest", "hf:owner/repo@Q4_K_M", "file:...").
    PullStarted { source: String },
    /// Bytes landed and digest-verified.
    PullCompleted { path: String, size_bytes: u64 },
    /// A runtime child was spawned and is warming.
    Loading { pid: u32, endpoint: String },
    /// The runtime answered its health probe.
    Resident { endpoint: String, warmup_ms: u64 },
    /// The model stopped being resident, with the named reason.
    Lost { reason: LostReason },
    /// Deliberate operator stop. Not a loss — an instruction.
    Unloaded,
}

impl Transition {
    fn kind(&self) -> &'static str {
        match self {
            Transition::Declared { .. } => "declared",
            Transition::PullStarted { .. } => "pull_started",
            Transition::PullCompleted { .. } => "pull_completed",
            Transition::Loading { .. } => "loading",
            Transition::Resident { .. } => "resident",
            Transition::Lost { .. } => "lost",
            Transition::Unloaded => "unloaded",
        }
    }

    fn facts(&self) -> Value {
        match self {
            Transition::Declared {
                vram_bytes,
                admitted,
            } => json!({ "vram_bytes": vram_bytes, "admitted": admitted }),
            Transition::PullStarted { source } => json!({ "source": source }),
            Transition::PullCompleted { path, size_bytes } => {
                json!({ "path": path, "size_bytes": size_bytes })
            }
            Transition::Loading { pid, endpoint } => {
                json!({ "pid": pid, "endpoint": endpoint })
            }
            Transition::Resident {
                endpoint,
                warmup_ms,
            } => json!({ "endpoint": endpoint, "warmup_ms": warmup_ms }),
            Transition::Lost { reason } => json!({ "reason": lost_reason_str(*reason) }),
            Transition::Unloaded => json!({}),
        }
    }
}

fn lost_reason_str(reason: LostReason) -> &'static str {
    match reason {
        LostReason::Evicted => "evicted",
        LostReason::GpuDetached => "gpu_detached",
        LostReason::ProcessExited => "process_exited",
        LostReason::Unhealthy => "unhealthy",
    }
}

/// One event read back out of the spine.
#[derive(Debug, Clone)]
pub struct Event {
    pub model: String,
    pub kind: String,
    pub facts: Value,
    pub seq: u64,
    pub hash: String,
    pub ts: f64,
}

impl Event {
    fn of(node: &Node) -> Option<Event> {
        let obj = node.data.as_object()?;
        Some(Event {
            model: node.id.clone(),
            kind: obj.get("kind")?.as_str()?.to_string(),
            facts: obj.get("facts").cloned().unwrap_or(Value::Null),
            seq: node.seq,
            hash: node.hash.clone(),
            ts: node.ts,
        })
    }
}

/// The spine itself: an embedded NEDB database.
pub struct Spine {
    db: Db,
}

impl Spine {
    /// Open (or create) an on-disk spine.
    pub fn open(root: &Path) -> Result<Spine, String> {
        let db = Db::open(root, None).map_err(|e| e.to_string())?;
        Ok(Spine { db })
    }

    /// Pure in-memory spine — tests and ephemeral runs. Nothing survives drop.
    pub fn in_memory() -> Spine {
        Spine {
            db: Db::in_memory(),
        }
    }

    /// Record a transition for `model`, causally linked to the events that
    /// produced it. Returns the ref future events should cite as cause.
    pub fn record(
        &self,
        model: &str,
        transition: &Transition,
        caused_by: &[EventRef],
    ) -> Result<EventRef, String> {
        let coll = match transition {
            Transition::PullStarted { .. } | Transition::PullCompleted { .. } => PULLS,
            _ => RESIDENCY,
        };
        let doc = json!({
            "kind": transition.kind(),
            "facts": transition.facts(),
        });
        let causes: Vec<String> = caused_by.iter().map(|r| r.hash.clone()).collect();
        let node = self
            .db
            .put(coll, model, doc, causes, None, None)
            .map_err(|e| e.to_string())?;
        Ok(EventRef::of(&node))
    }

    /// The model's current state — the latest residency event.
    pub fn latest(&self, model: &str) -> Option<Event> {
        self.db.get(RESIDENCY, model).as_ref().and_then(Event::of)
    }

    /// What the model's state was at sequence `seq`. Time travel.
    pub fn state_as_of(&self, model: &str, seq: u64) -> Option<Event> {
        self.db
            .get_as_of(RESIDENCY, model, seq)
            .as_ref()
            .and_then(Event::of)
    }

    /// The causal chain that produced the model's current state, newest
    /// first: the answer to "why is this model (not) resident?"
    pub fn why(&self, model: &str) -> Vec<Event> {
        let Some(latest) = self.latest(model) else {
            return Vec::new();
        };
        self.db
            .trace(&latest.hash, false, 64)
            .iter()
            .filter_map(Event::of)
            .collect()
    }

    /// Reconstruct `Residency` from the latest event, for handing back to
    /// the `hearth_core` state machine after a restart.
    pub fn residency(&self, model: &str, now_ms: u64) -> Option<Residency> {
        let ev = self.latest(model)?;
        let vram = ev
            .facts
            .get("vram_bytes")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        match ev.kind.as_str() {
            "resident" => Some(Residency::Resident {
                since: now_ms,
                vram_bytes: vram,
            }),
            "loading" => Some(Residency::Loading { since: now_ms }),
            "lost" => {
                let reason = match ev.facts.get("reason").and_then(Value::as_str) {
                    Some("gpu_detached") => LostReason::GpuDetached,
                    Some("process_exited") => LostReason::ProcessExited,
                    Some("unhealthy") => LostReason::Unhealthy,
                    _ => LostReason::Evicted,
                };
                Some(Residency::Lost { at: now_ms, reason })
            }
            "unloaded" => Some(Residency::Stopped { at: now_ms }),
            _ => None,
        }
    }

    /// Current global sequence — the clock `state_as_of` reads against.
    pub fn seq(&self) -> u64 {
        self.db.seq.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Merkle head of the whole history.
    pub fn head(&self) -> String {
        self.db.head()
    }

    /// Verify every node against its hash. `(checked, failures)`.
    pub fn verify(&self) -> (usize, Vec<String>) {
        self.db.verify()
    }

    /// Flush everything to disk (no-op in memory).
    pub fn flush(&self) {
        self.db.flush_all();
        self.db.flush_manifest_if_dirty();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spine() -> Spine {
        Spine::in_memory()
    }

    #[test]
    fn declared_is_recorded_and_latest() {
        let s = spine();
        s.record(
            "muse",
            &Transition::Declared {
                vram_bytes: 4_000_000_000,
                admitted: true,
            },
            &[],
        )
        .unwrap();
        let ev = s.latest("muse").unwrap();
        assert_eq!(ev.kind, "declared");
        assert_eq!(ev.facts["admitted"], json!(true));
    }

    #[test]
    fn refusal_is_a_fact_too() {
        let s = spine();
        s.record(
            "too-big",
            &Transition::Declared {
                vram_bytes: u64::MAX,
                admitted: false,
            },
            &[],
        )
        .unwrap();
        assert_eq!(s.latest("too-big").unwrap().facts["admitted"], json!(false));
    }

    #[test]
    fn full_lifecycle_chains_causally() {
        let s = spine();
        let declared = s
            .record(
                "muse",
                &Transition::Declared {
                    vram_bytes: 4_000_000_000,
                    admitted: true,
                },
                &[],
            )
            .unwrap();
        let pull = s
            .record(
                "muse",
                &Transition::PullStarted {
                    source: "ollama:library/muse:latest".into(),
                },
                &[declared.clone()],
            )
            .unwrap();
        let pulled = s
            .record(
                "muse",
                &Transition::PullCompleted {
                    path: "/blobs/sha256-abc".into(),
                    size_bytes: 4_000_000_000,
                },
                &[pull],
            )
            .unwrap();
        let loading = s
            .record(
                "muse",
                &Transition::Loading {
                    pid: 4242,
                    endpoint: "127.0.0.1:8080".into(),
                },
                &[pulled],
            )
            .unwrap();
        let resident = s
            .record(
                "muse",
                &Transition::Resident {
                    endpoint: "127.0.0.1:8080".into(),
                    warmup_ms: 43_000,
                },
                &[loading],
            )
            .unwrap();

        // why() walks the causal chain back through the whole story.
        let why = s.why("muse");
        let kinds: Vec<&str> = why.iter().map(|e| e.kind.as_str()).collect();
        assert_eq!(
            kinds,
            vec![
                "resident",
                "loading",
                "pull_completed",
                "pull_started",
                "declared"
            ]
        );
        assert_eq!(why[0].hash, resident.hash);
    }

    #[test]
    fn as_of_separates_then_from_now() {
        let s = spine();
        let loading = s
            .record(
                "muse",
                &Transition::Loading {
                    pid: 1,
                    endpoint: "127.0.0.1:8080".into(),
                },
                &[],
            )
            .unwrap();
        let resident = s
            .record(
                "muse",
                &Transition::Resident {
                    endpoint: "127.0.0.1:8080".into(),
                    warmup_ms: 100,
                },
                &[loading],
            )
            .unwrap();
        s.record(
            "muse",
            &Transition::Lost {
                reason: LostReason::GpuDetached,
            },
            &[resident.clone()],
        )
        .unwrap();

        // Now: lost. Then (at the resident event's seq): resident.
        assert_eq!(s.latest("muse").unwrap().kind, "lost");
        let then = s.state_as_of("muse", resident.seq).unwrap();
        assert_eq!(then.kind, "resident");
    }

    #[test]
    fn residency_reconstructs_for_the_state_machine() {
        let s = spine();
        let r = s
            .record(
                "muse",
                &Transition::Resident {
                    endpoint: "e".into(),
                    warmup_ms: 5,
                },
                &[],
            )
            .unwrap();
        assert!(matches!(
            s.residency("muse", 0),
            Some(Residency::Resident { .. })
        ));
        s.record(
            "muse",
            &Transition::Lost {
                reason: LostReason::Evicted,
            },
            &[r],
        )
        .unwrap();
        assert!(matches!(
            s.residency("muse", 0),
            Some(Residency::Lost {
                reason: LostReason::Evicted,
                ..
            })
        ));
    }

    #[test]
    fn lost_reason_survives_the_round_trip() {
        let s = spine();
        s.record(
            "muse",
            &Transition::Lost {
                reason: LostReason::GpuDetached,
            },
            &[],
        )
        .unwrap();
        assert!(matches!(
            s.residency("muse", 0),
            Some(Residency::Lost {
                reason: LostReason::GpuDetached,
                ..
            })
        ));
    }

    #[test]
    fn verify_is_clean_over_a_real_history() {
        let s = spine();
        for i in 0..5 {
            s.record(
                "muse",
                &Transition::Resident {
                    endpoint: format!("e{i}"),
                    warmup_ms: i,
                },
                &[],
            )
            .unwrap();
        }
        let (checked, failures) = s.verify();
        assert!(checked >= 5);
        assert!(failures.is_empty(), "verify failures: {failures:?}");
    }

    #[test]
    fn on_disk_history_survives_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let resident_seq;
        {
            let s = Spine::open(dir.path()).unwrap();
            let l = s
                .record(
                    "muse",
                    &Transition::Loading {
                        pid: 7,
                        endpoint: "127.0.0.1:9".into(),
                    },
                    &[],
                )
                .unwrap();
            let r = s
                .record(
                    "muse",
                    &Transition::Resident {
                        endpoint: "127.0.0.1:9".into(),
                        warmup_ms: 12,
                    },
                    &[l],
                )
                .unwrap();
            resident_seq = r.seq;
            s.record(
                "muse",
                &Transition::Lost {
                    reason: LostReason::Evicted,
                },
                &[r],
            )
            .unwrap();
            s.flush();
        }
        // Fresh process, same disk: history intact, AS OF still answers.
        let s = Spine::open(dir.path()).unwrap();
        assert_eq!(s.latest("muse").unwrap().kind, "lost");
        assert_eq!(
            s.state_as_of("muse", resident_seq).unwrap().kind,
            "resident"
        );
        let (checked, failures) = s.verify();
        assert!(checked >= 3);
        assert!(failures.is_empty());
    }
}
