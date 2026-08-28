# hearth-store

**The spine. Every residency transition is a bi-temporal, causally-linked, tamper-evident event.**

[![crates.io](https://img.shields.io/crates/v/hearth-store.svg)](https://crates.io/crates/hearth-store)
[![docs.rs](https://img.shields.io/docsrs/hearth-store)](https://docs.rs/hearth-store)

Not a log bolted onto the side of a supervisor. **The supervisor's memory IS the database.** `hearth-store` puts an embedded [NEDB](https://github.com/Eth-Interchained/nedb) under [hearth](https://github.com/aiassistsecure/hearth) so the three questions you actually have at 3am are *lookups* instead of theories.

```toml
[dependencies]
hearth-store = "0.3"
```

## Why a database and not a log file

A rotating log can tell you a line was printed. It cannot answer:

1. **What was resident *at the moment* the request failed?**
2. **Why did it go cold** — and is that my fault or my provider's?
3. **Is this history real,** or did something rewrite it?

Question 3 is what makes the other two worth anything. Operator reputation built on an unverifiable log is a rumour. If a scoring system can be argued with, it will be — so the record has to be provable, not merely present.

## The API is those three questions

```rust
use hearth_store::{Spine, Transition};
use hearth_core::LostReason;

let spine = Spine::open(&dir)?;            // or Spine::in_memory()

// Recorded BEFORE anything runs — a refusal is a fact too.
let declared = spine.record("stories", &Transition::Declared {
    vram_bytes: 1 << 30, admitted: true,
}, &[])?;

// Each event names the one that caused it.
let loading = spine.record("stories", &Transition::Loading {
    pid: 10_121, endpoint: "127.0.0.1:18080".into(),
}, &[declared])?;

spine.record("stories", &Transition::Resident {
    endpoint: "127.0.0.1:18080".into(), warmup_ms: 256,
}, &[loading])?;
```

| method | question it answers |
|---|---|
| `latest(model)` | what is true now |
| `state_as_of(model, seq)` | what was true **then** — the past does not move |
| `why(model)` | the causal chain, newest first |
| `residency(model, now_ms)` | hand the state machine its state back after a restart |
| `all_latest()` | every model the spine has ever seen |
| `verify()` | `(nodes_checked, problems)` — is this history real |
| `seq()` / `head()` | where the chain is right now |

### 1. What was true then is still answerable now

```
$ hearth as-of stories 2
as of seq 2: stories was `resident` {"endpoint":"127.0.0.1:18080","warmup_ms":256}
```

Bi-temporal means the past does not move when the present changes. The model going cold at 04:00 does not overwrite the fact that it was resident at 03:00 — both are true, each at its own sequence. Ask about a sequence before anything happened and you get *nothing* rather than a guess.

### 2. Why it went cold, as a walked chain

```
$ hearth why stories
● seq     3  unloaded       {}
└─ seq     2  resident       {"endpoint":"127.0.0.1:18080","warmup_ms":256}
└─ seq     1  loading        {"endpoint":"127.0.0.1:18080","pid":10121}
└─ seq     0  declared       {"admitted":true,"vram_bytes":1073741824}
```

`why()` follows `caused_by` links rather than sorting by timestamp, so two models failing in the same second never get their stories interleaved.

Note what `seq 0` buys you: **the declaration is recorded before any process starts,** refusals included. "It used to hold three models" is the first thing anyone says when a box gets slower, and the budget's refusals are exactly the history that answers it.

### 3. Is the history real

```
$ hearth verify
verify ok — 4 nodes checked, history intact
```

Content-addressed events over NEDB's Merkle roots. Nobody edited yesterday to make today look better.

## Fault attribution

The stored fact is the **reason** — `evicted` · `gpu_detached` · `process_exited` · `unhealthy`. Whether it counts against the operator is then a pure function of that reason, `LostReason::is_operator_fault()`, which returns `false` for `GpuDetached` and always will.

That split is deliberate. Storing a *derived* boolean alongside the reason invites the two to disagree after a refactor, and the version that gets believed is whichever one the reader happens to read. One stored fact, one total function over it, no drift.

What the store does guarantee is that a damaged record cannot buy an alibi. When a stored loss reason is unreadable, it decodes to `Evicted` — **the one that counts against us:**

```rust
// hearth-store/src/lib.rs — residency()
let reason = match ev.facts.get("reason").and_then(Value::as_str) {
    Some("gpu_detached")  => LostReason::GpuDetached,
    Some("process_exited") => LostReason::ProcessExited,
    Some("unhealthy")     => LostReason::Unhealthy,
    _                     => LostReason::Evicted,   // never the exonerating one
};
```

Absence has to be positively observed. Anywhere a fact is unreadable, the unknown answer is the one that costs us something — the same rule the GPU probe follows.

## What gets recorded

`Transition` covers the whole life of a model, pulls included:

```
Declared { vram_bytes, admitted }      a refusal is a fact too
PullStarted { source }                 ollama: · hf: · file:
PullCompleted { path, size_bytes }     bytes landed and digest-verified
Loading { pid, endpoint }
Resident { endpoint, warmup_ms }       measured, never guessed
Lost { reason }
Unloaded                               an instruction, not a failure
```

Durability is at **event granularity** — every transition flushes at write time. That is not the obvious default and it was not free: the first end-to-end run came back `no models in the spine yet` after serving real tokens, because `record()` never flushed and a kill took the entire history with it. A spine made of RAM is not a spine. Transitions are rare, so this is cheap; a resident model answering a probe every ten seconds buys no flush, but a state *change* always does.

## Proven, not asserted

The supervisor served real tokens under a real `llama-server`, took a `SIGTERM`, and then a **fresh process** — no shared memory, nothing warm — read the whole story back off disk and verified it. A kill is an event in the history, not the end of it.

## Try it

```bash
cargo run -p hearth-store --example the_night   # replays the night this came from
cargo test -p hearth-store
```

## The rest of the workspace

| crate | what it does |
|---|---|
| [`hearth-core`](https://crates.io/crates/hearth-core) | state machine, VRAM planner, fleet router. Pure logic. |
| [`hearth-resolve`](https://crates.io/crates/hearth-resolve) | one reference syntax over Ollama and HuggingFace |
| **hearth-store** | this crate — the NEDB spine |
| [`hearth-serve`](https://crates.io/crates/hearth-serve) | the supervisor and the `hearth` CLI |

---

Built by **Vex × Interchained**

© Interchained LLC · BUSL-1.1 (converts to Apache-2.0 on 2030-08-27)
