# hearth-core

**The decision procedure. No GPU required.**

[![crates.io](https://img.shields.io/crates/v/hearth-core.svg)](https://crates.io/crates/hearth-core)
[![docs.rs](https://img.shields.io/docsrs/hearth-core)](https://docs.rs/hearth-core)

The pure-logic heart of [hearth](https://github.com/aiassistsecure/hearth): a residency state machine, a VRAM planner, and a fleet router. It spawns nothing, probes nothing, and links no CUDA. You hand it facts; it hands you states and decisions. That separation is the point — every rule in here is tested once, in Rust, on a laptop, and the bindings and the supervisor consume the same answers.

```toml
[dependencies]
hearth-core = "0.3"
```

## The problem it exists to solve

Three completely different failures arrive at a caller as one timeout:

- the runtime **evicted** a model to free VRAM,
- the host **detached the GPU** and gave it to another tenant,
- a 32B model was simply **still loading**.

One is a capacity problem you own. One is your provider's, and no configuration you write will touch it. One is not a problem at all. A timeout cannot tell you which, so all three get "fixed" repeatedly and none of them go away.

`hearth-core` refuses to collapse them.

## Residency is a named state with a named reason

```rust
use hearth_core::residency::{Residency, Observation, LostReason};

let s = Residency::Unknown
    .observe(&Observation::LoadStarted, 1_000)
    .observe(&Observation::ProbeOk { vram_bytes: 21 << 30 }, 41_000);

assert!(s.is_ready());
println!("{}", s.explain(60_000));   // "resident for 19s"
```

When it goes wrong, the reason survives:

```rust
let lost = s.observe(
    &Observation::ProbeFailed { gpu_present: false, detail: "no CUDA device".into() },
    3_600_000,
);

match lost {
    Residency::Lost { reason, .. } => {
        assert_eq!(reason, LostReason::GpuDetached);
        // The bit that matters, and it is one boolean:
        assert!(!reason.is_operator_fault());
    }
    _ => unreachable!(),
}
```

`gpu_present` on a failed probe is the entire diagnosis. Present and cold means the runtime dropped it — that is capacity, and it is yours. Absent means the card left — that is the provider's, and `is_operator_fault()` returns `false` for it, permanently. A reputation system fed the other answer slowly deletes its own honest operators.

| state | meaning |
|---|---|
| `Unknown` | never probed. Said out loud rather than guessed. |
| `Loading { since }` | weights materializing. Reports elapsed; enforces no deadline. |
| `Resident { since, vram_bytes }` | loaded **and** answering **and** accounted for. |
| `Lost { at, reason }` | `Evicted` · `GpuDetached` · `ProcessExited` · `Unhealthy` |
| `Failed { at, reason }` | will not load, and what the runtime said about why. |
| `Stopped { at }` | unloaded on purpose. Not a failure. |

## The card's size is arithmetic, checked before anything loads

Declare five 20 GiB models on a 48 GiB card and no runtime will error. It loads, evicts, loads, evicts, forever, and presents to everyone as "the models got slow."

```rust
use hearth_core::budget::{Budget, Declared, plan, GIB};

let budget = Budget::with_reserve_pct(48 * GIB, 8);
let roster = vec![
    Declared { model: "muse-local:latest".into(),  weights_bytes: 20 * GIB, kv_bytes: GIB },
    Declared { model: "deepseek-r1:32b".into(),    weights_bytes: 20 * GIB, kv_bytes: GIB },
    Declared { model: "gemma4:26b".into(),         weights_bytes: 16 * GIB, kv_bytes: GIB },
];

let p = plan(budget, &roster);
println!("{}", p.explain());
for r in &p.rejected {
    println!("{} will never fit here — short by {:.1} GiB",
             r.model, hearth_core::budget::gib(r.short_bytes()));
}
```

Two rules in that planner are deliberate and both are about not surprising an operator at 3am:

**Declaration order is priority order.** First fit, never best fit. Reordering the roster to squeeze in one more model would silently demote whatever you listed first — and on a serving box, first means most important.

**The reserve is never planned into.** Weights are not the whole cost. KV cache grows with context and parallelism, each CUDA context is hundreds of megabytes, and fragmentation is real on a card that has been up for weeks.

## Routers get an answer they can act on

```rust
use hearth_core::fleet::{Fleet, Route};

let mut fleet = Fleet::declare(budget, roster);
fleet.set_endpoint("muse-local:latest", "127.0.0.1:8090");
fleet.observe("muse-local:latest", &Observation::LoadStarted, now);

// Every variant carries `model`, so `..` is not optional in these arms.
match fleet.route("muse-local:latest", now + 5_000) {
    Route::Ready { endpoint, .. }       => { /* send it */ }
    Route::Warming { for_ms, .. }       => { /* wait or try elsewhere — do NOT fault this node */ }
    Route::Lost { operator_fault, .. }  => { /* try elsewhere; score only if operator_fault */ }
    Route::NotAdmitted { short_bytes, .. } => { /* stop asking — it will never fit here */ }
    Route::Unknown { .. }               => { /* we genuinely do not know yet, and we say so */ }
    _                                   => {}
}
```

`Warming` is the one every stack gets wrong. Routing to a model that is still coming up, then calling the inevitable timeout an error, is the most common way a serving stack lies about itself.

## Design rules

**Nothing fails on a clock.** A 32B materializing over a network fabric can legitimately spend minutes before its first token. Killing it at an arbitrary deadline converts a slow success into a fast failure and then reports a crash loop it caused. `Loading` reports how long; deciding what to do about that belongs to whoever knows whether a human is waiting. Progress is reported — patience is a policy, not a constant.

**Nothing is evicted to make room.** If a model does not fit, the honest answer is that it does not fit. The planner already said so at declare time.

**Absence must be positively observed.** Anywhere a fact is unreadable, the unknown answer is the one that counts *against* us — never the one that hands out an alibi.

## Try it

Every snippet above is a compiled example, not prose — [`examples/quickstart.rs`](examples/quickstart.rs) *is* this README. A README no compiler reads rots silently and then wastes the first ten minutes of every newcomer's day.

```bash
cargo run -p hearth-core --example quickstart
```

```
== residency ==
  resident for 19s
  lost: GpuDetached, operator at fault: false
== budget ==
  2 of 3 admitted, 42.0 GiB committed of 44.2 GiB usable
  REJECTED gemma4:26b — needs 17.0 GiB, 2.2 GiB free, short by 14.8 GiB
== routing ==
  warming for 5000ms — wait or try elsewhere, but do NOT fault this node
  0.0 / 44.2 GiB held  (1 declared, 1 admitted)
  muse-local:latest            loading for 5s
```

That is real output, pasted. Two 21 GiB models fit on a 48 GiB card; the third is refused at declare time with the exact shortfall, instead of being discovered at 3am as "the models got slow."

```bash
cargo run -p hearth-core --example a6000   # the roster that started this, on a real card
cargo test -p hearth-core                  # no GPU needed
```

## The rest of the workspace

| crate | what it does |
|---|---|
| **hearth-core** | this crate — state machine, planner, router. Pure logic. |
| [`hearth-resolve`](https://crates.io/crates/hearth-resolve) | one reference syntax over the Ollama registry and HuggingFace GGUFs |
| [`hearth-store`](https://crates.io/crates/hearth-store) | the NEDB spine — bi-temporal, causal, tamper-evident history |
| [`hearth-serve`](https://crates.io/crates/hearth-serve) | the supervisor and the `hearth` CLI |

Also on [npm](https://www.npmjs.com/package/@interchained/hearth) and [PyPI](https://pypi.org/project/hearth-engine/).

---

Built by **Vex × Interchained**

© Interchained LLC · BUSL-1.1 (converts to Apache-2.0 on 2030-08-27)
