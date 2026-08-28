# hearth-serve

**The supervisor, and the `hearth` CLI.**

[![crates.io](https://img.shields.io/crates/v/hearth-serve.svg)](https://crates.io/crates/hearth-serve)
[![docs.rs](https://img.shields.io/docsrs/hearth-serve)](https://docs.rs/hearth-serve)

`llama-server` children, health probes that produce facts instead of conclusions, and a [NEDB](https://github.com/Eth-Interchained/nedb) event spine from commit one. This is where [hearth](https://github.com/aiassistsecure/hearth) stops being a state machine and starts holding real processes that hold real gigabytes.

```bash
cargo install hearth-serve      # installs the `hearth` binary
```

## Three layers, three jobs

| layer | job |
|---|---|
| [`hearth-core`](https://crates.io/crates/hearth-core) | the **decision procedure** — what state is each model in, what next, is there budget |
| [`hearth-store`](https://crates.io/crates/hearth-store) | the **system of record** — every transition, causal and verifiable, written as it happens |
| `server` / `probe` | the **mechanism** — children, and the probes that produce facts about them |

Keeping them apart is the design. A probe answers exactly one question — *did this endpoint answer HTTP within the deadline?* — and reports what it saw. Turning that into a residency state is the core's job. Conflating the two is how every serving stack ends up reporting a guess as a fact.

## Process-per-model is the whole architecture

It is why hearth can promise residency at all. A process holding a model **cannot be talked out of it by a cache policy**, so "stays warm" becomes "stays alive" — a problem operating systems solved decades ago.

The cost is honest and worth naming: one CUDA context per process, a few hundred megabytes each. That is exactly what the budget reserve is for.

## Use it

```bash
hearth serve --model muse --gguf ./muse.gguf --port 8080
hearth status
hearth why muse
hearth as-of muse 2
hearth verify
```

`hearth serve` flags: `--model` and `--gguf` are required; `--port`, `--ctx`, `--vram-gib`, `--total-gib`, `--binary` and `--once` are optional. `--once` starts, waits for ready, and exits — useful in CI, where supervising forever is the wrong shape.

Point it at a llama.cpp build with `--binary` or `HEARTH_LLAMA_SERVER`; state lives in `$HEARTH_HOME` (default `~/.hearth`).

## As a library

```rust
use hearth_serve::{Supervisor, server::ServerSpec};
use hearth_store::Spine;
use hearth_core::{Budget, Declared, GIB};

let mut sup = Supervisor::new(
    Spine::open(&dir)?,
    Budget::with_reserve_pct(48 * GIB, 8),
    vec![Declared { model: "muse".into(), weights_bytes: 20 * GIB, kv_bytes: GIB }],
);

sup.start(ServerSpec::new("muse", "./muse.gguf", 8080))?;
let endpoint = sup.wait_ready("muse", Duration::from_secs(600))?;

loop {
    let n = sup.tick();          // returns transitions recorded, not noise
    if n > 0 { eprint!("{}", sup.report()); }
}
```

`Supervisor::new` records every declaration — **including the ones the budget refused** — before any process starts. `start()` refuses outright to launch a model the budget rejected, rather than starting it and discovering VRAM later.

`tick()` returns *the number of transitions recorded*, so a caller logs honestly ("2 transitions") instead of narrating every uneventful pass. A supervisor that prints every ten seconds trains people to ignore it.

## The decisions that matter

**Nothing fails on a clock.** A 32B materializing over a network fabric can legitimately take minutes. A supervisor that kills it at a deadline converts a slow success into a fast failure and then reports a crash loop it caused itself. Load duration is reported, never enforced.

**A 503 while loading is not a failure.** `llama-server` answers 503 with `{"status":"loading model"}` during load — `ProbeResult::Warming`, which is *progress*. Reading that as broken restarts a model thirty seconds from ready, forever.

**Nothing is evicted to make room.** If a model does not fit, the honest answer is that it does not fit. The planner already said so at declare time.

**A missing GGUF fails before the spawn.** `llama-server`'s own error for a bad path arrives on stderr *after* a successful spawn, which a supervisor reads as "started, then died" — a crash loop instead of "that path does not exist." A `stat` call knew for free.

**A kill is an event, not the end of the history.** `SIGINT`/`SIGTERM` latch an atomic bool; all the real work — stop children, record `unloaded`, flush — happens on the supervise loop, because a signal handler that touches a database is a signal handler that corrupts one.

## The one boolean

When a probe fails, one question decides everything:

```rust
let gpu = probe::gpu_present().unwrap_or(true);
Observation::ProbeFailed { gpu_present: gpu, detail }
```

Present and cold means the runtime dropped it — `Evicted`, a capacity problem the operator owns. Absent means the host reclaimed the card — `GpuDetached`, the provider's, and `is_operator_fault()` returns `false` for it permanently.

`gpu_present()` returns `Option<bool>`, and the `None` matters: on a CPU-only box there is no way to know, and the caller's `unwrap_or(true)` makes *unknown count against us* rather than handing out an alibi. Absence has to be positively observed.

## Proven end to end

On a real GGUF under a real `llama-server`: resident with warmup **measured** (256ms, not guessed), real completion tokens served, `SIGTERM` taken, child reaped — and then a **fresh process** read the whole story back off disk:

```
$ hearth why stories
● seq     3  unloaded       {}
└─ seq     2  resident       {"endpoint":"127.0.0.1:18080","warmup_ms":256}
└─ seq     1  loading        {"endpoint":"127.0.0.1:18080","pid":10121}
└─ seq     0  declared       {"admitted":true,"vram_bytes":1073741824}

$ hearth verify
verify ok — 4 nodes checked, history intact
```

## Integration

hearth speaks OpenAI-compatible, so [pin-clientd](https://github.com/aiassistsecure/pin-clientd) works with it today: set `apiMode: "openai"` and point `inferenceUri` at hearth.

## Not done yet

`hearth pull` wired to the resolvers · the HTTP surface (OpenAI-compatible proxy + `/residency`) · multi-model fleets under one supervisor. `Transition::PullStarted` / `PullCompleted` already exist in the spine, so the history is ready before the downloader is.

---

Built by **Vex × Interchained**

© Interchained LLC · BUSL-1.1 (converts to Apache-2.0 on 2030-08-27)
