# examples

Runnable, and **run** — every number printed in the READMEs was pasted out of these, not composed by hand. A README no compiler or interpreter ever reads rots silently and then wastes the first ten minutes of every newcomer's day.

## Rust

```bash
cargo run -p hearth-core    --example quickstart        # the hearth-core README, compiled
cargo run -p hearth-core    --example a6000             # the roster that started this
cargo run -p hearth-store   --example the_night         # bi-temporal history over real NEDB
cargo run -p hearth-resolve --example against_reality   # parsers vs a real registry manifest
```

## Node

```bash
# From a checkout, build the addon once:
cd crates/hearth-node && npm install && npm run build && cd ../..

node examples/node/plan.js     # will this card hold this roster?
node examples/node/fleet.js    # the night, replayed — and who gets blamed
```

Or against the published package: `npm install @interchained/hearth` and change the
`require('../../crates/hearth-node')` at the top of each file to `require('@interchained/hearth')`.

## Python

```bash
# From a checkout:
cd crates/hearth-py && maturin develop && cd ../..

python examples/python/the_night.py
```

Or: `pip install hearth-engine`.

## What each one is actually showing

**`plan.js` / the planning half of `the_night.py`** — five models an operator wanted resident on one rented 48 GiB RTX A6000. No runtime refuses that roster: it loads, evicts, loads, evicts, and presents to everyone as *"the models got slow."* Here it is arithmetic, up front, with the exact shortfall per model.

**`fleet.js` / `the_night.py`** — the night hearth was built for. A model warms up, serves for an hour, and then the host reclaims the card. Watch the verdict at each step, and especially the last one:

```
GPU detached   muse-local:latest   try-elsewhere
               {"reason":"gpu_detached","operator_fault":false,...}

same probe failure, card still PRESENT:
               {"reason":"evicted","operator_fault":true,...}
```

Identical symptom — a health probe that stopped answering. Opposite diagnosis, opposite fault. One boolean, `gpu_present`, is the entire difference, and nothing else in the ecosystem reports it.

**`the_night.rs`** — the same story asked as *queries* instead of log lines: what was resident **as of** that sequence, why did it go cold (a walked causal chain, not a timestamp sort), and is the history real (`verify`).

## One thing to watch

Key names differ **by language on purpose**, and you cannot copy them between the two:

| | returns | accepts |
|---|---|---|
| Node | camelCase — `shortBytes`, `committedBytes` | `weightsBytes` *or* `weights_bytes`; `gpuPresent` |
| Python | snake_case — `short_bytes`, `committed_bytes` | `weights_bytes` *or* `weightsBytes`; `gpu_present` |

Both parsers accept both spellings in both languages — the roster always did, and `observe(...)` does as of 0.3.3. Before that, an unrecognised key in the detail dict fell back to "the GPU was present," which reported `evicted` when you meant `gpu_detached`, silently. The mapping now lives in `hearth-core` and is tested once, so the bindings cannot drift apart again.

What still differs is only what comes back **out**: Node returns camelCase, Python returns snake_case. Write keys the way your language does and read them the same way.

---

Built by **Vex × Interchained**

© Interchained LLC · BUSL-1.1
