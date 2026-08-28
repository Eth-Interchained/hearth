# @interchained/hearth

**Deterministic model residency, in Node.** Keep declared models warm, and tell the truth about which ones are.

[![npm](https://img.shields.io/npm/v/@interchained/hearth.svg)](https://www.npmjs.com/package/@interchained/hearth)

Native bindings to [hearth](https://github.com/aiassistsecure/hearth)'s Rust core, built with [napi-rs](https://napi.rs). No GPU required to use this package — it is the decision procedure, not the runtime.

```bash
npm install @interchained/hearth
```

Prebuilt for linux-x64-gnu · darwin-x64 · darwin-arm64 · win32-x64-msvc. Node >= 18.

## Why you would want this

Your router asks a node for a model and the request hangs, then fails. Three completely different things cause that, and they arrive looking identical:

- the runtime **evicted** the model to free VRAM,
- the host **detached the GPU** and gave it to another tenant,
- a 32B model was simply **still loading**.

One is a capacity problem the operator owns. One is their provider's, and no configuration will touch it. One is not a problem at all. A timeout cannot tell you which — so a router that treats them the same either retries into a wall or penalizes an operator for something they could not control.

## Will this card hold this roster?

Answered before anything loads.

```js
const { plan } = require('@interchained/hearth');

const GIB = 1024 ** 3;

const p = plan(48 * GIB, 8, [
  { model: 'muse-local:latest', weightsBytes: 20 * GIB, kvBytes: GIB },
  { model: 'deepseek-r1:32b',   weightsBytes: 20 * GIB, kvBytes: GIB },
  { model: 'gemma4:26b',        weightsBytes: 16 * GIB, kvBytes: GIB },
]);

console.log(p.explain);
// 2 of 3 admitted, 42.0 GiB committed of 44.2 GiB usable

for (const r of p.rejected) {
  console.log(`${r.model}: short by ${(r.shortBytes / GIB).toFixed(1)} GiB`);
  // gemma4:26b: short by 14.8 GiB
}
```

Declare five 20 GiB models on a 48 GiB card and no runtime will error. It loads, evicts, loads, evicts, forever, and presents to everyone as "the models got slow." This refuses the model that does not fit *and tells you the shortfall*.

Two rules in the planner are deliberate:

- **Declaration order is priority order.** First fit, never best fit — reordering to squeeze in one more model would silently demote whatever you listed first, and on a serving box first means most important.
- **The reserve is never planned into.** Weights are not the whole cost: KV cache grows with context and parallelism, each CUDA context is hundreds of megabytes, and fragmentation is real on a card that has been up for weeks.

`p.declared` echoes back how many models you sent. If you passed three and it says three, the plan is about your roster and not a silently emptied one — see *A bug worth knowing about* below.

## A live fleet, and an answer a router can act on

```js
const { HearthFleet } = require('@interchained/hearth');

const fleet = new HearthFleet(48 * GIB, 8, [
  { model: 'muse-local:latest', weightsBytes: 20 * GIB, kvBytes: GIB },
]);

fleet.setEndpoint('muse-local:latest', '127.0.0.1:8090');
fleet.observe('muse-local:latest', 'load_started', {}, Date.now());

const r = fleet.route('muse-local:latest', Date.now());
if (r.ready)            sendTo(r.endpoint);
else if (r.tryElsewhere) retryElsewhere({ scoreDown: r.operatorFault });
```

`route()` gives you three booleans that answer three different questions:

| field | question |
|---|---|
| `ready` | can I send this request here, right now |
| `tryElsewhere` | should I go find another node |
| `operatorFault` | **is this the operator's fault** — `false` for a detached GPU |

That last one is the one nothing else reports. A reputation system fed the wrong answer slowly deletes its own honest operators.

The `route` field names which case you are in, and each case carries its own facts:

```js
{ route: 'unknown',      ready: false, tryElsewhere: true,  operatorFault: false }
{ route: 'warming',      for_ms: 20000, ready: false, tryElsewhere: false }
{ route: 'ready',        endpoint: '127.0.0.1:8090', ready: true }
{ route: 'lost',         reason: 'gpu_detached', operatorFault: false, tryElsewhere: true }
{ route: 'lost',         reason: 'evicted',      operatorFault: true,  tryElsewhere: true }
{ route: 'not_admitted', short_bytes: 19155554136, tryElsewhere: true }
{ route: 'not_declared', tryElsewhere: true }
```

`warming` is the one every stack gets wrong: **wait or route around, but do not fault this node.** `not_admitted` is permanent until the hardware or the declaration changes — stop asking.

> **Field naming, honestly:** `plan()` returns camelCase throughout (`shortBytes`, `neededBytes`, `freeBytes`). `route()` is currently mixed — `ready`, `tryElsewhere` and `operatorFault` are camelCase, but `for_ms` and `short_bytes` come straight from the Rust enum in snake_case, and a `lost` route carries both `operatorFault` and `operator_fault`. Use the names exactly as printed above. camelCase aliases are a known follow-up; they will be added, not swapped, so nothing here breaks.

### Recording what you observed

```js
fleet.observe(model, kind, detail, nowMs);
```

`kind` is one of `load_started` · `probe_ok` · `probe_failed` · `process_exited` · `load_failed` · `stop`.

**The single most important field that crosses this boundary** is `gpuPresent` on a `probe_failed`:

```js
fleet.observe(model, 'probe_failed', {
  gpuPresent: false,          // the card is GONE — not this operator's fault
  detail: 'no CUDA device',
}, Date.now());
```

Omit it and it reads as `true` — "the card was still there" — so a missing field can never quietly exonerate an operator. Absence has to be positively observed.

You pass **facts**, never conclusions. What state that produces is the core's job, decided by one state machine tested once in Rust rather than three times in three languages.

## Everything, as one block

```js
console.log(fleet.report(Date.now()));
// 0.0 / 44.2 GiB held  (1 declared, 1 admitted)
//   muse-local:latest            loading for 5s
```

Same text `hearth status` prints — one truth in two places is how they stop matching.

## Why the API is JSON-shaped

Deliberately structural rather than idiomatic. Three languages describing the same state machine will drift, and JSON is the one shape all three already agree on. Every rule about residency and VRAM lives in `hearth-core` and is tested there once; a binding that reimplements any of it is a second source of truth waiting to disagree with the first.

Both spellings work — `weightsBytes` or `weights_bytes` — because the same core serves a camelCase language and a snake_case one, and neither should have to translate.

## A bug worth knowing about

In `0.1.0` this binding **silently ate every roster it was given.** JS has no integers, so every byte count arrived as an `f64`, `as_u64()` returned `None`, and a `filter_map` quietly discarded the lot. A roster of three became a roster of zero — and zero models trivially fit, so the answer came back `fits: true` with an empty admitted list.

Fixed in `0.1.1`: parsing moved into the core with its own tests, whole-valued floats are accepted, and an unreadable entry now **throws** instead of vanishing. Loud beats convenient.

It was found by *running* it, not by reading it. That is why the examples in this README are executed, not composed.

## Run the examples

```bash
node examples/node/plan.js
node examples/node/fleet.js
```

## Also available

| package | registry |
|---|---|
| [`hearth-engine`](https://pypi.org/project/hearth-engine/) | PyPI |
| [`hearth-core`](https://crates.io/crates/hearth-core) | crates.io — the pure logic |
| [`hearth-serve`](https://crates.io/crates/hearth-serve) | crates.io — the supervisor and CLI |

---

Built by **Vex × Interchained**

© Interchained LLC · BUSL-1.1 (converts to Apache-2.0 on 2030-08-27)
