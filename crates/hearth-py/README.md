# hearth-engine

**Deterministic model residency, in Python.** Keep declared models warm, and tell the truth about which ones are.

[![PyPI](https://img.shields.io/pypi/v/hearth-engine.svg)](https://pypi.org/project/hearth-engine/)

Native bindings to [hearth](https://github.com/aiassistsecure/hearth)'s Rust core, built with [PyO3](https://pyo3.rs) and [maturin](https://maturin.rs). No GPU required to use this package — it is the decision procedure, not the runtime.

```bash
pip install hearth-engine
```

```python
import hearth
```

One [abi3](https://docs.python.org/3/c-api/stable.html) wheel per platform covers Python 3.9+, so a new Python release does not need a new wheel.

## Why you would want this

Your inference call hangs, then fails. Three completely different things cause that, and they arrive looking identical:

- the runtime **evicted** the model to free VRAM,
- the host **detached the GPU** and gave it to another tenant,
- a 32B model was simply **still loading**.

One is a capacity problem you own. One is your provider's, and no configuration you write will touch it. One is not a problem at all. A timeout cannot tell you which — so all three get "fixed" repeatedly and none of them go away.

## Will this card hold this roster?

Answered before anything loads.

```python
from hearth import GIB, declare, plan

p = plan(48 * GIB, [
    declare("muse-local:latest", 20 * GIB, GIB),
    declare("deepseek-r1:32b",   20 * GIB, GIB),
    declare("gemma4:26b",        16 * GIB, GIB),
])

print(p["explain"])
# 2 of 3 admitted, 42.0 GiB committed of 44.2 GiB usable
#   REJECTED gemma4:26b — needs 17.0 GiB, 2.2 GiB free, short by 14.8 GiB

for r in p["rejected"]:
    print(f"{r['model']}: short by {r['short_bytes'] / GIB:.1f} GiB")
# gemma4:26b: short by 14.8 GiB
```

Declare five 20 GiB models on a 48 GiB card and no runtime will error. It loads, evicts, loads, evicts, forever, and presents to everyone as "the models got slow." This refuses the model that does not fit *and tells you the shortfall*.

Two rules in the planner are deliberate:

- **Declaration order is priority order.** First fit, never best fit — reordering to squeeze in one more model would silently demote whatever you listed first, and on a serving box first means most important.
- **The reserve is never planned into** (default 8%). Weights are not the whole cost: KV cache grows with context and parallelism, each CUDA context is hundreds of megabytes, and fragmentation is real on a card that has been up for weeks.

## A live fleet, and an answer you can act on

```python
from hearth import GIB, Fleet, declare

fleet = Fleet(48 * GIB, [declare("muse-local:latest", 20 * GIB, GIB)])

fleet.set_endpoint("muse-local:latest", "127.0.0.1:8090")
fleet.observe("muse-local:latest", "load_started")

r = fleet.route("muse-local:latest")
if r["ready"]:
    send_to(r["endpoint"])
elif r["try_elsewhere"]:
    retry_elsewhere(score_down=r["operator_fault"])
```

`route()` gives you three booleans answering three different questions:

| field | question |
|---|---|
| `ready` | can I send this request here, right now |
| `try_elsewhere` | should I go find another node |
| `operator_fault` | **is this the operator's fault** — `False` for a detached GPU |

That last one is the one nothing else reports. A reputation system fed the wrong answer slowly deletes its own honest operators.

The `route` key names which case you are in:

```python
{"route": "unknown",      "ready": False, "try_elsewhere": True,  "operator_fault": False}
{"route": "warming",      "for_ms": 20000, "ready": False, "try_elsewhere": False}
{"route": "ready",        "endpoint": "127.0.0.1:8090", "ready": True}
{"route": "lost",         "reason": "gpu_detached", "operator_fault": False, "try_elsewhere": True}
{"route": "lost",         "reason": "evicted",      "operator_fault": True,  "try_elsewhere": True}
{"route": "not_admitted", "short_bytes": 19155554136, "try_elsewhere": True}
{"route": "not_declared", "try_elsewhere": True}
```

`warming` is the one every stack gets wrong: **wait or route around, but do not fault this node.** Routing to a model that is still coming up, then calling the inevitable timeout an error, is the most common way a serving stack lies about itself.

> **Key naming:** every key this package returns is `snake_case`, and every key it *accepts* is too. The Node package returns camelCase for the same data — each is idiomatic for its own language, so do not copy key names between the two.

## Recording what you observed

```python
fleet.observe(model, kind, detail=None, now=None)
```

`kind` is one of `load_started` · `probe_ok` · `probe_failed` · `process_exited` · `load_failed` · `stop`. `now` defaults to `now_ms()`.

**The single most important field you will ever pass here** is `gpu_present` on a `probe_failed`:

```python
fleet.observe("muse-local:latest", "probe_failed", {
    "gpu_present": False,       # the card is GONE — not this operator's fault
    "detail": "no CUDA device",
})
```

Omit it and it reads as `True` — "the card was still there" — so a missing field can never quietly exonerate an operator. Absence has to be positively observed.

> **Spell it `gpu_present`, in snake_case.** As of 0.3.1 this parser accepts only that spelling, and an unrecognised key falls back to "the GPU was present" — so a dict written as `{"gpuPresent": False}` reports `evicted` with `operator_fault: True`, the exact opposite of what you meant, with no error. `probe_ok` takes `vram_bytes` the same way. Accepting both spellings here (as the roster parser already does) is a tracked fix.

You pass **facts**, never conclusions. What state those produce is the core's job, decided by one state machine tested once in Rust rather than three times in three languages.

## Everything, as one block

```python
print(fleet.report())
# 0.0 / 44.2 GiB held  (1 declared, 1 admitted)
#   muse-local:latest            loading for 5s
```

Same text `hearth status` prints — one truth in two places is how they stop matching.

## Run the example

```bash
python examples/python/the_night.py
```

Replays the night hearth was built for: a model warms up, serves for an hour, the host takes the card away — and the router is told, in words, that it was not the operator's fault.

## Also available

| package | registry |
|---|---|
| [`@interchained/hearth`](https://www.npmjs.com/package/@interchained/hearth) | npm |
| [`hearth-core`](https://crates.io/crates/hearth-core) | crates.io — the pure logic |
| [`hearth-serve`](https://crates.io/crates/hearth-serve) | crates.io — the supervisor and `hearth` CLI |

---

Built by **Vex × Interchained**

© Interchained LLC · BUSL-1.1 (converts to Apache-2.0 on 2030-08-27)
