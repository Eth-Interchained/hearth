# hearth-api

**One OpenAI-compatible port in front of a fleet — where the HTTP status carries the diagnosis.**

[![crates.io](https://img.shields.io/crates/v/hearth-api.svg)](https://crates.io/crates/hearth-api)
[![docs.rs](https://img.shields.io/docsrs/hearth-api)](https://docs.rs/hearth-api)

```bash
hearth up --model muse=/models/muse.gguf:20 \
          --model deepseek=/models/deepseek.gguf:20 \
          --total-gib 48 --port 11434
```

```
hearth: muse loading on 127.0.0.1:42499 …
hearth: whale not started — refused by the VRAM budget at declaration
hearth up on http://127.0.0.1:11434
  POST /v1/chat/completions   routed by the "model" field
  GET  /v1/models             what is declared, and what is ready
  GET  /residency             the truth the OpenAI shape cannot carry
```

Point any OpenAI client at it. Proxying is the boring part.

## The part that is not boring

Every gateway can forward a request. What none of them do is answer honestly when they **can't**. A model still loading, a model the runtime evicted, a model whose GPU the host reclaimed, and a model that will never fit on this card all arrive at a caller as *the same failed request*.

hearth already knows which one it is. This crate turns that into a status a router can act on:

| situation | status | `Retry-After` | `retryable` | `operator_fault` |
|---|---|---|---|---|
| resident | proxied | — | — | — |
| **warming** | 503 | **5s** | true | false |
| **GPU reclaimed by host** | 503 | 30s | true | **false** |
| **evicted by runtime** | 503 | 30s | true | **true** |
| process exited | 503 | 30s | true | true |
| **will never fit here** | **409** | **none** | **false** | false |
| not declared | 404 | none | false | — |
| no `model` in body | 400 | none | false | — |

### 409, not 503, for a model that does not fit

This is the one that matters most and the easiest to get wrong. `503` means *try again*. A model that is short by 396 GiB on this card is **arithmetically incapable** of ever being served here — a retryable status turns that into a router hammering the box forever, which presents to everyone as a slow node.

```bash
$ curl -s -D- -X POST localhost:11434/v1/chat/completions \
    -d '{"model":"whale","messages":[]}'
HTTP/1.1 409 Conflict
```
```json
{"error": {
  "message": "whale does not fit on this host — short by 396.0 GiB. This is permanent
              until the declaration or the hardware changes, so do not retry here.",
  "type": "model_not_admitted",
  "short_bytes": 425201762304,
  "retryable": false
}}
```

No `Retry-After` header, deliberately. Setting one on a permanent condition invites exactly the retry storm the status code was chosen to prevent.

### The body says whose fault it was

Same 503, opposite diagnosis. A router that reads only the status treats them alike — correct, both mean *go elsewhere*. One that reads the body learns who to score down, and **that is the number that follows an operator around**:

```json
{"type": "model_lost", "reason": "gpudetached", "operator_fault": false,
 "message": "muse lost its GPU — the host reclaimed the card; this is not the operator's doing"}

{"type": "model_lost", "reason": "evicted", "operator_fault": true,
 "message": "muse was evicted by the runtime to free VRAM — this node is over-committed"}
```

Each reason gets its own message rather than one keyed off the fault boolean. Telling an operator "this node is over-committed" because a process segfaulted sends them to inspect VRAM headroom that was never the problem — a wrong diagnosis is worse than a vague one.

### Warming is progress, not failure

```json
{"type": "model_warming", "loading_for_ms": 20000, "retryable": true,
 "operator_fault": false,
 "message": "muse is loading (20s so far) — this is progress, not a failure"}
```

The caller is told *how long*, and decides for itself whether to wait. hearth never makes that call on someone else's behalf — a 32B materializing over a network fabric can legitimately take minutes, and killing it at a deadline turns a slow success into a fast failure.

## `/residency` — what the OpenAI shape cannot express

```bash
$ curl -s localhost:11434/residency
```
```json
{
  "committed_bytes": 2147483648,
  "free_bytes": 4294967296,
  "report": "2.0 / 6.0 GiB held  (2 declared, 1 admitted)\n  muse   resident for 25s\n  whale  not admitted — short by 396.0 GiB",
  "models": [
    {"model": "muse",  "ready": true,  "operator_fault": false, "short_bytes": 0},
    {"model": "whale", "ready": false, "operator_fault": false, "short_bytes": 425201762304}
  ]
}
```

The refused model is **listed, not hidden**. "It used to hold that model" is the first thing anyone says when a box gets slower, and the exact shortfall is the actionable number.

`/v1/models` carries the same truth in OpenAI's shape, under an additive `hearth` key that a plain client ignores.

`/health` is about **the gateway**, not the models. A gateway that reported unhealthy because a model was loading would be pulled from a load balancer for doing its job correctly.

## Also speaks Ollama's paths

`/api/chat`, `/api/generate` and `/api/tags` are recognised, so a tool already pointed at Ollama does not need rewriting to try hearth.

## Design

**One lock, held only to decide.** The gateway locks the fleet to make a routing decision and releases it *before* relaying. Holding it across a generation would stall every other request and the supervisor with it.

**Streaming is relayed in chunks, flushed per chunk.** Buffering would hold every token until the model finished — a live stream becomes a long silence followed by a wall of text, which is the most noticeable way a gateway ruins a chat UI.

**No HTTP framework.** Same call as the rest of this workspace: `probe.rs` writes a raw GET because a health check is a status line; `hearth-pull` drives `curl` because TLS and resume are a real library's job. This is a localhost server whose whole surface is read a request, write a response, relay bytes. A framework would bring an async runtime to do what std already does, and the parsing — the only part with a rule in it — lives here, pure and tested.

**Loopback by default.** A gateway on `0.0.0.0` hands anyone on the network an unauthenticated GPU. Put TLS in front of it if it needs to leave the box.

## The rest of the workspace

| crate | what it does |
|---|---|
| [`hearth-core`](https://crates.io/crates/hearth-core) | the decision procedure — states, budget, routing |
| [`hearth-resolve`](https://crates.io/crates/hearth-resolve) | references to fetchable blobs, no network |
| [`hearth-pull`](https://crates.io/crates/hearth-pull) | the bytes, digest-verified and recorded |
| [`hearth-store`](https://crates.io/crates/hearth-store) | the NEDB spine |
| **hearth-api** | this crate — the gateway |
| [`hearth-serve`](https://crates.io/crates/hearth-serve) | the supervisor and the `hearth` CLI |

---

Built by **Vex × Interchained**

© Interchained LLC · BUSL-1.1 (converts to Apache-2.0 on 2030-08-27)
