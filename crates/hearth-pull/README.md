# hearth-pull

**Get the bytes, prove they are the right bytes, and write down where they came from.**

[![crates.io](https://img.shields.io/crates/v/hearth-pull.svg)](https://crates.io/crates/hearth-pull)
[![docs.rs](https://img.shields.io/docsrs/hearth-pull)](https://docs.rs/hearth-pull)

The half of the aggregator that fetches. [`hearth-resolve`](https://crates.io/crates/hearth-resolve) is deliberately network-free — it turns a reference into blob URLs, a chosen quantization and a byte count with `serde` and `serde_json` as its entire dependency list, which is what lets the VRAM planner refuse a model *before* forty gigabytes move. This crate moves them.

```bash
cargo install hearth-serve     # gets you the `hearth` binary
hearth pull tinyllama
```

```
hearth: pulling tinyllama …
######################################################################## 100.0%
tinyllama:latest pulled and verified — 0.59 GiB
  from  ollama:library/tinyllama:latest
  at    ~/.hearth/blobs/sha256-2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816

  hearth serve --model tinyllama:latest --gguf ~/.hearth/blobs/sha256-2af3b…
```

That is real output. 637 MB from `registry.ollama.ai` in 33 seconds, digest-verified.

## Three properties, and the third is the one nobody else offers

### 1. Digest-verified, and a failure deletes the file

Every blob is hashed and compared against the digest the registry published. A download that does not match is **deleted**, because keeping it is how a corrupt download becomes a permanent, cached, corrupt download — and a corrupt GGUF fails much later as an unexplainable tokenizer error, three layers from the truth.

Downloads land on a `.partial` sibling and are renamed only after the digest matches. **The rename is the commit.** Without that, an interrupted transfer leaves a file whose *name* claims a digest its *contents* do not have, and every later run trusts the name.

Proven against real corruption rather than asserted:

```
$ # flip 64 bytes at offset 300,000,000 — file SIZE unchanged
$ hearth pull tinyllama --verify-existing
hearth: …/sha256-2af3b818… does not match its own name — refetching
tinyllama:latest pulled and verified — 0.59 GiB
```

A size check cannot see that. Only a digest can.

### 2. Resumable, and nothing fails on a clock

A 40 GiB transfer that dies at 39 GiB resumes. Restarting is not a recovery strategy, it is a way to never finish on a bad link.

There is deliberately **no total timeout**. Retries handle a flaky connection; a deadline on the transfer would kill a legitimately slow model and report it as a failure. Same rule the supervisor follows — a slow model is slow, not broken.

### 3. Recorded, so provenance is a query

```
$ hearth why tinyllama:latest
where tinyllama:latest came from:

● seq     3  pull_completed {"path":"…/sha256-2af3b818…","size_bytes":637699456}
└─ seq     2  pull_started   {"source":"ollama:library/tinyllama:latest"}
(never served — the bytes are here, nothing has loaded them)
```

`PullStarted` is written **before any bytes move**, so a pull that dies halfway leaves evidence that it was attempted and from where — the alternative is a half-empty blob directory and no idea which model it belongs to. Months later, "where did this file come from" is a query against a verifiable causal chain instead of an archaeology project.

A pull is deliberately **not** a residency state. Having the bytes is not having them loaded, and letting a pull answer "is this model warm" would make a downloaded-but-never-served model report as resident.

## Why it drives `curl` instead of linking an HTTP client

The same call the rest of this workspace makes about dependencies, applied to a harder case. `probe.rs` writes a raw HTTP GET over a `TcpStream`, because a localhost health check is a status line and nothing else. A registry download is not that — it needs TLS, redirects to a CDN, resume-after-interrupt, and retry with backoff. Writing those is not declining a dependency; it is becoming one, badly.

`curl` is already on every machine that can run a model, is maintained by people who do this full time, and gets all four right. hearth already shells out to `nvidia-smi` and `llama-server`; this is the same bargain with a better-tested binary. What lives in this crate is the argv and the error reading — the parts with a rule in them — pure and tested without touching the network.

Failures are explained by cause, because "it failed" is the least useful thing a downloader can say:

| curl exit | what you are told |
|---|---|
| 6 | could not resolve the host — DNS, or no network from this box |
| 7 | could not connect — the host is reachable but refused us |
| 22 | the server returned an error status (with what it said) |
| 23 | could not write the file — check the disk and permissions |
| 33 | the server refused a resumed range; retry without resume |
| 35 / 60 | TLS failed — a proxy or a clock skew will do this |

`--fail` is not optional in that argv. Without it, curl writes a 404 page to disk and exits 0 — a 200-byte "not found" page named `muse.gguf`, which fails much later as a corrupt tokenizer.

## Only the model layer is weights

An Ollama manifest carries templates, params, system prompts and licenses alongside the model. Only `application/vnd.ollama.image.model` is weights; summing every layer would overstate VRAM and refuse models that fit. For real tinyllama that is 637,699,456 bytes and not 637,699,655.

If a manifest has no model layer, this **refuses** rather than guessing. Picking the largest blob and hoping is how you load a license file as weights.

## As a library

```rust
use hearth_pull::{pull, PullConfig};
use hearth_store::Spine;

let cfg = PullConfig {
    blobs_dir: "~/.hearth/blobs".into(),
    progress: true,
    verify_existing: false,   // the filename IS the digest
    ..Default::default()
};

let out = pull("tinyllama", &cfg, &Spine::open(&dir)?)?;
println!("{} at {}", out.model, out.weights_path.display());
```

`verify_existing` is off by default because re-hashing 40 GiB on every start is a real cost and the filename is the digest. Turn it on when you have reason to distrust the disk.

## Not done yet

**HuggingFace pulls are not wired.** `hearth-resolve` can already plan them — quant choice with longest-token-match, multipart shard sets, byte counts — but this crate's fetch path only speaks to Ollama's registry. It says so plainly rather than failing obscurely:

```
HuggingFace pulls are not wired yet — hearth-resolve can already plan
them; what is missing is this fetch path. Use an ollama: reference or
file: for now.
```

## The rest of the workspace

| crate | what it does |
|---|---|
| [`hearth-core`](https://crates.io/crates/hearth-core) | state machine, VRAM planner, router, SHA-256 |
| [`hearth-resolve`](https://crates.io/crates/hearth-resolve) | references to fetchable blobs, no network |
| **hearth-pull** | this crate — the bytes, verified and recorded |
| [`hearth-store`](https://crates.io/crates/hearth-store) | the NEDB spine |
| [`hearth-serve`](https://crates.io/crates/hearth-serve) | the supervisor and the `hearth` CLI |

---

Built by **Vex × Interchained**

© Interchained LLC · BUSL-1.1 (converts to Apache-2.0 on 2030-08-27)
