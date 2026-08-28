# hearth-resolve

**One reference syntax over Ollama's registry, HuggingFace GGUFs, and your own disk.**

[![crates.io](https://img.shields.io/crates/v/hearth-resolve.svg)](https://crates.io/crates/hearth-resolve)
[![docs.rs](https://img.shields.io/docsrs/hearth-resolve)](https://docs.rs/hearth-resolve)

Their catalogs, our runtime. [hearth](https://github.com/aiassistsecure/hearth) does not want to be a model registry — two good ones already exist, with the models people actually run. This crate turns any reference into *fetchable blobs* without downloading anything, so the VRAM planner can refuse a model before a single byte moves.

```toml
[dependencies]
hearth-resolve = "0.3"
```

## One syntax, three origins

```rust
use hearth_resolve::Reference;

// A bare name is Ollama's curated library — the shorthand everyone already types.
Reference::parse("llama3")?;                    // ollama library/llama3:latest
Reference::parse("deepseek-r1:32b")?;           // ollama library/deepseek-r1:32b
Reference::parse("ollama:user/custom:v2")?;     // a user namespace

// HuggingFace, with the quantization pinned or left to us.
Reference::parse("hf:TheBloke/Llama-2-7B-GGUF")?;
Reference::parse("hf:TheBloke/Llama-2-7B-GGUF@Q4_K_M")?;

// Already on disk. No catalog, no download, no opinions.
Reference::parse("file:/models/muse.gguf")?;
```

```rust
let r = Reference::parse("deepseek-r1:32b")?;
r.key();            // stable identity for the store and the models dir
r.display_name();   // what a human should read
r.needs_download(); // false for Local — nothing to fetch
```

A bare name resolving to Ollama is a deliberate compatibility choice: it is what every existing script and every operator's muscle memory already types, and gratuitously breaking that buys nothing.

## Choosing a quantization, and saying so

Point at a HuggingFace repo without pinning a quant and something has to choose. That choice is explicit and reported, never silent:

```rust
use hearth_resolve::{pick_gguf, quant_of, available_quants, total_parts};

quant_of("llama-2-7b.Q4_K_M.gguf");   // Some("Q4_K_M")
```

Preference order starts at **Q4_K_M** — the quality/size knee for most GGUF models, and the one a human would pick if they were reading the file list themselves.

Two details in here exist because getting them wrong is quiet and expensive:

**Longest-token-match wins.** `Q4_K` is a prefix of `Q4_K_M`, so naive matching silently hands you a different model than the one named in the file. The tokenizer takes the longest match, so `Q4_K_M` is never swallowed by `Q4_K`.

**Multi-part GGUFs are all-or-nothing.** A file named `-00001-of-00003` is one third of a model. `total_parts()` reads the count and `is_multipart()` flags it, so a plan either has every shard or is refused — rather than loading a third of a model and reporting a corrupt tokenizer.

## A plan is bytes, before any bytes move

```rust
use hearth_resolve::{plan_from_ollama_manifest, plan_from_hf_files};

let plan = plan_from_ollama_manifest(&reference, &manifest_json)?;
plan.total_bytes();   // Option<u64> — feed straight into the VRAM budget
plan.is_multipart();
```

`total_bytes()` is the whole point of resolving before downloading: it is what lets `hearth-core` say *"short by 14.8 GiB"* at declare time instead of after a 40 GiB download and an out-of-memory error.

From an Ollama manifest, only `application/vnd.ollama.image.model` is the weights layer — the manifest also carries templates, params and licenses, and summing all of them would overstate VRAM by a comfortable margin and refuse models that fit.

## Verified against the live registries

The parsers were checked against real manifests and real repo listings, not fixtures invented to match the code:

```bash
cargo run -p hearth-resolve --example against_reality
cargo test -p hearth-resolve
```

Fixtures written by the same person who wrote the parser agree with the parser by construction. Live registries do not.

## The rest of the workspace

| crate | what it does |
|---|---|
| [`hearth-core`](https://crates.io/crates/hearth-core) | state machine, VRAM planner, fleet router. Pure logic. |
| **hearth-resolve** | this crate — references to fetchable blobs |
| [`hearth-store`](https://crates.io/crates/hearth-store) | the NEDB spine — bi-temporal, causal, tamper-evident |
| [`hearth-serve`](https://crates.io/crates/hearth-serve) | the supervisor and the `hearth` CLI |

---

Built by **Vex × Interchained**

© Interchained LLC · BUSL-1.1 (converts to Apache-2.0 on 2030-08-27)
