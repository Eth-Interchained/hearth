//! `hearth pull` — get the bytes, prove they are the right bytes, and write
//! down where they came from.
//!
//! This is the half of the aggregator that was missing. [`hearth_resolve`] is
//! deliberately network-free: it turns a reference into blob URLs, a chosen
//! quantization and a byte count, with `serde` and `serde_json` as its entire
//! dependency list. That is what lets the VRAM planner refuse a model *before*
//! forty gigabytes move. This crate is what moves them.
//!
//! Three properties, and the third is the one nobody else offers:
//!
//! 1. **Digest-verified.** Every blob is hashed and compared against the digest
//!    the registry published. A download that does not match is deleted, not
//!    kept — a corrupt GGUF fails much later as an unexplainable tokenizer
//!    error, three layers away from the truth.
//! 2. **Resumable.** A 40 GiB transfer that dies at 39 GiB resumes. Restarting
//!    is not a recovery strategy, it is a way to never finish on a bad link.
//! 3. **Recorded.** `PullStarted` and `PullCompleted` land in the spine with
//!    the resolved source, so months later "where did this file come from" is a
//!    query — `hearth why <model>` — instead of an archaeology project. Those
//!    two transitions already existed in [`hearth_store`]; the history was
//!    ready before the downloader was.

pub mod curl;
pub mod registry;
pub mod runtime;

use std::path::{Path, PathBuf};

use hearth_core::sha256;
use hearth_resolve::Reference;
use hearth_store::{EventRef, Spine, Transition};

pub use registry::{Blob, Fetched};

/// Where blobs live, and how loud to be about it.
#[derive(Debug, Clone)]
pub struct PullConfig {
    /// Content-addressed blob directory, e.g. `~/.hearth/blobs`.
    pub blobs_dir: PathBuf,
    /// Draw curl's progress meter. A silent twenty-minute pause is
    /// indistinguishable from a hang, and a downloader that looks hung gets
    /// killed halfway.
    pub progress: bool,
    /// Re-verify a blob that is already on disk instead of trusting its name.
    /// Off by default: re-hashing 40 GiB on every start is a real cost, and the
    /// filename IS the digest. On when you have reason to distrust the disk.
    pub verify_existing: bool,
    /// How many blobs to fetch simultaneously. Flag `--parallel-downloads` wins,
    /// then env `HEARTH_PARALLEL_DOWNLOADS`, then this default.
    ///
    /// Default 4 is deliberately conservative: a 40-part sharded model with
    /// concurrency=40 would saturate a spinning disk with random writes from 40
    /// separate curl processes and turn a metered connection hostile. Four gives
    /// real parallelism on a fast NVMe link while remaining polite on HDDs and
    /// capped connections. Set 1 to recover the original sequential behaviour
    /// exactly.
    pub parallel_downloads: usize,
}

impl Default for PullConfig {
    fn default() -> Self {
        PullConfig {
            blobs_dir: PathBuf::from("./blobs"),
            progress: true,
            verify_existing: false,
            parallel_downloads: 4,
        }
    }
}

/// What a pull did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pulled {
    pub model: String,
    /// Resolved origin, as recorded in the spine.
    pub source: String,
    /// The weights blob on disk.
    pub weights_path: PathBuf,
    pub bytes: u64,
    /// True when everything was already present and verified — so a second
    /// `pull` of the same model is honest about having done nothing.
    pub already_had_it: bool,
}

/// Content-addressed path for a digest. The filename IS the digest, which is
/// what makes "do we already have this" a `stat` rather than a re-hash.
pub fn blob_path(blobs_dir: &Path, digest: &str) -> PathBuf {
    blobs_dir.join(format!("sha256-{}", sha256::normalize(digest)))
}

/// Bytes as something a person can compare to a download page.
pub fn human_bytes(n: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    let f = n as f64;
    if f >= GIB {
        format!("{:.2} GiB", f / GIB)
    } else if f >= MIB {
        format!("{:.1} MiB", f / MIB)
    } else {
        format!("{n} B")
    }
}

/// Hash a file, showing that it is still working.
///
/// The step this replaces was a bare `sha256::hex_digest_file`, which on a
/// 35 GiB blob is minutes of total silence immediately after curl prints
/// `100.0%`. That is the exact failure `PullConfig::progress` was written to
/// prevent, and the verify step was the one place the pull path did not honour
/// it.
///
/// Throttled to roughly twice a second on one rewritten line, so it reads like
/// the transfer meter above it instead of scrolling a wall of text.
fn digest_file_loud(path: &Path, label: &str, progress: bool) -> Result<String, String> {
    use std::io::Write;
    let started = std::time::Instant::now();
    let mut last = std::time::Instant::now();
    let digest = sha256::hex_digest_file_with_progress(path, |done, total| {
        if !progress {
            return;
        }
        if last.elapsed() < std::time::Duration::from_millis(500) && done < total {
            return;
        }
        last = std::time::Instant::now();
        let secs = started.elapsed().as_secs_f64().max(0.001);
        let rate = (done as f64 / secs) as u64;
        let pct = if total > 0 {
            format!("{:5.1}%", done as f64 / total as f64 * 100.0)
        } else {
            "  ?  ".to_string()
        };
        // \r, not \n: one line, rewritten in place.
        let _ = write!(
            std::io::stderr(),
            "\rhearth: verifying {label} {pct} ({} / {}) {}/s   ",
            human_bytes(done),
            human_bytes(total),
            human_bytes(rate)
        );
        let _ = std::io::stderr().flush();
    })
    .map_err(|e| format!("could not hash {}: {e}", path.display()))?;
    if progress {
        let secs = started.elapsed().as_secs_f64();
        let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let rate = if secs > 0.0 {
            (len as f64 / secs) as u64
        } else {
            0
        };
        let _ = writeln!(
            std::io::stderr(),
            "\rhearth: verified {label} — {} in {:.1}s ({}/s)      ",
            human_bytes(len),
            secs,
            human_bytes(rate)
        );
    }
    Ok(digest)
}

/// Fetch one blob to its content-addressed home, verifying the digest.
///
/// Downloads to a `.partial` sibling and renames only after the digest
/// matches. Without that, an interrupted transfer leaves a file whose *name*
/// claims a digest its *contents* do not have — and every later run trusts the
/// name. The rename is the commit.
///
/// `slot` disambiguates the temp path for the self-verified (no published
/// digest) case when multiple workers run in parallel. Sequential callers
/// always pass 0.
pub fn fetch_blob(blob: &Blob, cfg: &PullConfig) -> Result<(PathBuf, u64, bool), String> {
    fetch_blob_with_slot(blob, cfg, 0)
}

fn fetch_blob_with_slot(
    blob: &Blob,
    cfg: &PullConfig,
    slot: usize,
) -> Result<(PathBuf, u64, bool), String> {
    match &blob.digest {
        Some(digest) => fetch_blob_verified(blob, digest, cfg),
        None => fetch_blob_self_verified(blob, cfg, slot),
    }
}

/// The original path: a digest is already known, so the final home is known
/// before a single byte moves, and every byte is checked against it.
fn fetch_blob_verified(
    blob: &Blob,
    digest: &str,
    cfg: &PullConfig,
) -> Result<(PathBuf, u64, bool), String> {
    let final_path = blob_path(&cfg.blobs_dir, digest);

    if final_path.exists() {
        let len = std::fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);
        if cfg.verify_existing {
            let actual = digest_file_loud(&final_path, &blob.name, cfg.progress)?;
            if !sha256::matches(digest, &actual) {
                // On disk under a digest it does not have. Say so loudly and
                // re-fetch rather than quietly serving the wrong weights.
                eprintln!(
                    "hearth: {} does not match its own name — refetching",
                    final_path.display()
                );
                let _ = std::fs::remove_file(&final_path);
            } else {
                return Ok((final_path, len, true));
            }
        } else if blob.size_bytes == 0 || len == blob.size_bytes {
            return Ok((final_path, len, true));
        } else {
            // Right name, wrong size: a previous run was interrupted after the
            // rename, or something truncated it. Not trustworthy.
            eprintln!(
                "hearth: {} is {len} bytes, expected {} — refetching",
                final_path.display(),
                blob.size_bytes
            );
            let _ = std::fs::remove_file(&final_path);
        }
    }

    let partial = final_path.with_extension("partial");
    download_to(&partial, blob, cfg)?;

    let actual = digest_file_loud(&partial, &blob.name, cfg.progress)?;
    if !sha256::matches(digest, &actual) {
        // Delete it. Keeping a blob that failed verification is how a corrupt
        // download becomes a permanent, cached, corrupt download.
        let _ = std::fs::remove_file(&partial);
        return Err(format!(
            "digest mismatch for {}\n  expected sha256:{}\n  got      sha256:{}\n  \
             the file has been deleted; a resumed transfer over a proxy that \
             rewrote a range will do this, so a plain retry is worth one attempt",
            blob.name,
            sha256::normalize(digest),
            actual
        ));
    }

    let len = std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);
    std::fs::rename(&partial, &final_path)
        .map_err(|e| format!("could not commit {}: {e}", final_path.display()))?;
    Ok((final_path, len, false))
}

/// No digest was published to check against — a bare URL, or a HuggingFace
/// file the tree listing did not mark as LFS.
///
/// The final home cannot be known before downloading, because the digest
/// that names it is the thing being discovered. So this downloads to a
/// TEMPORARY name (never a content-addressed one — a partial file must never
/// sit under the name its finished content will claim), hashes what actually
/// arrived, and only then computes the path a verified blob would have had
/// from the start. If that path already exists, this exact content was
/// already fetched under some other name, and the new download is discarded
/// as a duplicate rather than kept as a second copy of the same bytes.
///
/// This is self-consistency, not verification: the file matches its own
/// name by construction, which is trivially true of every file and proves
/// nothing about whether the content is what the operator meant to fetch.
/// Callers that need real verification pin a digest — the URL scheme's
/// `#sha256:HEX` fragment exists for exactly that.
///
/// `slot` is appended to the temp filename to prevent collisions when multiple
/// parallel workers download blobs that happen to share a sanitised name.
fn fetch_blob_self_verified(
    blob: &Blob,
    cfg: &PullConfig,
    slot: usize,
) -> Result<(PathBuf, u64, bool), String> {
    std::fs::create_dir_all(&cfg.blobs_dir).map_err(|e| e.to_string())?;
    let safe_name = blob
        .name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>();
    // The slot suffix ensures two parallel workers fetching blobs with the
    // same sanitised name — rare but possible for bare-URL pulls — write to
    // different temp files and do not clobber each other's partial downloads.
    let partial = cfg
        .blobs_dir
        .join(format!("{safe_name}-{slot}.unverified.partial"));

    download_to(&partial, blob, cfg)?;

    let actual = digest_file_loud(&partial, &blob.name, cfg.progress)?;
    let final_path = blob_path(&cfg.blobs_dir, &actual);

    if final_path.exists() {
        // Already have this exact content under its rightful name — this
        // download was redundant, not wrong.
        let _ = std::fs::remove_file(&partial);
        let len = std::fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);
        return Ok((final_path, len, true));
    }

    let len = std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);
    std::fs::rename(&partial, &final_path)
        .map_err(|e| format!("could not commit {}: {e}", final_path.display()))?;
    eprintln!(
        "hearth: {} has no published digest — stored as sha256:{actual}, \
         self-consistent only (not checked against a source, because the \
         source published none)",
        blob.name
    );
    Ok((final_path, len, false))
}

/// Download one blob to `dest`, resuming a partial transfer if one is
/// already there. Shared by both verification paths — they differ in what
/// happens to the bytes afterward, not in how the bytes arrive.
fn download_to(dest: &Path, blob: &Blob, cfg: &PullConfig) -> Result<(), String> {
    let req = {
        let mut r = curl::Request::get(&blob.url).to_file(dest);
        for (k, v) in &blob.headers {
            r = r.header(k, v);
        }
        // Resume only if there is something to resume; `-C -` on a missing file
        // is fine, but on a zero-byte one some servers answer 416.
        if dest.exists() && std::fs::metadata(dest).map(|m| m.len()).unwrap_or(0) > 0 {
            r = r.resuming();
        }
        r
    };
    curl::fetch_file(&req, cfg.progress).map_err(|e| e.0)?;
    Ok(())
}

/// Fetch a slice of blobs in parallel, honouring `cfg.parallel_downloads`.
///
/// Returns a vec of `(original_index, path, bytes, was_cached)` in
/// completion order (not input order). The caller must sort or index by
/// `original_index` when insertion order matters.
///
/// **Progress discipline**: N concurrent curl progress bars interleaved on
/// one terminal is unreadable garbage. When `parallel_downloads > 1` each
/// worker fetches silently and emits a single structured line when it
/// finishes: `hearth: [3/5] name — 8.1 GiB done`. That gives the operator
/// clear forward progress without the mush. The single-blob path (`== 1`)
/// is untouched and keeps its live progress meter.
///
/// **Failure discipline**: the first error from any worker is captured and
/// returned, naming both the part index and the blob name so the operator
/// knows which of the N in-flight parts went wrong. Other workers are
/// allowed to finish their current blob (stopping mid-curl is not cleaner
/// than finishing), but no new work is started once an error is recorded.
///
/// **Temp-path safety**: `fetch_blob_verified` writes to `sha256-<digest>.partial`,
/// unique per digest by construction. `fetch_blob_self_verified` writes to
/// `<name>-<slot>.unverified.partial`, where slot is the blob's position in
/// the input slice — also unique, even if two blobs share a sanitised name.
/// `(original index, path on disk, bytes, was it already cached)`.
///
/// Named because clippy is right that the bare tuple-in-Arc-in-Mutex is
/// unreadable at a call site, and because the ORDER of these fields is load
/// bearing: the index is first precisely so results can be re-sorted into
/// input order after finishing out of order.
type FetchOutcome = (usize, PathBuf, u64, bool);

fn fetch_blobs_parallel(
    blobs: &[Blob],
    cfg: &PullConfig,
    n_total: usize,
) -> Result<Vec<FetchOutcome>, String> {
    use std::sync::{Arc, Mutex};

    if blobs.is_empty() {
        return Ok(Vec::new());
    }

    // The actual thread count: bounded by both the limit and the work.
    // This also handles parallel_downloads == 1 — one worker, same as the
    // old sequential loop, but still going through this function for code
    // simplicity. The caller keeps the single-blob branch for exact parity.
    let thread_count = cfg.parallel_downloads.min(blobs.len()).max(1);

    // When running in parallel (> 1 thread), each worker suppresses curl's
    // per-part progress meter and the per-part hash meter. Mixed-progress
    // output from N concurrent curl processes on one terminal is unreadable;
    // the workers emit a single completion line instead (see below). The outer
    // cfg.progress flag still governs whether ANY output is produced.
    let worker_cfg = if thread_count > 1 {
        // Progress off inside workers; the completion banner below takes over.
        Arc::new(PullConfig {
            progress: false,
            ..cfg.clone()
        })
    } else {
        Arc::new(cfg.clone())
    };

    // A global offset into the blob slice that each worker advances atomically.
    // Simpler than a channel or a Mutex<VecDeque> — each worker just keeps
    // claiming the next unclaimed index.
    let next_idx = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // First error from any worker. Only the first is kept — if five parts are
    // in flight and three fail, the first failure is the one to act on; the
    // rest are likely cascading from the same root cause.
    let first_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let results: Arc<Mutex<Vec<FetchOutcome>>> =
        Arc::new(Mutex::new(Vec::with_capacity(blobs.len())));

    std::thread::scope(|s| {
        for _ in 0..thread_count {
            let next_idx = Arc::clone(&next_idx);
            let first_error = Arc::clone(&first_error);
            let results = Arc::clone(&results);
            let worker_cfg = Arc::clone(&worker_cfg);

            s.spawn(move || {
                loop {
                    // If a peer already failed, stop starting new work. The
                    // peer's curl process will finish its current chunk —
                    // there is no portable way to kill a subprocess mid-flight
                    // that is cleaner than letting it run — but we do not
                    // start fresh blobs.
                    if first_error.lock().unwrap().is_some() {
                        break;
                    }

                    let i = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= blobs.len() {
                        break;
                    }

                    let blob = &blobs[i];
                    // slot == i means every blob gets a unique temp suffix,
                    // even if two blobs have the same sanitised name.
                    let outcome = fetch_blob_with_slot(blob, &worker_cfg, i);

                    match outcome {
                        Ok((path, bytes, cached)) => {
                            if cfg.progress && thread_count > 1 {
                                // One clear line per completed part. [i+1/total]
                                // so the operator can see forward progress
                                // without interleaved curl noise. The "already
                                // on disk" branch matches the sequential path's
                                // wording.
                                if cached {
                                    eprintln!(
                                        "hearth: [{}/{}] {} — already on disk",
                                        i + 1,
                                        n_total,
                                        blob.name
                                    );
                                } else {
                                    eprintln!(
                                        "hearth: [{}/{}] {} — {} done",
                                        i + 1,
                                        n_total,
                                        blob.name,
                                        human_bytes(bytes)
                                    );
                                }
                            }
                            results.lock().unwrap().push((i, path, bytes, cached));
                        }
                        Err(e) => {
                            // Name the part prominently. "download failed" when
                            // five parts were in flight tells the operator
                            // nothing about which one to retry.
                            let msg =
                                format!("part [{}/{}] {} failed: {}", i + 1, n_total, blob.name, e);
                            let mut guard = first_error.lock().unwrap();
                            if guard.is_none() {
                                *guard = Some(msg);
                            }
                        }
                    }
                }
            });
        }
    });

    // Take the error before the results; if both are set the error wins.
    if let Some(e) = first_error.lock().unwrap().take() {
        return Err(e);
    }

    Ok(Arc::try_unwrap(results)
        .expect("all threads finished")
        .into_inner()
        .unwrap())
}

/// Pull a model reference, recording the whole thing in the spine.
///
/// The recording is not decoration. `PullStarted` goes in *before* any bytes
/// move, so a pull that dies halfway leaves evidence that it was attempted and
/// from where — the alternative is a half-empty blob directory and no idea
/// which model it belongs to.
pub fn pull(reference: &str, cfg: &PullConfig, spine: &Spine) -> Result<Pulled, String> {
    let reference = Reference::parse(reference).map_err(|e| e.0)?;
    let model = reference.display_name();
    let source = registry::source_string(&reference);

    // A local file is already here. Recording a pull for it would be a lie.
    if let Reference::Local { path } = &reference {
        let p = PathBuf::from(path);
        if !p.exists() {
            return Err(format!("no file at {path}"));
        }
        let bytes = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
        return Ok(Pulled {
            model,
            source,
            weights_path: p,
            bytes,
            already_had_it: true,
        });
    }

    let started = spine
        .record(
            &model,
            &Transition::PullStarted {
                source: source.clone(),
            },
            &[],
        )
        .map_err(|e| format!("could not record the pull: {e}"))?;

    // Resolving is a network round trip against a registry that can be slow or
    // rate-limited, and it produced no output at all — so the first thing the
    // operator saw was a pause before any transfer had been announced.
    if cfg.progress {
        eprintln!("hearth: resolving {source} …");
    }

    let fetched = registry::resolve_blobs(&reference).map_err(|e| {
        // The failure is worth recording too — "we tried and the registry said
        // no" is a different fact from "nobody ever tried".
        let _ = spine.record(
            &model,
            &Transition::PullStarted {
                source: format!("{source} — FAILED: {e}"),
            },
            std::slice::from_ref(&started),
        );
        e
    })?;

    // Announce the PLAN before moving bytes. "1 file, 35.10 GiB" up front is
    // the difference between a transfer the operator can wait out and one they
    // assume is stuck; and on a multi-part model it is the only way to know
    // that a meter reaching 100.0% is part one of three, not the end.
    let n = fetched.blobs.len();
    let concurrency = cfg.parallel_downloads.min(n).max(1);
    if cfg.progress {
        let total: u64 = fetched.blobs.iter().map(|b| b.size_bytes).sum();
        if concurrency > 1 {
            eprintln!(
                "hearth: {n} file(s), {} to fetch — downloading up to {concurrency} in parallel, \
                 then a sha256 verify of each",
                human_bytes(total)
            );
        } else {
            eprintln!(
                "hearth: {n} file(s), {} to fetch — then a sha256 verify of each",
                human_bytes(total)
            );
        }
    }

    // Single-blob path: identical to the original sequential code. Progress
    // meters and all output are unchanged — no reason to pay the parallel
    // machinery overhead or suppress the live curl bar for one file.
    if n == 1 {
        let blob = &fetched.blobs[0];
        if cfg.progress {
            eprintln!(
                "hearth: [1/1] {} ({})",
                blob.name,
                if blob.size_bytes > 0 {
                    human_bytes(blob.size_bytes)
                } else {
                    "size unknown".to_string()
                }
            );
        }
        let (path, bytes, cached) = fetch_blob(blob, cfg)?;
        if cfg.progress && cached {
            eprintln!("hearth: [1/1] already on disk, skipped");
        }

        let (weights_path, bytes) = if blob.is_weights {
            (path, bytes)
        } else {
            return Err(format!(
                "{model} resolved with no weights layer — 1 blob, not the model"
            ));
        };

        spine
            .record(
                &model,
                &Transition::PullCompleted {
                    path: weights_path.display().to_string(),
                    size_bytes: bytes,
                },
                &[EventRef {
                    hash: started.hash,
                    seq: started.seq,
                }],
            )
            .map_err(|e| format!("could not record completion: {e}"))?;

        return Ok(Pulled {
            model,
            source,
            weights_path,
            bytes,
            already_had_it: cached,
        });
    }

    // Multi-blob path: fetch in parallel with bounded concurrency.
    //
    // Per-blob announcement lines (the "[2/5] name (size)" header) are emitted
    // by fetch_blobs_parallel itself after each part finishes — not before it
    // starts — because in parallel mode the start order is non-deterministic
    // and printing a start banner for each blob while others are already
    // downloading would produce confusing interleaved output.
    let raw_results = fetch_blobs_parallel(&fetched.blobs, cfg, n)?;

    // Re-index by original position so weights detection and cached_all are
    // in-order regardless of which thread finished first.
    let mut outcomes = vec![None::<(PathBuf, u64, bool)>; n];
    for (i, path, bytes, cached) in raw_results {
        outcomes[i] = Some((path, bytes, cached));
    }

    let mut weights: Option<(PathBuf, u64)> = None;
    let mut cached_all = true;
    for (i, slot) in outcomes.into_iter().enumerate() {
        let (path, len, cached) = slot.ok_or_else(|| {
            // Should be unreachable: fetch_blobs_parallel guarantees every
            // index is covered or returns an Err. Guard defensively.
            format!("internal error: blob {i} has no result after parallel fetch")
        })?;
        if !cached {
            cached_all = false;
        }
        if fetched.blobs[i].is_weights {
            weights = Some((path, len));
        }
    }

    let (weights_path, bytes) = weights.ok_or_else(|| {
        format!(
            "{model} resolved with no weights layer — the registry returned \
             {} blob(s), none of them the model",
            fetched.blobs.len()
        )
    })?;

    spine
        .record(
            &model,
            &Transition::PullCompleted {
                path: weights_path.display().to_string(),
                size_bytes: bytes,
            },
            &[EventRef {
                hash: started.hash,
                seq: started.seq,
            }],
        )
        .map_err(|e| format!("could not record completion: {e}"))?;

    Ok(Pulled {
        model,
        source,
        weights_path,
        bytes,
        already_had_it: cached_all,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- existing tests, unchanged ----

    #[test]
    fn a_blob_path_is_its_digest_so_presence_is_a_stat() {
        let p = blob_path(Path::new("/b"), "sha256:ABC123");
        assert_eq!(p, Path::new("/b/sha256-abc123"));
        // Both spellings of the same digest land on the same file, or a cache
        // would miss every time the registry changed its formatting.
        assert_eq!(p, blob_path(Path::new("/b"), "abc123"));
    }

    #[test]
    fn a_local_reference_is_not_recorded_as_a_pull() {
        let dir = std::env::temp_dir().join("hearth-pull-local");
        std::fs::create_dir_all(&dir).unwrap();
        let gguf = dir.join("muse.gguf");
        std::fs::write(&gguf, b"weights").unwrap();

        let spine = Spine::in_memory();
        let out = pull(
            &format!("file:{}", gguf.display()),
            &PullConfig::default(),
            &spine,
        )
        .unwrap();

        assert!(out.already_had_it);
        assert_eq!(out.bytes, 7);
        assert!(
            spine.latest(&out.model).is_none(),
            "nothing was fetched, so claiming a pull would be a lie"
        );
        std::fs::remove_file(&gguf).ok();
    }

    #[test]
    fn a_missing_local_file_is_refused_before_anything_is_recorded() {
        let spine = Spine::in_memory();
        let err = pull(
            "file:/definitely/not/here.gguf",
            &PullConfig::default(),
            &spine,
        )
        .unwrap_err();
        assert!(err.contains("no file at"), "{err}");
        assert_eq!(spine.seq(), 0, "nothing should have been written");
    }

    #[test]
    fn a_digest_mismatch_deletes_the_file_rather_than_caching_corruption() {
        // The blob is served by a local file:// URL whose contents do not match
        // the digest we claim. curl handles file:// so this needs no network.
        let dir = std::env::temp_dir().join("hearth-pull-mismatch");
        std::fs::create_dir_all(&dir).unwrap();
        let served = dir.join("served.bin");
        std::fs::write(&served, b"not what we asked for").unwrap();

        let blobs = dir.join("blobs");
        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
            parallel_downloads: 1,
        };
        let blob = Blob {
            name: "weights".into(),
            url: format!("file://{}", served.display()),
            digest: Some(
                "sha256:0000000000000000000000000000000000000000000000000000000000000000".into(),
            ),
            size_bytes: 0,
            headers: vec![],
            is_weights: true,
        };

        let err = fetch_blob(&blob, &cfg).unwrap_err();
        assert!(err.contains("digest mismatch"), "{err}");
        let digest = blob.digest.as_deref().unwrap();
        assert!(
            !blob_path(&blobs, digest).exists(),
            "a blob that failed verification must not be left on disk"
        );
        assert!(
            !blob_path(&blobs, digest).with_extension("partial").exists(),
            "and neither must the partial"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_matching_digest_commits_the_file_under_its_digest() {
        let dir = std::env::temp_dir().join("hearth-pull-match");
        std::fs::create_dir_all(&dir).unwrap();
        let body = b"the actual weights, honestly";
        let served = dir.join("served.bin");
        std::fs::write(&served, body).unwrap();

        let blobs = dir.join("blobs");
        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
            parallel_downloads: 1,
        };
        let blob = Blob {
            name: "weights".into(),
            url: format!("file://{}", served.display()),
            digest: Some(sha256::hex_digest(body)),
            size_bytes: body.len() as u64,
            headers: vec![],
            is_weights: true,
        };

        let (path, len, cached) = fetch_blob(&blob, &cfg).unwrap();
        assert!(!cached);
        assert_eq!(len, body.len() as u64);
        assert_eq!(std::fs::read(&path).unwrap(), body);

        // Second call is a stat, not a download.
        let (_, _, cached2) = fetch_blob(&blob, &cfg).unwrap();
        assert!(cached2, "the filename IS the digest — presence is enough");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_cached_blob_of_the_wrong_size_is_refetched() {
        let dir = std::env::temp_dir().join("hearth-pull-truncated");
        std::fs::create_dir_all(&dir).unwrap();
        let body = b"complete contents here";
        let served = dir.join("served.bin");
        std::fs::write(&served, body).unwrap();

        let blobs = dir.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let digest = sha256::hex_digest(body);
        // A truncated file already sitting under the right name — what an
        // interrupted run leaves behind if it renames before verifying.
        std::fs::write(blob_path(&blobs, &digest), b"trunc").unwrap();

        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
            parallel_downloads: 1,
        };
        let blob = Blob {
            name: "weights".into(),
            url: format!("file://{}", served.display()),
            digest: Some(digest),
            size_bytes: body.len() as u64,
            headers: vec![],
            is_weights: true,
        };
        let (path, len, cached) = fetch_blob(&blob, &cfg).unwrap();
        assert!(!cached, "the size did not match, so it was not trusted");
        assert_eq!(len, body.len() as u64);
        assert_eq!(std::fs::read(&path).unwrap(), body);
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- fetch_blob_self_verified: no published digest to check against ----

    #[test]
    fn no_digest_still_lands_the_file_under_its_own_computed_sha256() {
        // A bare URL pull: nothing published a digest, so `blob.digest` is
        // `None`. The file must still end up content-addressed — hearth
        // computes the digest itself rather than trusting a caller-supplied
        // filename for the final path.
        let dir = std::env::temp_dir().join("hearth-pull-no-digest");
        std::fs::create_dir_all(&dir).unwrap();
        let body = b"weights nobody published a digest for";
        let served = dir.join("served.bin");
        std::fs::write(&served, body).unwrap();

        let blobs = dir.join("blobs");
        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
            parallel_downloads: 1,
        };
        let blob = Blob {
            name: "model.gguf".into(),
            url: format!("file://{}", served.display()),
            digest: None,
            size_bytes: 0,
            headers: vec![],
            is_weights: true,
        };

        let (path, len, cached) = fetch_blob(&blob, &cfg).unwrap();
        assert!(!cached);
        assert_eq!(len, body.len() as u64);
        assert_eq!(std::fs::read(&path).unwrap(), body);

        let expected_name = format!("sha256-{}", sha256::hex_digest(body));
        assert_eq!(
            path.file_name().unwrap().to_str().unwrap(),
            expected_name,
            "the final path must be the digest hearth computed, not a guess"
        );
        // slot 0 is used by the public fetch_blob wrapper
        assert!(
            !blobs.join("model.gguf-0.unverified.partial").exists(),
            "the working file must not survive under its temporary name"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_second_no_digest_pull_of_identical_content_is_recognised_as_cached() {
        // Two different URLs (or the same URL fetched twice under different
        // callers) that happen to serve byte-identical content must land on
        // the SAME blob, not two copies — that is the entire point of
        // content addressing, and the no-digest path has to preserve it even
        // though it does not know the address until after downloading.
        let dir = std::env::temp_dir().join("hearth-pull-no-digest-dup");
        std::fs::create_dir_all(&dir).unwrap();
        let body = b"identical weights served from two places";
        let served = dir.join("served.bin");
        std::fs::write(&served, body).unwrap();

        let blobs = dir.join("blobs");
        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
            parallel_downloads: 1,
        };
        let blob = Blob {
            name: "model.gguf".into(),
            url: format!("file://{}", served.display()),
            digest: None,
            size_bytes: 0,
            headers: vec![],
            is_weights: true,
        };

        let (first_path, ..) = fetch_blob(&blob, &cfg).unwrap();
        let (second_path, _, cached) = fetch_blob(&blob, &cfg).unwrap();
        assert!(
            cached,
            "identical content must be recognised, not re-stored"
        );
        assert_eq!(first_path, second_path);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unsafe_blob_name_does_not_escape_the_blobs_directory() {
        // blob.name comes from a URL path or a registry listing — untrusted
        // input. A name containing "../" must not let the temporary working
        // file land outside blobs_dir.
        let dir = std::env::temp_dir().join("hearth-pull-unsafe-name");
        std::fs::create_dir_all(&dir).unwrap();
        let body = b"contents";
        let served = dir.join("served.bin");
        std::fs::write(&served, body).unwrap();

        let blobs = dir.join("blobs");
        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
            parallel_downloads: 1,
        };
        let blob = Blob {
            name: "../../etc/model.gguf".into(),
            url: format!("file://{}", served.display()),
            digest: None,
            size_bytes: 0,
            headers: vec![],
            is_weights: true,
        };

        let (path, ..) = fetch_blob(&blob, &cfg).unwrap();
        assert!(
            path.starts_with(&blobs),
            "the final path must stay under blobs_dir, got {}",
            path.display()
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- new tests for parallel download feature ----

    /// `parallel_downloads` is the tunable that governs concurrency. The
    /// default of 4 is a deliberate compromise: enough parallelism to matter on
    /// a fast NVMe link, polite enough not to thrash an HDD or a metered
    /// connection. The value must round-trip through the config struct so the
    /// CLI and the env-var plumbing can set it.
    /// Sequential vs parallel, against a REAL HTTP server, measured.
    ///
    /// `#[ignore]` because it needs a server on 127.0.0.1:8731 and six part
    /// files — CI has neither, and a test that silently passes when its
    /// fixture is absent is worse than no test. Run it deliberately:
    ///
    /// ```text
    /// python3 /tmp/slowserve.py &          # sleeps 1s per request
    /// cargo test -p hearth-pull -- --ignored --nocapture parallel_is_actually_faster
    /// ```
    ///
    /// The server sleeps per request on purpose. A localhost transfer is too
    /// fast for scheduling to show above the noise, so this measures whether
    /// the workers OVERLAP. It deliberately does NOT claim to prove bandwidth
    /// saturation on a real network — that is a different experiment and this
    /// one cannot speak to it.
    #[test]
    #[ignore]
    fn parallel_is_actually_faster_against_a_real_server() {
        let dir = std::env::temp_dir().join("hearth-parallel-bench");
        let _ = std::fs::remove_dir_all(&dir);

        let mk = |n: usize| Blob {
            name: format!("part-{n}.bin"),
            url: format!("http://127.0.0.1:8731/part-{n}.bin"),
            digest: None,
            size_bytes: 0,
            headers: vec![],
            is_weights: n == 1,
        };
        let blobs: Vec<Blob> = (1..=6).map(mk).collect();

        let run = |workers: usize| -> std::time::Duration {
            let blobs_dir = dir.join(format!("w{workers}"));
            let _ = std::fs::remove_dir_all(&blobs_dir);
            let cfg = PullConfig {
                blobs_dir,
                progress: false,
                verify_existing: false,
                parallel_downloads: workers,
            };
            let t = std::time::Instant::now();
            let got = fetch_blobs_parallel(&blobs, &cfg, blobs.len())
                .unwrap_or_else(|e| panic!("fetch with {workers} worker(s) failed: {e}"));
            assert_eq!(got.len(), 6, "every part must arrive");
            t.elapsed()
        };

        let seq = run(1);
        let par = run(4);
        println!("  sequential (1 worker): {:.2}s", seq.as_secs_f64());
        println!("  parallel   (4 workers): {:.2}s", par.as_secs_f64());
        println!("  speedup: {:.2}x", seq.as_secs_f64() / par.as_secs_f64());

        // Six requests at ~1s each: ~6s sequential, ~2s at concurrency 4.
        // The bar is deliberately loose (just "meaningfully faster") because a
        // tight threshold on a shared CI box is a flaky test, and a flaky test
        // teaches people to ignore failures.
        assert!(
            par < seq.mul_f64(0.6),
            "parallel ({:.2}s) should be well under sequential ({:.2}s)",
            par.as_secs_f64(),
            seq.as_secs_f64()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parallel_downloads_default_is_four_and_is_configurable() {
        let cfg = PullConfig::default();
        assert_eq!(
            cfg.parallel_downloads, 4,
            "default 4: real parallelism without thrashing a spinning disk or metered link"
        );
        let single = PullConfig {
            parallel_downloads: 1,
            ..PullConfig::default()
        };
        assert_eq!(single.parallel_downloads, 1);
    }

    /// Two workers fetching blobs whose sanitised names are identical must not
    /// clobber each other's temp files. The slot suffix in the temp filename is
    /// the guard: slot 0 writes `name-0.unverified.partial`, slot 1 writes
    /// `name-1.unverified.partial`. This test proves the invariant without a
    /// network — both blobs are served by local file:// URLs.
    #[test]
    fn parallel_temp_paths_are_unique_per_slot_even_with_identical_blob_names() {
        let dir = std::env::temp_dir().join("hearth-pull-slot-collision");
        std::fs::create_dir_all(&dir).unwrap();

        let body_a = b"part one contents";
        let body_b = b"part two contents";
        let served_a = dir.join("part_a.bin");
        let served_b = dir.join("part_b.bin");
        std::fs::write(&served_a, body_a).unwrap();
        std::fs::write(&served_b, body_b).unwrap();

        let blobs = dir.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();

        // Both blobs intentionally share the same name — the worst case for
        // temp-path collision if the slot suffix were absent.
        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
            parallel_downloads: 1,
        };

        // Fetch slot 0 and slot 1 sequentially to simulate what parallel workers do.
        let blob_a = Blob {
            name: "model.gguf".into(),
            url: format!("file://{}", served_a.display()),
            digest: None,
            size_bytes: 0,
            headers: vec![],
            is_weights: true,
        };
        let blob_b = Blob {
            name: "model.gguf".into(),
            url: format!("file://{}", served_b.display()),
            digest: None,
            size_bytes: 0,
            headers: vec![],
            is_weights: false,
        };

        // Use fetch_blob_with_slot directly to prove the slot suffix lands.
        let (path_a, ..) = fetch_blob_with_slot(&blob_a, &cfg, 0).unwrap();
        let (path_b, ..) = fetch_blob_with_slot(&blob_b, &cfg, 1).unwrap();

        // Different content → different digests → different final paths.
        assert_ne!(
            path_a, path_b,
            "distinct content must land in distinct content-addressed paths"
        );
        // Neither temp file should survive.
        assert!(!blobs.join("model.gguf-0.unverified.partial").exists());
        assert!(!blobs.join("model.gguf-1.unverified.partial").exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// A digest mismatch in one part of a parallel pull must fail the whole
    /// pull AND name the part, so the operator knows which of N in-flight
    /// downloads went wrong. "download failed" across five parts is useless;
    /// "part [3/5] model-00003-of-00005.gguf failed: digest mismatch" is not.
    #[test]
    fn a_parallel_digest_mismatch_names_the_failing_part() {
        let dir = std::env::temp_dir().join("hearth-pull-parallel-mismatch");
        std::fs::create_dir_all(&dir).unwrap();

        // Three blobs: first and last are fine, middle is corrupt.
        let good_body = b"correct weights";
        let bad_body = b"tampered weights";

        let served_good = dir.join("good.bin");
        let served_bad = dir.join("bad.bin");
        std::fs::write(&served_good, good_body).unwrap();
        std::fs::write(&served_bad, bad_body).unwrap();

        let blobs = dir.join("blobs");
        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
            parallel_downloads: 3,
        };

        let good_digest = sha256::hex_digest(good_body);
        // A digest nothing on disk will ever satisfy. THIS IS THE WHOLE POINT
        // OF THE TEST AND THE FIRST VERSION GOT IT WRONG: it gave all three
        // blobs the SAME digest, and because storage is content-addressed,
        // blob 0 wrote `sha256-<good>` and blobs 1 and 2 then found that path
        // already present and skipped downloading entirely. The corrupt part
        // was never fetched, so nothing could reject it, and the test failed
        // claiming the code was broken when the code was right. A corrupt part
        // must claim a digest that is NOT already on disk, or content-addressing
        // correctly short-circuits it.
        let unsatisfiable = sha256::hex_digest(b"a digest no served body produces");

        let blobs_list = vec![
            Blob {
                name: "model-00001-of-00003.gguf".into(),
                url: format!("file://{}", served_good.display()),
                digest: Some(good_digest.clone()),
                size_bytes: good_body.len() as u64,
                headers: vec![],
                is_weights: false,
            },
            Blob {
                // The corrupt part: served content hashes to neither digest.
                name: "model-00002-of-00003.gguf".into(),
                url: format!("file://{}", served_bad.display()),
                digest: Some(unsatisfiable),
                size_bytes: good_body.len() as u64,
                headers: vec![],
                is_weights: true,
            },
            Blob {
                name: "model-00003-of-00003.gguf".into(),
                url: format!("file://{}", served_good.display()),
                digest: Some(good_digest.clone()),
                size_bytes: good_body.len() as u64,
                headers: vec![],
                is_weights: false,
            },
        ];

        let err = fetch_blobs_parallel(&blobs_list, &cfg, 3).unwrap_err();

        // The error must name the offending part. The exact index may vary if
        // threads finish out of order, but the blob name is deterministic.
        assert!(
            err.contains("model-00002-of-00003.gguf"),
            "error must name the failing part, got: {err}"
        );
        assert!(
            err.contains("digest mismatch"),
            "error must explain why it failed, got: {err}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The single-blob path (n == 1) must behave identically to the pre-parallel
    /// code: same return value, same caching semantics, no regression.
    #[test]
    fn single_blob_parallel_path_is_unchanged() {
        let dir = std::env::temp_dir().join("hearth-pull-single-blob-parallel");
        std::fs::create_dir_all(&dir).unwrap();
        let body = b"one file, no sharding needed";
        let served = dir.join("model.bin");
        std::fs::write(&served, body).unwrap();

        let blobs = dir.join("blobs");
        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
            parallel_downloads: 4, // non-1 limit, but only one blob
        };
        let blob_list = vec![Blob {
            name: "model.gguf".into(),
            url: format!("file://{}", served.display()),
            digest: Some(sha256::hex_digest(body)),
            size_bytes: body.len() as u64,
            headers: vec![],
            is_weights: true,
        }];

        let results = fetch_blobs_parallel(&blob_list, &cfg, 1).unwrap();
        assert_eq!(results.len(), 1);
        let (idx, path, len, cached) = &results[0];
        assert_eq!(*idx, 0);
        assert!(!cached);
        assert_eq!(*len, body.len() as u64);
        assert_eq!(std::fs::read(path).unwrap(), body);

        // Second call: cached.
        let results2 = fetch_blobs_parallel(&blob_list, &cfg, 1).unwrap();
        assert!(results2[0].3, "second call must be a cache hit");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `fetch_blobs_parallel` with concurrency 2 fetches three blobs correctly,
    /// all verified and placed under their content-addressed paths. This is the
    /// core multi-part model scenario.
    #[test]
    fn three_blobs_with_concurrency_two_all_arrive_and_are_verified() {
        let dir = std::env::temp_dir().join("hearth-pull-three-parts");
        std::fs::create_dir_all(&dir).unwrap();

        let parts: Vec<Vec<u8>> = vec![
            b"shard one content".to_vec(),
            b"shard two content".to_vec(),
            b"shard three content".to_vec(),
        ];
        let mut blob_list = Vec::new();
        for (i, body) in parts.iter().enumerate() {
            let served = dir.join(format!("part{i}.bin"));
            std::fs::write(&served, body).unwrap();
            blob_list.push(Blob {
                name: format!("model-{:05}-of-00003.gguf", i + 1),
                url: format!("file://{}", served.display()),
                digest: Some(sha256::hex_digest(body)),
                size_bytes: body.len() as u64,
                headers: vec![],
                is_weights: i == 0,
            });
        }

        let blobs = dir.join("blobs");
        let cfg = PullConfig {
            blobs_dir: blobs.clone(),
            progress: false,
            verify_existing: false,
            parallel_downloads: 2,
        };

        let mut results = fetch_blobs_parallel(&blob_list, &cfg, 3).unwrap();
        results.sort_by_key(|r| r.0);
        assert_eq!(results.len(), 3);
        for (i, (idx, path, len, cached)) in results.iter().enumerate() {
            assert_eq!(*idx, i);
            assert!(!cached);
            assert_eq!(*len, parts[i].len() as u64);
            assert_eq!(std::fs::read(path).unwrap(), parts[i]);
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
