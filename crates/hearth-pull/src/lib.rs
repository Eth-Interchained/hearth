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
}

impl Default for PullConfig {
    fn default() -> Self {
        PullConfig {
            blobs_dir: PathBuf::from("./blobs"),
            progress: true,
            verify_existing: false,
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

/// Fetch one blob to its content-addressed home, verifying the digest.
///
/// Downloads to a `.partial` sibling and renames only after the digest
/// matches. Without that, an interrupted transfer leaves a file whose *name*
/// claims a digest its *contents* do not have — and every later run trusts the
/// name. The rename is the commit.
pub fn fetch_blob(blob: &Blob, cfg: &PullConfig) -> Result<(PathBuf, u64, bool), String> {
    let final_path = blob_path(&cfg.blobs_dir, &blob.digest);

    if final_path.exists() {
        let len = std::fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);
        if cfg.verify_existing {
            let actual = sha256::hex_digest_file(&final_path)
                .map_err(|e| format!("could not read {}: {e}", final_path.display()))?;
            if !sha256::matches(&blob.digest, &actual) {
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
    let req = {
        let mut r = curl::Request::get(&blob.url).to_file(&partial);
        for (k, v) in &blob.headers {
            r = r.header(k, v);
        }
        // Resume only if there is something to resume; `-C -` on a missing file
        // is fine, but on a zero-byte one some servers answer 416.
        if partial.exists() && std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0) > 0 {
            r = r.resuming();
        }
        r
    };

    curl::fetch_file(&req, cfg.progress).map_err(|e| e.0)?;

    let actual = sha256::hex_digest_file(&partial)
        .map_err(|e| format!("could not hash {}: {e}", partial.display()))?;
    if !sha256::matches(&blob.digest, &actual) {
        // Delete it. Keeping a blob that failed verification is how a corrupt
        // download becomes a permanent, cached, corrupt download.
        let _ = std::fs::remove_file(&partial);
        return Err(format!(
            "digest mismatch for {}\n  expected sha256:{}\n  got      sha256:{}\n  \
             the file has been deleted; a resumed transfer over a proxy that \
             rewrote a range will do this, so a plain retry is worth one attempt",
            blob.name,
            sha256::normalize(&blob.digest),
            actual
        ));
    }

    let len = std::fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);
    std::fs::rename(&partial, &final_path)
        .map_err(|e| format!("could not commit {}: {e}", final_path.display()))?;
    Ok((final_path, len, false))
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

    let mut weights: Option<(PathBuf, u64)> = None;
    let mut cached_all = true;
    for blob in &fetched.blobs {
        let (path, len, cached) = fetch_blob(blob, cfg)?;
        if !cached {
            cached_all = false;
        }
        if blob.is_weights {
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
        };
        let blob = Blob {
            name: "weights".into(),
            url: format!("file://{}", served.display()),
            digest: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                .into(),
            size_bytes: 0,
            headers: vec![],
            is_weights: true,
        };

        let err = fetch_blob(&blob, &cfg).unwrap_err();
        assert!(err.contains("digest mismatch"), "{err}");
        assert!(
            !blob_path(&blobs, &blob.digest).exists(),
            "a blob that failed verification must not be left on disk"
        );
        assert!(
            !blob_path(&blobs, &blob.digest)
                .with_extension("partial")
                .exists(),
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
        };
        let blob = Blob {
            name: "weights".into(),
            url: format!("file://{}", served.display()),
            digest: sha256::hex_digest(body),
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
        };
        let blob = Blob {
            name: "weights".into(),
            url: format!("file://{}", served.display()),
            digest,
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
}
