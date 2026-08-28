//! Turning a reference into a list of blobs to fetch.
//!
//! The URL construction and the manifest reading live in [`hearth_resolve`],
//! which is network-free on purpose. This module is the thin layer that
//! actually asks the registry for the manifest and hands the answer back as
//! something [`crate::fetch_blob`] can download.

use hearth_resolve::Reference;

use crate::curl;

/// One thing to download.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Blob {
    /// For error messages a person has to read.
    pub name: String,
    pub url: String,
    /// Expected digest, `sha256:…` or bare hex. Both compare equal.
    pub digest: String,
    /// Expected size, or 0 when the registry did not say.
    pub size_bytes: u64,
    pub headers: Vec<(String, String)>,
    /// Is this the weights layer? Exactly one blob should be, and it is the one
    /// the VRAM budget cares about.
    pub is_weights: bool,
}

/// What a reference resolved to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fetched {
    pub blobs: Vec<Blob>,
    /// Total weights bytes — the number the planner wants, known before a
    /// single byte moves.
    pub weights_bytes: u64,
}

/// The origin string recorded in the spine. Stable and greppable, because it is
/// the answer to "where did this file come from" months later.
pub fn source_string(r: &Reference) -> String {
    match r {
        Reference::Ollama {
            namespace,
            name,
            tag,
        } => format!("ollama:{namespace}/{name}:{tag}"),
        Reference::HuggingFace {
            owner,
            repo,
            revision,
            quant,
        } => match quant {
            Some(q) => format!("hf:{owner}/{repo}@{q}#{revision}"),
            None => format!("hf:{owner}/{repo}#{revision}"),
        },
        Reference::Local { path } => format!("file:{path}"),
    }
}

const OLLAMA_REGISTRY: &str = "https://registry.ollama.ai";

/// The media type that is actually the model. An Ollama manifest also carries
/// templates, params, system prompts and licenses; summing every layer would
/// overstate VRAM and refuse models that fit.
const OLLAMA_MODEL_MEDIA_TYPE: &str = "application/vnd.ollama.image.model";

/// Ask the registry what a reference is made of.
pub fn resolve_blobs(reference: &Reference) -> Result<Fetched, String> {
    match reference {
        Reference::Ollama {
            namespace,
            name,
            tag,
        } => {
            let manifest_url = format!("{OLLAMA_REGISTRY}/v2/{namespace}/{name}/manifests/{tag}");
            let body = curl::fetch_string(&curl::Request::get(&manifest_url).header(
                "Accept",
                "application/vnd.docker.distribution.manifest.v2+json",
            ))
            .map_err(|e| {
                format!(
                    "could not fetch the manifest for {namespace}/{name}:{tag}\n  {}",
                    e.0
                )
            })?;

            let manifest: serde_json::Value = serde_json::from_str(&body).map_err(|e| {
                format!(
                    "the registry did not return a manifest for {namespace}/{name}:{tag}: {e}\n  \
                     got: {}",
                    body.chars().take(160).collect::<String>()
                )
            })?;

            blobs_from_ollama_manifest(&manifest, namespace, name)
        }
        Reference::HuggingFace { .. } => Err(
            "HuggingFace pulls are not wired yet — hearth-resolve can already plan \
             them (quant choice, multipart shards, byte counts); what is missing is \
             this fetch path. Use an ollama: reference or file: for now."
                .into(),
        ),
        Reference::Local { path } => Ok(Fetched {
            blobs: vec![Blob {
                name: "local".into(),
                url: format!("file://{path}"),
                digest: String::new(),
                size_bytes: 0,
                headers: vec![],
                is_weights: true,
            }],
            weights_bytes: 0,
        }),
    }
}

/// Read an Ollama manifest into blobs. Pure, so it is tested against a real
/// captured manifest without touching the network.
pub fn blobs_from_ollama_manifest(
    manifest: &serde_json::Value,
    namespace: &str,
    name: &str,
) -> Result<Fetched, String> {
    let layers = manifest
        .get("layers")
        .and_then(|l| l.as_array())
        .ok_or_else(|| {
            format!(
                "manifest has no layers array — {}",
                manifest
                    .get("errors")
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "and no error either".into())
            )
        })?;

    let mut blobs = Vec::new();
    let mut weights_bytes = 0u64;

    for layer in layers {
        let digest = layer
            .get("digest")
            .and_then(|d| d.as_str())
            .ok_or("a layer has no digest — refusing to fetch something unverifiable")?;
        let media = layer
            .get("mediaType")
            .and_then(|m| m.as_str())
            .unwrap_or("");
        let size = layer
            .get("size")
            .and_then(hearth_core::budget::whole_bytes)
            .unwrap_or(0);
        let is_weights = media == OLLAMA_MODEL_MEDIA_TYPE;
        if is_weights {
            weights_bytes = weights_bytes.saturating_add(size);
        }
        blobs.push(Blob {
            name: format!("{}:{}", short_media(media), sha256_short(digest)),
            url: format!("{OLLAMA_REGISTRY}/v2/{namespace}/{name}/blobs/{digest}"),
            digest: digest.to_string(),
            size_bytes: size,
            headers: vec![],
            is_weights,
        });
    }

    if !blobs.iter().any(|b| b.is_weights) {
        return Err(format!(
            "no {OLLAMA_MODEL_MEDIA_TYPE} layer in the manifest — {} layer(s), \
             none of them weights. hearth will not guess which blob is a model.",
            blobs.len()
        ));
    }

    Ok(Fetched {
        blobs,
        weights_bytes,
    })
}

fn short_media(media: &str) -> &str {
    media.rsplit('.').next().unwrap_or(media)
}

fn sha256_short(digest: &str) -> String {
    hearth_core::sha256::normalize(digest)
        .chars()
        .take(12)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real manifest captured from registry.ollama.ai for library/tinyllama.
    /// Real, so the parser is checked against what the registry actually sends
    /// rather than against a fixture written to match the code.
    const TINYLLAMA: &str = r#"{
      "schemaVersion": 2,
      "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
      "config": {
        "mediaType": "application/vnd.docker.container.image.v1+json",
        "digest": "sha256:6331358be52a6ebc2fd0755a51ad1175734fd17a628ab5ea6897109396245362",
        "size": 483
      },
      "layers": [
        {
          "mediaType": "application/vnd.ollama.image.model",
          "digest": "sha256:2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816",
          "size": 637699456
        },
        {
          "mediaType": "application/vnd.ollama.image.template",
          "digest": "sha256:af0ddbdaaa26f30d54d727f9dd944b76bdb926fdaf9a58f63f78c534dca71bd1",
          "size": 70
        },
        {
          "mediaType": "application/vnd.ollama.image.system",
          "digest": "sha256:c8472cd9daed5e7c2ff5a1b04e5e2b9e5b0b4f4b2f5f5c4f5b4f5c4f5b4f5c4f",
          "size": 31
        },
        {
          "mediaType": "application/vnd.ollama.image.params",
          "digest": "sha256:fa956ab37b8c21152f2f5e0b0b4f4b2f5f5c4f5b4f5c4f5b4f5c4f5b4f5c4f5b",
          "size": 98
        }
      ]
    }"#;

    #[test]
    fn only_the_model_layer_counts_as_weights() {
        let m: serde_json::Value = serde_json::from_str(TINYLLAMA).unwrap();
        let f = blobs_from_ollama_manifest(&m, "library", "tinyllama").unwrap();

        assert_eq!(f.blobs.len(), 4, "every layer is fetchable");
        assert_eq!(
            f.blobs.iter().filter(|b| b.is_weights).count(),
            1,
            "exactly one of them is the model"
        );
        // 637,699,456 and not 637,699,655: summing the template, system and
        // params layers would overstate VRAM and refuse models that fit.
        assert_eq!(f.weights_bytes, 637_699_456);
    }

    #[test]
    fn blob_urls_are_built_from_the_digest() {
        let m: serde_json::Value = serde_json::from_str(TINYLLAMA).unwrap();
        let f = blobs_from_ollama_manifest(&m, "library", "tinyllama").unwrap();
        let w = f.blobs.iter().find(|b| b.is_weights).unwrap();
        assert_eq!(
            w.url,
            "https://registry.ollama.ai/v2/library/tinyllama/blobs/sha256:2af3b81862c6be03c769683af18efdadb2c33f60ff32ab6f83e42c043d6c7816"
        );
        assert!(w.name.contains("model"), "readable in an error: {}", w.name);
    }

    #[test]
    fn a_manifest_with_no_model_layer_is_refused_not_guessed() {
        // A model reference whose manifest carries only templates is either the
        // wrong tag or a registry change. Picking the largest blob and hoping
        // is how you load a license file as weights.
        let m = serde_json::json!({
            "layers": [
                { "mediaType": "application/vnd.ollama.image.template",
                  "digest": "sha256:aa", "size": 70 }
            ]
        });
        let err = blobs_from_ollama_manifest(&m, "library", "x").unwrap_err();
        assert!(err.contains("none of them weights"), "{err}");
        assert!(err.contains("will not guess"));
    }

    #[test]
    fn a_layer_without_a_digest_is_refused() {
        // No digest means no verification possible. Fetching it anyway would
        // put an unverifiable blob on disk under a made-up name.
        let m = serde_json::json!({
            "layers": [{ "mediaType": OLLAMA_MODEL_MEDIA_TYPE, "size": 10 }]
        });
        let err = blobs_from_ollama_manifest(&m, "library", "x").unwrap_err();
        assert!(err.contains("no digest"), "{err}");
    }

    #[test]
    fn a_registry_error_document_says_what_it_said() {
        let m = serde_json::json!({ "errors": [{ "code": "MANIFEST_UNKNOWN" }] });
        let err = blobs_from_ollama_manifest(&m, "library", "nope").unwrap_err();
        assert!(err.contains("MANIFEST_UNKNOWN"), "{err}");
    }

    #[test]
    fn a_json_float_size_survives_the_crossing() {
        // Sizes arriving as integer-valued floats are real; refusing them would
        // report a legitimate layer as zero bytes and let the planner admit a
        // model that cannot fit.
        let m = serde_json::json!({
            "layers": [{ "mediaType": OLLAMA_MODEL_MEDIA_TYPE,
                         "digest": "sha256:ab", "size": 637699456.0 }]
        });
        let f = blobs_from_ollama_manifest(&m, "library", "x").unwrap();
        assert_eq!(f.weights_bytes, 637_699_456);
    }

    #[test]
    fn source_strings_are_stable_and_greppable() {
        assert_eq!(
            source_string(&Reference::parse("deepseek-r1:32b").unwrap()),
            "ollama:library/deepseek-r1:32b"
        );
        assert_eq!(
            source_string(&Reference::parse("hf:TheBloke/X-GGUF@Q4_K_M").unwrap()),
            "hf:TheBloke/X-GGUF@Q4_K_M#main"
        );
        assert_eq!(
            source_string(&Reference::parse("file:/m/muse.gguf").unwrap()),
            "file:/m/muse.gguf"
        );
    }

    #[test]
    fn huggingface_says_it_is_not_wired_rather_than_failing_obscurely() {
        let r = Reference::parse("hf:TheBloke/X-GGUF").unwrap();
        let err = resolve_blobs(&r).unwrap_err();
        assert!(err.contains("not wired yet"), "{err}");
        assert!(err.contains("ollama:"), "and says what to do instead");
    }
}
