//! Two catalogs, one command.
//!
//! The point of aggregating is that a person types what they already type and
//! it works. So these tests are mostly about not surprising anyone: a bare
//! name means what Ollama users expect, a HuggingFace repo picks a sane
//! quantization and SAYS it chose, and nothing ever half-downloads a split
//! model and calls it done.

use hearth_resolve::plan::{
    plan_from_hf_files, plan_from_ollama_manifest, HfSource, QuantChoice, RepoFile,
};
use hearth_resolve::{available_quants, pick_gguf, quant_of, total_parts, Reference};
use serde_json::json;

fn f(path: &str, size: u64) -> RepoFile {
    RepoFile {
        path: path.into(),
        size_bytes: Some(size),
    }
}

// ---------------------------------------------------------------------------
// What did they mean?
// ---------------------------------------------------------------------------

#[test]
fn a_bare_name_is_ollama_because_that_is_what_people_type() {
    // Punishing existing muscle memory to make a point about neutrality is how
    // you build a tool nobody switches to.
    assert_eq!(
        Reference::parse("llama3").unwrap(),
        Reference::Ollama {
            namespace: "library".into(),
            name: "llama3".into(),
            tag: "latest".into()
        }
    );
    assert_eq!(
        Reference::parse("qwen3:14b").unwrap(),
        Reference::Ollama {
            namespace: "library".into(),
            name: "qwen3".into(),
            tag: "14b".into()
        }
    );
}

#[test]
fn a_namespaced_ollama_model_keeps_its_namespace() {
    assert_eq!(
        Reference::parse("ollama:someone/model:v2").unwrap(),
        Reference::Ollama {
            namespace: "someone".into(),
            name: "model".into(),
            tag: "v2".into()
        }
    );
    // Namespaces work without the scheme too — `user/model` is unambiguous.
    assert_eq!(
        Reference::parse("interchained/muse:latest").unwrap(),
        Reference::Ollama {
            namespace: "interchained".into(),
            name: "muse".into(),
            tag: "latest".into()
        }
    );
}

#[test]
fn huggingface_takes_a_quant_and_a_revision() {
    assert_eq!(
        Reference::parse("hf:TheBloke/Llama-2-7B-GGUF").unwrap(),
        Reference::HuggingFace {
            owner: "TheBloke".into(),
            repo: "Llama-2-7B-GGUF".into(),
            revision: "main".into(),
            quant: None,
        }
    );
    assert_eq!(
        Reference::parse("hf:TheBloke/Llama-2-7B-GGUF@Q5_K_M#refs/pr/3").unwrap(),
        Reference::HuggingFace {
            owner: "TheBloke".into(),
            repo: "Llama-2-7B-GGUF".into(),
            revision: "refs/pr/3".into(),
            quant: Some("Q5_K_M".into()),
        }
    );
    // hf:// and hf: are the same thing; people type both.
    assert_eq!(
        Reference::parse("hf://owner/repo").unwrap(),
        Reference::parse("hf:owner/repo").unwrap()
    );
}

#[test]
fn a_path_is_a_path_even_though_it_has_a_colon_in_it() {
    // Checked before schemes on purpose: a Windows path and a scheme prefix
    // both contain a colon, and the file on disk is the less surprising read.
    for p in [
        "./models/muse.gguf",
        "/opt/models/x.gguf",
        "~/m.gguf",
        "C:/models/x.gguf",
    ] {
        assert!(
            matches!(Reference::parse(p).unwrap(), Reference::Local { .. }),
            "{p} should be local"
        );
    }
}

#[test]
fn the_same_model_always_gets_the_same_key() {
    // Two references meaning one model must key identically, or the weights
    // download twice and the residency history splits in half.
    assert_eq!(
        Reference::parse("llama3").unwrap().key(),
        Reference::parse("ollama:library/llama3:latest")
            .unwrap()
            .key()
    );
}

#[test]
fn a_nonsense_reference_says_what_was_wrong_with_it() {
    for bad in ["", "hf:", "hf:justone", "hf:a/b/c", "llama3:", "hf:a/b@"] {
        let e = Reference::parse(bad).unwrap_err();
        assert!(!e.0.is_empty(), "{bad:?} needs a real message");
    }
    assert!(Reference::parse("hf:a/b/c")
        .unwrap_err()
        .0
        .contains("owner/repo"));
}

// ---------------------------------------------------------------------------
// Ollama manifests.
// ---------------------------------------------------------------------------

#[test]
fn only_the_model_layer_is_the_model() {
    // An Ollama manifest also carries the prompt template, the params and the
    // licence. Downloading those as though they were weights gives you a
    // 30-byte "model" and a very confusing load error.
    let manifest = json!({
        "layers": [
            { "mediaType": "application/vnd.ollama.image.license", "digest": "sha256:aaa", "size": 30 },
            { "mediaType": "application/vnd.ollama.image.template", "digest": "sha256:bbb", "size": 90 },
            { "mediaType": "application/vnd.ollama.image.model", "digest": "sha256:ccc", "size": 4_600_000_000u64 },
            { "mediaType": "application/vnd.ollama.image.params", "digest": "sha256:ddd", "size": 40 }
        ]
    });
    let p = plan_from_ollama_manifest(
        &manifest,
        "https://registry.ollama.ai",
        "library",
        "llama3",
        "k".into(),
        "llama3:latest".into(),
    )
    .unwrap();

    assert_eq!(p.blobs.len(), 1);
    assert_eq!(p.blobs[0].digest.as_deref(), Some("sha256:ccc"));
    assert_eq!(p.total_bytes(), Some(4_600_000_000));
    assert_eq!(
        p.blobs[0].url,
        "https://registry.ollama.ai/v2/library/llama3/blobs/sha256:ccc"
    );
}

#[test]
fn a_manifest_with_no_model_layer_says_what_it_did_have() {
    let manifest = json!({ "layers": [
        { "mediaType": "application/vnd.ollama.image.license", "digest": "sha256:a", "size": 1 }
    ]});
    let e = plan_from_ollama_manifest(
        &manifest,
        "https://r",
        "library",
        "x",
        "k".into(),
        "x".into(),
    )
    .unwrap_err();
    assert!(
        e.0.contains("license"),
        "name what was actually there: {}",
        e.0
    );
}

#[test]
fn ollama_never_claims_a_quantization_it_was_not_told() {
    // The tag encodes it; the manifest does not name it. Reporting one would
    // be inventing information.
    let manifest = json!({ "layers": [
        { "mediaType": "application/vnd.ollama.image.model", "digest": "sha256:c", "size": 10 }
    ]});
    let p = plan_from_ollama_manifest(
        &manifest,
        "https://r",
        "library",
        "x",
        "k".into(),
        "x".into(),
    )
    .unwrap();
    assert_eq!(p.quant, None);
    assert!(!p.quant_was_chosen_for_you);
}

// ---------------------------------------------------------------------------
// HuggingFace: eight files, one right answer.
// ---------------------------------------------------------------------------

#[test]
fn reads_the_quantization_out_of_a_filename_longest_match_first() {
    // Q4_K must not swallow Q4_K_M — that is a different, larger, better file.
    assert_eq!(
        quant_of("llama-2-7b.Q4_K_M.gguf").as_deref(),
        Some("Q4_K_M")
    );
    assert_eq!(
        quant_of("llama-2-7b.Q4_K_S.gguf").as_deref(),
        Some("Q4_K_S")
    );
    assert_eq!(quant_of("model-q5_k_m.gguf").as_deref(), Some("Q5_K_M"));
    assert_eq!(quant_of("Mistral.IQ3_XS.gguf").as_deref(), Some("IQ3_XS"));
    assert_eq!(quant_of("model.f16.gguf").as_deref(), Some("F16"));
    assert_eq!(quant_of("just-a-model.gguf"), None);
}

#[test]
fn picks_the_quantization_the_community_actually_uses_and_admits_it_chose() {
    let files = vec![
        f("llama.Q2_K.gguf", 1),
        f("llama.Q8_0.gguf", 8),
        f("llama.Q4_K_M.gguf", 4),
        f("llama.F16.gguf", 16),
    ];
    let (picked, quant, chosen_for_you) = pick_gguf(&files, None).unwrap();
    assert_eq!(quant, "Q4_K_M");
    assert_eq!(picked.len(), 1);
    assert!(chosen_for_you, "the caller must be able to tell we decided");
}

#[test]
fn a_pinned_quantization_is_honoured_and_is_not_our_choice() {
    let files = vec![f("m.Q4_K_M.gguf", 4), f("m.Q8_0.gguf", 8)];
    let (picked, quant, chosen_for_you) = pick_gguf(&files, Some("q8_0")).unwrap();
    assert_eq!(quant, "Q8_0");
    assert_eq!(picked[0].path, "m.Q8_0.gguf");
    assert!(!chosen_for_you);
}

#[test]
fn asking_for_a_quantization_that_is_not_there_lists_what_is() {
    let files = vec![f("m.Q4_K_M.gguf", 4), f("m.Q8_0.gguf", 8)];
    let e = pick_gguf(&files, Some("Q2_K")).unwrap_err();
    assert!(e.0.contains("Q4_K_M") && e.0.contains("Q8_0"), "{}", e.0);
}

#[test]
fn every_single_part_of_a_split_model_comes_back_in_order() {
    // Half a split GGUF is not a model. Picking one part is a download that
    // succeeds and then fails to load with an error nobody can read.
    let files = vec![
        f("big.Q4_K_M-00003-of-00003.gguf", 3),
        f("big.Q4_K_M-00001-of-00003.gguf", 1),
        f("big.Q4_K_M-00002-of-00003.gguf", 2),
        f("big.Q8_0.gguf", 9),
    ];
    let (picked, quant, _) = pick_gguf(&files, Some("Q4_K_M")).unwrap();
    assert_eq!(quant, "Q4_K_M");
    assert_eq!(picked.len(), 3);
    assert_eq!(picked[0].path, "big.Q4_K_M-00001-of-00003.gguf");
    assert_eq!(picked[2].path, "big.Q4_K_M-00003-of-00003.gguf");
}

#[test]
fn an_incomplete_split_is_refused_rather_than_downloaded() {
    let files = vec![
        f("big.Q4_K_M-00001-of-00003.gguf", 1),
        f("big.Q4_K_M-00002-of-00003.gguf", 2),
    ];
    let e = pick_gguf(&files, Some("Q4_K_M")).unwrap_err();
    assert!(e.0.contains("3 parts") || e.0.contains("into 3"), "{}", e.0);
}

#[test]
fn a_lone_unlabelled_gguf_is_still_a_usable_model() {
    // Plenty of repos publish exactly one file with no quant in the name.
    // Refusing on principle would be pedantry.
    let files = [f("model.gguf", 7)];
    let (picked, _, chosen) = pick_gguf(&files, None).unwrap();
    assert_eq!(picked.len(), 1);
    assert!(chosen);
}

#[test]
fn a_repo_with_no_gguf_says_why_rather_than_returning_nothing() {
    let files = vec![f("model.safetensors", 1), f("config.json", 2)];
    let e = pick_gguf(&files, None).unwrap_err();
    assert!(
        e.0.contains("safetensors"),
        "point at the actual problem: {}",
        e.0
    );
}

#[test]
fn total_parts_reads_the_of_suffix() {
    assert_eq!(total_parts("x-00001-of-00007.gguf"), Some(7));
    assert_eq!(total_parts("x.Q4_K_M.gguf"), None);
}

#[test]
fn available_quants_are_deduped_and_sorted_for_an_error_message() {
    let files = [
        f("a.Q8_0.gguf", 1),
        f("b.Q4_K_M.gguf", 1),
        f("c.Q4_K_M.gguf", 1),
    ];
    let refs: Vec<&RepoFile> = files.iter().collect();
    assert_eq!(available_quants(&refs), vec!["Q4_K_M", "Q8_0"]);
}

// ---------------------------------------------------------------------------
// The finished plan.
// ---------------------------------------------------------------------------

#[test]
fn a_hf_plan_builds_resolve_urls_and_admits_it_has_no_digest() {
    let p = plan_from_hf_files(
        HfSource {
            owner: "TheBloke",
            repo: "Llama-2-7B-GGUF",
            revision: "main",
        },
        vec![f("llama-2-7b.Q4_K_M.gguf", 4_080_000_000)],
        QuantChoice {
            quant: "Q4_K_M".into(),
            chosen_for_you: true,
        },
        "k".into(),
        "TheBloke/Llama-2-7B-GGUF@Q4_K_M".into(),
    );
    assert_eq!(
        p.blobs[0].url,
        "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf"
    );
    assert_eq!(p.blobs[0].filename, "llama-2-7b.Q4_K_M.gguf");
    assert_eq!(
        p.blobs[0].digest, None,
        "HuggingFace does not publish one — inventing it makes verification a \
         ritual that always passes",
    );
    assert!(p.quant_was_chosen_for_you);
}

#[test]
fn an_unknown_size_anywhere_makes_the_total_unknown() {
    // A partial sum understates the download and would let the VRAM planner
    // admit something that does not fit.
    let mut p = plan_from_hf_files(
        HfSource {
            owner: "o",
            repo: "r",
            revision: "main",
        },
        vec![
            f("a-00001-of-00002.gguf", 10),
            f("a-00002-of-00002.gguf", 10),
        ],
        QuantChoice {
            quant: "Q4_K_M".into(),
            chosen_for_you: false,
        },
        "k".into(),
        "n".into(),
    );
    assert_eq!(p.total_bytes(), Some(20));
    assert!(p.is_multipart());
    p.blobs[1].size_bytes = None;
    assert_eq!(p.total_bytes(), None);
}
