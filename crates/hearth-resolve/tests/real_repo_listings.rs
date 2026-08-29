//! Real GGUF repo listings, captured from HuggingFace.
//!
//! These are not fixtures anyone invented: they are the file lists that
//! `ggml-org/gpt-oss-20b-GGUF` and
//! `bartowski/moonshotai_Kimi-Linear-48B-A3B-Instruct-GGUF` actually serve.
//! The bugs below were found by running the resolver against them, not by
//! reading it.

use hearth_resolve::plan::{available_quants, is_auxiliary_gguf, pick_gguf, quant_of, RepoFile};

fn rf(p: &str, s: u64) -> RepoFile {
    RepoFile {
        path: p.to_string(),
        size_bytes: Some(s),
    }
}

/// ggml-org/gpt-oss-20b-GGUF — two EAGLE-3 draft heads and one real model.
fn gpt_oss_20b() -> Vec<RepoFile> {
    vec![
        rf("eagle3-gpt-oss-20b-Q8_0.gguf", 920_000_000),
        rf("eagle3-gpt-oss-20b-BF16.gguf", 1_720_000_000),
        rf("gpt-oss-20b-MXFP4.gguf", 12_110_000_000),
    ]
}

fn kimi_linear() -> Vec<RepoFile> {
    vec![
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-IQ1_M.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-IQ1_S.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-IQ2_M.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-IQ2_S.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-IQ2_XS.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-IQ2_XXS.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-IQ3_M.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-IQ3_XS.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-IQ3_XXS.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-IQ4_NL.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-IQ4_XS.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q2_K.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q2_K_L.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q3_K_L.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q3_K_M.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q3_K_S.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q3_K_XL.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q4_0.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q4_1.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q4_K_L.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q4_K_M.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q4_K_S.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q5_K_L.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q5_K_M.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q5_K_S.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q6_K.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q6_K_L.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q8_0/moonshotai_Kimi-Linear-48B-A3B-Instruct-Q8_0-00001-of-00002.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-Q8_0/moonshotai_Kimi-Linear-48B-A3B-Instruct-Q8_0-00002-of-00002.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-bf16/moonshotai_Kimi-Linear-48B-A3B-Instruct-bf16-00001-of-00003.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-bf16/moonshotai_Kimi-Linear-48B-A3B-Instruct-bf16-00002-of-00003.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-bf16/moonshotai_Kimi-Linear-48B-A3B-Instruct-bf16-00003-of-00003.gguf", 1),
        rf("moonshotai_Kimi-Linear-48B-A3B-Instruct-imatrix.gguf", 1)
    ]
}

// ---------------------------------------------------------------------------
// The bug: an unpinned pull served a draft head as the model.
// ---------------------------------------------------------------------------

#[test]
fn an_unpinned_gpt_oss_pull_gets_the_model_not_the_draft_head() {
    let (parts, chosen, auto) = pick_gguf(&gpt_oss_20b(), None).expect("must resolve");
    assert!(auto, "nothing was pinned");
    assert_eq!(chosen, "MXFP4");
    assert_eq!(parts.len(), 1);
    assert_eq!(
        parts[0].path, "gpt-oss-20b-MXFP4.gguf",
        "picked a 0.92 GB EAGLE-3 draft head instead of the 12.11 GB model"
    );
}

#[test]
fn mxfp4_is_a_quantization_hearth_can_name() {
    // `@MXFP4` used to fail with "no MXFP4 in this repo -- available: BF16,
    // Q8_0" while gpt-oss-20b-MXFP4.gguf sat in the listing. gpt-oss is
    // PUBLISHED in MXFP4, so not knowing the name made the reference weights
    // unpullable.
    assert_eq!(quant_of("gpt-oss-20b-MXFP4.gguf").as_deref(), Some("MXFP4"));
    let (parts, chosen, _) = pick_gguf(&gpt_oss_20b(), Some("MXFP4")).expect("must resolve");
    assert_eq!(chosen, "MXFP4");
    assert_eq!(parts[0].path, "gpt-oss-20b-MXFP4.gguf");
    // Case-insensitive, like every other quant.
    assert!(pick_gguf(&gpt_oss_20b(), Some("mxfp4")).is_ok());
}

#[test]
fn a_draft_head_is_not_offered_as_an_available_quantization() {
    let err = pick_gguf(&gpt_oss_20b(), Some("Q8_0"))
        .expect_err("Q8_0 exists only as a draft head, so it must not resolve");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("MXFP4"),
        "error should list the real options: {msg}"
    );
}

#[test]
fn auxiliary_markers_are_recognised() {
    for p in [
        "eagle3-gpt-oss-20b-Q8_0.gguf",
        "mmproj-Qwen3-VL-8B-f16.gguf",
        "model-draft-Q4_K_M.gguf",
        "some-lora-Q8_0.gguf",
    ] {
        assert!(is_auxiliary_gguf(p), "{p} should be auxiliary");
    }
    for p in [
        "gpt-oss-20b-MXFP4.gguf",
        "granite-4.2-8b-Q8_0.gguf",
        "moonshotai_Kimi-Linear-48B-A3B-Instruct-Q5_K_M.gguf",
    ] {
        assert!(!is_auxiliary_gguf(p), "{p} is the model, not auxiliary");
    }
}

// ---------------------------------------------------------------------------
// Quant names the vocabulary was missing. Each of these is a real file in the
// real bartowski repo that reported the WRONG quantization or none at all.
// ---------------------------------------------------------------------------

#[test]
fn the_bartowski_ladder_is_read_exactly() {
    for (file, want) in [
        (
            "moonshotai_Kimi-Linear-48B-A3B-Instruct-Q4_K_L.gguf",
            "Q4_K_L",
        ),
        (
            "moonshotai_Kimi-Linear-48B-A3B-Instruct-Q3_K_XL.gguf",
            "Q3_K_XL",
        ),
        (
            "moonshotai_Kimi-Linear-48B-A3B-Instruct-Q5_K_L.gguf",
            "Q5_K_L",
        ),
        (
            "moonshotai_Kimi-Linear-48B-A3B-Instruct-Q6_K_L.gguf",
            "Q6_K_L",
        ),
        (
            "moonshotai_Kimi-Linear-48B-A3B-Instruct-Q2_K_L.gguf",
            "Q2_K_L",
        ),
        (
            "moonshotai_Kimi-Linear-48B-A3B-Instruct-IQ1_M.gguf",
            "IQ1_M",
        ),
        (
            "moonshotai_Kimi-Linear-48B-A3B-Instruct-IQ2_M.gguf",
            "IQ2_M",
        ),
        (
            "moonshotai_Kimi-Linear-48B-A3B-Instruct-IQ2_S.gguf",
            "IQ2_S",
        ),
        (
            "moonshotai_Kimi-Linear-48B-A3B-Instruct-IQ3_M.gguf",
            "IQ3_M",
        ),
        // Longest match must still win over the shorter prefix.
        (
            "moonshotai_Kimi-Linear-48B-A3B-Instruct-Q4_K_M.gguf",
            "Q4_K_M",
        ),
        (
            "moonshotai_Kimi-Linear-48B-A3B-Instruct-Q4_K_S.gguf",
            "Q4_K_S",
        ),
    ] {
        assert_eq!(quant_of(file).as_deref(), Some(want), "misread {file}");
    }
}

#[test]
fn every_gguf_in_a_real_repo_is_named() {
    // A file whose quantization cannot be named is a file the resolver cannot
    // offer. Before the vocabulary was widened, several of the 33 came back
    // either mislabelled or None.
    let files = kimi_linear();
    let unnamed: Vec<&str> = files
        .iter()
        .filter(|f| !f.path.contains("imatrix"))
        .filter(|f| quant_of(&f.path).is_none())
        .map(|f| f.path.as_str())
        .collect();
    assert!(unnamed.is_empty(), "unnamed quantizations: {unnamed:?}");
}

#[test]
fn pinning_any_advertised_quant_resolves_it() {
    let files = kimi_linear();
    let refs: Vec<&RepoFile> = files.iter().collect();
    for q in available_quants(&refs) {
        let (parts, chosen, _) =
            pick_gguf(&files, Some(&q)).unwrap_or_else(|e| panic!("@{q} failed: {e:?}"));
        assert_eq!(chosen, q);
        assert!(!parts.is_empty(), "@{q} resolved to zero files");
    }
}

#[test]
fn an_unpinned_kimi_pull_still_lands_on_a_k_quant() {
    // The block-float formats were appended to the END of the preference list
    // precisely so this did not change.
    let (_, chosen, auto) = pick_gguf(&kimi_linear(), None).expect("must resolve");
    assert_eq!(chosen, "Q4_K_M");
    assert!(auto);
}

#[test]
fn a_repo_of_only_auxiliary_files_still_reports_what_it_has() {
    // Filtering must never turn a describable repo into "empty".
    let only_drafts = vec![
        rf("eagle3-gpt-oss-20b-Q8_0.gguf", 920_000_000),
        rf("eagle3-gpt-oss-20b-BF16.gguf", 1_720_000_000),
    ];
    let (parts, chosen, _) = pick_gguf(&only_drafts, Some("Q8_0"))
        .expect("fall back to the unfiltered list rather than claim no GGUFs");
    assert_eq!(chosen, "Q8_0");
    assert_eq!(parts[0].path, "eagle3-gpt-oss-20b-Q8_0.gguf");
}

#[test]
fn safetensors_only_still_says_so() {
    let st = vec![
        rf("model-00001-of-00002.safetensors", 1),
        rf("config.json", 1),
    ];
    let msg = format!("{:?}", pick_gguf(&st, None).expect_err("no gguf"));
    assert!(msg.contains("GGUF"), "{msg}");
}
