//! First-token warmup — because "loaded" and "fast" are different promises.
//!
//! Process-per-model already puts the WEIGHTS in VRAM at spawn; that is the
//! residency promise. But the first real request still pays a one-time cost —
//! graph build, KV cache allocation — so the first caller of the night gets a
//! noticeably slower answer than everyone after them. Ollama users work around
//! this by hand, curling an empty prompt at each model after boot.
//!
//! hearth does it for you: after a model turns `Resident`, one throwaway
//! single-token generation goes through it, and the first *real* request finds
//! everything hot. Which models get this is the operator's call:
//!
//!   --preload-models=N      warm the first N, in declaration order —
//!                           declaration order IS priority order
//!   --preload-model=NAME    warm exactly this one (repeatable)
//!   (neither)               warm everything admitted, because preloading is
//!                           the product; opting out is the special case
//!   --preload-models=0      the opt-out
//!
//! Warmups run one at a time. Firing them all simultaneously would contend for
//! the GPU during the minutes when models are still loading — turning a warmup
//! into a slowdown.

/// Which models to warm, decided purely so it can be tested without a fleet.
///
/// `n` is `--preload-models`, `named` is every `--preload-model`. Both given
/// means both apply: the first N plus the named ones, deduplicated, in
/// declaration order.
pub fn warmup_targets(
    declared_in_order: &[String],
    n: Option<usize>,
    named: &[String],
) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();

    match (n, named.is_empty()) {
        // No preference stated: warm everything. A model you declared is a
        // model you intend to serve, and the first caller should not be the
        // one who pays for that.
        (None, true) => out.extend(declared_in_order.iter().cloned()),
        (n_opt, _) => {
            if let Some(n) = n_opt {
                out.extend(declared_in_order.iter().take(n).cloned());
            }
            for name in named {
                // Named models must actually be declared — warming a model the
                // budget never admitted would just be a confusing 404 in the
                // log. The caller reports unknown names loudly instead.
                if declared_in_order.iter().any(|d| d == name) && !out.contains(name) {
                    out.push(name.clone());
                }
            }
        }
    }
    out
}

/// Names asked for that are not in the fleet — reported, never ignored.
/// A typo in --preload-model that silently warms nothing is a cold model
/// discovered by the first user of the day.
pub fn unknown_targets(declared: &[String], named: &[String]) -> Vec<String> {
    named
        .iter()
        .filter(|n| !declared.iter().any(|d| &d == n))
        .cloned()
        .collect()
}

/// The one-token request that makes a resident model hot.
///
/// llama-server's native `/completion`, not the OpenAI path, because this goes
/// straight to the model process (the gateway would just add a hop) and
/// `n_predict: 1` is the smallest legal generation. `cache_prompt: false` so
/// the warmup does not squat in the prompt cache the first real request wants.
pub fn warmup_request_body() -> &'static str {
    r#"{"prompt":"hi","n_predict":1,"cache_prompt":false}"#
}

/// Did the runtime actually generate? A warmup that 404s or errors must count
/// as a FAILED warmup — "I sent a request" is not "the model is hot".
pub fn warmup_succeeded(status: Option<u16>) -> bool {
    matches!(status, Some(s) if (200..300).contains(&s))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fleet() -> Vec<String> {
        vec![
            "muse".to_string(),
            "deepseek-r1:32b".to_string(),
            "gemma4:27b".to_string(),
        ]
    }

    #[test]
    fn no_preference_warms_everything_in_declaration_order() {
        assert_eq!(warmup_targets(&fleet(), None, &[]), fleet());
    }

    #[test]
    fn preload_models_n_takes_the_first_n_because_declaration_order_is_priority() {
        assert_eq!(
            warmup_targets(&fleet(), Some(2), &[]),
            vec!["muse", "deepseek-r1:32b"]
        );
    }

    #[test]
    fn zero_is_the_opt_out() {
        assert!(warmup_targets(&fleet(), Some(0), &[]).is_empty());
    }

    #[test]
    fn n_larger_than_the_fleet_is_just_everything() {
        assert_eq!(warmup_targets(&fleet(), Some(99), &[]), fleet());
    }

    #[test]
    fn named_models_warm_exactly_those() {
        assert_eq!(
            warmup_targets(&fleet(), None, &["gemma4:27b".into()]),
            vec!["gemma4:27b"]
        );
    }

    #[test]
    fn n_and_names_combine_without_duplicates() {
        // --preload-models=1 --preload-model=muse: muse is both the first N
        // and named. Warming it twice would be a pointless second generation.
        assert_eq!(
            warmup_targets(&fleet(), Some(1), &["muse".into(), "gemma4:27b".into()]),
            vec!["muse", "gemma4:27b"]
        );
    }

    #[test]
    fn a_name_not_in_the_fleet_is_not_warmed_and_is_reported() {
        // The typo case: --preload-model=musee. Silently warming nothing is a
        // cold model discovered by the first user of the day.
        let named = vec!["musee".to_string(), "muse".to_string()];
        assert_eq!(warmup_targets(&fleet(), None, &named), vec!["muse"]);
        assert_eq!(unknown_targets(&fleet(), &named), vec!["musee"]);
    }

    #[test]
    fn the_warmup_request_is_one_token_and_does_not_pollute_the_prompt_cache() {
        let body: serde_json::Value = serde_json::from_str(warmup_request_body()).unwrap();
        assert_eq!(body["n_predict"], 1, "smallest legal generation");
        assert_eq!(
            body["cache_prompt"], false,
            "the warmup must not squat in the cache the first real request wants"
        );
    }

    #[test]
    fn only_a_2xx_counts_as_warm() {
        assert!(warmup_succeeded(Some(200)));
        assert!(!warmup_succeeded(Some(503)), "still loading is not warm");
        assert!(!warmup_succeeded(Some(404)));
        assert!(!warmup_succeeded(None), "no answer is not warm");
    }
}
