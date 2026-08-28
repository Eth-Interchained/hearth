//! What to do with a request. Pure, and the whole product lives here.
//!
//! Every serving stack in front of a model can proxy. What none of them do is
//! answer honestly when they *can't*: a model that is still loading, a model
//! the runtime evicted, a model whose GPU the host reclaimed, and a model that
//! will never fit on this card all arrive at a caller as the same failed
//! request. hearth already knows which one it is — [`hearth_core::fleet`] draws
//! that distinction — and this module is where that knowledge becomes an HTTP
//! status a router can act on without a human reading logs.
//!
//! The distinction that matters most is **retryable vs not**:
//!
//!   * `Warming` is 503 with `Retry-After`. Come back; it is coming.
//!   * `Lost` is 503. Try elsewhere, and the body says whose fault it was.
//!   * `NotAdmitted` is **409, not 503** — it does not fit on this card and it
//!     never will. A 503 here would invite a router to retry forever against a
//!     box that is arithmetically incapable of serving it.
//!
//! Getting that last one wrong is a retry storm that looks like a slow node.

use hearth_core::fleet::{Fleet, Route};
use hearth_core::residency::{LostReason, Millis};

use serde_json::{json, Value};

/// What the gateway decided to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Forward the request body verbatim to a resident model.
    Proxy { model: String, endpoint: String },
    /// Answer it ourselves.
    Answer(Response),
}

/// A response we generate rather than proxy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    pub status: u16,
    pub body: String,
    /// Seconds. Set only when coming back later could plausibly work.
    pub retry_after: Option<u32>,
}

impl Response {
    pub fn json(status: u16, body: Value) -> Response {
        Response {
            status,
            body: serde_json::to_string_pretty(&body).unwrap_or_else(|_| "{}".into()),
            retry_after: None,
        }
    }

    pub fn retry_in(mut self, secs: u32) -> Response {
        self.retry_after = Some(secs);
        self
    }

    /// An OpenAI-shaped error, so existing clients surface the message instead
    /// of choking on a body they cannot parse. The extra fields are additive —
    /// a client that ignores them still gets a sensible error, and one that
    /// reads them gets the diagnosis.
    pub fn openai_error(
        status: u16,
        message: impl Into<String>,
        kind: &str,
        extra: Value,
    ) -> Response {
        let mut err = json!({
            "message": message.into(),
            "type": kind,
            "param": Value::Null,
            "code": kind,
        });
        if let (Some(e), Some(x)) = (err.as_object_mut(), extra.as_object()) {
            for (k, v) in x {
                e.insert(k.clone(), v.clone());
            }
        }
        Response::json(status, json!({ "error": err }))
    }
}

/// Which endpoint a path is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endpoint {
    /// Anything that names a model in its body and should reach the runtime.
    Inference,
    Models,
    /// Ollama's native listing. Same facts as `Models`, DIFFERENT dialect —
    /// an ollama-mode client deserializes `{"models":[…]}` and chokes on
    /// OpenAI's `{"data":[…]}`. Found in production: pin-clientd polled
    /// /api/tags on day one of the cutover and logged
    /// "error decoding response body" against a perfectly healthy fleet.
    ModelsOllama,
    Residency,
    Health,
    Unknown,
}

/// Classify a path. Query strings are stripped, and a trailing slash does not
/// make a different endpoint — clients add both.
pub fn classify(path: &str) -> Endpoint {
    let p = path.split('?').next().unwrap_or("").trim_end_matches('/');
    match p {
        "/v1/chat/completions"
        | "/v1/completions"
        | "/v1/embeddings"
        | "/api/chat"
        | "/api/generate" => Endpoint::Inference,
        "/v1/models" => Endpoint::Models,
        "/api/tags" => Endpoint::ModelsOllama,
        "/residency" | "/v1/residency" => Endpoint::Residency,
        "/health" | "/healthz" | "/v1/health" => Endpoint::Health,
        _ => Endpoint::Unknown,
    }
}

/// Pull the model name out of a request body.
///
/// Tolerant on purpose: this is the field every OpenAI-compatible client
/// already sends, and rejecting a request because the JSON had an unexpected
/// extra key would make hearth stricter than the API it is imitating.
pub fn model_of(body: &str) -> Option<String> {
    let v: Value = serde_json::from_str(body).ok()?;
    let m = v.get("model")?.as_str()?.trim();
    if m.is_empty() {
        None
    } else {
        Some(m.to_string())
    }
}

/// The decision. This is the function the product is.
pub fn decide(path: &str, body: &str, fleet: &Fleet, now: Millis) -> Decision {
    match classify(path) {
        Endpoint::Health => Decision::Answer(Response::json(
            200,
            json!({ "status": "ok", "service": "hearth" }),
        )),
        Endpoint::Models => Decision::Answer(Response::json(200, models_body(fleet, now))),
        Endpoint::ModelsOllama => {
            Decision::Answer(Response::json(200, ollama_tags_body(fleet, now)))
        }
        Endpoint::Residency => Decision::Answer(Response::json(200, residency_body(fleet, now))),
        Endpoint::Unknown => Decision::Answer(Response::openai_error(
            404,
            format!("no route for {path}"),
            "not_found",
            json!({}),
        )),
        Endpoint::Inference => {
            let Some(model) = model_of(body) else {
                return Decision::Answer(Response::openai_error(
                    400,
                    "request has no \"model\" field — hearth routes by model name, \
                     so it cannot guess which one you meant",
                    "invalid_request_error",
                    json!({}),
                ));
            };
            route_to_decision(&model, fleet.route(&model, now))
        }
    }
}

/// Turn the fleet's answer into an HTTP answer. Separated so the mapping —
/// which is the part with the judgement in it — is testable on its own.
pub fn route_to_decision(model: &str, route: Route) -> Decision {
    match route {
        Route::Ready { endpoint, .. } => Decision::Proxy {
            model: model.to_string(),
            endpoint,
        },

        // Coming up. NOT a failure, and the caller is told how long it has
        // been so it can decide for itself whether to wait — hearth never
        // makes that call on someone else's behalf.
        Route::Warming { for_ms, .. } => Decision::Answer(
            Response::openai_error(
                503,
                format!(
                    "{model} is loading ({}s so far) — this is progress, not a failure",
                    for_ms / 1000
                ),
                "model_warming",
                json!({
                    "state": "warming",
                    "loading_for_ms": for_ms,
                    "retryable": true,
                    "operator_fault": false,
                }),
            )
            // Short, because a load that is already underway usually finishes
            // in seconds-to-minutes and a long backoff wastes a warm model.
            .retry_in(5),
        ),

        // Gone. The body carries WHOSE fault, because that is the number that
        // follows an operator around and nothing else in the ecosystem reports
        // it.
        Route::Lost {
            reason,
            operator_fault,
            ..
        } => Decision::Answer(
            Response::openai_error(
                503,
                // Keyed off the REASON, not off the fault boolean. Saying
                // "over-committed" for a process that segfaulted sends an
                // operator to look at VRAM headroom that was never the
                // problem — a wrong diagnosis is worse than a vague one.
                match reason {
                    LostReason::GpuDetached => format!(
                        "{model} lost its GPU — the host reclaimed the card; this is not the operator's doing"
                    ),
                    LostReason::Evicted => format!(
                        "{model} was evicted by the runtime to free VRAM — this node is over-committed"
                    ),
                    LostReason::ProcessExited => format!(
                        "{model}'s serving process exited — check the runtime log, not the VRAM budget"
                    ),
                    LostReason::Unhealthy => format!(
                        "{model} stopped answering health checks while its GPU was still present"
                    ),
                },
                "model_lost",
                json!({
                    "state": "lost",
                    "reason": format!("{reason:?}").to_lowercase(),
                    "retryable": true,
                    "operator_fault": operator_fault,
                }),
            )
            .retry_in(30),
        ),

        // It will not load, and we know why.
        Route::Failed { reason, .. } => Decision::Answer(Response::openai_error(
            503,
            format!("{model} failed to load: {reason}"),
            "model_failed",
            json!({ "state": "failed", "retryable": false, "operator_fault": true }),
        )),

        // THE one that must not be a 503. It does not fit on this card and it
        // never will; a retryable status here is a router hammering a box that
        // is arithmetically incapable of serving the request.
        Route::NotAdmitted { short_bytes, .. } => Decision::Answer(Response::openai_error(
            409,
            format!(
                "{model} does not fit on this host — short by {:.1} GiB. This is \
                 permanent until the declaration or the hardware changes, so do \
                 not retry here.",
                hearth_core::budget::gib(short_bytes)
            ),
            "model_not_admitted",
            json!({
                "state": "not_admitted",
                "short_bytes": short_bytes,
                "retryable": false,
                "operator_fault": false,
            }),
        )),

        Route::NotDeclared { .. } => Decision::Answer(Response::openai_error(
            404,
            format!("{model} is not declared on this host"),
            "model_not_found",
            json!({ "state": "not_declared", "retryable": false }),
        )),

        // Declared, admitted, never observed. An honest "we do not know yet"
        // beats a confident wrong answer in either direction.
        Route::Unknown { .. } => Decision::Answer(
            Response::openai_error(
                503,
                format!("{model} is declared but has not been observed yet"),
                "model_unknown",
                json!({ "state": "unknown", "retryable": true, "operator_fault": false }),
            )
            .retry_in(2),
        ),
    }
}

/// `/v1/models`, in OpenAI's shape so existing clients can list.
fn models_body(fleet: &Fleet, now: Millis) -> Value {
    let data: Vec<Value> = fleet
        .slots()
        .iter()
        .map(|s| {
            json!({
                "id": s.declared.model,
                "object": "model",
                "owned_by": "hearth",
                // Additive: a plain OpenAI client ignores these, and anything
                // that reads them learns whether the model can serve right now.
                "hearth": {
                    "ready": s.state.is_ready(),
                    "state": s.state.explain(now),
                    "admitted": s.admitted,
                },
            })
        })
        .collect();
    json!({ "object": "list", "data": data })
}

/// `/api/tags`, in OLLAMA's shape, because that is the contract of the path.
///
/// An ollama-mode client (pin-clientd, the ollama CLI, anything built on the
/// ollama SDK) deserializes exactly this structure; answering with OpenAI's
/// listing on ollama's path is a parse error on every poll. Only admitted
/// models are listed — ollama's semantics are "models you can run", and a
/// budget-refused model is not that. /v1/models and /residency still show
/// everything, refusals included.
fn ollama_tags_body(fleet: &Fleet, now: Millis) -> Value {
    let models: Vec<Value> = fleet
        .slots()
        .iter()
        .filter(|s| s.admitted)
        .map(|s| {
            json!({
                "name": s.declared.model,
                "model": s.declared.model,
                // The moment of listing. hearth's real history lives in the
                // spine; this field exists because the dialect requires it.
                "modified_at": "1970-01-01T00:00:00Z",
                "size": s.declared.total_bytes(),
                "digest": "",
                "details": {
                    "format": "gguf",
                    "family": "",
                    "parameter_size": "",
                    "quantization_level": "",
                },
                // Additive, same as everywhere else: ollama clients ignore it,
                // and anything smarter learns whether the model can serve NOW.
                "hearth": {
                    "ready": s.state.is_ready(),
                    "state": s.state.explain(now),
                },
            })
        })
        .collect();
    json!({ "models": models })
}

/// `/residency` — the truth the OpenAI shape has no way to express.
fn residency_body(fleet: &Fleet, now: Millis) -> Value {
    let models: Vec<Value> = fleet
        .slots()
        .iter()
        .map(|s| {
            let route = fleet.route(&s.declared.model, now);
            json!({
                "model": s.declared.model,
                "state": s.state.explain(now),
                "ready": s.state.is_ready(),
                "admitted": s.admitted,
                "declared_bytes": s.declared.total_bytes(),
                "short_bytes": s.short_bytes,
                "endpoint": s.endpoint,
                "operator_fault": route.operator_fault(),
                "try_elsewhere": route.should_try_elsewhere(),
            })
        })
        .collect();
    json!({
        "committed_bytes": fleet.live_committed_bytes(),
        "free_bytes": fleet.live_free_bytes(),
        "report": fleet.report(now),
        "models": models,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hearth_core::budget::{Budget, Declared, GIB};
    use hearth_core::residency::Observation;

    fn fleet() -> Fleet {
        Fleet::declare(
            Budget::with_reserve_pct(48 * GIB, 8),
            vec![
                Declared {
                    model: "muse".into(),
                    weights_bytes: 20 * GIB,
                    kv_bytes: GIB,
                },
                Declared {
                    model: "whale".into(),
                    weights_bytes: 400 * GIB,
                    kv_bytes: 0,
                },
            ],
        )
    }

    fn answer(d: Decision) -> Response {
        match d {
            Decision::Answer(r) => r,
            Decision::Proxy { .. } => panic!("expected an answer, got a proxy"),
        }
    }

    fn facts(r: &Response) -> Value {
        serde_json::from_str::<Value>(&r.body).unwrap()["error"].clone()
    }

    // -- the retryable / not-retryable line ---------------------------------

    #[test]
    fn a_model_that_will_never_fit_is_not_a_503() {
        // The whole point. 503 means "try again"; this model is arithmetically
        // incapable of running here, and a retryable status turns that into a
        // router hammering the box forever.
        let r = answer(route_to_decision("whale", fleet().route("whale", 1_000)));
        assert_eq!(r.status, 409, "permanent, so not a retryable status");
        assert_eq!(r.retry_after, None, "and no Retry-After to invite one");
        assert_eq!(facts(&r)["retryable"], json!(false));
        assert!(facts(&r)["short_bytes"].as_u64().unwrap() > 0);
        assert!(r.body.contains("do not retry"), "{}", r.body);
    }

    #[test]
    fn warming_is_a_503_that_says_come_back_soon() {
        let mut f = fleet();
        f.observe("muse", &Observation::LoadStarted, 1_000);
        let r = answer(route_to_decision("muse", f.route("muse", 21_000)));
        assert_eq!(r.status, 503);
        assert_eq!(r.retry_after, Some(5), "short — a load in flight finishes");
        assert_eq!(facts(&r)["retryable"], json!(true));
        assert_eq!(facts(&r)["loading_for_ms"], json!(20_000));
        assert_eq!(
            facts(&r)["operator_fault"],
            json!(false),
            "still loading is nobody's fault"
        );
        assert!(r.body.contains("progress, not a failure"));
    }

    // -- the one boolean, over HTTP -----------------------------------------

    #[test]
    fn a_detached_gpu_says_in_the_body_that_it_is_not_the_operators_fault() {
        let mut f = fleet();
        f.observe("muse", &Observation::LoadStarted, 0);
        f.observe(
            "muse",
            &Observation::ProbeOk {
                vram_bytes: 21 * GIB,
            },
            1_000,
        );
        f.observe(
            "muse",
            &Observation::ProbeFailed {
                gpu_present: false,
                detail: "no CUDA device".into(),
            },
            2_000,
        );
        let r = answer(route_to_decision("muse", f.route("muse", 2_000)));
        assert_eq!(r.status, 503);
        assert_eq!(facts(&r)["operator_fault"], json!(false));
        assert_eq!(facts(&r)["reason"], json!("gpudetached"));
        assert!(r.body.contains("not the operator's doing"), "{}", r.body);
    }

    #[test]
    fn an_eviction_says_the_opposite_at_the_same_status() {
        // Same HTTP status, opposite diagnosis. A router that only reads the
        // status treats them the same, which is correct — both mean "go
        // elsewhere". One that reads the body learns who to score down.
        let mut f = fleet();
        f.observe("muse", &Observation::LoadStarted, 0);
        f.observe(
            "muse",
            &Observation::ProbeOk {
                vram_bytes: 21 * GIB,
            },
            1_000,
        );
        f.observe(
            "muse",
            &Observation::ProbeFailed {
                gpu_present: true,
                detail: "not loaded".into(),
            },
            2_000,
        );
        let r = answer(route_to_decision("muse", f.route("muse", 2_000)));
        assert_eq!(r.status, 503);
        assert_eq!(facts(&r)["operator_fault"], json!(true));
        assert!(r.body.contains("over-committed"), "{}", r.body);
    }

    #[test]
    fn each_loss_reason_gets_its_own_diagnosis_not_a_shared_one() {
        // Found by running it: a process that exited was being told "this node
        // is over-committed", which is the Evicted message. That sends an
        // operator to look at VRAM headroom that was never the problem.
        let cases = [
            (LostReason::GpuDetached, "reclaimed the card", false),
            (LostReason::Evicted, "over-committed", true),
            (LostReason::ProcessExited, "process exited", true),
            (LostReason::Unhealthy, "health checks", true),
        ];
        for (reason, phrase, fault) in cases {
            let r = answer(route_to_decision(
                "muse",
                Route::Lost {
                    model: "muse".into(),
                    reason,
                    operator_fault: reason.is_operator_fault(),
                },
            ));
            assert_eq!(r.status, 503);
            assert!(
                r.body.contains(phrase),
                "{reason:?} should say {phrase:?}, said: {}",
                r.body
            );
            assert_eq!(facts(&r)["operator_fault"], json!(fault), "{reason:?}");
        }
    }

    #[test]
    fn a_ready_model_is_proxied_not_answered() {
        let mut f = fleet();
        f.set_endpoint("muse", "127.0.0.1:8090");
        f.observe("muse", &Observation::LoadStarted, 0);
        f.observe(
            "muse",
            &Observation::ProbeOk {
                vram_bytes: 21 * GIB,
            },
            1_000,
        );
        assert_eq!(
            route_to_decision("muse", f.route("muse", 1_000)),
            Decision::Proxy {
                model: "muse".into(),
                endpoint: "127.0.0.1:8090".into()
            }
        );
    }

    #[test]
    fn an_undeclared_model_is_a_404_not_a_503() {
        // It is not coming. Retrying will not make this host declare it.
        let r = answer(route_to_decision("nope", fleet().route("nope", 0)));
        assert_eq!(r.status, 404);
        assert_eq!(facts(&r)["retryable"], json!(false));
    }

    #[test]
    fn never_observed_is_an_honest_short_retry() {
        let r = answer(route_to_decision("muse", fleet().route("muse", 0)));
        assert_eq!(r.status, 503);
        assert_eq!(r.retry_after, Some(2));
        assert_eq!(facts(&r)["state"], json!("unknown"));
    }

    // -- routing and parsing ------------------------------------------------

    #[test]
    fn the_paths_clients_actually_use_are_recognised() {
        for p in [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
            "/api/chat",
            "/api/generate",
        ] {
            assert_eq!(classify(p), Endpoint::Inference, "{p}");
        }
        assert_eq!(classify("/v1/models"), Endpoint::Models);
        // Ollama's own listing path, so a tool pointed at hearth instead of
        // ollama does not have to be rewritten.
        assert_eq!(classify("/api/tags"), Endpoint::ModelsOllama);
        assert_eq!(classify("/residency"), Endpoint::Residency);
        assert_eq!(classify("/health"), Endpoint::Health);
        assert_eq!(classify("/nope"), Endpoint::Unknown);
    }

    #[test]
    fn a_trailing_slash_or_query_is_the_same_endpoint() {
        // Clients add both, and a 404 for a trailing slash is a support ticket.
        assert_eq!(classify("/v1/models/"), Endpoint::Models);
        assert_eq!(
            classify("/v1/chat/completions?stream=true"),
            Endpoint::Inference
        );
    }

    #[test]
    fn the_model_comes_out_of_the_body() {
        assert_eq!(
            model_of(r#"{"model":"muse","messages":[]}"#),
            Some("muse".into())
        );
        assert_eq!(model_of(r#"{"model":"  muse  "}"#), Some("muse".into()));
        assert_eq!(model_of(r#"{"model":""}"#), None);
        assert_eq!(model_of(r#"{"messages":[]}"#), None);
        assert_eq!(model_of("not json"), None);
        assert_eq!(model_of(""), None);
    }

    #[test]
    fn a_request_with_no_model_says_so_rather_than_guessing() {
        // Picking the only resident model would work right up until there are
        // two, and then it would silently answer from the wrong one.
        let r = answer(decide(
            "/v1/chat/completions",
            r#"{"messages":[]}"#,
            &fleet(),
            0,
        ));
        assert_eq!(r.status, 400);
        assert!(r.body.contains("cannot guess"), "{}", r.body);
    }

    #[test]
    fn an_unknown_path_is_a_404_in_the_shape_clients_parse() {
        let r = answer(decide("/nope", "", &fleet(), 0));
        assert_eq!(r.status, 404);
        // OpenAI-shaped, so a client surfaces the message instead of choking.
        assert!(facts(&r)["message"].as_str().unwrap().contains("/nope"));
    }

    // -- the informational endpoints ----------------------------------------

    #[test]
    fn models_lists_everything_declared_including_what_does_not_fit() {
        let r = answer(decide("/v1/models", "", &fleet(), 0));
        assert_eq!(r.status, 200);
        let v: Value = serde_json::from_str(&r.body).unwrap();
        let ids: Vec<&str> = v["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["muse", "whale"]);
        // The refused one is listed and marked, not hidden. "It used to have
        // that model" is the first thing anyone says when a box gets slower.
        let whale = &v["data"][1];
        assert_eq!(whale["hearth"]["admitted"], json!(false));
        assert_eq!(whale["hearth"]["ready"], json!(false));
    }

    #[test]
    fn api_tags_speaks_ollama_not_openai() {
        // The production bug from cutover day one: pin-clientd in ollama mode
        // polled /api/tags, got OpenAI's {"data":[…]}, and logged "error
        // decoding response body" against a healthy fleet. The path IS the
        // contract: ollama's path gets ollama's dialect.
        let r = answer(decide("/api/tags", "", &fleet(), 0));
        assert_eq!(r.status, 200);
        let v: Value = serde_json::from_str(&r.body).unwrap();
        assert!(v.get("models").is_some(), "ollama's key: {}", r.body);
        assert!(v.get("data").is_none(), "not OpenAI's key");
        let m = &v["models"][0];
        for key in ["name", "model", "modified_at", "size", "digest", "details"] {
            assert!(m.get(key).is_some(), "ollama clients deserialize {key}");
        }
        assert_eq!(m["name"], json!("muse"));
        assert!(m["size"].as_u64().unwrap() > 0);
        // Only admitted models: ollama semantics are "models you can run",
        // and whale was refused by the budget.
        assert_eq!(v["models"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn residency_carries_what_the_openai_shape_cannot() {
        let mut f = fleet();
        f.observe("muse", &Observation::LoadStarted, 1_000);
        let r = answer(decide("/residency", "", &f, 5_000));
        let v: Value = serde_json::from_str(&r.body).unwrap();
        assert!(v["report"].as_str().unwrap().contains("muse"));
        let muse = &v["models"][0];
        assert_eq!(muse["ready"], json!(false));
        assert_eq!(muse["operator_fault"], json!(false));
        assert!(muse["state"].as_str().unwrap().contains("loading"));
        let whale = &v["models"][1];
        assert!(
            whale["short_bytes"].as_u64().unwrap() > 0,
            "the shortfall is the actionable number"
        );
    }

    #[test]
    fn health_is_about_the_gateway_not_the_models() {
        // A gateway that reports unhealthy because a model is loading would be
        // taken out of a load balancer for doing its job correctly.
        let r = answer(decide("/health", "", &fleet(), 0));
        assert_eq!(r.status, 200);
    }
}
