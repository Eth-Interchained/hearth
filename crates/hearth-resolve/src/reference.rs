//! What model did the user mean?
//!
//! hearth aggregates two catalogs that do not agree on anything. Ollama has a
//! curated few hundred models under short names people already have in muscle
//! memory (`llama3`, `qwen3:14b`). HuggingFace has hundreds of thousands of
//! GGUFs under `owner/repo`, several quantizations to a repo, and no opinion
//! about which one you want.
//!
//! The bet is that both should work in the same command, because the catalogs
//! are theirs and the serving is ours. So:
//!
//! ```text
//! llama3                             ollama library/llama3:latest
//! qwen3:14b                          ollama library/qwen3:14b
//! ollama:someone/model:tag           ollama someone/model:tag
//! hf:TheBloke/Llama-2-7B-GGUF        huggingface, quant chosen for you
//! hf:TheBloke/Llama-2-7B-GGUF@Q5_K_M huggingface, quant pinned
//! hf:owner/repo@Q4_K_M#refs/pr/3     ...at a specific revision
//! ./models/muse.gguf                 a file you already have
//! ```
//!
//! A bare name means Ollama on purpose. It is what people already type, and a
//! tool that punishes existing muscle memory to make a point about neutrality
//! is a tool nobody switches to.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Where a model comes from, once we have understood the reference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum Reference {
    Ollama {
        /// `library` for the curated catalog, a username otherwise.
        namespace: String,
        name: String,
        tag: String,
    },
    HuggingFace {
        owner: String,
        repo: String,
        /// Git revision. `main` unless asked otherwise.
        revision: String,
        /// Which quantization, if the caller pinned one. When absent we choose,
        /// and say which one we chose.
        quant: Option<String>,
    },
    /// A GGUF already on disk. No catalog, no download, no opinions.
    Local { path: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError(pub String);

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ParseError {}

fn bad(msg: impl Into<String>) -> ParseError {
    ParseError(msg.into())
}

impl Reference {
    /// Read a reference the way a person typed it.
    pub fn parse(input: &str) -> Result<Reference, ParseError> {
        let s = input.trim();
        if s.is_empty() {
            return Err(bad("empty model reference"));
        }

        // KNOWN schemes first. An explicit scheme is a statement of intent and
        // cannot be ambiguous; the path heuristic below is a guess for input
        // that made no statement.
        //
        // This used to be the other way round, to keep a Windows path like
        // `C:\models\x.gguf` from being read as a scheme. That concern is real
        // but it is already handled by only ever stripping the four schemes we
        // know — `c` is not one of them. Checking the heuristic first had a
        // worse consequence: `looks_like_path` matches anything ending in
        // `.gguf`, so EVERY `file:` URL pointing at a GGUF matched it and kept
        // its scheme, producing `Local { path: "file:/m/muse.gguf" }` — a path
        // that cannot be opened. The `file:` branch below was dead code for
        // every realistic input, and the heuristic meant to recognise GGUF
        // files was what broke GGUF file URLs.
        if let Some(rest) = strip_scheme(s, "hf") {
            return parse_hf(rest);
        }
        if let Some(rest) = strip_scheme(s, "huggingface") {
            return parse_hf(rest);
        }
        if let Some(rest) = strip_scheme(s, "ollama") {
            return parse_ollama(rest);
        }
        if let Some(rest) = strip_scheme(s, "file") {
            if rest.is_empty() {
                return Err(bad("file: needs a path"));
            }
            return Ok(Reference::Local {
                path: rest.to_string(),
            });
        }

        // No scheme. Does it look like a path?
        if looks_like_path(s) {
            return Ok(Reference::Local {
                path: s.to_string(),
            });
        }

        // No scheme. Ollama, because that is what the name looks like to
        // everyone who has ever used one of these tools.
        parse_ollama(s)
    }

    /// A stable identity for this model on disk and in the event log.
    ///
    /// Two references that mean the same model must produce the same key, or
    /// the same weights get downloaded twice and the residency history splits
    /// in half.
    pub fn key(&self) -> String {
        match self {
            Reference::Ollama {
                namespace,
                name,
                tag,
            } => {
                format!("ollama/{namespace}/{name}:{tag}")
            }
            Reference::HuggingFace {
                owner,
                repo,
                revision,
                quant,
            } => {
                let q = quant.as_deref().unwrap_or("auto");
                format!("hf/{owner}/{repo}@{revision}#{q}")
            }
            Reference::Local { path } => format!("local/{path}"),
        }
    }

    /// What to call it in a report, a route, or an API response.
    pub fn display_name(&self) -> String {
        match self {
            Reference::Ollama {
                namespace,
                name,
                tag,
            } if namespace == "library" => {
                format!("{name}:{tag}")
            }
            Reference::Ollama {
                namespace,
                name,
                tag,
            } => format!("{namespace}/{name}:{tag}"),
            Reference::HuggingFace {
                owner, repo, quant, ..
            } => match quant {
                Some(q) => format!("{owner}/{repo}@{q}"),
                None => format!("{owner}/{repo}"),
            },
            Reference::Local { path } => path
                .rsplit('/')
                .next()
                .unwrap_or(path)
                .trim_end_matches(".gguf")
                .to_string(),
        }
    }

    pub fn needs_download(&self) -> bool {
        !matches!(self, Reference::Local { .. })
    }
}

fn strip_scheme<'a>(s: &'a str, scheme: &str) -> Option<&'a str> {
    let lower = s.to_ascii_lowercase();
    let prefix = format!("{scheme}:");
    // `hf://owner/repo` and `hf:owner/repo` both mean the same thing; people
    // type both and neither is wrong.
    if lower.starts_with(&format!("{prefix}//")) {
        Some(&s[prefix.len() + 2..])
    } else if lower.starts_with(&prefix) {
        Some(&s[prefix.len()..])
    } else {
        None
    }
}

fn looks_like_path(s: &str) -> bool {
    s.starts_with('.')
        || s.starts_with('/')
        || s.starts_with('~')
        || s.ends_with(".gguf")
        // C:\models\x.gguf
        || (s.len() > 2 && s.as_bytes()[1] == b':' && (s.contains('\\') || s.contains('/')))
}

fn parse_hf(rest: &str) -> Result<Reference, ParseError> {
    if rest.is_empty() {
        return Err(bad("hf: needs an owner/repo"));
    }
    // Split the optional pieces off the back before touching the path, so a
    // '#' or '@' inside neither can confuse the owner/repo split.
    let (head, revision) = match rest.split_once('#') {
        Some((h, r)) if !r.is_empty() => (h, r.to_string()),
        Some(_) => return Err(bad("empty revision after '#'")),
        None => (rest, "main".to_string()),
    };
    let (path, quant) = match head.split_once('@') {
        Some((p, q)) if !q.is_empty() => (p, Some(q.to_string())),
        Some(_) => return Err(bad("empty quantization after '@'")),
        None => (head, None),
    };

    let mut parts = path.split('/');
    let owner = parts.next().unwrap_or("");
    let repo = parts.next().unwrap_or("");
    if owner.is_empty() || repo.is_empty() || parts.next().is_some() {
        return Err(bad(format!(
            "expected owner/repo, got \"{path}\" — a HuggingFace reference has exactly two parts"
        )));
    }

    Ok(Reference::HuggingFace {
        owner: owner.to_string(),
        repo: repo.to_string(),
        revision,
        quant,
    })
}

fn parse_ollama(rest: &str) -> Result<Reference, ParseError> {
    if rest.is_empty() {
        return Err(bad("ollama: needs a model name"));
    }
    let (path, tag) = match rest.rsplit_once(':') {
        Some((p, t)) if !t.is_empty() && !t.contains('/') => (p, t.to_string()),
        Some((_, "")) => return Err(bad("empty tag after ':'")),
        _ => (rest, "latest".to_string()),
    };

    let mut parts = path.split('/');
    let first = parts.next().unwrap_or("");
    let second = parts.next();
    if parts.next().is_some() {
        return Err(bad(format!(
            "expected name or namespace/name, got \"{path}\""
        )));
    }
    let (namespace, name) = match second {
        // Ollama's curated catalog lives under `library`, which nobody types.
        None => ("library".to_string(), first.to_string()),
        Some(n) => (first.to_string(), n.to_string()),
    };
    if name.is_empty() || namespace.is_empty() {
        return Err(bad(format!("incomplete model name: \"{rest}\"")));
    }

    Ok(Reference::Ollama {
        namespace,
        name,
        tag,
    })
}

#[cfg(test)]
mod file_scheme_tests {
    use super::*;

    /// The bug: `looks_like_path` matches anything ending in `.gguf`, and it
    /// used to be checked before the scheme strips. So every `file:` URL
    /// pointing at a GGUF — which is every realistic one — came back with its
    /// scheme still attached, as a path that cannot be opened.
    #[test]
    fn a_file_url_yields_a_path_you_can_actually_open() {
        for (input, want) in [
            ("file:/models/muse.gguf", "/models/muse.gguf"),
            ("file:./muse.gguf", "./muse.gguf"),
            ("file:muse.gguf", "muse.gguf"),
            // Not a .gguf, which is the one case that used to work — it fell
            // past the heuristic and reached the scheme branch.
            ("file:/models/model.bin", "/models/model.bin"),
        ] {
            match Reference::parse(input).expect(input) {
                Reference::Local { path } => assert_eq!(path, want, "for {input}"),
                other => panic!("{input} should be Local, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_bare_path_still_works_without_a_scheme() {
        for input in ["/models/muse.gguf", "./muse.gguf", "~/models/muse.gguf"] {
            match Reference::parse(input).expect(input) {
                Reference::Local { path } => assert_eq!(path, input),
                other => panic!("{input} should be Local, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_windows_path_is_not_mistaken_for_a_scheme() {
        // The reason the heuristic was first. It is safe under the new order
        // because only four known schemes are ever stripped, and `c` is not
        // one of them.
        for input in [r"C:\models\muse.gguf", "C:/models/muse.gguf"] {
            match Reference::parse(input).expect(input) {
                Reference::Local { path } => assert_eq!(path, input),
                other => panic!("{input} should be Local, got {other:?}"),
            }
        }
    }

    #[test]
    fn the_other_schemes_still_win_over_the_path_heuristic() {
        // `hf:owner/repo` ending in something path-ish must remain a HF
        // reference, not become a local file.
        assert!(matches!(
            Reference::parse("hf:TheBloke/X-GGUF").unwrap(),
            Reference::HuggingFace { .. }
        ));
        assert!(matches!(
            Reference::parse("ollama:library/llama3:latest").unwrap(),
            Reference::Ollama { .. }
        ));
    }

    #[test]
    fn an_empty_file_scheme_is_an_error_not_an_empty_path() {
        // `Local { path: "" }` would fail much later as "no such file: ''".
        assert!(Reference::parse("file:").is_err());
    }

    #[test]
    fn display_name_is_the_model_name_now_that_the_scheme_is_gone() {
        // Basename with the extension trimmed: a model is called "muse", not
        // "muse.gguf" and certainly not "file:/models/muse.gguf". Before the
        // reorder this returned the whole scheme-prefixed string, which then
        // became the model's key in the spine.
        let r = Reference::parse("file:/models/muse.gguf").unwrap();
        assert_eq!(r.display_name(), "muse");
        assert_eq!(r.key(), "local//models/muse.gguf");
    }
}
