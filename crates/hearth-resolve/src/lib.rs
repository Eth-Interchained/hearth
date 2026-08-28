//! Resolve a model reference to fetchable bytes.
//!
//! hearth aggregates catalogs it does not own: Ollama's curated few hundred,
//! and HuggingFace's hundreds of thousands of GGUFs. Neither is the product —
//! the serving is. This crate is the seam between "what a person typed" and
//! "these exact bytes, with this digest".
//!
//! Everything here is pure. Reference parsing, layer selection, quantization
//! choice and multi-part detection are all rules, and rules are testable
//! without a network. The HTTP that actually moves bytes lives elsewhere and
//! contains no decisions.
//!
//! ```
//! use hearth_resolve::Reference;
//! assert_eq!(
//!     Reference::parse("llama3").unwrap().key(),
//!     "ollama/library/llama3:latest",
//! );
//! ```

pub mod plan;
pub mod reference;

pub use plan::{
    available_quants, pick_gguf, plan_from_hf_files, plan_from_ollama_manifest, quant_of,
    total_parts, Blob, FetchPlan, HfSource, QuantChoice, RepoFile, ResolveError,
    OLLAMA_MODEL_MEDIA_TYPE, QUANT_PREFERENCE,
};
pub use reference::{ParseError, Reference};
