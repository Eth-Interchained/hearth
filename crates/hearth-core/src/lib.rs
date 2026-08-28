//! hearth — deterministic model residency.
//!
//! Keep declared models warm, and tell the truth about which ones are.
//!
//! Not an inference engine. llama.cpp and vLLM have spent years on kernels,
//! samplers and tokenizers, and none of that is the problem. The problem is
//! that no serving stack will promise a model stays loaded, and none of them
//! can say why one stopped being. hearth is the supervisor above the runtime:
//! it declares a set of models, refuses to declare more than the card holds,
//! keeps them pinned, and reports residency as a named state with a named
//! reason instead of an eventual timeout.
//!
//! Two states here do not exist anywhere else, and they are the two that cost
//! a night of debugging to tell apart:
//!
//!   * `Lost { reason: Evicted }`     — the runtime dropped it for VRAM.
//!   * `Lost { reason: GpuDetached }` — the host took the card away.
//!
//! One is a capacity problem you own. The other is your provider's, and no
//! configuration you write will fix it. Today both look identical from the
//! outside, which is why both get "fixed" repeatedly and neither goes away.

pub mod budget;
pub mod fleet;
pub mod residency;
pub mod sha256;

pub use budget::{plan, Budget, Declared, Plan, Rejection, GIB};
pub use fleet::{Fleet, Route, Slot};
pub use residency::{LostReason, Millis, Observation, Residency};
