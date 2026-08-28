//! What actually fits on one RTX A6000, given the roster that was registered
//! when PIN stopped answering.
use hearth_core::budget::gib;
use hearth_core::{plan, Budget, Declared, GIB};

fn m(model: &str, w: u64, kv: u64) -> Declared {
    Declared { model: model.into(), weights_bytes: w, kv_bytes: kv }
}

fn main() {
    let card = Budget::with_reserve_pct(48 * GIB, 8);
    println!("RTX A6000 — {:.0} GiB total, {:.1} GiB reserved, {:.1} GiB usable\n",
             gib(card.total_bytes), gib(card.reserve_bytes), gib(card.usable_bytes()));

    let roster = vec![
        m("muse-local:latest", 20 * GIB, GIB),
        m("deepseek-r1:32b",   20 * GIB, GIB),
        m("gemma4:26b",        16 * GIB, GIB),
        m("qwen3.6:27b",       17 * GIB, GIB),
        m("gemma4-extract:31b",19 * GIB, GIB),
    ];
    let p = plan(card, &roster);
    println!("{}", p.explain());
    println!("\nheadroom: {:.1} GiB", gib(p.headroom_bytes()));
}
