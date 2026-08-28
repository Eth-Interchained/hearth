//! Run the pure resolvers against payloads captured from the real registries.
//! Written after a binding shipped broken because it was never executed.
use hearth_resolve::plan::RepoFile;
use hearth_resolve::{
    pick_gguf, plan_from_hf_files, plan_from_ollama_manifest, HfSource, QuantChoice, Reference,
};

fn gib(b: u64) -> f64 {
    b as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn main() {
    // ---- real registry.ollama.ai manifest for library/llama3:latest -------
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("/tmp/ollama_manifest.json").unwrap())
            .unwrap();
    let r = Reference::parse("llama3").unwrap();
    let (ns, nm) = match &r {
        Reference::Ollama {
            namespace, name, ..
        } => (namespace.clone(), name.clone()),
        _ => unreachable!(),
    };
    let p = plan_from_ollama_manifest(
        &manifest,
        "https://registry.ollama.ai",
        &ns,
        &nm,
        r.key(),
        r.display_name(),
    )
    .unwrap();
    println!("OLLAMA  {}", p.display_name);
    println!("  key      {}", p.key);
    println!("  blobs    {} (filtered from 4 layers)", p.blobs.len());
    println!("  size     {:.2} GiB", gib(p.total_bytes().unwrap()));
    println!("  digest   {}", p.blobs[0].digest.as_deref().unwrap());
    println!("  url      {}", p.blobs[0].url);

    // ---- real huggingface listing ----------------------------------------
    let hf: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string("/tmp/hf.json").unwrap()).unwrap();
    let files: Vec<RepoFile> = hf["siblings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| RepoFile {
            path: s["rfilename"].as_str().unwrap().to_string(),
            size_bytes: s.get("size").and_then(|v| v.as_u64()),
        })
        .filter(|f| f.path.ends_with(".gguf"))
        .collect();
    println!("\nHUGGINGFACE  {} gguf files in the repo", files.len());

    let hr = Reference::parse("hf:TheBloke/Llama-2-7B-GGUF").unwrap();
    let (picked, quant, chose) = pick_gguf(&files, None).unwrap();
    let plan = plan_from_hf_files(
        HfSource {
            owner: "TheBloke",
            repo: "Llama-2-7B-GGUF",
            revision: "main",
        },
        picked,
        QuantChoice {
            quant: quant.clone(),
            chosen_for_you: chose,
        },
        hr.key(),
        hr.display_name(),
    );
    println!(
        "  auto     {} ({:.2} GiB){}",
        quant,
        gib(plan.total_bytes().unwrap()),
        if chose { "  <- we chose this" } else { "" }
    );
    println!("  url      {}", plan.blobs[0].url);

    let (pinned, q2, chose2) = pick_gguf(&files, Some("Q8_0")).unwrap();
    println!(
        "  pinned   {} ({:.2} GiB){}",
        q2,
        gib(pinned.iter().filter_map(|f| f.size_bytes).sum::<u64>()),
        if chose2 {
            "  <- we chose"
        } else {
            "  <- caller chose"
        }
    );

    match pick_gguf(&files, Some("Q9_MEGA")) {
        Ok(_) => println!("  BAD: accepted a quant that does not exist"),
        Err(e) => println!("  bogus quant -> {}", &e.0[..e.0.len().min(96)]),
    }
}
