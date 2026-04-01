mod gpu;
mod model;
use std::time::Instant;

const GGUF_PATH: &str = "/home/raz/.lmstudio/models/lmstudio-community/Phi-4-mini-reasoning-GGUF/Phi-4-mini-reasoning-Q4_K_M.gguf";
const SHADER_DIR: &str = "/home/raz/projects/torch-vulkan/csrc/shaders";

fn main() {
    println!("=== RRR — Phi-4 Mini Native K-Quant ===\n");

    let t0 = Instant::now();

    let gguf = model::GGUFModel::load(GGUF_PATH);
    eprintln!("[rrr] GGUF: v{}, {} tensors, {:.1}s", gguf.version, gguf.n_tensors, t0.elapsed().as_secs_f32());

    let tokenizer = model::BPETokenizer::from_gguf(&gguf);

    // Simple prompt (matching llama.cpp test)
    // Single token test — compare with llama.cpp
    let prompt_tokens = vec![976u32]; // "The"
    eprintln!("[rrr] Single token test: {:?}", prompt_tokens);

    eprintln!("[rrr] token 976 = \"{}\"", tokenizer.vocab.get(976).map(|s| s.as_str()).unwrap_or("?"));

    unsafe {
        let mut engine = gpu::ComputeEngine::new(SHADER_DIR);
        let phi4 = model::Phi4Model::load_from_gguf(&gguf, &mut engine);

        // Print first 5 embedding values for token 976
        let emb = phi4.embed(976);
        eprintln!("[rrr] embed(976)[:5] = {:?}", &emb[..5]);
        let emb_norm: f32 = emb.iter().map(|x| x*x).sum::<f32>().sqrt();
        eprintln!("[rrr] embed(976) norm = {:.4}", emb_norm);

        eprintln!("\n[rrr] Load: {:.1}s", t0.elapsed().as_secs_f32());

        // Run with chat template
        phi4.generate(&engine, &prompt_tokens, 20, Some(&tokenizer.vocab));
    }
}
