mod gpu;
mod model;
use std::time::Instant;

const GGUF_PATH: &str = "/home/raz/.lmstudio/models/lmstudio-community/Phi-4-mini-reasoning-GGUF/Phi-4-mini-reasoning-Q4_K_M.gguf";
const SHADER_DIR: &str = "/home/raz/projects/torch-vulkan/csrc/shaders";

fn main() {
    println!("=== RRR — Phi-4 Mini Inference Engine ===\n");

    let t0 = Instant::now();

    // Load GGUF
    eprintln!("[rrr] Loading GGUF from {}...", GGUF_PATH);
    let gguf = model::GGUFModel::load(GGUF_PATH);
    eprintln!("[rrr] GGUF loaded: v{}, {} tensors, {:.1}s",
        gguf.version, gguf.n_tensors, t0.elapsed().as_secs_f32());

    // Print key metadata
    if let Some(name) = gguf.metadata.get("general.name") {
        eprintln!("[rrr] Model: {:?}", name);
    }

    unsafe {
        // Init Vulkan engine
        let mut engine = gpu::ComputeEngine::new(SHADER_DIR);

        // Load model weights (dequantize to F32)
        let phi4 = model::Phi4Model::load_from_gguf(&gguf, &mut engine);

        eprintln!("\n[rrr] Total load time: {:.1}s", t0.elapsed().as_secs_f32());
        eprintln!("[rrr] Starting generation...\n");

        // Generate tokens from a test prompt
        // [1, 2, 3, 4] — arbitrary tokens, no real tokenizer
        // Will produce garbage text but proves the pipeline runs end-to-end
        let prompt = vec![1u32, 2, 3, 4];
        phi4.generate(&engine, &prompt, 10);
    }
}
