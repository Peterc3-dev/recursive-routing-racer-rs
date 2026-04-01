mod gpu;
mod model;
use std::io::Write;

const GGUF_PATH: &str = "/home/raz/.lmstudio/models/lmstudio-community/Phi-4-mini-reasoning-GGUF/Phi-4-mini-reasoning-Q4_K_M.gguf";
const SHADER_DIR: &str = "/home/raz/projects/torch-vulkan/csrc/shaders";

fn main() {
    let gguf = model::GGUFModel::load(GGUF_PATH);
    let tokenizer = model::BPETokenizer::from_gguf(&gguf);

    let prompt_text = "The capital of France is";
    let prompt_tokens = tokenizer.encode(prompt_text);
    eprintln!("[rrr] Prompt: {:?}", prompt_tokens);

    unsafe {
        let mut engine = gpu::ComputeEngine::new(SHADER_DIR);
        let phi4 = model::Phi4Model::load_from_gguf(&gguf, &mut engine);
        phi4.generate(&engine, &prompt_tokens, 20, Some(&tokenizer.vocab));
    }
}
