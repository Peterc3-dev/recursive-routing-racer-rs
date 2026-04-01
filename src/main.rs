mod gpu;
mod model;
use std::io::Write;
use std::time::Instant;

const GGUF_PATH: &str = "/home/raz/.lmstudio/models/lmstudio-community/Phi-4-mini-reasoning-GGUF/Phi-4-mini-reasoning-Q4_K_M.gguf";
const SHADER_DIR: &str = "/home/raz/projects/torch-vulkan/csrc/shaders";

const MAX_TOKENS: usize = 256;
const EOS_TOKEN: u32 = 199999;  // <|endoftext|> for Phi-4

fn main() {
    let batch = std::env::args().any(|a| a == "--batch");

    let gguf = model::GGUFModel::load(GGUF_PATH);
    let tokenizer = model::BPETokenizer::from_gguf(&gguf);

    unsafe {
        let mut engine = gpu::ComputeEngine::new(SHADER_DIR);
        let phi4 = model::Phi4Model::load_from_gguf(&gguf, &mut engine);

        if batch {
            // Batch mode: read one line from stdin, print generated text to stdout, exit.
            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).unwrap() == 0 { return; }
            let input = input.trim();
            if input.is_empty() { return; }

            let prompt_tokens = tokenizer.encode(input);
            let mut cache = model::KVCache::new();

            let pt = Instant::now();
            let mut logits = Vec::new();
            for &tok in &prompt_tokens {
                let mut h = phi4.embed(tok);
                logits = phi4.forward_gpu(&engine, &mut h, &mut cache);
            }
            eprintln!("[prefill {}ms, {} tokens]", pt.elapsed().as_millis(), prompt_tokens.len());

            let gt = Instant::now();
            let mut gen_count = 0usize;
            let mut output = String::new();

            for _ in 0..MAX_TOKENS {
                let (best_id, _) = logits.iter().enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
                let tok = best_id as u32;
                if tok == EOS_TOKEN { break; }

                output.push_str(&tokenizer.decode(&[tok]));
                gen_count += 1;

                let mut hidden = phi4.embed(tok);
                logits = phi4.forward_gpu(&engine, &mut hidden, &mut cache);
            }

            let gen_time = gt.elapsed().as_secs_f32();
            eprintln!("[{} tokens, {:.1} tok/s]", gen_count,
                if gen_time > 0.0 { gen_count as f32 / gen_time } else { 0.0 });

            print!("{}", output);
            std::io::stdout().flush().unwrap();
        } else {
            // Interactive mode
            println!("=== RRR — Phi-4 Mini (Vulkan Compute) ===");
            println!("Type a prompt and press Enter. Ctrl-C to quit.\n");

            let mut cache = model::KVCache::new();

            loop {
                print!("> ");
                std::io::stdout().flush().unwrap();
                let mut input = String::new();
                if std::io::stdin().read_line(&mut input).unwrap() == 0 { break; }
                let input = input.trim();
                if input.is_empty() { continue; }
                if input == "/reset" {
                    cache = model::KVCache::new();
                    println!("[context cleared]");
                    continue;
                }

                let prompt_tokens = tokenizer.encode(input);
                if prompt_tokens.is_empty() { continue; }

                let pt = Instant::now();
                let mut logits = Vec::new();
                for &tok in &prompt_tokens {
                    let mut h = phi4.embed(tok);
                    logits = phi4.forward_gpu(&engine, &mut h, &mut cache);
                }
                let prefill_ms = pt.elapsed().as_millis();
                eprint!("\x1b[90m[prefill {}ms, {} tokens]\x1b[0m ", prefill_ms, prompt_tokens.len());
                std::io::stdout().flush().unwrap();

                let gt = Instant::now();
                let mut gen_count = 0usize;

                for _ in 0..MAX_TOKENS {
                    let (best_id, _) = logits.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
                    let tok = best_id as u32;
                    if tok == EOS_TOKEN { break; }

                    let decoded = tokenizer.decode(&[tok]);
                    print!("{}", decoded);
                    std::io::stdout().flush().unwrap();
                    gen_count += 1;

                    let mut hidden = phi4.embed(tok);
                    logits = phi4.forward_gpu(&engine, &mut hidden, &mut cache);
                }

                let gen_time = gt.elapsed().as_secs_f32();
                println!();
                eprintln!("\x1b[90m[{} tokens, {:.1} tok/s]\x1b[0m",
                    gen_count, if gen_time > 0.0 { gen_count as f32 / gen_time } else { 0.0 });
            }
        }
    }
}
