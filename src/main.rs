mod gpu;
mod model;
use std::io::Write;
use std::time::Instant;

const DEFAULT_GGUF: &str = "/home/raz/models/Phi-4-mini-reasoning-Q4_K_M.gguf";
const SHADER_DIR: &str = "/home/raz/projects/torch-vulkan/csrc/shaders";

const MAX_TOKENS: usize = 256;
const EOS_TOKEN: u32 = 199999;  // <|endoftext|> for Phi-4

/// Parse a --key value pair from the argument list.
fn get_flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
}

/// Build KV cache compression config from CLI args.
fn parse_kv_config(args: &[String]) -> model::KVCompressConfig {
    let compress_mode = get_flag_value(args, "--kv-compress")
        .and_then(model::KVCompressMode::from_str)
        .unwrap_or(model::KVCompressMode::None);

    let evict_mode = get_flag_value(args, "--kv-evict")
        .and_then(model::KVEvictMode::from_str)
        .unwrap_or(model::KVEvictMode::None);

    let budget = get_flag_value(args, "--kv-budget")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);

    let config = model::KVCompressConfig {
        compress_mode,
        evict_mode,
        budget,
        sink_count: 4,
    };

    if config.is_active() {
        eprintln!("[kv-squeeze] compress={}, evict={}, budget={}",
            config.compress_mode, config.evict_mode,
            if config.budget > 0 { config.budget.to_string() } else { "unlimited".to_string() });
    }

    config
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let batch = args.iter().any(|a| a == "--batch");
    let speculative = args.iter().any(|a| a == "--speculative");
    let draft_layers: usize = 8;
    let draft_k: usize = 4;

    // Parse KV compression config from CLI flags
    let kv_config = parse_kv_config(&args);

    // Model path: first non-flag argument (skip flag values too)
    let flag_args = ["--kv-compress", "--kv-evict", "--kv-budget"];
    let gguf_path = {
        let mut skip_next = false;
        let mut found: Option<&str> = None;
        for a in args.iter().skip(1) {
            if skip_next { skip_next = false; continue; }
            if flag_args.contains(&a.as_str()) || a == "--batch" || a == "--speculative" {
                if flag_args.contains(&a.as_str()) { skip_next = true; }
                continue;
            }
            if a.starts_with("--") { continue; }
            found = Some(a.as_str());
            break;
        }
        found.unwrap_or(DEFAULT_GGUF).to_string()
    };

    eprintln!("[rrr] Loading {}", gguf_path);
    let gguf = model::GGUFModel::load(&gguf_path);
    let tokenizer = model::BPETokenizer::from_gguf(&gguf);

    unsafe {
        let mut engine = gpu::ComputeEngine::new(SHADER_DIR);
        let phi4 = model::Phi4Model::load_from_gguf(&gguf, &mut engine);
        let n_kv_heads = phi4.config.n_kv_heads;
        let head_dim = phi4.config.head_dim;
        let n_layers = phi4.config.n_layers;

        if batch {
            // Batch mode: read one line from stdin, print generated text to stdout, exit.
            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).unwrap() == 0 { return; }
            let input = input.trim();
            if input.is_empty() { return; }

            let prompt_tokens = tokenizer.encode(input);
            let mut cache = model::KVCache::with_config(kv_config);

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
            let mut accepted_total = 0usize;
            let mut verify_calls = 0usize;

            if speculative {
                while gen_count < MAX_TOKENS {
                    let (accepted, new_logits) = phi4.speculative_decode(
                        &engine, &logits, &mut cache, draft_layers, draft_k);
                    verify_calls += 1;

                    let mut eos = false;
                    for &tok in &accepted {
                        if tok == EOS_TOKEN { eos = true; break; }
                        output.push_str(&tokenizer.decode(&[tok]));
                        gen_count += 1;
                    }
                    accepted_total += accepted.len();

                    // Get the next token from verify logits
                    let (best, _) = new_logits.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
                    let correction = best as u32;
                    if correction == EOS_TOKEN || eos { break; }

                    output.push_str(&tokenizer.decode(&[correction]));
                    gen_count += 1;

                    // Run the correction token through full model to get next logits
                    let mut hidden = phi4.embed(correction);
                    logits = phi4.forward_gpu(&engine, &mut hidden, &mut cache);
                }
            } else {
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
            }

            let gen_time = gt.elapsed().as_secs_f32();
            if speculative {
                let accept_rate = if verify_calls > 0 { accepted_total as f32 / (verify_calls as f32 * draft_k as f32) } else { 0.0 };
                eprintln!("[{} tokens, {:.1} tok/s, speculative: {:.0}% accept rate, {} verify calls]",
                    gen_count, if gen_time > 0.0 { gen_count as f32 / gen_time } else { 0.0 },
                    accept_rate * 100.0, verify_calls);
            } else {
                eprintln!("[{} tokens, {:.1} tok/s]", gen_count,
                    if gen_time > 0.0 { gen_count as f32 / gen_time } else { 0.0 });
            }

            // Print KV cache compression stats on exit
            cache.print_compress_stats(n_kv_heads, head_dim, n_layers);

            print!("{}", output);
            std::io::stdout().flush().unwrap();
        } else {
            // Interactive mode
            println!("=== RRR — Phi-4 Mini (Vulkan Compute) ===");
            if kv_config.is_active() {
                println!("KV cache: compress={}, evict={}, budget={}",
                    kv_config.compress_mode, kv_config.evict_mode,
                    if kv_config.budget > 0 { kv_config.budget.to_string() } else { "unlimited".to_string() });
            }
            println!("Type a prompt and press Enter. Ctrl-C to quit.\n");

            let mut cache = model::KVCache::with_config(kv_config);

            loop {
                print!("> ");
                std::io::stdout().flush().unwrap();
                let mut input = String::new();
                if std::io::stdin().read_line(&mut input).unwrap() == 0 {
                    // Print stats on exit (Ctrl-D)
                    cache.print_compress_stats(n_kv_heads, head_dim, n_layers);
                    break;
                }
                let input = input.trim();
                if input.is_empty() { continue; }
                if input == "/reset" {
                    cache.print_compress_stats(n_kv_heads, head_dim, n_layers);
                    let cfg = cache.compress_config.clone();
                    cache = model::KVCache::with_config(cfg);
                    println!("[context cleared]");
                    continue;
                }

                let prompt_tokens = tokenizer.encode(input);
                if prompt_tokens.is_empty() { continue; }

                let pt = Instant::now();
                let logits = phi4.forward_prefill_batched(&engine, &prompt_tokens, &mut cache);
                let mut logits = logits;
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
