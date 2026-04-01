# Recursive Routing Racer (RRR)

A from-scratch Phi-4 inference engine in Rust, running GGUF models directly on Vulkan compute shaders. No PyTorch, no CUDA, no ONNX — raw SPIR-V shaders reading K-quant blocks on the fly.

## Status

**Working.** Generates coherent text from Phi-4-mini (3.8B, Q4_K_M) on an AMD Radeon 890M integrated GPU.

```
Prompt: "The capital of France is"
Output: known as Paris the capital of Europe.
Speed:  5.5 tok/s (Vulkan compute, Radeon 890M)
Corr:   0.991 logit correlation with llama.cpp at position 0
```

## Architecture

| Component | Implementation |
|-----------|---------------|
| GGUF loader | Zero-copy mmap, raw K-quant bytes to GPU |
| Embedding | Q6_K dequant on CPU, F32 on GPU for logits |
| RMS norm | SPIR-V shader, 256-thread parallel reduction |
| QKV / output proj | Native Q4_K/Q5_K SPIR-V matmul (dequant in shader) |
| FFN up/down | Native Q4_K/Q6_K SPIR-V matmul |
| RoPE | CPU, LongRoPE with short factors + attn_factor scaling |
| Attention | CPU, grouped-query (24 heads / 8 KV heads) |
| Logits | F32 register-tiled matmul (chunked for 200K vocab) |
| Tokenizer | BPE from GGUF vocab + merges |

### What's on GPU

All weight-heavy operations: RMS norm, every matmul (Q4_K, Q5_K, Q6_K, F32). Weights stay as raw GGUF K-quant bytes on GPU — ~1.9 GB for 32 layers. Dequantization happens inside the compute shaders, zero host-side requantization.

### What's on CPU

RoPE (cheap), attention score computation (the next optimization target), SiLU activation, softmax, tokenizer.

## Model Support

Tested with:
- `Phi-4-mini-reasoning-Q4_K_M.gguf` (lmstudio-community)

Architecture: Phi-3 (`phi3` in GGUF). Supports Q4_K, Q5_K, Q6_K quantization types. Mixed quantization per layer (Q5_K for QKV, Q4_K for projections, Q6_K for select FFN layers).

## Building

```bash
# Requires: Rust, Vulkan SDK (glslangValidator), Vulkan-capable GPU
cargo build --release

# Compile shaders (if modifying)
cd /path/to/torch-vulkan/csrc/shaders
glslangValidator -V matmul_q4k.comp -o matmul_q4k.spv
glslangValidator -V matmul_q5k.comp -o matmul_q5k.spv
glslangValidator -V matmul_q6k.comp -o matmul_q6k.spv
glslangValidator -V rmsnorm.comp -o rmsnorm.spv
```

## Running

Edit `src/main.rs` to point `GGUF_PATH` and `SHADER_DIR` at your model and shaders, then:

```bash
cargo run --release
```

## Bug History

Eight dequantization bugs fixed across Q4_K, Q5_K, and Q6_K shaders. The critical one:

**Bug #8 (Q6_K scale index):** Each Q6_K block has 256 elements and 16 scales. The code used `is = g * 8` to index scales, addressing only 8 of 16. The correct formula from ggml is `is = g * 8 + l / 16`, which splits each 32-element quad into two halves with different scales. This affected the embedding (Q6_K) and FFN-down in half the layers. Fixing it took logit correlation from negative to 0.991.

## Dependencies

- [ash](https://crates.io/crates/ash) — Vulkan bindings
- [memmap2](https://crates.io/crates/memmap2) — GGUF mmap
- [half](https://crates.io/crates/half) — F16 conversion
- [rayon](https://crates.io/crates/rayon) — parallel iteration
- Shaders from [torch-vulkan](https://github.com/Peterc3-dev/torch-vulkan)
