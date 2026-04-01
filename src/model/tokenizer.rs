//! BPE tokenizer — reads vocab/merges from GGUF metadata.
//! GPT-2 style byte-level BPE with the standard byte↔unicode mapping.

use std::collections::HashMap;
use crate::model::gguf::{GGUFModel, MetaValue};

pub struct BPETokenizer {
    pub vocab: Vec<String>,                       // id → token string
    token_to_id: HashMap<String, u32>,            // token string → id
    merge_ranks: HashMap<(String, String), usize>, // pair → priority (lower = merge first)
    byte_decoder: HashMap<char, u8>,              // GPT-2 unicode char → byte
    byte_encoder: HashMap<u8, char>,              // byte → GPT-2 unicode char
}

impl BPETokenizer {
    pub fn from_gguf(gguf: &GGUFModel) -> Self {
        // Read vocab
        let vocab: Vec<String> = match gguf.metadata.get("tokenizer.ggml.tokens") {
            Some(MetaValue::Array(arr)) => {
                arr.iter().filter_map(|v| {
                    if let MetaValue::Str(s) = v { Some(s.clone()) } else { None }
                }).collect()
            }
            _ => panic!("No tokenizer.ggml.tokens in GGUF"),
        };

        // Build token→id map
        let token_to_id: HashMap<String, u32> = vocab.iter().enumerate()
            .map(|(i, s): (usize, &String)| (s.clone(), i as u32)).collect();

        // Read merges
        let merge_ranks = match gguf.metadata.get("tokenizer.ggml.merges") {
            Some(MetaValue::Array(arr)) => {
                arr.iter().enumerate().filter_map(|(rank, v)| {
                    if let MetaValue::Str(s) = v {
                        let parts: Vec<&str> = s.splitn(2, ' ').collect();
                        if parts.len() == 2 {
                            Some(((parts[0].to_string(), parts[1].to_string()), rank))
                        } else { None }
                    } else { None }
                }).collect()
            }
            _ => HashMap::new(),
        };

        eprintln!("[tokenizer] {} vocab, {} merges", vocab.len(), merge_ranks.len());

        // Build GPT-2 byte↔unicode mapping
        let (byte_encoder, byte_decoder) = build_gpt2_byte_maps();

        BPETokenizer { vocab, token_to_id, merge_ranks, byte_decoder, byte_encoder }
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32]) -> String {
        let raw: String = tokens.iter()
            .filter_map(|&id| self.vocab.get(id as usize))
            .cloned()
            .collect();

        // Convert GPT-2 unicode encoding back to bytes
        let bytes: Vec<u8> = raw.chars()
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect();

        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Encode text to token IDs using BPE
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() { return vec![]; }

        // Convert bytes to GPT-2 unicode chars
        let encoded: String = text.bytes()
            .map(|b| self.byte_encoder[&b])
            .collect();

        // Simple pre-tokenization: split on spaces, keeping space as prefix of next word
        let words = pre_tokenize(&encoded);

        let mut token_ids = Vec::new();
        for word in &words {
            let bpe_tokens = self.bpe(word);
            for tok in &bpe_tokens {
                if let Some(&id) = self.token_to_id.get(tok) {
                    token_ids.push(id);
                } else {
                    // Fallback: encode each character individually
                    for c in tok.chars() {
                        let s = c.to_string();
                        if let Some(&id) = self.token_to_id.get(&s) {
                            token_ids.push(id);
                        }
                    }
                }
            }
        }
        token_ids
    }

    /// Apply BPE merges to a word
    fn bpe(&self, word: &str) -> Vec<String> {
        let mut pieces: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if pieces.len() <= 1 { return pieces; }

        loop {
            let mut best_rank = usize::MAX;
            let mut best_idx = 0;
            let mut found = false;

            for i in 0..pieces.len() - 1 {
                let pair = (pieces[i].clone(), pieces[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                        found = true;
                    }
                }
            }

            if !found { break; }

            let merged = format!("{}{}", pieces[best_idx], pieces[best_idx + 1]);
            pieces[best_idx] = merged;
            pieces.remove(best_idx + 1);

            if pieces.len() <= 1 { break; }
        }

        pieces
    }
}

/// GPT-2 pre-tokenization: split keeping spaces attached to following word
fn pre_tokenize(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let space_char = '\u{0120}'; // GPT-2's encoding of space (Ġ)

    for &c in &chars {
        if c == space_char && !current.is_empty() {
            words.push(current.clone());
            current.clear();
        }
        current.push(c);
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
}

/// Build the GPT-2 byte↔unicode char mapping
fn build_gpt2_byte_maps() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut bs: Vec<u8> = Vec::new();
    let mut cs: Vec<u32> = Vec::new();

    // Printable ranges that map to themselves
    for b in 33u8..=126 { bs.push(b); cs.push(b as u32); }
    for b in 161u8..=172 { bs.push(b); cs.push(b as u32); }
    for b in 174u8..=255 { bs.push(b); cs.push(b as u32); }

    // Remaining bytes map to 256+
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    let mut encoder = HashMap::new();
    let mut decoder = HashMap::new();
    for (&b, &c) in bs.iter().zip(cs.iter()) {
        let ch = char::from_u32(c).unwrap();
        encoder.insert(b, ch);
        decoder.insert(ch, b);
    }
    (encoder, decoder)
}
