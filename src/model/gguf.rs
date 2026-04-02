//! GGUF model loader — mmap, zero-copy, SIMD-ready.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub typ: u32,
    pub offset: u64,
}

pub struct GGUFModel {
    pub version: u32,
    pub n_tensors: u64,
    pub metadata: HashMap<String, MetaValue>,
    pub tensor_infos: Vec<TensorInfo>,
    pub data_offset: u64,
    mmap: Mmap,
}

#[derive(Debug, Clone)]
pub enum MetaValue {
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<MetaValue>),
}

impl MetaValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetaValue::U32(v) => Some(*v as u64),
            MetaValue::U64(v) => Some(*v),
            MetaValue::I32(v) => Some(*v as u64),
            MetaValue::I64(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetaValue::F32(v) => Some(*v),
            MetaValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self { MetaValue::Str(s) => Some(s), _ => None }
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self { Reader { data, pos: 0 } }
    
    fn u8(&mut self) -> u8 { let v = self.data[self.pos]; self.pos += 1; v }
    fn u32(&mut self) -> u32 { 
        let v = u32::from_le_bytes(self.data[self.pos..self.pos+4].try_into().unwrap());
        self.pos += 4; v
    }
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.data[self.pos..self.pos+8].try_into().unwrap());
        self.pos += 8; v
    }
    fn i32(&mut self) -> i32 { self.u32() as i32 }
    fn i64(&mut self) -> i64 { self.u64() as i64 }
    fn f32(&mut self) -> f32 { f32::from_le_bytes(self.data[self.pos..self.pos+4].try_into().unwrap()); let v = f32::from_le_bytes(self.data[self.pos-4..self.pos].try_into().unwrap()); self.pos += 0; v }
    
    fn f32_raw(&mut self) -> f32 {
        let v = f32::from_le_bytes(self.data[self.pos..self.pos+4].try_into().unwrap());
        self.pos += 4; v
    }
    fn f64_raw(&mut self) -> f64 {
        let v = f64::from_le_bytes(self.data[self.pos..self.pos+8].try_into().unwrap());
        self.pos += 8; v
    }
    
    fn string(&mut self) -> String {
        let len = self.u64() as usize;
        let s = std::str::from_utf8(&self.data[self.pos..self.pos+len]).unwrap().to_string();
        self.pos += len; s
    }
    
    fn value(&mut self, typ: u32) -> MetaValue {
        match typ {
            0 => MetaValue::U32(self.u32()),   // UINT8 -> stored as u32
            1 => MetaValue::I32(self.i32()),    // INT8
            2 => MetaValue::U32({ let v = u16::from_le_bytes(self.data[self.pos..self.pos+2].try_into().unwrap()); self.pos += 2; v as u32 }),
            3 => MetaValue::I32({ let v = i16::from_le_bytes(self.data[self.pos..self.pos+2].try_into().unwrap()); self.pos += 2; v as i32 }),
            4 => MetaValue::U32(self.u32()),
            5 => MetaValue::I32(self.i32()),
            6 => MetaValue::F32(self.f32_raw()),
            7 => MetaValue::Bool(self.u8() != 0),
            8 => MetaValue::Str(self.string()),
            9 => {
                let arr_type = self.u32();
                let arr_len = self.u64() as usize;
                let mut arr = Vec::with_capacity(arr_len);
                for _ in 0..arr_len { arr.push(self.value(arr_type)); }
                MetaValue::Array(arr)
            }
            10 => MetaValue::U64(self.u64()),
            11 => MetaValue::I64(self.i64()),
            12 => MetaValue::F64(self.f64_raw()),
            _ => { self.pos += 1; MetaValue::U32(0) }
        }
    }
}

impl GGUFModel {
    pub fn load(path: &str) -> Self {
        let file = File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };
        
        let mut r = Reader::new(&mmap);
        
        let magic = r.u32();
        assert_eq!(magic, GGUF_MAGIC, "Not a GGUF file");
        let version = r.u32();
        let n_tensors = r.u64();
        let n_kv = r.u64();
        
        eprintln!("[GGUF] v{}, {} tensors, {} kv pairs", version, n_tensors, n_kv);
        
        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let key = r.string();
            let typ = r.u32();
            let val = r.value(typ);
            metadata.insert(key, val);
        }
        
        // Read tensor infos
        let mut tensor_infos = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = r.string();
            let n_dims = r.u32();
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims { shape.push(r.u64()); }
            let typ = r.u32();
            let offset = r.u64();
            tensor_infos.push(TensorInfo { name, shape, typ, offset });
        }
        
        // Data starts at alignment boundary after header
        let alignment = metadata.get("general.alignment")
            .and_then(|v| v.as_u64()).unwrap_or(32) as usize;
        let data_offset = (r.pos + alignment - 1) / alignment * alignment;
        eprintln!("[GGUF] data_offset={}, header_end={}", data_offset, r.pos);
        
        GGUFModel {
            version, n_tensors, metadata, tensor_infos,
            data_offset: data_offset as u64, mmap,
        }
    }
    
    /// Get raw bytes for a tensor (zero-copy from mmap)
    pub fn tensor_bytes(&self, name: &str) -> &[u8] {
        let info = self.tensor_infos.iter().find(|t| t.name == name)
            .unwrap_or_else(|| panic!("Tensor not found: {}", name));
        
        let n_elements: u64 = info.shape.iter().product();
        let byte_size = match info.typ {
            0 => n_elements * 4,   // F32
            1 => n_elements * 2,   // F16
            2 => (n_elements / 32) * 18,   // Q4_0 (32 elems, 2B scale + 16B quants)
            3 => (n_elements / 32) * 20,   // Q4_1
            6 => (n_elements / 32) * 22,   // Q5_0 (32 elems, 2B scale + 4B qh + 16B qs)
            7 => (n_elements / 32) * 24,   // Q5_1
            8 => (n_elements / 32) * 34,   // Q8_0 (32 elems, 2B scale + 32B quants)
            9 => (n_elements / 32) * 36,   // Q8_1
            12 => (n_elements / 256) * 144,  // Q4_K
            13 => (n_elements / 256) * 176,  // Q5_K
            14 => (n_elements / 256) * 210,  // Q6_K
            _ => panic!("Unknown quant type {} for tensor {}", info.typ, name),
        };
        
        let start = self.data_offset as usize + info.offset as usize;
        &self.mmap[start..start + byte_size as usize]
    }
    
    /// Dequantize F32 tensor (zero-copy view)
    pub fn tensor_f32(&self, name: &str) -> &[f32] {
        let bytes = self.tensor_bytes(name);
        unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                bytes.len() / 4,
            )
        }
    }
    
    pub fn meta_u64(&self, key: &str) -> u64 {
        self.metadata.get(key).and_then(|v| v.as_u64()).unwrap_or(0)
    }
    
    pub fn meta_str(&self, key: &str) -> &str {
        self.metadata.get(key).and_then(|v| v.as_str()).unwrap_or("")
    }
}
