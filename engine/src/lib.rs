//! TTS frontend WASM engine: normalize + G2P + SPM + BERT align (SIMD).
//! Acoustic ONNX models stay outside; everything else is embedded here.

mod cmudict;
pub mod g2p;
mod normalize;
mod symbols;

use g2p::prepare;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use std::slice;

/// Tiny seedable PRNG (xorshift64*) — no getrandom / wasm-bindgen imports.
struct Rng64(u64);
impl Rng64 {
    fn new(seed: u32) -> Self {
        let mut s = seed as u64;
        if s == 0 {
            s = 0x9E3779B97F4A7C15;
        }
        Self(s)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// Embed leftover JSON assets so they live inside engine.wasm
const _CONFIG_JSON: &[u8] = include_bytes!("../assets/config.json");
const _DEBERTA_CONFIG_JSON: &[u8] = include_bytes!("../assets/deberta_config.json");
const _TOKENIZER_CONFIG_JSON: &[u8] = include_bytes!("../assets/tokenizer_config.json");

/// Packed prepare blob (little-endian):
/// magic u32 = 0x54505331 ("TPS1")
/// n_ids, n_phones, n_w2p : u32
/// input_ids[n_ids] i32
/// phones[n_phones] i32
/// tones[n_phones] i32
/// language[n_phones] i32
/// word2ph[n_w2p] i32
#[no_mangle]
pub extern "C" fn engine_prepare(text_ptr: *const u8, text_len: usize, out_len: *mut usize) -> *mut u8 {
    let text = unsafe {
        if text_ptr.is_null() {
            ""
        } else {
            std::str::from_utf8(slice::from_raw_parts(text_ptr, text_len)).unwrap_or("")
        }
    };
    let p = prepare(text);
    let mut buf: Vec<u8> = Vec::with_capacity(
        16 + 4 * (p.input_ids.len()
            + p.phones.len() * 3
            + p.word2ph.len()),
    );
    fn push_u32(b: &mut Vec<u8>, v: u32) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn push_i32s(b: &mut Vec<u8>, xs: &[i32]) {
        for x in xs {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    push_u32(&mut buf, 0x5450_5331);
    push_u32(&mut buf, p.input_ids.len() as u32);
    push_u32(&mut buf, p.phones.len() as u32);
    push_u32(&mut buf, p.word2ph.len() as u32);
    push_i32s(&mut buf, &p.input_ids);
    push_i32s(&mut buf, &p.phones);
    push_i32s(&mut buf, &p.tones);
    push_i32s(&mut buf, &p.language);
    push_i32s(&mut buf, &p.word2ph);

    let len = buf.len();
    unsafe {
        if !out_len.is_null() {
            *out_len = len;
        }
        let layout = Layout::from_size_align(len, 8).unwrap();
        let ptr = alloc(layout);
        if ptr.is_null() {
            return ptr::null_mut();
        }
        ptr::copy_nonoverlapping(buf.as_ptr(), ptr, len);
        ptr
    }
}

#[no_mangle]
pub extern "C" fn engine_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    unsafe {
        dealloc(ptr, Layout::from_size_align(len, 8).unwrap());
    }
}

/// Expand Deberta hidden [seq,bert_dim] → en_bert [n_phone,bert_dim] via word2ph,
/// and allocate zh/ja random U[0,1], emo zeros, zin ~ N(0,1)*sdp_noise.
///
/// Output packed f32 LE:
/// magic 0x54424E31 ("TBN1")
/// n_phone u32, bert_dim u32, emo_dim u32
/// en_bert[n_phone*bert_dim]
/// zh_bert[n_phone*bert_dim]
/// ja_bert[n_phone*bert_dim]
/// emo[emo_dim]
/// zin[2*n_phone]
#[no_mangle]
pub extern "C" fn engine_pack_bert(
    hidden_ptr: *const f32,
    seq: usize,
    word2ph_ptr: *const i32,
    n_w2p: usize,
    bert_dim: usize,
    emo_dim: usize,
    seed: u32,
    sdp_noise: f32,
    out_len: *mut usize,
) -> *mut u8 {
    if hidden_ptr.is_null() || word2ph_ptr.is_null() || seq == 0 || bert_dim == 0 {
        return ptr::null_mut();
    }
    let emo_dim = if emo_dim == 0 { 512 } else { emo_dim };
    let hidden = unsafe { slice::from_raw_parts(hidden_ptr, seq * bert_dim) };
    let word2ph = unsafe { slice::from_raw_parts(word2ph_ptr, n_w2p) };
    let n_phone: usize = word2ph.iter().map(|x| *x as usize).sum();

    let mut en = vec![0f32; n_phone * bert_dim];
    let mut o = 0usize;
    for (ti, &reps) in word2ph.iter().enumerate() {
        let src = ti.min(seq.saturating_sub(1));
        let row = &hidden[src * bert_dim..(src + 1) * bert_dim];
        for _ in 0..reps {
            copy_row_simd(row, &mut en[o * bert_dim..(o + 1) * bert_dim]);
            o += 1;
        }
    }

    let mut rng = Rng64::new(seed);
    let mut zh = vec![0f32; n_phone * bert_dim];
    let mut ja = vec![0f32; n_phone * bert_dim];
    for v in zh.iter_mut().chain(ja.iter_mut()) {
        *v = rng.next_f32();
    }
    let emo = vec![0f32; emo_dim];
    let mut zin = vec![0f32; 2 * n_phone];
    for c in 0..2 {
        for t in 0..n_phone {
            let u1 = rng.next_f32().max(1e-6);
            let u2 = rng.next_f32();
            let r = (-2.0 * u1.ln()).sqrt();
            let th = 2.0 * std::f32::consts::PI * u2;
            zin[c * n_phone + t] = r * th.cos() * sdp_noise;
        }
    }

    let mut buf: Vec<u8> = Vec::new();
    fn push_u32(b: &mut Vec<u8>, v: u32) {
        b.extend_from_slice(&v.to_le_bytes());
    }
    fn push_f32s(b: &mut Vec<u8>, xs: &[f32]) {
        for x in xs {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    push_u32(&mut buf, 0x5442_4E31);
    push_u32(&mut buf, n_phone as u32);
    push_u32(&mut buf, bert_dim as u32);
    push_u32(&mut buf, emo_dim as u32);
    push_f32s(&mut buf, &en);
    push_f32s(&mut buf, &zh);
    push_f32s(&mut buf, &ja);
    push_f32s(&mut buf, &emo);
    push_f32s(&mut buf, &zin);

    let len = buf.len();
    unsafe {
        if !out_len.is_null() {
            *out_len = len;
        }
        let layout = Layout::from_size_align(len, 8).unwrap();
        let ptr = alloc(layout);
        if ptr.is_null() {
            return ptr::null_mut();
        }
        ptr::copy_nonoverlapping(buf.as_ptr(), ptr, len);
        ptr
    }
}

#[inline]
fn copy_row_simd(src: &[f32], dst: &mut [f32]) {
    dst.copy_from_slice(src);
}

/// Allocate a float buffer the host can fill (e.g. Deberta output) before pack_bert.
#[no_mangle]
pub extern "C" fn engine_alloc_f32(n: usize) -> *mut f32 {
    if n == 0 {
        return ptr::null_mut();
    }
    unsafe {
        let layout = Layout::from_size_align(n * 4, 8).unwrap();
        alloc(layout) as *mut f32
    }
}

#[no_mangle]
pub extern "C" fn engine_free_f32(ptr: *mut f32, n: usize) {
    if ptr.is_null() || n == 0 {
        return;
    }
    unsafe {
        dealloc(ptr as *mut u8, Layout::from_size_align(n * 4, 8).unwrap());
    }
}

/// Allocate bytes for host→wasm string passing.
#[no_mangle]
pub extern "C" fn engine_alloc(n: usize) -> *mut u8 {
    if n == 0 {
        return ptr::null_mut();
    }
    unsafe { alloc(Layout::from_size_align(n, 1).unwrap()) }
}

#[no_mangle]
pub extern "C" fn engine_alloc_free(ptr: *mut u8, n: usize) {
    if ptr.is_null() || n == 0 {
        return;
    }
    unsafe {
        dealloc(ptr, Layout::from_size_align(n, 1).unwrap());
    }
}

/// Touch embedded JSON so LTO cannot strip them.
#[no_mangle]
pub extern "C" fn engine_assets_bytes() -> u32 {
    (_CONFIG_JSON.len() + _DEBERTA_CONFIG_JSON.len() + _TOKENIZER_CONFIG_JSON.len()) as u32
}
