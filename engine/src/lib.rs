//! TTS frontend WASM engine: normalize + G2P + SPM + BERT align.
//! Acoustic ONNX models stay outside; frontend assets are embedded here.

pub mod chinese;
pub mod japanese;
mod cmudict;
pub mod g2p;
mod normalize;
mod jieba_dict;
mod pinyin_util;
mod tone_sandhi;
mod symbols;

use g2p::{prepare, prepare_lang};
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

const _CONFIG_JSON: &[u8] = include_bytes!("../assets/config.json");
const _DEBERTA_CONFIG_JSON: &[u8] = include_bytes!("../assets/deberta_config.json");
const _TOKENIZER_CONFIG_JSON: &[u8] = include_bytes!("../assets/tokenizer_config.json");

fn pack_prepare(p: &g2p::Prepared, out_len: *mut usize) -> *mut u8 {
    let mut buf: Vec<u8> = Vec::with_capacity(
        20 + 4
            * (p.input_ids.len()
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
    // magic TPS1, then n_ids, n_phones, n_w2p, bert_lang
    push_u32(&mut buf, 0x5450_5331);
    push_u32(&mut buf, p.input_ids.len() as u32);
    push_u32(&mut buf, p.phones.len() as u32);
    push_u32(&mut buf, p.word2ph.len() as u32);
    push_u32(&mut buf, p.bert_lang as u32);
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

/// Packed prepare blob (little-endian):
/// magic u32 = 0x54505331 ("TPS1")
/// n_ids, n_phones, n_w2p, bert_lang : u32
/// input_ids[n_ids] i32
/// phones / tones / language [n_phones] i32
/// word2ph[n_w2p] i32
///
/// `lang`: 0=ZH, 1=JP, 2=EN (default EN if using engine_prepare).
#[no_mangle]
pub extern "C" fn engine_prepare_lang(
    text_ptr: *const u8,
    text_len: usize,
    lang: u32,
    out_len: *mut usize,
) -> *mut u8 {
    let text = unsafe {
        if text_ptr.is_null() {
            ""
        } else {
            std::str::from_utf8(slice::from_raw_parts(text_ptr, text_len)).unwrap_or("")
        }
    };
    let p = prepare_lang(text, lang as i32);
    pack_prepare(&p, out_len)
}

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
    pack_prepare(&p, out_len)
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

/// Expand hidden [seq,bert_dim] → primary bert slot via word2ph;
/// other language slots get U[0,1]; emo zeros; zin ~ N(0,1)*sdp_noise.
///
/// `bert_lang`: 0=zh, 1=ja, 2=en — which slot receives `hidden`.
///
/// Output packed f32 LE:
/// magic 0x54424E31 ("TBN1")
/// n_phone u32, bert_dim u32, emo_dim u32
/// en_bert / zh_bert / ja_bert / emo / zin
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
    // Default: EN slot (backward compatible with older hosts).
    engine_pack_bert_lang(
        hidden_ptr,
        seq,
        word2ph_ptr,
        n_w2p,
        bert_dim,
        emo_dim,
        seed,
        sdp_noise,
        2,
        out_len,
    )
}

#[no_mangle]
pub extern "C" fn engine_pack_bert_lang(
    hidden_ptr: *const f32,
    seq: usize,
    word2ph_ptr: *const i32,
    n_w2p: usize,
    bert_dim: usize,
    emo_dim: usize,
    seed: u32,
    sdp_noise: f32,
    bert_lang: u32,
    out_len: *mut usize,
) -> *mut u8 {
    if hidden_ptr.is_null() || word2ph_ptr.is_null() || seq == 0 || bert_dim == 0 {
        return ptr::null_mut();
    }
    let emo_dim = if emo_dim == 0 { 512 } else { emo_dim };
    let hidden = unsafe { slice::from_raw_parts(hidden_ptr, seq * bert_dim) };
    let word2ph = unsafe { slice::from_raw_parts(word2ph_ptr, n_w2p) };
    let n_phone: usize = word2ph.iter().map(|x| *x as usize).sum();

    let mut primary = vec![0f32; n_phone * bert_dim];
    let mut o = 0usize;
    for (ti, &reps) in word2ph.iter().enumerate() {
        let src = ti.min(seq.saturating_sub(1));
        let row = &hidden[src * bert_dim..(src + 1) * bert_dim];
        for _ in 0..reps {
            primary[o * bert_dim..(o + 1) * bert_dim].copy_from_slice(row);
            o += 1;
        }
    }

    let mut rng = Rng64::new(seed);
    let mut zh = vec![0f32; n_phone * bert_dim];
    let mut ja = vec![0f32; n_phone * bert_dim];
    let mut en = vec![0f32; n_phone * bert_dim];
    match bert_lang {
        0 => zh.copy_from_slice(&primary),
        1 => ja.copy_from_slice(&primary),
        _ => en.copy_from_slice(&primary),
    }
    for (slot, lang) in [(&mut zh, 0u32), (&mut ja, 1u32), (&mut en, 2u32)] {
        if lang != bert_lang {
            for v in slot.iter_mut() {
                *v = rng.next_f32();
            }
        }
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

#[no_mangle]
pub extern "C" fn engine_assets_bytes() -> u32 {
    (_CONFIG_JSON.len() + _DEBERTA_CONFIG_JSON.len() + _TOKENIZER_CONFIG_JSON.len()) as u32
}
