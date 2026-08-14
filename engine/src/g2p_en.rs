//! Minimal g2p_en OOV phoneme predictor (GRU seq2seq from checkpoint20.npz).
//! Used when a word is missing from Bert-VITS2 CMUdict — mirrors upstream `_g2p(word)`.

use once_cell::sync::Lazy;
use std::f32::consts::E;

const WEIGHTS: &[u8] = include_bytes!("../assets/g2p_en_weights.bin");

const PHONEMES: [&str; 74] = [
    "<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1",
    "AH2", "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH",
    "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1",
    "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1",
    "OY2", "P", "R", "S", "SH", "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V",
    "W", "Y", "Z", "ZH",
];

/// Grapheme index: 0=pad, 1=unk, 2=</s>, 3..=a..z
fn grapheme_idx(ch: char) -> usize {
    if ch.is_ascii_lowercase() {
        (ch as u8 - b'a') as usize + 3
    } else {
        1 // <unk> — g2p_en keeps hyphens etc. as unk, not stripped
    }
}

fn grapheme_seq(word: &str) -> Vec<usize> {
    let mut seq: Vec<usize> = word
        .chars()
        .flat_map(|c| c.to_lowercase())
        .map(grapheme_idx)
        .collect();
    seq.push(2); // </s>
    seq
}

struct Arrays {
    enc_emb: Vec<f32>,       // [29, 256]
    enc_w_ih: Vec<f32>,      // [768, 256]
    enc_w_hh: Vec<f32>,
    enc_b_ih: Vec<f32>,      // [768]
    enc_b_hh: Vec<f32>,
    dec_emb: Vec<f32>,       // [74, 256]
    dec_w_ih: Vec<f32>,
    dec_w_hh: Vec<f32>,
    dec_b_ih: Vec<f32>,
    dec_b_hh: Vec<f32>,
    fc_w: Vec<f32>,          // [74, 256]
    fc_b: Vec<f32>,          // [74]
    hidden: usize,
    emb: usize,
    n_phonemes: usize,
}

fn load_arrays() -> Arrays {
    let mut p = 0usize;
    assert_eq!(&WEIGHTS[p..p + 4], b"G2PE");
    p += 4;
    let n_arrays = u32::from_le_bytes(WEIGHTS[p..p + 4].try_into().unwrap()) as usize;
    p += 4;

    let mut arrs: Vec<(Vec<usize>, Vec<f32>)> = Vec::with_capacity(n_arrays);
    for _ in 0..n_arrays {
        p += 16; // name padding
        let ndim = u32::from_le_bytes(WEIGHTS[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        let size = u32::from_le_bytes(WEIGHTS[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(u32::from_le_bytes(WEIGHTS[p..p + 4].try_into().unwrap()) as usize);
            p += 4;
        }
        let mut data = vec![0f32; size];
        for v in &mut data {
            *v = f32::from_le_bytes(WEIGHTS[p..p + 4].try_into().unwrap());
            p += 4;
        }
        arrs.push((shape, data));
    }

    let hidden = arrs[1].0[1];
    let emb = arrs[0].0[1];
    Arrays {
        enc_emb: arrs[0].1.clone(),
        enc_w_ih: arrs[1].1.clone(),
        enc_w_hh: arrs[2].1.clone(),
        enc_b_ih: arrs[3].1.clone(),
        enc_b_hh: arrs[4].1.clone(),
        dec_emb: arrs[5].1.clone(),
        dec_w_ih: arrs[6].1.clone(),
        dec_w_hh: arrs[7].1.clone(),
        dec_b_ih: arrs[8].1.clone(),
        dec_b_hh: arrs[9].1.clone(),
        fc_w: arrs[10].1.clone(),
        fc_b: arrs[11].1.clone(),
        hidden,
        emb,
        n_phonemes: 74,
    }
}

static MODEL: Lazy<Arrays> = Lazy::new(load_arrays);

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

fn dot_row(mat: &[f32], cols: usize, row: usize, x: &[f32]) -> f32 {
    let off = row * cols;
    mat[off..off + cols]
        .iter()
        .zip(x.iter())
        .map(|(a, b)| a * b)
        .sum()
}

fn matvec_t(mat: &[f32], cols: usize, x: &[f32], out: &mut [f32]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = dot_row(mat, cols, i, x);
    }
}

fn grucell(x: &[f32], h: &mut [f32], w: &Arrays, dec: bool) {
    let (w_ih, w_hh, b_ih, b_hh) = if dec {
        (&w.dec_w_ih, &w.dec_w_hh, &w.dec_b_ih, &w.dec_b_hh)
    } else {
        (&w.enc_w_ih, &w.enc_w_hh, &w.enc_b_ih, &w.enc_b_hh)
    };
    let hsz = w.hidden;
    let emb = w.emb;
    let gate = hsz * 3;
    let mut rzn_ih = vec![0f32; gate];
    let mut rzn_hh = vec![0f32; gate];
    matvec_t(w_ih, emb, x, &mut rzn_ih);
    for (i, b) in b_ih.iter().enumerate() {
        rzn_ih[i] += b;
    }
    matvec_t(w_hh, hsz, h, &mut rzn_hh);
    for (i, b) in b_hh.iter().enumerate() {
        rzn_hh[i] += b;
    }
    let two_thirds = gate * 2 / 3;
    let mut h_new = vec![0f32; hsz];
    for i in 0..hsz {
        let r = sigmoid(rzn_ih[i] + rzn_hh[i]);
        let z = sigmoid(rzn_ih[hsz + i] + rzn_hh[hsz + i]);
        let n = (rzn_ih[two_thirds + i] + r * rzn_hh[two_thirds + i]).tanh();
        h_new[i] = (1.0 - z) * n + z * h[i];
    }
    h.copy_from_slice(&h_new);
}

fn emb_lookup(table: &[f32], dim: usize, idx: usize) -> Vec<f32> {
    let off = idx * dim;
    table[off..off + dim].to_vec()
}

/// Predict ARPAbet phones for an OOV word (lowercase a-z and apostrophe stripped by caller).
pub fn predict(word: &str) -> Vec<String> {
    let w = &*MODEL;
    let lower: String = word.chars().flat_map(|c| c.to_lowercase()).collect();
    if lower.is_empty() {
        return Vec::new();
    }

    let seq = grapheme_seq(&lower);

    let mut h = vec![0f32; w.hidden];
    for &idx in &seq {
        let x = emb_lookup(&w.enc_emb, w.emb, idx);
        grucell(&x, &mut h, w, false);
    }

    let mut dec = emb_lookup(&w.dec_emb, w.emb, 2); // <s>
    let mut preds = Vec::new();
    for _ in 0..20 {
        grucell(&dec, &mut h, w, true);
        let mut best_idx = 0usize;
        let mut best = f32::NEG_INFINITY;
        for (i, b) in w.fc_b.iter().enumerate() {
            let mut logit = *b;
            let row_off = i * w.emb;
            for (j, hj) in h.iter().enumerate() {
                logit += w.fc_w[row_off + j] * hj;
            }
            if logit > best {
                best = logit;
                best_idx = i;
            }
        }
        if best_idx == 3 {
            break;
        }
        preds.push(PHONEMES[best_idx].to_string());
        dec = emb_lookup(&w.dec_emb, w.emb, best_idx);
    }
    preds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evilcorp_matches_g2p_en() {
        let p = predict("evilcorp");
        assert_eq!(
            p,
            vec![
                "EH1", "V", "IH0", "L", "K", "AO2", "R", "P"
            ]
        );
    }

    #[test]
    fn ceo_matches_g2p_en() {
        let p = predict("CEO");
        assert_eq!(p, vec!["S", "IY1", "OW0"]);
    }

    #[test]
    fn twenty_four_hyphen_matches_g2p_en() {
        let p = predict("twenty-four");
        assert_eq!(
            p,
            vec!["T", "W", "EH1", "N", "T", "IY0", "F", "UH1", "R"]
        );
    }
}
