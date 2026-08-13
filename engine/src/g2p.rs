//! English G2P using CMUdict (+ punctuation / letter fallback for OOV).

use crate::cmudict::ENG_DICT;
use crate::symbols::{post_replace_ph, ARPA, SYMBOL_TO_ID, EN_TONE_START, LANG_EN};
use sentencepiece_rs::SentencePieceProcessor;
use once_cell::sync::Lazy;
use regex::Regex;

const SPM_BYTES: &[u8] = include_bytes!("../assets/spm.model");

pub static SPM: Lazy<SentencePieceProcessor> = Lazy::new(|| {
    SentencePieceProcessor::from_serialized_model(SPM_BYTES).expect("load spm.model")
});

static SEP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"([,;.\?\!\s]+)").unwrap());

fn refine_ph(phn: &str) -> (String, i32) {
    let bytes = phn.as_bytes();
    if let Some(last) = bytes.last() {
        if last.is_ascii_digit() {
            let tone = (last - b'0') as i32 + 1;
            let stem = &phn[..phn.len() - 1];
            return (stem.to_ascii_lowercase(), tone);
        }
    }
    (phn.to_ascii_lowercase(), 0)
}

fn refine_syllables(syllables: &[Vec<String>]) -> (Vec<String>, Vec<i32>) {
    let mut tones = Vec::new();
    let mut phonemes = Vec::new();
    for phn_list in syllables {
        for phn in phn_list {
            let (ph, tone) = refine_ph(phn);
            phonemes.push(ph);
            tones.push(tone);
        }
    }
    (phonemes, tones)
}

fn distribute_phone(n_phone: usize, n_word: usize) -> Vec<i32> {
    let n_word = n_word.max(1);
    let mut phones_per_word = vec![0i32; n_word];
    for _ in 0..n_phone {
        let min_index = phones_per_word
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| *v)
            .map(|(i, _)| i)
            .unwrap_or(0);
        phones_per_word[min_index] += 1;
    }
    phones_per_word
}

fn sep_text(text: &str) -> Vec<String> {
    // Python: re.split(r"([,;.\?\!\s+])", text) — char-class keeps each punct/space as a token.
    // We keep punctuation tokens; drop whitespace-only pieces.
    let mut out = Vec::new();
    let mut last = 0usize;
    for m in SEP_RE.find_iter(text) {
        if m.start() > last {
            let w = &text[last..m.start()];
            if !w.is_empty() {
                out.push(w.to_string());
            }
        }
        for ch in m.as_str().chars() {
            if matches!(ch, ',' | ';' | '.' | '?' | '!' | '+') {
                out.push(ch.to_string());
            }
            // whitespace discarded (Python keeps them then filters strip=="")
        }
        last = m.end();
    }
    if last < text.len() {
        out.push(text[last..].to_string());
    }
    out.into_iter().filter(|w| !w.trim().is_empty()).collect()
}

/// Letter-name / punctuation fallback when word missing from CMUdict (no neural g2p_en).
fn fallback_phones(word: &str) -> (Vec<String>, Vec<i32>) {
    let mut phns = Vec::new();
    let mut tns = Vec::new();
    for ch in word.chars() {
        if ch.is_ascii_alphabetic() {
            // crude: use single-letter English letter names via dict if present
            let name = ch.to_ascii_uppercase().to_string();
            if let Some(syls) = ENG_DICT.get(&name) {
                let (p, t) = refine_syllables(syls);
                phns.extend(p);
                tns.extend(t);
            } else {
                phns.push("UNK".into());
                tns.push(0);
            }
        } else {
            let s = ch.to_string();
            phns.push(post_replace_ph(&s));
            tns.push(0);
        }
    }
    if phns.is_empty() {
        phns.push(post_replace_ph(word));
        tns.push(0);
    }
    (phns, tns)
}

fn tokenize_word(word: &str) -> Vec<String> {
    SPM.encode(word).unwrap_or_default()
}

pub fn encode_bert_ids(text: &str) -> Vec<i32> {
    let ids = SPM.encode_to_ids(text).unwrap_or_default();
    let mut out = Vec::with_capacity(ids.len() + 2);
    out.push(1); // CLS
    out.extend(ids.into_iter().map(|x| x as i32));
    out.push(2); // SEP
    out
}

pub struct G2pOut {
    pub phones: Vec<String>,
    pub tones: Vec<i32>,
    pub word2ph: Vec<i32>,
}

pub fn g2p_english(text: &str) -> G2pOut {
    let words = sep_text(text);
    let tokens: Vec<Vec<String>> = words.iter().map(|w| tokenize_word(w)).collect();

    let mut phones_w: Vec<Vec<String>> = Vec::new();
    let mut tones_w: Vec<Vec<i32>> = Vec::new();

    for word in &words {
        let key = word.to_ascii_uppercase();
        if let Some(syls) = ENG_DICT.get(&key) {
            let (phns, tns) = refine_syllables(syls);
            phones_w.push(phns.into_iter().map(|p| post_replace_ph(&p)).collect());
            tones_w.push(tns);
        } else if word.len() == 1 && !word.chars().next().unwrap().is_ascii_alphanumeric() {
            phones_w.push(vec![post_replace_ph(word)]);
            tones_w.push(vec![0]);
        } else {
            // Try ARPA-like from dict of whole word without punctuation
            let cleaned: String = word
                .chars()
                .filter(|c| c.is_ascii_alphabetic() || *c == '\'')
                .collect();
            if let Some(syls) = ENG_DICT.get(&cleaned.to_ascii_uppercase()) {
                let (phns, tns) = refine_syllables(syls);
                phones_w.push(phns.into_iter().map(|p| post_replace_ph(&p)).collect());
                tones_w.push(tns);
            } else {
                let (phns, tns) = fallback_phones(word);
                phones_w.push(phns.into_iter().map(|p| post_replace_ph(&p)).collect());
                tones_w.push(tns);
            }
        }
    }

    let mut word2ph = Vec::new();
    for (token, phoneme) in tokens.iter().zip(phones_w.iter()) {
        word2ph.extend(distribute_phone(phoneme.len(), token.len().max(1)));
    }

    let mut phones = vec!["_".to_string()];
    let mut tones = vec![0i32];
    for (p, t) in phones_w.into_iter().zip(tones_w.into_iter()) {
        phones.extend(p);
        tones.extend(t);
    }
    phones.push("_".into());
    tones.push(0);
    word2ph.insert(0, 1);
    word2ph.push(1);

    assert_eq!(phones.len(), tones.len());
    assert_eq!(phones.len() as i32, word2ph.iter().sum());

    G2pOut {
        phones,
        tones,
        word2ph,
    }
}

pub fn cleaned_text_to_sequence(phones: &[String], tones: &[i32]) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let phone_ids: Vec<i32> = phones
        .iter()
        .map(|s| *SYMBOL_TO_ID.get(s.as_str()).unwrap_or(&SYMBOL_TO_ID["UNK"]))
        .collect();
    let tones: Vec<i32> = tones.iter().map(|t| t + EN_TONE_START).collect();
    let lang: Vec<i32> = vec![LANG_EN; phone_ids.len()];
    (phone_ids, tones, lang)
}

pub fn intersperse(lst: &[i32], item: i32) -> Vec<i32> {
    let mut result = vec![item; lst.len() * 2 + 1];
    for (i, v) in lst.iter().enumerate() {
        result[i * 2 + 1] = *v;
    }
    result
}

/// Full frontend tensors for Deberta + evil acoustics (with blanks).
pub struct Prepared {
    pub input_ids: Vec<i32>,
    pub phones: Vec<i32>,
    pub tones: Vec<i32>,
    pub language: Vec<i32>,
    pub word2ph: Vec<i32>,
}

pub fn prepare(text: &str) -> Prepared {
    let norm = crate::normalize::text_normalize(text);
    let g = g2p_english(&norm);
    let input_ids = encode_bert_ids(&norm);
    let (phones, tones, language) = cleaned_text_to_sequence(&g.phones, &g.tones);
    let phones = intersperse(&phones, 0);
    let tones = intersperse(&tones, 0);
    let language = intersperse(&language, 0);
    let mut word2ph: Vec<i32> = g.word2ph.iter().map(|n| n * 2).collect();
    if !word2ph.is_empty() {
        word2ph[0] += 1;
    }
    assert_eq!(word2ph.iter().sum::<i32>() as usize, phones.len());
    Prepared {
        input_ids,
        phones,
        tones,
        language,
        word2ph,
    }
}

#[allow(dead_code)]
fn _arpa_touch() {
    let _ = ARPA.len();
}
