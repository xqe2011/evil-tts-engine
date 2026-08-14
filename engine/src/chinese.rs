//! Simplified V220 Chinese G2P (opencpop + pinyin).
//! Omits jieba POS / ToneSandhi; quality is slightly lower than Bert-VITS2 Python.

use crate::symbols::post_replace_ph;
use once_cell::sync::Lazy;
use pinyin::ToPinyin;
use std::collections::HashMap;

const OPENCPOP: &str = include_str!("../assets/opencpop-strict.txt");

static PINYIN_TO_SYMBOL: Lazy<HashMap<&'static str, Vec<&'static str>>> = Lazy::new(|| {
    let mut m = HashMap::new();
    for line in OPENCPOP.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split('\t');
        let Some(py) = parts.next() else { continue };
        let Some(syms) = parts.next() else { continue };
        m.insert(py, syms.split_whitespace().collect());
    }
    m
});

const PUNCT: &[char] = &['!', '?', '…', ',', '.', '\'', '-'];

fn replace_punctuation(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        let r = match c {
            '嗯' => {
                out.push('恩');
                continue;
            }
            '呣' => {
                out.push('母');
                continue;
            }
            '：' | '；' | '，' | '、' | '·' => ",",
            '。' | '\n' | '$' => ".",
            '！' => "!",
            '？' => "?",
            '…' => "…",
            '“' | '”' | '"' | '‘' | '’' | '（' | '）' | '(' | ')' | '《' | '》' | '【' | '】'
            | '[' | ']' | '「' | '」' => "'",
            '—' | '～' | '~' => "-",
            _ => {
                out.push(c);
                continue;
            }
        };
        out.push_str(r);
    }
    // drop non-CJK / non-punct
    out.chars()
        .filter(|c| {
            PUNCT.contains(c)
                || ('\u{4e00}'..='\u{9fff}').contains(c)
                || *c == '…'
        })
        .collect()
}

/// Very small digit→hanzi (enough for short prompts).
fn an2cn_simple(text: &str) -> String {
    const DIGITS: &[char] = &['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
    let mut out = String::new();
    let mut num = String::new();
    let flush = |num: &mut String, out: &mut String| {
        if num.is_empty() {
            return;
        }
        for ch in num.chars() {
            if let Some(d) = ch.to_digit(10) {
                out.push(DIGITS[d as usize]);
            }
        }
        num.clear();
    };
    for c in text.chars() {
        if c.is_ascii_digit() {
            num.push(c);
        } else {
            flush(&mut num, &mut out);
            out.push(c);
        }
    }
    flush(&mut num, &mut out);
    out
}

pub fn text_normalize(text: &str) -> String {
    replace_punctuation(&an2cn_simple(text))
}

fn syllable_phones(plain: &str, tone: i32) -> Option<(Vec<String>, Vec<i32>)> {
    let syms = PINYIN_TO_SYMBOL.get(plain)?;
    let phones: Vec<String> = syms.iter().map(|s| (*s).to_string()).collect();
    let tones = vec![tone; phones.len()];
    Some((phones, tones))
}

fn char_to_phones(ch: char) -> (Vec<String>, Vec<i32>) {
    let s = ch.to_string();
    if PUNCT.contains(&ch) || ch == '…' {
        let ph = post_replace_ph(&s);
        return (vec![ph], vec![0]);
    }
    // Single-char pinyin
    let mut it = s.as_str().to_pinyin();
    let Some(Some(py)) = it.next() else {
        return (vec!["UNK".into()], vec![0]);
    };
    let with_tone = py.with_tone_num_end();
    let (plain, tone) = if let Some(last) = with_tone.chars().last() {
        if last.is_ascii_digit() {
            let tone = (last as u8 - b'0') as i32;
            let plain = &with_tone[..with_tone.len() - 1];
            (plain.to_string(), if tone == 0 { 5 } else { tone })
        } else {
            // neutral / missing tone → 5
            (with_tone.to_string(), 5)
        }
    } else {
        return (vec!["UNK".into()], vec![0]);
    };

    if let Some(out) = syllable_phones(&plain, tone) {
        return out;
    }
    // Fallback: try ü/v variants used by opencpop (e.g. lve, nve)
    let alt = plain.replace('ü', "v").replace("u:", "v");
    if alt != plain {
        if let Some(out) = syllable_phones(&alt, tone) {
            return out;
        }
    }
    (vec!["UNK".into()], vec![0])
}

pub struct ZhG2p {
    pub phones: Vec<String>,
    pub tones: Vec<i32>,
    pub word2ph: Vec<i32>,
    pub norm_text: String,
}

pub fn g2p_chinese(text: &str) -> ZhG2p {
    let norm = text_normalize(text);
    let mut phones = Vec::new();
    let mut tones = Vec::new();
    let mut word2ph = Vec::new();

    for ch in norm.chars() {
        let (p, t) = char_to_phones(ch);
        word2ph.push(p.len() as i32);
        phones.extend(p);
        tones.extend(t);
    }

    phones.insert(0, "_".into());
    tones.insert(0, 0);
    word2ph.insert(0, 1);
    phones.push("_".into());
    tones.push(0);
    word2ph.push(1);

    debug_assert_eq!(phones.len(), tones.len());
    debug_assert_eq!(phones.len() as i32, word2ph.iter().sum::<i32>());
    // Bert-VITS2: len(word2ph) == len(text) + 2
    debug_assert_eq!(word2ph.len(), norm.chars().count() + 2);

    ZhG2p {
        phones,
        tones,
        word2ph,
        norm_text: norm,
    }
}
