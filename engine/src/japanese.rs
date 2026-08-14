//! Bert-VITS2 V220 Japanese G2P (OpenJTalk-compatible via jpreprocess + NAIST-JDIC).
//! Omits num2words / alpha-symbol expansions from upstream Python.

use crate::symbols::post_replace_ph;
use jlabel::Label;
use jpreprocess::kind::JPreprocessDictionaryKind;
use jpreprocess::{DefaultTokenizer, JPreprocess, SystemDictionaryConfig};
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

const JP_VOCAB_TXT: &str = include_str!("../assets/jp_vocab.txt");
const JP_CLS: i32 = 1;
const JP_SEP: i32 = 2;
const JP_UNK: i32 = 3;

const PUNCT: &[&str] = &["!", "?", "…", ",", ".", "'", "-"];

static JP: Lazy<JPreprocess<DefaultTokenizer>> = Lazy::new(|| {
    let system = SystemDictionaryConfig::Bundled(JPreprocessDictionaryKind::NaistJdic)
        .load()
        .expect("load naist-jdic");
    JPreprocess::with_dictionaries(system, None)
});

static MARKS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"[^A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]",
    )
    .unwrap()
});

static JP_VOCAB: Lazy<HashMap<String, i32>> = Lazy::new(|| {
    let mut m = HashMap::new();
    for (i, line) in JP_VOCAB_TXT.lines().enumerate() {
        m.insert(line.to_string(), i as i32);
    }
    m
});

#[derive(Debug, Clone)]
struct FrontendNode {
    string: String,
    pron: String,
}

fn hira2kata(text: &str) -> String {
    text.chars()
        .map(|c| {
            if ('\u{3041}'..='\u{3096}').contains(&c) {
                char::from_u32(c as u32 + 0x60).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

fn replace_punctuation(text: &str) -> String {
    let rep: &[(&str, &str)] = &[
        ("：", ","),
        ("；", ","),
        ("，", ","),
        ("。", "."),
        ("！", "!"),
        ("？", "?"),
        ("\n", "."),
        ("．", "."),
        ("…", "..."),
        ("···", "..."),
        ("・・・", "..."),
        ("·", ","),
        ("・", ","),
        ("、", ","),
        ("$", "."),
        ("“", "'"),
        ("”", "'"),
        ("\"", "'"),
        ("‘", "'"),
        ("’", "'"),
        ("（", "'"),
        ("）", "'"),
        ("(", "'"),
        (")", "'"),
        ("《", "'"),
        ("》", "'"),
        ("【", "'"),
        ("】", "'"),
        ("[", "'"),
        ("]", "'"),
        ("—", "-"),
        ("−", "-"),
        ("～", "-"),
        ("~", "-"),
        ("「", "'"),
        ("」", "'"),
    ];
    let mut out = text.to_string();
    for (a, b) in rep {
        out = out.replace(a, b);
    }
    let keep: String = out
        .chars()
        .filter(|c| {
            PUNCT.contains(&c.to_string().as_str())
                || matches!(*c, '…')
                || ('\u{3040}'..='\u{309f}').contains(c)
                || ('\u{30a0}'..='\u{30ff}').contains(c)
                || ('\u{4e00}'..='\u{9fff}').contains(c)
                || ('\u{3400}'..='\u{4dbf}').contains(c)
                || *c == '\u{3005}'
        })
        .collect();
    keep
}

fn an2jp_simple(text: &str) -> String {
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
    let nfkc: String = text.nfkc().collect();
    let s = an2jp_simple(&nfkc);
    replace_punctuation(&s).replace('゙', "")
}

fn parse_frontend_line(line: &str) -> FrontendNode {
    let parts: Vec<&str> = line.split(',').collect();
    let string = parts.first().unwrap_or(&"").to_string();
    let pron = parts.get(9).unwrap_or(&"").to_string();
    FrontendNode { string, pron }
}

fn is_symbol_token(w: &str) -> bool {
    matches!(w, "・" | "、" | "。" | "？" | "！")
}

fn text2sep_kata(norm: &str) -> (Vec<String>, Vec<String>, Vec<(String, i32)>) {
    let lines = JP.run_frontend(norm).unwrap_or_default();
    let parsed: Vec<FrontendNode> = lines.iter().map(|l| parse_frontend_line(l)).collect();

    let mut res = Vec::new();
    let mut sep = Vec::new();
    for parts in parsed {
        let word = replace_punctuation(&parts.string);
        let mut yomi = parts.pron.replace('’', "");
        if !yomi.is_empty() {
            if MARKS.is_match(&yomi) {
                if word.chars().count() > 1 {
                    for ch in word.chars() {
                        let w = replace_punctuation(&ch.to_string());
                        res.push(w.clone());
                        sep.push(w);
                    }
                    continue;
                }
                yomi = word.clone();
            }
            res.push(yomi);
        } else if is_symbol_token(&word) {
            res.push(word.clone());
        } else if word == "っ" || word == "ッ" {
            res.push("ッ".into());
        } else if matches!(word.as_str(), "「" | "」" | "『" | "』" | "―" | "（" | "）" | "［" | "］" | "[" | "]") {
            // skip
        } else {
            res.push(word.clone());
        }
        sep.push(word);
    }

    let kata: Vec<String> = res.iter().map(|i| hira2kata(i)).collect();
    let acc = get_accent(norm);
    (sep, kata, acc)
}

fn get_accent(text: &str) -> Vec<(String, i32)> {
    let labels: Vec<Label> = match JP.extract_fullcontext(text) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let labels: Vec<String> = labels.iter().map(|l| l.to_string()).collect();
    let ph_re = Regex::new(r"-([^+]*)\+").unwrap();
    let a1_re = Regex::new(r"/A:(-?[0-9]+)\+").unwrap();
    let a2_re = Regex::new(r"\+(\d+)\+").unwrap();

    let mut phonemes = Vec::new();
    let mut accents = Vec::new();
    for (n, label) in labels.iter().enumerate() {
        let Some(ph_cap) = ph_re.captures(label) else { continue };
        let phoneme = ph_cap.get(1).unwrap().as_str();
        if phoneme == "sil" || phoneme == "pau" {
            continue;
        }
        phonemes.push(phoneme.replace("cl", "q").to_ascii_lowercase());
        let a1 = a1_re
            .captures(label)
            .and_then(|c| c.get(1))
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);
        let a2 = a2_re
            .captures(label)
            .and_then(|c| c.get(1))
            .and_then(|m| m.as_str().parse().ok())
            .unwrap_or(0);
        let a2_next = if n + 1 < labels.len() {
            let next_ph = ph_re
                .captures(&labels[n + 1])
                .and_then(|c| c.get(1))
                .map(|m| m.as_str())
                .unwrap_or("");
            if next_ph == "sil" || next_ph == "pau" {
                -1
            } else {
                a2_re
                    .captures(&labels[n + 1])
                    .and_then(|c| c.get(1))
                    .and_then(|m| m.as_str().parse().ok())
                    .unwrap_or(-1)
            }
        } else {
            -1
        };
        let tone = if a1 == 0 && a2_next == a2 + 1 {
            -1
        } else if a2 == 1 && a2_next == 2 {
            1
        } else {
            0
        };
        accents.push(tone);
    }
    phonemes.into_iter().zip(accents).collect()
}

fn kata2phoneme(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return Vec::new();
    }
    if text == "ー" {
        return vec!["ー".into()];
    }
    if text.starts_with('ー') {
        let rest: String = text.chars().skip(1).collect();
        let mut v = vec!["ー".into()];
        v.extend(kata2phoneme(&rest));
        return v;
    }
    if MARKS.is_match(text) {
        return vec![post_replace_ph(text)];
    }
    get_accent(text)
        .into_iter()
        .map(|(p, _)| post_replace_ph(&p))
        .collect()
}

fn handle_long(sep_phonemes: &mut [Vec<String>]) {
    for i in 0..sep_phonemes.len() {
        if sep_phonemes[i].first().map(|s| s.as_str()) == Some("ー") {
            if i > 0 {
                if let Some(last) = sep_phonemes[i - 1].last() {
                    sep_phonemes[i][0] = last.clone();
                }
            }
        }
        for j in 0..sep_phonemes[i].len() {
            if sep_phonemes[i][j] == "ー" {
                if j > 0 {
                    sep_phonemes[i][j] = sep_phonemes[i][j - 1].clone();
                }
            }
        }
    }
}

fn align_tones(phones: &[Vec<String>], mut tones: Vec<(String, i32)>) -> Vec<i32> {
    let mut out = Vec::new();
    for pho in phones {
        let mut temp = vec![0i32; pho.len()];
        for (idx, p) in pho.iter().enumerate() {
            if tones.is_empty() {
                break;
            }
            if p == &tones[0].0 {
                temp[idx] = tones[0].1;
                if idx > 0 {
                    temp[idx] += temp[idx - 1];
                }
                tones.remove(0);
            }
        }
        let mut temp = std::iter::once(0).chain(temp.into_iter()).collect::<Vec<_>>();
        temp.pop();
        if temp.iter().any(|&x| x < 0) {
            temp = temp.into_iter().map(|x| x + 1).collect();
        }
        out.extend(temp);
    }
    out
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

fn tokenize_word(word: &str) -> Vec<String> {
    if PUNCT.contains(&word) {
        vec![word.to_string()]
    } else {
        word.chars().map(|c| c.to_string()).collect()
    }
}

pub fn encode_jp_bert_ids(norm: &str) -> Vec<i32> {
    let mut out = Vec::with_capacity(norm.chars().count() + 2);
    out.push(JP_CLS);
    for ch in norm.chars() {
        let s = ch.to_string();
        let id = JP_VOCAB.get(&s).copied().unwrap_or(JP_UNK);
        out.push(id);
    }
    out.push(JP_SEP);
    out
}

pub struct JpG2p {
    pub phones: Vec<String>,
    pub tones: Vec<i32>,
    pub word2ph: Vec<i32>,
    pub norm_text: String,
    pub bert_text: String,
}

pub fn g2p_japanese(text: &str) -> JpG2p {
    let norm = text_normalize(text);
    let (sep_text, sep_kata, acc) = text2sep_kata(&norm);
    let bert_text: String = sep_text.join("");

    let sep_tokenized: Vec<Vec<String>> = sep_text.iter().map(|w| tokenize_word(w)).collect();
    let mut sep_phonemes: Vec<Vec<String>> = sep_kata.iter().map(|k| kata2phoneme(k)).collect();
    handle_long(&mut sep_phonemes);

    let mut tones = align_tones(&sep_phonemes, acc);

    let mut word2ph = Vec::new();
    for (token, phoneme) in sep_tokenized.iter().zip(sep_phonemes.iter()) {
        word2ph.extend(distribute_phone(phoneme.len(), token.len().max(1)));
    }

    let mut phones = vec!["_".to_string()];
    tones.insert(0, 0);
    for p in &sep_phonemes {
        for ph in p {
            phones.push(post_replace_ph(ph));
        }
    }
    phones.push("_".into());
    tones.push(0);
    word2ph.insert(0, 1);
    word2ph.push(1);

    debug_assert_eq!(phones.len(), tones.len());
    debug_assert_eq!(phones.len() as i32, word2ph.iter().sum::<i32>());

    JpG2p {
        phones,
        tones,
        word2ph,
        norm_text: norm,
        bert_text,
    }
}
