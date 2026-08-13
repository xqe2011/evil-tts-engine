//! V220 symbol tables (must match Bert-VITS2 oldVersion/V220/text/symbols.py).

use once_cell::sync::Lazy;
use std::collections::{HashMap, HashSet};

pub const NUM_ZH_TONES: i32 = 6;
pub const NUM_JA_TONES: i32 = 2;
pub const EN_TONE_START: i32 = NUM_ZH_TONES + NUM_JA_TONES; // 8
pub const LANG_EN: i32 = 2;

pub static SYMBOL_TO_ID: Lazy<HashMap<&'static str, i32>> = Lazy::new(|| {
    let punctuation = ["!", "?", "…", ",", ".", "'", "-"];
    let pu_symbols: Vec<&str> = punctuation
        .iter()
        .copied()
        .chain(["SP", "UNK"])
        .collect();
    let pad = "_";
    let zh_symbols = [
        "E", "En", "a", "ai", "an", "ang", "ao", "b", "c", "ch", "d", "e", "ei", "en", "eng",
        "er", "f", "g", "h", "i", "i0", "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong",
        "ir", "iu", "j", "k", "l", "m", "n", "o", "ong", "ou", "p", "q", "r", "s", "sh", "t",
        "u", "ua", "uai", "uan", "uang", "ui", "un", "uo", "v", "van", "ve", "vn", "w", "x",
        "y", "z", "zh", "AA", "EE", "OO",
    ];
    let ja_symbols = [
        "N", "a", "a:", "b", "by", "ch", "d", "dy", "e", "e:", "f", "g", "gy", "h", "hy", "i",
        "i:", "j", "k", "ky", "m", "my", "n", "ny", "o", "o:", "p", "py", "q", "r", "ry", "s",
        "sh", "t", "ts", "ty", "u", "u:", "w", "y", "z", "zy",
    ];
    let en_symbols = [
        "aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh", "eh", "er", "ey", "f", "g",
        "hh", "ih", "iy", "jh", "k", "l", "m", "n", "ng", "ow", "oy", "p", "r", "s", "sh", "t",
        "th", "uh", "uw", "V", "w", "y", "z", "zh",
    ];
    let mut normal: Vec<&str> = zh_symbols
        .iter()
        .chain(ja_symbols.iter())
        .chain(en_symbols.iter())
        .copied()
        .collect();
    normal.sort_unstable();
    normal.dedup();
    let mut symbols = Vec::with_capacity(1 + normal.len() + pu_symbols.len());
    symbols.push(pad);
    symbols.extend(normal);
    symbols.extend(pu_symbols);
    symbols
        .into_iter()
        .enumerate()
        .map(|(i, s)| (s, i as i32))
        .collect()
});

pub static ARPA: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "AH0", "S", "AH1", "EY2", "AE2", "EH0", "OW2", "UH0", "NG", "B", "G", "AY0", "M", "AA0",
        "F", "AO0", "ER2", "UH1", "IY1", "AH2", "DH", "IY0", "EY1", "IH0", "K", "N", "W", "IY2",
        "T", "AA1", "ER1", "EH2", "OY0", "UH2", "UW1", "Z", "AW2", "AW1", "V", "UW2", "AA2", "ER",
        "AW0", "UW0", "R", "OW1", "EH1", "ZH", "AE0", "IH2", "IH", "Y", "JH", "P", "AY1", "EY0",
        "OY2", "TH", "HH", "D", "ER0", "CH", "AO1", "AE1", "AO2", "OY1", "AY2", "IH1", "OW0",
        "L", "SH",
    ]
    .into_iter()
    .collect()
});

pub fn post_replace_ph(ph: &str) -> String {
    let mapped = match ph {
        "：" | "；" | "，" | "、" | "·" => ",",
        "。" | "\n" => ".",
        "！" => "!",
        "？" => "?",
        "…" | "···" | "・・・" => "...",
        "v" => "V",
        other => other,
    };
    if SYMBOL_TO_ID.contains_key(mapped) {
        mapped.to_string()
    } else {
        "UNK".to_string()
    }
}
