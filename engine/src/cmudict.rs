//! CMUdict loader from embedded `cmudict.rep`.

use once_cell::sync::Lazy;
use std::collections::HashMap;

const CMUDICT_BYTES: &[u8] = include_bytes!("../assets/cmudict.rep");

/// word (UPPER) -> list of syllables, each syllable is list of ARPAbet phones
pub type EngDict = HashMap<String, Vec<Vec<String>>>;

pub static ENG_DICT: Lazy<EngDict> = Lazy::new(load_cmudict);

fn load_cmudict() -> EngDict {
    let text = std::str::from_utf8(CMUDICT_BYTES).expect("cmudict utf8");
    let mut dict = EngDict::new();
    for (line_index, line) in text.lines().enumerate() {
        if line_index < 48 {
            continue; // Python used line_index < 49 with 1-based → skip first 48 lines
        }
        let line = line.trim();
        if line.is_empty() || line.starts_with("##") {
            continue;
        }
        let Some((word, rest)) = line.split_once("  ") else {
            continue;
        };
        let syllables: Vec<Vec<String>> = rest
            .split(" - ")
            .map(|syl| syl.split_whitespace().map(|s| s.to_string()).collect())
            .collect();
        dict.insert(word.to_string(), syllables);
    }
    dict
}
