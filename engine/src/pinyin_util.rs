//! Per-character initials + FINALS_TONE3 (pypinyin-compatible) for ToneSandhi / G2P.

use crate::symbols::punctuation;
use once_cell::sync::Lazy;
use std::collections::HashMap;

static PINYIN_MAP: Lazy<HashMap<char, (String, String)>> = Lazy::new(|| {
    let raw: HashMap<String, [String; 2]> =
        serde_json::from_str(include_str!("../assets/pypinyin_map.json")).expect("pypinyin_map");
    raw.into_iter()
        .filter_map(|(k, v)| {
            let ch = k.chars().next()?;
            Some((ch, (v[0].clone(), v[1].clone())))
        })
        .collect()
});

pub fn char_initial_final(ch: char) -> (String, String) {
    if punctuation().iter().any(|p| p.chars().next() == Some(ch)) {
        let s = ch.to_string();
        return (s.clone(), s);
    }
    if let Some(pair) = PINYIN_MAP.get(&ch) {
        return pair.clone();
    }
    (String::new(), "5".to_string())
}

pub fn word_initials_finals(word: &str) -> (Vec<String>, Vec<String>) {
    let mut initials = Vec::new();
    let mut finals = Vec::new();
    for ch in word.chars() {
        let (i, f) = char_initial_final(ch);
        initials.push(i);
        finals.push(f);
    }
    (initials, finals)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pypinyin_samples() {
        let cases: [(char, (String, String)); 6] = [
            ('你', ("n".into(), "i3".into())),
            ('好', ("h".into(), "ao3".into())),
            ('一', ("".into(), "i1".into())),
            ('个', ("g".into(), "e4".into())),
            ('二', ("".into(), "er4".into())),
            ('千', ("q".into(), "ian1".into())),
        ];
        for (ch, (ei, ef)) in cases {
            let (i, f) = char_initial_final(ch);
            assert_eq!(i, ei, "initial for {ch}");
            assert_eq!(f, ef, "final for {ch}");
        }
    }
}
