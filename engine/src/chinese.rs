//! Bert-VITS2 V220 Chinese G2P: jieba POS + ToneSandhi + opencpop mapping.

use crate::jieba_dict::JIEBA;
use crate::pinyin_util::word_initials_finals;
use crate::symbols::{post_replace_ph, punctuation};
use crate::tone_sandhi::ToneSandhi;
use chinese_number::{ChineseCase, ChineseCountMethod, ChineseVariant, NumberToChinese};
use once_cell::sync::Lazy;
use regex::Regex;
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

static NUM_RE: Lazy<regex::Regex> = Lazy::new(|| regex::Regex::new(r"\d+(?:\.\d+)?").unwrap());

fn split_sentences(text: &str) -> Vec<String> {
    let punc: std::collections::HashSet<char> =
        punctuation().iter().flat_map(|s| s.chars()).collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if punc.contains(&ch) {
            if !cur.trim().is_empty() {
                out.push(cur.clone());
            }
            cur.clear();
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

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
    let allowed: String = punctuation().iter().copied().collect();
    out.chars()
        .filter(|c| {
            allowed.contains(*c)
                || ('\u{4e00}'..='\u{9fff}').contains(c)
                || *c == '…'
        })
        .collect()
}

fn an2cn(num_str: &str) -> String {
    if num_str.contains('.') {
        let parts: Vec<&str> = num_str.splitn(2, '.').collect();
        let int_part = parts[0].parse::<u64>().ok().and_then(|n| {
            n.to_chinese(
                ChineseVariant::Simple,
                ChineseCase::Lower,
                ChineseCountMethod::TenThousand,
            )
            .ok()
        });
        let frac: String = parts
            .get(1)
            .unwrap_or(&"")
            .chars()
            .filter(|c| c.is_ascii_digit())
            .map(digit_char)
            .collect();
        match int_part {
            Some(i) if frac.is_empty() => i,
            Some(i) => format!("{i}点{frac}"),
            None => num_str.to_string(),
        }
    } else if let Ok(n) = num_str.parse::<u64>() {
        n.to_chinese(
            ChineseVariant::Simple,
            ChineseCase::Lower,
            ChineseCountMethod::TenThousand,
        )
        .unwrap_or_else(|_| num_str.to_string())
    } else {
        num_str.to_string()
    }
}

fn digit_char(c: char) -> char {
    match c {
        '0' => '零',
        '1' => '一',
        '2' => '二',
        '3' => '三',
        '4' => '四',
        '5' => '五',
        '6' => '六',
        '7' => '七',
        '8' => '八',
        '9' => '九',
        _ => c,
    }
}

pub fn text_normalize(text: &str) -> String {
    let mut text = text.to_string();
    let matches: Vec<_> = NUM_RE.find_iter(&text).map(|m| (m.start(), m.end(), m.as_str().to_string())).collect();
    let mut offset = 0i32;
    for (start, end, num) in matches {
        let cn = an2cn(&num);
        let s = (start as i32 + offset) as usize;
        let e = (end as i32 + offset) as usize;
        text.replace_range(s..e, &cn);
        offset += cn.len() as i32 - num.len() as i32;
    }
    replace_punctuation(&text)
}

fn finals_to_phones(c: &str, v: &str, word2ph: &mut Vec<i32>) -> (Vec<String>, Vec<i32>) {
    let punc: Vec<&str> = punctuation().to_vec();
    if c == v && c.chars().count() == 1 {
        let ch = c.chars().next().unwrap();
        if punc.iter().any(|p| p.chars().next() == Some(ch)) || punc.contains(&c) {
            let ph = post_replace_ph(c);
            word2ph.push(1);
            return (vec![ph], vec![0]);
        }
    }
    let v_without_tone = &v[..v.len().saturating_sub(1)];
    let tone = v.chars().last().unwrap_or('5');
    let mut pinyin = format!("{c}{v_without_tone}");
    if !c.is_empty() {
        let v_rep = [("uei", "ui"), ("iou", "iu"), ("uen", "un")];
        for (from, to) in v_rep {
            if v_without_tone == from {
                pinyin = format!("{c}{to}");
            }
        }
    } else {
        let pinyin_rep = [("ing", "ying"), ("i", "yi"), ("in", "yin"), ("u", "wu")];
        if let Some(&(_, rep)) = pinyin_rep.iter().find(|(k, _)| *k == pinyin.as_str()) {
            pinyin = rep.to_string();
        } else if let Some(first) = pinyin.chars().next() {
            let single_rep = [('v', "yu"), ('e', "e"), ('i', "y"), ('u', "w")];
            if let Some((_, rep)) = single_rep.iter().find(|(k, _)| *k == first) {
                pinyin = format!("{rep}{}", &pinyin[first.len_utf8()..]);
            }
        }
    }
    if let Some(syms) = PINYIN_TO_SYMBOL.get(pinyin.as_str()) {
        let phones: Vec<String> = syms.iter().map(|s| (*s).to_string()).collect();
        let tone_i: i32 = tone.to_digit(10).unwrap_or(5) as i32;
        let n = phones.len();
        word2ph.push(n as i32);
        (phones, vec![tone_i; n])
    } else {
        word2ph.push(1);
        (vec!["UNK".into()], vec![0])
    }
}

fn g2p_segment(seg: &str) -> (Vec<String>, Vec<i32>, Vec<i32>) {
    let seg = Regex::new("[a-zA-Z]+").unwrap().replace_all(seg, "");
    let seg = seg.as_ref();
    let tagged = JIEBA.tag(seg, true);
    let mut seg_cut: Vec<(String, String)> = tagged
        .into_iter()
        .map(|t| (t.word.to_string(), t.tag.to_string()))
        .collect();
    seg_cut = ToneSandhi::pre_merge_for_modify(seg_cut);

    let mut phones_list = Vec::new();
    let mut tones_list = Vec::new();
    let mut word2ph = Vec::new();

    for (word, pos) in seg_cut {
        if pos == "eng" {
            continue;
        }
        let (initials, mut finals) = word_initials_finals(&word);
        finals = ToneSandhi::modified_tone(&word, &pos, finals);
        for (c, v) in initials.iter().zip(finals.iter()) {
            let (phone, tone) = finals_to_phones(c, v, &mut word2ph);
            phones_list.extend(phone);
            tones_list.extend(tone);
        }
    }
    (phones_list, tones_list, word2ph)
}

pub struct ZhG2p {
    pub phones: Vec<String>,
    pub tones: Vec<i32>,
    pub word2ph: Vec<i32>,
    pub norm_text: String,
}

pub fn g2p_chinese(text: &str) -> ZhG2p {
    let norm = text_normalize(text);
    let sentences = split_sentences(&norm);

    let mut phones = Vec::new();
    let mut tones = Vec::new();
    let mut word2ph = Vec::new();

    for seg in &sentences {
        let (p, t, w) = g2p_segment(seg);
        phones.extend(p);
        tones.extend(t);
        word2ph.extend(w);
    }

    phones.insert(0, "_".into());
    tones.insert(0, 0);
    word2ph.insert(0, 1);
    phones.push("_".into());
    tones.push(0);
    word2ph.push(1);

    debug_assert_eq!(phones.len(), tones.len());
    debug_assert_eq!(phones.len() as i32, word2ph.iter().sum::<i32>());
    debug_assert_eq!(word2ph.len(), norm.chars().count() + 2);

    ZhG2p {
        phones,
        tones,
        word2ph,
        norm_text: norm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parity(text: &str) -> ZhG2p {
        g2p_chinese(text)
    }

    #[test]
    fn norm_2024() {
        assert_eq!(text_normalize("2024年"), "二千零二十四年");
    }

    #[test]
    fn greeting_tones() {
        let g = parity("你好，我是助手。");
        assert_eq!(g.norm_text, "你好,我是助手.");
        assert_eq!(g.tones, vec![0, 2, 2, 3, 3, 0, 3, 3, 4, 4, 4, 4, 3, 3, 0, 0]);
    }

    #[test]
    fn yige_pingguo() {
        let g = parity("一个苹果");
        assert_eq!(g.tones, vec![0, 2, 2, 5, 5, 2, 2, 3, 3, 0]);
    }

    #[test]
    fn year_2024_phones() {
        let g = parity("2024年");
        assert_eq!(
            g.phones,
            vec![
                "_", "EE", "er", "q", "ian", "l", "ing", "EE", "er", "sh", "ir", "s", "i0", "n",
                "ian", "_"
            ]
        );
    }
}
