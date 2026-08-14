//! Bert-VITS2 V220 ToneSandhi (ported from text/tone_sandhi.py).

use crate::jieba_dict::JIEBA;
use crate::pinyin_util::word_initials_finals;
use once_cell::sync::Lazy;
use std::collections::HashSet;

static MUST_NEURAL: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    include_str!("../assets/must_neural_tone_words.txt")
        .lines()
        .filter(|l| !l.is_empty())
        .collect()
});

static MUST_NOT_NEURAL: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    include_str!("../assets/must_not_neural_tone_words.txt")
        .lines()
        .filter(|l| !l.is_empty())
        .collect()
});

const PUNC: &str = "：，；。？！\"\"''':,;.?!";

pub type Seg = Vec<(String, String)>;

pub struct ToneSandhi;

impl ToneSandhi {
    pub fn pre_merge_for_modify(seg: Seg) -> Seg {
        let seg = Self::merge_bu(seg);
        let seg = Self::merge_yi(seg);
        let seg = Self::merge_reduplication(seg);
        let seg = Self::merge_continuous_three_tones(seg);
        let seg = Self::merge_continuous_three_tones_2(seg);
        Self::merge_er(seg)
    }

    pub fn modified_tone(word: &str, pos: &str, mut finals: Vec<String>) -> Vec<String> {
        finals = Self::bu_sandhi(word, finals);
        finals = Self::yi_sandhi(word, finals);
        finals = Self::neural_sandhi(word, pos, finals);
        Self::three_sandhi(word, finals)
    }

    fn neural_sandhi(word: &str, pos: &str, mut finals: Vec<String>) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        for j in 0..chars.len() {
            if j > 0
                && chars[j] == chars[j - 1]
                && pos.chars().next().is_some_and(|c| "nva".contains(c))
                && !MUST_NOT_NEURAL.contains(word)
            {
                finals[j] = replace_tone(&finals[j], '5');
            }
        }
        if let Some(&last) = chars.last() {
            if "吧呢啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶".contains(last) {
                let i = finals.len() - 1;
                finals[i] = replace_tone(&finals[i], '5');
            } else if "的地得".contains(last) {
                let i = finals.len() - 1;
                finals[i] = replace_tone(&finals[i], '5');
            } else if chars.len() > 1
                && "们子".contains(last)
                && (pos == "r" || pos.starts_with('n'))
                && !MUST_NOT_NEURAL.contains(word)
            {
                let i = finals.len() - 1;
                finals[i] = replace_tone(&finals[i], '5');
            } else if chars.len() > 1
                && "上下里".contains(last)
                && (pos == "s" || pos == "l" || pos == "f")
            {
                let i = finals.len() - 1;
                finals[i] = replace_tone(&finals[i], '5');
            } else if chars.len() > 1
                && "来去".contains(last)
                && chars.len() >= 2
                && "上下进出回过起开".contains(chars[chars.len() - 2])
            {
                let i = finals.len() - 1;
                finals[i] = replace_tone(&finals[i], '5');
            }
        }
        if let Some(ge_idx) = word.char_indices().find(|(_, c)| *c == '个').map(|(i, _)| i) {
            let byte_idx = ge_idx;
            let char_idx = word[..byte_idx].chars().count();
            let prev = word[..byte_idx].chars().last();
            if prev.is_some_and(is_number_char)
                || prev.is_some_and(|c| "几有两半多各整每做是".contains(c))
                || word == "个"
            {
                finals[char_idx] = replace_tone(&finals[char_idx], '5');
            }
        } else if MUST_NEURAL.contains(word) || MUST_NEURAL.contains(tail2(word).as_str()) {
            let i = finals.len() - 1;
            finals[i] = replace_tone(&finals[i], '5');
        }

        let word_list = Self::split_word(word);
        let split_at = word_list[0].chars().count();
        let (a, b) = finals.split_at(split_at.min(finals.len()));
        let mut finals_list = vec![a.to_vec(), b.to_vec()];
        for (i, sub) in finals_list.iter_mut().enumerate() {
            let w = if i == 0 { &word_list[0] } else { &word_list[1] };
            if MUST_NEURAL.contains(w.as_str()) || MUST_NEURAL.contains(tail2(w).as_str()) {
                if let Some(last) = sub.last_mut() {
                    *last = replace_tone(last, '5');
                }
            }
        }
        finals_list.into_iter().flatten().collect()
    }

    fn bu_sandhi(word: &str, mut finals: Vec<String>) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() == 3 && chars[1] == '不' {
            finals[1] = replace_tone(&finals[1], '5');
        } else {
            for (i, &ch) in chars.iter().enumerate() {
                if ch == '不' && i + 1 < chars.len() && tone_digit(&finals[i + 1]) == Some('4') {
                    finals[i] = replace_tone(&finals[i], '2');
                }
            }
        }
        finals
    }

    fn yi_sandhi(word: &str, mut finals: Vec<String>) -> Vec<String> {
        if word.contains('一') && word.chars().all(|c| c == '一' || c.is_ascii_digit()) {
            return finals;
        }
        let chars: Vec<char> = word.chars().collect();
        if chars.len() == 3 && chars[1] == '一' && chars[0] == chars[2] {
            finals[1] = replace_tone(&finals[1], '5');
            return finals;
        }
        if word.starts_with("第一") && finals.len() > 1 {
            finals[1] = replace_tone(&finals[1], '1');
            return finals;
        }
        for (i, &ch) in chars.iter().enumerate() {
            if ch == '一' && i + 1 < chars.len() {
                if tone_digit(&finals[i + 1]) == Some('4') {
                    finals[i] = replace_tone(&finals[i], '2');
                } else {
                    let next = chars[i + 1];
                    if !PUNC.contains(next) {
                        finals[i] = replace_tone(&finals[i], '4');
                    }
                }
            }
        }
        finals
    }

    fn split_word(word: &str) -> Vec<String> {
        let mut word_list: Vec<String> = JIEBA
            .cut_for_search(word, true)
            .into_iter()
            .map(|t| t.word.to_string())
            .collect();
        word_list.sort_by_key(|w| w.chars().count());
        let first = word_list.first().cloned().unwrap_or_default();
        if word.starts_with(&first) {
            vec![first.clone(), word[first.len()..].to_string()]
        } else {
            let second = word.strip_suffix(&first).unwrap_or("").to_string();
            vec![second, first]
        }
    }

    fn three_sandhi(word: &str, mut finals: Vec<String>) -> Vec<String> {
        let n = finals.len();
        if n == 2 && Self::all_tone_three(&finals) {
            finals[0] = replace_tone(&finals[0], '2');
        } else if n == 3 {
            let word_list = Self::split_word(word);
            if Self::all_tone_three(&finals) {
                if word_list[0].chars().count() == 2 {
                    finals[0] = replace_tone(&finals[0], '2');
                    finals[1] = replace_tone(&finals[1], '2');
                } else if word_list[0].chars().count() == 1 {
                    finals[1] = replace_tone(&finals[1], '2');
                }
            } else {
                let split_at = word_list[0].chars().count();
                let (a, b) = finals.split_at(split_at.min(finals.len()));
                let mut left = a.to_vec();
                let mut right = b.to_vec();
                if Self::all_tone_three(&left) && left.len() == 2 {
                    left[0] = replace_tone(&left[0], '2');
                }
                if !Self::all_tone_three(&right)
                    && tone_digit(right.first().unwrap()).is_some_and(|t| t == '3')
                    && tone_digit(left.last().unwrap()).is_some_and(|t| t == '3')
                {
                    let last = left.len() - 1;
                    left[last] = replace_tone(&left[last], '2');
                }
                finals = left.into_iter().chain(right).collect();
            }
        } else if n == 4 {
            let mut finals_list = vec![finals[..2].to_vec(), finals[2..].to_vec()];
            finals.clear();
            for sub in &mut finals_list {
                if Self::all_tone_three(sub) {
                    sub[0] = replace_tone(&sub[0], '2');
                }
                finals.extend(sub.iter().cloned());
            }
        }
        finals
    }

    fn all_tone_three(finals: &[String]) -> bool {
        finals.iter().all(|x| tone_digit(x) == Some('3'))
    }

    fn merge_bu(seg: Seg) -> Seg {
        let mut new_seg = Seg::new();
        let mut last_word = String::new();
        for (mut word, pos) in seg {
            if last_word == "不" {
                word = format!("不{word}");
            }
            if word != "不" {
                new_seg.push((word.clone(), pos));
            }
            last_word = word;
        }
        if last_word == "不" {
            new_seg.push(("不".into(), "d".into()));
        }
        new_seg
    }

    fn merge_yi(seg: Seg) -> Seg {
        let mut new_seg: Seg = Vec::new();
        for (i, (word, pos)) in seg.iter().cloned().enumerate() {
            if i > 0
                && word == "一"
                && i + 1 < seg.len()
                && seg[i - 1].0 == seg[i + 1].0
                && seg[i - 1].1 == "v"
            {
                if let Some(last) = new_seg.last_mut() {
                    last.0 = format!("{}{}一{}", last.0, last.0, last.0);
                }
            } else if i >= 2 && seg[i - 1].0 == "一" && seg[i - 2].0 == word && pos == "v" {
                continue;
            } else {
                new_seg.push((word, pos));
            }
        }
        let seg = new_seg;
        let mut out = Seg::new();
        for (word, pos) in seg {
            if out.last().is_some_and(|(w, _)| w == "一") {
                let prev = out.pop().unwrap();
                out.push((format!("{}{word}", prev.0), pos));
            } else {
                out.push((word, pos));
            }
        }
        out
    }

    fn merge_continuous_three_tones(seg: Seg) -> Seg {
        let sub_finals: Vec<Vec<String>> = seg
            .iter()
            .map(|(w, _)| word_initials_finals(w).1)
            .collect();
        let mut new_seg = Seg::new();
        let mut merge_last = vec![false; seg.len()];
        for (i, (word, pos)) in seg.into_iter().enumerate() {
            if i > 0
                && Self::all_tone_three(&sub_finals[i - 1])
                && Self::all_tone_three(&sub_finals[i])
                && !merge_last[i - 1]
            {
                let prev_word = &new_seg.last().unwrap().0;
                if !Self::is_reduplication(prev_word)
                    && prev_word.chars().count() + word.chars().count() <= 3
                {
                    new_seg.last_mut().unwrap().0.push_str(&word);
                    merge_last[i] = true;
                } else {
                    new_seg.push((word, pos));
                }
            } else {
                new_seg.push((word, pos));
            }
        }
        new_seg
    }

    fn merge_continuous_three_tones_2(seg: Seg) -> Seg {
        let sub_finals: Vec<Vec<String>> = seg
            .iter()
            .map(|(w, _)| word_initials_finals(w).1)
            .collect();
        let mut new_seg = Seg::new();
        let mut merge_last = vec![false; seg.len()];
        for (i, (word, pos)) in seg.into_iter().enumerate() {
            if i > 0
                && tone_digit(sub_finals[i - 1].last().unwrap()) == Some('3')
                && tone_digit(sub_finals[i].first().unwrap()) == Some('3')
                && !merge_last[i - 1]
            {
                let prev_word = &new_seg.last().unwrap().0;
                if !Self::is_reduplication(prev_word)
                    && prev_word.chars().count() + word.chars().count() <= 3
                {
                    new_seg.last_mut().unwrap().0.push_str(&word);
                    merge_last[i] = true;
                } else {
                    new_seg.push((word, pos));
                }
            } else {
                new_seg.push((word, pos));
            }
        }
        new_seg
    }

    fn merge_er(seg: Seg) -> Seg {
        let mut new_seg = Seg::new();
        for (i, (word, pos)) in seg.into_iter().enumerate() {
            if i > 0 && word == "儿" && new_seg.last().is_some_and(|(w, _)| w != "#") {
                new_seg.last_mut().unwrap().0.push('儿');
            } else {
                new_seg.push((word, pos));
            }
        }
        new_seg
    }

    fn merge_reduplication(seg: Seg) -> Seg {
        let mut new_seg = Seg::new();
        for (word, pos) in seg {
            if new_seg.last().is_some_and(|(w, _)| w == &word) {
                new_seg.last_mut().unwrap().0.push_str(&word);
            } else {
                new_seg.push((word, pos));
            }
        }
        new_seg
    }

    fn is_reduplication(word: &str) -> bool {
        let chars: Vec<char> = word.chars().collect();
        chars.len() == 2 && chars[0] == chars[1]
    }
}

fn is_number_char(c: char) -> bool {
    c.is_ascii_digit() || "零一二三四五六七八九十百千万亿两".contains(c)
}

fn tail2(word: &str) -> String {
    let chars: Vec<char> = word.chars().collect();
    if chars.len() >= 2 {
        chars[chars.len() - 2..].iter().collect()
    } else {
        word.to_string()
    }
}

fn tone_digit(s: &str) -> Option<char> {
    s.chars().last().filter(|c| c.is_ascii_digit())
}

fn replace_tone(s: &str, tone: char) -> String {
    if s.is_empty() {
        return s.to_string();
    }
    format!("{}{tone}", &s[..s.len().saturating_sub(1)])
}
