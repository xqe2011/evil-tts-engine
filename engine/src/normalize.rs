//! Number / punctuation normalization (EN Bert-VITS2 subset).

use once_cell::sync::Lazy;
use regex::Regex;

static COMMA_NUMBER: Lazy<Regex> = Lazy::new(|| Regex::new(r"([0-9][0-9\,]+[0-9])").unwrap());
static DECIMAL: Lazy<Regex> = Lazy::new(|| Regex::new(r"([0-9]+\.[0-9]+)").unwrap());
static POUNDS: Lazy<Regex> = Lazy::new(|| Regex::new(r"£([0-9\,]*[0-9]+)").unwrap());
static DOLLARS: Lazy<Regex> = Lazy::new(|| Regex::new(r"\$([0-9\.\,]*[0-9]+)").unwrap());
static ORDINAL: Lazy<Regex> = Lazy::new(|| Regex::new(r"[0-9]+(st|nd|rd|th)").unwrap());
static NUMBER: Lazy<Regex> = Lazy::new(|| Regex::new(r"[0-9]+").unwrap());
static PUNCT_WORD: Lazy<Regex> = Lazy::new(|| Regex::new(r"([,;.\?\!])([\w])").unwrap());

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

fn under_100(n: u64) -> String {
    if n < 20 {
        ONES[n as usize].to_string()
    } else {
        let t = TENS[(n / 10) as usize];
        let o = n % 10;
        if o == 0 {
            t.to_string()
        } else {
            format!("{}-{}", t, ONES[o as usize])
        }
    }
}

fn under_1000(n: u64) -> String {
    if n < 100 {
        under_100(n)
    } else {
        let h = n / 100;
        let r = n % 100;
        if r == 0 {
            format!("{} hundred", ONES[h as usize])
        } else {
            format!("{} hundred {}", ONES[h as usize], under_100(r))
        }
    }
}

/// Compact English cardinal (good enough for TTS prompts; not full `inflect`).
pub fn number_to_words(n: u64) -> String {
    if n < 1000 {
        return under_1000(n);
    }
    if n < 1_000_000 {
        let th = n / 1000;
        let r = n % 1000;
        if r == 0 {
            format!("{} thousand", under_1000(th))
        } else {
            format!("{} thousand {}", under_1000(th), under_1000(r))
        }
    } else {
        n.to_string()
    }
}

fn expand_dollars(match_str: &str) -> String {
    let parts: Vec<&str> = match_str.split('.').collect();
    if parts.len() > 2 {
        return format!("{match_str} dollars");
    }
    let dollars: u64 = parts[0].replace(',', "").parse().unwrap_or(0);
    let cents: u64 = if parts.len() > 1 {
        parts[1].parse().unwrap_or(0)
    } else {
        0
    };
    if dollars > 0 && cents > 0 {
        let du = if dollars == 1 { "dollar" } else { "dollars" };
        let cu = if cents == 1 { "cent" } else { "cents" };
        format!("{dollars} {du}, {cents} {cu}")
    } else if dollars > 0 {
        let du = if dollars == 1 { "dollar" } else { "dollars" };
        format!("{dollars} {du}")
    } else if cents > 0 {
        let cu = if cents == 1 { "cent" } else { "cents" };
        format!("{cents} {cu}")
    } else {
        "zero dollars".into()
    }
}

fn expand_number(num: u64) -> String {
    if (1000..3000).contains(&num) {
        if num == 2000 {
            return "two thousand".into();
        }
        if (2001..2010).contains(&num) {
            return format!("two thousand {}", number_to_words(num % 100));
        }
        if num % 100 == 0 {
            return format!("{} hundred", number_to_words(num / 100));
        }
        // approx inflect group=2
        return number_to_words(num);
    }
    number_to_words(num)
}

fn ordinal_to_words(s: &str) -> String {
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    let n: u64 = digits.parse().unwrap_or(0);
    // simplistic: "twenty-first" etc. not perfect; enough for rare cases
    format!("{} {}", number_to_words(n), &s[digits.len()..])
}

fn replace_punctuation(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        let repl = match c {
            '：' | '；' | '，' | '、' | '·' | '・' => Some(","),
            '。' | '．' | '$' => Some("."),
            '！' => Some("!"),
            '？' => Some("?"),
            '\n' => Some("."),
            '“' | '”' | '"' | '‘' | '’' | '（' | '）' | '(' | ')' | '《' | '》' | '【' | '】'
            | '[' | ']' | '「' | '」' => Some("'"),
            '—' | '−' | '～' | '~' => Some("-"),
            '…' => Some("..."),
            _ => None,
        };
        if let Some(r) = repl {
            out.push_str(r);
        } else {
            out.push(c);
        }
    }
    // multi-char ellipsis variants already partly handled
    out.replace("···", "...").replace("・・・", "...")
}

fn normalize_numbers(text: &str) -> String {
    let text = COMMA_NUMBER.replace_all(text, |caps: &regex::Captures| {
        caps[1].replace(',', "")
    });
    let text = POUNDS.replace_all(&text, "$1 pounds");
    let text = DOLLARS.replace_all(&text, |caps: &regex::Captures| expand_dollars(&caps[1]));
    let text = DECIMAL.replace_all(&text, |caps: &regex::Captures| {
        caps[1].replace('.', " point ")
    });
    let text = ORDINAL.replace_all(&text, |caps: &regex::Captures| ordinal_to_words(&caps[0]));
    NUMBER
        .replace_all(&text, |caps: &regex::Captures| {
            let n: u64 = caps[0].parse().unwrap_or(0);
            expand_number(n)
        })
        .into_owned()
}

pub fn text_normalize(text: &str) -> String {
    let text = normalize_numbers(text);
    let text = replace_punctuation(&text);
    PUNCT_WORD.replace_all(&text, "$1 $2").into_owned()
}
