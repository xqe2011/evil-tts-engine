//! Shared jieba instance (wasm-safe: no zstd/default-dict feature).

use jieba_rs::Jieba;
use once_cell::sync::Lazy;
use std::io::Cursor;

pub static JIEBA: Lazy<Jieba> = Lazy::new(|| {
    let mut jieba = Jieba::empty();
    let mut reader = Cursor::new(include_bytes!("../assets/jieba_dict.txt"));
    jieba
        .load_dict(&mut reader)
        .expect("load jieba_dict.txt");
    jieba
});
