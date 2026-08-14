//! Dump ZH g2p for parity checks against Bert-VITS2 V220 chinese.py.

fn main() {
    let tests = ["你好，我是助手。", "一个苹果", "2024年"];
    for t in tests {
        let g = tts_engine::chinese::g2p_chinese(t);
        println!("=== {t:?}");
        println!("  norm: {:?}", g.norm_text);
        println!("  phones: {:?}", g.phones);
        println!("  tones: {:?}", g.tones);
        println!("  word2ph: {:?}", g.word2ph);
    }
}
