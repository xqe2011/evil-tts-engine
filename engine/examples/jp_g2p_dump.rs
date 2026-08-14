//! Dump JP g2p for parity checks against Bert-VITS2 V220 japanese.py.

fn main() {
    let tests = [
        "こんにちは、世界！",
        "2024年",
        "100円",
        "$100",
        "100,000",
        "hello,こんにちは、世界ー！……",
    ];
    for t in tests {
        let g = tts_engine::japanese::g2p_japanese(t);
        println!("=== {t:?}");
        println!("  norm: {:?}", g.norm_text);
        println!("  phones: {:?}", g.phones);
        println!("  tones: {:?}", g.tones);
        println!("  word2ph: {:?}", g.word2ph);
    }
}
