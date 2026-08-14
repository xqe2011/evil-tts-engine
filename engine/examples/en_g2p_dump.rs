//! Dump EN g2p for parity checks against Bert-VITS2 V220 english.py.

fn main() {
    let tests = [
        "Hello, I am evil, an assistant by apple banana.",
        "evilcorp",
        "The CEO of Apple Inc. spoke.",
        "Dr. Smith went home.",
        "In 2024, the 1st place was $100.",
    ];
    for t in tests {
        let norm = tts_engine::text_normalize(t);
        let g = tts_engine::g2p::g2p_english(&norm);
        println!("=== {t:?}");
        println!("  norm: {norm:?}");
        println!("  phones: {:?}", g.phones);
        println!("  tones: {:?}", g.tones);
        println!("  word2ph: {:?}", g.word2ph);
    }
}
