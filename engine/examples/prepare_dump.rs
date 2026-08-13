//! Dump prepare() for the test sentence (native).

fn main() {
    let text = "Hello, I am evil, an assistant by apple banana.";
    let p = tts_engine::g2p::prepare(text); // need pub use
    println!("ids {:?}", p.input_ids);
    println!("phones {:?}", p.phones);
    println!("tones {:?}", p.tones);
    println!("word2ph {:?}", p.word2ph);
    println!("n_phones {}", p.phones.len());
}
