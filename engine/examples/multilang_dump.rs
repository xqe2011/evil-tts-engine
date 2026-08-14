//! Dump multilang prepare tensors for parity vs Bert-VITS2 V220 infer_multilang.

fn main() {
    let tests = [
        "[ZH]你好[EN]hello[JP]こんにちは",
        "[EN]Hello[ZH]世界",
    ];
    for t in tests {
        println!("=== {t}");
        let Some(m) = tts_engine::g2p::prepare_multilang(t, false, false) else {
            println!("  (no tags / empty)");
            continue;
        };
        println!("  phones: {:?}", m.phones);
        println!("  tones: {:?}", m.tones);
        println!("  language: {:?}", m.language);
        println!("  n_phones: {}", m.phones.len());
        for (i, seg) in m.segments.iter().enumerate() {
            println!(
                "  seg[{i}] bert_lang={} n_ids={} word2ph={:?}",
                seg.bert_lang,
                seg.input_ids.len(),
                seg.word2ph
            );
        }
    }
}
