# -*- coding: utf-8 -*-
"""Rebuild vocab.bin from tokenizer.json with the CORRECT byte decoder.

The Qwen3 tokenizer is byte-level BPE: every token string in tokenizer.json
uses GPT-2 style byte encoding, where raw UTF-8 bytes are shown as chars:
  - 0x21..0x7E          -> as-is (ASCII)
  - 0xA1..0xAC, 0xAE..0xFF -> as-is (latin-1 range)
  - 0x00..0x20          -> U+0100..U+0120   (0x0A->U+010A 'C', 0x20->U+0120 'G')
  - 0x7F..0xA0          -> U+0121..U+0142
  - 0xAD                -> U+0143 'N'   <-- the byte fix_vocab_local.py missed

The engine decodes token strings directly as UTF-8 text (no byte decoder at
runtime), so vocab.bin must store the UN-encoded UTF-8 text, e.g. "答" not
"çŃĶ". This script does that conversion. The 'Ġ'/'Ċ' marker chars are KEPT
(encode/piece_to_text rely on them). Special tokens (id >= 151643) are copied
from the existing vocab.bin.

Usage:
    python tools/build_vocab_bin.py <tokenizer.json> <vocab.bin> <vocab.bin.new>

  - <tokenizer.json> : HuggingFace tokenizer.json of the target model
  - <vocab.bin>      : an existing engine vocab.bin from the same model
                       (special tokens, id >= 151643, are copied from it)
  - <vocab.bin.new>  : output path (can equal the input vocab.bin in-place)
"""
import io, json, struct, sys

if len(sys.argv) != 4:
    sys.exit("usage: python build_vocab_bin.py <tokenizer.json> <vocab.bin> <out.bin>")
TOKJ, OLDV, OUTV = sys.argv[1], sys.argv[2], sys.argv[3]

def decode_token_str(s):
    """Un-encode a byte-level BPE token string to raw UTF-8 text."""
    out = bytearray()
    for ch in s:
        cp = ord(ch)
        if cp == 0x0143:          # byte 0xAD (the case the old fix got wrong)
            out.append(0xAD)
        elif cp == 0x0121:        # byte 0x7F
            out.append(0x7F)
        elif 0x0122 <= cp <= 0x0142:
            out.append(cp - 0xA2) # 0x80..0xA0
        elif 0x00A1 <= cp <= 0x00AC or 0x00AE <= cp <= 0x00FF:
            out.append(cp)        # latin-1 as-is
        elif 0x0080 <= cp <= 0x00A0:
            out.append(cp)        # C2 80..A0 -> byte (rare, keep old behavior)
        else:
            out.extend(ch.encode("utf-8"))  # keep 'Ġ'/'Ċ'/CJK/etc as-is
    return bytes(out)

def main():
    with io.open(TOKJ, "r", encoding="utf-8") as f:
        vocab = json.load(f)["model"]["vocab"]
    items = sorted(vocab.items(), key=lambda kv: kv[1])  # (str, id), id 0..151642

    fixed = 0
    kept = 0
    entries = []          # (tid, bytes)
    for s, tid in items:
        raw = s.encode("utf-8")
        rec = decode_token_str(s)
        # Store the RAW bytes (bytes_to_unicode inverse) for EVERY token.
        # Fragment tokens (mid-BPE emoji states) are invalid UTF-8 on their
        # own but MUST be stored as raw bytes so the engine's incremental
        # UTF-8 decoder can splice them across token boundaries. Storing the
        # marker STRING's UTF-8 (the old "keep original" branch) is what
        # produced the "ðŁĺ" garbage.
        entries.append((tid, rec))
        if rec != raw:
            fixed += 1
        else:
            kept += 1

    # copy special tokens (id >= 151643) from existing vocab.bin unchanged
    extras = []
    with open(OLDV, "rb") as f:
        vsz = struct.unpack("<I", f.read(4))[0]
        for _ in range(vsz):
            tid, slen = struct.unpack("<IH", f.read(6))
            bs = f.read(slen)
            if tid >= 151643:
                extras.append((tid, bs))
    entries.extend(extras)
    entries.sort(key=lambda e: e[0])

    with open(OUTV, "wb") as f:
        f.write(struct.pack("<I", len(entries)))
        for tid, bs in entries:
            f.write(struct.pack("<IH", tid, len(bs)))
            f.write(bs)

    print("total entries:", len(entries))
    print("converted:", fixed, " kept:", kept, " special copied:", len(extras))

    # ---- validation ----
    d = {tid: bs for tid, bs in entries}
    def count(bs_sub):
        return sum(1 for v in d.values() if bs_sub in v)
    print("\n-- common char coverage --")
    for ch in "的一是不了人我在有他这中大来上国个到说们为子和你地出道也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没日于起还发成事只作当想看文无开手十用主行方如前所本见经头面公同三已老从动两长知民样现分将外但身些与高意进把法此实回二理美点月明其种声全工己话儿者向情部正名定女答案语学习两":
        if count(ch.encode("utf-8")) == 0:
            print("   MISSING:", ch)
    print("   (no MISSING lines above = full coverage)")
    for probe in ["回答", "语言", "学习", "你好", "Ġgood", "Ċ"]:
        hits = [tid for tid, v in d.items() if probe.encode("utf-8") in v]
        print(f"   {probe!r}: {len(hits)} hits {hits[:3]}")
    print("\nDONE ->", OUTV)

if __name__ == "__main__":
    main()
