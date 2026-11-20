#!/usr/bin/env python3
"""Check nano's native tokenizer and chat template against the reference libraries.

Run with the reference Python (`$NANO_MLX_PYTHON -I`), which has `tokenizers` and
`transformers`. The checkpoint folder is read as data; nothing in it is executed.

  corpus   CHECKPOINT  generate the corpus (>= 20,000 strings), encode and decode it with
                       `tokenizers` and with `nano tokenize CHECKPOINT --batch FILE` (one
                       process), and require zero differences in ids and in decoded text
  classes  CHECKPOINT  every code point's class under the library's own pre-tokenizer regex
                       (letter / number / space / other) against src/mega/unicode_tables.zig
  template CHECKPOINT  the native chat prompt against the checkpoint's chat_template.jinja
                       rendered by transformers (text and ids), and the chat continuation
                       (one rendered turn + reply + next turn == the whole conversation)
  fixture  CHECKPOINT  write src/mega/testdata/: a small tokenizer trained with the library
                       in the checkpoint's format, and expected outputs for the unit tests

Exit 0 when every comparison is identical, 1 otherwise.
"""
import argparse
import base64
import glob
import json
import os
import pathlib
import random
import struct
import subprocess
import sys
import time
import unicodedata

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "zig-out/bin/nano"
SCRATCH = ROOT / "target/tokenizer-check"


def load_lib(checkpoint):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(pathlib.Path(checkpoint) / "tokenizer.json"))


# ---- native batch protocol (src/mega/cli.zig, `nano tokenize MODEL --batch FILE`) ----

def rec_encode(raw):
    return b"E" + struct.pack("<I", len(raw)) + raw


def rec_decode(ids):
    return b"D" + struct.pack("<I", len(ids)) + struct.pack("<%dI" % len(ids), *ids)


def rec_json(kind, obj):
    raw = json.dumps(obj).encode()
    return kind + struct.pack("<I", len(raw)) + raw


def run_native(binary, checkpoint, records, name):
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / ("%s.bin" % name)
    path.write_bytes(b"".join(records))
    start = time.perf_counter()
    proc = subprocess.run([str(binary), "tokenize", str(checkpoint), "--batch", str(path)],
                          capture_output=True, timeout=1800)
    wall = time.perf_counter() - start
    path.unlink()
    if proc.returncode != 0:
        sys.exit("native batch failed (%d): %s" % (proc.returncode, proc.stderr.decode(errors="replace")[-2000:]))
    out = [json.loads(line) for line in proc.stdout.decode().split("\n") if line]
    if len(out) != len(records):
        sys.exit("native batch answered %d records for %d" % (len(out), len(records)))
    return out, wall, proc.stderr.decode(errors="replace").strip()


# ---- corpus ----

SENTENCES = [
    "The quick brown fox jumps over the lazy dog. It's 12:45, isn't it? We'll see; they'd've known.",
    "Le cœur a ses raisons que la raison ne connaît point. L'été dernier, à Paris, il faisait 35 °C.",
    "Über den Wolken muss die Freiheit wohl grenzenlos sein. Straße, Maß, groß — ẞ.",
    "¿Dónde está la biblioteca? ¡Está allí, señor! Año 2024: 1.234,56 €.",
    "Ο Ηράκλειτος είπε: «τὰ πάντα ῥεῖ». Η Αθήνα είναι η πρωτεύουσα της Ελλάδας.",
    "Москва — столица России. Съешь же ещё этих мягких французских булок, да выпей чаю.",
    "Слава Україні! Ґава, їжак, єнот. Беларусь: ў.",
    "مرحبا بالعالم. القاهرة هي عاصمة مصر، وعدد سكانها ٢٠ مليون نسمة.",
    "שלום עולם! ירושלים היא עיר עתיקה. מספר 123 בעברית.",
    "नमस्ते दुनिया। भारत की राजधानी नई दिल्ली है। संख्या १२३४५।",
    "আমার সোনার বাংলা, আমি তোমায় ভালোবাসি। ১৯৭১ সালে।",
    "வணக்கம் உலகம். தமிழ் ஒரு பழமையான மொழி. ௧௨௩",
    "สวัสดีชาวโลก ภาษาไทยไม่มีการเว้นวรรคระหว่างคำ ๑๒๓",
    "你好，世界！北京是中国的首都。今天是2024年1月1日，气温零下五度。",
    "こんにちは世界。東京は日本の首都です。ひらがな、カタカナ、漢字を使います。ｶﾀｶﾅ。",
    "안녕하세요 세계! 서울은 대한민국의 수도입니다. 한글은 1443년에 만들어졌습니다.",
    "Γεια σου κόσμε; Привет мир; Hello world; 你好世界; مرحبا; שלום; नमस्ते.",
    "Tiếng Việt có dấu: Hà Nội, Đà Nẵng, Thành phố Hồ Chí Minh.",
    "Türkçe: İstanbul, Iğdır, ılık, şeker, çiçek, öğrenci, ü.",
    "Polski: Zażółć gęślą jaźń. Čeština: Příliš žluťoučký kůň úpěl ďábelské ódy.",
    "ქართული ენა: გამარჯობა მსოფლიო. Հայերեն: Բարեւ աշխարհ.",
    "አማርኛ: ሰላም ልዑል። ᚠᚢᚦᚨᚱᚲ ᚷᚹ. ᐊᐃᓐᓇᐃ ᓄᓇᕗᑦ.",
    "ꯃꯤꯇꯩ ꯂꯣꯟ; ꦗꦮ; ᜊᜌ᜔ᜊᜌᜒᜈ᜔; 𐌰𐌱𐌲 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝟙𝟚𝟛.",
    "Emoji: 👍🏽 👨‍👩‍👧‍👦 🇯🇵 🏳️‍🌈 ❤️ ☃ ✓ 🙂🙃 — keycap 1️⃣, ZWJ, variation selectors.",
    "Combining: é ä ñ ǫ́ Z͓͔͑͒ क्ष 각.",
    "Math: ∀x∈ℝ, ∃y: x² + y² ≤ 1 ⇒ |x| ≤ 1; ∑ᵢ aᵢ = ∫₀¹ f(t) dt ≈ 3.14159…",
    "Currency: $12.50, £7, ¥1000, ₹250, ₩5000, ₿0.001, 100 ₽.",
    "Fullwidth: ＡＢＣ１２３！？ and halfwidth ｱｲｳ, ideographic space　here.",
    "Zero-width: a​b‌c‍d﻿e⁠f and soft­hyphen.",
    "Bidi: ‮abc‬ ⁦x⁩ ‏‎ RTL marks.",
]

SCRIPTS = {
    "latin": (0x41, 0x24F), "greek": (0x391, 0x3C9), "cyrillic": (0x410, 0x44F), "armenian": (0x531, 0x586),
    "hebrew": (0x5D0, 0x5EA), "arabic": (0x621, 0x64A), "devanagari": (0x905, 0x939), "bengali": (0x985, 0x9B9),
    "gurmukhi": (0xA05, 0xA39), "gujarati": (0xA85, 0xAB9), "tamil": (0xB85, 0xBB9), "telugu": (0xC05, 0xC39),
    "kannada": (0xC85, 0xCB9), "malayalam": (0xD05, 0xD39), "thai": (0xE01, 0xE30), "lao": (0xE81, 0xEAE),
    "tibetan": (0xF40, 0xF6C), "myanmar": (0x1000, 0x102A), "georgian": (0x10D0, 0x10FA), "hangul_jamo": (0x1100, 0x1112),
    "ethiopic": (0x1200, 0x1248), "cherokee": (0x13A0, 0x13F4), "canadian": (0x1401, 0x166C), "khmer": (0x1780, 0x17B3),
    "mongolian": (0x1820, 0x1877), "hiragana": (0x3041, 0x3096), "katakana": (0x30A1, 0x30FA), "cjk": (0x4E00, 0x9FFF),
    "hangul": (0xAC00, 0xD7A3), "yi": (0xA000, 0xA48C), "cjk_ext_b": (0x20000, 0x2A6DF), "math_alnum": (0x1D400, 0x1D7FF),
}
MARKS = [0x300, 0x301, 0x308, 0x327, 0x94D, 0x93F, 0xE31, 0x5B0, 0x64E, 0x20D7, 0xFE0F, 0x200D]
DIGITS = [chr(c) for c in range(0x110000) if unicodedata.category(chr(c)) == "Nd"]
OTHER_NUMBERS = [chr(c) for c in range(0x110000) if unicodedata.category(chr(c)) in ("Nl", "No")]
WHITESPACE = ["\t", "\n", "\x0b", "\x0c", "\r", " ", "\x85", "\xa0", " ", " ", " ", " ",
              " ", " ", " ", " ", "　", "\r\n", "\x1c", "\x1f", "᠎", "​", "﻿"]
PUNCT = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") + ["«", "»", "—", "–", "…", "¿", "¡", "·", "、", "。", "「", "」", "‘", "’", "“", "”"]


def by_category():
    cats = {}
    for cp in range(0x110000):
        cats.setdefault(unicodedata.category(chr(cp)), []).append(cp)
    return cats


def synth_word(rng, lo, hi):
    return "".join(chr(rng.randint(lo, hi)) for _ in range(rng.randint(1, 9)))


def synth_prose(rng):
    name = rng.choice(list(SCRIPTS))
    lo, hi = SCRIPTS[name]
    words = []
    for _ in range(rng.randint(3, 40)):
        w = synth_word(rng, lo, hi)
        if rng.random() < 0.15:
            w += chr(rng.choice(MARKS))
        if rng.random() < 0.1:
            w = rng.choice(DIGITS) * rng.randint(1, 6)
        words.append(w)
        if rng.random() < 0.15:
            words.append(rng.choice(PUNCT))
    sep = rng.choice([" ", " ", " ", "　", "", "\n", "  "])
    return sep.join(words)


def code_snippets(rng, n):
    files = []
    for pattern in ("src/**/*.zig", "src/**/*.metal", "src/*.m", "tools/*.py", "docs/*.md", "*.md", "build.zig", "benchmarks/*.md"):
        files += [p for p in glob.glob(str(ROOT / pattern), recursive=True) if "unicode_tables" not in p]
    texts = [pathlib.Path(p).read_text(errors="replace") for p in sorted(files)]
    texts = [t for t in texts if len(t) > 100]
    out = []
    for _ in range(n):
        t = rng.choice(texts)
        size = rng.choice([20, 80, 300, 1000, 4000])
        start = rng.randrange(0, max(1, len(t) - size))
        snippet = t[start:start + size]
        if rng.random() < 0.2:
            snippet = snippet.replace("\n", "\r\n")
        if rng.random() < 0.1:
            snippet = snippet.replace("    ", "\t")
        out.append(snippet)
    return out


def whitespace_runs(rng, n):
    out = []
    atoms = WHITESPACE + ["a", "B", "1", "!", "'s", "é", "中", "\n\n", "    ", "\r\r\n"]
    for _ in range(n):
        out.append("".join(rng.choice(WHITESPACE if rng.random() < 0.7 else atoms) for _ in range(rng.randint(1, 30))))
    return out


def contractions(rng, n):
    forms = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'T", "'RE", "'Re", "'rE", "'VE", "'M", "'LL", "'lL",
             "'D", "'ſ", "'ﬆ", "'K", "'", "''", "'x", "'1", "' s", "'\ns", "’s"]
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 8)):
            parts.append(rng.choice(["it", "I", "we", "THEY", "x", "", " ", "1", "é", "中", "!"]))
            parts.append(rng.choice(forms))
            parts.append(rng.choice(["", " ", "a", "e", "l", "\n", "x"]))
        out.append("".join(parts))
    return out


def number_strings(rng, n):
    out = []
    for _ in range(n):
        k = rng.choice([1, 2, 3, 4, 5, 7, 10, 33])
        if rng.random() < 0.5:
            s = "".join(rng.choice("0123456789") for _ in range(k))
        else:
            s = "".join(rng.choice(DIGITS + OTHER_NUMBERS) for _ in range(k))
        out.append(rng.choice(["", "x", " ", "$", "-", "v"]) + s + rng.choice(["", "x", " ", ".", "%", "px"]))
    return out


def category_strings(rng, cats, per):
    out = []
    contexts = ["{}", "a{}", "{}a", " {}", "{} ", "1{}", "{}1", "'{}", "{}\n", "\n{}", "!{}", "{}!", "  {}  x"]
    for cat in sorted(cats):
        if cat == "Cs":
            continue  # surrogates are not text; their UTF-8 forms are in the invalid-UTF-8 group
        pool = cats[cat]
        for _ in range(per):
            body = "".join(chr(rng.choice(pool)) for _ in range(rng.randint(1, 12)))
            out.append(rng.choice(contexts).format(body))
    return out


def every_code_point():
    """Every scalar value, in chunks, adjacent to each other and in rotating contexts."""
    cps = [cp for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]
    contexts = ["a", " ", "'", "1", "\n", "\t", "!", " a", "　", "\r\n", "é", "  "]
    out = []
    for i in range(0, len(cps), 96):
        chunk = cps[i:i + 96]
        out.append("".join(map(chr, chunk)))
        out.append("".join(contexts[(j + i) % len(contexts)] + chr(c) for j, c in enumerate(chunk)))
    return out


def byte_soup(rng, n):
    raw = []
    for _ in range(n):
        k = rng.randint(1, 200)
        mode = rng.random()
        if mode < 0.4:
            b = bytes(rng.randrange(256) for _ in range(k))
        elif mode < 0.7:
            b = bytes(rng.choice([rng.randrange(0x80), rng.randrange(0x80, 0xC0), rng.randrange(0xC0, 0x100)]) for _ in range(k))
        elif mode < 0.8:
            b = chr(rng.randrange(0xD800, 0xE000)).encode("utf-8", "surrogatepass") * rng.randint(1, 3)
        else:
            # valid text with corrupted bytes
            t = bytearray(synth_prose(rng).encode())
            for _ in range(rng.randint(1, 5)):
                if t:
                    t[rng.randrange(len(t))] = rng.randrange(256)
            if t and rng.random() < 0.5:
                del t[rng.randrange(len(t)):]
            b = bytes(t)
        raw.append(b)
    return raw


def added_token_strings(rng, added):
    contents = [t["content"] for t in added]
    out = []
    contexts = ["{}", "a{}", "{}a", " {}", "{} ", "{}\n", "\n{}", "x {} y", "{}{}", "<{}>", "{}<", "|{}|", "1{}2", "é{}中"]
    for c in contents:
        for ctx in contexts:
            out.append(ctx.replace("{}", c))
        out.append(c[:-1])
        out.append(c[1:])
        out.append(c[: len(c) // 2] + " " + c[len(c) // 2:])
    for _ in range(1500):
        parts = []
        for _ in range(rng.randint(1, 10)):
            r = rng.random()
            if r < 0.5:
                parts.append(rng.choice(contents))
            elif r < 0.7:
                c = rng.choice(contents)
                parts.append(c[: rng.randint(1, len(c))])
            else:
                parts.append(rng.choice(["hello", " ", "\n", "<", "|", ">", "<|", "|>", "user", "中文", "'s", "  "]))
        out.append("".join(parts))
    return out


def long_inputs(rng, cats):
    out = ["a" * 100000, " " * 50000, "\n" * 30000, "1" * 70000, "ab" * 60000, "中" * 40000, "!" * 50000,
           "\r\n" * 20000, " a" * 40000, "　" * 20000, ("x " * 5000 + "\n") * 10, "'s" * 20000,
           "<|im_start|>" * 3000, "<|im_" * 5000]
    letters = cats["Lo"] + cats["Ll"] + cats["Lu"]
    for size in (50000, 200000, 1000000):
        out.append("".join(synth_prose(rng) + rng.choice(["\n", " ", "\n\n"]) for _ in range(size // 200))[:size])
        out.append("".join(chr(rng.choice(letters)) if rng.random() < 0.8 else rng.choice(PUNCT + WHITESPACE) for _ in range(size // 4)))
    return out


def build_corpus(seed, added):
    rng = random.Random(seed)
    cats = by_category()
    groups = {
        "source code": code_snippets(rng, 5000),
        "prose (written)": [s for s in SENTENCES for _ in range(1)] + [a + "\n" + b for a in SENTENCES for b in SENTENCES[:10]],
        "prose (synthetic, 32 scripts)": [synth_prose(rng) for _ in range(5000)],
        "general categories (29; Cs below)": category_strings(rng, cats, 100),
        "every code point": every_code_point(),
        "whitespace runs": whitespace_runs(rng, 3000),
        "contractions": contractions(rng, 1000),
        "numbers": number_strings(rng, 1000),
        "added tokens in text": added_token_strings(rng, added),
        "byte soup made valid": [b.decode("utf-8", "replace") for b in byte_soup(rng, 3000)],
        "long inputs": long_inputs(rng, cats),
        "empty": [""],
    }
    texts = [(name, s.encode("utf-8", "surrogatepass"), s) for name, items in groups.items() for s in items]
    # Invalid UTF-8 given raw to nano; the library gets the same bytes decoded with "replace".
    texts += [("invalid UTF-8 (raw)", b, b.decode("utf-8", "replace")) for b in byte_soup(rng, 3000)]
    return texts, rng


def random_id_sequences(rng, size, n):
    seqs = []
    for _ in range(n):
        k = rng.randint(1, 60)
        mode = rng.random()
        if mode < 0.5:
            seqs.append([rng.randrange(size) for _ in range(k)])
        elif mode < 0.8:
            seqs.append([rng.randrange(256) for _ in range(k)])  # single bytes: partial UTF-8 everywhere
        else:
            seqs.append([rng.randrange(size - 200, size + 3000) for _ in range(k)])  # added, special, unknown ids
    return seqs


def cmd_corpus(args):
    lib = load_lib(args.checkpoint)
    added = json.loads((pathlib.Path(args.checkpoint) / "tokenizer.json").read_text())["added_tokens"]
    texts, rng = build_corpus(args.seed, added)
    size = lib.get_vocab_size(with_added_tokens=True)
    t0 = time.perf_counter()
    lib_ids = [lib.encode(t[2], add_special_tokens=True).ids for t in texts]
    lib_seconds = time.perf_counter() - t0
    id_seqs = lib_ids + random_id_sequences(rng, size, args.random_ids)
    records = [rec_encode(raw) for _, raw, _ in texts] + [rec_decode(ids) for ids in id_seqs]
    native, wall, note = run_native(args.binary, args.checkpoint, records, "corpus")
    enc, dec = native[:len(texts)], native[len(texts):]
    lib_text = lib.decode_batch(id_seqs, skip_special_tokens=True)
    lib_special = lib.decode_batch(id_seqs, skip_special_tokens=False)
    groups, id_diffs, text_diffs = {}, [], []
    for (name, raw, text), want, got in zip(texts, lib_ids, enc):
        g = groups.setdefault(name, [0, 0, 0])
        g[0] += 1
        g[1] += len(raw)
        if got["ids"] != want:
            g[2] += 1
            id_diffs.append((name, raw, want, got["ids"]))
    for ids, got, t1, t2 in zip(id_seqs, dec, lib_text, lib_special):
        for key, want in (("text", t1), ("special", t2), ("stream", t1)):
            if got[key] != want:
                text_diffs.append((key, ids, want, got[key]))
    print("corpus seed %d: %d strings, %d bytes; %d decode checks (%d from the corpus ids, %d random id sequences), "
          "each as decode, decode keeping special tokens, and streaming decode"
          % (args.seed, len(texts), sum(len(t[1]) for t in texts), len(id_seqs), len(lib_ids), len(id_seqs) - len(lib_ids)))
    for name, (count, nbytes, bad) in groups.items():
        print("  %-32s %6d strings %10d bytes  %d id differences" % (name, count, nbytes, bad))
    print("library encode: %.2f s; native batch process (load + all records): %.2f s" % (lib_seconds, wall))
    if note:
        print("native: " + note)
    for name, raw, want, got in id_diffs[:10]:
        print("ID DIFF [%s] %r\n  library %s\n  native  %s" % (name, raw[:200], want[:40], got[:40]))
    for key, ids, want, got in text_diffs[:10]:
        print("TEXT DIFF (%s) ids %s\n  library %r\n  native  %r" % (key, ids[:20], want[:100], got[:100]))
    print("id differences: %d of %d strings; decoded-text differences: %d of %d checks"
          % (len(id_diffs), len(texts), len(text_diffs), 3 * len(id_seqs)))
    return 0 if not id_diffs and not text_diffs else 1


# ---- classes ----

def cmd_classes(args):
    from tokenizers import pre_tokenizers, Regex
    spec = json.loads((pathlib.Path(args.checkpoint) / "tokenizer.json").read_text())
    pattern = spec["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    split = pre_tokenizers.Split(Regex(pattern), "isolated")

    def n(s):
        return len(split.pre_tokenize_str(s))
    native = {}
    text = (ROOT / "src/mega/unicode_tables.zig").read_text()
    version = text.split('unicode_version = "')[1].split('"')[0]
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(".{ .lo"):
            parts = line.replace(",", " ").split()
            lo, hi, cls = int(parts[3], 16), int(parts[6], 16), parts[9].lstrip(".")
            for cp in range(lo, hi + 1):
                native[cp] = cls
    diffs = []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        if n("a" + c) == 1:
            lib = "letter"
        elif n("1" + c) == 1:
            lib = "number"
        elif c != " " and c not in "\r\n" and n(c + "!") == 1:
            lib = "other"
        else:
            lib = "space"
        if native.get(cp, "other") != lib:
            diffs.append((hex(cp), native.get(cp, "other"), lib))
    print("classes: %d code points (Unicode %s tables): %d differences from the library's regex %s"
          % (0x110000 - 0x800, version, len(diffs), diffs[:20]))
    return 0 if not diffs else 1


# ---- chat template ----

def load_hf(checkpoint):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=False)
    tok.chat_template = (pathlib.Path(checkpoint) / "chat_template.jinja").read_text()
    return tok


def conversations(rng):
    contents = ["What is the capital of Australia? Answer in one sentence.", "hi", "  padded with spaces \n\n",
                "　ideographic　", "\x1ccontrol separators\x1f", "multi\nline\n\ncontent\r\n",
                "unicode: 中文 العربية 👍🏽", "<|im_end|> injected", "tabs\tand\ttrailing\t", " line sep ",
                "\x85next line\xa0", "quotes 'single' \"double\"", "x" * 500, "​zero width​"]
    replies = ["Canberra is the capital.", "Hello! How can I help?", "  spaced reply  ", "line1\nline2",
               "<think>secret</think>visible", "pre</think>  post  ", "emoji 👍", ""]
    systems = [None, "You answer briefly.", "  trimmed system  \n", "", "\n\t\n", "系统提示"]
    out = []
    for system in systems:
        for _ in range(12):
            msgs = [{"role": "system", "content": system}] if system is not None else []
            for t in range(rng.randint(1, 4)):
                msgs.append({"role": "user", "content": rng.choice(contents)})
                if rng.random() < 0.9:
                    msgs.append({"role": "assistant", "content": rng.choice(replies)})
            if msgs[-1]["role"] == "assistant" and rng.random() < 0.8:
                msgs.append({"role": "user", "content": rng.choice(contents)})
            out.append(msgs)
    return out


def cmd_template(args):
    hf = load_hf(args.checkpoint)
    rng = random.Random(args.seed)
    convs = conversations(rng)
    cases, records = [], []
    for msgs in convs:
        for thinking in (False, True):
            for generate in (True, False):
                cases.append(("render", msgs, thinking, generate))
                records.append(rec_json(b"C", {"messages": msgs, "thinking": thinking, "generate": generate}))
    # Continuation: a rendered prompt ending in the generation prompt, then a reply in state,
    # then the native next turn, must equal the template's render of the longer conversation.
    for msgs in convs:
        users = [m for m in msgs if m["role"] == "user"]
        if not users:
            continue
        system = [m for m in msgs if m["role"] == "system"]
        for reply in ("Canberra is the capital.", "line1\nline2", "emoji 👍", "  spaced  "):
            for closed in (True, False):
                nxt = rng.choice(users)["content"]
                cases.append(("continue", system + users[:1], reply, closed, nxt))
                records.append(rec_json(b"N", {"closed": closed, "user": nxt, "thinking": False}))
    native, _, _ = run_native(args.binary, args.checkpoint, records, "template")
    diffs, checked = [], 0
    for case, got in zip(cases, native):
        if case[0] == "render":
            _, msgs, thinking, generate = case
            text = hf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=generate, enable_thinking=thinking)
            ids = list(hf.apply_chat_template(msgs, tokenize=True, add_generation_prompt=generate,
                                              enable_thinking=thinking, return_dict=False))
            checked += 1
            if got["text"] != text or got["ids"] != ids:
                diffs.append((msgs, thinking, generate, text, got["text"], ids == got["ids"]))
        else:
            _, first, reply, closed, nxt = case
            head = hf.apply_chat_template(first, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            whole = hf.apply_chat_template(first + [{"role": "assistant", "content": reply}, {"role": "user", "content": nxt}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
            # An unclosed reply (cut by --max-tokens) is closed by the next turn's text.
            state = head + reply + ("<|im_end|>" if closed else "")
            checked += 1
            if state + got["text"] != whole:
                diffs.append((first, reply, closed, whole, state + got["text"], None))
    for d in diffs[:8]:
        print("DIFF", repr(d[0])[:300], d[1], d[2], "\n  template %r\n  native   %r\n  ids equal %s" % (d[3][-300:], d[4][-300:], d[5]))
    print("template: %d conversations; %d comparisons (rendered text and ids, thinking on/off, with and without "
          "the generation prompt; chat continuation); %d differences" % (len(convs), checked, len(diffs)))
    return 0 if not diffs else 1


# ---- fixture ----

FIXTURE_TRAIN = SENTENCES * 3 + [
    "def main():\n    print('hello world')\n    return 0\n",
    "const std = @import(\"std\");\npub fn main() !void {\n    var x: u32 = 1234567;\n}\n",
    "The the THE they're we'll I'd you've she's it's don't can't won't.",
    "    indented\n\n\n\ttabbed\r\n\r\nwindows line endings  trailing   \n",
]
FIXTURE_CASES = [
    "", "a", "Hello world", "hello world hello", "The the THE", "it's they're we'll I'd you've don't 'S 'ſ 'LL",
    "12345 1 22 333 4444", "١٢٣٤٥ ๑๒๓", "x   y", "x   ", "   ", "\n", "\r\n\r\n", "  \n  x", "a\t\tb",
    " !!!\n\nz", "(abc) [def] {ghi}", "def main():\n    print('hello world')\n", "中文字符 和 日本語 カタカナ",
    "Ελληνικά Русский العربية עברית हिन्दी", "👍🏽 👨‍👩‍👧 🇯🇵", "é ä क्ष", "　x　",
    " nbsp thin narrow", "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think></think>",
    "<|im_start|><|im_end|><|im", "<|im<|im_start|>", "x<think>y</think>z", "<norm>a<norm>", "a<é>b", "aĠXb",
    "hello<|im_end|>hello", "<<|im_end|>>", "'s's's", "x" * 300, " " * 70, "ab" * 100,
]
FIXTURE_RAW = [b"\xff", b"a\xc3", b"\xed\xa0\x80x", b"\xf0\x9f\x98", b"ok\x80\x80ok", b"\xc0\xaf", b"\xf4\x90\x80\x80"]


def cmd_fixture(args):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, Regex, AddedToken
    spec = json.loads((pathlib.Path(args.checkpoint) / "tokenizer.json").read_text())
    pattern = spec["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    tok = Tokenizer(models.BPE(ignore_merges=True))
    tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(pattern), "isolated"),
                                                 pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(FIXTURE_TRAIN, trainer)
    tok.add_special_tokens([AddedToken(c, special=True, normalized=False) for c in
                            ("<|startoftext|>", "<|im_start|>", "<|im_end|>", "<|im")])
    tok.add_tokens([AddedToken("<think>", normalized=False), AddedToken("</think>", normalized=False),
                    AddedToken("<é>", normalized=False), AddedToken("ĠX", normalized=False),
                    AddedToken("<norm>", normalized=True), AddedToken("hello", normalized=False)])
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    doc = json.loads(tok.to_str())
    (out / "tokenizer-fixture.json").write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")))
    cases = []
    inputs = [(s.encode(), s) for s in FIXTURE_CASES] + [(b, b.decode("utf-8", "replace")) for b in FIXTURE_RAW]
    reloaded = Tokenizer.from_str(json.dumps(doc))
    for raw, text in inputs:
        ids = reloaded.encode(text).ids
        cases.append({"b64": base64.b64encode(raw).decode(), "ids": ids,
                      "text": reloaded.decode(ids), "text_special": reloaded.decode(ids, skip_special_tokens=False)})
    (out / "tokenizer-fixture-cases.json").write_text(json.dumps(cases, ensure_ascii=False, indent=0))
    print("fixture: vocab %d, merges %d, added %d, %d cases -> %s"
          % (len(doc["model"]["vocab"]), len(doc["model"]["merges"]), len(doc["added_tokens"]), len(cases), out))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("corpus", "classes", "template", "fixture", "all"))
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("--binary", type=pathlib.Path, default=DEFAULT_BINARY)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--random-ids", type=int, default=5000)
    parser.add_argument("--out", default=str(ROOT / "src/mega/testdata"))
    args = parser.parse_args()
    if args.command == "all":
        codes = [cmd_classes(args), cmd_corpus(args), cmd_template(args)]
        sys.exit(max(codes))
    sys.exit({"corpus": cmd_corpus, "classes": cmd_classes, "template": cmd_template, "fixture": cmd_fixture}[args.command](args))


if __name__ == "__main__":
    main()
