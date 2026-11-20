# `nano` on the command line

`nano` reads a checkpoint folder and answers in text: no Python, no network, no downloads. The tokenizer, the chat template and the sampler are native (`src/mega/tokenizer.zig`, the front end's template in `src/mega/lfm3.zig`, `src/mega/sample.zig`); the commands are in `src/mega/cli.zig`. Today the only model with a front end is LFM3-1.7B (its 4-bit MLX export).

## Install

```sh
zig build -Doptimize=ReleaseSafe          # Zig 0.17, Apple Silicon macOS
install zig-out/bin/nano /usr/local/bin/  # or anywhere on PATH; it is one file
export NANO_MODEL=/path/to/lfm3-1.7b-mlx-4bit
```

The checkpoint folder needs `config.json`, `tokenizer.json` and the weights; `generation_config.json` only for `--sampling model`. `chat_template.jinja` is not read: the template's text-only path is part of the model's front end and is checked against that file.

`MODEL` in every command below is that folder. When it is left out, `nano` uses `$NANO_MODEL`, then `$LFM3_CHECKPOINT`. A first argument is taken as `MODEL` only if it is a folder with `config.json` or `tokenizer.json` in it.

## Commands

```sh
nano run [MODEL] "prompt"               # one reply, streamed to standard output
nano chat [MODEL]                       # a conversation
nano tokenize [MODEL] "text"            # token ids as a JSON array
nano detokenize [MODEL] 597 5205 302    # text of ids
nano --help, nano COMMAND --help, nano --version
```

The engine commands (`inspect`, `eval`, `generate`, `session`, `bench`, `mlx-op`) are unchanged and still take token ids from JSON files; `nano tokenize` writes such a file.

### `run`

```sh
$ nano run "What is the capital of Australia? Answer in one sentence."
The capital of Australia is Canberra.
$ echo "Write a haiku about the sea." | nano run --stats
...
load 0.24 s | prompt 19 tokens, 0.133 s, 143.1 tok/s | first token 0.133 s | reply 26 tokens, 249.4 tok/s | stop: stop-token
```

The prompt is the argument, `--prompt TEXT`, `--file PATH`, or standard input. Only the reply goes to standard output (a final newline is added when standard output is a terminal); messages and `--stats` go to standard error.

| option | |
| --- | --- |
| `--system TEXT` | system message |
| `--max-tokens N` | reply length limit (default 1024) |
| `--raw` | no chat template: the text is the model's input as is (special-token strings in it still become special tokens, as the tokenizers library does) |
| `--think`, `--no-think` | the template's thinking switch (default off, as mlx-vlm does) |
| `--no-thinking-output` | leave the thinking out of the output |
| `--json` | one JSON object at the end instead of a stream: `text`, `thinking`, `stop`, prompt and reply ids and counts, timings |
| `--stats` | load time; prompt tokens, prefill seconds and tokens a second; time to first token; reply tokens and tokens a second (tokens after the first over the time after the first) |
| `--context N` | positions the state holds (default 4096; the model allows far more, memory grows by about 24 KB a position) |
| `--prefill exact\|mlx` | see below |
| `--knobs FILE`, `--knob name=value` | engine knobs (docs/MEGAKERNEL.md): `--knobs` replaces the preset, `--knob` applies on top |

A reply ends at the model's stop token (`<|im_end|>`, from `config.json` through the front end), at `--max-tokens`, or when the context is full, which is said on standard error. Special tokens are never printed.

### `chat`

```sh
$ nano chat
> My name is Ada.
Hello Ada! How can I help you today?
> What is my name?
Your name is Ada. How may I assist you?
```

The model keeps its state between turns and reads only the new tokens: after a reply, the next turn feeds the text the template puts between that reply and the new generation prompt (`\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think></think>`, preceded by `<|im_end|>` if the reply was cut short). Commands: `/reset`, `/system TEXT` (sets the system message and starts over), `/stats`, `/help`, `/exit`; a line starting with `//` sends the text after the first `/`. Ctrl-C stops the current reply and keeps the conversation; Ctrl-D leaves. Line editing is whatever the terminal gives.

When standard input is not a terminal, every non-empty line is one turn and nothing but the replies (one per line) goes to standard output; `--json` writes one JSON object per turn instead, including the ids fed and generated. Options are those of `run` except `--prompt`, `--file` and `--raw`; `--max-tokens` is per reply.

### `tokenize`, `detokenize`

```sh
$ nano tokenize "Hello world, it's 2024! 中文 <|im_end|>"
[35808,1530,20,435,589,229,1217,28,9,50525,4966,229,124900]
$ nano tokenize --chat "What is the capital of Australia? Answer in one sentence." > prompt.json
$ nano generate MODEL --tokens-file prompt.json
$ nano detokenize 597 5205 302 5881 355 2076 59405 22 124900
The capital of Australia is Canberra.
```

`tokenize --chat [--system TEXT] [--think]` wraps the text as one user turn with the generation prompt. `detokenize` takes ids as arguments or on standard input (spaces, commas or a JSON array) and leaves special tokens out unless `--special`.

## The two prefill modes

`run` and `chat` use the tuned knobs, not the reference executor: `src/mega/presets/exact.json` is embedded in the binary at build time.

- `--prefill exact` (default): the prompt is read in batches whose arithmetic is bit-identical to feeding it one token at a time, so a reply is the same however the prompt was chunked.
- `--prefill mlx`: the prompt is read through ports of MLX's batched kernels (`src/mega/presets/mlx-prefill.json`). Several times faster on long prompts; each kernel matches stock MLX bit for bit, so the result is not bit-identical to token-at-a-time reading and a long reply can diverge. Decoding is the same in both modes.

One measurement (M3 Ultra, 6,987-token prompt, 120 reply tokens, one run each, machine shared): exact prefill 9.03 s (774 tokens a second), mlx 1.65 s (4,235 tokens a second); decode 225 and 220 tokens a second; the two replies were the same text.

## Sampling

Greedy is the default: the engine's GPU argmax, deterministic, at full speed. `--temperature T`, `--top-k K`, `--top-p P`, `--repetition-penalty R` and `--seed S` switch to a host sampler over the logits read back from the GPU, applied in Hugging Face's order (penalty over every token in the conversation, temperature, top-k, top-p, then a draw). `--sampling model` takes them from the checkpoint's `generation_config.json` (LFM3: temperature 0.1, top-k 50, repetition penalty 1.1); flags given as well override it. The same seed gives the same reply; the draws are not those of any other implementation.

## Thinking

With `--think` the template's generation prompt ends in `<think>` and the model reasons first, then writes `</think>` and the answer. On a terminal the reasoning is shown dimmed; otherwise it is printed between literal `<think>` and `</think>`; `--no-thinking-output` leaves it out. In `--json` it is the `thinking` field. In a chat with thinking on, the reasoning stays in the model's state for later turns (the template itself would drop earlier turns' reasoning when re-rendering the conversation; `nano` does not re-render).

## Correctness

- The tokenizer is checked against the Hugging Face `tokenizers` library (0.23.3) by `tools/nano_tokenizer_check.py` (run it with the reference Python and `-I`): `corpus` generates 51,299 strings (22.8 MB: source code, written and synthetic prose in 32 scripts, every Unicode general category, every code point, whitespace runs, contractions, numbers, every added token in text, byte soup, invalid UTF-8, inputs up to 1 MB, the empty string) and requires identical ids, plus identical decoded text for the corpus ids and 5,000 random id sequences (plain, keeping special tokens, and streaming); `classes` compares every code point's class under the library's own regex; `template` compares the chat prompt with the checkpoint's `chat_template.jinja` rendered by `transformers` (text and ids) and checks the chat continuation. All three report 0 differences.
- The regex is matched by hand over Unicode 16.0.0 category tables (`src/mega/unicode_tables.zig`, written by `tools/nano_unicode_tables.py`), the version the library's Oniguruma uses; the case-insensitive contractions accept ASCII case pairs and U+017F for `s`, as Oniguruma does.
- Invalid UTF-8 input becomes U+FFFD per maximal invalid subpart (as Python's `"replace"`) before tokenizing; that is what the library sees for such bytes.
- `tools/nano_cli_check.py golden` checks that `nano run` reproduces the `short` golden's greedy tokens, and `chat` that a three-turn piped `nano chat` gives the same reply tokens as `nano session` fed the same ids (reference knobs and the exact preset). Run it under the inference lock.

## Limits

- Text only: tools, images, audio and video in the template are not supported.
- One tokenizer form: byte-level BPE with the Llama 3 split regex, no normalizer, `ByteLevel` decoder, added tokens without `lstrip`/`rstrip`/`single_word`. Anything else in a `tokenizer.json` is refused with a message naming the feature.
- One model front end (LFM3); a model without a chat template works with `--raw` only.
- The first forward of a process includes about 0.1 s of GPU warm-up, which shows in the first time to first token.
- Contexts beyond 4,096 tokens are not covered by any golden.
