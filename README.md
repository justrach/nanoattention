<h1 align="center">nanoattention</h1>

<p align="center"><strong>Inkling 2 Mini, Flint, and Liquid AI inference, close to the metal.</strong></p>

<p align="center">A multi-model native Zig + Metal engine for Apple Silicon.<br>Explicit memory. Persistent KV caches. Lean-checked execution contracts.</p>

<p align="center">
  <img alt="Apple Silicon macOS" src="https://img.shields.io/badge/platform-Apple%20Silicon-555">
  <img alt="Built in Zig 0.17" src="https://img.shields.io/badge/built%20in-Zig%200.17-f7a41d?logo=zig&logoColor=white">
  <img alt="GPU backend Metal" src="https://img.shields.io/badge/GPU-Metal-555">
  <img alt="Original code MIT licensed" src="https://img.shields.io/badge/original%20code-MIT-44cc11">
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#what-it-does">What it does</a> ·
  <a href="#inkling-2-mini-nano">Inkling 2 Mini</a> ·
  <a href="#use-with-nanoharness--codegraff">NanoHarness &amp; CodeGraff</a> ·
  <a href="#performance">Performance</a> ·
  <a href="#correctness-and-proofs">Proofs</a> ·
  <a href="#development">Development</a> ·
  <a href="#acknowledgements--credits">Credits</a>
</p>

Run [Thinking Machines Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini),
[RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), and Liquid AI
(`LFM3-1.7B` / `LFM2.5-1.2B`) locally on Apple Silicon with a small, inspectable
Zig + Metal stack. `nano` runs Inkling 2 Mini and Liquid AI models through a
model-agnostic megakernel; `flint` drives Recursive Flint. Zig maps the checkpoint,
owns buffers and KV/ring state, and schedules execution; Metal kernels run attention,
quantized projections, expert routing, and the vocabulary head on the GPU.

**The native runtime does not link MLX, PyTorch, or CUDA.** Optional Python tools
handle tokenization, chat templates, and the local HTTP bridge—not model math.
This is a focused, single-sequence inference engine, not a general tensor framework.

## Quick start

### 1. Build the engine

You need an Apple Silicon Mac, Xcode or Command Line Tools with the macOS SDK,
**Zig 0.17.0**, **Lean/Lake 4.35.0**, and **Python 3.9+**. The Lean project uses
`Std` only. Builds check the generated proof bindings before compiling the engine.

```sh
git clone https://github.com/justrach/nanoattention.git
cd nanoattention
zig build -Doptimize=ReleaseSafe
```

This repository is private; cloning requires access. Building produces both
`zig-out/bin/nano` (the model-agnostic megakernel for Inkling 2 Mini and Liquid AI
models) and `zig-out/bin/flint` (the dedicated engine for Recursive Flint). Both
binaries (`nano-aarch64-macos.tar.gz` and `flint-aarch64-macos.tar.gz`) also ship
in [GitHub Releases](https://github.com/justrach/nanoattention/releases) starting at
`v0.2.0`. There is no runtime or model auto-install step.

### 2. Get the weights

Flint is developed by **Recursive**. The official model and original BF16
checkpoint are on [Hugging Face: RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint).
The weights retain Recursive's **[Apache-2.0 license](https://huggingface.co/RecursiveSI/flint/blob/main/LICENSE)**;
we do not relicense them under this engine's MIT license.

[codegraff.com/flint](https://codegraff.com/flint) is an **unofficial download mirror**
for the tested, modified MLX-style affine-INT4 export—not a byte-for-byte mirror of
the original BF16 checkpoint or a model authored by us. It retains the same
upstream Apache-2.0 terms, original license, attribution, and modification notice;
no endorsement by Recursive is implied.

The mirror includes all five weight shards, tokenizer, chat template,
configuration, license, and checksums. Allow roughly **22.58 GB of disk space**,
plus enough unified memory for weights, working buffers, and your chosen context.
The download does not install or start an inference server.

Point the commands below at the directory containing the complete checkpoint:

```sh
./zig-out/bin/flint inspect /path/to/flint-int4
```

`inspect` checks model configuration; it is not a full checkpoint integrity check.
Use the download's checksums to verify the files.

### 3. Generate text

In a Python environment with `transformers` and `tokenizers` installed:

```sh
python3 -B tools/chat.py /path/to/flint-int4 \
  "Write a Python binary search function." \
  --context 2048 --max-tokens 64 --staged-row-tile 2
```

The helper uses only the checkpoint's local tokenizer and chat template. All
inference runs in the native executable. Generation is greedy.

<details>
<summary>Work directly with token IDs</summary>

Create `prompt.json` as an array of IDs produced by the checkpoint's tokenizer,
then run:

```sh
./zig-out/bin/flint generate /path/to/flint-int4 \
  --tokens-file prompt.json --context 2048 --max-tokens 64 --staged-row-tile 2
```

The result is a JSON object containing generated `tokens`. EOS is included when
reached. `--context` bounds prompt plus generated positions. Raw text is not a
valid token-ID prompt. `flint eval` exposes vocabulary logits for diagnostics.

</details>

## What it does

- **Native quantized inference.** Load mmap-backed safetensors, including sharded
  MLX affine INT4/INT8 exports, without expanding the full checkpoint to FP32.
- **GPU-side model work.** Packed projections, fused QKV launches, routed/shared
  experts, normalization, attention, and greedy selection use Metal kernels.
- **Persistent conversations.** A JSONL session keeps weights and per-layer KV
  alive. Compatible follow-up turns reuse the exact cached token prefix; changed
  histories trigger an explicit reset rather than a pretend cache hit.
- **Long-context attention.** Sliding layers use bounded ring caches; global
  layers retain the full configured prefix. Flint applies RoPE to sliding layers
  and NoPE to global layers.
- **Checked kernel contracts.** Lean proves selected integer scheduling and
  indexing properties; generated bindings connect those contracts to Zig and Metal.
- **Measured experiments.** Optional BF16 global KV, history-based speculation,
  and tuning candidates are separate from the ordinary FP32-KV decode baseline.

The loader also supports F32/F16/BF16 tensors and mixed-precision embeddings.
NVIDIA NVFP4 is not supported. The tested INT4 export has received more full-model
validation than the published BF16 checkpoint.

## Use with NanoHarness & CodeGraff

`justrach/nanoharness` integrates both `flint` and `nano` directly from **Settings → Agents** on Apple Silicon Macs (exposing `nanoattention/flint` and `nanoattention/inkling-2-mini` over a managed local OpenAI-compatible loopback server with unified-memory sizing).

Keep the model local while CodeGraff reads files, runs tools, and handles the coding
conversation. The owned launcher starts one loopback adapter and one persistent
native worker, then selects `nanoattention/flint` for the chosen workspace (`tools/inkling_adapter.py` and `tools/inkling_agent.py` run Inkling 2 Mini through the same provider slot).

```sh
python3 tools/guarded_run.py --lock /path/to/shared-inference.lock --timeout 1400 -- \
  python3 -B tools/graff_native.py /path/to/flint-int4 \
  --graff /path/to/graff --workspace /path/to/coding-workspace \
  --context 32768 --max-tokens 4096 --timeout 1200 -- \
  -p "Add tests for the parser, then fix any failures."
```

Use the tokenizer-enabled Python environment. Omit the arguments after `--` for
an interactive terminal session. Run local inference jobs sequentially under the
same exclusive lock.

Compatible Graff builds use the built-in local provider. Older binaries use a
temporary project router, without overwriting an existing router or changing global
provider settings. The launcher cleans up its owned processes on exit. Normal
Graff tool permissions still apply; the launcher is not itself an OS sandbox.

The Python bridge provides OpenAI-compatible chat and function-tool translation;
**it never executes tools**. It accepts one active model request at a time, and
SSE output is buffered rather than token-live. See the
[setup, lifecycle, and verification guide](docs/CODEGRAFF.md).

## Performance

Decode speed depends on context, kernel configuration, and workload. The benchmark
reports retain configurations, methodology, negative results, and correctness gates
rather than treating the fastest run as a universal speed claim.

| Workload | Recorded native decode rate | Evidence |
| --- | ---: | --- |
| Short coding prompt, 21-token prefix / 64 teacher-forced steps | 120.40 tok/s | [Matched local MLX comparison](benchmarks/README.md) |
| Seeded random raw IDs, filled 16K / 256 ordinary steps | 106.09 tok/s | [Raw decode experiments](benchmarks/raw-decode.md) |
| Seeded random raw IDs, filled 32K / 256 ordinary steps | 93.19 tok/s | [Raw decode experiments](benchmarks/raw-decode.md) |

These are historical measurements on an **Apple M3 Ultra, 60 GPU cores, 256 GB
unified memory**, with the INT4 checkpoint and native FP32 activations/KV. Loading,
prefill, and warmup are excluded. Rows come from different trials/configurations;
they are not a paired comparison or a fresh benchmark of this documentation change.
Raw teacher-forced rates are not free-running coding-agent throughput.

The short-context matched trial measured stock MLX FP32 at 96.84 tok/s and BF16 at
91.96 tok/s. That establishes only that workload: native serial prefill remains
substantially slower than MLX's batched prefill. Recursive's published `rec`
numbers use different hardware and protocols; **this is not a matched win over
`rec`, overall MLX parity, or a claim of 130 tok/s at long context**.

More: [long-context sessions](benchmarks/larger-context-sessions.md),
[raw attention tuning](benchmarks/raw-attention-hillclimb.md), and
[prepared-input experiments that did not improve speed](benchmarks/raw-prepared-inputs.md).

<details>
<summary>Experimental modes</summary>

- **BF16 global KV** reduces global-cache storage, while sliding KV and model math
  remain FP32. It can change logits and generated tokens; it is not stock MLX BF16
  inference. See [measurements](benchmarks/bf16-kv.md).
- **Greedy speculation** uses committed-history drafts and batched target
  verification with sliding-cache rollback. It is disabled by default, preserves
  target greedy tokens in the tested cases, and can be slower on real coding
  tasks. See [contracts and limitations](docs/SPECULATIVE.md) and
  [workload-dependent results](benchmarks/speculative-hillclimb.md).
- **Resident execution** interprets the generated plan in one GPU dispatch, with
  `--resident-workers 1..32` and checked dimension/context limits. It is slower than
  the optimized staged path. Ordinary staged execution uses multiple kernels;
  one command-buffer submission is not one kernel.

</details>

## Correctness and proofs

Correctness is checked at several levels: headless Metal tests, synthetic
checkpoints against independent NumPy, bounded full-model MLX comparisons, and
Lean-generated binding checks. Numerical agreement is reported with its tested
precision, context, and tolerance—not as a blanket quality or bitwise-MLX claim.

For long-context coding, recorded sequential FP32 MLX checks cover every greedy
choice through 28,623 cached tokens and six full-vocabulary boundary rows, not
all logits at every position. Raw kernel candidates use separate bitwise admission
against the ordinary native reference. Neither establishes full-window task quality.

**Lean proves selected integer properties:** execution-plan order, guarded coverage,
index bounds, packed INT4 reconstruction, and launch geometry. Builds verify exact
bindings consumed by Zig and Metal. **It does not prove floating-point equivalence,
whole-shader correctness, GPU memory semantics, forward progress, or speed.**
See [proof scope and assumptions](proofs/README.md).

## Liquid AI models (`nano`)

`zig build` also builds `zig-out/bin/nano`, a separate Zig/Metal inference engine
for the tested 4-bit MLX checkpoints of Liquid AI's LFM3-1.7B and
LFM2.5-1.2B-Instruct. It does not link MLX. Both models support raw-token
inference; the native text and chat commands currently support LFM3 only.

```sh
zig-out/bin/nano inspect "$CHECKPOINT"
zig-out/bin/nano generate "$CHECKPOINT" --tokens-file prompt.json --max-tokens 128 \
  --knobs src/mega/presets/exact.json --knob decode_attention=3 \
  --knob decode_q4_static=true --knob fuse_add_norm=true
```

The optimizations are opt-in. At a 4,096-token prompt on the tested M3 Ultra,
LFM3 beat pipelined MLX on decode; LFM2.5 did not reliably beat it. MLX was
faster on prefill for both. See [measurements and reproduction commands](benchmarks/liquid-attention.md),
[the engine design](docs/MEGAKERNEL.md), and [LFM3 CLI usage](docs/CLI.md).

## Inkling 2 Mini (`nano`)

`nano` natively runs [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (`inkling2_loop`: 24 physical layers expanded to 48 logical loop visits, 128 routed NVFP4/MXFP8 experts, relative & depth attention, BF16 dense projections, ~19.5 GB checkpoint) with **zero MLX or PyTorch runtime dependency**:

```sh
./zig-out/bin/nano inspect "$CKPT"
./zig-out/bin/nano bench "$CKPT" --knobs src/mega/presets/inkling.json \
  --context 4200 --prompt-tokens 4096 --decode-tokens 64 --repeats 3
./zig-out/bin/nano session "$CKPT" --context 32768 --knobs src/mega/presets/inkling.json
python3 tools/inkling_adapter.py "$CKPT" --engine native --binary zig-out/bin/nano
```

Matched 3-repeat `nano bench` throughput on an **Apple M3 Ultra (60 GPU cores, 256 GB unified memory)** (`--decode-tokens 64 --repeats 3 --decode-detail 1`, `0` gate redos) with [`src/mega/presets/inkling.json`](src/mega/presets/inkling.json) — incorporating six megakernel optimization designs inspired by [`qwegnai/qwegn`](https://github.com/qwegnai/qwegn) ([`@Andrei-cloud`](https://github.com/Andrei-cloud) / QwegnAI, Apache-2.0):

| Prompt (context) | Long-Context Base (Pre / Dec) | Pre-Qwegn `megakernel-v1` (Pre / Dec) | `inkling.json` `v0.2.0` (Pre / Dec) | Speedup vs Base |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 (`4,200`) | `1,263.44` / `39.82` tok/s | `1,403.25` / `54.60` tok/s | **`1,570.07` / `64.49` tok/s** | **+24.3% / +62.0%** |
| 16,384 (`16,512`) | `1,197.98` / `38.37` tok/s | `1,330.69` / `51.83` tok/s | **`1,477.74` / `61.24` tok/s** | **+23.4% / +59.6%** |
| 32,768 (`32,900`) | `1,069.70` / `36.72` tok/s | `1,249.00` / `48.74` tok/s | **`1,376.23` / `58.06` tok/s** | **+28.7% / +58.1%** |

Against published [`ml-explore/mlx-lm#2086`](https://github.com/ml-explore/mlx-lm/pull/2086) numbers (authored by **Jun Kim** [`@jundot`](https://github.com/jundot) with contributions from Pedro Cuenca and Lucas Gelfond, run on an **80-core M3 Ultra / 512 GB**, reporting `1,516.3` / `32.07` tok/s at 4K and `1,472.3` / `32.50` tok/s at 16K), `nano v0.2.0` on 60 cores achieves **`2.01×`** higher decode at 4K and **`1.88×`** at 16K, and exceeds MLX's raw 80-core prefill at both 4K (`+3.5%`) and 16K (`+0.4%`).

#### `v0.3.0`: DFlash 2 Speculative Decoding + AngelSpec `D-cut` + Lean-Maxx Kernels

`nano v0.3.0` wires native [`DFlash2DraftModel`](src/mega/dflash2.zig) speculative drafting (`--draft`), LoopCD contrastive decoding (`--loop-passes 3 --loopcd hidden --loopcd-omega 0.5`), and **AngelSpec `D-cut`** adaptive verification depth pruning ([`dflash_dcut`](src/mega/knobs.zig#L76) + unified target `bf16_verify_mma` enabled in [`src/mega/presets/inkling.json`](src/mega/presets/inkling.json)):

```sh
./zig-out/bin/nano session "$CKPT" --knobs src/mega/presets/inkling.json \
  --context 8192 --loop-passes 3 --loopcd hidden --loopcd-omega 0.5 \
  --draft /path/to/inkling-2-mini-dflash2
```

- **Why `D-cut` wins without retraining:** Because Inkling 2 Mini is a 128-expert MoE target, verifying a full `r=8` draft block (`45.6 ms`) costs **`2.72×`** more than `r=1` (`16.8 ms`) while DFlash 2 takes `~4.1–4.7 ms/batch`. Computing exact post-selector softmax confidence ($c_t \in [0, 1]$ in [`df_top16_part` / `df_top16_merge` / `df_select_path`](src/mega/dflash2.metal)) and truncating low-confidence tails online via $r^* = \arg\max_r U(r)/(t_{\text{draft}} + t_{\text{verify}}(r))$ improves every workload while preserving **100% exact greedy parity** (`base == draft`).
- **Metal kernel optimizations:** Target path adds `mk_bf16_mma_m8n32_softcap_direct` (unified rows `1..8` `unembed`), `splitk_vec8`, fused `mk_route_topk_group_tiles`, hoisted grouped MoE NVFP4/MXFP8 decode, batched `mk_store_inkling_short_conv_ring_batch`, and `mk_copy_slots4`. Draft path adds `df_bf16_mma_m8n32_direct`, `df_multi_bf16_mma_m8n16_direct` (`Q+K+V` / `K+V`), `df_bf16_gate_up_swiglu_mma`, `df_bf16_linear_splitk_rows_vec8`, `df_grouped_dyn_conv_vec4`, `df_kv_ring_store_vec4`, zero-barrier `df_rmsnorm_rope` / `df_rmsnorm_rope_ring_store`, `df_attention_gqa4`, and `d2_concurrent` overlaps.

Verified coherent 3-pass **Suite A** (`nano session`, M3 Ultra 60-core / 256 GB, `--knobs src/mega/presets/inkling.json`, `0` parity failures, `SafeAllocator` leak = `False`):

| Config | Workload | Base (`tok/s`) | Baseline Draft (`tok/s`) | **Final `v0.3.0` Draft (`tok/s`)** | Speedup vs Base | `mean_step` | Parity / Coherence |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: | :--- |
| **`3p_hidden`** (`--loop-passes 3 --loopcd hidden --loopcd-omega 0.5`) | Coding (`54` tok in, `192` out) | `64.76` | `90.18` | **`104.35`** | **`1.61×`** | `4.00` | `True` / Clean |
| **`3p_hidden`** | Math (`49` tok in, `192` out) | `64.42` | `123.08` | **`125.56`** (`131.07` A/B) | **`1.95×`** | `5.49` | `True` / Clean |
| **`3p_hidden`** | 4K Code (`4,067` tok in, `256` out) | `62.33` | `75.07` | **`88.48`** | **`1.42×`** | `3.28` | `True` / Clean |
| **`3p_off`** (`--loop-passes 3 --loopcd off`) | Coding | `60.03` | `89.18` | **`107.00`** | **`1.78×`** | `4.17` | `True` / Clean |
| **`3p_off`** | Math | `64.33` | `124.41` | **`130.36`** (`132.41` A/B) | **`2.03×`** | `5.65` | `True` / Clean |
| **`3p_off`** | 4K Code | `62.38` | `54.81` | **`83.83`** | **`1.34×`** | `2.98` | `True` / Clean |

*2-pass limit (`[5, 23, 29]` remap):* `dflash2` automatically remaps 2-pass 37-trace schedules to `{5, 23, 29}` (same physical layers `[4, 10, 16]` as 3-pass `[5, 23, 41]`). While raw 2-pass throughput is fast (`108.31–113.68 tok/s` coding, `138.78 tok/s` loop-inflated `code_4k`), `--loop-passes 2` (`--loopcd off` and `--loopcd hidden --loopcd-omega 0.15..0.40`) still degrades into syntax glitches and repetition loops on math/4K code and fails `32K` needle retrieval (`0/3`), so **3-pass** (`3p_hidden` / `3p_off`) is the recommended coherent setting.

#### Long-Context Q4 Multi-Head LSA, Cross-Loop Reuse & `.nkv` Cartridges (`32K / 64K / 131K`)

Combine `--draft` with `--knob lsa=true --knob lsa_cross_loop_reuse=true` (and optional `--kv-in` / `--kv-out` `.nkv` persistent KV cartridges) for high-speed coherent speculative decode up to 131K–1M context indexing:

```sh
./zig-out/bin/nano session "$CKPT" --knobs src/mega/presets/inkling.json \
  --context 131072 --loop-passes 3 --loopcd hidden --loopcd-omega 0.5 \
  --knob lsa=true --knob lsa_cross_loop_reuse=true \
  --draft /path/to/inkling-2-mini-dflash2
```

Verified **Suite B** multi-turn needle retrieval + technical summary (`32K / 64K / 131K`, `SafeAllocator` leak = `False`):

| Context (`prompt_tokens`) | Config | Needle | Exact Parity (`T1,T2`) | Linear Prefill (`tok/s`) | Turn 1 (`base -> draft tok/s`, `ms`) | Turn 2 (`base -> draft tok/s`, `ms`) |
| ---: | :--- | :---: | :---: | ---: | ---: | ---: |
| **`32K`** (`32,182`) | **`3p_hidden`** | `True` | `(True, True)` | `1,353.8` | `59.0 ->` **`69.7`** (`2.67`) | `60.8 ->` **`59.6`** (`2.00`) |
| **`32K`** (`32,182`) | **`3p_off`** | `True` | `(True, True)` | `1,358.2` | `60.9 ->` **`68.9`** (`2.67`) | `60.4 ->` **`70.8`** (`2.61`) |
| **`64K`** (`64,601`) | **`3p_hidden`** | `True` | `(True, True)` | `1,380.2` | `59.9 ->` **`73.9`** (`2.78`) | `56.7 ->` **`60.9`** (`2.13`) |
| **`64K`** (`64,601`) | **`3p_off`** | `True` | `(True, True)` | `1,331.2` | `56.0 ->` **`75.3`** (`2.78`) | `56.7 ->` **`77.3`** (`2.98`) |
| **`131K`** (`129,125`) | **`3p_hidden`** | `True` | `(True, True)` | `1,299.5` | `59.8 ->` **`65.8`** (`2.46`) | `60.2 ->` **`64.9`** (`2.56`) |
| **`131K`** (`129,125`) | **`3p_off`** | `True` | `(True, True)` | `1,328.1` | `60.3 ->` **`68.1`** (`2.46`) | `60.7 ->` **`70.4`** (`2.56`) |

#### `v0.3.1`: Native Inkling CLI (`--draft` + `o200k_base`) & BF16 MMA LoopCD Dual-Unembed

`nano v0.3.1` brings zero-Python native text CLI & chat directly to Inkling 2 Mini with DFlash 2 speculative drafting and fixes the `logits` / `logits-adaptive` LoopCD verification bottleneck:

- **Native Inkling `run` & `chat` (`--draft` + `o200k_base`):** Added Unicode 16.0.0 `o200k_base` pre-tokenization ([`src/mega/tokenizer.zig`](src/mega/tokenizer.zig), [`src/mega/unicode_tables.zig`](src/mega/unicode_tables.zig) — verified `0` ID diffs across `49,939` strings and `0` decoded-text diffs across `164,817` checks vs HuggingFace `tokenizers`), Inkling chat template ([`src/mega/inkling2_loop.zig`](src/mega/inkling2_loop.zig)), auto-selected [`src/mega/presets/inkling.json`](src/mega/presets/inkling.json), live streaming `--draft` in [`src/mega/cli.zig`](src/mega/cli.zig) / [`src/mega/dflash2.zig`](src/mega/dflash2.zig), and `--draft` / `--loopcd` / `--lsa` flags in [`tools/inkling_adapter.py`](tools/inkling_adapter.py):

```sh
./zig-out/bin/nano run "$CKPT" --draft /path/to/inkling-2-mini-dflash2 \
  --loop-passes 3 --loopcd hidden --loopcd-omega 0.5 "Explain Metal simdgroup_matrix."
./zig-out/bin/nano chat "$CKPT" --draft /path/to/inkling-2-mini-dflash2 \
  --loop-passes 3 --loopcd hidden --loopcd-omega 0.5
```

- **BF16 MMA Dual-Unembed LoopCD Kernels:** Single-weight-read hardware BF16 `simdgroup_matrix` dual-unembed kernels (`mk_bf16_mma_m8n{16,32}_softcap_{loopcd,dual}_direct` in [`src/mega/inkling_fast.metal`](src/mega/inkling_fast.metal) + [`src/mega/exec.zig`](src/mega/exec.zig)) accelerate `.head_softcap_loopcd` (`rows=1..8`), verified on M3 Ultra (`exact=True`, `leak=False`):

| Config | Coding (`tok/s`, vs Base) | Math (`tok/s`, vs Base) | 4K Code (`tok/s`, vs Base) |
| :--- | ---: | ---: | ---: |
| **`3p_logits`** (`--loop-passes 3 --loopcd logits --loopcd-omega 0.5`) | `105.85 ->` **`117.11`** (`1.79×`) | `85.92 ->` **`104.93`** (`1.60×`) | `40.88 ->` **`62.38`** (`+52.6%`, `0.99×`) |
| **`3p_adaptive`** (`--loop-passes 3 --loopcd logits-adaptive --loopcd-omega 0.7`) | `106.79 ->` **`112.14`** (`1.77×`) | `70.56 ->` **`96.37`** (`1.52×`) | `33.58 ->` **`72.16`** (`2.15×` faster, `1.15×` vs base) |
| **`3p_hidden`** (`--loop-passes 3 --loopcd hidden --loopcd-omega 0.5`, ref) | **`103.36`** (`1.57×`) | **`133.26`** (`2.02×`) | **`92.17`** (`1.46×`) |

#### Canonical Eval & Perf Suite (`tools/nano_eval.py`)

[`tools/nano_eval.py`](tools/nano_eval.py) is the canonical zero-dependency harness for measuring both task quality and native Metal throughput (`nano tokenize --batch`, `nano session`, `nano bench`) side by side:

- **Suites:** `core` (executable Python coding including an instrumented `O(log n)` binary-search read-count check, modular/algebraic math, strict JSON extraction, and a 3,990-token Zig/Metal audit), `long` (32K/64K/131K multi-turn needle retrieval + summary with auto LSA), `scale` (256K–1M), and `bench` (raw synthetic `nano bench`).
- **Gates:** Grades task outputs, flags repetition loops / tail-diversity collapses via coherence checks, enforces `100%` exact `base == draft` greedy parity and `0` leaks, separates `Coherent Decode (tok/s)` from broken fast modes, and runs CI regression checks via `compare --gate`.

Verified 2-repeat M3 Ultra `core` baseline ([`benchmarks/inkling-eval-v031.md`](benchmarks/inkling-eval-v031.md), `--mode both --repeats 2`):

| Config | Pass | Coherent | Parity | Leaks | Prefill (`tok/s`) | Coherent Decode (`tok/s`) | 4K+ (`tok/s`) | `mean_step` | Speedup |
| :--- | :---: | :---: | :---: | :---: | ---: | ---: | ---: | :---: | :---: |
| **`3p_hidden`** | **6/6** | 6/6 | 6/6 | 0 | `1,153.9` | **`110.23`** | **`92.43`** | `4.39` | **`1.69×`** |
| **`3p_off`** | **6/6** | 6/6 | 6/6 | 0 | `1,155.1` | **`107.90`** | `90.52` | `4.18` | `1.66×` |
| **`3p_adaptive`** | **6/6** | 6/6 | 6/6 | 0 | `1,117.7` | **`99.98`** | `79.71` | `3.78` | `1.54×` |
| **`2p_off`** | `0/6` | `2/6` | 6/6 | 0 | `1,567.6` | `117.81` (`132.98` all) | `152.08` *(loop)* | `4.06` | `1.55×` |

```sh
python3 tools/nano_eval.py run "$CKPT" --draft "$DRAFT" \
  --suite core --config 3p_hidden,3p_off,3p_adaptive,2p_off --mode both --repeats 2 \
  --out benchmarks/inkling-eval-v031.json --md benchmarks/inkling-eval-v031.md
python3 tools/nano_eval.py compare benchmarks/inkling-eval-v031.json candidate.json --gate
```

See [`src/mega/presets/inkling.json`](src/mega/presets/inkling.json), the MAP-Elites QD grid across 4K/16K/32K in [`benchmarks/inkling-qd/elites.md`](benchmarks/inkling-qd/elites.md), the Lean 4 contracts in [`proofs/Inkling.lean`](proofs/Inkling.lean) (`59` public / `65` total theorems) and [`proofs/Speculative.lean`](proofs/Speculative.lean) (`17` theorems), and [`docs/MEGAKERNEL.md`](docs/MEGAKERNEL.md).

**Parity & limits:** `inkling.json` preserves exact greedy token parity across `bf16_verify_mma` + `dflash_dcut` (`0` parity failures across Suite A and Suite B `32K/64K/131K`), is bit-identical (`max_abs = 0.0`) on `bf16_verify_mma=false` to pre-Qwegn `megakernel-v1` across `cake_toks` (`feed=1, 8`), `real_512` (`feed=512`, `560/560` top-1, `32/32` greedy), and `long_real` (`feed=512`, `568/568` top-1, `32/32` greedy), matches `64/64` greedy tokens on `cake_toks` against baseline, and passes [`tools/nano_inkling_attention_check.py`](tools/nano_inkling_attention_check.py) (`24/24` top-1 agreement across six arm pairs). Native `--draft` DFlash 2 speculative decoding and the native `inkling2_vision` image encoder (`--image` / `--max-image-tokens` in `nano run`/`chat`, `images` in `nano session`, OpenAI `image_url` in `tools/inkling_adapter.py`, and `nano_generate_multimodal` in `include/nano.h`) are wired; the checkpoint's bundled `mtp.safetensors` drafter remains separate.

## Development

Run from the repository root. NumPy is needed for synthetic numerical checks;
MLX/MLX-LM are optional reference-test dependencies, not runtime dependencies.

```sh
zig build test -Doptimize=ReleaseSafe
zig build test-mega -Doptimize=ReleaseSafe
zig build proofs
python3 -B proofs/test_verifier.py
zig test --dep plan -Mroot=proofs/check_bindings.zig -Mplan=src/resident_plan.zig
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tools -p 'test_*.py'
python3 tools/verify.py zig-out/bin/flint --staged-row-tile 2
```

| Location | Responsibility |
| --- | --- |
| [`src/model.zig`](src/model.zig) | Model execution, expert scheduling, and persistent KV state |
| [`src/weights.zig`](src/weights.zig) | Checkpoint mappings and quantized tensor descriptors |
| [`src/metal.zig`](src/metal.zig), [`src/metal.m`](src/metal.m) | Metal bridge, cached pipelines, barriers, and completion handling |
| [`src/`](src/) | Metal kernels, generated plans, and native tests |
| [`proofs/`](proofs/) | Lean contracts, emitters, and binding verification |
| [`tools/`](tools/) | Tokenizer/HTTP bridges, parity checks, and benchmark drivers |
| [`benchmarks/`](benchmarks/) | Reproducible measurements and rejected experiments |

### Current boundaries

Apple Silicon only. Single sequence, inference only. No training/autograd, general
lazy tensor graph, multi-device execution, or concurrent multi-user batching.
The Flint executable has no native tokenizer/template renderer. Its ordinary
inference is greedy, not Flint's recommended stochastic sampling, and its Python
adapter rejects unsupported sampling and reasoning controls. Full batched prefill
for Flint and broad model-quality parity remain unverified. mmap avoids an extra full weight copy; it
does not remove physical-memory or Metal-buffer limits.

## Acknowledgements & credits

`nanoattention` builds directly on ideas, kernel designs, and reference implementations from the open-source Apple Silicon ML community:

- **Qwegn** ([`qwegnai/qwegn`](https://github.com/qwegnai/qwegn)) — authored by **[`@Andrei-cloud`](https://github.com/Andrei-cloud)** / QwegnAI (Apache-2.0): source of the six Inkling 2 Mini optimization designs adapted in `v0.2.0` (`expert_fp_lut`, `expert_compact_tiles`, `expert_bf16_io`, `fuse_qkvr_prep`, `relative_inline_bias`, and `fuse_merge_norm`) across [`src/mega/inkling_fast.metal`](src/mega/inkling_fast.metal), `src/mega/plan.zig`, `src/mega/exec.zig`, and `proofs/Inkling.lean`.
- **`mlx-lm` Inkling 2 support** ([`ml-explore/mlx-lm#2086`](https://github.com/ml-explore/mlx-lm/pull/2086) / [`jundot/mlx-lm`](https://github.com/jundot/mlx-lm)) — authored by **Jun Kim** ([`@jundot`](https://github.com/jundot)), with contributions from **Pedro Cuenca** ([`@pcuenca`](https://github.com/pcuenca)) and **Lucas Gelfond** ([`@lucasgelfond`](https://github.com/lucasgelfond)) (MIT, Apple Inc.): batched relative attention tile kernel adapted in [`src/mega/inkling_tiles.metal`](src/mega/inkling_tiles.metal) and the `mlx-lm` reference baseline.
- **MLX** ([`ml-explore/mlx`](https://github.com/ml-explore/mlx)) — Apple Inc. and the MLX contributors (MIT): INT4 `qmv_fast` / `qdot`, vector SDPA, Steel GEMM, and Steel attention kernel designs.
- **Recursive `rec`** ([`recursive-oss/rec`](https://github.com/recursive-oss/rec)) — Recursive (**Chris Cummins** [`@ChrisCummins`](https://github.com/ChrisCummins), **Audrey Tang** [`@audreyt`](https://github.com/audreyt), **Oliver Yu** [`@tc3oliver`](https://github.com/tc3oliver); Apache-2.0): resident decode and atomic arrival/phase barrier design adapted in [`src/resident.metal`](src/resident.metal).
- **AngelSpec** ([`Tencent/AngelSpec`](https://github.com/Tencent/AngelSpec)) — source of the `D-cut` adaptive speculative verification depth truncation algorithm adapted in [`src/mega/dflash2.zig`](src/mega/dflash2.zig) and [`src/mega/dflash2.metal`](src/mega/dflash2.metal).
- **Model & architecture authors** — **Thinking Machines Lab** ([Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)), **Recursive** ([Flint](https://huggingface.co/RecursiveSI/flint), Apache-2.0), and **Liquid AI** (`LFM3-1.7B` / `LFM2.5-1.2B-Instruct`).

## License

Original code and contributions are **[MIT licensed](LICENSE)**. Adapted Recursive
`rec` and Qwegn ([`qwegnai/qwegn`](https://github.com/qwegnai/qwegn)) components
retain Apache-2.0 terms; adapted MLX and `mlx-lm` components retain MIT terms
and notices. The top-level MIT license does not replace those upstream obligations.
Model weights are not bundled and remain separately licensed; Recursive Flint,
Thinking Machines Inkling 2 Mini, and Liquid AI checkpoints retain their
respective upstream licenses.

See [third-party notices](THIRD_PARTY_NOTICES.md), [provenance](PROVENANCE.md), and
[`licenses/`](licenses/).

<p align="center"><sub>Built in Zig · Executed on Metal · Contracts checked in Lean</sub></p>
