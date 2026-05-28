# VcLLM Reproduction Plan

**Operational commands and troubleshooting** live in [`README.md`](README.md). This document is the **roadmap and phase checklist** for research reproducibility—not a substitute for the runbook.

**Snapshot (maintainer note):** update the milestone tables when merging features; keep code references aligned with actual filenames under the repo root.

---

## 0. Project Goal

Reproduce the core mechanisms of **VcLLM (Video Coded LLM)** / **LLM.265** starting from small models (`EleutherAI/pythia-160m`), not from LLaMA-3-70B-scale builds.

**Strategy**

1. Validate **offline** tensor compression: `Tensor → Quantization → Video Codec → Tensor Reconstruction`.
2. Integrate the codec path into **inference** (weights, KV cache); later into **training / distributed** paths.
3. Grow a modular codebase that can scale to larger models and multi-node experiments.

**Primary codec stack (paper-aligned, non-negotiable)**

VcLLM’s **core systems claim** is **fixed-function video hardware** on the datapath: tensors are compressed and decompressed through **real GPU video IP**, not a CPU reimplementation.

- **Encode:** NVIDIA **NVENC** only (`hevc_nvenc` via FFmpeg). **No CPU-encoder fallback** is part of the reproduction story for this project.
- **Decode:** NVIDIA **NVDEC** only (FFmpeg CUDA / **CUVID-class** hardware decode, e.g. `hevc_cuvid`). **CPU software decode fallback is explicitly rejected** for paper-aligned experiments and should not be used to claim successful VcLLM reproduction.

If NVENC or NVDEC is unavailable, treat the **machine / FFmpeg build / container GPU pass-through** as broken and fix it—do not substitute libx265 or software HEVC decode and still call it a systems reproduction of VcLLM.

---

## Non-Goals

- Do **not** anchor the first iteration on 70B-scale reproduction or paper-final absolute numbers.
- Do **not** block the MVP on pipeline-parallel / data-parallel compression (Phases 4–5)—those are **explicit later milestones**.
- **CPU-only or “soft decode” setups are not valid targets** for claiming VcLLM / LLM.265 systems reproduction in this repo: **hardware encode + hardware decode** are required for official experiment logs. (Lightweight unit tests on pure math may still run on CPU; **codec integration is GPU/NVENC/NVDEC-only by policy**.)

---

## Initial Model Targets

| Stage | Model | Purpose |
|-------|--------|---------|
| MVP / single-node | `EleutherAI/pythia-160m` | Codec smoke tests, weight + KV hooks, perplexity / lm_eval |
| System-scale experiments | `EleutherAI/pythia-1.4b` | Planned: activation / gradient communication experiments |

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| Offline pipeline | Working end-to-end `tensor → RTN → frames → HEVC → decode → tensor` with metrics **on NVENC + NVDEC** |
| Weight compression | Layer-wise bitstreams + perplexity; optional lm_eval JSON comparison |
| KV cache | Compressed cache path during `generate`, footprint + token-level comparison vs baseline |
| Communication | (Future) activation / gradient compression in DP/PP with logged bytes and step time |
| Modularity | Clear separation: `compression/`, `codec/`, `hooks/`, `evaluation/` |

---

## 1. Repository Structure (current)

```text
project/
├── codec/                  # frame mapping, FFmpeg HEVC, metadata, codec jobs
├── compression/            # RTN, incoherence, weight pipeline
├── hooks/                    # weight loader, KV hooks, VcLLMCompressedCache
├── communication/            # placeholder (__init__.py) — Phase 4+
├── training/                 # placeholder (__init__.py) — Phase 4+
├── evaluation/               # perplexity, tensor metrics, kv_cache_eval, lm_eval_integration
├── configs/                  # placeholder (__init__.py)
├── scripts/                  # phase scripts + smoke tests
├── tests/                    # placeholder (__init__.py) — add real tests over time
├── results/                  # optional: JSON outputs from experiments (gitignored or local)
├── Dockerfile
├── requirements.txt
├── README.md                 # how to run experiments
├── MODEL_AND_EVAL_GUIDE.md   # datasets / evaluation design notes
├── PLAN.md                   # this file
└── run.py                    # CLI: test_codec, compress_weights, eval_weight_codec, eval_kv_cache
```

### Module Responsibilities

| Directory | Responsibility |
|-----------|----------------|
| `codec/` | Tensor ↔ frame layouts; **FFmpeg + NVENC encode / NVDEC decode**; bitstreams + JSON metadata (paper path; no soft-decode story) |
| `compression/` | RTN; optional incoherence (Hadamard / block size); `weight_pipeline` for layer-wise compress/decompress |
| `hooks/` | `weight_loader`; `KVCacheCompressionHook`; `VcLLMCompressedCache` for transformers DynamicCache |
| `communication/` | **Planned:** compressed send/recv, compressed collectives |
| `training/` | **Planned:** PP/DP loops with profiling hooks |
| `evaluation/` | WikiText perplexity; tensor metrics; KV eval CLI; `lm_eval_integration.py` |
| `configs/` | **Planned:** YAML/TOML experiment configs |
| `scripts/` | Standalone tests (tensor codec, weight smoke, KV generate parity, optional paper-style script) |

---

## 2. System Decomposition

### Inputs

- **Static:** model weights  
- **Inference runtime:** KV cache (implemented); activations (planned Phase 4)  
- **Training runtime:** activation gradients, weight gradients (planned Phase 5)  
- **System:** model config, QP / frame size / RTN settings; **paper runs assume hardware codecs only** (CLI flags that disable hardware are **not** part of the official reproduction contract—see `README.md`)

### Core Layers (logical)

- **Model:** HuggingFace `AutoModelForCausalLM`, `generate()` with `use_cache=True` for KV experiments  
- **Compression:** RTN + optional incoherence; shared semantics for weights and KV  
- **Codec:** Intra-oriented HEVC (see `codec/hevc_backend.py`); **NVENC encode + NVDEC decode only** for systems-aligned claims  
- **Communication:** **Not yet implemented**  
- **Evaluation:** Perplexity, tensor metrics, KV footprint / token match, lm_eval JSON  

### Outputs

- Bitstreams + sidecar JSON metadata under `--compressed-dir`  
- Reconstructed tensors in memory  
- Metrics: perplexity, lm_eval accuracies, compression ratio summaries, KV dense vs packed byte estimates  

---

## 3. End-to-End Pipeline (reference)

Same logical diagram as before; **implemented paths today** are **Weight** and **KV Cache** through RTN + frame mapping + HEVC. **Activation** and **Gradient** paths are **future work**.

```text
[FP16/BF16 Tensor]
        |
        v
[Tensor Type Router]
  |       |        |        |
  |       |        |        +--> Gradient Path     (Phase 5 — not implemented)
  |       |        +------------> Activation Path (Phase 4 — not implemented)
  |       +---------------------> KV Cache Path   (implemented)
  +-----------------------------> Weight Path     (implemented)

Weight Path:
[Optional Incoherence]
        |
        v
[RTN -> INT8]
        |
        v
[Tensor -> Frame Mapping]
        |
        v
[HEVC Encode (NVENC)]
        |
        v
[Bitstream + Metadata]
        |
        v
[HEVC Decode (NVDEC / hardware)]
        |
        v
[Frame -> Tensor]
        |
        v
[Load into LLM]

KV Cache Path:
[Runtime K/V tensors]
        |
        v
[RTN + Frame Mapping + HEVC]
        |
        v
[Compressed store in DynamicCache layers]
        |
        v
[Decode before attention]
```

---

## 4. Milestones (rolling status)

| ID | Milestone | Status |
|----|-----------|--------|
| M1 | Offline tensor codec pipeline end-to-end | **Done** (`run.py --mode test_codec`) |
| M2 | Weight compression + perplexity (small model) | **Done** (`compress_weights`, `eval_weight_codec`; lm_eval via `evaluation/lm_eval_integration.py`) |
| M3 | KV cache compression during generation | **Done** (`eval_kv_cache`, `scripts/test_kv_vcllm_cache_generate.py`) |
| M4 | Pipeline-parallel activation compression | **Not started** (`communication/`, `training/` stubs) |
| M5 | Data-parallel gradient compression | **Not started** |
| M6 | Unified evaluation report + reproducibility bundle | **Partial** (metrics exist; formal report / pinned artifact list TBD) |

---

## 5. Phase 0 — Project Bootstrap

**Status: complete**

| Task | Done |
|------|------|
| Repository layout (`codec`, `compression`, `hooks`, `evaluation`, `scripts`, placeholders) | ✓ |
| `run.py`, `requirements.txt`, `README.md`, `PLAN.md`, `Dockerfile` | ✓ |
| Core deps: torch, transformers, datasets, ffmpeg available on PATH | ✓ |
| Smoke: `python run.py --mode test_codec` | ✓ |

**Exit criteria:** repo runs locally or in Docker; `pythia-160m` loads for weight/KV modes.

---

## 6. Phase 1 — Tensor → Video Codec Validation

**Status: complete**

**Code:** `compression/rtn.py`, `compression/incoherence.py` (optional), `codec/frame_mapper.py`, `codec/hevc_backend.py`, `codec/metadata.py`, `codec/codec_job.py`, `evaluation/tensor_metrics.py`, `scripts/test_tensor_codec.py`.

**Entry:** `python run.py --mode test_codec`

| Task | Done |
|------|------|
| RTN quantize / dequantize + metadata | ✓ |
| Tensor ↔ frames + chunking | ✓ |
| HEVC encode/decode (intra-oriented), FFmpeg subprocess | ✓ |
| Random / weight-shaped tensor tests + bitrate / error reporting | ✓ |
| Reconstruction metrics (MSE, MAE, relative error, cosine, compression stats as implemented) | ✓ |

**Residual risks:** tensor layout vs codec efficiency; document **GPU generation, driver, FFmpeg NVENC/NVDEC capabilities** in every experiment log. Do not compare “soft decode” runs to hardware runs as equivalent systems points.

---

## 7. Phase 2 — Weight Compression Inference

**Status: complete**

**Code:** `compression/weight_pipeline.py`, `hooks/weight_loader.py`, `evaluation/perplexity.py`, `evaluation/lm_eval_integration.py`, `run.py`.

**CLI:**

- `python run.py --mode compress_weights`
- `python run.py --mode eval_weight_codec`
- `python evaluation/lm_eval_integration.py --model ... [--compressed-dir ...] --output ...`

| Task | Done |
|------|------|
| Layer-wise compress → bitstreams + JSON | ✓ |
| Decompress into a fresh model instance | ✓ |
| WikiText-2 perplexity baseline vs decompressed | ✓ |
| lm_eval JSON for baseline vs decompressed | ✓ |

**Important:** default `--compress-mode rtn_lossy_hevc` can **severely** hurt perplexity if QP/bitrate is wrong—use `rtn_lossless_hevc` or `rtn_only` first to validate the pipeline (`README.md`).

**Residual risks:** layer sensitivity; outlier layers; disk layout vs lazy decode optimizations.

---

## 8. Phase 3 — KV Cache Compression

**Status: complete (inference hook path)**

**Code:** `hooks/kv_cache_hook.py`, `hooks/compressed_kv_cache.py`, `evaluation/kv_cache_eval.py`, `run.py --mode eval_kv_cache`, `scripts/test_kv_vcllm_cache_generate.py`.

**Note:** There is **no** separate `compression/runtime_quant.py` or `codec/stream_buffer.py` in this repo—KV reuses the same RTN + frame + HEVC stack as weights, configured via `KVCacheCompressionHook`.

| Task | Done |
|------|------|
| Inject compressed cache (`VcLLMCompressedCache`) compatible with transformers DynamicCache | ✓ |
| Greedy generation parity / match-rate reporting vs baseline | ✓ |
| Dense vs packed byte footprint estimates | ✓ |
| Longer generate smoke test script | ✓ |

**Residual risks:** decode latency vs memory win; lossy KV (`--kv-lossy`) vs quality—measure task-by-task.

---

## 9. Phase 4 — Pipeline-Parallel Activation Compression

**Status: not started**

**Target model:** `EleutherAI/pythia-1.4b` (when started).

**Planned code areas (to be created):** `communication/*`, `hooks/*` for activation transport, `training/pipeline_engine.py` or equivalent.

| Task | Status |
|------|--------|
| Compressed send/recv for forward activations | ☐ |
| Decode at consumer stage | ☐ |
| Backward activation-gradient path | ☐ |
| Profiling: encode / comm / decode | ☐ |

---

## 10. Phase 5 — Data-Parallel Gradient Compression

**Status: not started**

**Planned code areas:** `communication/compressed_allreduce.py` (or similar), `compression/residual_comp.py`, gradient hooks, `training/data_parallel_engine.py`.

| Task | Status |
|------|--------|
| Gradient quantize + encode before comm | ☐ |
| Metadata for variable-length payloads | ☐ |
| Optional residual compensation | ☐ |
| Training stability vs baseline DP | ☐ |

---

## 11. Evaluation Matrix

### Tensor-level

- MSE, MAE, relative error, cosine similarity, compression ratio / bits-per-value — **supported** in tensor codec path (`evaluation/tensor_metrics.py`, scripts).

### Inference-level

- WikiText-2 perplexity — **supported** (`evaluation/perplexity.py`)
- lm_eval accuracies — **supported** (`evaluation/lm_eval_integration.py`)
- Logits drift / long-context KV — **partially** (KV scripts focus on token match + footprint; extend as needed)

### System-level

- Memory / KV packed bytes — **partial** (KV eval estimates)
- Encode/decode latency breakdown — **manual** (optional timers in hooks/codecs)
- Communication bytes / step time — **Phase 4–5**

---

## 12. Experiment Matrix (recommended order)

### Weight compression

1. FP16 baseline perplexity  
2. `rtn_only`  
3. `rtn_lossless_hevc`  
4. `rtn_lossy_hevc` + QP sweep  
5. Optional: lm_eval before/after  

### KV cache

1. Lossless HEVC default (`eval_kv_cache` without `--kv-lossy`)  
2. `--kv-lossy` + QP sweep  
3. Long prompts via `--kv-prompt`; longer decode via `test_kv_vcllm_cache_generate.py`  

### Activation / gradient (future)

- Deferred until Phases 4–5 land.

---

## 13. Paper Alignment Checklist

| Item | Status |
|------|--------|
| HEVC intra-oriented encoding (no inter-frame dependence for tensor slices) | Implemented; must be exercised **on NVENC/NVDEC** for systems claims |
| Tensor types: weights | Done |
| Tensor types: KV | Done |
| Tensor types: activations | Not done |
| Tensor types: gradients | Not done |
| Language quality: perplexity + lm_eval | Done |
| System benefits: compression ratio / KV footprint | Partial |
| Distributed: PP activation / DP gradient compression | Not done |

---

## 14. Critical Failure Checklist (engineering)

- [ ] Frame mapping preserves enough structure for the codec to be worthwhile  
- [ ] Inter coding disabled where the paper/design requires **intra-only** tensor slices  
- [ ] Metadata sufficient to invert shape/dtype/scale  
- [ ] Lossy runs always record **QP**, **compress-mode**, and **GPU model + driver + FFmpeg build** (hardware path only for official logs)  
- [ ] Hooks do not change attention semantics (shape/debug assertions on failure)  
- [ ] Measure latency where claiming wall-clock wins  

---

## 15. Execution Schedule (suggested remaining work)

This replaces the old fixed “Week 1–5” calendar with **priority-ordered** tasks aligned to **current** repo state.

| Priority | Focus |
|----------|--------|
| P0 | Keep `README.md` and CLI flags accurate when changing defaults |
| P1 | Tests: RTN / frame mapping on CPU where pure math; **integration tests require NVENC+NVDEC** for codec paths |
| P2 | Phase 4 scaffold: minimal PP dummy two-stage pipeline + compressed activation tensor path |
| P3 | Phase 5 scaffold: compressed gradient bucket + correctness checks |
| P4 | Repro bundle: pinned `requirements.txt`, FFmpeg version note, example `results/*.json` naming convention |

---

## 16. Cursor Working Rules

- Implement **one phase** at a time; keep **offline tensor tests green** before merging risky codec changes.  
- Prefer extending **`run.py`** for user-facing experiments so `README.md` stays stable.  
- Log **quality + system** metrics for each experiment (QP, mode, **NVENC/NVDEC confirmation**—no soft-decode baselines for VcLLM claims).  
- Save baselines **before** aggressive bitrate / lossy sweeps.  

---

## 17. Immediate Next Actions (maintainers)

1. **Testing:** add unit tests for `compression/rtn.py` and `codec/frame_mapper.py` (CPU OK); add **GPU-backed** codec smoke tests that assert NVENC/NVDEC are used on paper-aligned runs.  
2. **Docs:** when changing `run.py` defaults, update `README.md` and this file’s milestone table.  
3. **Phase 4 spike:** design minimal API for `communication/` (tensor → bitstream → tensor + rank metadata).  
4. **Repro:** document FFmpeg compile flags / NVENC availability in experiment logs for paper-aligned runs.  

---

## Changelog

| Date | Change |
|------|--------|
| 2026-05 | Rewrote milestones to match implemented codebase; fixed file references (`lm_eval_integration.py`); marked Phases 0–3 complete and 4–5 not started; linked `README.md`; replaced stale week-by-week schedule with prioritized backlog. |
| 2026-05 | **Policy:** hardware-only NVENC/NVDEC for paper-aligned reproduction; **rejected CPU soft-decode (and CPU codec) fallback** as part of the VcLLM systems story; updated `README.md` + this file accordingly. |
