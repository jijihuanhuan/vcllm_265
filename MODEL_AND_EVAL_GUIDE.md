# VcLLM Reproduction: Model, Dataset, and Comparison Guide

## 0. Purpose

This document summarizes practical recommendations for:

- model selection
- dataset selection
- compressed vs uncompressed weight comparison
- compressed vs uncompressed model result comparison
- prompt design for inference-side evaluation

The goal is to support an **engineering-first reproduction** of VcLLM starting from small models and progressively scaling toward distributed system experiments.

---

## 1. Model Selection

### 1.1 Recommended Model Roadmap

Use a two-stage model strategy:

- **Phase 1 to Phase 3**
  - `EleutherAI/pythia-160m`

- **Phase 4 to Phase 5**
  - `EleutherAI/pythia-1.4b`

### 1.2 Why `pythia-160m` for the MVP

`pythia-160m` is a strong first model because:

- it is small enough for fast iteration
- weight export, compression, and reconstruction are quick
- perplexity evaluation is cheap
- KV cache hook debugging is manageable
- failures are easier to localize between codec logic and model behavior

### 1.3 Why `pythia-1.4b` for System Experiments

`pythia-1.4b` is a better fit for activation and gradient communication experiments because:

- it is large enough for communication overhead to matter
- it remains much easier to debug than 7B or larger models
- it better reflects the system-level motivation of the VcLLM paper

### 1.4 Models Not Recommended for First Pass

- `LLaMA-3-70B`
  - too expensive and too complex for first-pass system validation

- `OPT-125M`
  - usable, but less convenient than Pythia for modern LLM engineering workflows

- `TinyLlama`
  - fine for smoke tests, but weaker as a paper reproduction backbone

### 1.5 Final Recommendation

Use the following default setup:

- `pythia-160m` for offline tensor validation, weight compression, and KV cache compression
- `pythia-1.4b` for pipeline-parallel activation compression and data-parallel gradient compression

---

## 2. Dataset Selection

Different phases need different datasets. Do not use a single dataset for all purposes.

### 2.1 Weight Compression Quality Evaluation

Use these datasets to evaluate whether compressed weights preserve language modeling quality:

- `WikiText-2`
- optional: `Penn Treebank`

Primary metric:

- perplexity

Recommended use:

- compare FP16 baseline vs RTN-only vs RTN+HEVC
- run bitrate sweeps
- analyze layer sensitivity

### 2.2 Downstream Task Evaluation

Use these datasets to evaluate whether compressed models preserve reasoning and instruction-following behavior:

- `PIQA`
- `WinoGrande`
- `HellaSwag`
- `ARC-Easy`

Primary metrics:

- zero-shot accuracy
- task-level accuracy deltas before and after compression

Recommended use:

- validate that improvements are not limited to perplexity
- expose failure cases where generated answers remain fluent but become less correct

### 2.3 Training and Communication Compression Evaluation

Use lightweight training datasets to evaluate activation and gradient compression:

- `The Pile` subset
- or a fixed reproducible text subset sampled from `C4`

Primary metrics:

- training loss
- validation perplexity
- communication volume
- step time

Recommended use:

- activation compression in pipeline parallel
- gradient compression in data parallel

### 2.4 Long-Context Evaluation for KV Cache Compression

Use:

- `WikiText-2` long context slices
- a custom `long_context_prompts.jsonl`

The custom prompt file should include:

- long multi-turn QA
- long-document retrieval
- long technical notes
- code-oriented long-context prompts

Primary metrics:

- long-context perplexity
- retrieval correctness
- output drift under long prompts

---

## 3. Phase-by-Phase Model and Dataset Mapping

### Phase 1: Tensor -> Codec Validation

Model:

- no model required for the first smoke test
- optionally use tensors sampled from `pythia-160m`

Data:

- random tensors
- real model weight tensors

Goal:

- validate the codec pipeline independently from model semantics

### Phase 2: Weight Compression Inference

Model:

- `pythia-160m`

Data:

- `WikiText-2`
- `PIQA`
- `WinoGrande`
- `HellaSwag`

Goal:

- validate compressed weight loading and quality retention

### Phase 3: KV Cache Compression

Model:

- `pythia-160m`

Data:

- `WikiText-2` long slices
- `long_context_prompts.jsonl`

Goal:

- validate memory reduction and long-context quality retention

### Phase 4: Pipeline Parallel Activation Compression

Model:

- `pythia-1.4b`

Data:

- `The Pile` subset

Goal:

- validate activation communication compression under PP

### Phase 5: Data Parallel Gradient Compression

Model:

- start with `pythia-160m`
- optionally extend to `pythia-1.4b`

Data:

- `The Pile` subset
- or a fixed `C4` subset

Goal:

- validate compressed gradient communication and training stability

---

## 4. Weight Comparison: Before vs After Compression

Weight comparison should be done at two levels:

- static tensor comparison
- functional model comparison

### 4.1 Static Weight Comparison

This answers:

> How far are the reconstructed weights from the original weights?

Recommended metrics:

- `MSE`
- `MAE`
- `Relative Error`
- `Cosine Similarity`
- `Max Absolute Error`

Recommended granularity:

- full-model aggregate
- per-layer aggregate
- per-module-type aggregate

Suggested module types:

- embedding
- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- MLP up projection
- MLP down projection
- `lm_head`

### 4.2 Suggested Weight Comparison Output Table

```text
| layer_name | module_type | mse | mae | rel_error | cosine_sim | max_abs_err | bitrate |
|------------|-------------|-----|-----|-----------|------------|-------------|---------|
```

### 4.3 Interpretation Guidance

- low MSE alone is not enough
- some layers are much more sensitive than others
- reconstructed weights can look numerically close but still degrade perplexity noticeably
- layer-wise error concentration is often more informative than global averages

---

## 5. Model Result Comparison: Before vs After Compression

This answers:

> Does the model still behave correctly after compression?

### 5.1 Recommended Metrics

- perplexity
- zero-shot task accuracy
- logits drift
- token-level output differences
- generation stability
- long-context retrieval accuracy

### 5.2 Comparison Dimensions

Always compare across these model states:

- FP16 baseline
- RTN-only baseline
- RTN + HEVC
- RTN + HEVC + KV compression
- later: activation-compressed / gradient-compressed training variants

### 5.3 Suggested Result Comparison Table

```text
| run_name | model | compression_mode | bitrate | ppl | piqa | hellaswag | winogrande | notes |
|----------|-------|------------------|---------|-----|------|-----------|------------|-------|
```

### 5.4 Suggested Per-Prompt Comparison Table

```text
| prompt_id | task_type | model_state | bitrate | original_output | compressed_output | logits_cosine | token_diff_ratio | manual_notes |
|-----------|-----------|-------------|---------|-----------------|-------------------|---------------|------------------|--------------|
```

---

## 6. Prompt Design for Compressed vs Uncompressed Comparison

Do not use only one prompt. Build a fixed prompt set and reuse it across all experiments.

Recommended categories:

- basic continuation
- commonsense QA
- long-context retrieval
- instruction-following / structured output

### 6.1 Basic Continuation Prompts

Use to evaluate fluency and local output stability.

Prompt template:

```text
Continue the following paragraph in a coherent and factual style:

Artificial intelligence systems are increasingly used in research, engineering, and communication. However, the efficiency of deploying large models depends not only on parameter count, but also on memory, bandwidth, and system design.

Continuation:
```

Alternative template:

```text
Continue the news article in a coherent style:

In a surprising development, researchers found that system-level bottlenecks, rather than model size alone, often determine whether large models can be deployed efficiently in practice.

Continuation:
```

What to observe:

- fluency
- factual coherence
- degeneration
- sudden stylistic drift after compression

### 6.2 Commonsense QA Prompts

Use to evaluate whether compression affects basic reasoning and answer correctness.

Prompt template:

```text
Answer the question briefly and accurately.

Question: Why can compressing KV cache reduce memory usage during long-context inference?
Answer:
```

Alternative template:

```text
Question: If a person walks through wet mud while wearing shoes, what is most likely to become dirty?
Answer:
```

What to observe:

- correctness
- brevity vs rambling
- whether compressed outputs become less grounded

### 6.3 Long-Context Retrieval Prompts

Use specifically for KV cache compression evaluation.

Prompt template:

```text
Read the following passage carefully and answer the question at the end.

[Insert long passage here]

Question: What was the main reason the team switched from raw tensor transmission to codec-based transmission?
Answer:
```

Alternative template:

```text
Below is a long technical note.

[Insert long note here]

Question: According to the note, which module is responsible for metadata synchronization?
Answer:
```

What to observe:

- retrieval accuracy from distant context
- drift under longer sequence lengths
- hallucination after KV compression

### 6.4 Instruction-Following and Structured Output Prompts

Use to evaluate whether compression hurts output formatting and instruction compliance.

Prompt template:

```text
Summarize the following paragraph in exactly 3 bullet points.

[Insert text]

Summary:
```

Alternative template:

```text
Read the text and return a JSON object with the following fields:
- method
- tensor_type
- compression_stage
- expected_benefit

Text:
[Insert technical passage]
```

What to observe:

- formatting correctness
- instruction compliance
- JSON validity
- structural degradation under compression

---

## 7. Recommended Fixed Prompt Set

Create a reusable file:

- `evaluation/prompts/compare_prompts.jsonl`

Recommended composition:

- 10 basic continuation prompts
- 10 commonsense QA prompts
- 10 long-context retrieval prompts
- 10 instruction-following prompts

Benefits:

- consistent comparison across experiments
- easier regression tracking
- easier manual inspection

---

## 8. What to Log for Every Prompt Comparison

For every prompt, store:

- `prompt_id`
- `task_type`
- `model_name`
- `compression_mode`
- `bitrate`
- `original_output`
- `compressed_output`
- `logits_cosine`
- `token_diff_ratio`
- `manual_notes`

This is important because once multiple compression modes and bitrates are involved, manual comparison becomes very hard without structured records.

---

## 9. Recommended Minimal Experiment Configuration

### Model

- `pythia-160m`

### Datasets

- `WikiText-2`
- `PIQA`
- `HellaSwag`
- custom `long_context_prompts.jsonl`

### Comparison Groups

- FP16 baseline
- INT8 RTN-only baseline
- RTN + HEVC
- RTN + HEVC + KV compression

### Output Metrics

- weight MSE / cosine similarity
- perplexity
- zero-shot accuracy
- prompt-by-prompt generation difference
- KV memory saving

---

## 10. Practical Experimental Advice

### 10.1 For Weight Compression

Start with:

- whole-model RTN baseline
- then RTN + HEVC
- then layer-wise sensitivity analysis

Do not start with very aggressive bitrate targets before baseline behavior is stable.

### 10.2 For KV Cache Compression

Always compare:

- short context
- medium context
- long context

A compression scheme that looks fine at 512 tokens may fail at 4k or 8k tokens.

### 10.3 For Prompt Evaluation

Use a fixed prompt set.

Do not rely only on a few hand-picked examples, because compressed models often fail unevenly:

- some prompts remain unchanged
- some prompts drift slightly
- some prompts collapse only under long context or structured output constraints

### 10.4 For Reporting

When writing reproduction results, separate:

- tensor-level fidelity
- model-level quality
- system-level efficiency

Do not treat them as interchangeable.

---

## 11. Recommended Next Files to Create

To make this guide actionable, the next useful files are:

- `evaluation/prompts/compare_prompts.jsonl`
- `evaluation/prompts/long_context_prompts.jsonl`
- `evaluation/result_schema.json`

Suggested purpose:

- `compare_prompts.jsonl`
  - reusable compressed vs uncompressed prompt evaluation set

- `long_context_prompts.jsonl`
  - dedicated KV cache stress test set

- `result_schema.json`
  - unified storage format for metrics and outputs

