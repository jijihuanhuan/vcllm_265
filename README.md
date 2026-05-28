# VcLLM：视频编码式 LLM 张量压缩 — 复现实验手册

面向第一次使用该仓库的研究者：按章节复制命令即可跑通流水线；遇到问题先看「常见错误与排查」。

---

## 1. 项目目标（Project Overview）

本仓库实现 **VcLLM / LLM.265** 思路下的工程原型：**把量化后的张量布局成类图像帧，再走真实 HEVC 视频编解码**，用于：

- **离线验证**：随机张量或权重张量 → RTN 量化 → 帧映射 → HEVC → 解码 → 重建误差与码率
- **权重压缩**：按层压缩 HuggingFace CausalLM 权重到磁盘，再解压回内存并评测困惑度等
- **KV 压缩**：在解码生成过程中对 KV Cache 做 RTN+HEVC，对比显存/比特流占用与输出一致性

**硬件编解码是 VcLLM 论文侧的核心系统贡献**：本仓库的**研究复现路线**只承认 **NVIDIA NVENC（`hevc_nvenc`）编码** 与 **NVDEC（如 FFmpeg `hevc_cuvid` / CUDA 硬解路径）解码** 构成的闭环。**不考虑 CPU 软解回退**；也不把 libx265 / 纯软件解码链路作为可接受的论文对齐实验配置。若硬解或硬编不可用，应视为**环境未就绪**，需修复驱动、容器 GPU 透传与 FFmpeg 构建，而不是改用软解跑通。

`rtn_only` 等**无量化后视频码流**的配置可作为**量化基线**，与「固定功能视频硬件承担编解码」的系统主张分开表述，不得与 NVENC+NVDEC 结果混为一谈。

---

## 2. 系统整体流程（文字流程图）

**离线张量编解码（与模型语义无关）**

1. 张量 →（可选）不相干变换 → **RTN 量化**  
2. 量化整数 → **帧映射**（铺成 PNG 序列）  
3. PNG → **FFmpeg + NVENC（`hevc_nvenc`）** → HEVC 比特流  
4. 比特流 → **FFmpeg + NVDEC（硬解）** → PNG → **逆映射** →反量化张量  
5. 与原始张量比 **MSE / 相对误差 / 码率**

**权重压缩与评测**

1. `from_pretrained` 加载模型 → 逐层 **压缩到 `--compressed-dir`**（JSON 元数据 + 比特流）  
2. 新骨架模型 → **解压权重覆盖参数**  
3. 在 WikiText-2 上算 **困惑度**；可选 **lm_eval** 零样本任务

**KV 压缩与评测**

1. 正常 `generate(..., use_cache=True)` 时，每层逐步累积 **Key/Value**  
2. 注入 **VcLLMCompressedCache**：在 cache 更新路径中对 K/V 做与权重侧一致的压缩/解压  
3. 对比 ** baseline DynamicCache** 与 **压缩 Cache** 的 token 序列、估算 **稠密 KV 字节数 vs 压缩比特流字节数**

---

## 3. 仓库结构说明（源码与脚本）

下列为**仓库内应阅读的源码与脚本**（不含运行时生成的 `compressed_weights/`、`output/`、缓存目录等）。

| 路径 | 作用 |
|------|------|
| `run.py` | **唯一推荐的主入口**：`--mode` 切换离线编解码、压权重、困惑度、KV 评测 |
| `requirements.txt` | Python 依赖版本下界 |
| `Dockerfile` | CUDA base 镜像 + 系统级 `ffmpeg` 等；容器内需再 `pip install -r requirements.txt`（若与镜像预装不一致） |
| `PLAN.md` | 分阶段复现计划（路线图，非操作步骤） |
| `MODEL_AND_EVAL_GUIDE.md` | 模型/数据集/评测设计建议 |
| **codec/** | |
| `codec/frame_mapper.py` | 张量 ↔ 帧（PNG）布局 |
| `codec/hevc_backend.py` | 调用 FFmpeg：**NVENC 编码、NVDEC 硬解**（论文对齐路径；不以 CPU 软解为复现前提） |
| `codec/metadata.py` | 层名、shape、scale 等序列化 |
| `codec/codec_job.py` | 单次编解码任务参数封装 |
| **compression/** | |
| `compression/rtn.py` | RTN 量化 / 反量化 |
| `compression/incoherence.py` | 可选 Hadamard 等「不相干」预处理（与 `--no-incoherence` 对应） |
| `compression/transform.py` | 张量变换辅助 |
| `compression/topology_router.py` | 层路由等辅助逻辑 |
| `compression/weight_pipeline.py` | **权重压缩/解压主流程** |
| **hooks/** | |
| `hooks/kv_cache_hook.py` | KV 压缩钩子配置（RTN、QP、帧大小等） |
| `hooks/compressed_kv_cache.py` | `VcLLMCompressedCache`：与 `transformers` DynamicCache 对接 |
| `hooks/weight_loader.py` | 从压缩目录解压并载入权重（供扩展使用） |
| **evaluation/** | |
| `evaluation/perplexity.py` | WikiText-2 困惑度 |
| `evaluation/tensor_metrics.py` | 张量重建指标 |
| `evaluation/kv_cache_eval.py` | KV 评测 CLI：`run.py --mode eval_kv_cache` 调用 |
| `evaluation/lm_eval_integration.py` | **lm_eval** 零样本任务（JSON 结果） |
| **scripts/** | |
| `scripts/test_tensor_codec.py` | 离线编解码自测（由 `test_codec` 调用） |
| `scripts/test_weight_compression*.py` | 权重压缩冒烟脚本（可直接 `python scripts/...`） |
| `scripts/test_compression_quality.py` | 压缩质量相关测试脚本 |
| `scripts/test_kv_vcllm_cache_generate.py` | **长序列 greedy generate** 与 baseline 对比 token（默认 256 tokens） |
| `scripts/test_vcllm_paper.py` | 论文风格单项实验脚本（路径写死 `/tmp` 等，适合参考而非首选入口） |
| `scripts/debug_frame_mapping.py` | 帧映射调试 |
| **debug_*.py**（仓库根目录） | 单层/单步调试：权重、RTN、整管线（日常实验可忽略） |
| `code_test.py` | 零散测试入口（非正式 harness） |
| **configs/**、`communication/`、`training/`、`tests/`** | 目前多为占位（仅 `__init__.py`），预留给分布式与训练扩展 |

---

## 4. 核心模块职责说明

| 模块 | 职责 |
|------|------|
| **compression** | 量化与（可选）预处理；权重/KV 共用 RTN 与参数语义 |
| **codec** | 与 FFmpeg 打交道：帧序列 ↔ HEVC 比特流；记录元数据 |
| **hooks** | 把编解码挂到 **权重加载** 或 **KV Cache 更新** |
| **evaluation** | 困惑度、张量指标、KV 对比、可选 lm_eval |
| **scripts** | 不经过 `run.py` 的独立实验与调试脚本 |

---

## 5. 环境安装步骤（一步一步）

以下假定 **Linux + NVIDIA GPU + 可用 NVENC/NVDEC**（论文对齐实验的**硬前提**；无硬编硬解环境请先解决环境，勿以软解替代）。

1. **进入仓库根目录**（包含 `run.py` 的目录）。

2. **创建并激活虚拟环境**（推荐）：
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **安装 PyTorch（CUDA 版）**（按 [PyTorch 官网](https://pytorch.org) 选择与驱动匹配的 CUDA 构建；**CPU-only 环境不满足本仓库论文复现前提**。）

4. **安装 Python 依赖**：
   ```bash
   pip install -U pip
   pip install -r requirements.txt
   ```

5. **安装系统 FFmpeg**（Ubuntu 示例）：
   ```bash
   sudo apt-get update && sudo apt-get install -y ffmpeg
   ```
   编解码由 **子进程调用 `ffmpeg`**，未安装会直接报错。

6. **可选：Docker**（镜像内已含 `ffmpeg`；GPU 需 NVIDIA Container Toolkit）：
   ```bash
   docker build -t vcllm:latest .
   docker run -it --gpus all --ipc=host --shm-size=32g -v "$(pwd)":/workspace vcllm:latest
   ```
   进入容器后在 `/workspace` 执行 `pip install -r requirements.txt`（若需与镜像内 PyTorch 对齐请自行固定版本）。

---

## 6. GPU / FFmpeg 依赖检查方法

**GPU**

```bash
nvidia-smi
```
**用途**：确认驱动与显卡可见；无输出则 NVENC/NVDEC 不可用。

**PyTorch CUDA**

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.version.cuda)"
```
**用途**：确认 PyTorch 能否使用 GPU；论文对齐实验要求为 `True`。

**FFmpeg 是否存在**

```bash
ffmpeg -version
```
**用途**：确认命令行可调 FFmpeg。

**HEVC 硬件编码器（必查）**

```bash
ffmpeg -hide_banner -encoders 2>/dev/null | grep -i hevc
```
**用途**：必须能看到 **`hevc_nvenc`**。看不到则说明当前 FFmpeg/驱动/环境无法走论文对齐编码路径，需换带 NVENC 的 FFmpeg 构建或修正安装，**不要**依赖 CPU 编码冒充系统贡献。

**HEVC 硬件解码器（必查）**

```bash
ffmpeg -hide_banner -decoders 2>/dev/null | grep -i hevc
```
**用途**：必须能走 **NVDEC / CUVID 类硬解**（如 `hevc_cuvid` 等，具体名称随 FFmpeg 版本略有差异）。**禁止**把「软解能跑」当作 VcLLM 复现成功标准。

---

## 7. 如何运行实验（完整命令）

**约定**：下列命令均在仓库根目录执行，`python` 即当前虚拟环境中的解释器。

### 7.1 离线张量编解码自测

```bash
python run.py --mode test_codec
```

### 7.2 仅压缩权重到磁盘

```bash
python run.py --mode compress_weights \
  --model EleutherAI/pythia-160m \
  --compressed-dir ./compressed_weights \
  --compress-mode rtn_lossless_hevc \
  --qp 0 \
  --frame-size 1024
```

### 7.3 权重压缩 + WikiText 困惑度前后对比

```bash
python run.py --mode eval_weight_codec \
  --model EleutherAI/pythia-160m \
  --compressed-dir ./compressed_weights \
  --compress-mode rtn_lossless_hevc \
  --qp 0 \
  --frame-size 1024
```

### 7.4 KV Cache 压缩评测（短序列冒烟：Footprint + greedy 对比）

**用途**：快速检查 VcLLM KV 路径相对 baseline 的 **token 一致性** 与 **稠密 KV vs 压缩比特** 的 footprint；默认只多生成 32 个 token。

```bash
python run.py --mode eval_kv_cache \
  --model EleutherAI/pythia-160m \
  --max-new-tokens 32 \
  --frame-size 1024
```

有损 KV（`--qp` 生效）示例：

```bash
python run.py --mode eval_kv_cache \
  --model EleutherAI/pythia-160m \
  --kv-lossy \
  --qp 28 \
  --max-new-tokens 32
```

### 7.5 长上下文：显存与压缩 Profiling（512 token 续写）

**用途**：把续写长度拉到 **512**，在 **Chunking / 分块固化的 KV 压缩** 下观察：随上下文变长，**有效压缩强度**（团队常 track 的每元素比特 / 与稠密 KV 的体积比；若你们内部记为 BPE，可对照是否稳定在约 **2.9** 附近）以及 **相对稠密 FP16 KV 的显存/驻留优势**。日志中关注 `eval_kv_cache` 打印的 **dense vs packed 字节** 与生成是否仍与 baseline 一致（无损 HEVC 下通常一致）。

```bash
python run.py --mode eval_kv_cache \
  --model EleutherAI/pythia-160m \
  --max-new-tokens 512 \
  --frame-size 1024
```

**说明**：长序列对 **RTN+HEVC KV** 压力更大；若 OOM，可先减小模型或调 `--frame-size`（与分块实验设计一致即可）。

### 7.6 PIQA 全量零样本：智商保真度（三组拉踩对照）

**用途**：在 **整份 PIQA 测试集**（约 **1800+** 条，**不要**加 `--limit`）上跑零样本；有了分块/Chunking 后，**完整跑 VcLLM HEVC KV** 的耗时可控。请固定 **同一模型权重**（此处为原始 FP16 基线权重），仅切换 **`--kv-cache-mode`**，输出三组 JSON 便于并列对比。

**① 标准 FP16 KV（无压缩）**

```bash
python evaluation/lm_eval_integration.py \
  --model EleutherAI/pythia-160m \
  --tasks piqa \
  --kv-cache-mode none \
  --output results/piqa_kv_none_full.json
```

**② 仅 RTN 量化 KV（默认 3-bit，`rtn3`）——视频 codec 对照组**

```bash
python evaluation/lm_eval_integration.py \
  --model EleutherAI/pythia-160m \
  --tasks piqa \
  --kv-cache-mode rtn3 \
  --output results/piqa_kv_rtn3_full.json
```

**③ VcLLM：RTN + HEVC KV（论文对齐硬件路径）**

```bash
python evaluation/lm_eval_integration.py \
  --model EleutherAI/pythia-160m \
  --tasks piqa \
  --kv-cache-mode vcllm_hevc \
  --kv-frame-size 1024 \
  --output results/piqa_kv_vcllm_hevc_full.json
```

**说明**：`vcllm_hevc` 全量远慢于 `none` / `rtn3`；可先加 `--limit 0.05` 做冒烟，**正式表数务必去掉 `--limit`**。需要压权重再评测时，加 `--compressed-dir ./compressed_weights`（与 `--kv-cache-mode` 正交，按实验设计选择）。

### 7.7 WikiText-2 困惑度：全量测试集 + KV 自回归（三组拉踩对照）

**用途**：在 **WiKiText-2 test** 上、**按 token 自回归 + 真实 KV 路径** 累加 NLL，**不**设 `--limit-tokens` 即跑满可参与贡献的整段语料，用于检查 **分块固化（避免对同一段 KV 反复重压缩）** 后是否消除长程 **量化漂移**。结果 stdout 之外可写入 JSON。

**① 稠密 FP16 KV**

```bash
python evaluation/eval_wikitext_ppl.py \
  --model EleutherAI/pythia-160m \
  --kv-cache-mode none \
  --output results/wikitext_kv_none_full.json
```

**② RTN KV（`rtn3`）**

```bash
python evaluation/eval_wikitext_ppl.py \
  --model EleutherAI/pythia-160m \
  --kv-cache-mode rtn3 \
  --output results/wikitext_kv_rtn3_full.json
```

**③ VcLLM HEVC KV**

```bash
python evaluation/eval_wikitext_ppl.py \
  --model EleutherAI/pythia-160m \
  --kv-cache-mode vcllm_hevc \
  --frame-size 1024 \
  --output results/wikitext_kv_vcllm_hevc_full.json
```

**说明**：默认 `--max-length` / `--stride` 与脚本一致（长文档见 `evaluation/eval_wikitext_ppl.py` 顶部）；冒烟可加 `--limit-tokens 4096`，**全量表数勿加 `limit`**。

### 7.8 长序列生成一致性（VcLLMCompressedCache vs 基线）

**用途**：更长 greedy **decode**，严格对比压缩 KV 与 baseline 的 **token id** 是否一致（默认无损链路）。

```bash
python scripts/test_kv_vcllm_cache_generate.py \
  --model EleutherAI/pythia-160m \
  --max-new-tokens 256
```

### 7.9 lm_eval 多任务（可选：基线 vs 解压权重）

**用途**：多任务零样本；第二条需在 **`compress_weights`** 之后，评测 **解压权重** 质量。

```bash
python evaluation/lm_eval_integration.py \
  --model EleutherAI/pythia-160m \
  --output results/baseline.json
```

```bash
python evaluation/lm_eval_integration.py \
  --model EleutherAI/pythia-160m \
  --compressed-dir ./compressed_weights \
  --output results/vcllm_decompressed.json
```

---

## 8. 每个运行命令的用途说明

| 命令 | 用途 |
|------|------|
| `run.py --mode test_codec` | **不加载完整模型**，跑张量→量化→HEVC→重建，验证 codec 链路是否正确 |
| `run.py --mode compress_weights` | 加载模型，**逐层写出压缩产物**到 `--compressed-dir`，用于归档或后续解压评测 |
| `run.py --mode eval_weight_codec` | **端到端质量**：原始权重困惑度 → 压缩 → 解压载入 → 再算困惑度 |
| `run.py --mode eval_kv_cache` | **推理路径 KV**：baseline vs 压缩 cache；**Footprint + greedy**；增大 `--max-new-tokens` 做长上下文 profiling |
| `evaluation/lm_eval_integration.py` | lm_eval 零样本任务；**`--tasks`** 指定任务；**`--kv-cache-mode`**=`none` / `rtn3` / `vcllm_hevc` 在 **loglikelihood 路径注入 KV**；**不加 `--limit`** 即全量样本 |
| `evaluation/eval_wikitext_ppl.py` | WikiText-2 test、**token-by-token KV 自回归**困惑度；**`--kv-cache-mode`** 三组对照；**`--output`** 写 JSON 指标 |
| `scripts/test_kv_vcllm_cache_generate.py` | 更长 greedy **decode**，严格对比 token id 是否与未压缩 KV 一致（默认无损链路） |

---

## 9. 实验参数解释

| 参数 | 含义 |
|------|------|
| `--compress-mode` | `rtn_only`：仅 RTN，无 HEVC；`rtn_lossless_hevc`：RTN + **无损** HEVC（NVENC 无损模式，取决于 FFmpeg/GPU）；`rtn_lossy_hevc`：RTN + **有损** HEVC，`--qp` 生效 |
| `--qp` | **有损** HEVC 的量化参数；数值越大通常码率越低、失真越大。**无损模式与 `rtn_only` 下可不关心** |
| `--frame-size` | 张量切块映射到「帧」时的块大小，影响内存峰值与编码粒度 |
| `--no-hardware-accel` | **非论文复现路径**（若仍存在）：强制 CPU 编码。**正式实验禁止使用**；与「硬件编解码为核心贡献」立场不一致。 |
| `--no-hardware-decode` | **非论文复现路径**（若仍存在）：关闭 NVDEC。**正式实验禁止使用**；本仓库**不考虑** CPU 软解回退。 |
| `--no-incoherence` | 关闭默认的 **不相干（Hadamard 等）预处理**；用于消融 |
| `--incoherence-block-size` | 不相干变换块大小，须为 **2 的幂**；过大可能 **OOM** |
| `--device-map` | `auto`：多卡 / 大模型时分片；`none`：单设备加载 |
| `--torch-dtype` | 模型在 GPU 上加载时的权重 dtype（`float16` / `bfloat16` / `float32`） |
| `--kv-lossy` | **仅 eval_kv_cache**：默认无损 HEVC；加上后变为 RTN+有损 HEVC，**此时 `--qp` 影响 KV** |
| `--max-new-tokens` | **仅 eval_kv_cache**：greedy 多生成的 token 数 |
| `--kv-prompt` | **仅 eval_kv_cache**：自定义 prompt；不设则用内置短文本 |
| `--tasks` | **仅 lm_eval_integration**：逗号分隔任务名（如 `piqa`） |
| `--kv-cache-mode` | **lm_eval_integration / eval_wikitext_ppl**：`none`（稠密 KV）、`rtn3`（RTN 量化 KV）、`vcllm_hevc`（RTN+HEVC KV） |
| `--limit` | **lm_eval_integration**：每任务样本上限；**正式全表勿传**（默认跑满） |
| `--kv-frame-size` | **lm_eval_integration** 在 `vcllm_hevc` 下与 `tensor_to_frames` 块长一致（默认 1024） |
| `--limit-tokens` | **eval_wikitext_ppl**：达到 N 个「计入困惑度的 next-token 步」后提前停止；**全量勿传** |

---

## 10. 推理流程中 KV Cache 的作用说明

自回归生成每一步只喂入**最后一个新 token**，但要让注意力看到**此前所有 token**，因此需要缓存每一层、每一步的 **Key 和 Value**（即 KV Cache）。

- **无压缩**：KV 以稠密张量形式增长，显存随序列长度近似线性增加。  
- **本仓库压缩路径**：在 cache 更新时把 K/V 经 RTN+帧映射+HEVC 压成比特流，需要参与注意力时再解压回稠密张量（具体策略见 `hooks/compressed_kv_cache.py`）。  
- **评测意义**：在 `eval_kv_cache` 与 `test_kv_vcllm_cache_generate.py` 中对比 **显存/字节占用** 与 **生成 token 是否与 baseline 一致**（有损设置下可能不一致，需看指标）。

---

## 11. 常见错误与排查方法

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `ffmpeg` 未找到或报错 | 系统未安装 FFmpeg 或不在 `PATH` | 安装 FFmpeg；容器内 `apt-get install ffmpeg` |
| NVENC 失败或报错 | 驱动、Docker 未挂载 GPU、会话数占满、FFmpeg 未编译 NVENC | 检查 `nvidia-smi`、`docker run --gpus all`、换用带 NVENC 的 FFmpeg；**不得**改用 CPU 编码作为论文对齐结果 |
| NVDEC / 硬解失败 | 驱动、FFmpeg CUVID、无头环境权限 | 修复硬解链路；**不得**改用 CPU 软解作为复现结论依据 |
| 困惑度从 ~26 暴涨到数百 | **有损默认**：`eval_weight_codec` 默认 `--compress-mode rtn_lossy_hevc`，QP=0 仍可能不适配 | 先用 `rtn_lossless_hevc` 或 `rtn_only` 验证链路；再扫 QP |
| CUDA OOM | 模型过大或 `incoherence_block_size` 过大 | 减小 `--incoherence-block-size`；`--device-map auto`；换更小模型 |
| `datasets` 下载 WikiText 失败 | 网络或 HuggingFace 访问限制 | 配置 `HF_ENDPOINT` 或离线缓存 |
| KV 评测与 baseline token 不一致 | 使用了 **`--kv-lossy`** 或数值路径不一致 | 先用默认无损链路；对齐 `--qp`；确保全程 **NVENC/NVDEC**，勿混用软解 |

---

## 12. 推荐实验流程（新用户应该先做什么）

1. **`ffmpeg -version` + `nvidia-smi`**：环境与 GPU 正常。  
2. **`python run.py --mode test_codec`**：确认离线编解码无报错。  
3. **`compress_weights`**：使用 `rtn_lossless_hevc` 生成 `./compressed_weights`。  
4. **`eval_weight_codec`**：同样使用 **`rtn_lossless_hevc`（或 `rtn_only`）** 确认困惑度接近基线，证明权重链路正确。  
5. 再改为 **`rtn_lossy_hevc`**，扫 `--qp`，记录困惑度与磁盘体积（见 `compressed_weights/compression_summary.json`）。  
6. **`eval_kv_cache`**（短序列冒烟 → **`--max-new-tokens 512`** 长上下文 profiling）。  
7. **PIQA 全量三组**：§7.6 `none` / `rtn3` / `vcllm_hevc`。  
8. **WikiText 全量三组**：§7.7 `eval_wikitext_ppl.py`。  
9. 可选：**§7.9** 多任务 lm_eval；**`test_kv_vcllm_cache_generate.py`** 做长序列 token 一致性。

---

## 13. 最小可运行示例（Quick Start）

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt
# 确保已安装 ffmpeg、nvidia-smi 正常，且 ffmpeg 可见 hevc_nvenc / 硬解能力

python run.py --mode test_codec

python run.py --mode eval_weight_codec \
  --model EleutherAI/pythia-160m \
  --compress-mode rtn_lossless_hevc \
  --compressed-dir ./compressed_weights

python run.py --mode eval_kv_cache --model EleutherAI/pythia-160m --max-new-tokens 32
```


## 参考文献与许可

- 论文方向：*LLM.265: Video Codecs are Secretly Tensor Codecs*（VcLLM / LLM.265）  
- 困惑度数据：WikiText-2（经 `datasets` 加载）  
- 本仓库用于研究目的；详细方法论与扩展实验设计见 `MODEL_AND_EVAL_GUIDE.md` 与 `PLAN.md`。


离线张量验证（tensor → codec → tensor）

python run.py --mode test_codec

权重压缩
执行压缩（生成压缩权重）
因为你已经完美重构了 topology_router.py 并从 run.py 中删除了 --hadamard-dim，所以现在的压缩命令非常清爽，程序会自动判断每一层该怎么旋转：
Bash
python run.py --mode compress_weights \
  --model EleutherAI/pythia-160m \
  --incoherence-block-size 1024
(注：它会自动根据 gpt_neox 的拓扑结构对输入层做 in_features 旋转，对输出层做 out_features 旋转。)


权重压缩完整的流程：

压缩权重（VcLLM 解压后评测）— 一次跑完 8 个 Zero-shot 任务
在仓库根目录执行（按你的压缩目录和模型名改路径）：
cd /workspace
python evaluation/lm_eval_integration.py \
--model EleutherAI/pythia-160m \
--compressed-dir ./compressed_weights \
--output ./results/vcllm_pythia160m_8task.json \
--num-fewshot 0 \
--batch-size 8 \
--tasks piqa,copa,arc_easy,arc_challenge,winogrande,hellaswag,rte,openbookqa
--tasks 可省略：默认就是上述 8 个任务。
若 NVDEC 解码报错，可加：--no-hardware-decode。
需要论文里的 stderr / 置信区间 时，把 --bootstrap-iters 设为例如 100000（会明显变慢）；默认 0 不估计 stderr。
若某任务提示需确认不安全代码，加：--confirm-run-unsafe-code。
冒烟测试：--limit 32（每任务只评少量样本）。

16-bit 全精度基线（Baseline）
不传 --compressed-dir 即可（同一模型、同一任务列表、同一 harness 设置）：
cd /workspace
python evaluation/lm_eval_integration.py \
--model EleutherAI/pythia-160m \
--output ./results/baseline_pythia160m_8task.json \
--num-fewshot 0 \
--batch-size 8 \
--tasks piqa,copa,arc_easy,arc_challenge,winogrande,hellaswag,rte,openbookqa
对比时保持 --num-fewshot、--batch-size、--bootstrap-iters、--limit 等与压缩版一致。





KV压缩

1. 显存与 BPE 终极 Profiling (测试长文本性能)
我们将生成长度拉到 512 个 token，用来验证 Chunking 机制下，随着文本变长，BPE 是否稳定在 2.9 左右，以及节省的显存有多么夸张。
Bash
python run.py --mode eval_kv_cache --model EleutherAI/pythia-160m --max-new-tokens 512
2. PIQA 零样本准确度 (完整验证智商保真度)
拿掉 --limit，跑满 PIQA 测试集的所有样本（约 1800+ 条）。有了 Chunking，原来无法容忍的耗时现在变得完全可控。
Bash
python evaluation/lm_eval_integration.py \
  --model EleutherAI/pythia-160m \
  --tasks piqa \
  --kv-cache-mode vcllm_hevc \
  --output results/piqa_vcllm_full.json
(同时你也需要运行 --kv-cache-mode none 和 rtn3 的完整版，用于最终的拉踩对比。)
3. WikiText-2 困惑度 (完整验证语言流畅度)
跑满整个测试集，彻底验证我们分块固化（不再重复压缩）后，是否完美消除了“量化漂移 (Quantization Drift)”。
Bash
python evaluation/eval_wikitext_ppl.py \
  --model EleutherAI/pythia-160m \
  --kv-cache-mode vcllm_hevc \
  --output results/wikitext_vcllm_full.json




diffusion验证VCLLM对权重压缩前后的图片生成质量对比
1. 
用于验证环境是否配置正确，以及 4D-2D 维度变换是否会破坏图像。

Bash
# 只做降维展开和填充，不做量化和压缩。用于排查内存指针错位。
HF_ENDPOINT=https://hf-mirror.com python evaluation/eval_diffusion_weight_compression.py --debug-mode reshape_only

# 只做 INT8 极值量化。用于排查是否存在离群值导致量化崩塌。
HF_ENDPOINT=https://hf-mirror.com python evaluation/eval_diffusion_weight_compression.py --debug-mode rtn_only
2. 真·无损基准测试 (True Lossless Baseline)
强制 FFmpeg 进入绝对无损模式（-tune lossless -qp 0），只压缩一部分层（up_blocks 以节省时间）。此时体积最大，但画质应与 FP16 原始模型 100% 相同。

Bash
HF_ENDPOINT=https://hf-mirror.com python evaluation/eval_diffusion_weight_compression.py --debug-mode partial_hevc
3. 裸奔的有损压缩 (无 Hadamard 对照组)
模拟业界粗暴使用视频编码器的错误做法。采用高强度有损压缩（QP=12）。这会产出一张带有明显劣化痕迹的图片，用于论文中的 Negative Baseline 对比图。

Bash
HF_ENDPOINT=https://hf-mirror.com python evaluation/eval_diffusion_weight_compression.py --hevc-lossy --qp 12
4. 终极火力全开版 (论文核心成果)
完整运用 VcLLM 理论 + Channel-wise Hadamard + 智能显存卸载 + 全局 UNet 压缩。
这是你汇报时最闪亮的一组数据，能够在获得极致显存压缩比（低 BPE）的同时，输出高清保真的图像。

Bash
HF_ENDPOINT=https://hf-mirror.com python evaluation/eval_diffusion_weight_compression.py --hevc-lossy --qp 12 --use-hadamard
(注：如果显存大于 40GB 且想加速实验，可附加 --no-offload-during-codec)



生成100张图片：
python evaluation/eval_diffusion_weight_compression.py \
  --num-images 100 \
  --hevc-lossy \
  --qp 0 \
  --output-dir outputs

评估输出指标：
python evaluation/calc_diffusion_metrics.py \
  --baseline-dir outputs/baseline \
  --compressed-dir outputs/compressed \
  --prompt-manifest outputs/prompts.csv \
  --blind-output-dir outputs/human_eval \
  --skip-clip

  