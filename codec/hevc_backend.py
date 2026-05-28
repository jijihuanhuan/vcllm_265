import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from codec.codec_job import CodecJob

_ROOT_TMP = Path(os.environ.get("VCLLM_TMP", "/tmp")) / "vcllm"

# Cached result of ``assert_gpu_hevc_hw_codecs_available``: None = not yet probed, "ok" / "fail".
_gpu_hevc_hw_probe: str | None = None
_gpu_hevc_hw_fail_reason: str | None = None

ERR_GPU_HW_CODECS_UNAVAILABLE = (
    "GPU 硬件 HEVC 不可用：ffmpeg 未列出 hevc_nvenc 或 hevc_cuvid（或 ffmpeg 不可用）。"
    "禁止回退到 CPU 软件编解码（libx265 / PNG 路径）。"
)

ERR_NVENC_FAILED = (
    "GPU 硬件编码器 (hevc_nvenc) 调用失败，禁止回退到 CPU/libx265。"
    "ffmpeg stderr:\n{stderr}"
)

ERR_CUVID_FAILED = (
    "GPU 硬件解码器 (hevc_cuvid) 调用失败或未产出帧，禁止回退到 CPU/PNG 解码。"
    "ffmpeg stderr:\n{stderr}"
)

_NV_YIELD_SLEEP_SEC = 0.1


def _ffmpeg_input_fullrange_prefix() -> list[str]:
    """Input-side: treat following image2/PNG gray frames as PC / full swing (0–255)."""
    return ["-color_range", "pc"]


def _ffmpeg_hevc_nvenc_fullrange_output_tags() -> list[str]:
    """
    Output-side VUI / tagging for hevc_nvenc so bitstream is not interpreted as TV (16–235).
    Pairs with input ``-color_range pc`` for uint8 tensor round-trips.
    """
    return [
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-color_range",
        "pc",
    ]


def _yield_cuda_before_ffmpeg_subprocess() -> None:
    """
    Before spawning ffmpeg (NVENC/CUVID), drain PyTorch CUDA streams and sleep briefly so
    the GPU scheduler / power firmware can yield to the video engine (mitigates timeouts
    and hard hangs when compute is saturated).
    """
    try:
        import torch
    except ImportError:
        time.sleep(_NV_YIELD_SLEEP_SEC)
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time.sleep(_NV_YIELD_SLEEP_SEC)


def _ffmpeg_cuda_hw_failure_hint(stderr: str) -> str:
    """Extra context when NVENC/CUVID fails in the ffmpeg child process."""
    s = stderr or ""
    if "CUDA_ERROR_NO_DEVICE" in s or ("cuInit" in s and "failed" in s):
        return (
            "\n\n[诊断] ffmpeg 的 GPU HEVC 路径（hevc_nvenc / hevc_cuvid）无法在子进程中初始化 CUDA，"
            "这与「PyTorch 能使用 torch.cuda」并不等价：系统 ffmpeg 可能链接了另一套 libcuda，"
            "或容器/环境未把 GPU 设备暴露给 ffmpeg 子进程。"
            "建议：在同一 shell 运行 "
            "`ffmpeg -hide_banner -f lavfi -i color=gray:s=256x256:r=1 -frames:v 1 -c:v hevc_nvenc -f null -` "
            "（NVENC HEVC 常见最小边长 ≥128/144；仓库探针与 tensor 填充使用 256×256。）"
            "若仍失败，请更换与主机 NVIDIA 驱动匹配的 ffmpeg 构建，或检查 NVIDIA_VISIBLE_DEVICES / 容器 GPU。"
        )
    if "minimum supported" in s or "InitializeEncoder failed" in s:
        return (
            "\n\n[诊断] NVENC 拒绝当前帧分辨率：HEVC NVENC 有最小宽高要求。"
            "请使用至少 256×256 的探针/帧缓冲（见 ``codec.frame_mapper.NVENC_MIN_FRAME_HW``）。"
        )
    if "CUDA_ERROR_OUT_OF_MEMORY" in s or "out of memory" in s.lower():
        return (
            "\n\n[诊断] ffmpeg 创建 CUDA 解码上下文时显存不足：PyTorch 若已占满整卡，"
            "hevc_cuvid 再申请设备内存会失败。可将大模型临时 offload 到 CPU 再跑 NVENC/NVDEC，"
            "或见 ``evaluation/eval_diffusion_weight_compression.py`` 默认在 HEVC 阶段前的 pipeline CPU 卸载逻辑。"
        )
    return ""


def _probe_hevc_nvenc_minimal_subprocess() -> None:
    """
    One-frame grayscale NVENC encode via lavfi (256×256, NVENC-safe minimum).

    Catches ``ffmpeg -encoders`` listing hevc_nvenc while runtime cuInit fails.
    """
    work = _unique_work_dir("nvenc_probe")
    try:
        out_hevc = work / "probe.hevc"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            *_ffmpeg_input_fullrange_prefix(),
            "-f",
            "lavfi",
            "-i",
            "color=gray:s=256x256:r=1",
            "-frames:v",
            "1",
            "-c:v",
            "hevc_nvenc",
            "-g",
            "1",
            "-bf",
            "0",
            "-tune",
            "lossless",
            "-qp",
            "0",
            "-lossless",
            "1",
            *_ffmpeg_hevc_nvenc_fullrange_output_tags(),
            str(out_hevc),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stderr = result.stderr or ""
        if result.returncode != 0:
            raise RuntimeError(
                "GPU 硬件编码器 (hevc_nvenc) 运行时探测失败：单帧灰图编码未成功。"
                f"ffmpeg stderr:\n{stderr}"
                + _ffmpeg_cuda_hw_failure_hint(stderr)
            )
        if not out_hevc.is_file() or out_hevc.stat().st_size == 0:
            raise RuntimeError(
                "GPU 硬件编码器 (hevc_nvenc) 运行时探测失败：未生成有效码流文件。"
                + _ffmpeg_cuda_hw_failure_hint(stderr)
            )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def assert_gpu_hevc_hw_codecs_available() -> None:
    """
    Verify ``ffmpeg`` lists ``hevc_nvenc`` and ``hevc_cuvid``. Raises ``RuntimeError`` on failure.
    Successful probes are cached for the process.
    """
    global _gpu_hevc_hw_probe, _gpu_hevc_hw_fail_reason
    if _gpu_hevc_hw_probe == "ok":
        return
    if _gpu_hevc_hw_probe == "fail":
        raise RuntimeError(_gpu_hevc_hw_fail_reason or ERR_GPU_HW_CODECS_UNAVAILABLE)

    try:
        enc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        dec = subprocess.run(
            ["ffmpeg", "-hide_banner", "-decoders"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as e:
        _gpu_hevc_hw_probe = "fail"
        _gpu_hevc_hw_fail_reason = f"{ERR_GPU_HW_CODECS_UNAVAILABLE} (未找到 ffmpeg 可执行文件。)"
        raise RuntimeError(_gpu_hevc_hw_fail_reason) from e
    except subprocess.TimeoutExpired as e:
        _gpu_hevc_hw_probe = "fail"
        _gpu_hevc_hw_fail_reason = f"{ERR_GPU_HW_CODECS_UNAVAILABLE} (ffmpeg 探测超时。)"
        raise RuntimeError(_gpu_hevc_hw_fail_reason) from e

    blob = (enc.stdout or "") + (enc.stderr or "") + (dec.stdout or "") + (dec.stderr or "")
    if "hevc_nvenc" not in blob or "hevc_cuvid" not in blob:
        _gpu_hevc_hw_probe = "fail"
        _gpu_hevc_hw_fail_reason = ERR_GPU_HW_CODECS_UNAVAILABLE
        raise RuntimeError(ERR_GPU_HW_CODECS_UNAVAILABLE)

    try:
        _probe_hevc_nvenc_minimal_subprocess()
    except RuntimeError as e:
        _gpu_hevc_hw_probe = "fail"
        _gpu_hevc_hw_fail_reason = str(e)
        raise

    _gpu_hevc_hw_probe = "ok"
    _gpu_hevc_hw_fail_reason = None


def _unique_work_dir(prefix: str) -> Path:
    _ROOT_TMP.mkdir(parents=True, exist_ok=True)
    d = _ROOT_TMP / f"{prefix}_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _nvenc_intra_args(job: CodecJob) -> list[str]:
    extra: list[str] = []
    if job.intra_only:
        extra.extend(["-g", "1", "-bf", "0"])
    return extra


def _libx265_intra_args(job: CodecJob) -> list[str]:
    extra: list[str] = ["-g", "1", "-an"]
    if job.intra_only:
        pass  # -g 1 already forces all-intra for typical HEVC usage
    return extra


def encode_frames_to_bitstream(
    frames: np.ndarray,
    output_path: str | os.PathLike[str],
    codec_job: CodecJob,
    *,
    allow_software_encoder_fallback: bool = True,
) -> int:
    """
    Encode grayscale frames (N, H, W) uint8 to HEVC bitstream.
    Uses a unique temp directory per call to avoid concurrent /tmp races.
    Input PNGs and encoder VUI are tagged with PC / full range (``-color_range pc``) so
    0–255 values are not interpreted as limited TV range (16–235).
    """
    if frames.ndim != 3:
        raise ValueError(f"frames must be (N, H, W), got {frames.shape}")
    n, h, w = frames.shape
    if n == 0:
        raise ValueError("empty frame stack")
    if h != codec_job.height or w != codec_job.width:
        raise ValueError(
            f"frame shape ({h}, {w}) does not match CodecJob ({codec_job.height}, {codec_job.width})"
        )

    if not allow_software_encoder_fallback:
        assert_gpu_hevc_hw_codecs_available()
        if codec_job.backend == "libx265":
            raise RuntimeError(
                "禁止 CPU 软件编码：当前路径要求 GPU hevc_nvenc，但 CodecJob.backend 为 libx265。"
            )

    work = _unique_work_dir("enc_frames")
    try:
        for i in range(n):
            img = Image.fromarray(frames[i], mode="L")
            img.save(work / f"frame_{i:04d}.png")

        fps = max(codec_job.fps, 1e-6)
        cmd: list[str] = [
            "ffmpeg",
            "-y",
            *_ffmpeg_input_fullrange_prefix(),
            "-framerate",
            str(fps),
            "-start_number",
            "0",
            "-i",
            str(work / "frame_%04d.png"),
        ]

        backend = codec_job.backend
        if backend == "auto":
            backend = "hevc_nvenc"

        if backend == "hevc_nvenc":
            cmd.extend(["-c:v", "hevc_nvenc"])
            cmd.extend(_nvenc_intra_args(codec_job))
            if codec_job.lossless:
                # True lossless: avoid -rc constqp (rate control conflicts). NVENC needs
                # -tune lossless + -qp 0 together with -lossless 1 for a strict lossless path.
                cmd.extend(["-tune", "lossless", "-qp", "0", "-lossless", "1"])
            else:
                cmd.extend(["-rc", "constqp", "-qp", str(codec_job.qp)])
                if not codec_job.visual_optimization:
                    # Disable AQ for tensor-like data. Do not use ``-tune psnr``: many ffmpeg 4.x
                    # hevc_nvenc builds reject it (only e.g. hq/uhq/lossless/ll are valid).
                    cmd.extend(["-spatial-aq", "0", "-temporal-aq", "0"])
            cmd.extend(_ffmpeg_hevc_nvenc_fullrange_output_tags())
        elif backend == "libx265":
            cmd.extend(["-c:v", "libx265"])
            cmd.extend(["-preset", codec_job.preset])
            cmd.extend(_libx265_intra_args(codec_job))
            cmd.extend(["-qp", str(codec_job.qp)])
            if codec_job.lossless:
                cmd.extend(["-x265-params", "lossless=1:range=full"])
            else:
                if codec_job.visual_optimization:
                    cmd.extend(["-x265-params", "range=full"])
                else:
                    cmd.extend(["-tune", "psnr"])
                    cmd.extend(
                        [
                            "-x265-params",
                            "range=full:aq-mode=0:no-sao=1:no-deblock=1",
                        ]
                    )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd.append(str(out))

        _yield_cuda_before_ffmpeg_subprocess()
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            if (
                allow_software_encoder_fallback
                and codec_job.backend == "auto"
                and backend == "hevc_nvenc"
            ):
                print(f"Encoding failed with code {result.returncode}")
                print(f"Error: {result.stderr}")
                print("Falling back to software encoding (libx265)...")
                fallback = CodecJob(
                    width=codec_job.width,
                    height=codec_job.height,
                    fps=codec_job.fps,
                    intra_only=codec_job.intra_only,
                    qp=codec_job.qp,
                    lossless=codec_job.lossless,
                    backend="libx265",
                    preset=codec_job.preset,
                    visual_optimization=codec_job.visual_optimization,
                )
                return encode_frames_to_bitstream(
                    frames,
                    output_path,
                    fallback,
                    allow_software_encoder_fallback=False,
                )
            stderr = result.stderr or ""
            if backend == "hevc_nvenc" and not allow_software_encoder_fallback:
                raise RuntimeError(
                    ERR_NVENC_FAILED.format(stderr=stderr) + _ffmpeg_cuda_hw_failure_hint(stderr)
                )
            raise RuntimeError(f"Encoding failed: {stderr}")

        return result.returncode
    finally:
        shutil.rmtree(work, ignore_errors=True)


def decode_bitstream_to_frames(
    input_path: str | os.PathLike[str],
    codec_job: CodecJob,
    *,
    force_software_decode: bool = False,
    prefer_hardware_decode: bool = False,
    allow_software_decoder_fallback: bool = True,
) -> np.ndarray:
    """
    Decode HEVC bitstream to grayscale frames (N, H, W) uint8.
    `codec_job.width` / `codec_job.height` must match encoded frame geometry
    (required for GPU raw YUV path). Software PNG decode must match the same size.

    Uses ``-color_range pc`` on decode so uint8 tensor round-trips are not clamped to TV range.

    For tensor/weight pipelines that require bit-exact uint8 recovery, pass
    ``force_software_decode=True``: CUDA (hevc_cuvid) can return frames with the
    correct shape while corrupting pixel values for some NVENC bitstreams.

    If ``prefer_hardware_decode=True`` (and not forcing software), try hevc_cuvid first
    even when ``codec_job.backend`` is ``libx265``.

    ``allow_software_decoder_fallback=False``: require ``hevc_cuvid`` to succeed; raise
    instead of PNG/software decode.
    """
    w, h = codec_job.width, codec_job.height

    if not allow_software_decoder_fallback:
        assert_gpu_hevc_hw_codecs_available()

    work = _unique_work_dir("dec_frames")
    try:
        if not allow_software_decoder_fallback and force_software_decode:
            raise ValueError(
                "allow_software_decoder_fallback=False conflicts with force_software_decode=True"
            )

        try_cuda = (not force_software_decode) and (
            prefer_hardware_decode or codec_job.backend in ("auto", "hevc_nvenc")
        )

        if not allow_software_decoder_fallback and not try_cuda:
            raise RuntimeError(
                "Hardware decode required (allow_software_decoder_fallback=False) but the "
                "CUDA decode path is not selected for this CodecJob "
                f"(backend={codec_job.backend!r}, prefer_hardware_decode={prefer_hardware_decode})."
            )

        if try_cuda:
            # ``-f image2`` is required so ``frame_%04d.yuv`` is expanded to numbered
            # files. Without it, ffmpeg writes a single file literally named
            # ``frame_%04d.yuv`` and our reader finds no ``frame_0000.yuv``.
            cmd = [
                "ffmpeg",
                "-y",
                "-hwaccel",
                "cuda",
                "-c:v",
                "hevc_cuvid",
                *_ffmpeg_input_fullrange_prefix(),
                "-i",
                str(input_path),
                "-vsync",
                "0",
                "-color_range",
                "pc",
                "-f",
                "image2",
                "-start_number",
                "0",
                "-c:v",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-s",
                f"{w}x{h}",
                str(work / "frame_%04d.yuv"),
            ]
            _yield_cuda_before_ffmpeg_subprocess()
            result = subprocess.run(cmd, capture_output=True, text=True)
            cuda_stderr = result.stderr or ""

            frames_list: list[np.ndarray] = []
            if result.returncode == 0:
                i = 0
                while True:
                    frame_path = work / f"frame_{i:04d}.yuv"
                    if frame_path.is_file():
                        frame_data = frame_path.read_bytes()
                        expected = w * h
                        if len(frame_data) < expected:
                            break
                        frame = np.frombuffer(frame_data, dtype=np.uint8, count=expected).reshape(
                            h, w
                        )
                        frames_list.append(frame)
                        frame_path.unlink(missing_ok=True)
                        i += 1
                    else:
                        break

            if len(frames_list) > 0:
                return np.array(frames_list)

            if not allow_software_decoder_fallback:
                raise RuntimeError(
                    ERR_CUVID_FAILED.format(stderr=cuda_stderr) + _ffmpeg_cuda_hw_failure_hint(cuda_stderr)
                )

            if codec_job.backend == "auto":
                print("CUDA decode failed or unavailable; falling back to software PNG decode...")
            elif codec_job.backend == "hevc_nvenc" or prefer_hardware_decode:
                print("hevc_cuvid decode failed or produced no frames; falling back to software PNG decode...")

        cmd_sw = [
            "ffmpeg",
            "-y",
            *_ffmpeg_input_fullrange_prefix(),
            "-i",
            str(input_path),
            "-vsync",
            "0",
            "-start_number",
            "0",
            str(work / "frame_%04d.png"),
        ]
        _yield_cuda_before_ffmpeg_subprocess()
        result_sw = subprocess.run(cmd_sw, capture_output=True, text=True)
        if result_sw.returncode != 0:
            raise RuntimeError(f"Decoding failed: {result_sw.stderr}")

        frames_png: list[np.ndarray] = []
        i = 0
        while True:
            frame_path = work / f"frame_{i:04d}.png"
            if frame_path.is_file():
                img = Image.open(frame_path).convert("L")
                arr = np.array(img)
                frames_png.append(arr)
                frame_path.unlink(missing_ok=True)
                i += 1
            else:
                break

        if len(frames_png) == 0:
            raise RuntimeError("No frames decoded. Check if the input file is valid.")

        out = np.array(frames_png)
        if out.shape[1] != h or out.shape[2] != w:
            raise ValueError(
                f"Decoded frame size {out.shape[1:]} does not match CodecJob ({h}, {w})"
            )
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)
