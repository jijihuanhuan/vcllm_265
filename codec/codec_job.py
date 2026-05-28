"""
Single configuration object for tensor→video encode/decode jobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BackendName = Literal["auto", "libx265", "hevc_nvenc"]


@dataclass
class CodecJob:
    """One encode/decode job: frame geometry, rate control, and backend selection."""

    width: int
    height: int
    fps: float = 1.0
    intra_only: bool = True
    qp: int = 0
    lossless: bool = True
    backend: BackendName = "auto"
    #: When ``False`` (default), lossy encode uses tensor-friendly settings (NVENC: spatial/temporal
    #: AQ off; libx265: ``-tune psnr`` with SAO/deblock/AQ disabled). When ``True``, encoder defaults
    #: apply — intended for real video, not weight tensors.
    visual_optimization: bool = False

    preset: str = field(default="slow", repr=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width and height must be positive, got {self.width}x{self.height}")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "intra_only": self.intra_only,
            "qp": self.qp,
            "lossless": self.lossless,
            "backend": self.backend,
            "preset": self.preset,
            "visual_optimization": self.visual_optimization,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CodecJob:
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            fps=float(data.get("fps", 1.0)),
            intra_only=bool(data.get("intra_only", True)),
            qp=int(data.get("qp", 0)),
            lossless=bool(data.get("lossless", True)),
            backend=data.get("backend", "auto"),  # type: ignore[arg-type]
            preset=str(data.get("preset", "slow")),
            visual_optimization=bool(data.get("visual_optimization", False)),
        )

    @classmethod
    def square(cls, frame_size: int, **kwargs: Any) -> CodecJob:
        """Convenience for square frames (current tensor tiling layout)."""
        return cls(width=frame_size, height=frame_size, **kwargs)
