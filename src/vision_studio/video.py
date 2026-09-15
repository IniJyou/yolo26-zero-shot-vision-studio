from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np

from .config import (
    DEFAULT_MAX_VIDEO_SECONDS,
    SUPPORTED_VIDEO_SUFFIXES,
)
from .detectors import BaseDetector
from .schemas import Detection

# processing_fps 是整个程序每秒处理多少帧。
# metadata.fps 是原视频播放帧率。
# total_frame_detections 是所有帧检测框数量之和，不是去重后的真实目标数。
# 当前版本输出无音频视频；音频保留放到下一阶段。
# yuv420p + H.264 用于提高浏览器兼容性。
# 常见视频尺寸通常是偶数；奇数尺寸可能被 H.264 编码器拒绝，之后再做自动填充。


ProgressCallback = Callable[
    [int, int],
    None,
]


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": (
                self.duration_seconds
            ),
        }


@dataclass(frozen=True)
class FrameInferenceResult:
    frame_index: int
    timestamp_seconds: float
    inference_ms: float
    detections: list[Detection]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": (
                self.timestamp_seconds
            ),
            "inference_ms": self.inference_ms,
            "detection_count": len(
                self.detections
            ),
            "detections": [
                detection.to_dict()
                for detection in self.detections
            ],
        }


@dataclass(frozen=True)
class VideoInferenceResult:
    source_path: Path
    output_path: Path
    metadata: VideoMetadata
    processed_frames: int
    total_frame_detections: int
    elapsed_seconds: float
    processing_fps: float
    average_inference_ms: float
    frames: list[FrameInferenceResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source_path),
            "output": str(self.output_path),
            "metadata": self.metadata.to_dict(),
            "processed_frames": (
                self.processed_frames
            ),
            "total_frame_detections": (
                self.total_frame_detections
            ),
            "elapsed_seconds": (
                self.elapsed_seconds
            ),
            "processing_fps": (
                self.processing_fps
            ),
            "average_inference_ms": (
                self.average_inference_ms
            ),
            "frames": [
                frame.to_dict()
                for frame in self.frames
            ],
        }


def read_video_metadata(
    video_path: Path,
) -> VideoMetadata:
    if not video_path.is_file():
        raise FileNotFoundError(
            f"输入视频不存在：{video_path}"
        )

    if (
        video_path.suffix.lower()
        not in SUPPORTED_VIDEO_SUFFIXES
    ):
        raise ValueError(
            f"不支持的视频格式："
            f"{video_path.suffix}"
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    try:
        if not capture.isOpened():
            raise ValueError(
                f"无法打开视频：{video_path}"
            )

        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )
        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )
        fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )
        frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

    finally:
        capture.release()

    if width <= 0 or height <= 0:
        raise ValueError(
            "视频宽度或高度无效"
        )

    if fps <= 0:
        raise ValueError(
            "无法读取视频帧率"
        )

    duration_seconds = (
        frame_count / fps
        if frame_count > 0
        else 0.0
    )

    return VideoMetadata(
        width=width,
        height=height,
        fps=round(fps, 3),
        frame_count=frame_count,
        duration_seconds=round(
            duration_seconds,
            3,
        ),
    )


class VideoProcessor:
    """读取、检测并编码视频。"""

    def __init__(
        self,
        max_duration_seconds: float = (
            DEFAULT_MAX_VIDEO_SECONDS
        ),
    ) -> None:
        if max_duration_seconds <= 0:
            raise ValueError(
                "最大视频时长必须大于0"
            )

        self.max_duration_seconds = (
            max_duration_seconds
        )

    def process(
        self,
        detector: BaseDetector,
        video_path: Path,
        output_path: Path,
        confidence: float,
        iou: float,
        image_size: int,
        progress_callback: (
            ProgressCallback | None
        ) = None,
    ) -> VideoInferenceResult:
        metadata = read_video_metadata(
            video_path
        )

        if (
            metadata.duration_seconds
            > self.max_duration_seconds
        ):
            raise ValueError(
                "视频时长不能超过"
                f"{self.max_duration_seconds:.0f}秒"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            capture.release()
            raise ValueError(
                f"无法打开视频：{video_path}"
            )

        writer = imageio_ffmpeg.write_frames(
            path=str(output_path),
            size=(
                metadata.width,
                metadata.height,
            ),
            pix_fmt_in="bgr24",
            pix_fmt_out="yuv420p",
            fps=metadata.fps,
            codec="libx264",
            quality=7,
            macro_block_size=1,
            ffmpeg_log_level="error",
            output_params=[
                "-movflags",
                "+faststart",
            ],
        )

        start_time = perf_counter()
        frame_results: list[
            FrameInferenceResult
        ] = []
        processed_frames = 0
        inference_time_sum = 0.0

        try:
            writer.send(None)

            while True:
                success, frame_bgr = (
                    capture.read()
                )

                if not success:
                    break

                inference_result = (
                    detector.predict_frame(
                        frame_bgr=frame_bgr,
                        source_path=video_path,
                        confidence=confidence,
                        iou=iou,
                        image_size=image_size,
                    )
                )

                annotated_frame = (
                    np.ascontiguousarray(
                        inference_result
                        .annotated_image
                    )
                )

                writer.send(annotated_frame)

                frame_result = (
                    FrameInferenceResult(
                        frame_index=(
                            processed_frames
                        ),
                        timestamp_seconds=round(
                            processed_frames
                            / metadata.fps,
                            3,
                        ),
                        inference_ms=(
                            inference_result
                            .inference_ms
                        ),
                        detections=(
                            inference_result
                            .detections
                        ),
                    )
                )

                frame_results.append(
                    frame_result
                )

                processed_frames += 1
                inference_time_sum += (
                    inference_result
                    .inference_ms
                )

                if (
                    progress_callback
                    is not None
                ):
                    progress_callback(
                        processed_frames,
                        metadata.frame_count,
                    )

        finally:
            capture.release()
            writer.close()

        if processed_frames == 0:
            output_path.unlink(
                missing_ok=True
            )
            raise ValueError(
                "视频中没有可处理的帧"
            )

        elapsed_seconds = (
            perf_counter() - start_time
        )

        total_frame_detections = sum(
            len(frame.detections)
            for frame in frame_results
        )

        return VideoInferenceResult(
            source_path=video_path,
            output_path=output_path,
            metadata=metadata,
            processed_frames=processed_frames,
            total_frame_detections=(
                total_frame_detections
            ),
            elapsed_seconds=round(
                elapsed_seconds,
                3,
            ),
            processing_fps=round(
                processed_frames
                / elapsed_seconds,
                3,
            ),
            average_inference_ms=round(
                inference_time_sum
                / processed_frames,
                3,
            ),
            frames=frame_results,
        )