from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import cast
from uuid import uuid4

from .config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    RUNS_DIR,
)
from .detectors import YOLOEDetector
from .io_utils import save_json
from .model_cache import (
    DetectorCache,
    ModelMode,
)
from .prompt_utils import parse_prompt_text
from .video import (
    ProgressCallback,
    VideoInferenceResult,
    VideoProcessor,
)


@dataclass(frozen=True)
class VideoServiceResponse:
    video_path: Path
    json_path: Path
    summary: str
    inference_result: VideoInferenceResult = field(
        repr=False,
        compare=False,
    )


class VideoInferenceService:
    """组织视频模型选择、处理和结果保存。"""

    def __init__(
        self,
        detector_cache: DetectorCache | None = None,
        processor: VideoProcessor | None = None,
        output_root: Path | None = None,
    ) -> None:
        self._cache = (
            detector_cache
            if detector_cache is not None
            else DetectorCache()
        )

        self._processor = (
            processor
            if processor is not None
            else VideoProcessor()
        )

        self._output_root = (
            output_root
            if output_root is not None
            else RUNS_DIR / "video"
        )

        self._inference_lock = Lock()

    def run(
        self,
        mode: ModelMode,
        video_path: Path,
        prompt_text: str = "",
        confidence: float = (
            DEFAULT_CONFIDENCE
        ),
        iou: float = DEFAULT_IOU,
        image_size: int = DEFAULT_IMAGE_SIZE,
        progress_callback: (
            ProgressCallback | None
        ) = None,
    ) -> VideoServiceResponse:
        prompt_classes: list[str] = []

        if mode == "YOLOE":
            prompt_classes = parse_prompt_text(
                prompt_text
            )

        request_dir = (
            self._output_root
            / uuid4().hex
        )
        output_path = (
            request_dir / "result.mp4"
        )

        with self._inference_lock:
            detector = self._cache.get(mode)

            if mode == "YOLOE":
                yoloe_detector = cast(
                    YOLOEDetector,
                    detector,
                )
                yoloe_detector.set_classes(
                    prompt_classes
                )

            result = self._processor.process(
                detector=detector,
                video_path=video_path,
                output_path=output_path,
                confidence=confidence,
                iou=iou,
                image_size=image_size,
                progress_callback=(
                    progress_callback
                ),
            )

        output_data = result.to_dict()
        output_data["mode"] = mode
        output_data["prompts"] = (
            prompt_classes
        )

        json_path = save_json(
            output_data=output_data,
            json_path=(
                request_dir / "result.json"
            ),
        )

        summary = (
            f"模型：{mode}\n"
            f"原视频："
            f"{result.metadata.width}"
            f"x{result.metadata.height}，"
            f"{result.metadata.fps:.2f} FPS\n"
            f"处理帧数："
            f"{result.processed_frames}\n"
            f"逐帧检测框合计："
            f"{result.total_frame_detections}\n"
            f"平均模型推理："
            f"{result.average_inference_ms:.2f} ms\n"
            f"整体处理速度："
            f"{result.processing_fps:.2f} FPS"
        )

        return VideoServiceResponse(
            video_path=result.output_path,
            json_path=json_path,
            summary=summary,
            inference_result=result,
        )