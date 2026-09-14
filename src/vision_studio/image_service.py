from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, cast
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

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
from .schemas import InferenceResult



# uuid4() 为每次请求创建独立目录，避免不同用户覆盖文件。
# 即使 save_image=False，仍会生成 JSON，因为项目要求支持 JSON 下载。
# [:, :, ::-1] 把 BGR 三通道反转为 RGB。
# Lock 保护缓存的 YOLOE，避免请求之间的提示词互相覆盖。


@dataclass(frozen=True)
class ImageServiceResponse:
    """适合交给Web页面显示的一次图片检测结果。"""

    annotated_rgb: NDArray[np.uint8] = field(
        repr=False,
        compare=False,
    )
    detection_table: list[list[Any]]
    summary: str
    json_path: Path
    image_path: Path | None
    inference_result: InferenceResult = field(
        repr=False,
        compare=False,
    )


class ImageInferenceService:
    """组织模型选择、推理和Web输出转换。"""

    def __init__(
        self,
        detector_cache: DetectorCache | None = None,
        output_root: Path | None = None,
    ) -> None:
        self._cache = (
            detector_cache
            if detector_cache is not None
            else DetectorCache()
        )

        self._output_root = (
            output_root
            if output_root is not None
            else RUNS_DIR / "web"
        )

        # YOLOE的set_classes()会改变模型状态。
        # 暂时串行保护“设置类别+推理”过程。
        self._inference_lock = Lock()

    def run(
        self,
        mode: ModelMode,
        image_path: Path,
        prompt_text: str = "",
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
        image_size: int = DEFAULT_IMAGE_SIZE,
        save_image: bool = True,
    ) -> ImageServiceResponse:
        prompt_classes: list[str] = []

        if mode == "YOLOE":
            prompt_classes = parse_prompt_text(
                prompt_text
            )

        request_id = uuid4().hex
        request_dir = (
            self._output_root / request_id
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

            inference_result = detector.predict(
                image_path=image_path,
                output_dir=(
                    request_dir
                    if save_image
                    else None
                ),
                confidence=confidence,
                iou=iou,
                image_size=image_size,
            )

        # Ultralytics plot()是BGR，网页组件通常按RGB显示。
        annotated_rgb = (
            inference_result.annotated_image[
                :, :, ::-1
            ].copy()
        )

        detection_table = [
            [
                detection.label,
                detection.confidence,
                detection.bbox_xyxy[0],
                detection.bbox_xyxy[1],
                detection.bbox_xyxy[2],
                detection.bbox_xyxy[3],
            ]
            for detection
            in inference_result.detections
        ]

        summary = (
            f"模型：{mode}\n"
            f"目标数量："
            f"{inference_result.detection_count}\n"
            f"推理耗时："
            f"{inference_result.inference_ms:.2f} ms\n"
            f"处理阶段合计："
            f"{inference_result.total_ms:.2f} ms"
        )

        output_data = (
            inference_result.to_dict()
        )
        output_data["mode"] = mode
        output_data["prompts"] = (
            prompt_classes
        )

        json_path = save_json(
            output_data=output_data,
            json_path=request_dir / "result.json",
        )

        return ImageServiceResponse(
            annotated_rgb=annotated_rgb,
            detection_table=detection_table,
            summary=summary,
            json_path=json_path,
            image_path=(
                inference_result.output_path
            ),
            inference_result=inference_result,
        )