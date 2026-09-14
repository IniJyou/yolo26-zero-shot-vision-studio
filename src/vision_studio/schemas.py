from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from ultralytics.engine.results import Results


@dataclass(frozen=True)
class DetectorConfig:
    model_path: Path
    device: int | str
    confidence: float = 0.25
    iou: float = 0.70
    image_size: int = 640

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "confidence必须位于0到1之间"
            )

        if not 0.0 <= self.iou <= 1.0:
            raise ValueError(
                "iou必须位于0到1之间"
            )

        if self.image_size <= 0:
            raise ValueError(
                "image_size必须大于0"
            )


@dataclass(frozen=True)
class Detection:
    """单个检测目标。"""
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[
        float,
        float,
        float,
        float,
    ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
        }


@dataclass(frozen=True)
class InferenceResult:
    """一次图片推理产生的统一结果。"""
    source_path: Path
    model_name: str
    image_width: int
    image_height: int
    detections: list[Detection]
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float

    # Ultralytics的plot()返回BGR格式的NumPy图片。
    # 不把它写入JSON，只留给后续Web页面显示。
    annotated_image: NDArray[np.uint8] = field(
        repr=False,
        compare=False,
    )

    output_path: Path | None = None

    @property
    def detection_count(self) -> int:
        return len(self.detections)

    @property
    def total_ms(self) -> float:
        return round(
            self.preprocess_ms
            + self.inference_ms
            + self.postprocess_ms,
            2,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source_path),
            "model": self.model_name,
            "image_size": {
                "width": self.image_width,
                "height": self.image_height,
            },
            "speed_ms": {
                "preprocess": self.preprocess_ms,
                "inference": self.inference_ms,
                "postprocess": self.postprocess_ms,
                "total": self.total_ms,
            },
            "detection_count": self.detection_count,
            "detections": [
                detection.to_dict()
                for detection in self.detections
            ],
            "output_path": (
                str(self.output_path)
                if self.output_path is not None
                else None
            ),
        }


def parse_detections(
    result: Results,
) -> list[Detection]:
    """把Ultralytics检测框转换为Detection列表。"""
    detections: list[Detection] = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        coordinates = box.xyxy[0].tolist()

        detection = Detection(
            class_id=class_id,
            label=result.names[class_id],
            confidence=round(confidence, 4),
            bbox_xyxy=(
                round(coordinates[0], 1),
                round(coordinates[1], 1),
                round(coordinates[2], 1),
                round(coordinates[3], 1),
            ),
        )

        detections.append(detection)

    return detections


def build_inference_result(
    result: Results,
    source_path: Path,
    model_name: str,
    output_path: Path | None = None,
) -> InferenceResult:
    """把Ultralytics Results转换为项目统一结果。"""
    height, width = result.orig_shape

    return InferenceResult(
        source_path=source_path,
        model_name=model_name,
        image_width=width,
        image_height=height,
        detections=parse_detections(result),
        preprocess_ms=round(
            float(result.speed.get("preprocess", 0.0)),
            2,
        ),
        inference_ms=round(
            float(result.speed.get("inference", 0.0)),
            2,
        ),
        postprocess_ms=round(
            float(result.speed.get("postprocess", 0.0)),
            2,
        ),
        annotated_image=result.plot(),
        output_path=output_path,
    )