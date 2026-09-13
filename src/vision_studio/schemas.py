from typing import Any

from ultralytics.engine.results import Results

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DetectorConfig:
    model_path: Path
    device: int | str
    confidence: float = 0.25
    iou: float = 0.70
    image_size: int = 640

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence必须位于0到1之间")

        if not 0.0 <= self.iou <= 1.0:
            raise ValueError("iou必须位于0到1之间")

        if self.image_size <= 0:
            raise ValueError("image_size必须大于0")


def parse_detections(
    result: Results,
) -> list[dict[str, Any]]:
    """把Ultralytics结果转换成可序列化的检测列表。"""
    detections = []

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        coordinates = [
            round(value, 1)
            for value in box.xyxy[0].tolist()
        ]
        label = result.names[class_id]

        detections.append(
            {
                "class_id": class_id,
                "label": label,
                "confidence": round(confidence, 4),
                "bbox_xyxy": coordinates,
            }
        )

    return detections