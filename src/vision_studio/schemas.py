from typing import Any

from ultralytics.engine.results import Results


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