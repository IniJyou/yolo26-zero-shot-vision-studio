from pathlib import Path

from vision_studio.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_DEVICE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    PROJECT_ROOT,
    YOLOE_MODEL_PATH,
)
from vision_studio.detectors import YOLOEDetector
from vision_studio.io_utils import save_json
from vision_studio.prompt_utils import parse_prompt_text
from vision_studio.schemas import (
    DetectorConfig,
    parse_detections,
)


def main() -> None:
    config = DetectorConfig(
        model_path=YOLOE_MODEL_PATH,
        device=DEFAULT_DEVICE,
        confidence=DEFAULT_CONFIDENCE,
        iou=DEFAULT_IOU,
        image_size=DEFAULT_IMAGE_SIZE,
    )

    image_path = PROJECT_ROOT / "weights" / "bus.jpg"
    output_dir = PROJECT_ROOT / "runs" / "yoloe_first"

    prompt_text = "double-decker bus, person"
    classes = parse_prompt_text(prompt_text)

    detector = YOLOEDetector(config)
    detector.set_classes(classes)

    result = detector.predict(
        image_path=image_path,
        output_dir=output_dir,
    )

    height, width = result.orig_shape
    detections = parse_detections(result)
    pipeline_ms = sum(result.speed.values())

    print(f"运行设备：{config.device}")
    print(f"提示类别：{classes}")
    print(f"原图尺寸：{width}x{height}")
    print(f"检测数量：{len(detections)}")
    print(f"处理阶段合计：{pipeline_ms:.1f} ms")

    for detection in detections:
        print(
            f"类别={detection['label']}, "
            f"置信度={detection['confidence']:.3f}, "
            f"坐标={detection['bbox_xyxy']}"
        )

    output_data = {
        "source": str(image_path),
        "model": config.model_path.name,
        "mode": "yoloe_zero_shot",
        "prompts": classes,
        "image_size": {
            "width": width,
            "height": height,
        },
        "speed_ms": {
            "preprocess": round(
                result.speed["preprocess"],
                2,
            ),
            "inference": round(
                result.speed["inference"],
                2,
            ),
            "postprocess": round(
                result.speed["postprocess"],
                2,
            ),
            "total": round(pipeline_ms, 2),
        },
        "detection_count": len(detections),
        "detections": detections,
    }

    json_path = save_json(
        output_data=output_data,
        json_path=output_dir / "bus.json",
    )

    print(f"JSON结果已保存：{json_path}")


if __name__ == "__main__":
    main()