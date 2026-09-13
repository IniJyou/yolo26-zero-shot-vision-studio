from pathlib import Path

from vision_studio.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_DEVICE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU,
    PROJECT_ROOT,
    YOLO26_MODEL_PATH,
)
from vision_studio.detectors import YOLO26Detector
from vision_studio.io_utils import save_json
from vision_studio.schemas import DetectorConfig, parse_detections


def main() -> None:
    # 1. 构建检测器配置
    config = DetectorConfig(
        model_path=YOLO26_MODEL_PATH,
        device=DEFAULT_DEVICE,
        confidence=DEFAULT_CONFIDENCE,
        iou=DEFAULT_IOU,
        image_size=DEFAULT_IMAGE_SIZE,
    )

    # 2. 定义输入和输出路径
    image_path = PROJECT_ROOT / "weights" / "bus.jpg"
    output_dir = PROJECT_ROOT / "runs" / "python_first"

    # 3. 创建检测器并推理（验证输入和创建输出目录都在 detector 内部自动完成）
    detector = YOLO26Detector(config)
    result = detector.predict(
        image_path=image_path,
        output_dir=output_dir,
    )

    # 4. 打印推理概览
    height, width = result.orig_shape
    pipeline_ms = sum(result.speed.values())

    print(f"运行设备：{config.device}")
    print(f"置信度阈值：{config.confidence}")
    print(f"原图宽度：{width}")
    print(f"原图高度：{height}")
    print(f"检测数量：{len(result.boxes)}")
    print(f"各阶段耗时：{result.speed}")
    print(f"处理阶段合计：{pipeline_ms:.1f} ms")

    # 5. 解析检测框
    detections = parse_detections(result)

    for detection in detections:
        print(
            f"类别={detection['label']}, "
            f"置信度={detection['confidence']:.3f}, "
            f"坐标={detection['bbox_xyxy']}"
        )

    # 6. 构建输出数据
    output_data = {
        "source": str(image_path),
        "model": config.model_path.name,
        "image_size": {
            "width": width,
            "height": height,
        },
        "speed_ms": {
            "preprocess": round(result.speed["preprocess"], 2),
            "inference": round(result.speed["inference"], 2),
            "postprocess": round(result.speed["postprocess"], 2),
            "total": round(pipeline_ms, 2),
        },
        "detection_count": len(detections),
        "detections": detections,
    }

    # 7. 保存 JSON
    json_path = save_json(
        output_data=output_data,
        json_path=output_dir / "bus.json",
    )

    print(f"JSON结果已保存：{json_path}")


if __name__ == "__main__":
    main()