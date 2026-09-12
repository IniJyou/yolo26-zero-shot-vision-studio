import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.results import Results

# 找到项目根目录
from vision_studio.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_DEVICE,
    PROJECT_ROOT,
    SUPPORTED_IMAGE_SUFFIXES,
    YOLO26_MODEL_PATH,
)

def validate_inputs(
    model_path: Path,
    image_path: Path,
) -> None:
    """检查模型权重和输入图片是否合法。"""
    if not model_path.is_file():
        raise FileNotFoundError(f"模型权重不存在：{model_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"输入图片不存在：{image_path}")

    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"不支持的图片格式：{image_path.suffix}")


def parse_detections(result: Results) -> list[dict[str, object]]:
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


def save_json(
    output_data: dict[str, object],
    json_path: Path,
) -> Path:
    """将检测结果保存为UTF-8 JSON文件。"""
    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_text = json.dumps(
        output_data,
        ensure_ascii=False,
        indent=2,
    )

    json_path.write_text(
        json_text,
        encoding="utf-8",
    )

    return json_path


def main() -> None:
    # 1. 定义路径
    model_path = YOLO26_MODEL_PATH
    image_path = PROJECT_ROOT / "weights" / "bus.jpg"
    output_dir = PROJECT_ROOT / "runs" / "python_first"

    # 2. 校验输入
    validate_inputs(
        model_path=model_path,
        image_path=image_path,
    )

    # 3. 创建输出目录
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 4. 加载模型并执行推理
    print(f"运行设备：{DEFAULT_DEVICE}")
    print(f"置信度阈值：{DEFAULT_CONFIDENCE}")

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(image_path),
        device=DEFAULT_DEVICE,
        conf=DEFAULT_CONFIDENCE,
        save=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )

    # 5. 处理推理结果
    result = results[0]

    height, width = result.orig_shape
    pipeline_ms = sum(result.speed.values())

    print(f"原图宽度：{width}")
    print(f"原图高度：{height}")
    print(f"检测数量：{len(result.boxes)}")
    print(f"各阶段耗时：{result.speed}")
    print(f"处理阶段合计：{pipeline_ms:.1f} ms")

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
        "model": model_path.name,
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