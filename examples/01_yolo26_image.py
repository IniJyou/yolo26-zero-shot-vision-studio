from ultralytics import YOLO
from vision_studio.io_utils import save_json, validate_inputs
from vision_studio.schemas import parse_detections

# 找到项目根目录
from vision_studio.config import (
    DEFAULT_CONFIDENCE,
    DEFAULT_DEVICE,
    PROJECT_ROOT,
    SUPPORTED_IMAGE_SUFFIXES,
    YOLO26_MODEL_PATH,
)



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