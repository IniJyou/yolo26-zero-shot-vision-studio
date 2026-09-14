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
from vision_studio.schemas import DetectorConfig


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

    # 3. 创建检测器并推理
    detector = YOLO26Detector(config)

    inference_result = detector.predict(
        image_path=image_path,
        output_dir=output_dir,
    )

    # 4. 打印推理概览
    print(f"运行设备：{config.device}")
    print(f"置信度阈值：{config.confidence}")
    print(
        "原图尺寸："
        f"{inference_result.image_width}"
        f"x{inference_result.image_height}"
    )
    print(
        f"检测数量："
        f"{inference_result.detection_count}"
    )
    print(
        f"处理阶段合计："
        f"{inference_result.total_ms:.1f} ms"
    )

    for detection in inference_result.detections:
        print(
            f"类别={detection.label}, "
            f"置信度={detection.confidence:.3f}, "
            f"坐标={list(detection.bbox_xyxy)}"
        )

    # 5. 导出数据并保存 JSON
    output_data = inference_result.to_dict()

    json_path = save_json(
        output_data=output_data,
        json_path=output_dir / "bus.json",
    )

    print(f"标注图片：{inference_result.output_path}")
    print(f"JSON结果已保存：{json_path}")

if __name__ == "__main__":
    main()