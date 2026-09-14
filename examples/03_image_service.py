from vision_studio.config import PROJECT_ROOT
from vision_studio.image_service import (
    ImageInferenceService,
)


def main() -> None:
    service = ImageInferenceService()

    response = service.run(
        mode="YOLOE",
        image_path=(
            PROJECT_ROOT
            / "weights"
            / "bus.jpg"
        ),
        prompt_text=(
            "double-decker bus, person"
        ),
        confidence=0.20,
        iou=0.70,
        image_size=640,
        save_image=True,
    )

    print(response.summary)
    print(f"检测表格：{response.detection_table}")
    print(f"标注图片：{response.image_path}")
    print(f"JSON文件：{response.json_path}")


if __name__ == "__main__":
    main()