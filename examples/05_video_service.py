from vision_studio.config import (
    DATASETS_DIR,
)
from vision_studio.video_service import (
    VideoInferenceService,
)


def show_progress(
    current: int,
    total: int,
) -> None:
    print(
        f"\r处理进度：{current}/{total}",
        end="",
    )


def main() -> None:
    service = VideoInferenceService()

    response = service.run(
        mode="YOLO26",
        video_path=(
            DATASETS_DIR / "demo.mp4"
        ),
        confidence=0.25,
        iou=0.70,
        image_size=640,
        progress_callback=show_progress,
    )

    print()
    print(response.summary)
    print(f"结果视频：{response.video_path}")
    print(f"JSON文件：{response.json_path}")


if __name__ == "__main__":
    main()