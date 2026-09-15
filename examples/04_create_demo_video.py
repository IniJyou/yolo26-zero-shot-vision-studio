import cv2

from vision_studio.config import (
    DATASETS_DIR,
    WEIGHTS_DIR,
)


def main() -> None:
    image_path = WEIGHTS_DIR / "bus.jpg"
    output_path = DATASETS_DIR / "demo.mp4"

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"无法读取图片：{image_path}"
        )

    frame = cv2.resize(
        image,
        (640, 480),
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        5.0,
        (640, 480),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "无法创建测试视频"
        )

    try:
        for frame_index in range(10):
            current_frame = frame.copy()

            cv2.putText(
                current_frame,
                f"Frame {frame_index}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            writer.write(current_frame)

    finally:
        writer.release()

    print(f"测试视频已生成：{output_path}")


if __name__ == "__main__":
    main()