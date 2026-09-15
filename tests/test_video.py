from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from vision_studio.schemas import (
    InferenceResult,
)
from vision_studio.video import (
    VideoProcessor,
    read_video_metadata,
)


def make_capture(
    width: int = 640,
    height: int = 480,
    fps: float = 5.0,
    frame_count: int = 2,
) -> MagicMock:
    capture = MagicMock()
    capture.isOpened.return_value = True

    properties = {
        cv2.CAP_PROP_FRAME_WIDTH: width,
        cv2.CAP_PROP_FRAME_HEIGHT: height,
        cv2.CAP_PROP_FPS: fps,
        cv2.CAP_PROP_FRAME_COUNT: (
            frame_count
        ),
    }

    capture.get.side_effect = (
        lambda property_id:
        properties[property_id]
    )

    return capture


def make_frame_result(
    video_path: Path,
) -> InferenceResult:
    return InferenceResult(
        source_path=video_path,
        model_name="model.pt",
        image_width=640,
        image_height=480,
        detections=[],
        preprocess_ms=1.0,
        inference_ms=2.0,
        postprocess_ms=1.0,
        annotated_image=np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        ),
    )


def test_read_video_metadata(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"fake video")

    capture = make_capture()

    with patch(
        "vision_studio.video.cv2.VideoCapture",
        return_value=capture,
    ):
        metadata = read_video_metadata(
            video_path
        )

    assert metadata.width == 640
    assert metadata.height == 480
    assert metadata.fps == 5.0
    assert metadata.frame_count == 2
    assert metadata.duration_seconds == 0.4
    capture.release.assert_called_once()


def test_video_processor_writes_all_frames(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "input.mp4"
    output_path = tmp_path / "result.mp4"
    video_path.write_bytes(b"fake video")

    metadata_capture = make_capture()
    frame_capture = make_capture()

    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    frame_capture.read.side_effect = [
        (True, frame),
        (True, frame),
        (False, None),
    ]

    detector = MagicMock()
    detector.predict_frame.return_value = (
        make_frame_result(video_path)
    )

    writer = MagicMock()

    with (
        patch(
            "vision_studio.video.cv2.VideoCapture",
            side_effect=[
                metadata_capture,
                frame_capture,
            ],
        ),
        patch(
            "vision_studio.video."
            "imageio_ffmpeg.write_frames",
            return_value=writer,
        ),
    ):
        processor = VideoProcessor(
            max_duration_seconds=10.0
        )

        result = processor.process(
            detector=detector,
            video_path=video_path,
            output_path=output_path,
            confidence=0.25,
            iou=0.70,
            image_size=640,
        )

    assert result.processed_frames == 2
    assert result.metadata.fps == 5.0
    assert result.average_inference_ms == 2.0

    assert detector.predict_frame.call_count == 2

    # 一次send(None)初始化，加两次视频帧。
    assert writer.send.call_count == 3
    writer.close.assert_called_once()
    frame_capture.release.assert_called_once()


def test_video_rejects_long_duration(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "long.mp4"
    output_path = tmp_path / "result.mp4"
    video_path.write_bytes(b"fake video")

    capture = make_capture(
        fps=10.0,
        frame_count=400,
    )

    with patch(
        "vision_studio.video.cv2.VideoCapture",
        return_value=capture,
    ):
        processor = VideoProcessor(
            max_duration_seconds=30.0
        )

        with pytest.raises(
            ValueError,
            match="视频时长不能超过30秒",
        ):
            processor.process(
                detector=MagicMock(),
                video_path=video_path,
                output_path=output_path,
                confidence=0.25,
                iou=0.70,
                image_size=640,
            )