import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vision_studio.video import (
    VideoInferenceResult,
    VideoMetadata,
)
from vision_studio.video_service import (
    VideoInferenceService,
)


def make_video_result(
    tmp_path: Path,
) -> VideoInferenceResult:
    return VideoInferenceResult(
        source_path=tmp_path / "input.mp4",
        output_path=tmp_path / "result.mp4",
        metadata=VideoMetadata(
            width=640,
            height=480,
            fps=5.0,
            frame_count=10,
            duration_seconds=2.0,
        ),
        processed_frames=10,
        total_frame_detections=20,
        elapsed_seconds=1.0,
        processing_fps=10.0,
        average_inference_ms=5.0,
        frames=[],
    )


def test_video_service_saves_json(
    tmp_path: Path,
) -> None:
    detector = MagicMock()

    cache = MagicMock()
    cache.get.return_value = detector

    processor = MagicMock()
    processor.process.return_value = (
        make_video_result(tmp_path)
    )

    service = VideoInferenceService(
        detector_cache=cache,
        processor=processor,
        output_root=tmp_path / "runs",
    )

    response = service.run(
        mode="YOLO26",
        video_path=tmp_path / "input.mp4",
        confidence=0.25,
        iou=0.70,
        image_size=640,
    )

    cache.get.assert_called_once_with("YOLO26")
    assert response.json_path.is_file()
    assert "处理帧数：10" in response.summary

    output_data = json.loads(
        response.json_path.read_text(
            encoding="utf-8"
        )
    )

    assert output_data["mode"] == "YOLO26"
    assert output_data["processed_frames"] == 10


def test_video_service_rejects_empty_prompt(
    tmp_path: Path,
) -> None:
    cache = MagicMock()
    processor = MagicMock()

    service = VideoInferenceService(
        detector_cache=cache,
        processor=processor,
        output_root=tmp_path / "runs",
    )

    with pytest.raises(
        ValueError,
        match="提示词不能为空",
    ):
        service.run(
            mode="YOLOE",
            video_path=tmp_path / "input.mp4",
            prompt_text=" , ",
        )

    cache.get.assert_not_called()
    processor.process.assert_not_called()