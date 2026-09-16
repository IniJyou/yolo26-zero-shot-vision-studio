from pathlib import Path
from unittest.mock import MagicMock

import gradio as gr
import pytest

from vision_studio.web_controller import (
    VideoWebController,
)


def make_video_response(
    tmp_path: Path,
) -> MagicMock:
    response = MagicMock()
    response.video_path = (
        tmp_path / "result.mp4"
    )
    response.json_path = (
        tmp_path / "result.json"
    )
    response.summary = (
        "模型：YOLO26\n处理帧数：10"
    )
    return response


def test_video_controller_returns_outputs(
    tmp_path: Path,
) -> None:
    service = MagicMock()
    service.run.return_value = (
        make_video_response(tmp_path)
    )

    progress = MagicMock()
    controller = VideoWebController(service)

    result = controller.predict(
        video_path=str(tmp_path / "input.mp4"),
        mode="YOLO26",
        prompt_text="",
        confidence=0.25,
        iou=0.70,
        image_size=640,
        progress=progress,
    )

    call_arguments = (
        service.run.call_args.kwargs
    )

    assert call_arguments["mode"] == "YOLO26"
    assert call_arguments["video_path"] == (
        tmp_path / "input.mp4"
    )
    assert callable(
        call_arguments["progress_callback"]
    )

    progress_callback = call_arguments[
        "progress_callback"
    ]
    progress_callback(5, 10)

    assert result[0].endswith("result.mp4")
    assert "处理帧数：10" in result[1]
    assert result[2].endswith("result.json")
    assert result[3].endswith("result.mp4")


def test_video_controller_rejects_missing_video(
) -> None:
    controller = VideoWebController(
        service=MagicMock()
    )

    with pytest.raises(gr.Error):
        controller.predict(
            video_path=None,
            mode="YOLO26",
            prompt_text="",
            confidence=0.25,
            iou=0.70,
            image_size=640,
            progress=MagicMock(),
        )


def test_video_controller_converts_value_error(
) -> None:
    service = MagicMock()
    service.run.side_effect = ValueError(
        "视频时长不能超过30秒"
    )

    controller = VideoWebController(service)

    with pytest.raises(gr.Error):
        controller.predict(
            video_path="input.mp4",
            mode="YOLO26",
            prompt_text="",
            confidence=0.25,
            iou=0.70,
            image_size=640,
            progress=MagicMock(),
        )