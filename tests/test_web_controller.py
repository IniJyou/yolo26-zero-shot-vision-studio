from pathlib import Path
from unittest.mock import MagicMock

import gradio as gr
import numpy as np
import pytest

from vision_studio.web_controller import (
    ImageWebController,
)


def make_service_response(
    tmp_path: Path,
) -> MagicMock:
    response = MagicMock()

    response.annotated_rgb = np.zeros(
        (10, 10, 3),
        dtype=np.uint8,
    )

    response.detection_table = [
        [
            "person",
            0.9,
            10.0,
            20.0,
            100.0,
            200.0,
        ]
    ]

    response.summary = (
        "模型：YOLO26\n目标数量：1"
    )
    response.json_path = (
        tmp_path / "result.json"
    )
    response.image_path = (
        tmp_path / "result.jpg"
    )

    return response


def test_controller_returns_gradio_outputs(
    tmp_path: Path,
) -> None:
    service = MagicMock()
    service.run.return_value = (
        make_service_response(tmp_path)
    )

    controller = ImageWebController(service)

    result = controller.predict(
        image_path=str(tmp_path / "input.jpg"),
        mode="YOLO26",
        prompt_text="",
        confidence=0.25,
        iou=0.70,
        image_size=640,
    )

    service.run.assert_called_once_with(
        mode="YOLO26",
        image_path=tmp_path / "input.jpg",
        prompt_text="",
        confidence=0.25,
        iou=0.70,
        image_size=640,
        save_image=True,
    )

    assert result[0].shape == (10, 10, 3)
    assert result[1][0][0] == "person"
    assert result[2].startswith("模型：YOLO26")
    assert result[3].endswith("result.json")
    assert result[4] is not None
    assert result[4].endswith("result.jpg")


def test_controller_rejects_missing_image() -> None:
    controller = ImageWebController(
        service=MagicMock()
    )

    with pytest.raises(gr.Error):
        controller.predict(
            image_path=None,
            mode="YOLO26",
            prompt_text="",
            confidence=0.25,
            iou=0.70,
            image_size=640,
        )


def test_controller_converts_value_error() -> None:
    service = MagicMock()
    service.run.side_effect = ValueError(
        "YOLOE提示词不能为空"
    )

    controller = ImageWebController(service)

    with pytest.raises(gr.Error):
        controller.predict(
            image_path="image.jpg",
            mode="YOLOE",
            prompt_text="",
            confidence=0.25,
            iou=0.70,
            image_size=640,
        )