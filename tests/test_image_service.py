import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from vision_studio.image_service import (
    ImageInferenceService,
)
from vision_studio.schemas import (
    Detection,
    InferenceResult,
)


def make_inference_result(
    tmp_path: Path,
) -> InferenceResult:
    annotated_bgr = np.zeros(
        (2, 2, 3),
        dtype=np.uint8,
    )

    # BGR：蓝=10，绿=20，红=30
    annotated_bgr[0, 0] = [10, 20, 30]

    return InferenceResult(
        source_path=tmp_path / "image.jpg",
        model_name="model.pt",
        image_width=640,
        image_height=480,
        detections=[
            Detection(
                class_id=0,
                label="person",
                confidence=0.9,
                bbox_xyxy=(
                    10.0,
                    20.0,
                    100.0,
                    200.0,
                ),
            )
        ],
        preprocess_ms=1.0,
        inference_ms=2.0,
        postprocess_ms=3.0,
        annotated_image=annotated_bgr,
        output_path=None,
    )


def test_service_builds_web_outputs(
    tmp_path: Path,
) -> None:
    detector = MagicMock()
    detector.predict.return_value = (
        make_inference_result(tmp_path)
    )

    cache = MagicMock()
    cache.get.return_value = detector

    service = ImageInferenceService(
        detector_cache=cache,
        output_root=tmp_path / "runs",
    )

    response = service.run(
        mode="YOLO26",
        image_path=tmp_path / "image.jpg",
        confidence=0.30,
        iou=0.60,
        image_size=320,
        save_image=False,
    )

    cache.get.assert_called_once_with("YOLO26")

    detector.predict.assert_called_once_with(
        image_path=tmp_path / "image.jpg",
        output_dir=None,
        confidence=0.30,
        iou=0.60,
        image_size=320,
    )

    # BGR [10, 20, 30] 应变成 RGB [30, 20, 10]
    assert response.annotated_rgb[
        0, 0
    ].tolist() == [30, 20, 10]

    assert response.detection_table == [
        [
            "person",
            0.9,
            10.0,
            20.0,
            100.0,
            200.0,
        ]
    ]

    assert response.json_path.is_file()
    assert response.image_path is None

    output_data = json.loads(
        response.json_path.read_text(
            encoding="utf-8"
        )
    )

    assert output_data["mode"] == "YOLO26"
    assert output_data["prompts"] == []


def test_service_configures_yoloe_prompts(
    tmp_path: Path,
) -> None:
    detector = MagicMock()
    detector.predict.return_value = (
        make_inference_result(tmp_path)
    )

    cache = MagicMock()
    cache.get.return_value = detector

    service = ImageInferenceService(
        detector_cache=cache,
        output_root=tmp_path / "runs",
    )

    service.run(
        mode="YOLOE",
        image_path=tmp_path / "image.jpg",
        prompt_text=(
            "person, bus, Person"
        ),
        save_image=False,
    )

    detector.set_classes.assert_called_once_with(
        ["person", "bus"]
    )


def test_service_rejects_empty_yoloe_prompt(
    tmp_path: Path,
) -> None:
    cache = MagicMock()

    service = ImageInferenceService(
        detector_cache=cache,
        output_root=tmp_path / "runs",
    )

    with pytest.raises(
        ValueError,
        match="提示词不能为空",
    ):
        service.run(
            mode="YOLOE",
            image_path=tmp_path / "image.jpg",
            prompt_text=" , ",
        )

    cache.get.assert_not_called()