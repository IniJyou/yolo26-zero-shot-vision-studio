from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
from ultralytics.engine.results import Results

from vision_studio.schemas import (
    Detection,
    build_inference_result,
    parse_detections,
)


def make_fake_result() -> Results:
    fake_box = SimpleNamespace(
        cls=np.array([0]),
        conf=np.array([0.87654]),
        xyxy=np.array(
            [[10.04, 20.06, 100.08, 200.02]]
        ),
    )

    fake_result = SimpleNamespace(
        boxes=[fake_box],
        names={0: "person"},
        orig_shape=(480, 640),
        speed={
            "preprocess": 2.345,
            "inference": 10.678,
            "postprocess": 1.234,
        },
        plot=lambda: np.zeros(
            (480, 640, 3),
            dtype=np.uint8,
        ),
    )

    return cast(Results, fake_result)


def test_parse_detections_returns_detection() -> None:
    result = make_fake_result()

    detections = parse_detections(result)

    assert detections == [
        Detection(
            class_id=0,
            label="person",
            confidence=0.8765,
            bbox_xyxy=(
                10.0,
                20.1,
                100.1,
                200.0,
            ),
        )
    ]


def test_build_inference_result() -> None:
    raw_result = make_fake_result()

    result = build_inference_result(
        result=raw_result,
        source_path=Path("image.jpg"),
        model_name="model.pt",
    )

    assert result.image_width == 640
    assert result.image_height == 480
    assert result.detection_count == 1
    assert result.inference_ms == 10.68
    assert result.total_ms == 14.26
    assert result.output_path is None
    assert result.annotated_image.shape == (
        480,
        640,
        3,
    )


def test_inference_result_to_dict() -> None:
    raw_result = make_fake_result()

    result = build_inference_result(
        result=raw_result,
        source_path=Path("image.jpg"),
        model_name="model.pt",
    )

    output_data = result.to_dict()

    assert output_data["model"] == "model.pt"
    assert output_data["detection_count"] == 1
    assert output_data["detections"][0]["label"] == (
        "person"
    )
    assert "annotated_image" not in output_data