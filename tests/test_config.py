from pathlib import Path

import pytest

from vision_studio.schemas import DetectorConfig


def test_detector_config_accepts_valid_values() -> None:
    config = DetectorConfig(
        model_path=Path("model.pt"),
        device="cpu",
        confidence=0.25,
        iou=0.70,
        image_size=640,
    )

    assert config.confidence == 0.25
    assert config.iou == 0.70
    assert config.image_size == 640


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("iou", -0.1),
        ("iou", 1.1),
        ("image_size", 0),
        ("image_size", -640),
    ],
)
def test_detector_config_rejects_invalid_values(
    field_name: str,
    invalid_value: float | int,
) -> None:
    values = {
        "model_path": Path("model.pt"),
        "device": "cpu",
        "confidence": 0.25,
        "iou": 0.70,
        "image_size": 640,
    }

    values[field_name] = invalid_value

    with pytest.raises(ValueError):
        DetectorConfig(**values)