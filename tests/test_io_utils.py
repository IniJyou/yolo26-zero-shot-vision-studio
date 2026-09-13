from pathlib import Path

import pytest

from vision_studio.io_utils import validate_inputs


def test_validate_inputs_accepts_existing_jpg(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    image_path = tmp_path / "image.jpg"

    model_path.write_bytes(b"fake model")
    image_path.write_bytes(b"fake image")

    validate_inputs(
        model_path=model_path,
        image_path=image_path,
    )


def test_validate_inputs_rejects_missing_image(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    image_path = tmp_path / "missing.jpg"

    model_path.write_bytes(b"fake model")

    with pytest.raises(
        FileNotFoundError,
        match="输入图片不存在",
    ):
        validate_inputs(
            model_path=model_path,
            image_path=image_path,
        )


def test_validate_inputs_rejects_unsupported_suffix(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pt"
    image_path = tmp_path / "image.txt"

    model_path.write_bytes(b"fake model")
    image_path.write_bytes(b"fake image")

    with pytest.raises(
        ValueError,
        match="不支持的图片格式",
    ):
        validate_inputs(
            model_path=model_path,
            image_path=image_path,
        )