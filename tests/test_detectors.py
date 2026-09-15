from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vision_studio.detectors import (
    YOLO26Detector,
    YOLOEDetector,
)
from vision_studio.schemas import DetectorConfig


def make_config(
    tmp_path: Path,
) -> DetectorConfig:
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"fake model")

    return DetectorConfig(
        model_path=model_path,
        device="cpu",
        confidence=0.30,
        iou=0.60,
        image_size=320,
    )


def make_image(tmp_path: Path) -> Path:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake image")
    return image_path


def make_raw_result() -> MagicMock:
    result = MagicMock()
    result.boxes = None
    result.names = {}
    result.orig_shape = (480, 640)
    result.speed = {
        "preprocess": 1.0,
        "inference": 2.0,
        "postprocess": 3.0,
    }
    result.plot.return_value = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    return result


def test_yolo26_predict_without_saving(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    image_path = make_image(tmp_path)
    raw_result = make_raw_result()

    with patch(
        "vision_studio.detectors.YOLO"
    ) as yolo_factory:
        model = yolo_factory.return_value
        model.predict.return_value = [raw_result]

        detector = YOLO26Detector(config)
        result = detector.predict(image_path)

    model.predict.assert_called_once_with(
        source=str(image_path),
        device="cpu",
        conf=0.30,
        iou=0.60,
        imgsz=320,
        save=False,
        verbose=False,
    )

    raw_result.save.assert_not_called()
    assert result.output_path is None
    assert result.detection_count == 0


def test_yolo26_predict_with_output_dir(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    image_path = make_image(tmp_path)
    output_dir = tmp_path / "results"
    raw_result = make_raw_result()

    with patch(
        "vision_studio.detectors.YOLO"
    ) as yolo_factory:
        model = yolo_factory.return_value
        model.predict.return_value = [raw_result]

        detector = YOLO26Detector(config)
        result = detector.predict(
            image_path=image_path,
            output_dir=output_dir,
        )

    expected_path = output_dir / "image.jpg"

    raw_result.save.assert_called_once_with(
        filename=str(expected_path)
    )
    assert result.output_path == expected_path


def test_yoloe_requires_classes(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    image_path = make_image(tmp_path)

    with patch(
        "vision_studio.detectors.YOLOE"
    ):
        detector = YOLOEDetector(config)

        with pytest.raises(
            ValueError,
            match="必须先设置提示词",
        ):
            detector.predict(image_path)


def test_yoloe_normalizes_classes(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    with patch(
        "vision_studio.detectors.YOLOE"
    ) as yoloe_factory:
        model = yoloe_factory.return_value
        detector = YOLOEDetector(config)

        detector.set_classes(
            [" person ", "Person", "bus"]
        )

    model.set_classes.assert_called_once_with(
        ["person", "bus"]
    )
    assert detector.classes == [
        "person",
        "bus",
    ]

def test_yolo26_uses_runtime_parameters(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    image_path = make_image(tmp_path)
    raw_result = make_raw_result()

    with patch(
        "vision_studio.detectors.YOLO"
    ) as yolo_factory:
        model = yolo_factory.return_value
        model.predict.return_value = [raw_result]

        detector = YOLO26Detector(config)

        detector.predict(
            image_path=image_path,
            confidence=0.15,
            iou=0.50,
            image_size=640,
        )

    model.predict.assert_called_once_with(
        source=str(image_path),
        device="cpu",
        conf=0.15,
        iou=0.50,
        imgsz=640,
        save=False,
        verbose=False,
    )


def test_predict_rejects_invalid_runtime_confidence(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    image_path = make_image(tmp_path)

    with patch(
        "vision_studio.detectors.YOLO"
    ):
        detector = YOLO26Detector(config)

        with pytest.raises(
            ValueError,
            match="confidence必须位于0到1之间",
        ):
            detector.predict(
                image_path=image_path,
                confidence=1.5,
            )


def test_yolo26_predicts_numpy_frame(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    raw_result = make_raw_result()

    frame = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    with patch(
        "vision_studio.detectors.YOLO"
    ) as yolo_factory:
        model = yolo_factory.return_value
        model.predict.return_value = [raw_result]

        detector = YOLO26Detector(config)

        result = detector.predict_frame(
            frame_bgr=frame,
            source_path=tmp_path / "video.mp4",
            confidence=0.20,
            iou=0.60,
            image_size=320,
        )

    call_arguments = (
        model.predict.call_args.kwargs
    )

    assert call_arguments["source"] is frame
    assert call_arguments["conf"] == 0.20
    assert call_arguments["iou"] == 0.60
    assert call_arguments["imgsz"] == 320
    assert call_arguments["save"] is False
    assert call_arguments["verbose"] is False

    assert result.image_width == 640
    assert result.image_height == 480


def test_predict_frame_rejects_empty_array(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    with patch(
        "vision_studio.detectors.YOLO"
    ):
        detector = YOLO26Detector(config)

        with pytest.raises(
            ValueError,
            match="视频帧必须是非空",
        ):
            detector.predict_frame(
                frame_bgr=np.array([]),
                source_path=tmp_path / "video.mp4",
            )