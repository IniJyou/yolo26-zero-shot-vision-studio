from unittest.mock import MagicMock, patch

import pytest

from vision_studio.model_cache import DetectorCache


def test_cache_reuses_yolo26_detector() -> None:
    fake_detector = MagicMock()

    with patch(
        "vision_studio.model_cache.YOLO26Detector",
        return_value=fake_detector,
    ) as detector_factory:
        cache = DetectorCache()

        first = cache.get("YOLO26")
        second = cache.get("YOLO26")

    assert first is fake_detector
    assert second is fake_detector
    assert first is second
    assert cache.size == 1
    detector_factory.assert_called_once()


def test_cache_stores_two_model_modes() -> None:
    with (
        patch(
            "vision_studio.model_cache.YOLO26Detector"
        ) as yolo26_factory,
        patch(
            "vision_studio.model_cache.YOLOEDetector"
        ) as yoloe_factory,
    ):
        cache = DetectorCache()

        yolo26 = cache.get("YOLO26")
        yoloe = cache.get("YOLOE")

    assert yolo26 is yolo26_factory.return_value
    assert yoloe is yoloe_factory.return_value
    assert cache.size == 2


def test_cache_rejects_unknown_mode() -> None:
    cache = DetectorCache()

    with pytest.raises(
        ValueError,
        match="不支持的模型模式",
    ):
        cache.get("unknown")