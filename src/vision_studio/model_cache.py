from threading import Lock
from typing import Literal, cast

from .config import (
    DEFAULT_DEVICE,
    YOLO26_MODEL_PATH,
    YOLOE_MODEL_PATH,
)
from .detectors import (
    BaseDetector,
    YOLO26Detector,
    YOLOEDetector,
)
from .schemas import DetectorConfig


ModelMode = Literal["YOLO26", "YOLOE"]


class DetectorCache:
    """按模型模式缓存检测器和已经加载的权重。"""

    def __init__(self) -> None:
        self._detectors: dict[
            ModelMode,
            BaseDetector,
        ] = {}

        self._lock = Lock()

    def get(
        self,
        mode: str,
    ) -> BaseDetector:
        if mode not in ("YOLO26", "YOLOE"):
            raise ValueError(
                f"不支持的模型模式：{mode}"
            )

        model_mode = cast(ModelMode, mode)

        # 防止两个请求同时创建同一个模型。
        with self._lock:
            detector = self._detectors.get(
                model_mode
            )

            if detector is None:
                detector = self._create(
                    model_mode
                )
                self._detectors[model_mode] = (
                    detector
                )

        return detector

    def _create(
        self,
        mode: ModelMode,
    ) -> BaseDetector:
        if mode == "YOLO26":
            config = DetectorConfig(
                model_path=YOLO26_MODEL_PATH,
                device=DEFAULT_DEVICE,
            )
            return YOLO26Detector(config)

        config = DetectorConfig(
            model_path=YOLOE_MODEL_PATH,
            device=DEFAULT_DEVICE,
        )
        return YOLOEDetector(config)

    @property
    def size(self) -> int:
        return len(self._detectors)