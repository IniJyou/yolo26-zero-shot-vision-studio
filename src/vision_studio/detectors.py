from pathlib import Path

from ultralytics import YOLO, YOLOE
from ultralytics.engine.model import Model

from .io_utils import validate_inputs
from .prompt_utils import normalize_classes
from .schemas import (
    DetectorConfig,
    InferenceResult,
    build_inference_result,
)


class BaseDetector:
    """YOLO26和YOLOE共用的图片推理流程。"""

    def __init__(
        self,
        config: DetectorConfig,
    ) -> None:
        self.config = config
        self.model: Model

        if not config.model_path.is_file():
            raise FileNotFoundError(
                f"模型权重不存在：{config.model_path}"
            )

    def predict(
        self,
        image_path: Path,
        output_dir: Path | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        image_size: int | None = None,
    ) -> InferenceResult:
        validate_inputs(
            model_path=self.config.model_path,
            image_path=image_path,
        )
      # 这里重新创建 runtime_config 的一个作用是复用已有参数校验。
        runtime_config = DetectorConfig(
            model_path=self.config.model_path,
            device=self.config.device,
            confidence=(
                self.config.confidence
                if confidence is None
                else confidence
            ),
            iou=(
                self.config.iou
                if iou is None
                else iou
            ),
            image_size=(
                self.config.image_size
                if image_size is None
                else image_size
            ),
        )

        raw_results = self.model.predict(
            source=str(image_path),
            device=runtime_config.device,
            conf=runtime_config.confidence,
            iou=runtime_config.iou,
            imgsz=runtime_config.image_size,
            save=False,
        )

        if not raw_results:
            raise RuntimeError(
                "模型没有返回图片推理结果"
            )

        raw_result = raw_results[0]
        output_path: Path | None = None

        if output_dir is not None:
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                output_dir / image_path.name
            )

            raw_result.save(
                filename=str(output_path)
            )

        return build_inference_result(
            result=raw_result,
            source_path=image_path,
            model_name=self.config.model_path.name,
            output_path=output_path,
        )


class YOLO26Detector(BaseDetector):
    """COCO固定类别的YOLO26检测器。"""

    def __init__(
        self,
        config: DetectorConfig,
    ) -> None:
        super().__init__(config)
        self.model = YOLO(str(config.model_path))


class YOLOEDetector(BaseDetector):
    """通过英文文本类别进行零样本检测的YOLOE检测器。"""

    def __init__(
        self,
        config: DetectorConfig,
    ) -> None:
        super().__init__(config)
        self.model = YOLOE(str(config.model_path))
        self.classes: list[str] = []

    def set_classes(
        self,
        classes: list[str],
    ) -> None:
        normalized = normalize_classes(classes)

        self.model.set_classes(normalized)
        self.classes = normalized

    def predict(
        self,
        image_path: Path,
        output_dir: Path | None = None,
        confidence: float | None = None,
        iou: float | None = None,
        image_size: int | None = None,
    ) -> InferenceResult:
        if not self.classes:
            raise ValueError(
                "运行YOLOE前必须先设置提示词"
            )

        return super().predict(
            image_path=image_path,
            output_dir=output_dir,
            confidence=confidence,
            iou=iou,
            image_size=image_size,
        )