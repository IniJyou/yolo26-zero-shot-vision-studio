from pathlib import Path

from ultralytics import YOLO, YOLOE
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

from .io_utils import validate_inputs
from .prompt_utils import normalize_classes
from .schemas import DetectorConfig


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
        output_dir: Path,
    ) -> Results:
        validate_inputs(
            model_path=self.config.model_path,
            image_path=image_path,
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = self.model.predict(
            source=str(image_path),
            device=self.config.device,
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.image_size,
            save=True,
            project=str(output_dir.parent),
            name=output_dir.name,
            exist_ok=True,
        )

        return results[0]


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
        output_dir: Path,
    ) -> Results:
        if not self.classes:
            raise ValueError(
                "运行YOLOE前必须先设置提示词"
            )

        return super().predict(
            image_path=image_path,
            output_dir=output_dir,
        )