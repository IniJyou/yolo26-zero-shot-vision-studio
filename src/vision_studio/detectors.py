from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.results import Results

from .io_utils import validate_inputs
from .schemas import DetectorConfig
# 模型保存在对象中，因此一个检测器可以连续处理多张图片，不需要每次重新加载权重。

class YOLO26Detector:
    """封装YOLO26模型加载和图片推理。"""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

        if not config.model_path.is_file():
            raise FileNotFoundError(
                f"模型权重不存在：{config.model_path}"
            )

        self.model = YOLO(str(config.model_path))

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