import json
from pathlib import Path
from typing import Any

from .config import SUPPORTED_IMAGE_SUFFIXES


def validate_inputs(
    model_path: Path,
    image_path: Path,
) -> None:
    """检查模型权重和输入图片是否合法。"""
    if not model_path.is_file():
        raise FileNotFoundError(f"模型权重不存在：{model_path}")

    if not image_path.is_file():
        raise FileNotFoundError(f"输入图片不存在：{image_path}")

    if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"不支持的图片格式：{image_path.suffix}")


def save_json(
    output_data: dict[str, Any],
    json_path: Path,
) -> Path:
    """将检测结果保存为UTF-8 JSON文件。"""
    json_path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(
        output_data,
        ensure_ascii=False,
        indent=2,
    )

    json_path.write_text(
        json_text,
        encoding="utf-8",
    )

    return json_path