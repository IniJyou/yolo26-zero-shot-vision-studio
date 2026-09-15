from pathlib import Path

import torch

DEFAULT_IOU = 0.70
DEFAULT_IMAGE_SIZE = 640

# 这里不写死 D:\aiworkspace\yolo。通过当前文件位置推导项目根目录，仓库被其他人克隆后仍能运行。

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WEIGHTS_DIR = PROJECT_ROOT / "weights"
DATASETS_DIR = PROJECT_ROOT / "datasets"
RUNS_DIR = PROJECT_ROOT / "runs"

YOLO26_MODEL_PATH = WEIGHTS_DIR / "yolo26n.pt"
YOLOE_MODEL_PATH = WEIGHTS_DIR / "yoloe-26n-seg.pt"

DEFAULT_CONFIDENCE = 0.25
DEFAULT_DEVICE = 0 if torch.cuda.is_available() else "cpu"

SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

SUPPORTED_VIDEO_SUFFIXES = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}
# 暂时限制 30 秒，是为了避免一次任务占用过多磁盘和处理时间
DEFAULT_MAX_VIDEO_SECONDS = 30.0
