from pathlib import Path
from typing import Any, cast

import gradio as gr
import numpy as np
import torch
from numpy.typing import NDArray

from .image_service import ImageInferenceService
from .model_cache import ModelMode

from .video_service import VideoInferenceService

# 检查 Gradio 输入
#     ↓
# 调用 ImageInferenceService
#     ↓
# 把 Path 转成 Gradio 接受的字符串
#     ↓
# 把异常转换成中文 gr.Error


WebImageResult = tuple[
    NDArray[np.uint8],
    list[list[Any]],
    str,
    str,
    str | None,
]

WebVideoResult = tuple[
    str,
    str,
    str,
    str,
]


class ImageWebController:
    """连接Gradio组件与图片推理服务。"""

    def __init__(
        self,
        service: ImageInferenceService | None = None,
    ) -> None:
        self._service = (
            service
            if service is not None
            else ImageInferenceService()
        )

    def predict(
        self,
        image_path: str | None,
        mode: str,
        prompt_text: str,
        confidence: float,
        iou: float,
        image_size: int,
    ) -> WebImageResult:
        if not image_path:
            raise gr.Error("请先上传一张图片")

        if mode not in ("YOLO26", "YOLOE"):
            raise gr.Error(
                f"不支持的模型模式：{mode}"
            )

        model_mode = cast(ModelMode, mode)

        try:
            response = self._service.run(
                mode=model_mode,
                image_path=Path(image_path),
                prompt_text=prompt_text,
                confidence=confidence,
                iou=iou,
                image_size=image_size,
                save_image=True,
            )

        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()

            raise gr.Error(
                "GPU显存不足。请降低输入尺寸，"
                "或关闭其他占用显存的程序。"
            ) from error

        except FileNotFoundError as error:
            raise gr.Error(str(error)) from error

        except ValueError as error:
            raise gr.Error(str(error)) from error

        except Exception as error:
            raise gr.Error(
                f"检测失败：{error}"
            ) from error

        return (
            response.annotated_rgb,
            response.detection_table,
            response.summary,
            str(response.json_path),
            (
                str(response.image_path)
                if response.image_path is not None
                else None
            ),
        )



class VideoWebController:
    """连接Gradio组件与视频推理服务。"""

    def __init__(
        self,
        service: VideoInferenceService | None = None,
    ) -> None:
        self._service = (
            service
            if service is not None
            else VideoInferenceService()
        )

    def predict(
        self,
        video_path: str | None,
        mode: str,
        prompt_text: str,
        confidence: float,
        iou: float,
        image_size: int,
        progress: gr.Progress = gr.Progress(),
    ) -> WebVideoResult:
        if not video_path:
            raise gr.Error("请先上传一个视频")

        if mode not in ("YOLO26", "YOLOE"):
            raise gr.Error(
                f"不支持的模型模式：{mode}"
            )

        model_mode = cast(ModelMode, mode)

        def update_progress(
            processed_frames: int,
            total_frames: int,
        ) -> None:
            if total_frames > 0:
                progress_value = min(
                    processed_frames / total_frames,
                    1.0,
                )
                description = (
                    "正在处理视频："
                    f"{processed_frames}/{total_frames} 帧"
                )
            else:
                progress_value = 0.0
                description = (
                    "正在处理视频："
                    f"{processed_frames} 帧"
                )

            progress(
                progress_value,
                desc=description,
            )

        try:
            response = self._service.run(
                mode=model_mode,
                video_path=Path(video_path),
                prompt_text=prompt_text,
                confidence=confidence,
                iou=iou,
                image_size=image_size,
                progress_callback=update_progress,
            )

        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()

            raise gr.Error(
                "GPU显存不足。请降低输入尺寸，"
                "缩短视频，或关闭其他占用显存的程序。"
            ) from error

        except FileNotFoundError as error:
            raise gr.Error(str(error)) from error

        except ValueError as error:
            raise gr.Error(str(error)) from error

        except Exception as error:
            raise gr.Error(
                f"视频检测失败：{error}"
            ) from error

        progress(
            1.0,
            desc="视频处理完成",
        )

        result_video_path = str(
            response.video_path
        )

        return (
            result_video_path,
            response.summary,
            str(response.json_path),
            result_video_path,
        )

