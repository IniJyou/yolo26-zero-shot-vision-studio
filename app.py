import gradio as gr

from vision_studio.config import (
    DATASETS_DIR,
    RUNS_DIR,
    WEIGHTS_DIR,
)
from vision_studio.web_controller import (
    ImageWebController,
)



# 说明：
# - server_name="127.0.0.1"：只能本机访问。
# - share=False：不产生互联网公开地址。
# - max_file_size="20mb"：限制单张上传大小。Gradio 官方建议对上传文件设置大小限制。Gradio 文件访问说明
# - allowed_paths：只额外允许页面读取结果目录。
# - blocked_paths：禁止通过 Gradio 访问权重和数据集目录。
# - concurrency_limit=1：RTX 4060 同一时间只处理一个推理任务。
# - delete_cache=(86400, 86400)：每天检查并清理超过一天的 Gradio 临时缓存。
# - 队列最多等待 8 个任务。Gradio 的 Blocks.queue() 支持队列大小和并发限制。Gradio Blocks 文档

controller = ImageWebController()


def update_prompt_visibility(
    mode: str,
) -> gr.Textbox:
    """只有YOLOE模式需要显示提示词输入。"""
    return gr.Textbox(
        visible=(mode == "YOLOE")
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="YOLO26 Zero-Shot Vision Studio",
        theme=gr.themes.Soft(),
        delete_cache=(86400, 86400),
    ) as demo:
        gr.Markdown(
            """
# YOLO26 Zero-Shot Vision Studio

对比 YOLO26 固定类别检测与 YOLOE-26
开放词汇零样本检测。

- **YOLO26**：检测 COCO 固定类别
- **YOLOE**：使用英文提示词指定目标类别
"""
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="上传图片",
                    type="filepath",
                    image_mode="RGB",
                    sources=["upload"],
                    height=500,
                )

                model_mode = gr.Radio(
                    choices=["YOLO26", "YOLOE"],
                    value="YOLO26",
                    label="模型模式",
                )

                prompt_text = gr.Textbox(
                    label="YOLOE 英文提示词",
                    placeholder=(
                        "例如：person, bus, "
                        "cartoon sheep"
                    ),
                    value=(
                        "double-decker bus, person"
                    ),
                    visible=False,
                    lines=2,
                )

                with gr.Accordion(
                    "推理参数",
                    open=False,
                ):
                    confidence = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.25,
                        step=0.05,
                        label="置信度阈值",
                    )

                    iou = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.70,
                        step=0.05,
                        label="IoU 阈值",
                    )

                    image_size = gr.Dropdown(
                        choices=[320, 640, 960],
                        value=640,
                        label="输入尺寸",
                    )

                detect_button = gr.Button(
                    "开始检测",
                    variant="primary",
                )

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="标注结果",
                    type="numpy",
                    format="png",
                    interactive=False,
                    height=500,
                )

                summary = gr.Textbox(
                    label="推理摘要",
                    lines=4,
                    interactive=False,
                )

        detection_table = gr.Dataframe(
            headers=[
                "类别",
                "置信度",
                "x1",
                "y1",
                "x2",
                "y2",
            ],
            datatype=[
                "str",
                "number",
                "number",
                "number",
                "number",
                "number",
            ],
            value=[],
            interactive=False,
            label="检测结果",
        )

        with gr.Row():
            json_download = gr.File(
                label="下载 JSON 结果"
            )

            image_download = gr.File(
                label="下载标注图片"
            )

        model_mode.change(
            fn=update_prompt_visibility,
            inputs=model_mode,
            outputs=prompt_text,
            queue=False,
            api_visibility="private",
        )

        detect_button.click(
            fn=controller.predict,
            inputs=[
                input_image,
                model_mode,
                prompt_text,
                confidence,
                iou,
                image_size,
            ],
            outputs=[
                output_image,
                detection_table,
                summary,
                json_download,
                image_download,
            ],
            api_name="predict_image",
            show_progress="full",
            concurrency_limit=1,
        )

    return demo


demo = build_app()


if __name__ == "__main__":
    demo.queue(
        max_size=8,
        default_concurrency_limit=1,
    ).launch(
        server_name="127.0.0.1",
        inbrowser=True,
        share=False,
        show_error=True,
        max_file_size="20mb",
        allowed_paths=[
            str((RUNS_DIR / "web").resolve())
        ],
        blocked_paths=[
            str(WEIGHTS_DIR.resolve()),
            str(DATASETS_DIR.resolve()),
        ],
    )