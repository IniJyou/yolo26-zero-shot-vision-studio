from pathlib import Path

from ultralytics import YOLO

# 找到项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 准备输入、模型和输出路径
model_path = PROJECT_ROOT / "weights" / "yolo26n.pt"
image_path = PROJECT_ROOT / "weights" / "bus.jpg"
output_dir = PROJECT_ROOT / "runs" / "python_first"

# 加载模型
model = YOLO(str(model_path))

# 执行推理
results = model.predict(
    source=str(image_path),
    device=0,
    conf=0.25,
    save=True,
    project=str(output_dir.parent),
    name=output_dir.name,
    exist_ok=True,
)

# 一张输入图片对应一个结果对象
result = results[0]
height, width=result.orig_shape
pipeline_ms=sum(result.speed.values())

print(f"原图宽度：{width}")
print(f"原图高度：{height}")
print(f"检测数量：{len(result.boxes)}")
print(f"各阶段耗时：{result.speed}")
print(f"处理阶段合计：{pipeline_ms:.1f} ms")

# 遍历每个检测框
for box in result.boxes:
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())
    coordinates = [round(value,1) 
                   for value in box.xyxy[0].tolist()]
    label = result.names[class_id]

    print(
        f"类别={label}, "
        f"置信度={confidence:.3f}, "
        f"坐标={coordinates}"
    )