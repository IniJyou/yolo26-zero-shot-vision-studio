# YOLO26 Zero-Shot Vision Studio 学习日志

本文档记录项目开发过程中学到的知识、亲手完成的工作、遇到的问题及解决方法。它既是学习复盘，也是后续准备项目介绍和面试回答的依据。

## 2026-09-11：开发环境配置

### 学习目标

搭建能够通过 NVIDIA GPU 运行 YOLO26 的独立 Python 开发环境，并避免模型与缓存占用 C 盘。

### 学到的知识

1. Python 虚拟环境可以隔离不同项目的依赖，避免污染全局 Python。
2. 安装 NVIDIA 驱动不代表 PyTorch 一定支持 GPU，还必须安装带 CUDA 的 PyTorch 构建。
3. `torch.cuda.is_available()` 可以验证 PyTorch 能否真正调用显卡。
4. 模型权重、数据集、运行结果和工具缓存需要分别管理。
5. `.venv` 放在项目根目录是常见做法，但不能上传 GitHub。

### 亲手完成的工作

- 在 `D:\aiworkspace\yolo\.venv` 创建并激活虚拟环境。
- 安装 PyTorch CUDA、Ultralytics 和 OpenCV。
- 使用 `yolo checks` 验证开发环境。
- 将 Ultralytics 配置迁移到 `D:\ai-cache\Ultralytics`。
- 将数据集、模型权重和运行结果固定在 D 盘项目目录。

### 验证结果

| 组件 | 版本或状态 |
|---|---|
| Python | 3.12.4 |
| PyTorch | 2.14.0+cu130 |
| CUDA Runtime | 13.0 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB |
| Ultralytics | 8.4.147 |
| OpenCV | 5.0.0 |
| CUDA 可用 | `True` |

### 遇到的问题与解决方法

#### PyTorch 无法使用 GPU

最初安装的是 CPU 版 PyTorch，表现为：

```text
torch: 2.12.1+cpu
cuda available: False
```

解决方法是在项目虚拟环境中安装带 CUDA 13.0 的 PyTorch。安装后，PyTorch 成功识别 RTX 4060。

#### Ultralytics 默认使用 C 盘配置目录

Ultralytics 首次运行时默认尝试访问：

```text
C:\Users\luoyu\AppData\Roaming\Ultralytics
```

通过设置 `YOLO_CONFIG_DIR`，最终将配置文件迁移到：

```text
D:\ai-cache\Ultralytics\settings.json
```

当前资源目录为：

```text
datasets_dir = D:\aiworkspace\yolo\datasets
weights_dir  = D:\aiworkspace\yolo\weights
runs_dir     = D:\aiworkspace\yolo\runs
```

## 2026-09-11：第一次 YOLO26 GPU 推理

### 学习目标

使用官方 YOLO26n 预训练权重检测示例图片，验证模型下载、GPU 推理和结果保存的完整流程。

### 使用的关键参数

- `model=yolo26n.pt`：使用体积较小的 YOLO26n 预训练模型。
- `source=.../bus.jpg`：指定待检测图片。
- `device=0`：使用第一张 CUDA 显卡。
- `conf=0.25`：只保留置信度不低于 0.25 的结果。
- `save=True`：保存绘制检测框后的图片。

### 实际结果

```text
4 persons
1 bus
inference: 21.8 ms
```

结果保存于：

```text
D:\aiworkspace\yolo\runs\first_predict\bus.jpg
```

模型权重保存于：

```text
D:\aiworkspace\yolo\weights\yolo26n.pt
```

### 学到的知识

1. 本次不需要自行训练，因为使用的是已经训练好的官方权重。
2. `CUDA:0` 表示推理使用了第一张 NVIDIA GPU。
3. Ultralytics 不一定显示单独的 `Success`，出现检测结果与 `Results saved to ...` 就表示运行成功。
4. `21.8 ms inference` 只表示模型推理阶段的耗时，不包含完整的图片读取、预处理和结果保存时间。
5. PowerShell 中的 `>>` 是多行命令续行提示，不是报错。

## 项目目标与边界

项目名称：**YOLO26 Zero-Shot Vision Studio**。

计划开发一个基于 YOLO26 与 YOLOE-26 的开源目标检测项目：

- 使用 YOLO26 演示固定 COCO 类别的免训练检测。
- 使用 YOLOE-26 演示文本提示驱动的开放词汇零样本检测。
- 支持图片和视频输入。
- 使用 Gradio 制作本地 Web 演示页面。
- 输出类别、置信度、边界框、耗时和 JSON 结果。
- 使用至少 40 张自建测试图片比较目标命中率与推理性能。

动漫图片将作为零样本检测的特色案例，但项目不会声称能够可靠区分喜羊羊、美羊羊等具体角色。检测到 `cartoon sheep` 不等于完成角色身份识别。

## 2026-09-11：项目骨架与 GitHub 代码管理

### 学习目标

建立清晰的项目目录，用 Git 保存开发历史，并把安全的项目骨架推送到 GitHub。

### 学到的知识

1. Git 保存的是一次次可恢复的项目状态，GitHub 是远程代码托管平台。
2. `.gitignore` 用于阻止本地环境、模型权重、数据集、运行结果和密钥进入版本库。
3. `.gitkeep` 是用于保留空目录的普通占位文件，不是 Git 的特殊语法。
4. `git status --short` 用于快速查看修改；`git check-ignore -v` 可以说明文件被哪条规则忽略。
5. `origin` 是远程仓库的常用别名，`main` 是当前主分支。
6. 小而清晰的提交比一次提交大量无关修改更容易理解和回退。

### 亲手完成的工作

- 建立 `src`、`tests`、`benchmarks`、`assets` 和 `docs` 等项目目录。
- 编写 `.gitignore`，确认 `.venv`、`yolo26n.pt` 和 `runs` 结果不会上传。
- 编写项目首页 `README.md`。
- 初始化本地 Git 仓库并建立 `main` 分支。
- 添加 AGPL-3.0 许可证。
- 创建并连接 GitHub 仓库：`IniJyou/yolo26-zero-shot-vision-studio`。
- 将本地提交推送到 `origin/main`。

### 提交记录

```text
58f3b8f chore: initialize project structure
73c4d8d docs: add AGPL-3.0 license
```

## 2026-09-11：VS Code 环境与 Python 图片推理

### 学习目标

在 VS Code 中使用项目虚拟环境运行 Python，并通过代码读取 YOLO26 的结构化检测结果。

### 遇到的问题与解决方法

VS Code 曾提示“无法解析导入 ultralytics”。实际原因不是安装失败，而是编辑器使用了全局 Python：

```text
D:\python\python.exe
```

Ultralytics 安装在项目虚拟环境：

```text
D:\aiworkspace\yolo\.venv\Scripts\python.exe
```

通过 VS Code 的 `Python: Select Interpreter` 选择 `.venv` 解释器并重新加载窗口后，导入警告消失。

终端是否显示 `(.venv)` 与 VS Code 代码分析器选择哪个解释器是两个相关但独立的状态。运行项目文件应使用 Python 扩展提供的 `Run Python File`，避免其他运行扩展绕过虚拟环境。

### 亲手完成的工作

- 创建 `examples/01_yolo26_image.py`。
- 使用 `pathlib.Path` 从脚本位置计算项目根目录。
- 从本地 `weights/yolo26n.pt` 加载 YOLO26n。
- 使用 RTX 4060 对 `bus.jpg` 执行推理。
- 遍历 `result.boxes`，读取类别、置信度与边界框。
- 对坐标进行一位小数格式化。
- 使用 VS Code 直接运行 Python 文件。

### 代码知识

- `Path(__file__).resolve().parents[1]` 可以稳定找到项目根目录，减少对当前工作目录的依赖。
- Ultralytics 支持批量输入，因此 `model.predict()` 返回结果列表；单张图片使用 `results[0]`。
- `box.cls` 是类别编号，`result.names[class_id]` 将编号映射为类别名称。
- `box.conf` 是置信度，取值通常位于 0 到 1。
- `box.xyxy` 表示 `[x1, y1, x2, y2]`，即左上角和右下角坐标。
- `result.orig_shape` 的顺序是 `(height, width)`，不是 `(width, height)`。
- 终端显示的 `640x480` 是推理输入尺寸，检测框坐标已经映射回原始图片尺寸。
- `result.speed` 分别记录预处理、模型推理和后处理耗时。

### 本次运行结果

```text
原图尺寸：(1080, 810)
检测数量：5
检测类别：4 persons，1 bus
模型推理耗时：约 10.2 ms
```

本次单张图片运行包含首次调用波动，只能用于确认功能，不能作为最终性能指标。正式评测需要预热模型并重复运行多次。

## 2026-09-11：检测结果导出为 JSON

### 学习目标

将终端中的检测信息转换成结构化数据，并保存为后续 Web 页面、接口和评测工具能够读取的 JSON 文件。

### 学到的知识

1. Python 字典使用键值对描述一个对象，列表用于保存数量不固定的多个检测对象。
2. `json.dumps()` 将 Python 字典和列表序列化为 JSON 文本。
3. `ensure_ascii=False` 保留可读的中文，`indent=2` 生成方便检查的缩进格式。
4. `Path.write_text(..., encoding="utf-8")` 可以明确使用 UTF-8 保存文本文件。
5. PyTorch Tensor 不能直接写入 JSON，需要使用 `.item()`、`float()`、`int()` 和 `.tolist()` 转换成普通 Python 数据。
6. 程序没有报错、列表长度正确，都不能证明字段内容一定正确；还需要验证关键值。

### 亲手完成的工作

- 将每个检测目标转换成包含 `class_id`、`label`、`confidence` 和 `bbox_xyxy` 的字典。
- 使用 `detections` 列表收集五个检测对象。
- 保存原图尺寸、模型名称、各阶段耗时和检测数量。
- 生成 `runs/python_first/bus.json`。
- 验证 JSON 可以重新解析，检测数量与列表长度均为 5。
- 验证类别顺序为 `bus` 和四个 `person`。

### 遇到的问题与解决方法

#### 创建了字典但检测列表仍为空

第一次只在循环中写了一个独立字典，没有保存它。Python 创建字典后立即丢弃，因此最终得到：

```json
{
  "detection_count": 0,
  "detections": []
}
```

解决方法是先将字典赋给变量，再追加到列表：

```python
detection = {
    "class_id": class_id,
    "label": label,
    "confidence": round(confidence, 4),
    "bbox_xyxy": coordinates,
}
detections.append(detection)
```

#### 把变量误写成固定字符串

第二次把类别写成了：

```python
"label": "label"
```

这会让所有检测对象的类别都变成固定文本 `label`。正确写法是：

```python
"label": label
```

前者是字符串字面量，后者会读取当前循环中 `label` 变量的实际值。修复后，第一个检测对象正确保存为：

```json
{
  "class_id": 5,
  "label": "bus",
  "confidence": 0.8806,
  "bbox_xyxy": [0.0, 230.4, 803.2, 750.7]
}
```

## 三周学习与开发计划

### 第 1 周：图片推理基础

- 建立项目目录和 Git 仓库。
- 编写最小版 Python 图片推理程序。
- 读取目标类别、置信度、边界框和耗时。
- 将检测逻辑封装成可复用模块。
- 学习基础异常处理和单元测试。

预期成果：可以通过命令行检测本地图片并输出结构化结果。

### 第 2 周：YOLOE 与 Web 应用

- 理解固定类别检测与开放词汇检测的区别。
- 接入 YOLOE-26 和英文提示词。
- 使用 Gradio 建立图片检测页面。
- 实现视频逐帧处理和结果导出。

预期成果：可以在浏览器中切换 YOLO26 与 YOLOE，处理图片和视频。

### 第 3 周：评测与开源展示

- 准备至少 40 张合法使用的测试图片。
- 记录预期类别、模型、提示词和检测结果。
- 统计目标命中率、推理耗时和端到端耗时。
- 完善 README、截图、演示 GIF 和限制说明。
- 发布 GitHub 开源仓库并整理简历描述。

预期成果：形成一个能够公开展示、指标真实、可以在面试中解释的完整项目。

## 2026-09-12：配置常量与输入校验

### 学习目标

把经常调整的推理参数集中为配置常量，并在模型运行前主动检查输入，避免让错误进入耗时的推理阶段。

### 学到的知识

1. 全大写变量名通常表示配置常量，例如 `CONFIDENCE_THRESHOLD`。
2. 条件表达式可以根据 CUDA 状态选择运行设备：`0 if torch.cuda.is_available() else "cpu"`。
3. Python 集合适合保存允许的图片扩展名并进行成员判断。
4. `Path.is_file()` 可以判断模型或图片是否为真实文件。
5. `Path.suffix.lower()` 可以取得并统一比较文件扩展名。
6. `FileNotFoundError` 表示需要的文件不存在，`ValueError` 表示输入值或格式不符合要求。
7. `Path.mkdir(parents=True, exist_ok=True)` 可以安全创建输出目录。

### 亲手完成的工作

- 增加置信度、运行设备和允许图片格式等配置。
- 在 CUDA 不可用时为设备配置准备 CPU 回退值。
- 增加模型权重不存在检查。
- 增加输入图片不存在和格式不支持检查。
- 主动创建推理输出目录。
- 使用 `missing.jpg` 验证异常分支，得到清晰的 `FileNotFoundError`。
- 恢复正常输入并再次完成检测，JSON仍包含 `bus` 和四个 `person`。

### 遇到的问题与解决方法

模型文件不存在时曾误用 `FileExistsError`。该异常表示目标已经存在，与实际语义相反；最终改为 `FileNotFoundError`。这说明异常类型不仅控制程序停止，还应准确表达失败原因，方便调用者、测试和日志判断问题。

## 2026-09-12：函数封装与程序入口

### 学习目标

将单文件中的连续执行代码按职责拆分为函数，为后续迁移到 `src` 模块、接入 YOLOE 和 Web 页面做准备。

### 亲手完成的工作

- 使用 `validate_inputs()` 封装模型、图片和扩展名校验。
- 使用 `parse_detections()` 将 Ultralytics `Results` 转换为可序列化列表。
- 使用 `save_json()` 负责目录创建、JSON序列化和UTF-8写入。
- 使用 `main()` 收拢路径、模型推理、打印和保存流程。
- 使用 `if __name__ == "__main__"` 建立明确程序入口。
- 验证脚本编译成功，输出JSON仍包含5个正确目标。

### 关键工程认识

- 函数应尽量只有一个主要职责，便于测试和复用。
- 解析函数返回普通Python数据，使业务代码不必到处依赖Tensor细节。
- `save_json()` 返回最终路径，调用者可以继续显示、记录或下载该文件。
- 类型标注帮助编辑器发现错误，但Python运行时通常不会自动强制检查类型。
- 入口保护让文件被其他模块导入时只提供函数，不会立即加载模型并执行推理。
- “解析数据、展示数据、保存数据”分离后，更容易替换成Gradio页面或API。

## 2026-09-12：建立标准 Python 包

### 学习目标

把核心代码组织为可安装、可跨目录导入的 `vision_studio` 包，为后续示例、测试和 Web 应用共享代码建立基础。

### 亲手完成的工作

- 将原先直属于 `src` 的空模块迁移到 `src/vision_studio/`。
- 创建 `pyproject.toml`，声明项目名称、Python版本、依赖和包发现目录。
- 使用 `python -m pip install -e . --no-deps` 完成可编辑安装。
- 将项目路径和默认参数迁移到 `vision_studio.config`。
- 修改示例脚本，从核心包导入配置。
- 在 VS Code 中直接运行示例，仍检测到 `bus` 和四个 `person`。

### 关键工程认识

- `src` 是源码容器，`vision_studio` 是实际的 Python 导入包。
- 发布名称可以使用连字符 `yolo26-zero-shot-vision-studio`，导入名使用下划线 `vision_studio`。
- 可编辑安装不会复制源码；修改 `src/vision_studio` 后，虚拟环境会直接使用最新代码。
- 配置通过 `Path(__file__)` 推导项目根目录，没有把本机的 `D:\aiworkspace\yolo` 写死到公开源码中。
- `*.egg-info/` 是安装产生的元数据，不属于需要提交的项目源码。

### 验证结果

```text
package = D:\aiworkspace\yolo\src\vision_studio\__init__.py
root = D:\aiworkspace\yolo
model = D:\aiworkspace\yolo\weights\yolo26n.pt
检测数量 = 5
```

## 项目目录职责

| 路径 | 职责 | 是否提交 GitHub |
|---|---|---|
| `.venv/` | Python 虚拟环境和第三方依赖 | 否 |
| `src/` | 核心 Python 源代码 | 是 |
| `tests/` | 自动测试 | 是 |
| `benchmarks/` | 评测清单、脚本和结果 | 是 |
| `assets/` | 自有截图和演示素材 | 是 |
| `docs/` | 设计说明与补充文档 | 是 |
| `weights/` | 模型权重 | 否 |
| `datasets/` | 本地数据集 | 默认否 |
| `runs/` | 推理产生的结果 | 否 |
| `app.py` | Gradio Web 应用入口 | 是 |
| `requirements.txt` | 可复现的项目依赖 | 是 |
| `.gitignore` | Git 忽略规则 | 是 |

## 当前进度

- [x] 创建 Python 虚拟环境
- [x] 安装 PyTorch CUDA
- [x] 安装 Ultralytics 和 OpenCV
- [x] 验证 RTX 4060 GPU 推理环境
- [x] 完成第一次 YOLO26 图片检测
- [x] 将配置、权重、数据集和结果放在 D 盘
- [x] 制定三周项目计划
- [x] 创建学习日志
- [x] 建立项目文件夹框架
- [x] 编写 `.gitignore`
- [x] 编写最小版 `README.md`
- [x] 初始化并上传 GitHub 仓库
- [x] 编写第一个 Python 推理程序
- [x] 将图片检测结果导出为 JSON
- [x] 增加配置与输入文件校验
- [x] 将输入校验与结果转换封装成函数
- [x] 建立可编辑安装的 `vision_studio` Python 包
- [ ] 将示例逻辑迁移到 `src` 核心模块

## 后续记录规范

每完成一个学习任务，继续追加以下内容：

```markdown
## 日期：学习主题

### 学习目标

### 学到的知识

### 亲手完成的工作

### 遇到的问题与解决方法

### 我的理解与反思

### 下一步
```

其中“我的理解与反思”由项目开发者本人填写，用自己的语言解释技术选择、错误原因和解决过程。
