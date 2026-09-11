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
- [ ] 建立项目文件夹框架
- [ ] 编写 `.gitignore`
- [ ] 编写最小版 `README.md`
- [ ] 初始化并上传 GitHub 仓库
- [ ] 编写第一个 Python 推理程序

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
