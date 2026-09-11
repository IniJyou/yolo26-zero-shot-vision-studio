# YOLO26 Zero-Shot Vision Studio

一个用于学习目标检测和计算机视觉工程化的开源项目。

## 项目目标

本项目计划实现：

- 使用YOLO26进行固定类别免训练检测
- 使用YOLOE进行文本提示零样本检测
- 支持图片和视频输入
- 使用Gradio制作Web演示页面
- 输出目标类别、置信度、坐标和推理耗时
- 对至少40张测试图片进行效果与性能评测

## 当前进度

- [x] 配置Python虚拟环境
- [x] 配置PyTorch CUDA
- [x] 安装Ultralytics
- [x] 完成第一次YOLO26 GPU推理
- [x] 建立项目目录
- [ ] 编写Python图片检测程序
- [ ] 接入YOLOE
- [ ] 制作Gradio页面
- [ ] 实现视频检测
- [ ] 完成对比评测

## 已验证环境

- Windows 11
- Python 3.12.4
- PyTorch 2.14.0+cu130
- CUDA 13.0
- Ultralytics 8.4.147
- NVIDIA GeForce RTX 4060 Laptop GPU

## 项目说明

YOLO26使用预训练固定类别进行检测；YOLOE支持通过文本提示指定类别。

动漫图片将作为零样本检测案例，但本项目不保证能够准确区分具体动漫角色。

## 学习记录

开发过程、问题和解决方法记录在 [LEARN.md](LEARN.md)。
