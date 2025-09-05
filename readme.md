# Captcha OCR 项目说明

本项目提供两种模型方案：
- RNN 方案（CRNNModel）：CNN 特征 + 双向 LSTM 序列建模，准确率高、结构通用。
- 优化方案（ResNetCaptcha Tiny）：纯 CNN（TinyResNet），支持静态 ONNX 导出与 ONNXRuntime 量化，CPU 单张推理延迟可压至 <10ms（batch=1）。

涉及核心文件：
- 模型定义：`src/models/advanced_models.py`（`CRNNModel`、`ResNetCaptcha`、`DenseNetCaptcha`）
- 数据集（含可选预处理）：`src/data/enhanced_dataset.py`
- 推理部署：`src/deploy/inference.py`
- 导出 ONNX：`export_onnx.py`
- ONNX 量化与优化：`src/deploy/optimize.py`
- 配置：`config/config.yaml`

数据输入尺寸：200×70（W×H），配置为 `data.image_size: [70, 200]`（H, W），训练与推理预处理保持一致。

## 快速开始

1) 安装依赖
```bash
pip install -r requirements.txt
```

关键依赖：
- torch, torchvision, Pillow, PyYAML, tqdm
- onnxruntime, onnx
- opencv-python
- fastapi, uvicorn（如需服务化）

2) 数据准备
- 训练集与验证集目录在 `config/config.yaml` 的 `data.train_dir` 和 `data.val_dir` 指定。
- 图片命名形如 `LABEL_xxx.png`，其中 `LABEL` 为验证码字符串，长度由 `captcha.length` 控制（默认 6）。

## 方案对比：RNN 方案 vs 优化方案

- RNN 方案（`CRNNModel`）
  - 结构：多层卷积下采样 + 双向 LSTM + 位置分类（无 CTC）。
  - 参数规模：约 7–8M（与输入分辨率无关），CPU 推理相对较慢，延迟受 LSTM 影响明显。
  - 适用：对序列建模更敏感的场景，或有 GPU 推理资源。

- 优化方案（`ResNetCaptcha` Tiny）
  - 结构：纯 CNN（TinyResNet），全局池化 + 线性层直接输出位置分类结果。
  - 参数规模：数百万级以下，CPU 友好；配合静态 ONNX + ORT 量化，单张延迟可 <10ms。
  - 适用：CPU 上小延迟场景、批量=1 的在线服务。

## 配置说明（`config/config.yaml`）

关键字段：
- `data.image_size: [70, 200]` 保持训练与推理一致
- `captcha.length: 6`，`captcha.charset: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"`（36 类）
- `model.type`
  - RNN 方案：`crnn`
  - 优化方案：`resnet`（默认 tiny 通道配置已在代码中启用）
- `model.use_se`：优化方案建议 `false`（去除 SE，进一步降低延迟）
- `inference`
  - `batch_size: 1`（静态 ONNX）
  - `num_threads: 2~6`（建议网格搜索 2/3/4/6，以延迟优先）
  - `providers: ["CPUExecutionProvider"]`（Mac/通用 CPU）
  - `use_mkldnn: true`（启用 DNNL 内存/图优化）

示例（优化方案核心片段）：
```yaml
model:
  type: "resnet"
  use_se: false
inference:
  batch_size: 1
  num_threads: 4
  use_mkldnn: true
  providers: ["CPUExecutionProvider"]
```

## 训练（可选）

如果切换结构（如从 CRNN 到 ResNet），建议重新训练或至少 Finetune 一段时间，以适配纯 CNN 的分布。

示例（如有自定义训练脚本）：
```bash
python main.py --config config/config.yaml
```

## 导出 ONNX（静态）

导出脚本：`export_onnx.py`

- 将模型按当前 `config/config.yaml` 的结构实例化
- 加载 checkpoint（可选）
- 导出静态 ONNX（固定 batch=1，无动态轴），更利于 CPU 极致优化

示例命令：
```bash
python export_onnx.py \
  --config config/config.yaml \
  --checkpoint checkpoints/best_model.pth \
  --out checkpoints/best_model.onnx \
  --opset 11
```

导出结果：`checkpoints/best_model.onnx`

## ONNX 量化与优化（推荐）

文件：`src/deploy/optimize.py`

提供两种量化方式：

1) 动态量化（推荐先用）
- 简单易用，对 CPU 延迟收益大，通常几分钟内完成
```python
from src.deploy.optimize import onnx_dynamic_quantize
onnx_dynamic_quantize(
    "checkpoints/best_model.onnx",
    "checkpoints/best_model.int8.onnx"
)
```

2) 静态量化（QDQ，需要校准数据）
- 一般精度最好、延迟更低
- 需要实现一个 `CalibrationDataReader`（文件中提供了 `DummyCalibrationDataReader` 示例），喂入 N 张预处理后的图片张量（numpy，形状 (1, C, H, W)）
```python
from src.deploy.optimize import onnx_static_quantize, DummyCalibrationDataReader

images = [...]  # 若干预处理后的 Numpy 数组：(1, 3, 70, 200), float32, 归一化至 (-1,1)
reader = DummyCalibrationDataReader(images, input_name="input", image_size=(70, 200))
onnx_static_quantize(
    "checkpoints/best_model.onnx",
    reader,
    "checkpoints/best_model.qdq.onnx"
)
```

产出文件：
- 动态量化：`*.int8.onnx`
- 静态量化：`*.qdq.onnx`

## 推理（ONNXRuntime）

入口：`src/deploy/inference.py`（类 `CaptchaPredictor`）

- 支持按 `config.inference` 配置线程与 Provider
- 建议批量=1、顺序执行模式，降低调度与缓存开销

示例：
```python
import yaml
from src.deploy.inference import CaptchaPredictor

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

model_path = "checkpoints/best_model.int8.onnx"  # 原始/量化 ONNX 均可
predictor = CaptchaPredictor(model_path, config)
text = predictor.predict("path/to/captcha.png")
print(text)
```

## 基准测试（延迟）

建议方法：
- 使用静态 ONNX（batch=1）
- 先 warmup 5–10 次
- 再计时 100–1000 次，计算均值/中位数/分位数（p50/p95）

示例：
```python
import time
from PIL import Image
import yaml
from src.deploy.inference import CaptchaPredictor

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

predictor = CaptchaPredictor("checkpoints/best_model.int8.onnx", config)
img = Image.open("path/to/captcha.png").convert("RGB")

# warmup
for _ in range(10):
    predictor.predict(img)

# benchmark
N = 200
t0 = time.perf_counter()
for _ in range(N):
    predictor.predict(img)
t1 = time.perf_counter()
print(f"avg latency: {(t1 - t0) / N * 1000:.2f} ms")
```

线程数调优：
- 将 `config.inference.num_threads` 在 2/3/4/6（甚至 8）间做网格搜索，挑选最小延迟的设置。
- 小 batch 下，线程过多反而会增加调度开销。

## 常见问题（FAQ）

- 为什么推荐静态 ONNX（固定 batch=1）？
  - 静态形状更容易被 ORT/DNNL 做图优化与算子融合，小 batch 的延迟通常比动态轴更低。

- 为什么取消 SE/注意力？
  - 这些模块有益于精度，但会带来额外算子与内存访问，CPU 单张延迟敏感时优先移除。

- RNN 方案是否被完全替代？
  - 没有。RNN 对某些序列任务更鲁棒，若你有更充足的算力（如 GPU）或精度优先，可以继续使用 `CRNNModel`。在纯 CPU 延迟优先的场景，优先 TinyResNet 方案。

## 目录结构（简要）

- `config/`
  - `config.yaml`：全局配置
- `src/models/advanced_models.py`：`CRNNModel`、`ResNetCaptcha`、`DenseNetCaptcha`
- `src/data/enhanced_dataset.py`：可配置预处理的数据集
- `src/deploy/inference.py`：ONNXRuntime 推理封装
- `src/deploy/optimize.py`：ONNX 导出、动态/静态量化工具
- `export_onnx.py`：静态 ONNX 导出入口
- `checkpoints/`：存放权重与导出的 ONNX

## 达成 <10ms/张 的建议路径

- 使用 `ResNetCaptcha` Tiny（纯 CNN，去注意力/SE）
- 导出静态 ONNX（batch=1）
- 优先应用 ORT 动态量化（INT8）；若仍需更低延迟，改用静态量化（QDQ）
- 调整 `num_threads`（2–6）寻优
- 如仍不达标，进一步瘦身通道数（如 `24, 48, 96`）或减少 block 数
