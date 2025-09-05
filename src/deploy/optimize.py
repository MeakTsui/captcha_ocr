import os
from typing import Optional

import torch
import torch.nn as nn

try:
    from onnxruntime.quantization import quantize_dynamic, quantize_static, CalibrationDataReader, QuantFormat, QuantType
except Exception:
    quantize_dynamic = None
    quantize_static = None
    CalibrationDataReader = object
    QuantFormat = None
    QuantType = None


def optimize_for_inference(model: nn.Module, input_shape=(1, 3, 70, 200)):
    """保留 TorchScript trace（可用于 CPU 推理），不在此处做 PyTorch 量化。
    注意：原先对 Conv2d 的动态量化是无效的（PyTorch 仅支持 Linear/LSTM 的动态量化）。
    """
    model.eval()
    example = torch.rand(*input_shape)
    traced_model = torch.jit.trace(model, example)
    return traced_model


def export_onnx(model: nn.Module, save_path: str, input_shape=(1, 3, 70, 200), opset: int = 11):
    """导出为静态 ONNX（固定 batch=1）。"""
    model.eval()
    dummy_input = torch.randn(*input_shape)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )
    return save_path


# ---------------- ONNXRuntime 量化工具 ----------------
class DummyCalibrationDataReader(CalibrationDataReader):
    """示例校准数据读取器。请替换为你的真实数据集读取逻辑。"""
    def __init__(self, images, input_name: str, image_size=(70, 200)):
        self.images = images
        self.input_name = input_name
        self.image_size = image_size
        self._iter = iter(self.images)

    def get_next(self):
        try:
            arr = next(self._iter)
        except StopIteration:
            return None
        # 期望 arr 已是 (1, C, H, W) float32 标准化后的 numpy 数组
        return {self.input_name: arr}


def onnx_dynamic_quantize(model_path: str, out_path: Optional[str] = None):
    """对 ONNX 模型进行动态量化（权重 int8）。对 CPU 延迟影响大、部署便捷。"""
    if quantize_dynamic is None:
        raise RuntimeError('onnxruntime.quantization 未安装，请先安装 onnx 和 onnxruntime')
    if out_path is None:
        root, ext = os.path.splitext(model_path)
        out_path = root + '.int8' + ext
    quantize_dynamic(model_input=model_path,
                     model_output=out_path,
                     weight_type=QuantType.QInt8,
                     optimize_model=True)
    return out_path


def onnx_static_quantize(model_path: str, dataloader: CalibrationDataReader, out_path: Optional[str] = None):
    """对 ONNX 模型进行静态量化（QDQ），需提供校准数据。一般精度更好，延迟更低。"""
    if quantize_static is None:
        raise RuntimeError('onnxruntime.quantization 未安装，请先安装 onnx 和 onnxruntime')
    if out_path is None:
        root, ext = os.path.splitext(model_path)
        out_path = root + '.qdq' + ext
    quantize_static(model_input=model_path,
                    model_output=out_path,
                    calibration_data_reader=dataloader,
                    quant_format=QuantFormat.QDQ,
                    per_channel=True,
                    optimize_model=True)
    return out_path