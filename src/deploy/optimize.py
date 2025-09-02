import torch
import torch.nn as nn

def optimize_for_inference(model, input_shape=(1, 3, 70, 200)):
    """优化模型用于推理"""
    model.eval()  # 设置为推理模式
    
    # 1. 融合批归一化层（如果有）
    model = torch.quantization.fuse_modules(model, ['conv1', 'bn1'])
    
    # 2. 转换为TorchScript
    example = torch.rand(*input_shape)
    traced_model = torch.jit.trace(model, example)
    
    # 3. 量化模型（可选）
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    
    return traced_model, quantized_model

def export_onnx(model, save_path, input_shape=(1, 3, 70, 200)):
    """导出为ONNX格式"""
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    ) 