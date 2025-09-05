import argparse
import os
import yaml
import torch

# Model imports
from src.models.cnn import LightCaptchaModel
from src.models.advanced_models import CRNNModel, ResNetCaptcha, DenseNetCaptcha


def build_model(config):
    model_type = config['model'].get('type', 'light')
    if model_type == 'crnn':
        return CRNNModel(config)
    elif model_type == 'resnet':
        return ResNetCaptcha(config)
    elif model_type == 'densenet':
        return DenseNetCaptcha(config)
    else:
        return LightCaptchaModel(config)


def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def load_checkpoint_if_any(model: torch.nn.Module, ckpt_path: str):
    if not ckpt_path:
        return
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # Support both raw state_dict and full trainer-style dict
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)


def export_onnx(model: torch.nn.Module, config: dict, out_path: str, opset: int = 11):
    model.eval()
    h, w = config['data']['image_size']
    c = config['model'].get('input_channels', 3)
    dummy = torch.randn(1, c, h, w, dtype=torch.float32)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # 静态导出：固定 batch=1，无 dynamic_axes
    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
    )
    print(f"ONNX model exported to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Export captcha OCR model to ONNX (static)')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth', help='Path to .pth checkpoint (optional)')
    parser.add_argument('--out', default='checkpoints/best_model.onnx', help='Output ONNX path')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    args = parser.parse_args()

    config = load_config(args.config)
    model = build_model(config)
    load_checkpoint_if_any(model, args.checkpoint)

    export_onnx(model, config, args.out, opset=args.opset)


if __name__ == '__main__':
    main()
