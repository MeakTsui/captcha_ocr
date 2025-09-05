import argparse
import os
import yaml
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F

# Optional OpenCV for Otsu preprocessing
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# Model imports
from src.models.cnn import LightCaptchaModel
from src.models.advanced_models import CRNNModel, ResNetCaptcha, DenseNetCaptcha

try:
    import onnxruntime as ort
    HAS_ORT = True
except Exception:
    HAS_ORT = False


def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


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


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    return model


def _apply_preprocess_from_cfg(img_rgb: Image.Image, pre_cfg: dict | None) -> Image.Image:
    """Mirror EnhancedCaptchaDataset.preprocess_image to keep parity."""
    if not HAS_CV2:
        # Fallback: no OpenCV, return original RGB
        print('[WARN] OpenCV not available, skipping config preprocessing.')
        return img_rgb

    cfg = pre_cfg or {}
    # defaults
    use_otsu = cfg.get('use_otsu', True)
    invert_back = cfg.get('invert_back', True)
    hough = cfg.get('hough_remove_lines', {}) or {}
    mo = cfg.get('morph_open', {}) or {}
    mc = cfg.get('morph_close', {}) or {}
    cc = cfg.get('cc_filter', {}) or {}

    rgb = np.array(img_rgb.convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # 1) Threshold
    if use_otsu:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 2) Hough line removal
    if hough.get('enable', False):
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=int(hough.get('threshold', 50)),
            minLineLength=int(hough.get('minLineLength', 30)),
            maxLineGap=int(hough.get('maxLineGap', 10)),
        )
        if lines is not None:
            thickness = int(hough.get('thickness', 1))
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv2.line(binary, (int(x1), int(y1)), (int(x2), int(y2)), color=255, thickness=thickness)

    # 3) Morph open/close
    if mo.get('enable', False):
        k = max(1, int(mo.get('ksize', 2)))
        it = max(1, int(mo.get('iterations', 1)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=it)
    if mc.get('enable', False):
        k = max(1, int(mc.get('ksize', 3)))
        it = max(1, int(mc.get('iterations', 1)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=it)

    # 4) Connected components filter
    if cc.get('enable', False):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        mask = np.zeros_like(binary)
        min_area = int(cc.get('min_area', 50))
        min_h = int(cc.get('min_h', 10))
        min_w = int(cc.get('min_w', 5))
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area >= min_area and h >= min_h and w >= min_w:
                mask[labels == i] = 255
        binary = mask

    # 5) invert back
    processed = cv2.bitwise_not(binary) if invert_back else binary
    return Image.fromarray(processed).convert('RGB')


def preprocess_image(image_path: str, image_size_hw, pre_cfg: dict | None):
    """Return (np_input: (1,C,H,W) float32 in [-1,1], torch_input: tensor on CPU)."""
    img = Image.open(image_path).convert('RGB')

    # Apply config-driven preprocessing (if cv2 present)
    img = _apply_preprocess_from_cfg(img, pre_cfg)

    # Resize: config stores [H, W], PIL expects (W, H)
    h, w = image_size_hw
    img = img.resize((w, h))

    np_img = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    np_img = (np_img - 0.5) / 0.5  # Normalize to [-1,1]
    np_input = np.expand_dims(np_img, 0).astype(np.float32)

    torch_input = torch.from_numpy(np_input)  # CPU tensor
    return np_input, torch_input


def decode_prediction(logits: torch.Tensor, charset: str):
    """logits: (B,L,C) tensor -> list of strings"""
    probs = F.softmax(logits, dim=-1)
    idx = probs.argmax(dim=-1)  # (B,L)
    codes = []
    for row in idx.cpu().numpy():
        chars = [charset[i] for i in row]
        codes.append(''.join(chars))
    return codes


def pytorch_predict(image_path: str, config: dict, checkpoint: str, device: str = 'cpu'):
    dev = torch.device(device)
    model = build_model(config).to(dev)
    model.eval()
    load_checkpoint(model, checkpoint, dev)

    np_in, torch_in = preprocess_image(
        image_path,
        config['data']['image_size'],
        pre_cfg=config.get('preprocessing', {}),
    )
    with torch.no_grad():
        logits = model(torch_in.to(dev))  # (1,L,C)
    codes = decode_prediction(logits, config['captcha']['charset'])
    return codes[0], logits.cpu().numpy()


def onnx_predict(image_path: str, config: dict, onnx_path: str):
    if not HAS_ORT:
        raise RuntimeError('onnxruntime not installed. Please pip install onnxruntime')
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f'ONNX model not found: {onnx_path}')

    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])

    np_in, _ = preprocess_image(
        image_path,
        config['data']['image_size'],
        pre_cfg=config.get('preprocessing', {}),
    )
    out = sess.run(None, {'input': np_in})[0]  # (1,L,C)

    # decode
    idx = out.argmax(axis=-1)[0]
    charset = config['captcha']['charset']
    code = ''.join(charset[i] for i in idx)
    return code, out


def preview_preprocessing(image_path: str, config: dict, show: bool = False, save_path: str | None = None):
    """Create a side-by-side preview of original vs preprocessed image.
    Does not affect prediction pipeline.
    """
    orig = Image.open(image_path).convert('RGB')
    # proc = _apply_preprocess_from_cfg(orig, config.get('preprocessing', {}))
    proc = orig

    # Make same height
    h = max(orig.height, proc.height)
    def resize_h(img: Image.Image, target_h: int) -> Image.Image:
        if img.height == target_h:
            return img
        w = int(round(img.width * (target_h / img.height)))
        return img.resize((w, target_h), Image.LANCZOS)

    orig_r = resize_h(orig, h)
    proc_r = resize_h(proc, h)

    pad = 10
    W = orig_r.width + proc_r.width + pad
    H = h + 24  # space for labels
    canvas = Image.new('RGB', (W, H), (30, 30, 30))
    canvas.paste(orig_r, (0, 24))
    canvas.paste(proc_r, (orig_r.width + pad, 24))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((4, 4), 'Original', fill=(220, 220, 220), font=font)
    draw.text((orig_r.width + pad + 4, 4), 'Preprocessed', fill=(220, 220, 220), font=font)

    if save_path:
        canvas.save(save_path)
        print(f'[preview] saved to {save_path}')
    if show:
        canvas.show()


def main():
    parser = argparse.ArgumentParser(description='Compare PyTorch and ONNX predictions for a captcha image')
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    parser.add_argument('--onnx', default='checkpoints/best_model.onnx')
    parser.add_argument('--image', default='dataset/test/ZXB8OJ_0269.png', help='Path to image to predict')
    parser.add_argument('--device', default='cpu', help='cpu|cuda|mps (for PyTorch path)')
    parser.add_argument('--show-pre', action='store_true', help='Show original vs preprocessed preview window')
    parser.add_argument('--save-preview', default='', help='Save side-by-side preview image to this path')
    args = parser.parse_args()

    config = load_config(args.config)

    # Optional: preview preprocessing
    if args.show_pre or args.save_preview:
        save_path = args.save_preview if args.save_preview else None
        preview_preprocessing(args.image, config, show=args.show_pre, save_path=save_path)

    # PyTorch
    pt_code, pt_logits = pytorch_predict(args.image, config, args.checkpoint, device=args.device)

    # ONNX
    onnx_code, onnx_logits = onnx_predict(args.image, config, args.onnx)

    print('=== Prediction Comparison ===')
    print(f'Image: {args.image}')
    print(f'PyTorch: {pt_code}')
    print(f'ONNX   : {onnx_code}')

    # Optional: quick per-position check
    L = config['captcha']['length']
    print('\nPer-position argmax (PyTorch vs ONNX):')
    pt_idx = pt_logits.argmax(axis=-1)[0]
    onnx_idx = onnx_logits.argmax(axis=-1)[0]
    charset = config['captcha']['charset']
    for i in range(L):
        print(f'  pos {i+1}: {charset[pt_idx[i]]} ({pt_idx[i]}) vs {charset[onnx_idx[i]]} ({onnx_idx[i]})')


if __name__ == '__main__':
    main()
