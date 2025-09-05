import argparse
import os
import glob
import yaml
from copy import deepcopy
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
from collections import defaultdict

# Optional OpenCV for preprocessing
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# Model imports
from src.models.cnn import LightCaptchaModel
from src.models.advanced_models import CRNNModel, ResNetCaptcha, DenseNetCaptcha


# ---------------------- utils ----------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def build_model(config: Dict[str, Any]):
    model_type = config['model'].get('type', 'light')
    if model_type == 'crnn':
        return CRNNModel(config)
    elif model_type == 'resnet':
        return ResNetCaptcha(config)
    elif model_type == 'densenet':
        return DenseNetCaptcha(config)
    else:
        return LightCaptchaModel(config)


def try_get_ckpt_config(ckpt: Dict[str, Any]) -> Dict[str, Any] | None:
    # common places people store config
    for key in ['config', 'hparams', 'cfg']:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return None


def merge_eval_config(base_cfg: Dict[str, Any], ckpt_cfg: Dict[str, Any] | None, use_current: bool) -> Tuple[Dict[str, Any], List[str]]:
    """Return the effective config for building model/eval and a list of warnings."""
    eff = deepcopy(base_cfg)
    warns: List[str] = []
    if ckpt_cfg and not use_current:
        # merge critical fields from ckpt into eff
        def set_path(path: List[str]):
            src = ckpt_cfg
            dst = eff
            try:
                for p in path[:-1]:
                    src = src[p]
                    dst = dst.setdefault(p, {})
                if path[-1] in src:
                    dst[path[-1]] = src[path[-1]]
            except Exception:
                pass
        # critical keys
        for p in [
            ['captcha', 'charset'],
            ['captcha', 'length'],
            ['data', 'image_size'],
            ['model', 'type'],
            ['model', 'use_se'],
            ['model', 'se_reduction'],
            ['preprocessing'],
        ]:
            set_path(p)
        warns.append('[info] 使用checkpoint内保存的训练配置进行评估（如有）。使用 --use-current-config 可禁用该行为。')
    # sanity checks
    if ckpt_cfg:
        def get_nested(d: Dict[str, Any], path: List[str], default=None):
            cur = d
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    return default
                cur = cur[p]
            return cur
        cs1 = get_nested(ckpt_cfg, ['captcha', 'charset'])
        cs2 = get_nested(eff, ['captcha', 'charset'])
        if isinstance(cs1, str) and isinstance(cs2, str) and cs1 != cs2:
            warns.append(f"[warn] 评估字符集与训练不一致: ckpt_charset_len={len(cs1)} vs eval_charset_len={len(cs2)}")
        se1 = get_nested(ckpt_cfg, ['model', 'use_se'])
        se2 = get_nested(eff, ['model', 'use_se'])
        if se1 is not None and se2 is not None and bool(se1) != bool(se2):
            warns.append(f"[warn] use_se 不一致: ckpt={se1} eval={se2}")
        is1 = get_nested(ckpt_cfg, ['data', 'image_size'])
        is2 = get_nested(eff, ['data', 'image_size'])
        if is1 is not None and is2 is not None and tuple(is1) != tuple(is2):
            warns.append(f"[warn] image_size 不一致: ckpt={is1} eval={is2}")
    return eff, warns


# ---------------------- preprocessing ----------------------

def _apply_preprocess_from_cfg(img_rgb: Image.Image, pre_cfg: Dict[str, Any] | None) -> Image.Image:
    if not HAS_CV2:
        return img_rgb
    cfg = pre_cfg or {}
    use_otsu = cfg.get('use_otsu', True)
    invert_back = cfg.get('invert_back', True)
    hough = cfg.get('hough_remove_lines', {}) or {}
    mo = cfg.get('morph_open', {}) or {}
    mc = cfg.get('morph_close', {}) or {}
    cc = cfg.get('cc_filter', {}) or {}

    rgb = np.array(img_rgb.convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if use_otsu:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

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

    processed = cv2.bitwise_not(binary) if invert_back else binary
    return Image.fromarray(processed).convert('RGB')


def preprocess_image(image_path: str, image_size_hw, pre_cfg: Dict[str, Any] | None):
    img = Image.open(image_path).convert('RGB')
    # img = _apply_preprocess_from_cfg(img, pre_cfg)
    h, w = image_size_hw
    img = img.resize((w, h))

    np_img = np.array(img).transpose(2, 0, 1).astype(np.float32) / 255.0
    np_img = (np_img - 0.5) / 0.5
    np_input = np.expand_dims(np_img, 0).astype(np.float32)
    torch_input = torch.from_numpy(np_input)
    return np_input, torch_input


# ---------------------- data & metrics ----------------------

def load_test_images(test_dir: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(exts)]


def get_label_from_filename(path: str) -> str:
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split('_')
    return parts[0] if len(parts) >= 1 else name


def predict_codes(model: torch.nn.Module, tin: torch.Tensor, charset: str) -> Tuple[str, float, List[int]]:
    logits = model(tin)  # (1,L,C)
    probs = F.softmax(logits, dim=-1)
    idx = probs.argmax(dim=-1)[0].cpu().numpy().tolist()
    code = ''.join(charset[i] for i in idx)
    conf = probs.max(dim=-1).values[0].mean().item()
    return code, conf, idx


def save_error_preview(orig_path: str, config: Dict[str, Any], pred: str, label: str, save_path: str):
    orig = Image.open(orig_path).convert('RGB')
    proc = _apply_preprocess_from_cfg(orig, config.get('preprocessing', {}))
    # unify height
    h = max(orig.height, proc.height)
    def resize_h(img: Image.Image, target_h: int) -> Image.Image:
        if img.height == target_h:
            return img
        w = int(round(img.width * (target_h / img.height)))
        return img.resize((w, target_h), Image.LANCZOS)
    o = resize_h(orig, h)
    p = resize_h(proc, h)
    pad = 8
    W = o.width + p.width + pad
    H = h + 28
    canvas = Image.new('RGB', (W, H), (28, 28, 28))
    canvas.paste(o, (0, 28))
    canvas.paste(p, (o.width + pad, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((4, 4), f"GT:{label}", fill=(220, 220, 220), font=font)
    draw.text((o.width + pad + 4, 4), f"Pred:{pred}", fill=(220, 220, 220), font=font)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)


def evaluate_checkpoint(ckpt_path: str, base_config: Dict[str, Any], image_paths: List[str], device: str = 'cpu',
                        use_current_config: bool = False, save_errors: int = 0, errors_dir: str = '') -> Dict[str, Any]:
    dev = torch.device(device)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_cfg = try_get_ckpt_config(ckpt)
    eff_config, warns = merge_eval_config(base_config, ckpt_cfg, use_current=use_current_config)

    for w in warns:
        print(w)

    # [diag] 打印有效配置关键信息，确保与训练一致
    try:
        print('[config] model.type =', eff_config.get('model', {}).get('type'))
        print('[config] model.use_se =', eff_config.get('model', {}).get('use_se'))
        print('[config] model.se_reduction =', eff_config.get('model', {}).get('se_reduction'))
        print('[config] captcha.length =', eff_config.get('captcha', {}).get('length'))
        charset = eff_config.get('captcha', {}).get('charset', '')
        print('[config] captcha.charset_len =', len(charset))
        print('[config] captcha.charset =', charset)
        print('[config] data.image_size =', eff_config.get('data', {}).get('image_size'))
    except Exception:
        pass

    model = build_model(eff_config).to(dev)
    model.eval()
    state = ckpt.get('model_state_dict', ckpt)

    # [diag] 尝试严格加载，失败则回退并打印缺失/多余键，定位不匹配层（如分类头）
    try:
        info = model.load_state_dict(state, strict=True)
        # 当 strict=True 成功时，通常 info.missing_keys/ unexpected_keys 为空
        print('[load] strict=True success')
    except Exception as e:
        print(f'[load] strict=True failed: {e}')
        info = model.load_state_dict(state, strict=False)
        try:
            mk = getattr(info, 'missing_keys', [])
            uk = getattr(info, 'unexpected_keys', [])
        except Exception:
            mk, uk = [], []
        print(f"[load] strict=False: missing={len(mk)} unexpected={len(uk)}")
        if mk:
            print('  missing keys (head):', mk[:10])
        if uk:
            print('  unexpected keys (head):', uk[:10])

    charset = eff_config['captcha']['charset']
    L = eff_config['captcha']['length']

    total = 0
    exact_correct = 0
    pos_correct = 0
    conf_sum = 0.0

    err_saved = 0
    with torch.no_grad():
        for img_path in image_paths:
            label = get_label_from_filename(img_path)
            _, tin = preprocess_image(img_path, eff_config['data']['image_size'], pre_cfg=eff_config.get('preprocessing', {}))
            tin = tin.to(dev)
            pred, conf, idx = predict_codes(model, tin, charset)

            conf_sum += conf
            total += 1
            if pred == label:
                exact_correct += 1
            for i, ch in enumerate(label[:L]):
                if i < len(idx) and ch == charset[idx[i]]:
                    pos_correct += 1
            # [diag] 统计字符混淆与每字符准确率
            # 初始化容器（放在循环外声明更好，但为保持上下文简洁，这里做存在性检查后再创建）
            if 'confusion_counts' not in locals():
                confusion_counts = defaultdict(int)
                char_total = defaultdict(int)
                char_correct = defaultdict(int)
            for i, ch in enumerate(label[:L]):
                pred_ch = charset[idx[i]] if i < len(idx) else ''
                char_total[ch] += 1
                if pred_ch == ch:
                    char_correct[ch] += 1
                else:
                    confusion_counts[(ch, pred_ch)] += 1
            # save some error previews
            if save_errors > 0 and pred != label and err_saved < save_errors and errors_dir:
                fname = os.path.splitext(os.path.basename(img_path))[0]
                save_path = os.path.join(errors_dir, f"{os.path.basename(ckpt_path)}__{fname}.png")
                try:
                    save_error_preview(img_path, eff_config, pred, label, save_path)
                    err_saved += 1
                except Exception as e:
                    print(f"[warn] 保存错误样本失败: {e}")

    metrics = {
        'checkpoint': os.path.basename(ckpt_path),
        'num_samples': total,
        'exact_match': exact_correct / total if total else 0.0,
        'per_pos_acc': pos_correct / (total * L) if total else 0.0,
        'mean_conf': conf_sum / total if total else 0.0,
    }
    # [diag] 打印混淆对 Top-15 与每字符准确率 Top-10（按出现频次排序）
    try:
        if 'confusion_counts' in locals() and confusion_counts:
            top_pairs = sorted(confusion_counts.items(), key=lambda kv: kv[1], reverse=True)[:15]
            print('[confusion] top mis-pairs (label->pred : count):')
            for (gt, pd), c in top_pairs:
                print(f'  {gt}->{pd} : {c}')
        if 'char_total' in locals() and char_total:
            per_char = []
            for ch, tot in char_total.items():
                cor = char_correct.get(ch, 0)
                acc = (cor / tot) if tot else 0.0
                per_char.append((ch, tot, acc))
            per_char.sort(key=lambda x: x[1], reverse=True)
            print('[char-acc] per-char accuracy (top by frequency):')
            for ch, tot, acc in per_char[:10]:
                print(f'  {ch}: acc={acc:.3f} (count={tot})')
    except Exception as e:
        print('[diag] confusion summary failed:', e)
    return metrics


# ---------------------- main ----------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate all .pth in a directory on test dataset and report metrics')
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--checkpoints', default='checkpoints')
    parser.add_argument('--test-dir', default='dataset/test')
    parser.add_argument('--device', default='cpu', help='cpu|cuda|mps')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of test images (0=all)')
    parser.add_argument('--csv', default='', help='Optional path to save CSV metrics')
    parser.add_argument('--use-current-config', action='store_true', help='Ignore config stored in checkpoint and use current config.yaml only')
    parser.add_argument('--save-errors', type=int, default=0, help='Save up to N error previews per checkpoint')
    parser.add_argument('--errors-dir', default='errors_eval', help='Directory to save error previews')
    args = parser.parse_args()

    base_config = load_config(args.config)

    image_paths = load_test_images(args.test_dir)
    image_paths.sort()
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    ckpts = sorted(glob.glob(os.path.join(args.checkpoints, '*.pth')))
    if not ckpts:
        print(f'No .pth found in {args.checkpoints}')
        return

    results = []
    for i, ckpt in enumerate(ckpts, 1):
        print(f'[{i}/{len(ckpts)}] Evaluating {os.path.basename(ckpt)} ...')
        metrics = evaluate_checkpoint(
            ckpt, base_config, image_paths,
            device=args.device,
            use_current_config=args.use_current_config,
            save_errors=args.save_errors,
            errors_dir=args.errors_dir,
        )
        results.append(metrics)
        print(f"  exact_match={metrics['exact_match']:.4f}  per_pos_acc={metrics['per_pos_acc']:.4f}  mean_conf={metrics['mean_conf']:.4f}")

    results.sort(key=lambda m: (m['exact_match'], m['per_pos_acc'], m['mean_conf']), reverse=True)

    print('\n=== Summary (best first) ===')
    for m in results:
        print(f"{m['checkpoint']}: EM={m['exact_match']:.4f}, PosAcc={m['per_pos_acc']:.4f}, Conf={m['mean_conf']:.4f}, N={m['num_samples']}")

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['checkpoint', 'num_samples', 'exact_match', 'per_pos_acc', 'mean_conf'])
            w.writeheader()
            for m in results:
                w.writerow(m)
        print(f'[CSV] saved to {args.csv}')


if __name__ == '__main__':
    main()
