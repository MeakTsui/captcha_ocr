#!/usr/bin/env python3
"""
Generate a transparent PNG with gradient-filled text.

Default text: "Cricket AI"
Default gradient: left-to-right from #60a5fa to #c084fc

Usage examples:
  python scripts/generate_gradient_text.py \
      --text "Cricket AI" \
      --out scripts/cricket_ai.png \
      --font-size 128 \
      --padding 20

Optional font:
  python scripts/generate_gradient_text.py --font /System/Library/Fonts/SFNS.ttf

Note: If the specified font is unavailable, the script will try DejaVuSans.ttf,
then fall back to PIL's default bitmap font.
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.strip().lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b


def lerp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


def lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t)


def load_font(font_path: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # 1) Use user-provided font if valid
    if font_path and os.path.isfile(font_path):
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            pass
    # 2) Try a common bundled font in Pillow
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        pass
    # 3) Fallback to default bitmap font (size is fixed)
    return ImageFont.load_default()


def measure_text(text: str, font: ImageFont.ImageFont) -> Tuple[int, int, Tuple[int, int, int, int]]:
    # Use a temporary image to get precise bbox via textbbox
    temp_img = Image.new("L", (1, 1), 0)
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    # bbox = (left, top, right, bottom)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width, height, bbox


def create_gradient(size: Tuple[int, int], start_rgb: Tuple[int, int, int], end_rgb: Tuple[int, int, int], direction: str = "horizontal") -> Image.Image:
    w, h = size
    grad = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(grad)

    if direction == "horizontal":
        for x in range(w):
            t = 0 if w == 1 else x / (w - 1)
            color = lerp_color(start_rgb, end_rgb, t)
            draw.line([(x, 0), (x, h)], fill=(*color, 255))
    elif direction == "vertical":
        for y in range(h):
            t = 0 if h == 1 else y / (h - 1)
            color = lerp_color(start_rgb, end_rgb, t)
            draw.line([(0, y), (w, y)], fill=(*color, 255))
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")

    return grad


def render_gradient_text(
    text: str = "Cricket AI",
    font_path: str | None = None,
    font_size: int = 128,
    padding: int = 20,
    start_color: str = "#60a5fa",
    end_color: str = "#c084fc",
    direction: str = "horizontal",
) -> Image.Image:
    font = load_font(font_path, font_size)
    text_w, text_h, bbox = measure_text(text, font)

    # Calculate final image size with padding
    W = text_w + padding * 2
    H = text_h + padding * 2
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # Create gradient the size of text box only
    gradient = create_gradient((text_w, text_h), hex_to_rgb(start_color), hex_to_rgb(end_color), direction)

    # Create a mask for the text (white text on black background)
    mask = Image.new("L", (text_w, text_h), 0)
    mask_draw = ImageDraw.Draw(mask)

    # Note: textbbox might have negative offsets; we render on mask at (0,0) using anchor
    # Draw text onto mask at the origin; align by using baseline offset from bbox
    offset_x, offset_y = -bbox[0], -bbox[1]
    mask_draw.text((offset_x, offset_y), text, font=font, fill=255)

    # Paste gradient through the mask onto the main transparent image at (padding, padding)
    img.paste(gradient, (padding, padding), mask)
    return img


def render_fit_width(
    text: str,
    target_width: int,
    font_path: str | None = None,
    base_font_size: int = 128,
    padding: int = 20,
    start_color: str = "#60a5fa",
    end_color: str = "#c084fc",
    direction: str = "horizontal",
    supersample: int = 1,
) -> Image.Image:
    """Render text to fit a target width without post-resizing blur.

    Strategy:
    1) Estimate a font size so that (text width + 2*padding) ≈ target_width.
    2) Re-measure and refine once.
    3) Optionally render at (target_width * supersample), then downscale to target_width for crisp edges.
    """
    supersample = max(1, int(supersample))

    # First pass: measure with base font size
    font = load_font(font_path, base_font_size)
    text_w, _, _ = measure_text(text, font)
    inner_target = max(1, target_width - 2 * padding)
    if text_w <= 0:
        text_w = 1
    scale = inner_target / text_w
    est_font_size = max(1, int(round(base_font_size * scale)))

    # Second pass: refine with estimated font size
    font = load_font(font_path, est_font_size)
    text_w, text_h, bbox = measure_text(text, font)

    # Adjust once more if slightly off
    if text_w != inner_target and text_w > 0:
        est_font_size = max(1, int(round(est_font_size * (inner_target / text_w))))
        font = load_font(font_path, est_font_size)
        text_w, text_h, bbox = measure_text(text, font)

    W = text_w + 2 * padding
    H = text_h + 2 * padding

    # If supersampling, render larger and then downscale
    render_W = W * supersample
    render_H = H * supersample

    # Create gradient and mask at render scale
    gradient = create_gradient((text_w * supersample, text_h * supersample), hex_to_rgb(start_color), hex_to_rgb(end_color), direction)
    mask = Image.new("L", (text_w * supersample, text_h * supersample), 0)
    mask_draw = ImageDraw.Draw(mask)

    # For supersampled text, we need a correspondingly larger font
    ss_font = load_font(font_path, est_font_size * supersample)
    offset_x, offset_y = -bbox[0] * supersample, -bbox[1] * supersample
    mask_draw.text((offset_x, offset_y), text, font=ss_font, fill=255)

    render_img = Image.new("RGBA", (render_W, render_H), (0, 0, 0, 0))
    render_img.paste(gradient, (padding * supersample, padding * supersample), mask)

    # Downscale to exact target width if supersampled
    if supersample > 1:
        target_H = max(1, round(render_H / supersample))
        render_img = render_img.resize((W, target_H), resample=Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

    return render_img


def main():
    parser = argparse.ArgumentParser(description="Generate a transparent PNG with gradient text.")
    parser.add_argument("--text", default="Cricket AI", help="Text content to render")
    parser.add_argument("--font", default=None, help="Path to a .ttf/.otf font file")
    parser.add_argument("--font-size", type=int, default=12800, help="Font size in points")
    parser.add_argument("--padding", type=int, default=20, help="Padding around text in pixels")
    parser.add_argument("--start", default="#60a5fa", help="Start hex color (e.g. #60a5fa)")
    parser.add_argument("--end", default="#c084fc", help="End hex color (e.g. #c084fc)")
    parser.add_argument("--direction", choices=["horizontal", "vertical"], default="horizontal", help="Gradient direction")
    parser.add_argument("--width", type=int, default=None, help="Fit text to this final image width (avoids blur)")
    parser.add_argument("--supersample", type=int, default=2, help="Render at N× size and downsample for sharper edges")
    parser.add_argument("--out", default="scripts/cricket_ai.png", help="Output PNG path")

    args = parser.parse_args()

    if args.width is not None and args.width > 0:
        img = render_fit_width(
            text=args.text,
            target_width=args.width,
            font_path=args.font,
            base_font_size=args.font_size,
            padding=args.padding,
            start_color=args.start,
            end_color=args.end,
            direction=args.direction,
            supersample=args.supersample,
        )
    else:
        img = render_gradient_text(
            text=args.text,
            font_path=args.font,
            font_size=args.font_size,
            padding=args.padding,
            start_color=args.start,
            end_color=args.end,
            direction=args.direction,
        )

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    img.save(args.out, format="PNG")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
