import tkinter as tk
from tkinter import filedialog, messagebox
from dataclasses import dataclass
import yaml
import cv2
import numpy as np
from PIL import Image, ImageTk
import os


@dataclass
class PreprocessConfig:
    use_otsu: bool = True
    invert_back: bool = True
    hough_remove_lines_enable: bool = False
    hough_threshold: int = 50
    hough_minLineLength: int = 30
    hough_maxLineGap: int = 10
    hough_thickness: int = 1
    morph_open_enable: bool = False
    morph_open_ksize: int = 2
    morph_open_iterations: int = 1
    morph_close_enable: bool = False
    morph_close_ksize: int = 3
    morph_close_iterations: int = 1
    cc_filter_enable: bool = False
    cc_min_area: int = 50
    cc_min_h: int = 10
    cc_min_w: int = 5

    def to_yaml_dict(self):
        return {
            'use_otsu': self.use_otsu,
            'invert_back': self.invert_back,
            'hough_remove_lines': {
                'enable': self.hough_remove_lines_enable,
                'threshold': int(self.hough_threshold),
                'minLineLength': int(self.hough_minLineLength),
                'maxLineGap': int(self.hough_maxLineGap),
                'thickness': int(self.hough_thickness),
            },
            'morph_open': {
                'enable': self.morph_open_enable,
                'ksize': int(self.morph_open_ksize),
                'iterations': int(self.morph_open_iterations),
            },
            'morph_close': {
                'enable': self.morph_close_enable,
                'ksize': int(self.morph_close_ksize),
                'iterations': int(self.morph_close_iterations),
            },
            'cc_filter': {
                'enable': self.cc_filter_enable,
                'min_area': int(self.cc_min_area),
                'min_h': int(self.cc_min_h),
                'min_w': int(self.cc_min_w),
            },
        }


def apply_preprocess(image_pil: Image.Image, cfg: PreprocessConfig) -> Image.Image:
    rgb = np.array(image_pil.convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if cfg.use_otsu:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    if cfg.hough_remove_lines_enable:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi/180,
            threshold=int(cfg.hough_threshold),
            minLineLength=int(cfg.hough_minLineLength),
            maxLineGap=int(cfg.hough_maxLineGap),
        )
        if lines is not None:
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                cv2.line(binary, (int(x1), int(y1)), (int(x2), int(y2)), 255, int(cfg.hough_thickness))

    if cfg.morph_open_enable:
        k = max(1, int(cfg.morph_open_ksize))
        it = max(1, int(cfg.morph_open_iterations))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=it)

    if cfg.morph_close_enable:
        k = max(1, int(cfg.morph_close_ksize))
        it = max(1, int(cfg.morph_close_iterations))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=it)

    if cfg.cc_filter_enable:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        mask = np.zeros_like(binary)
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if int(area) >= int(cfg.cc_min_area) and int(h) >= int(cfg.cc_min_h) and int(w) >= int(cfg.cc_min_w):
                mask[labels == i] = 255
        binary = mask

    if cfg.invert_back:
        processed = cv2.bitwise_not(binary)
    else:
        processed = binary

    return Image.fromarray(processed).convert('RGB')


class PreprocessTunerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('Captcha Preprocess Tuner')
        self.cfg = PreprocessConfig()
        self.image_path = None
        self.orig_img: Image.Image | None = None
        self.proc_img: Image.Image | None = None

        # UI
        self._build_ui()

    def _build_ui(self):
        # Left: controls
        ctrl = tk.Frame(self.root)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        # Buttons
        tk.Button(ctrl, text='加载图片', command=self.load_image).pack(fill=tk.X)
        tk.Button(ctrl, text='保存处理结果', command=self.save_processed).pack(fill=tk.X, pady=(4, 0))
        tk.Button(ctrl, text='导出YAML片段', command=self.export_yaml).pack(fill=tk.X, pady=(4, 8))

        # Otsu & invert
        self.var_use_otsu = tk.BooleanVar(value=self.cfg.use_otsu)
        tk.Checkbutton(ctrl, text='使用 Otsu', variable=self.var_use_otsu, command=self.update_and_refresh).pack(anchor='w')
        self.var_invert = tk.BooleanVar(value=self.cfg.invert_back)
        tk.Checkbutton(ctrl, text='反相回 RGB', variable=self.var_invert, command=self.update_and_refresh).pack(anchor='w')

        # Hough
        tk.Label(ctrl, text='Hough 去线').pack(anchor='w', pady=(8, 0))
        self.var_hough_enable = tk.BooleanVar(value=self.cfg.hough_remove_lines_enable)
        tk.Checkbutton(ctrl, text='启用', variable=self.var_hough_enable, command=self.update_and_refresh).pack(anchor='w')
        self.s_hough_thr = self._scale(ctrl, 'threshold', 1, 200, self.cfg.hough_threshold)
        self.s_hough_minlen = self._scale(ctrl, 'minLineLength', 5, 200, self.cfg.hough_minLineLength)
        self.s_hough_maxgap = self._scale(ctrl, 'maxLineGap', 0, 50, self.cfg.hough_maxLineGap)
        self.s_hough_thick = self._scale(ctrl, 'thickness', 1, 5, self.cfg.hough_thickness)

        # Morph open
        tk.Label(ctrl, text='形态学 Open').pack(anchor='w', pady=(8, 0))
        self.var_open_enable = tk.BooleanVar(value=self.cfg.morph_open_enable)
        tk.Checkbutton(ctrl, text='启用', variable=self.var_open_enable, command=self.update_and_refresh).pack(anchor='w')
        self.s_open_ksize = self._scale(ctrl, 'ksize', 1, 7, self.cfg.morph_open_ksize)
        self.s_open_iter = self._scale(ctrl, 'iterations', 1, 5, self.cfg.morph_open_iterations)

        # Morph close
        tk.Label(ctrl, text='形态学 Close').pack(anchor='w', pady=(8, 0))
        self.var_close_enable = tk.BooleanVar(value=self.cfg.morph_close_enable)
        tk.Checkbutton(ctrl, text='启用', variable=self.var_close_enable, command=self.update_and_refresh).pack(anchor='w')
        self.s_close_ksize = self._scale(ctrl, 'ksize', 1, 7, self.cfg.morph_close_ksize)
        self.s_close_iter = self._scale(ctrl, 'iterations', 1, 5, self.cfg.morph_close_iterations)

        # CC filter
        tk.Label(ctrl, text='连通域过滤').pack(anchor='w', pady=(8, 0))
        self.var_cc_enable = tk.BooleanVar(value=self.cfg.cc_filter_enable)
        tk.Checkbutton(ctrl, text='启用', variable=self.var_cc_enable, command=self.update_and_refresh).pack(anchor='w')
        self.s_cc_area = self._scale(ctrl, 'min_area', 0, 1000, self.cfg.cc_min_area)
        self.s_cc_h = self._scale(ctrl, 'min_h', 0, 200, self.cfg.cc_min_h)
        self.s_cc_w = self._scale(ctrl, 'min_w', 0, 200, self.cfg.cc_min_w)

        # Right: canvas
        right = tk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(right, bg='#333')
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _scale(self, parent, label, frm, to, init):
        tk.Label(parent, text=label).pack(anchor='w')
        var = tk.IntVar(value=int(init))
        scale = tk.Scale(parent, from_=frm, to=to, orient=tk.HORIZONTAL, variable=var, command=lambda e: self.update_and_refresh())
        scale.pack(fill=tk.X)
        return scale

    def load_image(self):
        # macOS Tk requires filetypes as list of (label, pattern) or (label, (patterns,...))
        path = filedialog.askopenfilename(
            title='选择验证码图片',
            filetypes=[
                ('Images', ('*.png', '*.jpg', '*.jpeg', '*.bmp')),
                ('PNG', '*.png'),
                ('JPEG', '*.jpg'),
                ('JPEG', '*.jpeg'),
                ('BMP', '*.bmp'),
                ('All Files', '*.*'),
            ]
        )
        if not path:
            return
        self.image_path = path
        self.orig_img = Image.open(path).convert('RGB')
        self.refresh_preview()

    def refresh_preview(self):
        if self.orig_img is None:
            return
        self._sync_cfg_from_ui()
        self.proc_img = apply_preprocess(self.orig_img, self.cfg)
        self._show_on_canvas(self.proc_img)

    def update_and_refresh(self):
        self.refresh_preview()

    def _sync_cfg_from_ui(self):
        self.cfg.use_otsu = self.var_use_otsu.get()
        self.cfg.invert_back = self.var_invert.get()
        self.cfg.hough_remove_lines_enable = self.var_hough_enable.get()
        self.cfg.hough_threshold = self.s_hough_thr.get()
        self.cfg.hough_minLineLength = self.s_hough_minlen.get()
        self.cfg.hough_maxLineGap = self.s_hough_maxgap.get()
        self.cfg.hough_thickness = self.s_hough_thick.get()
        self.cfg.morph_open_enable = self.var_open_enable.get()
        self.cfg.morph_open_ksize = self.s_open_ksize.get()
        self.cfg.morph_open_iterations = self.s_open_iter.get()
        self.cfg.morph_close_enable = self.var_close_enable.get()
        self.cfg.morph_close_ksize = self.s_close_ksize.get()
        self.cfg.morph_close_iterations = self.s_close_iter.get()
        self.cfg.cc_filter_enable = self.var_cc_enable.get()
        self.cfg.cc_min_area = self.s_cc_area.get()
        self.cfg.cc_min_h = self.s_cc_h.get()
        self.cfg.cc_min_w = self.s_cc_w.get()

    def _show_on_canvas(self, img_pil: Image.Image):
        # Fit to canvas size
        self.root.update_idletasks()
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 300
        img = img_pil.copy()
        img.thumbnail((cw, ch), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete('all')
        self.canvas.create_image(cw // 2, ch // 2, image=self.tk_img)

    def save_processed(self):
        if self.proc_img is None:
            messagebox.showwarning('提示', '请先加载图片并生成处理结果')
            return
        default_name = 'processed.png'
        if self.image_path:
            base = os.path.basename(self.image_path)
            name, _ = os.path.splitext(base)
            default_name = f'{name}_processed.png'
        path = filedialog.asksaveasfilename(defaultextension='.png', initialfile=default_name)
        if not path:
            return
        self.proc_img.save(path)
        messagebox.showinfo('成功', f'已保存: {path}')

    def export_yaml(self):
        cfg_yaml = yaml.safe_dump(self.cfg.to_yaml_dict(), sort_keys=False, allow_unicode=True)
        # Save to file
        path = filedialog.asksaveasfilename(defaultextension='.yaml', initialfile='preprocessing_snippet.yaml')
        if not path:
            return
        with open(path, 'w', encoding='utf-8') as f:
            f.write(cfg_yaml)
        messagebox.showinfo('成功', f'已导出 YAML 片段到: {path}')


def main():
    root = tk.Tk()
    app = PreprocessTunerApp(root)
    root.geometry('1100x520')
    root.mainloop()


if __name__ == '__main__':
    main()
