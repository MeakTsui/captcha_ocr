import onnxruntime as ort
import numpy as np
from PIL import Image

class CaptchaPredictor:
    def __init__(self, model_path, config):
        # 优化的ONNX运行时设置
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = config['inference']['num_threads']
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 如果CPU支持AVX2/AVX512，启用MKL-DNN
        if config['inference']['use_mkldnn']:
            sess_options.enable_cpu_mem_arena = True
            sess_options.enable_mem_pattern = True
            
        self.session = ort.InferenceSession(
            model_path, 
            sess_options,
            providers=['CPUExecutionProvider']
        )
        self.config = config
        
    def preprocess(self, image):
        # 图像预处理：与 eval_checkpoints.py 一致，仅 resize + 标准化
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        else:
            # 确保是 RGB
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image))
            if image.mode != 'RGB':
                image = image.convert('RGB')

        # 训练时 torchvision.Resize 接收 [H, W]，而 PIL.resize 需要 (W, H)
        h, w = self.config['data']['image_size']
        processed_image = image.resize((w, h))

        arr = np.array(processed_image).transpose(2, 0, 1).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = arr[np.newaxis, :]
        # 可选断言：确保尺寸为 (1, C, H, W)
        # assert arr.shape[2:] == (h, w), f"preprocess got shape {arr.shape}, expected (H,W)=({h},{w})"
        return arr
        
    def predict(self, image):
        # 预处理
        input_data = self.preprocess(image)
        
        # 推理
        output = self.session.run(
            None, 
            {'input': input_data}
        )[0]
        
        # 后处理
        pred_chars = []
        for i in range(self.config['captcha']['length']):
            idx = output[0, i].argmax()
            pred_chars.append(self.config['captcha']['charset'][idx])
            
        return ''.join(pred_chars) 