import os
import base64
import io
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

# 允许以 "python -m uvicorn src.deploy.server:app" 方式运行
try:
    from src.deploy.inference import CaptchaPredictor
except ImportError:
    # 直接脚本运行的兜底导入
    from src.deploy.inference import CaptchaPredictor  # type: ignore


class PredictRequest(BaseModel):
    image_base64: str


def load_config(config_path: str = "config/config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def decode_base64_image(b64: str) -> Image.Image:
    # 去掉 data URL 头
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64, validate=True)
    except Exception:
        # 某些客户端可能不使用严格 base64，尝试宽松解码
        img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# 初始化应用与模型
app = FastAPI(title="Captcha OCR Service", version="1.0.0")
_config = load_config()
_model_path = os.getenv("MODEL_PATH", os.getenv("CAPTCHA_ONNX_PATH", "checkpoints/best_model.onnx"))

_predictor: CaptchaPredictor | None = None


@app.on_event("startup")
def _startup():
    global _predictor
    if not os.path.exists(_model_path):
        raise RuntimeError(f"模型文件不存在: {_model_path}. 请设置环境变量 MODEL_PATH 指向 ONNX 模型路径")
    _predictor = CaptchaPredictor(_model_path, _config)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if _predictor is None:
        raise HTTPException(status_code=500, detail="模型未初始化")
    try:
        image = decode_base64_image(req.image_base64)
        pred = _predictor.predict(image)
        code = "".join(pred)
        return {"code": code}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"推理失败: {e}")


# 便于 `python -m uvicorn src.deploy.server:app --reload` 运行
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
