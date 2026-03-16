import base64
import io
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

logger = logging.getLogger("matting")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

app = FastAPI(title="Jing's Video Matting Studio", version="0.9.0")
_cors_origins = [
    x.strip() for x in os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if x.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
TEMP_DIR = DATA_DIR / "temp"
MODELS_DIR = BASE_DIR / "models"
for _d in (DATA_DIR, UPLOAD_DIR, RESULT_DIR, TEMP_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
logger.info("推理设备: %s", DEVICE)

ModelKind = Literal["person_fast", "person_quality", "general_object", "general_object_hq"]

# ---------------------------------------------------------------------------
# RVM（人物抠像）
# ---------------------------------------------------------------------------
_rvm_models: Dict[str, torch.nn.Module] = {}


def get_rvm_model(variant: str) -> torch.nn.Module:
    if variant in _rvm_models:
        return _rvm_models[variant]
    logger.info("加载 RVM %s …", variant)
    model = torch.hub.load("PeterL1n/RobustVideoMatting", variant, trust_repo=True)
    model = model.eval().to(DEVICE)
    _rvm_models[variant] = model
    logger.info("RVM %s 就绪", variant)
    return model


# ---------------------------------------------------------------------------
# ONNX 显著目标检测模型
# ---------------------------------------------------------------------------
U2NET_PATH = MODELS_DIR / "u2net.onnx"
ISNET_PATH = MODELS_DIR / "isnet-general-use.onnx"

_u2net_session: Optional[ort.InferenceSession] = None
_isnet_session: Optional[ort.InferenceSession] = None
_onnx_pool = ThreadPoolExecutor(max_workers=2)


def _load_onnx(path: Path) -> ort.InferenceSession:
    logger.info("加载 ONNX: %s", path.name)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    logger.info("  %s 就绪", path.name)
    return sess


def get_u2net() -> Optional[ort.InferenceSession]:
    global _u2net_session
    if _u2net_session is not None:
        return _u2net_session
    if not U2NET_PATH.exists():
        return None
    _u2net_session = _load_onnx(U2NET_PATH)
    return _u2net_session


def get_isnet() -> Optional[ort.InferenceSession]:
    global _isnet_session
    if _isnet_session is not None:
        return _isnet_session
    if not ISNET_PATH.exists():
        return None
    _isnet_session = _load_onnx(ISNET_PATH)
    return _isnet_session


def _preprocess_for_onnx(frame_rgb: np.ndarray, size: int) -> np.ndarray:
    img = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_LANCZOS4)
    img = img.astype(np.float32)
    max_val = img.max()
    if max_val > 0:
        img = img / max_val
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


def _postprocess_alpha(raw: np.ndarray) -> np.ndarray:
    if raw.min() < -0.5 or raw.max() > 1.5:
        raw = 1.0 / (1.0 + np.exp(-np.clip(raw, -20, 20)))
    ma, mi = raw.max(), raw.min()
    if ma - mi > 1e-6:
        raw = (raw - mi) / (ma - mi)
    else:
        raw = np.zeros_like(raw)
    return np.clip(raw, 0.0, 1.0).astype(np.float32)


def _downscale_for_inference(frame_rgb: np.ndarray, max_side: int = 720) -> Tuple[np.ndarray, int, int]:
    """Downscale frame if larger than max_side, return (scaled_frame, orig_h, orig_w)."""
    h, w = frame_rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame_rgb, h, w
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA), h, w


def _run_onnx_model(sess: ort.InferenceSession, frame_rgb: np.ndarray, model_size: int) -> np.ndarray:
    orig_h, orig_w = frame_rgb.shape[:2]
    inp = _preprocess_for_onnx(frame_rgb, model_size)
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: inp})
    pred = outputs[0][0, 0]
    alpha = _postprocess_alpha(pred)
    alpha = cv2.resize(alpha, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0.0, 1.0)


def predict_alpha_fast(frame_rgb: np.ndarray) -> np.ndarray:
    """U2-Net only (320x320) — fast mode for general objects."""
    scaled, orig_h, orig_w = _downscale_for_inference(frame_rgb, 720)
    u2 = get_u2net()
    if u2 is None:
        isnet = get_isnet()
        if isnet is None:
            raise RuntimeError("没有可用的通用抠像模型")
        alpha = _run_onnx_model(isnet, scaled, 1024)
    else:
        alpha = _run_onnx_model(u2, scaled, 320)
    if scaled.shape[:2] != (orig_h, orig_w):
        alpha = cv2.resize(alpha, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return alpha


def predict_alpha_hq(frame_rgb: np.ndarray) -> np.ndarray:
    """Dual-model fusion (U2-Net + IS-Net in parallel) — high quality mode."""
    u2 = get_u2net()
    isnet = get_isnet()

    if u2 is not None and isnet is not None:
        fut_u2 = _onnx_pool.submit(_run_onnx_model, u2, frame_rgb, 320)
        fut_is = _onnx_pool.submit(_run_onnx_model, isnet, frame_rgb, 1024)
        return np.maximum(fut_u2.result(), fut_is.result())

    if u2 is not None:
        return _run_onnx_model(u2, frame_rgb, 320)
    if isnet is not None:
        return _run_onnx_model(isnet, frame_rgb, 1024)
    raise RuntimeError("没有可用的通用抠像模型")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tensor_to_numpy_uint8(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().clamp(0.0, 1.0).mul(255).to(torch.uint8)
    try:
        return t.numpy()
    except Exception:
        return np.array(t.tolist(), dtype=np.uint8)


def compute_downsample_ratio(h: int, w: int, ref: int = 512) -> float:
    longest = max(h, w)
    if longest <= 0:
        return 1.0
    return min(max(ref / float(longest), 0.1), 1.0)


def save_rgba_png(rgba: np.ndarray, path: Path) -> None:
    """Save RGBA as PNG with reduced compression for speed."""
    img = Image.fromarray(rgba, "RGBA")
    img.save(str(path), format="PNG", compress_level=3)


def save_frames_to_zip(frames_dir: Path, zip_path: Path) -> Path:
    with ZipFile(str(zip_path), "w", compression=ZIP_DEFLATED) as zf:
        for f in sorted(frames_dir.glob("*.png")):
            zf.write(str(f), arcname=f.name)
    logger.info("ZIP: %s (%.2f MB)", zip_path.name, zip_path.stat().st_size / 1048576)
    shutil.rmtree(str(frames_dir), ignore_errors=True)
    return zip_path


# ---------------------------------------------------------------------------
# Single-frame matting
# ---------------------------------------------------------------------------
def matting_frame_general(frame_rgb: np.ndarray, hq: bool = False) -> np.ndarray:
    alpha = predict_alpha_hq(frame_rgb) if hq else predict_alpha_fast(frame_rgb)
    return np.dstack([frame_rgb, (alpha * 255).astype(np.uint8)])


def matting_frame_person(
    frame_rgb: np.ndarray, model: torch.nn.Module,
    rec: List[Optional[torch.Tensor]], ref: int,
) -> Tuple[np.ndarray, List[Optional[torch.Tensor]]]:
    h, w = frame_rgb.shape[:2]
    src = (torch.from_numpy(frame_rgb.copy()).float().div_(255.0)
           .permute(2, 0, 1).unsqueeze(0).to(DEVICE))
    ratio = compute_downsample_ratio(h, w, ref)
    with torch.no_grad():
        fgr, pha, *new_rec = model(src, *rec, downsample_ratio=ratio)
    fgr_np = tensor_to_numpy_uint8(fgr[0].permute(1, 2, 0))
    pha_np = tensor_to_numpy_uint8(pha[0, 0])
    return np.dstack([fgr_np, pha_np]), list(new_rec)


# ---------------------------------------------------------------------------
# Preview rendering
# ---------------------------------------------------------------------------
def render_checkerboard(rgba: np.ndarray, cell: int = 16) -> np.ndarray:
    h, w = rgba.shape[:2]
    ys = np.arange(h)[:, None] // cell
    xs = np.arange(w)[None, :] // cell
    board = np.where((ys + xs) % 2 == 0, 220, 255).astype(np.uint8)
    bg = np.dstack([board, board, board]).astype(np.float32)
    rgb = rgba[:, :, :3].astype(np.float32)
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return np.clip(rgb * a + bg * (1.0 - a), 0, 255).astype(np.uint8)


def img_to_b64(rgb: np.ndarray, fmt: str = "JPEG", q: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format=fmt, quality=q)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Video batch processing
# ---------------------------------------------------------------------------
def process_person(path: Path, wd: Path, variant: str, ref: int) -> Path:
    model = get_rvm_model(variant)
    wd.mkdir(parents=True, exist_ok=True)
    fd = wd / "frames"; fd.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("无法打开视频")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    rec: List[Optional[torch.Tensor]] = [None] * 4
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgba, rec = matting_frame_person(rgb, model, rec, ref)
        save_rgba_png(rgba, fd / f"frame_{idx:06d}.png")
        idx += 1
        if idx % 30 == 0 or idx == 1:
            logger.info("%d / %d", idx, total)
    cap.release()
    if idx == 0:
        raise RuntimeError("未读取到帧")
    return save_frames_to_zip(fd, wd / "output_sequence.zip")


def process_general(path: Path, wd: Path, hq: bool = False) -> Path:
    wd.mkdir(parents=True, exist_ok=True)
    fd = wd / "frames"; fd.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("无法打开视频")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgba = matting_frame_general(rgb, hq=hq)
        save_rgba_png(rgba, fd / f"frame_{idx:06d}.png")
        idx += 1
        if idx % 20 == 0 or idx == 1:
            logger.info("%d / %d", idx, total)
    cap.release()
    if idx == 0:
        raise RuntimeError("未读取到帧")
    return save_frames_to_zip(fd, wd / "output_sequence.zip")


def process_video(path: Path, wd: Path, kind: ModelKind) -> Path:
    if kind == "person_fast":
        return process_person(path, wd, "mobilenetv3", 384)
    if kind == "person_quality":
        return process_person(path, wd, "resnet50", 512)
    if kind == "general_object_hq":
        return process_general(path, wd, hq=True)
    return process_general(path, wd, hq=False)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
@app.post("/api/preview")
async def preview_endpoint(file: UploadFile = File(...), model_kind: ModelKind = Form("person_fast")):
    if not file.filename:
        raise HTTPException(400, "未选择文件")
    suffix = Path(file.filename).suffix or ".mp4"
    tmp = UPLOAD_DIR / f"prev_{uuid4().hex}{suffix}"
    contents = await file.read()
    with open(str(tmp), "wb") as fp:
        fp.write(contents)

    cap = cv2.VideoCapture(str(tmp))
    if not cap.isOpened():
        raise HTTPException(500, "无法打开视频")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    mid = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, bgr = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, bgr = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(500, "无法读取帧")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if model_kind in ("person_fast", "person_quality"):
        v = "mobilenetv3" if model_kind == "person_fast" else "resnet50"
        r = 384 if model_kind == "person_fast" else 512
        rgba, _ = matting_frame_person(rgb, get_rvm_model(v), [None] * 4, r)
    else:
        hq = model_kind == "general_object_hq"
        rgba = matting_frame_general(rgb, hq=hq)

    try:
        tmp.unlink()
    except OSError:
        pass

    return JSONResponse({
        "original": f"data:image/jpeg;base64,{img_to_b64(rgb)}",
        "preview": f"data:image/jpeg;base64,{img_to_b64(render_checkerboard(rgba))}",
        "frame_index": mid,
        "total_frames": total,
    })


@app.post("/api/upload")
async def upload_endpoint(file: UploadFile = File(...), model_kind: ModelKind = Form("person_fast")):
    if not file.filename:
        raise HTTPException(400, "未选择文件")
    suffix = Path(file.filename).suffix or ".mp4"
    jid = uuid4().hex
    up = UPLOAD_DIR / f"{jid}{suffix}"
    contents = await file.read()
    with open(str(up), "wb") as fp:
        fp.write(contents)
    logger.info("保存: %s (%.1f MB) model=%s", up.name, len(contents) / 1048576, model_kind)
    try:
        zp = process_video(up, RESULT_DIR / jid, model_kind)
    except Exception as exc:
        logger.exception("处理失败")
        raise HTTPException(500, str(exc)) from exc
    return FileResponse(path=str(zp), media_type="application/zip", filename="output_sequence.zip")
