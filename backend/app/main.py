import asyncio
import base64
import io
import logging
import os
import shutil
import threading
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


def _preload_models() -> None:
    """后台预加载常用模型。可通过 DISABLE_PRELOAD=1 禁用，避免免费档内存不足导致启动失败。"""
    try:
        logger.info("预加载 person_fast (mobilenetv3) …")
        get_rvm_model("mobilenetv3")
        logger.info("person_fast 预加载完成")
    except Exception as e:
        logger.warning("预加载 RVM 失败: %s，首次预览将较慢", e)
    try:
        if U2NET_PATH.exists():
            logger.info("预加载 U2-Net …")
            get_u2net()
            logger.info("U2-Net 预加载完成")
    except Exception as e:
        logger.warning("预加载 U2-Net 失败: %s", e)


@app.on_event("startup")
def startup_preload():
    """可选预加载。DISABLE_PRELOAD=1 时跳过，确保服务快速启动（首次请求会较慢）。"""
    if os.environ.get("DISABLE_PRELOAD", "").strip() == "1":
        logger.info("DISABLE_PRELOAD=1，跳过模型预加载")
        return
    threading.Thread(target=_preload_models, daemon=True).start()
_cors_raw = os.environ.get("CORS_ORIGINS", "*")
_cors_origins = ["*"] if _cors_raw.strip() == "*" else [
    x.strip() for x in _cors_raw.split(",") if x.strip()
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
torch.set_num_threads(min(4, os.cpu_count() or 4))
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
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = min(4, (os.cpu_count() or 4))
    sess = ort.InferenceSession(
        str(path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
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


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess_for_onnx(frame_rgb: np.ndarray, size: int) -> np.ndarray:
    img = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    img = np.ascontiguousarray(img).astype(np.float32)
    max_val = float(img.max())
    if max_val > 0:
        np.divide(img, max_val, out=img)
    np.subtract(img, _MEAN, out=img)
    np.divide(img, _STD, out=img)
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


def predict_alpha_fast(frame_rgb: np.ndarray, max_side: int = 720) -> np.ndarray:
    """U2-Net only (320x320) — fast mode for general objects."""
    scaled, orig_h, orig_w = _downscale_for_inference(frame_rgb, max_side)
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


def predict_alpha_hq(frame_rgb: np.ndarray, max_side: int = 720) -> np.ndarray:
    """Dual-model fusion (U2-Net + IS-Net in parallel) — high quality mode."""
    scaled, orig_h, orig_w = _downscale_for_inference(frame_rgb, max_side)
    u2 = get_u2net()
    isnet = get_isnet()

    if u2 is not None and isnet is not None:
        fut_u2 = _onnx_pool.submit(_run_onnx_model, u2, scaled, 320)
        fut_is = _onnx_pool.submit(_run_onnx_model, isnet, scaled, 1024)
        alpha = np.maximum(fut_u2.result(), fut_is.result())
    elif u2 is not None:
        alpha = _run_onnx_model(u2, scaled, 320)
    elif isnet is not None:
        alpha = _run_onnx_model(isnet, scaled, 1024)
    else:
        raise RuntimeError("没有可用的通用抠像模型")
    if scaled.shape[:2] != (orig_h, orig_w):
        alpha = cv2.resize(alpha, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0.0, 1.0)


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
    """Save RGBA as PNG，compress_level=1 优先速度。"""
    img = Image.fromarray(rgba, "RGBA")
    img.save(str(path), format="PNG", compress_level=1)


def save_frames_to_zip(frames_dir: Path, zip_path: Path) -> Path:
    with ZipFile(str(zip_path), "w", compression=ZIP_DEFLATED, compresslevel=1) as zf:
        for f in sorted(frames_dir.glob("*.png")):
            zf.write(str(f), arcname=f.name)
    logger.info("ZIP: %s (%.2f MB)", zip_path.name, zip_path.stat().st_size / 1048576)
    shutil.rmtree(str(frames_dir), ignore_errors=True)
    return zip_path


# ---------------------------------------------------------------------------
# Single-frame matting
# ---------------------------------------------------------------------------
def matting_frame_general(frame_rgb: np.ndarray, hq: bool = False, max_side: int = 720) -> np.ndarray:
    alpha = predict_alpha_hq(frame_rgb, max_side) if hq else predict_alpha_fast(frame_rgb, max_side)
    return np.dstack([frame_rgb, (alpha * 255).astype(np.uint8)])


def matting_frame_person(
    frame_rgb: np.ndarray, model: torch.nn.Module,
    rec: List[Optional[torch.Tensor]], ref: int,
    max_side: int = 0,
) -> Tuple[np.ndarray, List[Optional[torch.Tensor]]]:
    if max_side > 0 and max(frame_rgb.shape[:2]) > max_side:
        scaled, _, _ = _downscale_for_inference(frame_rgb, max_side)
        frame_rgb = scaled
    h, w = frame_rgb.shape[:2]
    src = (torch.from_numpy(frame_rgb).float().div_(255.0)
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
    bg = np.dstack([board, board, board])
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return (rgba[:, :, :3].astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)).clip(0, 255).astype(np.uint8)


def img_to_b64(rgb: np.ndarray, fmt: str = "JPEG", q: int = 80) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format=fmt, quality=q, optimize=False)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Video batch processing
# ---------------------------------------------------------------------------
def process_person(path: Path, wd: Path, variant: str, ref: int, max_side: int = 720) -> Path:
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
        rgba, rec = matting_frame_person(rgb, model, rec, ref, max_side=max_side)
        save_rgba_png(rgba, fd / f"frame_{idx:06d}.png")
        idx += 1
        if idx % 30 == 0 or idx == 1:
            logger.info("%d / %d", idx, total)
    cap.release()
    if idx == 0:
        raise RuntimeError("未读取到帧")
    return save_frames_to_zip(fd, wd / "output_sequence.zip")


def process_general(path: Path, wd: Path, hq: bool = False, max_side: int = 720) -> Path:
    wd.mkdir(parents=True, exist_ok=True)
    fd = wd / "frames"; fd.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("无法打开视频")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idx = 0
    frame_idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if frame_idx % _FRAME_SKIP == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgba = matting_frame_general(rgb, hq=hq, max_side=max_side)
            save_rgba_png(rgba, fd / f"frame_{idx:06d}.png")
            idx += 1
            if idx % 20 == 0 or idx == 1:
                logger.info("%d / %d", idx, total)
        frame_idx += 1
    cap.release()
    if idx == 0:
        raise RuntimeError("未读取到帧")
    return save_frames_to_zip(fd, wd / "output_sequence.zip")


_VIDEO_MAX_SIDE = int(os.environ.get("MATTING_VIDEO_MAX_SIDE", "720"))
_FRAME_SKIP = max(1, int(os.environ.get("MATTING_FRAME_SKIP", "1")))  # 2=隔帧处理，约 2x 速度


def process_video(path: Path, wd: Path, kind: ModelKind) -> Path:
    ms = _VIDEO_MAX_SIDE
    if kind == "person_fast":
        return process_person(path, wd, "mobilenetv3", 384, max_side=ms)
    if kind == "person_quality":
        return process_person(path, wd, "resnet50", 512, max_side=ms)
    if kind == "general_object_hq":
        return process_general(path, wd, hq=True, max_side=ms)
    return process_general(path, wd, hq=False, max_side=ms)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "Jing's Video Matting Studio"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/warmup")
def warmup():
    """页面加载时调用，提前加载模型，减少首次预览等待。"""
    if "mobilenetv3" not in _rvm_models:
        threading.Thread(target=lambda: get_rvm_model("mobilenetv3"), daemon=True).start()
    if _u2net_session is None and U2NET_PATH.exists():
        threading.Thread(target=get_u2net, daemon=True).start()
    if _isnet_session is None and ISNET_PATH.exists():
        threading.Thread(target=get_isnet, daemon=True).start()
    return {"status": "ok", "person_fast_ready": "mobilenetv3" in _rvm_models}


def _do_preview_frame(rgb: np.ndarray, model_kind: ModelKind, max_side: int = 480) -> Tuple[np.ndarray, np.ndarray]:
    """同步执行预览抠像，供线程池调用。预览用较小分辨率加速。"""
    if model_kind in ("person_fast", "person_quality"):
        v = "mobilenetv3" if model_kind == "person_fast" else "resnet50"
        r = 384 if model_kind == "person_fast" else 512
        rgba, _ = matting_frame_person(rgb, get_rvm_model(v), [None] * 4, r, max_side=max_side)
    else:
        hq = model_kind == "general_object_hq"
        rgba = matting_frame_general(rgb, hq=hq, max_side=max_side)
    return rgb, rgba


@app.post("/api/preview-frame")
async def preview_frame_endpoint(file: UploadFile = File(...), model_kind: ModelKind = Form("person_fast")):
    """预览单帧：接收客户端提取的 JPEG 帧，无需上传整段视频，响应更快。"""
    if not file.filename:
        raise HTTPException(400, "未选择文件")
    contents = await file.read()
    arr = np.frombuffer(contents, dtype=np.uint8)
    rgb = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if rgb is None:
        raise HTTPException(400, "无法解析图片，请确保为 JPEG/PNG")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    max_side = 360 if model_kind == "general_object_hq" else 480
    try:
        rgb_out, rgba = await asyncio.to_thread(_do_preview_frame, rgb, model_kind, max_side)
    except RuntimeError as e:
        if "没有可用的通用抠像模型" in str(e):
            raise HTTPException(503, "通用抠像模型未就绪，请稍候重试")
        raise

    return JSONResponse({
        "original": f"data:image/jpeg;base64,{img_to_b64(rgb_out)}",
        "preview": f"data:image/jpeg;base64,{img_to_b64(render_checkerboard(rgba))}",
        "frame_index": 0,
        "total_frames": 1,
    })


@app.post("/api/preview")
async def preview_endpoint(file: UploadFile = File(...), model_kind: ModelKind = Form("person_fast")):
    """兼容旧版：上传整段视频提取中间帧预览。建议前端改用 /api/preview-frame 传单帧以加速。"""
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

    try:
        tmp.unlink()
    except OSError:
        pass

    max_side = 360 if model_kind == "general_object_hq" else 480
    try:
        rgb_out, rgba = await asyncio.to_thread(_do_preview_frame, rgb, model_kind, max_side)
    except RuntimeError as e:
        if "没有可用的通用抠像模型" in str(e):
            raise HTTPException(503, "通用抠像模型未就绪，请稍候重试")
        raise

    return JSONResponse({
        "original": f"data:image/jpeg;base64,{img_to_b64(rgb_out)}",
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
    with open(str(up), "wb") as fp:
        while chunk := await file.read(1024 * 1024):
            fp.write(chunk)
    size_mb = up.stat().st_size / 1048576
    logger.info("保存: %s (%.1f MB) model=%s", up.name, size_mb, model_kind)
    try:
        zp = await asyncio.to_thread(process_video, up, RESULT_DIR / jid, model_kind)
    except Exception as exc:
        logger.exception("处理失败")
        raise HTTPException(500, str(exc)) from exc
    return FileResponse(path=str(zp), media_type="application/zip", filename="output_sequence.zip")
