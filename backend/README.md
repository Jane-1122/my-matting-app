## Backend - FastAPI Video Matting

FastAPI 后端，负责接收视频、调度 RVM（或类似模型）进行抠像，并输出透明背景 PNG 序列帧的 ZIP 包。

### 主要接口设计（占位实现）

- `POST /api/jobs`：上传视频，创建一个抠像任务，返回 `job_id`
- `GET /api/jobs/{job_id}/status`：轮询任务状态，返回 `status` / `progress`
- `GET /api/jobs/{job_id}/result`：任务完成后下载 PNG 序列帧 ZIP 包

当前 `app/main.py` 中仅为占位逻辑：把上传文件保存到 `data/uploads`，并生成一个空 ZIP 作为结果，方便前后端联调。后续可在 `create_job` 内接入 RVM 推理。

### 本地启动

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

