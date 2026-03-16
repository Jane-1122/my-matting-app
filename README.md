## 精细视频抠像网站（Next.js + FastAPI）

本项目是一个前后端分离的精细视频抠像网站，用于上传视频、实时查看处理进度，并在完成后预览和下载“透明背景 PNG 序列帧 ZIP 包”。

- **前端技术栈**：Next.js（App Router）+ TypeScript + Tailwind CSS，极简现代设计风格
- **后端技术栈**：Python + FastAPI，集成 Robust Video Matting（RVM）或类似高精度抠像模型

### 目录结构概览

```text
my-matting-app/
  frontend/            # Next.js 前端
  backend/             # FastAPI 后端
  启动网站.bat          # 本地开发
  启动网站-Docker.bat   # Docker 启动
  云端部署说明.md       # 云端部署指南
```

### 运行方式

| 方式 | 说明 |
|------|------|
| 本地开发 | 双击 `启动网站.bat` → http://localhost:3000 |
| Docker | 双击 `启动网站-Docker.bat` → http://localhost |
| 云端部署 | 见 [云端部署说明.md](云端部署说明.md) |

