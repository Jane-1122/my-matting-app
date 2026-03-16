## Frontend - Next.js + Tailwind

Next.js App Router + Tailwind CSS 前端，提供视频上传、实时进度展示、结果下载的极简现代界面。

### 主要页面

- `app/page.tsx`：单页界面，包含：
  - 视频上传区域（拖拽/点击）
  - 上传和处理进度条
  - 错误提示
  - 结果 ZIP 下载按钮（后端返回后显示）

前端默认请求的后端地址为：

- 创建任务：`POST http://127.0.0.1:8000/api/jobs`
- 查询状态：`GET  http://127.0.0.1:8000/api/jobs/:id/status`
- 获取结果：`GET  http://127.0.0.1:8000/api/jobs/:id/result`

### 本地启动

```bash
cd frontend
npm install   # 或 pnpm install / yarn
npm run dev   # 默认 http://localhost:3000
```

