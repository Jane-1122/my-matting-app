# Jing's Video Matting Studio - Render 免费部署指南

按以下步骤在 Render 上免费部署，每月约 750 小时额度。

---

## 第一步：打开 Render 并连接仓库

1. 打开 **https://dashboard.render.com**
2. 登录或注册（支持 GitHub 一键登录）
3. 点击 **New** → **Blueprint**
4. 选择 **Build and deploy from a Git repository**
5. 连接 GitHub，选择仓库 **Jane-1122/my-matting-app**
6. 分支选择 **main**
7. 点击 **Apply**

---

## 第二步：等待首次构建

- Render 会自动识别根目录的 `render.yaml`，创建 **matting-backend** 和 **matting-frontend** 两个服务
- 首次构建约 **15–25 分钟**（需下载 PyTorch、ONNX 模型等）
- 可在 Dashboard 查看构建日志

---

## 第三步：配置环境变量

构建完成后，需要配置两个服务的环境变量：

### 1. 获取后端地址

- 进入 **matting-backend** 服务
- 在顶部复制其 **URL**（如 `https://matting-backend-xxxx.onrender.com`）

### 2. 配置 matting-frontend

- 进入 **matting-frontend** → **Environment**
- 添加变量：
  - **Key**: `NEXT_PUBLIC_API_URL`
  - **Value**: `https://matting-backend-xxxx.onrender.com/api`（替换为你的后端实际 URL）
- 保存后点击 **Manual Deploy** → **Deploy latest commit**

### 3. 配置 matting-backend（可选）

- 进入 **matting-backend** → **Environment**
- `CORS_ORIGINS` 可留空或填 `*`（代码已默认 `*`，一般无需修改）

---

## 第四步：使用

- 前端 URL 如：`https://matting-frontend-xxxx.onrender.com`
- 直接访问即可使用
- 可分享该链接给他人

---

## 注意事项

| 项目 | 说明 |
|------|------|
| **冷启动** | 15 分钟无访问会休眠，首次访问需等待 30–60 秒唤醒 |
| **免费额度** | 每月 750 小时，两个服务共享，约 375 小时/服务 |
| **视频大小** | 建议控制在 100MB 以内 |

---

## 快速链接

- Render 控制台：https://dashboard.render.com
- 本仓库：https://github.com/Jane-1122/my-matting-app
