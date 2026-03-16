@echo off
chcp 65001 >nul
title Jing's Video Matting Studio

echo ============================================
echo   Jing's Video Matting Studio
echo   正在启动，请稍候...
echo ============================================
echo.

cd /d "%~dp0"

:: 启动后端
echo [1/2] 启动后端服务 (FastAPI)...
start "Matting-Backend" cmd /k "cd /d "%~dp0" && python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"

:: 等待后端启动
timeout /t 3 /nobreak >nul

:: 启动前端
echo [2/2] 启动前端服务 (Next.js)...
start "Matting-Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

:: 等待前端启动
timeout /t 5 /nobreak >nul

echo.
echo ============================================
echo   启动完成！
echo.
echo   浏览器打开: http://localhost:3000
echo.
echo   关闭方法: 关掉弹出的两个黑色窗口即可
echo ============================================
echo.

:: 自动打开浏览器
start http://localhost:3000

pause
