@echo off
chcp 65001 >nul
title Jing's Video Matting Studio (Docker)

echo ============================================
echo   Jing's Video Matting Studio
echo   Docker 模式启动中...
echo ============================================
echo.

:: 检查 Docker 是否可用
docker --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Docker，请先安装 Docker Desktop。
    echo.
    echo 下载地址: https://www.docker.com/products/docker-desktop/
    echo 安装后请重启电脑，再运行本脚本。
    echo.
    pause
    exit /b 1
)

cd /d "%~dp0"

echo [1/2] 正在启动 Docker 容器...
docker compose up -d
if errorlevel 1 (
    echo.
    echo [错误] 启动失败，请检查 Docker Desktop 是否已运行。
    pause
    exit /b 1
)

echo.
echo [2/2] 等待服务就绪...
timeout /t 8 /nobreak >nul

echo.
echo ============================================
echo   启动完成！
echo.
echo   访问地址: http://localhost
echo.
echo   关闭方法: 双击 停止网站-Docker.bat
echo ============================================
echo.

:: 自动打开浏览器
start http://localhost

pause
