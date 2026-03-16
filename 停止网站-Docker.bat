@echo off
chcp 65001 >nul
title 停止 Jing's Video Matting Studio (Docker)

echo ============================================
echo   正在停止 Jing's Video Matting Studio...
echo ============================================
echo.

cd /d "%~dp0"

docker compose down

echo.
echo 已停止。
timeout /t 2 /nobreak >nul
