@echo off
chcp 65001 >nul
title 停止 Matting Studio

echo 正在停止所有服务...

:: 关闭 uvicorn
taskkill /FI "WINDOWTITLE eq Matting-Backend*" /F >nul 2>&1
:: 关闭 node/next
taskkill /FI "WINDOWTITLE eq Matting-Frontend*" /F >nul 2>&1

echo 已停止。
timeout /t 2 /nobreak >nul
