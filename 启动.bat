@echo off
setlocal
chcp 936 >nul 2>&1
title zhmem 中文信息检索

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

rem --- 1. 找 Python ---
set "PY="
where python >nul 2>&1 && set "PY=python"
if not defined PY (
  where py >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
  echo 没找到 Python。请先安装 Python 3.10 或更高版本：https://www.python.org/downloads/
  echo 安装时记得勾选 "Add Python to PATH"。
  pause
  exit /b 1
)

rem --- 2. 已在项目目录里就直接从源码运行，省去安装 ---
cd /d "%ROOT%"
%PY% -c "import zhmem" >nul 2>&1
if errorlevel 1 (
  echo 首次运行：正在安装依赖并注册 zhmem ...
  %PY% -m pip install -e . --disable-pip-version-check
  if errorlevel 1 (
    echo 安装失败。请手动执行： %PY% -m pip install -e .
    pause
    exit /b 1
  )
)

rem --- 3. 打开桌面窗口；窗口版起不来时退回命令行问答 ---
echo.
echo 正在打开 zhmem 窗口 ...
echo.
%PY% -m zhmem gui
if errorlevel 1 (
  echo.
  echo 窗口版没能打开（可能缺少图形环境）。改用命令行问答：
  echo 输入问题直接检索；输入 /help 看命令，/quit 退出。
  echo.
  %PY% -m zhmem
)
pause
