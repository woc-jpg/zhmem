#!/usr/bin/env bash
# zhmem 一键启动（macOS / Linux）。Windows 用同目录的 启动.bat。
# Usage: ./start.sh          — 交互式问答，必要时自动安装
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "需要 Python 3.10+：https://www.python.org/downloads/" >&2; exit 1; }

# 在项目目录里能 import 就说明源码可用；否则装成可编辑包（含知识库数据）。
if ! "$PY" -c 'import zhmem' >/dev/null 2>&1; then
  echo "首次运行：安装 zhmem ..."
  "$PY" -m pip install -e .
fi

echo
echo "正在打开 zhmem 窗口 ..."
echo

# Prefer the desktop window; without a display (or without tkinter) fall back to
# the terminal so the script is never a dead end over SSH.
if "$PY" -m zhmem gui; then
  exit 0
fi

echo
echo "窗口版没能打开（可能没有图形环境）。改用命令行问答："
echo "输入问题直接检索；/help 看命令，/quit 退出。"
echo
exec "$PY" -m zhmem
