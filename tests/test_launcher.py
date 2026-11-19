"""The 'download and just run it' path: zero-arg REPL + one-click launchers.

These guard the promise in the README that a fresh GitHub clone is usable
without any setup beyond having Python installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from zhmem import cli
from zhmem.retriever import Retriever as RealRetriever

_ROOT = Path(__file__).resolve().parents[1]


class _OfflineRetriever:
    """Stand-in for cli.Retriever so interactive tests never touch the network."""

    def __init__(self, *a, **k):
        self._r = RealRetriever.offline()

    @staticmethod
    def offline(*a, **k):
        return RealRetriever.offline()

    def __getattr__(self, name):
        return getattr(self._r, name)


def test_no_subcommand_enters_interactive_mode(monkeypatch, capsys):
    """`python -m zhmem` with no args must not error out on a required subcommand."""
    lines = iter(["深蹲怎么做才标准", "/quit"])
    monkeypatch.setattr("builtins.input", lambda *_: next(lines))
    monkeypatch.setattr(cli, "Retriever", _OfflineRetriever)

    rc = cli.main([])

    out = capsys.readouterr().out
    assert rc == 0
    assert "zhmem 中文信息检索" in out
    assert "深蹲正确动作" in out
    assert "再见" in out


def test_interactive_commands_are_handled(monkeypatch, capsys):
    lines = iter(["/help", "/images", "/offline", "/no-such-command", "/quit"])
    monkeypatch.setattr("builtins.input", lambda *_: next(lines))
    monkeypatch.setattr(cli, "Retriever", _OfflineRetriever)

    rc = cli.main([])

    out = capsys.readouterr().out
    assert rc == 0
    assert "输入问题直接检索" in out  # /help
    assert "图片：关" in out           # /images toggle off
    assert "已切换为离线模式" in out   # /offline
    assert "未知命令" in out           # /no-such-command


def test_windows_launcher_is_gbk_crlf_and_finds_python():
    bat = _ROOT / "启动.bat"
    assert bat.exists()
    raw = bat.read_bytes()
    assert raw[:3] != b"\xef\xbb\xbf", "a .bat must not carry a UTF-8 BOM"
    assert b"\r\n" in raw, "Windows cmd needs CRLF line endings"
    assert raw.count(b"\n") == raw.count(b"\r\n"), "a bare LF truncates commands in cmd"
    text = raw.decode("gbk")
    assert "where python" in text and "pip install -e ." in text
    assert "python -m zhmem" in text or "%PY% -m zhmem" in text


def test_the_windows_launcher_opens_the_window_and_falls_back_to_the_repl():
    # Double-clicking must land on the desktop window; if tkinter or a display is
    # missing the same script has to keep working in the terminal.
    text = (_ROOT / "启动.bat").read_bytes().decode("gbk").replace("\r\n", "\n")
    assert "%PY% -m zhmem gui" in text
    assert "if errorlevel 1" in text
    assert text.index("%PY% -m zhmem gui") < text.index("%PY% -m zhmem\n")


def test_unix_launcher_is_executable_script():
    sh = _ROOT / "start.sh"
    assert sh.exists()
    text = sh.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    assert "pip install -e ." in text
    assert "exec" in text
    assert "-m zhmem gui" in text, "the shell launcher should try the window first"
