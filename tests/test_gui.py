"""Desktop window tests.

Tk needs a display; on a headless CI box these are skipped rather than failed.
The window is built but never enters `mainloop()`, so nothing blocks.
"""

from __future__ import annotations

import pytest

tk = pytest.importorskip("tkinter", reason="tkinter is not installed")

from zhmem import gui  # noqa: E402


def _make_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:  # pragma: no cover - headless CI
        pytest.skip(f"no display available: {exc}")
    root.withdraw()
    return root


def test_the_import_does_not_need_a_display():
    # Importing must stay side-effect free: `zhmem.cli` imports this module on
    # `gui` subcommand only, but a broken import would break `--help` too.
    assert callable(gui.main)
    assert isinstance(gui.NAMESPACES, list) and gui.NAMESPACES[0] == ("全部", None)
    assert len(gui.EXAMPLES) >= 4


def test_namespaces_map_every_label_to_a_real_namespace():
    namespaces = {ns for _, ns in gui.NAMESPACES if ns}
    assert namespaces == {("daily",), ("travel",), ("fitness",)}


def test_the_window_builds_with_its_sidebar_and_composer():
    root = _make_root()
    try:
        app = gui.ChatApp(root)
        root.update()
        assert "zhmem" in root.title()
        assert app.send_btn.cget("text") == "发送"
        assert app.entry.get("1.0", "end").strip() == ""
    finally:
        root.destroy()


def test_a_user_message_is_tagged_without_its_newlines():
    # A newline inside the tagged range paints a full empty line of background
    # colour above the text — the bubble then looks twice as tall as it is.
    root = _make_root()
    try:
        app = gui.ChatApp(root)
        root.update()
        app._append_user("深蹲怎么做才标准")
        root.update()
        start, end = app.text.tag_ranges("user")
        assert app.text.get(start, end) == "深蹲怎么做才标准"
        assert app.text.index(end) == app.text.index(f"{start} lineend")
    finally:
        root.destroy()


def test_the_header_reflects_the_selected_filters():
    root = _make_root()
    try:
        app = gui.ChatApp(root)
        root.update()
        app.online.set(False)
        app.namespace.set("健身知识")
        app._filters_changed()
        assert "离线" in app.title_label.cget("text")
        assert "健身知识" in app.title_label.cget("text")
        assert app._current_namespaces() == ("fitness",)
    finally:
        root.destroy()


def test_an_answer_is_rendered_and_the_status_line_reports_the_counts(monkeypatch):
    class FakeRetriever:
        def ask(self, query, recent=(), images=True, namespaces=None):
            from zhmem.retriever import Answer
            from zhmem.store import Recalled

            class _Memory:
                subject = "深蹲正确动作"
                body = "双脚与肩同宽。"

            return Answer(query=query, local=[Recalled(_Memory(), 1.44, {})], web=[], images=[])

        def close(self):
            pass

        @classmethod
        def offline(cls):
            return cls()

    monkeypatch.setattr(gui, "Retriever", FakeRetriever)

    root = _make_root()
    try:
        app = gui.ChatApp(root)
        root.update()
        app.online.set(False)
        app._ask("深蹲怎么做才标准")

        import time

        deadline = time.time() + 10
        while app.busy and time.time() < deadline:
            root.update()
            time.sleep(0.02)
        for _ in range(10):
            root.update()

        body = app.text.get("1.0", "end")
        assert "深蹲正确动作" in body
        assert "本地知识库" in body
        assert "本地 1 条" in app.status.cget("text")
        assert not app.busy
    finally:
        root.destroy()
