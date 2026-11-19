"""Desktop app for zhmem — a chat window instead of a command line.

Started with ``python -m zhmem gui`` (or the packaged ``zhmem-gui`` script / the
Windows 启动.bat). Pure tkinter from the standard library: no Electron, no web
server, no extra install. Pillow is used only if it happens to be available, for
downscaling web images; without it images fall back to plain links.

The UI thread never blocks: retrieval runs in a worker thread and answers are
delivered back through a queue polled by `after()`. A slow web request therefore
leaves the window responsive, and the status line keeps saying what it is doing.
"""

from __future__ import annotations

import io
import queue
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from . import __version__
from .kb import load_entries
from .retriever import Answer, Retriever

# --- palette (a light, quiet theme; the accent is only used for the user's own
# messages and the send button, so the answer text stays the focus) -------------
BG = "#f5f6f8"
PANEL = "#ffffff"
BORDER = "#e3e6ea"
TEXT = "#1f2328"
MUTED = "#6b7280"
ACCENT = "#2563eb"
ACCENT_SOFT = "#eef3ff"
USER_BG = "#e8f0ff"

NAMESPACES = [("全部", None), ("日常生活", ("daily",)), ("旅游攻略", ("travel",)), ("健身知识", ("fitness",))]

EXAMPLES = [
    "敦煌莫高窟怎么预约",
    "被烫伤了怎么处理",
    "深蹲怎么做才标准",
    "每天盐摄入量多少合适",
    "北京旅游攻略",
    "第一次去西藏要注意什么",
]


def _pick_font(*candidates: str) -> str:
    """First installed family from `candidates`, else whatever Tk defaults to."""
    try:
        families = set(tkfont.families())
    except Exception:  # pragma: no cover - only if Tk failed to init fonts
        return "TkDefaultFont"
    for name in candidates:
        if name in families:
            return name
    return "TkDefaultFont"


@dataclass
class _Job:
    query: str
    namespaces: tuple[str, ...] | None
    web: bool
    images: bool


class ChatApp:
    """A single-window chat client over `Retriever`."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.busy = False
        self.events: list[str] = []  # recent user turns, fed back as `recent=`
        self._photos: list[Any] = []  # PhotoImage refs; Tk does not keep them alive

        self.online = tk.BooleanVar(value=True)
        self.with_images = tk.BooleanVar(value=True)
        self.namespace = tk.StringVar(value="全部")

        self.ui_font = _pick_font("Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC")
        self.mono_font = _pick_font("Cascadia Mono", "Consolas", "Menlo", "DejaVu Sans Mono")

        root.title(f"zhmem 中文信息检索  v{__version__}")
        root.geometry("1080x760")
        root.minsize(820, 560)
        root.configure(bg=BG)

        self._build_style()
        self._build_layout()
        self._greet()
        self._load_stats_async()
        root.after(80, self._pump)

    # ---------------------------------------------------------------- widgets
    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:  # pragma: no cover
            pass
        style.configure("TButton", font=(self.ui_font, 10), padding=(10, 6))
        style.configure("Send.TButton", font=(self.ui_font, 10, "bold"),
                        foreground="#ffffff", background=ACCENT, borderwidth=0, padding=(18, 8))
        style.map("Send.TButton", background=[("active", "#1d4ed8"), ("disabled", "#9db4e8")])
        style.configure("TRadiobutton", font=(self.ui_font, 10), background=PANEL, anchor="w", padding=(6, 4))
        style.configure("TCheckbutton", font=(self.ui_font, 10), background=PANEL, anchor="w", padding=(6, 2))
        style.configure("TSeparator", background=BORDER)

    def _build_layout(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        side = tk.Frame(self.root, bg=PANEL, width=236, highlightthickness=0)
        side.grid(row=0, column=0, sticky="ns")
        side.grid_propagate(False)
        tk.Frame(self.root, bg=BORDER, width=1).grid(row=0, column=1, sticky="ns")
        self._build_sidebar(side)

        main = tk.Frame(self.root, bg=BG)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        self._build_header(main)
        self._build_transcript(main)
        self._build_composer(main)

    def _build_sidebar(self, parent: tk.Frame) -> None:
        head = tk.Frame(parent, bg=PANEL)
        head.pack(fill="x", padx=18, pady=(20, 6))
        tk.Label(head, text="zhmem", bg=PANEL, fg=TEXT,
                 font=(self.ui_font, 17, "bold")).pack(anchor="w")
        tk.Label(head, text="中文信息检索 · 本地知识库 + 联网",
                 bg=PANEL, fg=MUTED, font=(self.ui_font, 8)).pack(anchor="w", pady=(2, 0))

        ttk.Separator(parent).pack(fill="x", padx=18, pady=12)

        tk.Label(parent, text="检索范围", bg=PANEL, fg=MUTED,
                 font=(self.ui_font, 9)).pack(anchor="w", padx=18, pady=(0, 2))
        for label, _ns in NAMESPACES:
            ttk.Radiobutton(parent, text=label, value=label, variable=self.namespace,
                            command=self._filters_changed).pack(fill="x", padx=12)

        ttk.Separator(parent).pack(fill="x", padx=18, pady=12)

        tk.Label(parent, text="选项", bg=PANEL, fg=MUTED,
                 font=(self.ui_font, 9)).pack(anchor="w", padx=18, pady=(0, 2))
        ttk.Checkbutton(parent, text="允许联网检索", variable=self.online,
                        command=self._filters_changed).pack(fill="x", padx=12)
        ttk.Checkbutton(parent, text="显示相关图片", variable=self.with_images,
                        command=self._filters_changed).pack(fill="x", padx=12)

        ttk.Separator(parent).pack(fill="x", padx=18, pady=12)

        tk.Label(parent, text="试试这些", bg=PANEL, fg=MUTED,
                 font=(self.ui_font, 9)).pack(anchor="w", padx=18, pady=(0, 2))
        for text in EXAMPLES:
            btn = tk.Label(parent, text=text, bg=PANEL, fg=ACCENT, cursor="hand2",
                           font=(self.ui_font, 9), anchor="w", padx=18)
            btn.pack(fill="x", pady=1)
            btn.bind("<Button-1>", lambda _e, q=text: self._ask(q))
            btn.bind("<Enter>", lambda e: e.widget.configure(fg="#1d4ed8", bg=ACCENT_SOFT))
            btn.bind("<Leave>", lambda e: e.widget.configure(fg=ACCENT, bg=PANEL))

        self.stats_label = tk.Label(parent, text="知识库加载中…", bg=PANEL, fg=MUTED,
                                    font=(self.ui_font, 8), justify="left", anchor="w", padx=18)
        self.stats_label.pack(side="bottom", fill="x", pady=(0, 16))

    def _build_header(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent, bg=BG, height=52)
        bar.grid(row=0, column=0, sticky="ew")
        bar.columnconfigure(0, weight=1)
        self.title_label = tk.Label(bar, text="随便问点什么", bg=BG, fg=TEXT,
                                    font=(self.ui_font, 12, "bold"), anchor="w")
        self.title_label.grid(row=0, column=0, sticky="w", padx=20, pady=(14, 0))
        self.status = tk.Label(bar, text="", bg=BG, fg=MUTED, font=(self.ui_font, 9), anchor="e")
        self.status.grid(row=0, column=1, sticky="e", padx=20, pady=(14, 0))

    def _build_transcript(self, parent: tk.Frame) -> None:
        wrap = tk.Frame(parent, bg=BG)
        wrap.grid(row=1, column=0, sticky="nsew", padx=(0, 0), pady=(8, 0))
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=1)

        self.text = tk.Text(wrap, bg=BG, fg=TEXT, wrap="word", bd=0, highlightthickness=0,
                            padx=20, pady=8, spacing1=2, spacing3=6, cursor="arrow",
                            font=(self.ui_font, 11), selectbackground=ACCENT_SOFT,
                            selectforeground=TEXT)
        self.text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(wrap, orient="vertical", command=self.text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.text.configure(yscrollcommand=scroll.set)

        # A tagged background also paints the tag's paragraph spacing, so keep
        # spacing small here and put the breathing room in untagged lines.
        self.text.tag_configure("user", justify="right", background=USER_BG, foreground="#12305f",
                                lmargin1=140, lmargin2=140, rmargin=4,
                                spacing1=0, spacing3=0, font=(self.ui_font, 11))
        self.text.tag_configure("role", foreground=MUTED, font=(self.ui_font, 8))
        self.text.tag_configure("h1", font=(self.ui_font, 13, "bold"), spacing1=12, spacing3=4,
                                foreground=TEXT)
        self.text.tag_configure("h2", font=(self.ui_font, 11, "bold"), spacing1=8, spacing3=2,
                                foreground=ACCENT)
        self.text.tag_configure("body", font=(self.ui_font, 11), lmargin1=6, lmargin2=6,
                                spacing3=4, foreground=TEXT)
        self.text.tag_configure("meta", foreground=MUTED, font=(self.ui_font, 9))
        self.text.tag_configure("url", foreground="#0b6bcb", underline=True)
        self.text.tag_configure("warn", foreground="#a15c00", font=(self.ui_font, 10))
        self.text.tag_configure("caption", foreground=MUTED, font=(self.ui_font, 8))
        self.text.tag_configure("gap", spacing1=10)
        self.text.configure(state="disabled")

    def _build_composer(self, parent: tk.Frame) -> None:
        box = tk.Frame(parent, bg=BG)
        box.grid(row=2, column=0, sticky="ew", padx=18, pady=(6, 14))
        box.columnconfigure(0, weight=1)

        border = tk.Frame(box, bg=BORDER, padx=1, pady=1)
        border.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        inner = tk.Frame(border, bg=PANEL)
        inner.pack(fill="both", expand=True)
        inner.columnconfigure(0, weight=1)

        self.entry = tk.Text(inner, height=3, bg=PANEL, fg=TEXT, bd=0, highlightthickness=0,
                             wrap="word", font=(self.ui_font, 11), padx=12, pady=10,
                             insertbackground=ACCENT)
        self.entry.grid(row=0, column=0, sticky="ew")
        self.entry.bind("<Return>", self._on_enter)
        self.entry.bind("<Shift-Return>", lambda _e: None)

        self.hint = tk.Label(inner, text="Enter 发送 · Shift+Enter 换行", bg=PANEL, fg=MUTED,
                             font=(self.ui_font, 8), anchor="w", padx=12)
        # A widget's pady is a single distance; only geometry managers take a pair.
        self.hint.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        self.send_btn = ttk.Button(box, text="发送", style="Send.TButton", command=self._send_clicked)
        self.send_btn.grid(row=0, column=1, sticky="ns")
        self.entry.focus_set()

    # -------------------------------------------------------------- behaviour
    def _filters_changed(self) -> None:
        parts = []
        if not self.online.get():
            parts.append("离线")
        parts.append(self.namespace.get())
        if not self.with_images.get():
            parts.append("无图")
        self.title_label.configure(text="随便问点什么" if len(parts) == 1 else " · ".join(parts))

    def _current_namespaces(self) -> tuple[str, ...] | None:
        for label, ns in NAMESPACES:
            if label == self.namespace.get():
                return ns
        return None

    def _on_enter(self, event: tk.Event) -> str:
        if event.state & 0x0001:  # Shift held → let the Text insert a newline
            return ""
        self._send_clicked()
        return "break"

    def _send_clicked(self) -> None:
        query = self.entry.get("1.0", "end").strip()
        if not query:
            return
        self.entry.delete("1.0", "end")
        self._ask(query)

    def _ask(self, query: str) -> None:
        if self.busy:
            self._set_status("还在检索上一条…")
            return
        self.busy = True
        self.send_btn.state(["disabled"])
        self._set_status("检索中…")
        self._append_user(query)
        job = _Job(query, self._current_namespaces(), self.online.get(), self.with_images.get())
        threading.Thread(target=self._work, args=(job, self.events[-3:]), daemon=True).start()

    def _work(self, job: _Job, recent: list[str]) -> None:
        """Runs off the UI thread. Never touches widgets — only the queue."""
        started = time.perf_counter()
        try:
            r = Retriever.offline() if not job.web else Retriever()
            try:
                answer = r.ask(job.query, recent=recent, images=job.images,
                               namespaces=job.namespaces)
            finally:
                r.close()
            self.queue.put(("answer", (answer, (time.perf_counter() - started) * 1000)))
            if job.images and answer.images:
                self.queue.put(("images", self._fetch_thumbnails(answer)))
        except Exception as exc:  # a broken provider must not kill the window
            self.queue.put(("error", f"{type(exc).__name__}: {exc}"))

    def _fetch_thumbnails(self, answer: Answer) -> list[tuple[Any, str, str]]:
        """Download up to 3 thumbnails. Returns (PIL image or None, title, url)."""
        out: list[tuple[Any, str, str]] = []
        try:
            from PIL import Image  # type: ignore
        except Exception:
            Image = None  # type: ignore
        for img in answer.images[:3]:
            title, url = img.title, img.thumb or img.url
            pic = None
            if Image is not None and url:
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 zhmem/1.0"})
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        data = resp.read(3_000_000)
                    pil = Image.open(io.BytesIO(data))
                    pil.thumbnail((220, 160))
                    pic = pil.convert("RGB")
                except Exception:
                    pic = None
            out.append((pic, title, img.url))
        return out

    def _pump(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "answer":
                    answer, ms = payload
                    self._append_answer(answer, ms)
                    if not answer.images:
                        self._finish()
                elif kind == "images":
                    self._append_images(payload)
                    self._finish()
                elif kind == "error":
                    self._append_error(payload)
                    self._finish()
                elif kind == "stats":
                    self.stats_label.configure(text=payload)
        except queue.Empty:
            pass
        self.root.after(80, self._pump)

    def _finish(self) -> None:
        self.busy = False
        self.send_btn.state(["!disabled"])
        self.entry.focus_set()

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)

    # ----------------------------------------------------------- rendering
    def _w(self, text: str, *tags: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", text, tags or ("body",))
        self.text.configure(state="disabled")
        self.text.see("end")

    def _greet(self) -> None:
        self._w("zhmem 中文信息检索\n", "h1")
        self._w("内置 84 条生活常识 / 全国旅游 / 健身知识；本地命中不够时才联网，"
                "问到能配图的话题会附上图片。左边可以限定检索范围。\n", "meta")

    def _append_user(self, query: str) -> None:
        self.events.append(query)
        # Insert the newlines *outside* the tagged range: a newline carrying the
        # background tag paints an empty coloured block the height of a line.
        self.text.configure(state="normal")
        self.text.insert("end", "\n")
        self.text.insert("end", query, "user")
        self.text.insert("end", "\n")
        self.text.configure(state="disabled")
        self.text.see("end")

    def _append_answer(self, answer: Answer, ms: float) -> None:
        self._w(f"\n", "gap")
        if answer.local:
            self._w("本地知识库\n", "h2")
            for hit in answer.local:
                title = hit.memory.subject or "（未命名）"
                self._w("· ", "meta")
                self._w(f"{title}", "h2")
                self._w(f"  相关度 {hit.score:.2f}\n", "caption")
                self._w(f"{hit.memory.body}\n", "body")
        if answer.web:
            self._w("网络资料\n", "h2")
            for res in answer.web:
                self._w(f"· {res.title}\n", "body")
                if res.url:
                    self._w(f"  {res.url}\n", "url")
                if res.snippet:
                    self._w(f"  {res.snippet}\n", "caption")
        if not answer.local and not answer.web:
            self._w("没找到可靠资料，换个说法试试。\n", "warn")
        self._set_status(f"{ms:.0f} ms · 本地 {len(answer.local)} 条 · 网络 {len(answer.web)} 条")

    def _append_images(self, items: list[tuple[Any, str, str]]) -> None:
        if not items:
            return
        self._w("\n相关图片\n", "h2")
        for pic, title, url in items:
            shown = False
            if pic is not None:
                try:
                    from PIL import ImageTk  # type: ignore
                    photo = ImageTk.PhotoImage(pic)
                    self.text.configure(state="normal")
                    self.text.image_create("end", image=photo, padx=6, pady=4)
                    self.text.configure(state="disabled")
                    # Tk does not own the PhotoImage: keep a reference or it
                    # vanishes the moment this loop variable is rebound.
                    self._photos.append(photo)
                    shown = True
                except Exception:
                    shown = False
            if not shown and url:
                self._w(f"[图片] {title}\n  {url}\n", "caption")
            elif shown:
                self._w(f"{title}\n", "caption")
        self.text.see("end")

    def _append_error(self, message: str) -> None:
        self._w(f"\n出错了：{message}\n", "warn")
        self._w("（本地知识库不受影响，可以再问一次或勾掉“允许联网检索”。）\n", "meta")

    def _load_stats_async(self) -> None:
        def run() -> None:
            try:
                counts: dict[str, int] = {}
                for e in load_entries():
                    counts[e.namespace] = counts.get(e.namespace, 0) + 1
                text = "知识库 " + str(sum(counts.values())) + " 条\n" + "\n".join(
                    f"  {k} {v}" for k, v in sorted(counts.items()))
            except Exception as exc:  # pragma: no cover
                text = f"知识库读取失败：{exc}"
            self.queue.put(("stats", text))
        threading.Thread(target=run, daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    try:  # crisp text on high-DPI Windows panels; harmless elsewhere
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
    except Exception:
        pass
    root = tk.Tk()
    ChatApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
