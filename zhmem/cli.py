"""Command-line interface for zhmem.

    python -m zhmem                          # 交互式问答（下载后直接问）
    python -m zhmem gui                      # 桌面窗口版
    python -m zhmem ask "敦煌莫高窟怎么预约"
    python -m zhmem ask "深蹲怎么做" --offline --no-images
    python -m zhmem stats
    python -m zhmem eval
"""

from __future__ import annotations

import argparse
import sys

from .kb import load_entries
from .retriever import Retriever

# Commands understood inside the interactive prompt. Everything else that does
# not start with '/' is treated as a question.
_HELP = """输入问题直接检索；以 / 开头的命令：
  /offline   切换为只查本地知识库（不联网）
  /online    切换为本地 + 联网检索（默认）
  /images    切换是否附带图片
  /help      显示这段帮助
  /quit      退出（也可直接 Ctrl+C 或输入 q）"""

_FAREWELL = "再见。"


def _interactive(offline: bool = False, no_images: bool = False) -> int:
    """Run a tiny read-eval-print loop so a fresh download is usable with zero
    setup: just `python -m zhmem`, type a question, get a cited answer.

    The Retriever is created once and reused: rebuilding the FTS5 store and
    reloading 84 JSON entries on every question would be needlessly slow, and
    more importantly State like offline/images should survive across turns.
    """
    use_images = not no_images
    try:
        r = Retriever.offline() if offline else Retriever()
    except Exception as exc:  # pragma: no cover - depends on local environment
        print(f"初始化失败：{exc}", file=sys.stderr)
        return 1
    print("zhmem 中文信息检索。输入 /help 查看命令。")
    try:
        while True:
            try:
                line = input("zhmem> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in {"q", "quit", "exit", "/quit"}:
                break
            if line in {"/help", "/?"}:
                print(_HELP)
                continue
            if line == "/offline":
                offline = True
                # Rebuild with no-network providers; close the old one.
                r.close()
                r = Retriever.offline()
                print("已切换为离线模式。")
                continue
            if line == "/online":
                offline = False
                r.close()
                r = Retriever()
                print("已切换为在线模式。")
                continue
            if line == "/images":
                use_images = not use_images
                print("图片：" + ("开" if use_images else "关"))
                continue
            if line.startswith("/"):
                print(f"未知命令：{line}（输入 /help 查看）")
                continue
            try:
                ans = r.ask(line, images=use_images)
            except Exception as exc:
                # One flaky web/image request must not kill the whole session.
                print(f"检索出错：{exc}", file=sys.stderr)
                continue
            print()
            print(ans.render(images=use_images))
            print()
    finally:
        r.close()
    print(_FAREWELL)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="zhmem", description="中文混合信息检索")
    sub = p.add_subparsers(dest="cmd")

    ask = sub.add_parser("ask", help="检索并输出 Markdown 答案")
    ask.add_argument("query", help="要问的问题")
    ask.add_argument("--offline", action="store_true", help="只查本地知识库")
    ask.add_argument("--no-images", action="store_true", help="不附带图片")
    ask.add_argument("--limit", type=int, default=5)
    ask.add_argument("--recent", nargs="*", default=[], help="上一轮用户提问（可选）")

    sub.add_parser("stats", help="显示内置知识库统计")
    sub.add_parser("gui", help="打开桌面窗口版")

    ev = sub.add_parser("eval", help="运行记忆检索评测")
    ev.add_argument("--all", action="store_true", help="同时输出各查询明细")

    args = p.parse_args(argv)
    if args.cmd is None:
        return _interactive()
    if args.cmd == "gui":
        from .gui import main as gui_main
        return gui_main()
    if args.cmd == "ask":
        r = Retriever.offline() if args.offline else Retriever()
        try:
            ans = r.ask(args.query, recent=args.recent, images=not args.no_images, limit=args.limit)
        finally:
            r.close()
        print(ans.render(images=not args.no_images, limit=args.limit))
        return 0
    if args.cmd == "stats":
        entries = load_entries()
        by: dict[str, int] = {}
        for e in entries:
            by[e.namespace] = by.get(e.namespace, 0) + 1
        print(f"内置知识库：{len(entries)} 条")
        for ns, n in by.items():
            print(f"  {ns}: {n}")
        return 0
    if args.cmd == "eval":
        from .eval.metrics import evaluate, format_report
        from .eval.corpus import QUERIES, build
        from .store import MemoryStore

        store = MemoryStore()
        try:
            mapping = build(store)
            print(format_report(evaluate(store, mapping, QUERIES), show_details=args.all))
        finally:
            store.close()
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
