# -*- coding: utf-8 -*-
"""
敦煌文化遗产智能知识库系统 — 一键启动入口

使用方法：
    python run.py              # 启动 Web 界面（Flask + 前端）
    python run.py --mode api   # 仅启动 API 服务器
    python run.py --mode ui    # 启动 Streamlit 管理界面

功能：
    - Web 模式：Flask 后端 + 精美 HTML 前端（推荐）
    - API 模式：仅提供 RESTful API 接口
    - UI 模式：Streamlit 管理后台
"""

import os
import sys
import subprocess
import argparse
import webbrowser
import threading


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── 自动加载 .env 环境变量 ──────────────────────────────────
def load_dotenv():
    """从项目根目录的 .env 文件加载环境变量（如果存在）。"""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # 不覆盖已存在的环境变量
            if key and key not in os.environ:
                os.environ[key] = value
    print("  环境变量已从 .env 加载")


load_dotenv()


def ensure_config():
    """确保配置文件存在并加载。"""
    from config import get_config
    cfg = get_config()
    print(f"  配置加载成功")
    print(f"  PDF 目录：{cfg['paths']['pdf_dir']}")
    return cfg


def ensure_dependencies():
    """检查关键依赖是否已安装。"""
    missing = []
    try:
        import flask
    except ImportError:
        missing.append("flask")
    try:
        import flask_cors
    except ImportError:
        missing.append("flask-cors")

    if missing:
        print(f"  正在安装缺失依赖：{', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing, "-q"],
            cwd=PROJECT_ROOT,
        )


def launch_flask(port=5000, open_browser=True):
    """启动 Flask API 服务器（阻塞）。"""
    from api.server import app

    if open_browser:
        # 延迟 2 秒后打开浏览器，等待服务器启动
        def _open():
            import time
            time.sleep(2)
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open, daemon=True).start()

    print(f"\n  Web 界面：http://localhost:{port}")
    print(f"  API 接口：http://localhost:{port}/api/")
    print()
    app.run(host="0.0.0.0", port=port, debug=False)


def launch_streamlit():
    """启动 Streamlit 管理界面（阻塞）。"""
    ui_path = os.path.join(PROJECT_ROOT, "ui", "app.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", ui_path],
        cwd=PROJECT_ROOT,
    )


def main():
    parser = argparse.ArgumentParser(
        description="敦煌文化遗产智能知识库系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python run.py                启动 Web 界面（默认，端口 5000）
  python run.py --port 8080    使用自定义端口
  python run.py --mode api     仅启动 API 服务器
  python run.py --mode ui      启动 Streamlit 管理界面
        """,
    )
    parser.add_argument(
        "--mode", choices=["web", "api", "ui"], default="web",
        help="启动模式：web（默认）= Flask + 前端, api = 仅 API, ui = Streamlit",
    )
    parser.add_argument("--port", type=int, default=5000, help="Web/API 监听端口（默认 5000）")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    print("=" * 55)
    print("   敦煌文化遗产智能知识库系统")
    print("   Dunhuang Cultural Heritage AI Knowledge Base")
    print("=" * 55)
    print()

    ensure_config()
    ensure_dependencies()

    if args.mode == "web":
        print(f"\n  模式：Web 界面（Flask 后端 + 精美前端）")
        launch_flask(port=args.port, open_browser=not args.no_browser)

    elif args.mode == "api":
        print(f"\n  模式：API 服务器")
        launch_flask(port=args.port, open_browser=False)

    elif args.mode == "ui":
        print(f"\n  模式：Streamlit 管理界面")
        launch_streamlit()


if __name__ == "__main__":
    main()
