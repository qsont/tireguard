import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from tireguard.config import AppConfig
from tireguard.ui import run_app


def _spawn_web_process(host: str, port: int) -> subprocess.Popen:
    # Supports both normal Python run and frozen EXE run
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "--web-only", "--host", host, "--port", str(port)]
    else:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--web-only",
            "--host",
            host,
            "--port",
            str(port),
        ]
    return subprocess.Popen(cmd)


def _run_web_only(host: str, port: int):
    import uvicorn
    from tireguard.api import app as api_app

    uvicorn.run(api_app, host=host, port=port, reload=False)


def _stop_process(p: subprocess.Popen | None):
    if not p or p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--web-only", action="store_true", help="Run web API only")
    parser.add_argument("--desktop-only", action="store_true", help="Run desktop UI only")
    parser.add_argument("--simple-ui", action="store_true", help="Run simplified 800x480 touchscreen UI")
    parser.add_argument("--compact-ui", action="store_true", help="Force compact UI layout (recommended for 800x480 displays)")
    parser.add_argument("--rpi-ui", action="store_true", help="Raspberry Pi touchscreen preset (compact + fullscreen)")
    parser.add_argument(
        "--compact-video-ratio",
        type=float,
        default=None,
        help="Compact UI video height ratio (0.20-0.80, default 0.40)",
    )
    parser.add_argument(
        "--rpi-video-ratio",
        type=float,
        default=None,
        help="Raspberry Pi UI video height ratio (0.20-0.80, default 0.38)",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Web host bind address")
    parser.add_argument("--port", type=int, default=8000, help="Web port")
    args = parser.parse_args()

    if args.web_only:
        _run_web_only(args.host, args.port)
        raise SystemExit(0)

    web_proc = None
    if not args.desktop_only:
        web_proc = _spawn_web_process(args.host, args.port)
        time.sleep(1.0)
        if not args.no_browser:
            webbrowser.open(f"http://127.0.0.1:{args.port}/")

    try:
        cfg = AppConfig()
        compact_override = True if (args.compact_ui or args.rpi_ui) else None
        run_app(
            cfg,
            simple_ui=bool(args.simple_ui),
            compact_ui=compact_override,
            fullscreen=bool(args.rpi_ui),
            rpi_ui=bool(args.rpi_ui),
            compact_video_ratio=args.compact_video_ratio,
            rpi_video_ratio=args.rpi_video_ratio,
        )
    finally:
        _stop_process(web_proc)
