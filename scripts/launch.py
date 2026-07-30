"""
DAY 6 — one-command launcher: refresh the eval snapshot with live data, then
start the API server (chat + dashboard), so neither has to be typed by hand.

Usage:
  python scripts/launch.py               # live eval refresh, then serve on :8000
  python scripts/launch.py --no-eval     # skip the eval refresh, just serve
  python scripts/launch.py --port 8080
  python scripts/launch.py --reload      # uvicorn --reload, for editing api/*.py
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def run_eval() -> bool:
    print("=" * 70)
    print("Refreshing eval snapshot (python eval/evaluate.py) — live Gemini calls,")
    print("throttled ~4.2s apart across the golden set, so this takes a few minutes.")
    print("=" * 70)
    result = subprocess.run([sys.executable, "eval/evaluate.py"], cwd=ROOT)
    if result.returncode != 0:
        print("\n[warn] eval run failed or was interrupted — the dashboard will fall")
        print("       back to the last saved snapshot (or show 'no snapshot yet').\n")
        return False
    return True


def run_server(port: int, reload: bool) -> None:
    print("=" * 70)
    print(f"Starting server on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    print(f"  Chat:      http://127.0.0.1:{port}/")
    print(f"  Dashboard: http://127.0.0.1:{port}/dashboard.html")
    print("=" * 70)
    cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(port)]
    if reload:
        cmd.append("--reload")
    subprocess.run(cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-eval", action="store_true", help="skip the live eval refresh")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="uvicorn --reload (dev only — see module docstring)")
    args = parser.parse_args()

    if args.no_eval:
        print("Skipping eval refresh (--no-eval) — dashboard shows the last saved snapshot.\n")
    else:
        run_eval()

    run_server(args.port, args.reload)


if __name__ == "__main__":
    main()
