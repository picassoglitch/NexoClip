"""Boot the NexoClip dashboard locally with auto-drains on.

Convenience launcher so you can `python run.py` from the project root
instead of typing the uvicorn factory invocation. Reads `NEXOCLIP_DB_PATH`
+ `NEXOCLIP_HOST` + `NEXOCLIP_PORT` from the environment with sensible
defaults.

Usage (after activating the venv):

    .venv\\Scripts\\activate            # PowerShell / cmd
    source .venv/Scripts/activate       # Git Bash
    source .venv/bin/activate           # macOS / Linux

    python run.py
    # → http://127.0.0.1:8000/dashboard/login
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

import uvicorn

from nexoclip.api import create_app
from nexoclip.db import Database, apply_migrations


def _ensure_ffmpeg_on_path() -> None:
    """Best-effort: if ffmpeg isn't on PATH but winget installed it under
    %LOCALAPPDATA%\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg*, prepend that
    bin dir to PATH for this process.

    The winget ffmpeg packages install the binary under a versioned dir
    inside the package — they DON'T add it to the user PATH automatically,
    which leaves the user with `ffmpeg : not recognized` after a successful
    install. Fix that here so `python run.py` works on a fresh box without
    a PATH-editing step.
    """
    if shutil.which("ffmpeg"):
        return
    if os.name != "nt":
        return
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        return
    winget_packages = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
    if not winget_packages.exists():
        return
    # Both Gyan.FFmpeg and Gyan.FFmpeg.Essentials nest a versioned
    # ffmpeg-*-build/bin/ffmpeg.exe inside their package dir.
    for pkg in winget_packages.glob("Gyan.FFmpeg*"):
        for bin_dir in pkg.glob("ffmpeg-*/bin"):
            if (bin_dir / "ffmpeg.exe").exists():
                os.environ["PATH"] = (
                    f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                )
                print(f"Located ffmpeg at {bin_dir} (added to PATH for this run)")
                return


def _resolve_python_check() -> None:
    """Hard-stop if you launched via system Python 3.14 instead of the venv.

    faster-whisper requires Python <= 3.12; running under 3.14 will fail
    with cryptic CTranslate2 errors deep in the transcription path.
    """
    if sys.version_info >= (3, 13):
        print(
            f"ERROR: NexoClip requires Python 3.11 or 3.12, got {sys.version_info.major}."
            f"{sys.version_info.minor}.\n"
            f"Activate the venv first:\n"
            f"  PowerShell:  .venv\\Scripts\\Activate.ps1\n"
            f"  Git Bash:    source .venv/Scripts/activate\n"
            f"  macOS/Linux: source .venv/bin/activate",
            file=sys.stderr,
        )
        sys.exit(1)


async def _boot() -> None:
    db_path = Path(os.environ.get("NEXOCLIP_DB_PATH", "./nexoclip.db"))
    host = os.environ.get("NEXOCLIP_HOST", "127.0.0.1")
    port = int(os.environ.get("NEXOCLIP_PORT", "8000"))

    db = Database(db_path)
    await apply_migrations(db)
    app = create_app(db=db, enable_background_drains=True)

    print(f"\nNexoClip dashboard: http://{host}:{port}/dashboard/login\n")
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()


def main() -> None:
    _resolve_python_check()
    _ensure_ffmpeg_on_path()
    try:
        asyncio.run(_boot())
    except KeyboardInterrupt:
        # Ctrl+C is the documented way to stop the dev server; exit
        # cleanly so the user doesn't see a scary-looking traceback.
        print("\nNexoClip dashboard stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
