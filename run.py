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


def _load_dotenv() -> None:
    """Read `.env` into os.environ before anything else imports settings.

    Without this, ANTHROPIC_API_KEY (and friends) never make it into the
    process environment. The LLM router then caches every provider as
    None at init time and runs report the misleading
    "all providers failed for purpose=X: provider not available: openai"
    error — even though the user has a real Anthropic key in .env.

    We import dotenv lazily because it's optional: if the user has env vars
    set globally (CI, docker, manual export), we don't want to require the
    package.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        # `override=True` because Windows can leak empty-string values into
        # the environment from user/system profiles. Without override, an
        # ANTHROPIC_API_KEY="" set globally silently shadows the real value
        # in .env — and the LLM router then reports "provider not available"
        # for every Claude call. The .env file is the source of truth.
        load_dotenv(env_path, override=True)
    except ImportError:
        print(
            "[warn] python-dotenv not installed — .env file ignored. "
            "Run: pip install python-dotenv"
        )


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


def _verify_or_fallback_to_cpu() -> None:
    """If `NEXOCLIP_WHISPER_DEVICE=cuda` (the default) but cuBLAS isn't actually
    loadable, override to CPU for this process.

    Without this, the dashboard happily accepts uploads, runs analyze_video,
    and only blows up at the transcribe step with a cryptic
    'Library cublas64_12.dll is not found' — burning 30+ seconds of user time
    per attempt. We try to load cublas64_12.dll up front; if it fails, we
    print a loud warning and switch to CPU + int8 + base model so the rest
    of the pipeline still works.
    """
    if os.name != "nt":
        return
    device = os.environ.get("NEXOCLIP_WHISPER_DEVICE", "cuda").strip().lower()
    if device != "cuda":
        return  # User already opted into cpu — nothing to verify.
    import ctypes

    try:
        ctypes.WinDLL("cublas64_12.dll")
        return  # Loaded fine — CUDA path is genuinely available.
    except OSError:
        pass

    # CUDA 13.x installs ship cublas64_13.dll, not cublas64_12.dll —
    # CTranslate2 / faster-whisper hard-code the v12 filename. If we see
    # nvcc on PATH but cuBLAS v12 still fails, surface that specific mismatch.
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path and "v13" in cuda_path.lower():
        print(
            f"\n[CUDA fallback] CUDA 13.x is installed at {cuda_path} but "
            f"faster-whisper / CTranslate2 needs the v12 cuBLAS library "
            f"(cublas64_12.dll). Fix: 'pip install nvidia-cublas-cu12 "
            f"nvidia-cudnn-cu12' inside the venv — the pip packages provide "
            f"the v12 DLLs alongside your existing v13 install. Falling back "
            f"to CPU for this run.\n"
        )
    else:
        print(
            "\n[CUDA fallback] cublas64_12.dll could not be loaded — falling "
            "back to CPU transcription for this run. Install the full CUDA "
            "Toolkit from NVIDIA if you want GPU speed; for now, CPU + base "
            "+ int8 will work.\n"
        )
    os.environ["NEXOCLIP_WHISPER_DEVICE"] = "cpu"
    os.environ["NEXOCLIP_WHISPER_COMPUTE_TYPE"] = "int8"
    # Only downgrade the model size if the user hasn't picked one explicitly.
    # `medium` on CPU is bearable but slow; `base` is the better default.
    if not os.environ.get("NEXOCLIP_WHISPER_MODEL"):
        os.environ["NEXOCLIP_WHISPER_MODEL"] = "base"
    # Bust the settings cache so the override takes effect.
    try:
        from nexoclip.settings import get_settings

        get_settings.cache_clear()
    except Exception:
        pass


def _ensure_system_cuda_on_path() -> None:
    """Add the NVIDIA CUDA Toolkit `bin\\` dir to PATH if installed system-wide.

    The Toolkit installer adds its bin dir to the *system* PATH at install
    time, but already-open shells don't pick up the change. Result: the
    user installs CUDA, restarts python run.py from the same shell, and
    still hits 'cublas64_12.dll not found' until they reopen PowerShell.

    Walk `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*` and add
    every versioned bin dir we find. No-op on Linux / macOS.
    """
    if os.name != "nt":
        return
    roots = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "NVIDIA GPU Computing Toolkit"
        / "CUDA",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "NVIDIA GPU Computing Toolkit"
        / "CUDA",
    ]
    # CUDA_PATH is the canonical env var the installer sets when it succeeds.
    cuda_path_env = os.environ.get("CUDA_PATH")
    if cuda_path_env:
        roots.insert(0, Path(cuda_path_env).parent)

    added: list[str] = []
    for cuda_root in roots:
        if not cuda_root.exists():
            continue
        for version_dir in sorted(cuda_root.iterdir(), reverse=True):
            bin_dir = version_dir / "bin"
            if not bin_dir.is_dir():
                continue
            # Only add if cuBLAS is actually there (filters out half-installed
            # versions / non-bin subdirs).
            if not any(bin_dir.glob("cublas64_*.dll")):
                continue
            try:
                os.add_dll_directory(str(bin_dir))
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            added.append(f"{cuda_root.name}/{version_dir.name}")
    if added:
        print(
            f"Registered system CUDA Toolkit bin dirs for the DLL loader: "
            f"{', '.join(added)}"
        )


def _ensure_cuda_libs_on_path() -> None:
    """Best-effort: add bundled nvidia-* pip packages' DLL dirs to the search path.

    faster-whisper / CTranslate2 dynamically link cublas64_12.dll and the
    cuDNN libraries. The CUDA Toolkit installer adds them to PATH, but most
    users just `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` — which
    drops the DLLs at .venv/Lib/site-packages/nvidia/*/bin/* but DOESN'T
    register them with Windows' DLL loader. Result: a confusing
    'Library cublas64_12.dll is not found or cannot be loaded' the first
    time transcribe runs, even though `pip list` shows the package.

    Walk site-packages/nvidia/<lib>/bin once at startup and register each
    via os.add_dll_directory + PATH so faster-whisper's load-time link
    resolves cleanly. No-op on Linux / macOS — they use rpath / DYLD.
    """
    if os.name != "nt":
        return
    try:
        import sysconfig

        site_packages = Path(sysconfig.get_paths()["purelib"])
    except Exception:
        return
    nvidia_root = site_packages / "nvidia"
    if not nvidia_root.exists():
        # Helpful breadcrumb when the user thinks they pip-installed but
        # didn't, or installed into a different interpreter.
        print(
            f"No nvidia/* CUDA pip packages found under {site_packages}. "
            f"If you intended GPU transcription, set NEXOCLIP_WHISPER_DEVICE=cpu "
            f"in .env (CPU works without CUDA libs) or install the full "
            f"CUDA Toolkit from NVIDIA's website."
        )
        return
    added: list[str] = []
    for lib_dir in nvidia_root.iterdir():
        bin_dir = lib_dir / "bin"
        if not bin_dir.is_dir():
            continue
        try:
            os.add_dll_directory(str(bin_dir))
        except (OSError, AttributeError):
            pass
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        added.append(lib_dir.name)
    if added:
        print(
            f"Registered CUDA pip libs for the DLL loader: {', '.join(sorted(added))}"
        )


def _print_whisper_config() -> None:
    """Echo the resolved Whisper settings at boot.

    Helps the user catch a stale .env (or one that's not being loaded) before
    the first upload. If they see `device: cuda` but expected `cpu`, the
    settings file isn't where pydantic-settings is looking.
    """
    try:
        from nexoclip.settings import get_settings

        s = get_settings()
        print(
            f"Whisper config: device={s.whisper_device} model={s.whisper_model} "
            f"compute={s.whisper_compute_type}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[warn] couldn't read settings: {e}")


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
    _load_dotenv()
    _ensure_ffmpeg_on_path()
    _ensure_system_cuda_on_path()
    _ensure_cuda_libs_on_path()
    _verify_or_fallback_to_cpu()
    _print_whisper_config()
    try:
        asyncio.run(_boot())
    except KeyboardInterrupt:
        # Ctrl+C is the documented way to stop the dev server; exit
        # cleanly so the user doesn't see a scary-looking traceback.
        print("\nNexoClip dashboard stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
