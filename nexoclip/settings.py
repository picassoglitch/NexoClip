"""Process-wide settings driven by environment + `.env`.

Per CLAUDE.md: config flows through Pydantic Settings, not scattered
`os.getenv` calls. Phase 0 only needs the pieces that have an env var in
`.env.example`; YAML loading arrives when the detection phrase list does.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Read once from environment / `.env`, then pass into service functions."""

    model_config = SettingsConfigDict(
        env_prefix="NEXOCLIP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    default_tenant_id: str = "default"
    default_output_dir: str = "./out"
    db_path: str = "./nexoclip.db"

    whisper_device: str = "cuda"
    whisper_model: str = "medium"
    whisper_compute_type: str = "float16"

    # Slice F.8 — TranscribeProvider selection. "local" runs
    # faster-whisper on this host (current production behavior).
    # Other values map to cloud STT vendors (assemblyai / deepgram /
    # openai) when their providers ship in F.10+. Each cloud option
    # reads its own API key from the matching `<vendor>_api_key`
    # setting below.
    transcribe_provider: str = "local"
    assemblyai_api_key: str | None = None
    deepgram_api_key: str | None = None
    openai_api_key: str | None = None

    # Slice F.8 — JobDispatcher selection. "in_process" runs pipeline
    # work via FastAPI BackgroundTasks on this host (current behavior).
    # "modal" hands the job off to a Modal app (planned in F.10+ once
    # the nexo-ai deployment lands).
    job_dispatcher: str = "in_process"
    # Hard cap on the transcribe step. faster-whisper occasionally hangs on
    # CUDA — silent stall, no exception, the asyncio.to_thread call never
    # returns. Without a timeout the whole pipeline waits forever and the
    # dashboard's progress card stays stuck on the pulsing dot. 30 min is
    # generous even for hour-long VODs on a slow GPU.
    whisper_timeout_s: float = 1800.0

    log_level: str = "INFO"
    log_format: str = "console"

    llm_default_provider: str = "anthropic"
    llm_default_quality: str = "standard"

    # yt-dlp browser-cookie pass-through. Set to "chrome" / "edge" / "firefox"
    # / "brave" / "chromium" to authenticate yt-dlp using cookies from a
    # logged-in browser session. Required for Kick (which 403s anonymous
    # scraping) and useful for age-gated YouTube. The browser must have
    # visited the platform at least once.
    cookies_from_browser: str | None = None

    # Alternative to cookies_from_browser: an absolute path to a Netscape-
    # format cookies.txt file (export with a "Get cookies.txt" browser
    # extension). yt-dlp reads the file directly so the browser can stay
    # open. When both this and cookies_from_browser are set, the file wins.
    cookies_file: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor; tests can call `get_settings.cache_clear()`."""
    return Settings()
