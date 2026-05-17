"""Process-wide settings driven by environment + `.env`.

Per CLAUDE.md: config flows through Pydantic Settings, not scattered
`os.getenv` calls. Phase 0 only needs the pieces that have an env var in
`.env.example`; YAML loading arrives when the detection phrase list does.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
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
    # dashboard's progress card stays stuck on the pulsing dot.
    #
    # Slice O.12 — the OLD default of 1800 (30 min) was too tight for
    # multi-hour VODs. The pipeline now scales timeouts proportional to
    # stream duration via `whisper_timeout_multiplier`. The static
    # `whisper_timeout_s` value is the FLOOR (minimum) — actual timeout
    # = `max(whisper_timeout_s, duration_s * whisper_timeout_multiplier)`.
    #
    # 4× realtime is a generous cap on a healthy CUDA setup (whisper
    # typically runs 3-10× realtime; the buffer absorbs cold-start +
    # disk-flush + paging hiccups). On a slow CPU/GPU bump to 8×.
    whisper_timeout_s: float = 1800.0
    whisper_timeout_multiplier: float = 4.0

    # Slice O.12 — analyze_video (PySceneDetect + visual signals)
    # timeout multiplier. The static config value is the floor; actual
    # = max(floor, duration_s * multiplier). On a 3-hour stream a
    # 0.5× cap = 90 min, plenty even on CPU-only hosts.
    analyze_video_timeout_multiplier: float = 0.5

    # Admin tenants (NEXOCLIP_ADMIN_TENANT_IDS) bypass the per-step
    # ceilings entirely. Useful for the operator dogfooding multi-hour
    # streams before paying users see them. Set to False to apply the
    # same caps to admins (safer for cost control once paying users land).
    admin_uncapped_pipeline: bool = True

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

    # Nexo AI integration — slice NX.1.
    # Both are symmetric secrets shared with Nexo AI (which sees them as
    # NEXOCLIP_ADMIN_TOKEN and NEXOCLIP_SSO_SECRET on its side). When unset,
    # the new admin + SSO endpoints reject all traffic with 503 so we never
    # accept un-authenticated provisioning calls.
    #
    # We override the auto-prefix so the env vars are the simple
    # NEXO_AI_ADMIN_TOKEN / NEXO_AI_SSO_SECRET names that the spec doc
    # promises (instead of NEXOCLIP_NEXO_AI_... which is the prefix the rest
    # of this Settings class uses).
    nexo_ai_admin_token: str | None = Field(
        default=None,
        validation_alias="NEXO_AI_ADMIN_TOKEN",
    )
    nexo_ai_sso_secret: str | None = Field(
        default=None,
        validation_alias="NEXO_AI_SSO_SECRET",
    )
    # Slice NX.3 — outbound URL to Nexo AI. Used to push usage events back
    # to the platform after each LLM call. When unset, usage reporting is a
    # no-op (NexoClip can run standalone without Nexo AI knowing about it).
    nexo_ai_base_url: str | None = Field(
        default=None,
        validation_alias="NEXO_AI_BASE_URL",
    )
    # Where NexoClip is reachable from the public internet. Used as the
    # post-SSO landing origin if we ever need to build absolute URLs.
    # Defaults to localhost for dev. Same override pattern.
    public_url: str = Field(
        default="http://localhost:8000",
        validation_alias="NEXOCLIP_PUBLIC_URL",
    )

    # Slice O.9 — admin tenant allowlist. Comma-separated tenant IDs that
    # see operator-only nav items (LLM spend + LLM settings). All other
    # tenants get a creator-only nav. Empty / unset → no tenants are
    # admins (creator UX for everyone, including local dev). Set this to
    # your own tenant id to see the admin pages back.
    admin_tenant_ids: str = ""

    # Slice O.22 — nexo-ai is the gatekeeper now. NexoClip no longer
    # serves a token-paste login form. Anyone hitting `/dashboard/login`
    # or any /dashboard/* page without a valid session cookie is bounced
    # to this URL. Production: https://nexo-ai.world/login. Set blank
    # to fall back to the legacy in-house login page (kept as
    # emergency-admin backstop; see dashboard.login_form).
    nexo_ai_login_url: str = Field(
        default="https://nexo-ai.world/login",
        validation_alias="NEXO_AI_LOGIN_URL",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor; tests can call `get_settings.cache_clear()`."""
    return Settings()
