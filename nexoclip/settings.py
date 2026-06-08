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
    # Slice O.46 — variants generation is OFF by default. Operator
    # feedback: "they take a lot and don't do much". The variants step
    # used to fan out per-persona LLM caption drafts; with the editor's
    # overlay system + hook generator carrying most of the same value
    # without an LLM call per persona, the per-clip variant generation
    # was wasteful. When False, the pipeline still creates a single
    # stub VariantRow per clip so the publish flow keeps working
    # (publishers consume variant.caption / variant.hashtags).
    variants_enabled: bool = False

    transcribe_provider: str = "local"
    assemblyai_api_key: str | None = None
    deepgram_api_key: str | None = None
    openai_api_key: str | None = None

    # AssemblyAI tuning.
    #
    # `assemblyai_language_mode` — what the AAI submit-time language
    # contract looks like per call. Three values:
    #
    #   "auto" (DEFAULT) — submit with `language_detection=true` +
    #     the code-switching prompt. AAI's Universal-3 Pro model
    #     handles ES/EN code-switching natively; the prompt
    #     instructs it to preserve each phrase in its original
    #     language instead of translating. This is the LATAM-creator
    #     default — every Mexican / Argentine / Spanish-speaking
    #     streamer who drops English filler ("clip that", "GG", etc)
    #     gets faithfully-transcribed captions in BOTH languages.
    #
    #   "es" | "en" — lock to a specific language. Submits
    #     `language_code=<code>` and drops language_detection + the
    #     code-switching prompt. Best accuracy for monolingual
    #     creators (most US English streamers, dedicated ES-only
    #     channels). Per-stream overrides through the existing
    #     `language` field on the pipeline call.
    #
    # `assemblyai_speech_models` — model ladder. The first model is
    # the quality target (`universal-3-pro` — best ES/EN accuracy +
    # native code-switching). The second is the 99-language fallback
    # (`universal-2`) for the rare case U3-Pro misses on an exotic
    # accent. AAI walks the ladder in order.
    #
    # `assemblyai_speaker_labels` — per-video diarization labels
    # (A / B / C). The pipeline reads these instead of running
    # pyannote separately (Task A2). Default True; flip False to
    # drop the ~$0.02/hr diarization upcharge if a particular run
    # doesn't need speaker separation.
    assemblyai_language_mode: str = "auto"
    assemblyai_speech_models: list[str] = Field(
        default_factory=lambda: ["universal-3-pro", "universal-2"]
    )
    assemblyai_speaker_labels: bool = True
    # Slice O.44 — Modal Whisper provider. Wired when
    # `transcribe_provider="modal"`. The endpoint is the URL Modal
    # exposes for the `transcribe` web function (printed by
    # `modal deploy infra/modal_whisper_app.py`). The token is a
    # shared bearer that the Modal app verifies — pick something
    # long and random.
    #   - modal_endpoint_url: e.g. https://username--nexoclip-whisper-transcribe.modal.run
    #   - modal_token: shared secret in the Authorization: Bearer header
    #   - modal_model: faster-whisper model size; "small" is the sweet
    #     spot on a T4 GPU (3-5 min per VOD-hour, accurate Spanish).
    #     "base" is faster but loses on accented speech; "medium" / "large-v3"
    #     are slower + costlier on Modal.
    modal_endpoint_url: str | None = None
    modal_token: str | None = None
    modal_model: str = "small"
    # Slice O.44 — HMAC secret for the internal signed-URL audio
    # fetch endpoint Modal pulls from. Different secret than the
    # SSO HMAC — this one is purely internal. Pick something random;
    # the audio URL only stays valid for 30 min so a leak is bounded.
    # Phase L.1 reuses this same secret for the MediaMTX webhook
    # bearer (internal-trust-only; not user-facing).
    internal_signing_secret: str | None = None

    # Phase L.1 — RTMP base URL the operator pastes into OBS. This is
    # the MediaMTX endpoint, NOT the NexoClip API host. Example:
    # `rtmp://live.nexoclip.nexo-ai.world/live`
    # The dashboard appends the active stream key to this base when
    # displaying the OBS URL. Unset = the live dashboard renders a
    # "live ingest not configured" panel instead.
    live_rtmp_base_url: str | None = None

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
    # timeout multiplier. Actual cap = max(floor, duration_s * multiplier).
    # Slice O.27 — bumped 0.5× → 4× because the 0.5× value assumed GPU
    # but most prod hosts (Railway, Fly, Render) run on CPU where
    # PySceneDetect runs at ~1× realtime. A 87-second video hit the
    # 120-second floor with the old multiplier and skipped the step
    # entirely; 4× gives PySceneDetect 4 minutes on that video (plenty)
    # and proportionally more on longer ones.
    analyze_video_timeout_multiplier: float = 4.0

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

    # Task 2b — VOD download speed.
    #
    # `ytdlp_concurrent_fragments` — yt-dlp's `concurrent_fragment_downloads`
    # option. Twitch / Kick / YouTube VODs are HLS / DASH segmented; the
    # native downloader fetches one segment at a time by default. Bumping
    # this to 16 typically gives 3-10× speedup on bandwidth-constrained
    # hosts (Railway egress, fly.io edge). Drop to 1 if the upstream CDN
    # rate-limits parallel fetches and you see 429s.
    ytdlp_concurrent_fragments: int = 16
    #
    # `ytdlp_use_aria2c` — opt into aria2c as the external downloader.
    # aria2c parallelises within a single segment via byte ranges; on
    # multi-GB sources it beats yt-dlp's native loop. Only takes effect
    # when aria2c is actually on PATH (we probe at request time and
    # silently fall back to the native downloader otherwise). False by
    # default because aria2c isn't installed on the Railway image yet.
    ytdlp_use_aria2c: bool = False
    #
    # `max_source_height` — cap the resolution yt-dlp asks for. Clips top
    # out at 1080×1920 for free/pro and 4K for all_access, so pulling
    # the platform's "best available" (often 4K+ on Kick / YouTube)
    # wastes bandwidth. Set to 1080 for Phase 0 defaults; tenants on the
    # all_access tier can bump this per-stream once the resolution-tier
    # plumbing surfaces it. 0 disables the cap (use platform's best).
    max_source_height: int = 1080

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
    # Token T3 — per-run BASE CHARGE in USD micros. The raw API cost of a
    # run (transcription + occasional Claude) can be near-zero, but each run
    # still consumes server time (ffmpeg, transcription orchestration,
    # rendering), storage, and bandwidth. This flat charge — reported as an
    # `engine.base` usage event after every successful run — covers that
    # overhead + margin so a near-free-API run still draws down the quota.
    # Default $0.05; at Nexo AI's ~$4/1M-token rate that's ~12,500
    # token-equivalents per run. Set to 0 to disable. Tune to your real
    # infra cost per run.
    pipeline_base_charge_usd_micros: int = Field(
        default=50_000,
        validation_alias="NEXOCLIP_PIPELINE_BASE_CHARGE_USD_MICROS",
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

    # ------------------------------------------------------------------
    # upload-post.com integration (replaces the in-house OAuth Connect
    # surface that lived in commits 774e2d9 + 9a89e1a + was scrapped
    # in fe5da71). upload-post is the multi-platform publish layer:
    # we hand them a video URL + target platforms, they do the actual
    # OAuth + posting against TikTok / IG / YT / X / LinkedIn / etc.
    # Multi-tenant via their "user profile" model — one upload-post
    # profile per NexoClip tenant.
    # ------------------------------------------------------------------
    #
    # upload_post_api_key — single company-wide API key from
    # upload-post.com. Authenticates every call. Tenants never see it.
    # Header shape: `Authorization: Apikey <key>`.
    #
    # Env var: NEXOCLIP_UPLOAD_POST_API_KEY (per the class-level
    # env_prefix="NEXOCLIP_"). Be explicit about this in the deploy
    # docs — the raw `UPLOAD_POST_API_KEY` (what upload-post's own
    # docs use as a placeholder) is silently ignored.
    upload_post_api_key: str | None = None

    # Base URL. Override only for tests / staging environments.
    # Env var: NEXOCLIP_UPLOAD_POST_BASE_URL.
    upload_post_base_url: str = "https://api.upload-post.com"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor; tests can call `get_settings.cache_clear()`."""
    return Settings()
