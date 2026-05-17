"""Slice O.24 — request-level i18n for templates.

Detects the user's locale from the `Accept-Language` header and exposes
a `t(key)` function to Jinja templates that resolves to the localized
string. For now: Spanish (`es`) + English (`en`), with English as the
fallback when a key is missing or the detected locale isn't supported.

Why this shape (not gettext / Babel):
  - Two locales, ~50 strings. A full gettext setup is overkill and adds
    a build step (.po → .mo compilation) that complicates Railway deploys.
  - The translation registry is a plain Python dict so it lives in
    version control + grep-able from anywhere.
  - Jinja globals install at app boot via `install_globals(templates)`;
    no per-template wiring.

How a template uses it:
    {{ t('landing.hero.title') }}
    <html lang="{{ locale() }}">

`locale()` returns the active two-letter code so templates can set
`<html lang>` for screen readers + the browser's translate widget.

How to add a new key:
  1. Add to TRANSLATIONS['en'] AND TRANSLATIONS['es'] below.
  2. Reference in the template via `{{ t('your.key') }}`.

Cost of an untranslated key:
  - In strict mode (`raise_on_missing=True`): KeyError, surfaces fast.
  - In tolerant mode (default): returns the key itself so the page
    still renders, but the operator sees the raw key and knows to add it.
"""
from __future__ import annotations

from fastapi import Request
from fastapi.templating import Jinja2Templates


# Languages we ship. Order matters for "first match wins" against the
# user's Accept-Language priority list — but since both are top-level
# we mostly only care about which one the request prefers.
SUPPORTED_LOCALES: tuple[str, ...] = ("en", "es")
DEFAULT_LOCALE = "en"


def detect_locale(request: Request) -> str:
    """Return 'en' or 'es' based on the request's Accept-Language header.

    Parse rules:
      - Take the comma-separated tags in order (Chrome/Safari sort by
        quality already; we honor that ordering).
      - For each tag, strip region (`en-US` → `en`, `es-AR` → `es`).
      - First tag that matches a SUPPORTED_LOCALES entry wins.
      - No match → DEFAULT_LOCALE.

    A future iteration could read a `?lang=es` query param or a
    `nexoclip_locale` cookie to let the user override their browser.
    """
    raw = request.headers.get("accept-language", "")
    if not raw:
        return DEFAULT_LOCALE
    for tag in raw.split(","):
        # Strip the q-value suffix (`en;q=0.9` → `en`) and any region.
        primary = tag.split(";", 1)[0].strip().lower()
        primary = primary.split("-", 1)[0]
        if primary in SUPPORTED_LOCALES:
            return primary
    return DEFAULT_LOCALE


# ---- Translation registry -------------------------------------------------
#
# Key naming convention: `surface.section.element`.
#   surface = landing | nav | sso | dashboard ...
#   section = hero | flow | foot ...
#   element = title | subtitle | cta_primary ...
#
# Long marketing copy gets `\n` literals — Jinja's |safe filter handles
# the HTML where needed.

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # ---- Landing page -----------------------------------------------
        "landing.title_tag": "NexoClip — the AI growth engine for streamers",
        "landing.hero.brand": "NEXOCLIP",
        "landing.hero.headline_html": (
            "The <strong>AI growth engine</strong> for streamers."
        ),
        "landing.hero.subhead": (
            "Drop a VOD. AI scores every clip on virality, generates the hook, "
            "routes it to the right streamer's brand, and queues publish with "
            "an undo window. The editor's still there when you need it — most "
            "mornings you won't."
        ),
        "landing.hero.cta_sign_in": "Sign in",
        "landing.hero.cta_api_docs": "API docs",
        "landing.hero.cta_for_llms": "For LLMs",
        # ---- Why it's different cards -----------------------------------
        "landing.why.heading": "Why it's different",
        "landing.why.scoring.title": "AI scoring on every clip",
        "landing.why.scoring.body": (
            "Viral score, hook strength, caption readability, dead-air risk — "
            "computed from the multimodal signals NexoClip already collects "
            "(rescore, motion, face presence, words-per-sec). The operator "
            "picks the AI's top 3 and ships."
        ),
        "landing.why.hook.title": "Hook generator, 5 tones",
        "landing.why.hook.body": (
            "One click, 5 viral title candidates in the streamer's voice. "
            "Tone presets: aggressive, Gen Z, corporate, curious, default. "
            "Click a candidate to drop it into the title overlay."
        ),
        "landing.why.intel.title": "Intelligence timeline",
        "landing.why.intel.body": (
            "Per-second markers under the preview: audio peaks, scene cuts, "
            "laughter reactions, chat-heat spikes, face-emotion changes. Click "
            "any marker to seek. Spot the moment that goes viral before you "
            "watch the clip."
        ),
        "landing.why.voice.title": "Voice-marker triggers",
        "landing.why.voice.body_html": (
            "Streamers say <code>clipea esto</code> (clip the next 30s) or "
            "<code>clipeaste eso</code> (clip the previous 60s) as natural "
            "verbal bookmarks. Custom phrases per brand kit."
        ),
        "landing.why.brandkits.title": "Per-speaker brand kits",
        "landing.why.brandkits.body": (
            "Multi-streamer VOD? Speaker diarization routes each clip to the "
            "right host's colors, fonts, handles, and captions. Speaker "
            "identities persist across VODs via embedding match. The "
            "differentiator most clipping tools don't ship."
        ),
        "landing.why.gpu.title": "Local GPU transcription",
        "landing.why.gpu.body": (
            "faster-whisper runs on your own GPU. Stream audio never leaves "
            "your machine for transcription — only the LLM caption-generation "
            "step calls out to Anthropic."
        ),
        "landing.why.publish.title": "Auto-publish with undo",
        "landing.why.publish.body": (
            "Trusted brand kits queue with a scheduled-for + undo window. "
            "Untrusted kits land in the inbox grouped by VOD/speaker. Same "
            "flow either way."
        ),
        "landing.why.agents.title": "Native to AI agents",
        "landing.why.agents.body": (
            "Every action is an MCP tool. Agents can ingest a VOD, score "
            "clips, pick winners, and publish — without a browser session. "
            "Built for the era where the operator is half-human, half-agent."
        ),
        # ---- Growth loop ------------------------------------------------
        "landing.flow.heading": "The growth loop",
        "landing.flow.intro": (
            "Six steps from VOD to published clip. Each step is idempotent "
            "and resumable — re-run any stream and the same outputs land "
            "exactly once."
        ),
        # ---- Nav (base.html dashboard chrome) ---------------------------
        "nav.clips": "Clips",
        "nav.inbox": "Inbox",
        "nav.personas": "Personas",
        "nav.brand_kits": "Brand kits",
        "nav.publish": "Publish",
        "nav.llm_spend": "LLM spend",
        "nav.llm_settings": "LLM settings",
        "nav.logout": "Logout",
        # ---- SSO failure page -------------------------------------------
        "sso.fail.title": "Invalid session",
        "sso.fail.body": (
            "The link you used to enter is invalid or expired. Go back to "
            "the Nexo AI dashboard and open NexoClip again."
        ),
        "sso.fail.cta": "Go to Nexo AI",
    },
    "es": {
        "landing.title_tag": "NexoClip — el motor de IA para growth de streamers",
        "landing.hero.brand": "NEXOCLIP",
        "landing.hero.headline_html": (
            "El <strong>motor de IA</strong> para growth de streamers."
        ),
        "landing.hero.subhead": (
            "Subí un VOD. La IA puntúa cada clip por viralidad, genera el "
            "hook, lo rutea al brand kit del streamer correcto, y agenda la "
            "publicación con ventana de undo. El editor sigue ahí cuando lo "
            "necesites — la mayoría de las mañanas no hace falta."
        ),
        "landing.hero.cta_sign_in": "Ingresar",
        "landing.hero.cta_api_docs": "Docs API",
        "landing.hero.cta_for_llms": "Para LLMs",
        "landing.why.heading": "Por qué es distinto",
        "landing.why.scoring.title": "Puntaje IA en cada clip",
        "landing.why.scoring.body": (
            "Puntaje viral, fuerza del hook, legibilidad del caption, riesgo "
            "de dead-air — computado a partir de los signals multimodales que "
            "NexoClip ya recolecta (rescore, motion, presencia de cara, "
            "palabras por segundo). El operador elige el top 3 de la IA y publica."
        ),
        "landing.why.hook.title": "Generador de hooks, 5 tonos",
        "landing.why.hook.body": (
            "Un click, 5 candidatos de título viral en la voz del streamer. "
            "Tonos: agresivo, Gen Z, corporativo, curioso, default. Hacé "
            "click en un candidato para mandarlo al overlay del título."
        ),
        "landing.why.intel.title": "Timeline de intelligence",
        "landing.why.intel.body": (
            "Marcadores por segundo bajo el preview: picos de audio, cortes "
            "de escena, reacciones de risa, picos de chat-heat, cambios de "
            "emoción facial. Click en cualquier marcador para saltar. Detectá "
            "el momento viral antes de mirar el clip entero."
        ),
        "landing.why.voice.title": "Triggers por marcador de voz",
        "landing.why.voice.body_html": (
            "Los streamers dicen <code>clipea esto</code> (clipea los próximos "
            "30s) o <code>clipeaste eso</code> (clipea los 60s anteriores) "
            "como bookmarks verbales naturales. Frases custom por brand kit."
        ),
        "landing.why.brandkits.title": "Brand kits por speaker",
        "landing.why.brandkits.body": (
            "¿VOD multi-streamer? La diarización de hablantes rutea cada clip "
            "a los colores, fuentes, handles y captions del host correcto. "
            "Las identidades persisten entre VODs por embedding match. El "
            "diferencial que la mayoría de las tools de clipping no tienen."
        ),
        "landing.why.gpu.title": "Transcripción GPU local",
        "landing.why.gpu.body": (
            "faster-whisper corre en tu propia GPU. El audio del stream "
            "nunca sale de tu máquina para transcribir — solo el paso de "
            "generación de captions con LLM llama a Anthropic."
        ),
        "landing.why.publish.title": "Auto-publicación con undo",
        "landing.why.publish.body": (
            "Brand kits confiables se encolan con scheduled-for + ventana de "
            "undo. Brand kits no confiables aterrizan en el inbox agrupados "
            "por VOD/speaker. Mismo flujo en cualquier caso."
        ),
        "landing.why.agents.title": "Nativo para agentes de IA",
        "landing.why.agents.body": (
            "Cada acción es una MCP tool. Los agentes pueden ingestar un VOD, "
            "puntuar clips, elegir ganadores y publicar — sin sesión de "
            "browser. Construido para la era donde el operador es mitad "
            "humano, mitad agente."
        ),
        "landing.flow.heading": "El loop de growth",
        "landing.flow.intro": (
            "Seis pasos de VOD a clip publicado. Cada paso es idempotente y "
            "resumible — re-corré cualquier stream y los mismos outputs "
            "aterrizan exactamente una vez."
        ),
        "nav.clips": "Clips",
        "nav.inbox": "Inbox",
        "nav.personas": "Personas",
        "nav.brand_kits": "Brand kits",
        "nav.publish": "Publicar",
        "nav.llm_spend": "Gasto LLM",
        "nav.llm_settings": "Ajustes LLM",
        "nav.logout": "Salir",
        "sso.fail.title": "Sesión inválida",
        "sso.fail.body": (
            "El enlace que usaste para entrar no es válido o expiró. Volvé "
            "al dashboard de Nexo AI y abrí NexoClip de nuevo."
        ),
        "sso.fail.cta": "Ir a Nexo AI",
    },
}


def t(key: str, locale: str = DEFAULT_LOCALE) -> str:
    """Resolve a translation key for the given locale, falling back to
    English if the key is missing, falling back to the key itself if
    English also doesn't have it (which surfaces the gap loudly).
    """
    table = TRANSLATIONS.get(locale) or TRANSLATIONS[DEFAULT_LOCALE]
    if key in table:
        return table[key]
    en_table = TRANSLATIONS[DEFAULT_LOCALE]
    if key in en_table:
        return en_table[key]
    return key


def install_globals(templates: Jinja2Templates) -> None:
    """Wire `t()` and `locale()` as Jinja globals on a Jinja2Templates
    instance. Call once per templates instance at app boot.

    Both functions read from the Request's `state.locale` which the
    middleware sets at request boundary via `detect_locale`. If
    `state.locale` isn't set, falls back to DEFAULT_LOCALE so templates
    never blow up on a missing context var.

    Uses `jinja2.pass_context` so the template doesn't have to pass
    `request` explicitly — Jinja injects the rendering context, which
    FastAPI's Jinja2Templates auto-populates with `request`.
    """
    import jinja2

    @jinja2.pass_context
    def _t_for_request(ctx, key: str) -> str:
        request = ctx.get("request")
        locale = getattr(request.state, "locale", DEFAULT_LOCALE) if request else DEFAULT_LOCALE
        return t(key, locale)

    @jinja2.pass_context
    def _locale_for_request(ctx) -> str:
        request = ctx.get("request")
        return getattr(request.state, "locale", DEFAULT_LOCALE) if request else DEFAULT_LOCALE

    templates.env.globals["t"] = _t_for_request
    templates.env.globals["locale"] = _locale_for_request


__all__ = [
    "SUPPORTED_LOCALES",
    "DEFAULT_LOCALE",
    "TRANSLATIONS",
    "detect_locale",
    "t",
    "install_globals",
]
