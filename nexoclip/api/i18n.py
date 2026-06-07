"""Slice O.24 — request-level i18n for templates.

Detects the user's locale from the `Accept-Language` header and exposes
a `t(key)` function to Jinja templates that resolves to the localized
string. Spanish (`es`) + English (`en`), English fallback.

How a template uses it:
    {{ t('landing.hero.h1_line1') }}
    <html lang="{{ locale() }}">

VOICE NOTES (Spanish):
  - Voseo platense/latino. "subí", "transmití", "agendá", "decí",
    "clipeá", "tenés", "querés", "podés", "imaginate", "dejá", "empezá".
  - Direct-response, OpusClip-style. Short paragraphs. Declarative.
    Sell outcomes (more views, more time, less editing, less spend).
  - Numbers ratified by the operator: hero says "20+ clips ready to
    post", emotional checklist uses "17 clips generados" as a concrete
    realistic instance. These override the earlier "12+" guard rail.
  - Posting language: "Publicá mientras dormís" is the headline copy on
    the feature card; the body says "Agendá clips automáticamente. Review
    o undo cuando quieras." Undo window is the trust layer — keep both.

How to add a new key:
  1. Add to TRANSLATIONS['en'] AND TRANSLATIONS['es'].
  2. Reference in the template via `{{ t('your.key') }}`.
"""
from __future__ import annotations

from fastapi import Request
from fastapi.templating import Jinja2Templates


SUPPORTED_LOCALES: tuple[str, ...] = ("en", "es")
DEFAULT_LOCALE = "en"


def detect_locale(request: Request) -> str:
    """Return 'en' or 'es' based on the request's Accept-Language header."""
    raw = request.headers.get("accept-language", "")
    if not raw:
        return DEFAULT_LOCALE
    for tag in raw.split(","):
        primary = tag.split(";", 1)[0].strip().lower()
        primary = primary.split("-", 1)[0]
        if primary in SUPPORTED_LOCALES:
            return primary
    return DEFAULT_LOCALE


# ---- Translation registry -------------------------------------------------
TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # ============================================================
        # LANDING PAGE — OpusClip-style outcome copy
        # ============================================================
        "landing.title_tag": (
            "NexoClip — turn one stream into 20+ ready-to-post clips"
        ),

        # ---- Hero ---------------------------------------------------
        "landing.hero.eyebrow": "NEXOCLIP",
        "landing.hero.h1_line1": "Stop paying editors.",
        "landing.hero.h1_line2": "Start publishing viral clips every day.",
        "landing.hero.sub_line1": (
            "Turn a 4-hour stream into 20+ ready-to-post clips in minutes."
        ),
        "landing.hero.sub_line2": (
            "NexoClip finds the best moments, creates hooks, subtitles, "
            "captions, hashtags, crops for every platform, and schedules "
            "everything automatically."
        ),
        "landing.hero.tagline_a": "You stream.",
        "landing.hero.tagline_b": "NexoClip handles the rest.",
        "landing.hero.cta_primary": "Upload my first stream",
        "landing.hero.cta_primary_micro": "Your first stream, free. No card.",
        "landing.hero.cta_secondary": "Watch demo",

        # ---- Trust bar ---------------------------------------------
        "landing.trust.t1": "TikTok Ready",
        "landing.trust.t2": "Instagram Reels",
        "landing.trust.t3": "YouTube Shorts",
        "landing.trust.t4": "Twitch VODs",
        "landing.trust.t5": "Kick Streams",

        # ---- Pain section ------------------------------------------
        "landing.pain.heading": "Editing clips is killing your growth.",
        "landing.pain.line1": "Every stream creates content.",
        "landing.pain.line2": "Most creators never publish it.",
        "landing.pain.line3": "Because editing takes:",
        "landing.pain.item1": "Hours of clipping",
        "landing.pain.item2": "Hours of captions",
        "landing.pain.item3": "Hours of thumbnails",
        "landing.pain.item4": "Hours of posting",
        "landing.pain.line4": "Or hundreds of dollars a month in editors.",
        "landing.pain.line5": "Meanwhile the stream is already dead.",

        # ---- "NexoClip fixes that" ---------------------------------
        "landing.fix.heading": "NexoClip fixes that.",
        "landing.fix.s1": "Upload stream.",
        "landing.fix.s2": "Get clips.",
        "landing.fix.s3": "Publish.",
        "landing.fix.s4": "Done.",

        # ---- Value Prop --------------------------------------------
        "landing.value.heading_line1": "One stream.",
        "landing.value.heading_line2": "Endless content.",
        "landing.value.intro": "A single stream becomes:",
        "landing.value.item1": "Viral clips",
        "landing.value.item2": "Hooks",
        "landing.value.item3": "Titles",
        "landing.value.item4": "Captions",
        "landing.value.item5": "Hashtags",
        "landing.value.item6": "Shorts",
        "landing.value.item7": "Reels",
        "landing.value.item8": "TikToks",
        "landing.value.without1": "Without touching Premiere.",
        "landing.value.without2": "Without hiring editors.",
        "landing.value.without3": "Without learning anything.",

        # ---- Money section -----------------------------------------
        "landing.math.heading": "What does one editor cost?",
        "landing.math.freelancer_label": "Freelance editor",
        "landing.math.freelancer_price": "$300 – $1,500 / month",
        "landing.math.agency_label": "Agency",
        "landing.math.agency_price": "$1,000 – $5,000 / month",
        "landing.math.us_label": "NexoClip",
        # TODO(operator): replace $__ with the real launch price.
        "landing.math.us_price": "$__ / month",
        "landing.math.us_line1": "A fraction of the cost.",
        "landing.math.us_line2": "Works 24/7.",
        "landing.math.us_line3": "Never sleeps.",
        "landing.math.us_line4": "Never misses a stream.",
        "landing.math.us_line5": "Never takes vacation.",

        # ---- Emotional section -------------------------------------
        "landing.imagine.heading": "Imagine waking up to this.",
        "landing.imagine.line1": "You finish streaming at midnight.",
        "landing.imagine.line2": "You go to sleep.",
        "landing.imagine.line3": "When you wake up:",
        "landing.imagine.item1": "17 clips generated",
        "landing.imagine.item2": "Best moments identified",
        "landing.imagine.item3": "Hooks written",
        "landing.imagine.item4": "Captions created",
        "landing.imagine.item5": "Scheduled for TikTok",
        "landing.imagine.item6": "Scheduled for Reels",
        "landing.imagine.item7": "Scheduled for Shorts",
        "landing.imagine.tagline": "Your content machine never stopped working.",

        # ---- Results section ---------------------------------------
        "landing.results.heading": "More clips = More chances to go viral",
        "landing.results.line1": (
            "Creators who publish consistently grow faster."
        ),
        "landing.results.line2": "The problem isn't talent.",
        "landing.results.line3": "The problem is output.",
        "landing.results.line4": (
            "NexoClip turns every stream into a content factory."
        ),

        # ---- Feature cards (rewritten) -----------------------------
        "landing.features.heading": "Everything you needed an editor for.",
        "landing.features.intro": "Automatic. Every night.",

        "landing.features.viral.title": (
            "Finds your viral moments automatically"
        ),
        "landing.features.viral.body": (
            "NexoClip analyzes every second of your stream and identifies "
            "the clips most likely to perform. No manual reviewing."
        ),
        "landing.features.hook.title": "Never write another title again",
        "landing.features.hook.body": (
            "Get multiple viral title options for every clip. Choose one "
            "with a click."
        ),
        "landing.features.timeline.title": "Skip the boring parts",
        "landing.features.timeline.body": (
            "Jump directly to the funniest, loudest, most emotional and "
            "most engaging moments."
        ),
        "landing.features.voice.title": "Clip moments instantly",
        "landing.features.voice.body_html": (
            "Say: <code>clip that.</code> "
            "And NexoClip remembers it automatically."
        ),
        "landing.features.brand.title": "Your content always looks on-brand",
        "landing.features.brand.body": (
            "Logos. Fonts. Colors. Captions. Everything applied automatically."
        ),
        "landing.features.publish.title": "Post while you sleep",
        "landing.features.publish.body": (
            "Schedule clips automatically across platforms. Review or undo "
            "anytime."
        ),

        # ---- Big statement -----------------------------------------
        "landing.statement_line1": "Stream more.",
        "landing.statement_line2": "Edit less.",
        "landing.statement_line3": "Grow faster.",

        # ---- Final CTA after statement -----------------------------
        "landing.final.heading": "Turn one stream into 20 pieces of content.",
        "landing.final.body": (
            "Finally grow your channel without hiring an editor and "
            "spending 4 hours a day cutting clips."
        ),
        "landing.final.cta": "Upload my first stream",
        "landing.final.cta_micro": "Free. No card.",

        # ---- FAQ ---------------------------------------------------
        "landing.faq.heading": "Frequently asked",
        "landing.faq.q1": (
            "How is NexoClip different from a clip editor?"
        ),
        "landing.faq.a1": (
            "Editors give you a timeline and trim handles. NexoClip is "
            "upstream: AI scores every clip on viral potential, generates "
            "the hook, applies the right brand kit, and schedules with an "
            "undo window. Your job shifts from 'find and cut clips' to "
            "'pick the AI's top 3 and let it ship'. The editor's still "
            "there when you need it — most mornings you won't."
        ),
        "landing.faq.q2": "Who is NexoClip for?",
        "landing.faq.a2": (
            "Streamers who want clips ready by morning. Multi-host "
            "collectives that need per-streamer branding within one VOD. "
            "Agencies running clipping for several creators at once. Agents "
            "/ MCP clients that want to drive a clipping pipeline "
            "programmatically."
        ),
        "landing.faq.q3": "Do I need a GPU?",
        "landing.faq.a3": (
            "A consumer GPU (RTX 4060 or better recommended) handles "
            "Whisper + pyannote comfortably for ~4hr VODs. CPU-only works "
            "for short clips but is much slower."
        ),
        "landing.faq.q4": "Which LLM does NexoClip use?",
        "landing.faq.a4": (
            "Anthropic Claude exclusively for the live surface. The router "
            "supports adding more providers later without code changes."
        ),
        "landing.faq.q5": "Can AI agents drive NexoClip?",
        "landing.faq.a5_html": (
            "Yes. NexoClip ships an MCP (Model Context Protocol) server "
            "for Claude Code, Cursor, and other LLM clients. Technical "
            "details on <a href=\"/agents\">/agents</a>."
        ),
        "landing.faq.q6": "What about retention and privacy?",
        "landing.faq.a6": (
            "Per-tenant retention windows (default 30 days for VODs, 90 "
            "for clips, 365 for transcripts) — all configurable. Audio and "
            "video stay on your machine for transcription; only LLM "
            "caption-generation calls out."
        ),
        "landing.faq.q7": "Is there a direct API?",
        "landing.faq.a7_html": (
            "Yes. <a href=\"/openapi.json\">OpenAPI spec</a> + "
            "<a href=\"/docs\">interactive Swagger UI</a>. Issue a token "
            "via <code>nexoclip tokens issue --tenant &lt;id&gt; --scope "
            "full</code> and pass as "
            "<code>Authorization: Bearer &lt;token&gt;</code>."
        ),

        # ---- Footer -----------------------------------------------
        "landing.foot.tagline": (
            "NexoClip · turn one stream into 20+ ready-to-post clips"
        ),
        "landing.foot.link_agents": "For agents / agencies",
        "landing.foot.link_llms": "llms.txt",
        "landing.foot.link_docs": "API docs",
        "landing.foot.link_signin": "Sign in",

        # ============================================================
        # /agents PAGE — the technical second door (unchanged from
        # previous version — already lives on /agents with deep tech).
        # ============================================================
        "agents.title_tag": (
            "NexoClip for agencies, integrators and AI agents"
        ),
        "agents.hero.eyebrow": "FOR AGENCIES · AGENTS · INTEGRATORS",
        "agents.hero.h1": (
            "Built for the era where the operator is half-human, half-agent."
        ),
        "agents.hero.sub": (
            "Multimodal signal fan-in, per-speaker diarization, local GPU "
            "transcription, MCP-native control surface, fully-typed REST API. "
            "Drive NexoClip from a browser, Claude Code, Cursor, or any "
            "MCP-aware client."
        ),
        "agents.hero.cta_docs": "OpenAPI docs",
        "agents.hero.cta_llms": "llms.txt",
        "agents.hero.cta_back": "← Back to the landing",
        "agents.signals.title": "Multimodal signal fan-in",
        "agents.signals.body": (
            "Voice-marker triggers, chat heat, audio peaks, scene cuts, "
            "face presence and motion all feed a single candidate-window "
            "scorer. Each candidate carries viral score, hook strength, "
            "caption readability and dead-air risk."
        ),
        "agents.diarization.title": "Per-speaker diarization",
        "agents.diarization.body": (
            "pyannote-audio 3.1 labels every speech segment; cosine-"
            "similarity embeddings resolve persistent identities across "
            "VODs. Multi-host streams route each clip to the right host's "
            "brand kit — without manual tagging."
        ),
        "agents.gpu.title": "Local GPU transcription",
        "agents.gpu.body": (
            "faster-whisper runs on the operator's own GPU. Stream audio "
            "never leaves the machine for transcription. The only LLM "
            "round-trip is caption + hook generation through Anthropic "
            "Claude."
        ),
        "agents.mcp.title": "MCP-native control surface",
        "agents.mcp.body_html": (
            "Every operator action is also an MCP tool. Run "
            "<code>nexoclip mcp serve --token &lt;api-token&gt;</code> to "
            "expose stdio MCP. Claude Code, Cursor, and any other MCP "
            "client see tools for listing streams, kicking off pipelines, "
            "fetching clips, managing brand kits, inspecting LLM spend."
        ),
        "agents.api.title": "Fully-typed REST + OpenAPI",
        "agents.api.body_html": (
            "Auto-generated OpenAPI 3.1 spec at <code>/openapi.json</code>, "
            "interactive Swagger at <code>/docs</code>. Pydantic-validated "
            "request and response bodies; no string parsing on either side."
        ),
        "agents.persona.title": (
            "Half-human, half-agent operator"
        ),
        "agents.persona.body": (
            "NexoClip ships assuming the operator is a hybrid: an agent "
            "that drafts the batch overnight, a human that approves it in "
            "30 seconds with morning coffee. The undo window is the trust "
            "boundary between those two halves."
        ),
        "agents.cta.heading": "Start integrating",
        "agents.cta.body": (
            "Read the OpenAPI spec, drop the llms.txt into your LLM "
            "client's context, or run the MCP server locally. The tenant "
            "token is the only credential you need."
        ),

        # ============================================================
        # NAV (dashboard chrome — base.html)
        # ============================================================
        "nav.clips": "Streams",
        "nav.inbox": "Clips",
        "nav.personas": "Personas",
        "nav.brand_kits": "Brand kits",
        "nav.publish": "Publish",
        "nav.llm_spend": "LLM spend",
        "nav.llm_settings": "LLM settings",
        "nav.docs_agents": "Docs / Agents",
        "nav.logout": "Logout",

        # ============================================================
        # SSO failure page
        # ============================================================
        "sso.fail.title": "Invalid session",
        "sso.fail.body": (
            "The link you used to enter is invalid or expired. Go back to "
            "the Nexo AI dashboard and open NexoClip again."
        ),
        "sso.fail.cta": "Go to Nexo AI",
    },

    # ================================================================
    # SPANISH — voseo platense/latino, OpusClip rhythm
    # ================================================================
    "es": {
        "landing.title_tag": (
            "NexoClip — convertí un stream en 20+ clips listos para publicar"
        ),

        # ---- Hero ---------------------------------------------------
        "landing.hero.eyebrow": "NEXOCLIP",
        "landing.hero.h1_line1": "Dejá de pagar editores.",
        "landing.hero.h1_line2": "Empezá a publicar clips virales todos los días.",
        "landing.hero.sub_line1": (
            "Convertí un stream de 4 horas en 20+ clips listos para "
            "publicar, en minutos."
        ),
        "landing.hero.sub_line2": (
            "NexoClip encuentra los mejores momentos, crea hooks, "
            "subtítulos, captions, hashtags, recorta para cada plataforma, "
            "y agenda todo automáticamente."
        ),
        "landing.hero.tagline_a": "Vos transmitís.",
        "landing.hero.tagline_b": "NexoClip hace el resto.",
        "landing.hero.cta_primary": "Subir mi primer stream",
        "landing.hero.cta_primary_micro": "Tu primer stream, gratis. Sin tarjeta.",
        "landing.hero.cta_secondary": "Ver demo",

        # ---- Trust bar ---------------------------------------------
        "landing.trust.t1": "TikTok",
        "landing.trust.t2": "Instagram Reels",
        "landing.trust.t3": "YouTube Shorts",
        "landing.trust.t4": "Twitch VODs",
        "landing.trust.t5": "Kick Streams",

        # ---- Pain section ------------------------------------------
        "landing.pain.heading": "Editar clips te está matando el crecimiento.",
        "landing.pain.line1": "Cada stream genera contenido.",
        "landing.pain.line2": "La mayoría de los creators nunca lo publica.",
        "landing.pain.line3": "Porque editar lleva:",
        "landing.pain.item1": "Horas de clipping",
        "landing.pain.item2": "Horas de captions",
        "landing.pain.item3": "Horas de miniaturas",
        "landing.pain.item4": "Horas de publicar",
        "landing.pain.line4": "O cientos de dólares al mes en editores.",
        "landing.pain.line5": "Mientras tanto el stream ya está muerto.",

        # ---- "NexoClip fixes that" ---------------------------------
        "landing.fix.heading": "NexoClip lo arregla.",
        "landing.fix.s1": "Subí el stream.",
        "landing.fix.s2": "Recibí los clips.",
        "landing.fix.s3": "Publicá.",
        "landing.fix.s4": "Listo.",

        # ---- Value Prop --------------------------------------------
        "landing.value.heading_line1": "Un stream.",
        "landing.value.heading_line2": "Contenido infinito.",
        "landing.value.intro": "Un solo stream se convierte en:",
        "landing.value.item1": "Clips virales",
        "landing.value.item2": "Hooks",
        "landing.value.item3": "Títulos",
        "landing.value.item4": "Captions",
        "landing.value.item5": "Hashtags",
        "landing.value.item6": "Shorts",
        "landing.value.item7": "Reels",
        "landing.value.item8": "TikToks",
        "landing.value.without1": "Sin tocar Premiere.",
        "landing.value.without2": "Sin contratar editores.",
        "landing.value.without3": "Sin aprender nada.",

        # ---- Money section -----------------------------------------
        "landing.math.heading": "¿Cuánto cuesta un editor?",
        "landing.math.freelancer_label": "Editor freelance",
        "landing.math.freelancer_price": "$300 – $1.500 / mes",
        "landing.math.agency_label": "Agencia",
        "landing.math.agency_price": "$1.000 – $5.000 / mes",
        "landing.math.us_label": "NexoClip",
        # TODO(operator): poner el precio real de launch.
        "landing.math.us_price": "$__ / mes",
        "landing.math.us_line1": "Una fracción del costo.",
        "landing.math.us_line2": "Trabaja 24/7.",
        "landing.math.us_line3": "Nunca duerme.",
        "landing.math.us_line4": "Nunca se pierde un stream.",
        "landing.math.us_line5": "Nunca se va de vacaciones.",

        # ---- Emotional section -------------------------------------
        "landing.imagine.heading": "Imaginate despertarte con esto.",
        "landing.imagine.line1": "Terminás de transmitir a medianoche.",
        "landing.imagine.line2": "Te vas a dormir.",
        "landing.imagine.line3": "Cuando te despertás:",
        "landing.imagine.item1": "17 clips generados",
        "landing.imagine.item2": "Mejores momentos identificados",
        "landing.imagine.item3": "Hooks escritos",
        "landing.imagine.item4": "Captions creados",
        "landing.imagine.item5": "Agendados para TikTok",
        "landing.imagine.item6": "Agendados para Reels",
        "landing.imagine.item7": "Agendados para Shorts",
        "landing.imagine.tagline": (
            "Tu máquina de contenido nunca paró de trabajar."
        ),

        # ---- Results section ---------------------------------------
        "landing.results.heading": (
            "Más clips = más chances de hacer viral"
        ),
        "landing.results.line1": (
            "Los creators que publican consistente crecen más rápido."
        ),
        "landing.results.line2": "El problema no es el talento.",
        "landing.results.line3": "El problema es el output.",
        "landing.results.line4": (
            "NexoClip convierte cada stream en una fábrica de contenido."
        ),

        # ---- Feature cards (rewritten) -----------------------------
        "landing.features.heading": (
            "Todo lo que necesitabas un editor para hacer."
        ),
        "landing.features.intro": "Automático. Cada noche.",

        "landing.features.viral.title": (
            "Encuentra tus momentos virales solo"
        ),
        "landing.features.viral.body": (
            "NexoClip analiza cada segundo de tu stream e identifica los "
            "clips con más chances de performar. Sin revisar nada a mano."
        ),
        "landing.features.hook.title": "Nunca más escribas otro título",
        "landing.features.hook.body": (
            "Recibí múltiples opciones de título viral para cada clip. "
            "Elegí uno con un click."
        ),
        "landing.features.timeline.title": "Saltate las partes aburridas",
        "landing.features.timeline.body": (
            "Andá directo a los momentos más divertidos, ruidosos, "
            "emocionales y enganchadores."
        ),
        "landing.features.voice.title": "Clipeá momentos al instante",
        "landing.features.voice.body_html": (
            "Decí: <code>clipeá esto.</code> "
            "Y NexoClip lo recuerda automáticamente."
        ),
        "landing.features.brand.title": (
            "Tu contenido siempre se ve on-brand"
        ),
        "landing.features.brand.body": (
            "Logos. Fuentes. Colores. Captions. Todo aplicado automáticamente."
        ),
        "landing.features.publish.title": "Publicá mientras dormís",
        "landing.features.publish.body": (
            "Agendá clips automáticamente en todas las plataformas. "
            "Review o undo cuando quieras."
        ),

        # ---- Big statement -----------------------------------------
        "landing.statement_line1": "Transmití más.",
        "landing.statement_line2": "Editá menos.",
        "landing.statement_line3": "Crecé más rápido.",

        # ---- Final CTA --------------------------------------------
        "landing.final.heading": (
            "Convertí un stream en 20 piezas de contenido."
        ),
        "landing.final.body": (
            "Finalmente hacé crecer tu canal sin contratar un editor y "
            "sin pasarte 4 horas por día cortando clips."
        ),
        "landing.final.cta": "Subir mi primer stream",
        "landing.final.cta_micro": "Gratis. Sin tarjeta.",

        # ---- FAQ ---------------------------------------------------
        "landing.faq.heading": "Preguntas frecuentes",
        "landing.faq.q1": "¿En qué se diferencia NexoClip de un editor de clips?",
        "landing.faq.a1": (
            "Los editores te dan un timeline con handles para recortar. "
            "NexoClip está upstream: la IA puntúa cada clip por potencial "
            "viral, genera el hook, aplica el brand kit correcto y agenda "
            "con ventana de undo. Tu laburo cambia de \"encontrar y "
            "cortar\" a \"elegir el top 3 de la IA y dejar que se "
            "publique\". El editor sigue ahí — la mayoría de las mañanas "
            "no lo abrís."
        ),
        "landing.faq.q2": "¿Para quién es NexoClip?",
        "landing.faq.a2": (
            "Streamers que quieren los clips listos para la mañana. "
            "Colectivos multi-host que necesitan branding per-streamer "
            "dentro de un VOD. Agencias que clipean para varios creators "
            "en paralelo. Agentes / clientes MCP que quieren manejar el "
            "pipeline programáticamente."
        ),
        "landing.faq.q3": "¿Necesito una GPU?",
        "landing.faq.a3": (
            "Una GPU consumer (RTX 4060+ recomendada) maneja Whisper + "
            "pyannote cómodamente para VODs de ~4hs. Modo CPU-only anda "
            "para clips cortos pero bastante más lento."
        ),
        "landing.faq.q4": "¿Qué LLM usa NexoClip?",
        "landing.faq.a4": (
            "Anthropic Claude exclusivamente para la surface en vivo. El "
            "router soporta agregar más providers sin cambios de código."
        ),
        "landing.faq.q5": "¿Los agentes de IA pueden manejar NexoClip?",
        "landing.faq.a5_html": (
            "Sí. NexoClip ship un servidor MCP para Claude Code, Cursor y "
            "otros clientes LLM. Detalles técnicos en "
            "<a href=\"/agents\">/agents</a>."
        ),
        "landing.faq.q6": "¿Qué pasa con retention y privacy?",
        "landing.faq.a6": (
            "Ventanas de retention per-tenant (default 30 días para VODs, "
            "90 para clips, 365 para transcripciones) — todo configurable. "
            "Audio y video se quedan en tu máquina para transcripción; "
            "solo el paso de generación de captions LLM llama afuera."
        ),
        "landing.faq.q7": "¿Hay API directa que pueda llamar?",
        "landing.faq.a7_html": (
            "Sí. <a href=\"/openapi.json\">Spec OpenAPI</a> + "
            "<a href=\"/docs\">Swagger UI interactivo</a>. Generás un "
            "token con <code>nexoclip tokens issue --tenant &lt;id&gt; "
            "--scope full</code> y lo pasás como "
            "<code>Authorization: Bearer &lt;token&gt;</code>."
        ),

        # ---- Footer -----------------------------------------------
        "landing.foot.tagline": (
            "NexoClip · convertí un stream en 20+ clips listos para publicar"
        ),
        "landing.foot.link_agents": "Para agentes / agencias",
        "landing.foot.link_llms": "llms.txt",
        "landing.foot.link_docs": "Docs API",
        "landing.foot.link_signin": "Ingresar",

        # ============================================================
        # /agents PAGE — second door technical (unchanged)
        # ============================================================
        "agents.title_tag": (
            "NexoClip para agencias, integradores y agentes de IA"
        ),
        "agents.hero.eyebrow": "PARA AGENCIAS · AGENTES · INTEGRADORES",
        "agents.hero.h1": (
            "Construido para la era donde el operador es mitad humano, "
            "mitad agente."
        ),
        "agents.hero.sub": (
            "Fan-in de signals multimodales, diarización per-speaker, "
            "transcripción GPU local, surface MCP-nativa, REST API "
            "fully-typed. Manejá NexoClip desde un browser, desde Claude "
            "Code, Cursor, o cualquier cliente MCP-aware."
        ),
        "agents.hero.cta_docs": "Docs OpenAPI",
        "agents.hero.cta_llms": "llms.txt",
        "agents.hero.cta_back": "← Volver al landing",
        "agents.signals.title": "Fan-in de signals multimodales",
        "agents.signals.body": (
            "Voice-marker triggers, chat heat, picos de audio, cortes de "
            "escena, presencia de cara y motion alimentan un solo scorer "
            "de ventanas candidatas. Cada candidata trae puntaje viral, "
            "fuerza del hook, legibilidad del caption y riesgo de dead-air."
        ),
        "agents.diarization.title": "Diarización per-speaker",
        "agents.diarization.body": (
            "pyannote-audio 3.1 etiqueta cada segmento de habla; "
            "embeddings por cosine-similarity resuelven identidades "
            "persistentes entre VODs. Streams multi-host rutean cada clip "
            "al brand kit del host correcto — sin tagging manual."
        ),
        "agents.gpu.title": "Transcripción GPU local",
        "agents.gpu.body": (
            "faster-whisper corre en la GPU del operador. El audio del "
            "stream nunca sale de la máquina para transcribir. El único "
            "round-trip LLM es generación de captions + hooks vía "
            "Anthropic Claude."
        ),
        "agents.mcp.title": "Surface de control MCP-nativa",
        "agents.mcp.body_html": (
            "Cada acción del operador es también una MCP tool. Corré "
            "<code>nexoclip mcp serve --token &lt;api-token&gt;</code> "
            "para exponer MCP por stdio. Claude Code, Cursor y cualquier "
            "cliente MCP ven tools para listar streams, kick-off de "
            "pipelines, fetch de clips, manejar brand kits, inspeccionar "
            "el gasto LLM."
        ),
        "agents.api.title": "REST fully-typed + OpenAPI",
        "agents.api.body_html": (
            "Spec OpenAPI 3.1 auto-generado en <code>/openapi.json</code>, "
            "Swagger interactivo en <code>/docs</code>. Request y response "
            "bodies validados con Pydantic; cero string parsing en ambos "
            "lados."
        ),
        "agents.persona.title": (
            "Operador: mitad humano, mitad agente"
        ),
        "agents.persona.body": (
            "NexoClip ship asumiendo que el operador es híbrido: un "
            "agente que dibuja el batch durante la noche, un humano que "
            "lo aprueba en 30 segundos con el café de la mañana. La "
            "ventana de undo es el límite de confianza entre las dos mitades."
        ),
        "agents.cta.heading": "Empezá a integrar",
        "agents.cta.body": (
            "Leé el spec OpenAPI, droppeá el llms.txt en el contexto de "
            "tu cliente LLM, o corré el servidor MCP local. El token de "
            "tenant es la única credencial que necesitás."
        ),

        # ============================================================
        # NAV
        # ============================================================
        "nav.clips": "Streams",
        "nav.inbox": "Clips",
        "nav.personas": "Personas",
        "nav.brand_kits": "Brand kits",
        "nav.publish": "Publicar",
        "nav.llm_spend": "Gasto LLM",
        "nav.llm_settings": "Ajustes LLM",
        "nav.docs_agents": "Docs / Agentes",
        "nav.logout": "Salir",

        # ============================================================
        # SSO failure page
        # ============================================================
        "sso.fail.title": "Sesión inválida",
        "sso.fail.body": (
            "El enlace que usaste para entrar no es válido o expiró. "
            "Volvé al dashboard de Nexo AI y abrí NexoClip de nuevo."
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
    """
    import jinja2

    @jinja2.pass_context
    def _t_for_request(ctx, key: str) -> str:
        request = ctx.get("request")
        locale = (
            getattr(request.state, "locale", DEFAULT_LOCALE)
            if request else DEFAULT_LOCALE
        )
        return t(key, locale)

    @jinja2.pass_context
    def _locale_for_request(ctx) -> str:
        request = ctx.get("request")
        return (
            getattr(request.state, "locale", DEFAULT_LOCALE)
            if request else DEFAULT_LOCALE
        )

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
