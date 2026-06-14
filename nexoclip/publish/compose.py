"""Post-text composer — the single source of truth for "given a clip,
what caption/title do we actually publish".

Reused by the dashboard auto-fill button, the Auto-program (whole-queue)
scheduler, and both autopilot paths (on_approve + hands_free). It stitches
together the pieces the rest of the system already produces:

  * the clip's caption + hashtags — from the stored `variants` (the LLM
    variant generator writes these per (clip, persona))
  * a viral hook line — the variant's `title_card_text` (the short overlay
    hook), or, on demand, a freshly generated one via `generate_hook_line`
  * the tenant's fixed handle/hashtag suffix — the @handles + brand tags the
    creator wants on every post (autopublish_settings.tag_suffix)

`build_post` is DB-only and deterministic (no network) so the bulk paths
stay cheap and the assembly is unit-testable. `generate_hook_line` is the
opt-in LLM path the user-triggered single-clip surfaces use to fill a hook
when the variant didn't carry one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from nexoclip.db import Database, VariantsRepo


@dataclass(frozen=True)
class ComposedPost:
    """The final text for one post: a platform `title` (used where the
    platform supports one, e.g. YouTube), the full `caption` body, the
    `hashtags` (with leading `#`), and the bare `hook` line."""

    title: str | None
    caption: str
    hashtags: list[str] = field(default_factory=list)
    hook: str = ""


def _as_hashtag(tag: str) -> str:
    """Normalize a stored tag (no leading `#`) into `#tag`. Empty → ``."""
    t = tag.strip().lstrip("#").strip()
    return f"#{t}" if t else ""


def assemble_caption(
    *, hook: str, body: str, hashtags: list[str], handle_suffix: str
) -> str:
    """Stitch the caption blocks in publish order: hook, body, hashtags,
    fixed suffix — each separated by a blank line, empties dropped. Pure
    and deterministic so it's trivially testable."""
    parts: list[str] = []
    hook = (hook or "").strip()
    body = (body or "").strip()
    if hook:
        parts.append(hook)
    if body and body != hook:
        parts.append(body)
    tagline = " ".join(t for t in (hashtags or []) if t)
    if tagline:
        parts.append(tagline)
    suffix = (handle_suffix or "").strip()
    if suffix:
        parts.append(suffix)
    return "\n\n".join(parts)


async def build_post(
    db: Database,
    clip_id: str,
    *,
    handle_suffix: str = "",
    hook_override: str | None = None,
    persona_id: str | None = None,
) -> ComposedPost:
    """Compose the final post text for `clip_id` from its stored variant.

    DB-only and deterministic. Uses the first variant (newest persona run
    is appended, oldest first — `list_for_clip` orders by created_at) as
    the caption/hashtag source. The hook is `hook_override` when given,
    else the variant's `title_card_text`. The tenant's `handle_suffix`
    rides at the end. Callers must run inside a bound-tenant context
    (the same requirement as `VariantsRepo.list_for_clip`)."""
    variants = await VariantsRepo(db).list_for_clip(clip_id, persona_id=persona_id)
    v = variants[0] if variants else None
    hook = (hook_override if hook_override is not None else (v.title_card_text if v else "")).strip()
    body = (v.caption if v else "").strip()
    hashtags = [h for h in (_as_hashtag(t) for t in (v.hashtags if v else [])) if h]
    caption = assemble_caption(
        hook=hook, body=body, hashtags=hashtags, handle_suffix=handle_suffix
    )
    # Title falls back to the first line of the body so YouTube still gets
    # a usable title even when no hook exists.
    title = hook or (body.splitlines()[0][:140] if body else None) or None
    return ComposedPost(title=title, caption=caption, hashtags=hashtags, hook=hook)


async def generate_hook_line(
    db: Database,
    tenant_id: str,
    clip_id: str,
    *,
    tone: str = "default",
    persona_id: str | None = None,
) -> str:
    """Best-effort single viral-hook line for a clip via the LLM.

    Resolves the persona voice + transcript snippet the same way the hooks
    surface does, then asks `generate_hooks` for one candidate. Returns ""
    on any failure so callers can fall back to the variant's hook — this
    must never raise into a publish path."""
    import json as _json
    from pathlib import Path

    try:
        from nexoclip.db import CandidatesRepo, ClipsRepo, PersonasRepo, TranscriptsRepo
        from nexoclip.llm import LLMRouter, load_llm_config
        from nexoclip.settings import get_settings
        from nexoclip.variants import generate_hooks
        from nexoclip.variants.personas import (
            get_persona as get_yaml_persona,
        )
        from nexoclip.variants.personas import (
            load_personas as load_yaml_personas,
        )

        clip = await ClipsRepo(db).get(clip_id)
        if clip is None:
            return ""

        # Persona: explicit > first tenant persona > first YAML persona.
        persona_voice = ""
        persona_language = "en"
        if persona_id:
            db_persona = await PersonasRepo(db).get(persona_id)
            if db_persona is not None:
                persona_voice = db_persona.voice_prompt
                persona_language = db_persona.primary_language
            else:
                try:
                    yp = get_yaml_persona(persona_id)
                    persona_voice = yp.voice_prompt
                    persona_language = yp.primary_language
                except Exception:
                    pass
        if not persona_voice:
            db_personas = await PersonasRepo(db).list_for_tenant()
            if db_personas:
                persona_voice = db_personas[0].voice_prompt
                persona_language = db_personas[0].primary_language
            else:
                try:
                    yps = list(load_yaml_personas().values())
                    if yps:
                        persona_voice = yps[0].voice_prompt
                        persona_language = yps[0].primary_language
                except Exception:
                    pass

        # Transcript snippet: candidate evidence is cheapest; fall back to
        # the transcript segments overlapping the clip window.
        snippet = ""
        if clip.candidate_id:
            for cand in await CandidatesRepo(db).list_for_stream(clip.stream_id):
                if cand.id == clip.candidate_id:
                    ev = cand.evidence or {}
                    snippet = str(
                        ev.get("transcript_snippet") or ev.get("phrase") or ""
                    ).strip()
                    break
        if not snippet:
            try:
                tx = await TranscriptsRepo(db).get(clip.stream_id)
                if tx is not None:
                    segs = _json.loads(tx.segments_json)
                    if isinstance(segs, list):
                        pieces = [
                            (s.get("text") or "").strip()
                            for s in segs
                            if isinstance(s, dict)
                            and (s.get("end") or 0) >= clip.start_s
                            and (s.get("start") or 0) <= clip.end_s
                            and (s.get("text") or "").strip()
                        ]
                        snippet = " ".join(pieces).strip()[:600]
            except Exception:
                snippet = ""

        tone_id = (tone or "default").strip().lower()
        if tone_id not in ("default", "aggressive", "gen_z", "corporate", "curious"):
            tone_id = "default"

        output_dir = Path(get_settings().default_output_dir)
        router = LLMRouter(
            config=load_llm_config(),
            call_log_path=output_dir / "llm_calls_hooks.jsonl",
            db=db,
        )
        hooks = await generate_hooks(
            tenant_id=tenant_id,
            persona_voice=persona_voice,
            persona_language=persona_language,
            transcript_snippet=snippet,
            tone=tone_id,  # type: ignore[arg-type]
            n=1,
            router=router,
        )
        return hooks[0].text.strip() if hooks else ""
    except Exception:
        return ""
