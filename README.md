## NexoClip

Multi-tenant SaaS that turns streamer VODs into multi-platform short-form clips.

**Status:** Phase 1 complete. The Phase 0 CLI (`nexoclip process …`) still works; on top of that the project now ships:

- Multi-tenant core: SQLite + tenancy contract, API tokens, dual-write through repos.
- 4-way detector fan-in (voice + chat heat + audio energy + visual signals).
- Local vision pipeline (PySceneDetect + OpenCV motion + Haar Cascade face presence).
- Smart 9:16 crop + auto-thumbnail per clip.
- LLM router with vision capability (`complete_multimodal` + frame cache).
- FastAPI REST API + HTMX dashboard (`uvicorn nexoclip.api.app:create_app`).
- Buffer publisher (drained by `nexoclip publish` or the API lifespan task).

---

## Prerequisites

| Tool | Why | How (Windows) | How (macOS / Linux) |
|---|---|---|---|
| Python 3.11 | runtime (faster-whisper requires <= 3.11/3.12) | `uv python install 3.11` | `uv python install 3.11` |
| `uv` | Python + venv + dep manager | `pip install --user uv` | `pipx install uv` or `pip install --user uv` |
| ffmpeg | cuts + reformats clips | `winget install Gyan.FFmpeg` | `brew install ffmpeg` / `apt install ffmpeg` |
| NVIDIA GPU + driver | faster-whisper CUDA path (CPU works, just slow) | install the NVIDIA driver matching your GPU | same |
| Anthropic API key | variant generation via Claude | `setx ANTHROPIC_API_KEY ...` (new shell after) | `export ANTHROPIC_API_KEY=...` |

> Note: `uv` and `winget` install ffmpeg into a directory that's only added to PATH after a fresh shell. Open a new terminal before running `ffmpeg -version` to verify.

---

## Install

```bash
git clone <repo>
cd QuantorClipAI

# Create the venv with Python 3.11 specifically
uv venv --python 3.11 .venv

# Install runtime + dev deps (editable)
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
# (drop --python on macOS / Linux if your default python is the venv one)

# Copy config templates
cp .env.example .env                                  # then add ANTHROPIC_API_KEY
cp config/nexoclip.example.yaml config/nexoclip.yaml
cp config/personas.example.yaml config/personas.yaml
cp config/llm.example.yaml      config/llm.yaml
```

### Verify the install

```bash
# Activate the venv first:
#   Windows:  .venv\Scripts\activate
#   *nix:     source .venv/bin/activate

ffmpeg -version                                        # binary on PATH?
python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"   # 1+ means CUDA works
nexoclip version                                       # entry-point installed?
pytest                                                  # smoke + unit tests
```

> The faster-whisper CUDA check uses **CTranslate2**, not PyTorch — `import torch` won't work because PyTorch isn't a dependency.

---

## Run

```bash
# Full pipeline against a real Kick / Twitch / YouTube VOD
nexoclip process "https://kick.com/<channel>/videos/<vod_id>" \
    --persona aldo_villanueva \
    --language es \
    --output-dir ./out
```

Per-step subcommands (handy when iterating on one stage):

```bash
nexoclip ingest      "<vod_url>" -o ./out
nexoclip transcribe  <stream_id> -o ./out
nexoclip detect      <stream_id> -o ./out
nexoclip cut         <stream_id> -o ./out
nexoclip variants    <clip_id>   --persona aldo_villanueva
```

Every command supports `--json` (machine-readable output) and `--force` (ignore the on-disk cache).

### Multi-tenant DB + dashboard (Phase 1)

```bash
# 1. Initialize SQLite + apply migrations
nexoclip db init

# 2. Create a tenant + issue an API token (the raw token is shown ONCE)
nexoclip tenants add aldo "Aldo Villanueva"
nexoclip tokens issue --tenant aldo --scope full
# → tok_01H...

# 3. Boot the API + HTMX dashboard
uvicorn 'nexoclip.api.app:create_app' --factory \
    --host 0.0.0.0 --port 8000

# 4. Open http://localhost:8000/dashboard/login and paste your tok_...

# 5. Drain pending publish_jobs manually (the API lifespan also kicks
#    a drain every 60s when started in production mode):
nexoclip publish --tenant aldo
```

The CLI keeps writing JSON manifests to disk and dual-writes through the
DB when `NEXOCLIP_DB_PATH` is set; the dashboard reads from the DB.

### Output layout (Phase 0)

```
out/<stream_id>/
├── manifest.json            # everything about this run, plus llm spend
├── stream.json              # ingest output
├── candidates.json          # detect output
├── clips_manifest.json      # cut output
├── llm_calls.jsonl          # one row per LLM call (tenant, purpose, tokens, cost)
├── source/
│   ├── video.mp4
│   ├── audio.wav            # 16 kHz mono PCM
│   └── transcript.json      # whisper output with word-level timestamps
└── clips/
    └── <clp_*>/
        ├── clip.mp4         # 9:16, libx264 + aac
        ├── metadata.json    # trigger, score, transcript snippet, window
        └── variants.json    # N persona-flavored captions
```

---

## Configuration

| File | Purpose |
|---|---|
| `.env` | secrets + per-machine knobs (`ANTHROPIC_API_KEY`, `NEXOCLIP_WHISPER_*`, `NEXOCLIP_LOG_*`) |
| `config/nexoclip.yaml` | trigger phrases, fuzzy distance, clip windows, encoder settings |
| `config/personas.yaml` | persona voice prompts (the system prompt fragment fed to the LLM) |
| `config/llm.yaml` | provider/model selection per purpose, retry + pricing tables |

Env vars override YAML, YAML overrides defaults (see CLAUDE.md).

---

## Repo layout

```
.
├── CLAUDE.md                # coding rules + project context (read first)
├── PHASE_0.md               # the current phase's tasks
├── README.md                # this file
├── docs/
│   └── nexoclip_spec.md     # full architectural spec (v0.5)
├── config/                  # YAML templates → personal copies
├── nexoclip/                # source
│   ├── ingest/              # yt-dlp + ffmpeg audio extract
│   ├── transcribe/          # faster-whisper
│   ├── detect/              # voice trigger fuzzy match
│   ├── clip/                # ffmpeg fast-cut + 9:16 reformat
│   ├── llm/                 # router + Anthropic provider + structured output
│   ├── variants/            # persona-flavored caption generation
│   ├── pipeline.py          # process_vod orchestrator + manifest
│   ├── settings.py          # pydantic-settings for env vars
│   ├── config.py            # YAML config models
│   ├── logging.py           # structlog setup
│   ├── errors.py            # typed exception hierarchy
│   ├── ids.py               # ULID minting (clp_, str_, var_, ...)
│   └── cli.py               # typer entry point
└── tests/
    └── test_end_to_end.py   # phase-0 exit-criterion smoke test
```

---

## Reading order for contributors / Claude Code

1. `CLAUDE.md` — coding rules + conventions
2. `docs/nexoclip_spec.md` — what we're building end-to-end
3. `PHASE_0.md` — what to build *right now*
4. The code

---

## Status (today)

Phase 0 + Phase 1 complete (Tasks 0-12). `pytest` runs 374 tests in ~18s, `ruff` and `mypy --strict` are clean across 72 source files. The pipeline runs against real VODs once you've set `ANTHROPIC_API_KEY` and downloaded a Whisper model on the first run (faster-whisper pulls the `medium` model, ~770 MB, on demand).

Phase 2 backlog: vision-LLM scoring (cheap-heuristic → premium-vision two-stage funnel), real face emotion classifier, native TikTok / YT Shorts publishers, MCP server, webhook subscribers, OAuth flows, scipy/librosa-driven audio classifiers. See [PHASE_2.md](PHASE_2.md).

---

## License

Private. All rights reserved (for now).
