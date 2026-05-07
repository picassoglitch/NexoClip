# Phase 2 - backlog stub

Phase 1 (Tasks 0-12) is complete: multi-tenant core, 4-way detector fan-in,
local vision pipeline, smart 9:16 crop + thumbnail, LLM router with vision
capability, FastAPI + HTMX dashboard, Buffer publisher. Phase 2 picks up
where Phase 1 stopped.

This file is a backlog, not a fully scoped phase plan. Each item below
should turn into its own task list (a la PHASE_1.md) before work starts.

---

## Headline goals

- **Quality leap on candidate selection.** Phase 1 ships a 4-signal heuristic
  that's good enough to surface "obviously clippable" moments. Phase 2 adds
  a multimodal-LLM pass that re-scores the top-K heuristic candidates with
  full visual context, so we publish fewer, better clips.
- **Real emotion classification.** Phase 1's face_emotion module signals
  presence only (Haar Cascade gives a bbox, not landmarks). Phase 2 either
  trains a small smile/laugh/shock classifier or routes those frames to the
  vision LLM with a tight schema.
- **Native publishers.** Buffer is a stop-gap. Phase 2 adds direct TikTok
  Content Posting API + YouTube Data API integrations behind the same
  `Publisher` protocol the worker already uses.
- **Webhooks.** Phase 1 emits to the `events` table from day one but has
  no subscribers. Phase 2 adds outbound webhook dispatch (HMAC-signed) so
  customers can wire in their own stacks.

---

## Backlog by area

### Detection + vision

- [ ] **Vision-LLM scoring funnel.** Two-stage: cheap heuristic produces
      top-K candidates (already done); premium model rescores with 5-frame
      multimodal context. Plug into `LLMRouter.complete_multimodal` (Task 8
      already shipped the surface). Add a "premium" routing rule for
      `candidate_scoring`.
- [ ] **Real face-emotion classifier.** Either:
        a) train a small CNN on a smile/laugh/shock dataset, or
        b) route 2-fps frames through the vision LLM with a Literal[
           "neutral", "smile", "laugh", "shock", "anger", "sad"] schema.
      Either way the visual_signals fan-in already accepts the labels;
      only `nexoclip/vision/face_emotion.py` changes.
- [ ] **Vision-LLM smart crop / thumbnail.** Phase 1 ships a face-detect
      heuristic; Phase 2 swaps in a vision-LLM picker that reasons about
      composition + on-screen text + gaze. The on-disk shape doesn't change.
- [ ] **OCR-based on-screen-text detector.** `visual_signals.text_changed`
      column is reserved but unfilled; OCR (PaddleOCR or AWS Textract via
      worker) populates it. Useful for trigger-from-overlay events.
- [ ] **Audio classifier upgrade.** Phase 1 uses `numpy.fft` + RMS only.
      Phase 2 evaluates `librosa` + a small audio CNN for laughter /
      applause / silence detection. `scipy` may also enter the dep tree.

### Publishers

- [ ] **TikTok Content Posting API.** OAuth flow, video upload, caption
      formatting. Same `Publisher` protocol as Buffer.
- [ ] **YouTube Data API (Shorts).** Same.
- [ ] **OAuth refresh.** Phase 1 stores raw access tokens inside
      `connected_accounts.oauth_blob_json`. Phase 2 stores refresh tokens
      and rotates on demand; the publisher catches 401, refreshes, retries.

### Platform

- [ ] **MCP server.** Expose stream / clip / variant resources over MCP so
      Claude Desktop can drive review. Auth via the same api_tokens table.
- [ ] **Webhook dispatch.** Subscribe to `events` rows by type + tenant;
      POST HMAC-signed payloads to subscriber URLs with retry/give-up.
- [ ] **Per-tenant rate limiting.** Buffer-style rate limits per platform;
      back-pressure into the publish queue.

### DX + ops

- [ ] **Live progress on the dashboard.** HTMX SSE endpoint streams
      pipeline-step events so users see "ingesting", "transcribing", etc.
      while a stream processes.
- [ ] **Cost projections.** Roll-up cards on /dashboard/llm-calls
      (per-purpose, per-day, with model + quality breakdown).
- [ ] **Persona prompt iteration UI.** Edit voice prompts in-place, then
      re-run variants on a stored clip without re-cutting.
- [ ] **DB migrations 002+.** Whatever Phase 2 needs (likely: webhook
      subscriptions, TikTok/YT external_id columns on publish_jobs).

### Cloud (deferred to Phase 3)

The following stay out of Phase 2 - they belong with the cloud migration:
Aurora/Postgres, ECS, SQS-backed worker pool, S3 for clip + frame storage,
real auth (Cognito or similar), billing/Stripe, marketing site, multi-region.

---

## Acceptance demo for Phase 2

A streamer connects Buffer **and** their TikTok account on the dashboard,
runs a VOD through the pipeline, and the vision-LLM rescore reorders the
top candidates so the published clip captures the actual reaction
(smile/shock onset) rather than the loudest moment. Webhooks fire for
each clip.published event so the streamer's own automation can pick up
from there.
