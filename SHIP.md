# SHIP.md — NexoClip go-live checklist

The native Connect tab (TikTok / Instagram / YouTube OAuth) is
fully built. The remaining blocker before paying strangers can
complete the full loop is **platform review — one approval per
platform, total, not per tenant**.

Under Pattern A, NexoClip owns one app per platform. Each platform
gates "publish on behalf of a non-test user" behind a review pass.
Without these approvals, the OAuth flows still work — but the
tokens come back scoped to your dev test users only, and any
attempt to publish for a real signup gets rejected with
permission/scope errors.

This file lists the three review packets, what each platform asks
for, and current status. Update the status lines as we move
through them.

---

## 1. TikTok — Content Posting API audit

**Portal**: developer.tiktok.com → your app → Content Posting API tab
**Reviewer turnaround**: 5-10 business days, often longer on first pass
**Current status**: ⬜ Not submitted

### Required materials

- [ ] **Working demo video** showing the FULL Connect → Publish loop
      end-to-end. ~3 minutes. Screen capture from a non-TikTok-staff
      test account, showing:
      1. Operator clicks "Connect with TikTok" on /dashboard/connect
      2. Authorizes our app on the standard TikTok login flow
      3. Returns to the dashboard with handle + avatar visible
      4. Publishes a real Reels-shaped clip from NexoClip to TikTok
      5. Verifies the post appeared on the TikTok account
      Voiceover or captions explaining what NexoClip is and why each
      step happens. TikTok is strict about the demo matching the
      live app exactly — no roadmap features.

- [ ] **Privacy policy URL**, publicly reachable. Must enumerate:
      * what data we collect from the TikTok user (open_id,
        display_name, avatar_url, access/refresh tokens)
      * what we DON'T collect (TikTok's data archive, follower lists,
        DMs — nothing past the publish surface)
      * how long we retain tokens (until revocation / disconnect)
      * who has access (NexoClip systems only, encrypted at rest)
      * deletion request flow (operator clicks Disconnect or emails
        a deletion request to a posted address)

- [ ] **Terms of Service URL**, publicly reachable.

- [ ] **Scope justification** in the form, copy-pastable:
      * `user.info.basic` — display_name + avatar shown on the
        Connect tab so the operator can confirm they connected the
        right account
      * `video.upload` — upload the operator's clip MP4 to TikTok's
        upload endpoint
      * `video.publish` — finalize + publish the uploaded video to
        the operator's TikTok timeline

- [ ] **Use-case description**: "NexoClip is a SaaS that turns
      streamer VODs into short-form clips and helps the operator
      publish those clips natively to TikTok with their own
      branding overlay. The operator authorizes NexoClip via
      standard TikTok OAuth; NexoClip uses the granted token solely
      to publish clips the operator created in NexoClip, and never
      reads, modifies, or distributes any other content on the
      operator's TikTok account."

### Submission steps

1. Move from Sandbox to Production on the app's main page
2. Fill out the audit form fields with the materials above
3. Submit + wait. TikTok usually replies with a "polish this" list
   before approving — budget for 2-3 round-trips.

### What "approved" unlocks

- Users outside the explicit Test User list can complete OAuth
- The `video.publish` call accepts non-sandbox content
- The 25-videos/day per-account cap takes effect (we already track
  it in `connected_accounts.daily_publish_count`)
- 6 req/min per-token rate cap takes effect (planned async semaphore
  in `nexoclip.publish.tiktok` — Wave 3 if not landed yet)

---

## 2. Meta (Instagram via Facebook) — App Review + Business Verification

**Portal**: developers.facebook.com → your app → App Review
**Reviewer turnaround**: 5-15 business days; Business Verification
adds another 5-10 days
**Current status**: ⬜ Not submitted

### Required materials

- [ ] **Business Verification first**. Meta gates publish permissions
      behind a verified Business. Requires:
      * legal business name
      * a tax document, utility bill, or business license
      * a corroborating public source (a website with matching
        contact info, an official directory listing)

- [ ] **Working demo video** showing the FULL Connect → Publish loop.
      Same shape as TikTok's: Connect with Instagram → operator
      picks the FB Page with linked IG-Business → handle + avatar
      appear → publish a real Reels-shaped clip → verify it appears
      on the IG account. ~3-5 minutes. Voice/captions explaining
      why we ask for each permission.

- [ ] **Permission justifications** (one entry per scope; each gets
      reviewed independently):
      * `instagram_basic` — "Display the user's IG username and
        avatar on the Connect tab so they can confirm they
        authorized the right account."
      * `instagram_content_publish` — "The core operation: publish
        clips the operator created in NexoClip to their IG Business
        account as Reels."
      * `pages_show_list` — "List the Facebook Pages the user
        administers so we can find the one with the linked IG
        Business account."
      * `pages_read_engagement` — "Required by Meta as a co-
        requisite of pages_show_list for managed Pages."
      * `business_management` — "Confirm the user's Business
        relationship to the IG Business account so we can call
        publish endpoints on the right Page."

- [ ] **Privacy policy URL** + **Terms URL** (same as TikTok; should
      already exist by then).

- [ ] **Data deletion** instructions URL — Meta specifically asks
      for a user-facing way to request deletion. Implementation:
      operator clicks Disconnect on the Connect tab (clears the
      encrypted token), OR emails a posted address for a full
      account deletion. Document both.

### Submission steps

1. Complete Business Verification (do this FIRST — App Review
   submissions before BV often get bounced)
2. App Review → Request advanced access for each permission in
   the list above
3. Each permission gets its own review — submit them as a single
   batch so reviewers can see the full flow at once

### What "approved" unlocks

- Non-test FB accounts can complete OAuth
- `instagram_content_publish` accepts non-sandbox media
- The 60-day long-lived token model takes effect (we already wire
  the proactive refresh job in
  `nexoclip.integrations.instagram.refresh`)

---

## 3. Google (YouTube Data API v3) — OAuth verification

**Portal**: console.cloud.google.com → APIs & Services → OAuth
consent screen → Verification
**Reviewer turnaround**: 4-8 weeks for full review; faster if no
sensitive scopes are involved
**Current status**: ⬜ Not submitted

### Required materials

- [ ] **App homepage URL** (the public landing — already at
      `https://nexoclip.nexo-ai.world`).

- [ ] **Privacy policy URL** + **Terms URL** (same docs as TikTok / Meta).

- [ ] **Authorized domains** in the OAuth consent screen config —
      verify ownership of `nexoclip.nexo-ai.world` via DNS TXT or
      file-based verification.

- [ ] **Justification per scope**:
      * `youtube.upload` — "Upload the operator's clip MP4 to their
        own YouTube channel. We do not read or modify any other
        content on the channel. youtube.upload is the minimum
        sufficient scope for publishing."

      Note: we deliberately ship `youtube.upload` only (no
      `youtube.readonly`) to keep this question short. The
      channelId is captured from the first videos.insert response
      in the publish adapter, not from a channels.list read at
      connect time.

- [ ] **Demo video** showing the full connect + upload loop. Google
      is more lenient than TikTok / Meta on production polish but
      still wants to see the flow.

- [ ] If `youtube.upload` is treated as a "sensitive scope" by
      Google's reviewer (it usually is for publish-class scopes),
      a **security assessment** may be required from an independent
      auditor. This is the most expensive step on this list. If we
      hit this requirement, factor 6-12 weeks AND ~$5-15k into the
      timeline.

### Submission steps

1. Configure the OAuth consent screen — pick "External" user type
2. Add `youtube.upload` to the scope list
3. Submit for verification with the materials above
4. Respond to reviewer questions promptly (Google usually replies
   in batches of ~1-2 questions per week)

### What "approved" unlocks

- Removes the "Google hasn't verified this app" warning screen
  every user otherwise sees on Connect
- Lifts the 100-user cap on the OAuth client (currently we can
  only OAuth 100 distinct Google accounts; with verification this
  becomes unlimited)
- Quota stays at the default `videos.insert` daily limit
  (10,000 uploads / day per project) — no separate quota
  application needed at this scale

---

## Pre-launch checklist (after the three approvals land)

- [ ] All three platform `status` lines above flipped to ✅ Approved
- [ ] Test the full Connect → Publish loop with a non-staff TikTok
      account, IG-Business account, and YouTube channel
- [ ] Verify the IG refresh scheduler runs in production (look for
      `instagram_refresh_drain` log lines with `refreshed > 0`)
- [ ] Verify the TikTok 25/day cap correctly blocks the 26th
      publish in a single UTC day
- [ ] Set the public privacy policy + ToS pages live at the URLs
      submitted in the review packets
- [ ] Land the Wave 2 cleanup migration: drop the
      `oauth_blob_json` plaintext mirror once the publish adapter
      reads `access_token_encrypted` directly

## Status legend

- ⬜ Not submitted
- 🟡 Submitted, awaiting reviewer response
- 🔄 Reviewer asked for changes; working on them
- ✅ Approved
- ❌ Rejected (see notes)

Update the status lines above as we move through them — they're the
single source of truth for "what's blocking go-live".
