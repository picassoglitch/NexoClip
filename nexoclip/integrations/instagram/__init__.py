"""Instagram platform integration — auth + (existing) publish helpers.

`auth.py` covers the Wave 2 OAuth connect flow: Facebook Login dialog →
short-lived user token → long-lived user token (~60 days) → resolve
Facebook Page → resolve linked Instagram Business account →
fetch_user_info for the post-connect display.

The existing `nexoclip.publish.instagram` carries the Reels publish
client; Wave 2 cleanup is expected to consolidate them once the
publisher reads the encrypted credential column directly.
"""
