"""TikTok platform integration — auth + publish helpers.

`auth.py` covers the Wave 1 OAuth connect flow (authorize URL,
code exchange, user-info fetch, creator-info query, refresh).
The existing `nexoclip.publish.tiktok` carries the publish-time
client; Wave 2 cleanup is expected to consolidate them once the
publisher reads the encrypted credential column directly.
"""
