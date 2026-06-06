"""YouTube (Google OAuth 2.0) platform integration — auth + publish.

`auth.py` covers the Wave 2 OAuth connect flow: authorize URL with
access_type=offline + prompt=consent (mandatory for refresh_token
reliability per Google docs), code → access+refresh exchange,
on-demand refresh.

We deliberately do NOT call channels.list at connect time — that
needs the youtube.readonly scope and broadens our verification
surface. The channelId is captured from the FIRST videos.insert
response in the existing publish adapter, written back to
platform_user_id when it lands.

The existing `nexoclip.publish.youtube` carries the upload client;
Wave 2 cleanup is expected to consolidate them.
"""
