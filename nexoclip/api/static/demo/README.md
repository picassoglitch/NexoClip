# Landing hero demo thumbnails

The five clip cards in the landing hero (the "1 stream → 5 platforms"
animation) render real images from this folder. Drop your files here:

| File        | Card / badge |
|-------------|--------------|
| `clip1.jpg` | TikTok       |
| `clip2.jpg` | Reels        |
| `clip3.jpg` | Shorts       |
| `clip4.jpg` | Kick         |
| `clip5.jpg` | Twitch       |

## Sizing

- **Aspect ratio:** 9:16 (vertical). The card crops to this with
  `object-fit: cover`, focal point ~28% from the top (faces sit well).
- **Recommended:** 360×640 px, optimized JPG/WebP, < ~80 KB each so the
  hero stays fast. (`.jpg` is what the template references; rename or
  change the `src` in `landing.html` if you use `.webp`.)

## Missing files are safe

If a file isn't present, the `<img>` removes itself (`onerror`) and the
original gradient placeholder shows. The page never breaks on a missing
asset — so you can add these one at a time.

## ⚠️ Rights / licensing — read before adding faces

These appear on the public, commercial landing page. Using a real,
identifiable person's image here implies they endorse / use NexoClip and
carries real legal exposure:

- **Right of publicity** — commercial use of someone's likeness needs
  consent.
- **False endorsement** (Lanham Act §43(a)) — implying a sponsorship
  that doesn't exist.
- **Copyright** — the photo itself is owned by whoever shot it.

Only drop in images you actually have the rights to use commercially:
your own creators/clients (with written permission), licensed stock of
content creators, or AI-generated faces (no real person). Do **not** use
scraped photos of famous streamers.
