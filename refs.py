"""Reference-link helpers for the pre-rendered gallery.

Deliberately dependency-free (stdlib only) so it can be imported by both app.py (gradio,
no torch) and render_gallery.py (torch, no gradio) without pulling either stack in.

Why this exists: the per-cross "reference photo" links were hand-collected hotlinks to
third-party commerce/blog/registry image files. Those rot (404), hotlink-block (403), or
were SPA soft-404s from the start. For any cross without a verified-live direct photo we
fall back to a Google Images search link, which never 404s and always surfaces real photos.
"""

from __future__ import annotations

import urllib.parse


def species_name(display: str) -> str:
    """'C. Hardyana' -> 'Cattleya Hardyana' (expand the abbreviated genus for search/labels)."""
    return display.replace("C. ", "Cattleya ", 1)


def search_url(display: str) -> str:
    """Stable Google Images search link for a cross. Used as the reference when no
    verified-live direct photo URL exists, so every cross has a working reference link."""
    query = f"{species_name(display)} orchid"
    return "https://www.google.com/search?tbm=isch&q=" + urllib.parse.quote(query)
