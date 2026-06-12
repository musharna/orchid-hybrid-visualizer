"""Interactive 'where the hybrid lands' latent map for the Space.

Reads the precomputed chord-plane coordinates (gallery/latent_coords.json, built offline by
build_latent_coords.py) and builds a Plotly figure per cross. Dependency-light: stdlib + plotly
only (no numpy/torch at runtime).

The figure shows, in each cross's parent chord-plane:
  - the two parent species at x = -1 / +1 (the chord), on the y=0 axis
  - the predicted F1 blend at the midpoint (0, 0)
  - for crosses with real examples in orchid-clip-v8 space: the real-hybrid centroid, offset
    perpendicular to the chord — the transgressive residual (validated Stage 16/18) — with an
    arrow from the midpoint and the off-chord angle annotated.
"""

from __future__ import annotations

import json
import os

_COORDS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "gallery", "latent_coords.json"
)

_PARENT_COLOR = "#7b2d8e"  # orchid purple
_MID_COLOR = "#888888"
_HYBRID_COLOR = "#d81b60"  # magenta


def load_coords(path: str = _COORDS) -> list[dict]:
    """Parse latent_coords.json -> list of per-cross entries. [] if absent."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)["crosses"]


def has_plane(entry: dict) -> bool:
    """True if this cross has a drawable chord-plane (both parents embeddable)."""
    return bool(entry.get("midpoint"))


def build_figure(entry: dict):
    """Build a Plotly Figure for one cross's chord-plane. Requires plotly."""
    import plotly.graph_objects as go

    fig = go.Figure()
    if not has_plane(entry):
        fig.add_annotation(
            text="Parent embeddings unavailable for this cross",
            showarrow=False,
            font=dict(size=14),
        )
        fig.update_layout(
            template="plotly_white", height=460, title=entry.get("display", "")
        )
        return fig

    parents = entry["parents"]
    px = [p["xy"][0] for p in parents]
    py = [p["xy"][1] for p in parents]
    plabels = [p["epithet"] for p in parents]
    # parents (on the chord)
    fig.add_trace(
        go.Scatter(
            x=px,
            y=py,
            mode="markers+text",
            name="parent species",
            text=plabels,
            textposition="bottom center",
            marker=dict(
                size=15,
                color=_PARENT_COLOR,
                symbol="circle",
                line=dict(width=1, color="white"),
            ),
            hovertemplate="%{text}<extra>parent</extra>",
        )
    )
    # predicted F1 blend = midpoint
    mx, my = entry["midpoint"]
    fig.add_trace(
        go.Scatter(
            x=[mx],
            y=[my],
            mode="markers+text",
            name="predicted F1 blend",
            text=["midpoint"],
            textposition="top center",
            marker=dict(size=14, color=_MID_COLOR, symbol="x"),
            hovertemplate="predicted blend (midpoint)<extra></extra>",
        )
    )
    # real hybrid (off-chord) + transgression arrow
    if entry.get("hybrid"):
        hx, hy = entry["hybrid"]
        fig.add_trace(
            go.Scatter(
                x=[mx, hx],
                y=[my, hy],
                mode="lines",
                name="transgression",
                line=dict(color=_HYBRID_COLOR, width=2, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[hx],
                y=[hy],
                mode="markers+text",
                name="real hybrid (avg)",
                text=["real hybrid"],
                textposition="top center",
                marker=dict(
                    size=16,
                    color=_HYBRID_COLOR,
                    symbol="star",
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="real-hybrid centroid<extra></extra>",
            )
        )
        if entry.get("angle_deg") is not None:
            fig.add_annotation(
                x=hx,
                y=hy,
                ax=mx,
                ay=my,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=_HYBRID_COLOR,
                text=f"{entry['angle_deg']:.0f}° off-chord",
                font=dict(size=12),
            )

    fig.update_layout(
        template="plotly_white",
        height=460,
        title=dict(text=entry["display"], x=0.5),
        xaxis=dict(
            title="parent blend axis (chord)",
            zeroline=True,
            zerolinecolor="#cccccc",
            zerolinewidth=2,
            range=[-1.6, 1.6],
        ),
        yaxis=dict(
            title="off-chord (transgressive)",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
            range=[-0.6, 1.6],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def caption(entry: dict) -> str:
    """Short markdown caption explaining the selected cross's geometry."""
    if not has_plane(entry):
        return f"**{entry['display']}** — parent embeddings unavailable."
    if entry.get("hybrid"):
        return (
            f"**{entry['display']}** — real hybrid photos sit near the parent **midpoint** along "
            f"the chord (the F1 blend), but **{entry['angle_deg']:.0f}° off-chord**: the deviation "
            f"is *transgressive* (beyond both parents), not a lean toward one parent. This is the "
            f"Stage 16/18 result, validated by permutation tests + DINOv2 replication. "
            f"_{entry['note']}._"
        )
    return (
        f"**{entry['display']}** — predicted as the F1 blend at the parent midpoint. "
        f"_No real examples of this cross in the orchid-clip-v8 hybrid set, so the transgressive "
        f"residual can't be shown here._"
    )
