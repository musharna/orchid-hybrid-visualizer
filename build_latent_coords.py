"""Offline precompute: project each cross into its parent 'chord-plane' for the Space's
interactive latent map. Reads the Stage-16 v8 embeddings (data/stage16/g0_arrays.npz) and
writes space/gallery/latent_coords.json — small, numpy-free at runtime.

The chord-plane makes the Stage 16/18 result visible per cross:
  x-axis = the parent-A -> parent-B direction (the 'chord'), origin at the weighted midpoint
  y-axis = the component of the real-hybrid centroid PERPENDICULAR to the chord (off-chord)
So both parents and the predicted midpoint lie on y=0, and a real hybrid's vertical offset IS
the transgression (Stelkens 2009; Stage 16 G1a rho parent-divergence~off-chord +0.518).

Crosses with no real examples in the 1002-hybrid set get parents+midpoint on the chord only
(hybrid=null). Distances are cosine-space (embeddings L2-normalized); coords are reported as a
fraction of the chord half-length so the scale is comparable across crosses.

Run from the repo root:  python3 space/build_latent_coords.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, REPO_ROOT)

from crosses import CROSSES  # noqa: E402

NPZ = os.path.join(REPO_ROOT, "data", "stage16", "g0_arrays.npz")
OUT = os.path.join(THIS_DIR, "gallery", "latent_coords.json")


def _epithet(name: str) -> str:
    """'Cattleya warneri' / 'Guarianthe bowringiana' -> 'warneri' / 'bowringiana'."""
    parts = str(name).strip().lower().split()
    return parts[-1] if parts else ""


def _norm_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():
    d = np.load(NPZ, allow_pickle=True)
    H, P1, P2 = d["H"], d["P1"], d["P2"]
    p1_ids, p2_ids, grex_ids = d["p1_ids"], d["p2_ids"], d["grex_ids"]

    # species centroid (mean parent embedding across all rows naming that species), unit-norm
    sp_rows: dict[str, list] = {}
    for ids, E in ((p1_ids, P1), (p2_ids, P2)):
        for i, s in enumerate(ids):
            sp_rows.setdefault(_epithet(s), []).append(E[i])
    sp_cent = {k: _norm_unit(np.mean(v, axis=0)) for k, v in sp_rows.items()}

    # real-hybrid centroid per grex (by normalized full name), unit-norm
    grex_rows: dict[str, list] = {}
    for i, g in enumerate(grex_ids):
        key = str(g).lower().replace("cattleya", "").strip()
        grex_rows.setdefault(key, []).append(H[i])
    grex_cent = {k: _norm_unit(np.mean(v, axis=0)) for k, v in grex_rows.items()}

    out = []
    n_real = 0
    for stem, display, ancestry, _ref in CROSSES:
        # available parent centroids + weights
        parents = []
        for sp, pct in ancestry.items():
            ep = sp.replace("_aurea", "")
            c = sp_cent.get(ep)
            if c is not None:
                parents.append((sp, float(pct), c))
        entry = {
            "stem": stem,
            "display": display,
            "hybrid": None,
            "parents": [],
            "midpoint": None,
            "angle_deg": None,
            "note": "",
        }
        if len(parents) < 2:
            entry["note"] = "parent embeddings unavailable"
            out.append(entry)
            continue

        # two highest-weight parents anchor the chord; midpoint = weighted blend over all available
        parents.sort(key=lambda t: -t[1])
        (aN, aW, aC), (bN, bW, bC) = parents[0], parents[1]
        wsum = sum(p[1] for p in parents)
        mid = _norm_unit(sum(p[1] / wsum * p[2] for p in parents))

        u = _norm_unit(bC - aC)  # chord direction
        half = float(np.dot(bC - aC, u)) / 2.0  # half chord length (cosine-space)
        if half <= 0:
            entry["note"] = "degenerate chord"
            out.append(entry)
            continue

        def coords(vec, v_axis=None):
            x = float(np.dot(vec - mid, u)) / half
            y = 0.0 if v_axis is None else float(np.dot(vec - mid, v_axis)) / half
            return [round(x, 4), round(y, 4)]

        # real hybrid -> defines the off-chord (v) axis
        gkey = display.lower().replace("c.", "").strip()
        hyb = None
        for k, c in grex_cent.items():
            if k == gkey:
                hyb = c
                break
        v_axis = None
        if hyb is not None:
            resid = hyb - mid
            off = resid - np.dot(resid, u) * u
            if np.linalg.norm(off) > 1e-6:
                v_axis = _norm_unit(off)
            n_real += 1

        entry["parents"] = [
            {
                "name": aN,
                "epithet": aN.replace("_aurea", ""),
                "weight": aW,
                "xy": coords(aC, v_axis),
            },
            {
                "name": bN,
                "epithet": bN.replace("_aurea", ""),
                "weight": bW,
                "xy": coords(bC, v_axis),
            },
        ]
        # any extra parents (3-4 parent complex crosses) plotted too, on the chord
        for eN, eW, eC in parents[2:]:
            entry["parents"].append(
                {
                    "name": eN,
                    "epithet": eN.replace("_aurea", ""),
                    "weight": eW,
                    "xy": coords(eC, v_axis),
                }
            )
        entry["midpoint"] = coords(mid, v_axis)
        if hyb is not None:
            entry["hybrid"] = coords(hyb, v_axis)
            # transgression angle: angle of (hybrid - midpoint) off the chord
            r = hyb - mid
            ang = np.degrees(
                np.arctan2(
                    np.linalg.norm(r - np.dot(r, u) * u), abs(np.dot(r, u)) + 1e-9
                )
            )
            entry["angle_deg"] = round(float(ang), 1)
            entry["note"] = f"real examples: {len(grex_rows[gkey])}"
        else:
            entry["note"] = "no real examples in 1002-hybrid set"
        out.append(entry)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(
            {"crosses": out, "space": "v8 (orchid-clip-v8) latent, cosine"}, f, indent=2
        )

    plotted = sum(1 for e in out if e["midpoint"])
    print(
        f"wrote {OUT}: {len(out)} crosses, {plotted} with chord-plane, {n_real} with real-hybrid off-chord"
    )
    for e in out:
        if e["hybrid"]:
            print(
                f"  {e['display']:24s} off-chord angle {e['angle_deg']:5.1f}°  hybrid y={e['hybrid'][1]:+.3f}"
            )


if __name__ == "__main__":
    main()
