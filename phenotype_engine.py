"""Phenotype Blending Engine for Cattleya orchid hybrids.

Loads species trait profiles and dominance rules, then produces
natural-language appearance descriptions for arbitrary ancestry blends.
"""

from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ordinal scales for interpolation
# ---------------------------------------------------------------------------

SIZE_SCALE = ["miniature", "small", "small-medium", "medium", "medium-large", "large"]
SUBSTANCE_SCALE = ["thin", "thin-medium", "medium", "medium-heavy", "heavy"]
SATURATION_SCALE = ["low", "medium", "high"]
PIGMENT_INTENSITY_SCALE = ["none", "low", "medium", "high"]

# Default trait profile used when a species is not found in the database.
DEFAULT_TRAITS: dict[str, Any] = {
    "color": {
        "primary": "lavender",
        "secondary": None,
        "lip": "magenta with darker markings",
        "saturation": "medium",
        "distribution": "even",
    },
    "form": {
        "petal_shape": "rounded",
        "symmetry": "good",
        "lip_shape": "medium",
        "flatness": "moderately flat",
    },
    "size": "medium",
    "substance": "medium",
    "texture": "waxy",
}


class PhenotypeEngine:
    """Blend orchid phenotype traits from an ancestry dictionary."""

    def __init__(
        self,
        phenotype_db_path: str | Path | None = None,
        dominance_rules_path: str | Path | None = None,
    ) -> None:
        base = Path(__file__).resolve().parent
        phenotype_db_path = phenotype_db_path or base / "phenotype_db.json"
        dominance_rules_path = dominance_rules_path or base / "dominance_rules.json"

        with open(phenotype_db_path) as f:
            self.db: dict[str, dict] = json.load(f)

        with open(dominance_rules_path) as f:
            raw_rules: dict = json.load(f)

        # Strip documentation keys (prefixed with _)
        self.rules: dict[str, dict] = {
            k: v for k, v in raw_rules.items() if not k.startswith("_")
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def blend(
        self,
        ancestry: dict[str, float],
        generation: int = 1,
        forms: dict[str, str] | None = None,
        transgression: float = 0.0,
    ) -> dict[str, Any]:
        """Return a merged trait dict for the given ancestry proportions.

        *ancestry* maps species names to percentages (should sum to ~100).
        Percentages are normalised internally.

        *generation* controls recessive expression thresholds:
          F1 (1) = 80%  — default, backward compatible
          F2 (2) = 50%  — recessives can show in 50/50 crosses
          F3+(3) = 25%  — recessives show at lower ancestry

        *forms* maps species names to AOS color form names (e.g.,
        ``{"violacea": "coerulea"}``).  Available forms vary by species;
        use :meth:`list_forms` to query.

        *transgression* in [0, 1] is the Stage 17 transgressive-segregation knob: it
        pushes pigment intensity / saturation BEYOND the parental range, scaled by the
        parental divergence (`_trait_divergence`). 0 (default) = exact prior behaviour.
        See specs/stage17_generator_transgression.md. NB: this is the *magnitude* control
        applied in a default extremity direction; the species-structured *direction*
        (Phase 2) is not yet modelled, and the coarse 4-level pigment scale caps the
        effect for crosses whose stronger parent is already at the scale ceiling.
        """
        if not ancestry:
            raise ValueError("ancestry must not be empty")

        recessive_threshold = max(25, 80 - (generation - 1) * 30)

        ancestry = self._normalise(ancestry)
        profiles = self._resolve_profiles(ancestry)

        # Apply color form overrides before merging
        if forms:
            for sp in profiles:
                form_name = self._match_form_key(sp, forms)
                if form_name:
                    profiles[sp] = self._apply_form(profiles[sp], sp, form_name)

        merge_profiles = self._suppress_recessive(
            ancestry, profiles, recessive_threshold
        )

        merged = self._weighted_merge(ancestry, merge_profiles)
        merged = self._apply_dominance(ancestry, profiles, merged, recessive_threshold)
        if transgression and transgression > 0:
            merged = self._apply_transgression(
                ancestry, profiles, merged, transgression
            )
        return merged

    def describe(
        self,
        ancestry: dict[str, float],
        generation: int = 1,
        forms: dict[str, str] | None = None,
        transgression: float = 0.0,
    ) -> str:
        """Return a natural-language appearance string for *ancestry*."""
        traits = self.blend(
            ancestry,
            generation=generation,
            forms=forms,
            transgression=transgression,
        )
        description = self._compose_description(traits)
        # Prepend parentage for CLIP differentiation
        parentage = self._parentage_prefix(ancestry)
        if parentage:
            return f"{parentage}. {description}"
        return description

    def describe_anchor(
        self,
        ancestry: dict[str, float],
        generation: int = 1,
        forms: dict[str, str] | None = None,
    ) -> str:
        """Like :meth:`describe`, but emit only the anchor + shape skeleton.

        Colors (lip color, petal color, distribution, secondary markings) are
        omitted. Intended to leave room in the 77-token CLIP budget for a
        PSM-derived trait suffix (see :func:`trait_prompt.compose_appearance`).
        """
        traits = self.blend(ancestry, generation=generation, forms=forms)
        size = traits.get("size", "medium")
        form = traits.get("form", {})
        parts = [f"photograph of a {size} cattleya orchid plant in flower"]

        lip_shape = (self._simplify_blend(form.get("lip_shape", "")) or "").lower()
        if any(k in lip_shape for k in ("small", "tubular", "narrow")):
            parts.append("small lip")
        elif any(k in lip_shape for k in ("large", "frilled", "ruffled")):
            parts.append("large ruffled lip")
        else:
            parts.append("lip")

        petal_shape = self._simplify_shape(form.get("petal_shape", ""))
        if petal_shape:
            parts.append(f"{petal_shape} petals")
        else:
            parts.append("petals")

        description = ", ".join(parts)
        parentage = self._parentage_prefix(ancestry)
        return f"{parentage}. {description}" if parentage else description

    def list_forms(self, species: str) -> list[str]:
        """Return available AOS color form names for a species."""
        profile = self._lookup(self.db, species)
        if not profile:
            return []
        forms = profile.get("color", {}).get("forms", {})
        return list(forms.keys())

    def negative_for(self, ancestry: dict[str, float]) -> str:
        """Return extra negative-prompt terms to counteract LoRA color bias.

        When the blended color is warm (red, orange, yellow, coral, salmon),
        returns terms to suppress the LoRA's purple/lavender tendency.
        When the blended color is cool/purple, returns terms to suppress
        warm color bleed. Returns empty string when no bias correction needed.
        """
        traits = self.blend(ancestry)
        primary = (traits.get("color", {}).get("primary") or "").lower()
        lip = (traits.get("color", {}).get("lip") or "").lower()
        combined = f"{primary} {lip}"

        warm_kw = (
            "orange",
            "red",
            "scarlet",
            "coral",
            "salmon",
            "burgundy",
            "crimson",
            "vermillion",
            "golden",
            "yellow",
            "amber",
        )
        cool_kw = ("lavender", "purple", "violet", "lilac", "mauve", "magenta")

        primary_warm = any(kw in primary for kw in warm_kw)
        primary_cool = any(kw in primary for kw in cool_kw)

        # If primary color is warm (regardless of lip), suppress purple bias
        if primary_warm and not primary_cool:
            return "purple, lavender, violet, lilac, mauve"
        return ""

    @staticmethod
    def _parentage_prefix(ancestry: dict[str, float]) -> str:
        """Build 'cross of Cattleya X and Cattleya Y' prefix from ancestry."""
        if not ancestry:
            return ""
        # Format species names: "dowiana_aurea" -> "dowiana aurea"
        names = []
        for sp in sorted(ancestry, key=lambda s: ancestry[s], reverse=True):
            clean = sp.replace("_", " ")
            # Add genus if not already present
            if not clean.startswith(("Cattleya", "Guarianthe")):
                clean = f"Cattleya {clean}"
            names.append(clean)
        if len(names) == 1:
            return f"cattleya orchid, {names[0]}"
        elif len(names) == 2:
            return f"cattleya hybrid orchid, cross of {names[0]} and {names[1]}"
        else:
            return f"cattleya hybrid orchid, cross of {', '.join(names[:-1])} and {names[-1]}"

    # ------------------------------------------------------------------
    # Normalisation & profile look-up
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(ancestry: dict[str, float]) -> dict[str, float]:
        total = sum(ancestry.values())
        if total == 0:
            raise ValueError("ancestry percentages sum to zero")
        return {sp: pct / total * 100 for sp, pct in ancestry.items()}

    def _lookup(self, store: dict, name: str) -> Any | None:
        """Look up a species in a dict, trying multiple name forms.

        Handles short names (``labiata``), underscored names
        (``dowiana_aurea``), and full names (``Cattleya dowiana aurea``).
        """
        # Exact match
        if name in store:
            return store[name]
        # Replace underscores with spaces for multi-word epithets
        clean = name.replace("_", " ")
        if clean in store:
            return store[clean]
        for genus in ("Cattleya", "Guarianthe"):
            full = f"{genus} {clean}"
            if full in store:
                return store[full]
        return None

    def _resolve_profiles(
        self, ancestry: dict[str, float]
    ) -> dict[str, dict[str, Any]]:
        """Return trait profiles for each species, falling back to defaults."""
        out: dict[str, dict] = {}
        for sp in ancestry:
            profile = self._lookup(self.db, sp)
            if profile is not None:
                # Strip non-trait keys like "source"
                out[sp] = {
                    k: copy.deepcopy(v)
                    for k, v in profile.items()
                    if k in ("color", "form", "size", "substance", "texture")
                }
            else:
                out[sp] = copy.deepcopy(DEFAULT_TRAITS)
        return out

    def _apply_form(self, profile: dict, species: str, form_name: str) -> dict:
        """Apply a named AOS color form's overrides to a species profile.

        Form overrides are sparse — they only contain fields that differ
        from the base (tipo) profile.  Returns a new profile with the
        overrides applied.
        """
        db_entry = self._lookup(self.db, species)
        if not db_entry:
            raise ValueError(f"Unknown species: {species}")

        forms = db_entry.get("color", {}).get("forms", {})
        if form_name not in forms:
            available = list(forms.keys())
            raise ValueError(
                f"Unknown form '{form_name}' for {species}. Available: {available}"
            )

        overrides = forms[form_name]
        if not overrides:
            return profile  # tipo — no changes

        result = copy.deepcopy(profile)
        color = result.get("color", {})

        # Apply each override field
        for key, value in overrides.items():
            if key == "pigments":
                color["pigments"] = copy.deepcopy(value)
            else:
                color[key] = value

        result["color"] = color
        return result

    @staticmethod
    def _match_form_key(species: str, forms: dict[str, str]) -> str | None:
        """Match a species name to a forms dict, trying short/full name forms."""
        if species in forms:
            return forms[species]
        clean = species.replace("_", " ")
        if clean in forms:
            return forms[clean]
        # Try short name (epithet only)
        for key, val in forms.items():
            key_clean = key.replace("_", " ")
            if species.endswith(key_clean) or key_clean.endswith(
                species.split()[-1] if " " in species else species
            ):
                return val
        return None

    def _suppress_recessive(
        self,
        ancestry: dict[str, float],
        profiles: dict[str, dict],
        recessive_threshold: int = 80,
    ) -> dict[str, dict]:
        """Return profiles with recessive traits nullified for the merge step.

        Recessive traits should not participate in blending unless the species
        is at >= *recessive_threshold* or all parents share the same value.
        We replace recessive trait values with the other parent's value so the
        merge ignores them.
        """
        out = copy.deepcopy(profiles)
        for species, pct in ancestry.items():
            rules = self._lookup(self.rules, species)
            if not rules:
                continue
            for trait_path, rule in rules.items():
                if rule.get("type") != "recessive":
                    continue
                value = rule.get("value") or self._get_nested(
                    profiles[species], trait_path
                )
                if value is None:
                    continue
                # Check if recessive would express
                if pct >= recessive_threshold:
                    continue  # let it merge normally
                if all(
                    self._get_nested(profiles[sp], trait_path) == value
                    for sp in ancestry
                ):
                    continue  # all parents share the value
                # Suppress: replace with the dominant parent's value
                for other_sp, other_pct in ancestry.items():
                    if other_sp != species and other_pct >= pct:
                        other_val = self._get_nested(profiles[other_sp], trait_path)
                        if other_val is not None:
                            self._set_nested(out[species], trait_path, other_val)
                            break
        return out

    # ------------------------------------------------------------------
    # Weighted merge
    # ------------------------------------------------------------------

    def _weighted_merge(
        self,
        ancestry: dict[str, float],
        profiles: dict[str, dict],
    ) -> dict[str, Any]:
        """Merge profiles weighted by ancestry percentages."""
        ranked = sorted(ancestry.items(), key=lambda x: x[1], reverse=True)
        primary_sp, primary_pct = ranked[0]
        base = copy.deepcopy(profiles[primary_sp])

        if len(ranked) == 1:
            return base

        # Merge secondary parents in
        for sec_sp, sec_pct in ranked[1:]:
            sec = profiles[sec_sp]
            base = self._merge_pair(base, primary_pct, sec, sec_pct, ancestry)

        return base

    def _merge_pair(
        self,
        base: dict,
        base_pct: float,
        sec: dict,
        sec_pct: float,
        full_ancestry: dict[str, float],
    ) -> dict:
        """Merge a secondary parent into the base traits."""
        result = copy.deepcopy(base)

        # --- Color ---
        result["color"] = self._merge_color(
            base.get("color", {}), sec.get("color", {}), base_pct, sec_pct
        )

        # --- Size / Substance (ordinal interpolation) ---
        result["size"] = self._interpolate_ordinal(
            SIZE_SCALE,
            base.get("size", "medium"),
            sec.get("size", "medium"),
            base_pct,
            sec_pct,
        )
        result["substance"] = self._interpolate_ordinal(
            SUBSTANCE_SCALE,
            base.get("substance", "medium"),
            sec.get("substance", "medium"),
            base_pct,
            sec_pct,
        )

        # --- Form (primary dominates unless secondary >= 40%) ---
        if sec_pct >= 40:
            result["form"] = self._merge_form(
                base.get("form", {}), sec.get("form", {}), base_pct, sec_pct
            )
        # else: keep base form

        # --- Texture (primary dominates unless secondary >= 40%) ---
        if sec_pct >= 40:
            base_tex = base.get("texture", "waxy")
            sec_tex = sec.get("texture", "waxy")
            if base_tex != sec_tex:
                result["texture"] = f"{base_tex} to {sec_tex}"
            else:
                result["texture"] = base_tex

        return result

    # ------------------------------------------------------------------
    # Sub-merges
    # ------------------------------------------------------------------

    def _merge_color(
        self,
        base_color: dict,
        sec_color: dict,
        base_pct: float,
        sec_pct: float,
    ) -> dict:
        result = copy.deepcopy(base_color)

        # --- Pigment-channel merge (preferred when data available) ---
        base_pig = base_color.get("pigments")
        sec_pig = sec_color.get("pigments")
        if base_pig and sec_pig:
            merged_pig = self._merge_pigments(base_pig, sec_pig, base_pct, sec_pct)
            result["pigments"] = merged_pig
            # Generate primary color string from merged pigments
            pig_desc = self._pigments_to_description(merged_pig)
            if pig_desc:
                result["primary"] = pig_desc
                # Skip string-based primary blending below
                bp = None
                sp = None
            else:
                bp = base_color.get("primary", "")
                sp = sec_color.get("primary", "")
        else:
            bp = base_color.get("primary", "")
            sp = sec_color.get("primary", "")

        # Primary colour blending (string fallback)
        if bp and sp and bp != sp:
            if 45 <= sec_pct <= 55:
                result["primary"] = f"{bp} blending with {sp}"
            elif sec_pct >= 20:
                result["primary"] = f"{bp} with {sp} tones"

        # Secondary colour: inherit if present
        sec_secondary = sec_color.get("secondary")
        if sec_secondary and sec_pct >= 20:
            if result.get("secondary"):
                result["secondary"] = f"{result['secondary']} and {sec_secondary}"
            else:
                result["secondary"] = sec_secondary

        # Lip blending
        bl = base_color.get("lip", "")
        sl = sec_color.get("lip", "")
        if bl and sl and bl != sl and sec_pct >= 30:
            result["lip"] = f"{bl} blending with {sl}"

        # Saturation interpolation
        result["saturation"] = self._interpolate_ordinal(
            SATURATION_SCALE,
            base_color.get("saturation", "medium"),
            sec_color.get("saturation", "medium"),
            base_pct,
            sec_pct,
        )

        # Distribution: keep base unless secondary >= 40%
        if sec_pct >= 40:
            sd = sec_color.get("distribution")
            bd = base_color.get("distribution")
            if sd and bd and sd != bd:
                result["distribution"] = f"{bd} to {sd}"

        return result

    def _merge_pigments(
        self,
        base_pig: dict,
        sec_pig: dict,
        base_pct: float,
        sec_pct: float,
    ) -> dict:
        """Merge pigment channels independently.

        Each channel (anthocyanin, carotenoid, co_pigment) blends intensity on
        an ordinal scale.  Hue follows the higher-intensity parent.  Pattern
        follows existing distribution merge rules.
        """
        result = {}
        for channel in ("anthocyanin", "carotenoid", "co_pigment"):
            bc = base_pig.get(channel, {})
            sc = sec_pig.get(channel, {})
            b_int = bc.get("intensity", "none")
            s_int = sc.get("intensity", "none")

            merged_intensity = self._interpolate_ordinal(
                PIGMENT_INTENSITY_SCALE, b_int, s_int, base_pct, sec_pct
            )

            # Hue: use whichever parent has higher intensity; if tied, use base
            b_idx = (
                PIGMENT_INTENSITY_SCALE.index(b_int)
                if b_int in PIGMENT_INTENSITY_SCALE
                else 0
            )
            s_idx = (
                PIGMENT_INTENSITY_SCALE.index(s_int)
                if s_int in PIGMENT_INTENSITY_SCALE
                else 0
            )
            if s_idx > b_idx:
                merged_hue = sc.get("hue")
            elif b_idx > s_idx:
                merged_hue = bc.get("hue")
            else:
                # Tied intensity — prefer base parent's hue
                merged_hue = bc.get("hue") or sc.get("hue")

            # If merged intensity is "none", clear hue
            if merged_intensity == "none":
                merged_hue = None

            result[channel] = {"hue": merged_hue, "intensity": merged_intensity}

        # Pattern: keep base unless secondary >= 40%
        bp = base_pig.get("pattern", {})
        sp = sec_pig.get("pattern", {})
        if sec_pct >= 40 and sp.get("type") and sp["type"] != bp.get("type"):
            result["pattern"] = {
                "type": sp["type"],
                "contrast": sp.get("contrast", bp.get("contrast", "low")),
            }
        else:
            result["pattern"] = (
                copy.deepcopy(bp) if bp else {"type": "even", "contrast": "low"}
            )

        return result

    @staticmethod
    def _pigments_to_description(pigments: dict) -> str:
        """Convert structured pigment data to a natural-language color string.

        Examples:
          anthocyanin=magenta/high, carotenoid=none -> "magenta"
          anthocyanin=red/high, carotenoid=orange/medium -> "red-orange"
          anthocyanin=none, carotenoid=yellow/medium -> "yellow"
          anthocyanin=none, carotenoid=none -> ""
        """
        anth = pigments.get("anthocyanin", {})
        caro = pigments.get("carotenoid", {})
        co = pigments.get("co_pigment", {})

        anth_hue = anth.get("hue")
        anth_int = anth.get("intensity", "none")
        caro_hue = caro.get("hue")
        caro_int = caro.get("intensity", "none")
        co_hue = co.get("hue")
        co_int = co.get("intensity", "none")

        parts = []

        # Intensity prefix
        int_scale = {"high": "deep", "medium": "", "low": "pale", "none": ""}

        if anth_hue and anth_int != "none":
            prefix = int_scale.get(anth_int, "")
            parts.append(f"{prefix} {anth_hue}".strip() if prefix else anth_hue)

        if caro_hue and caro_int != "none":
            prefix = int_scale.get(caro_int, "")
            caro_str = f"{prefix} {caro_hue}".strip() if prefix else caro_hue
            parts.append(caro_str)

        if co_hue and co_int != "none":
            parts.append(co_hue)

        if not parts:
            return ""

        if len(parts) == 1:
            return parts[0]

        # Combine: "magenta with orange tones" or "red-orange"
        if len(parts) == 2:
            # If both are single-word hues, hyphenate
            if " " not in parts[0] and " " not in parts[1]:
                return f"{parts[0]}-{parts[1]}"
            return f"{parts[0]} with {parts[1]} tones"

        return f"{parts[0]} with {parts[1]} and {parts[2]} tones"

    def _merge_form(
        self, base_form: dict, sec_form: dict, base_pct: float, sec_pct: float
    ) -> dict:
        result = copy.deepcopy(base_form)
        for key in ("petal_shape", "lip_shape"):
            bv = base_form.get(key, "")
            sv = sec_form.get(key, "")
            if bv and sv and bv != sv:
                result[key] = f"{bv} to {sv}"
        return result

    # ------------------------------------------------------------------
    # Ordinal interpolation
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate_ordinal(
        scale: list[str],
        val_a: str,
        val_b: str,
        pct_a: float,
        pct_b: float,
    ) -> str:
        """Weighted interpolation between two values on an ordinal scale."""

        def idx(v: str) -> int:
            v_lower = v.strip().lower()
            for i, s in enumerate(scale):
                if s == v_lower:
                    return i
            # Fuzzy: check containment
            for i, s in enumerate(scale):
                if v_lower in s or s in v_lower:
                    return i
            return len(scale) // 2  # fallback to middle

        ia = idx(val_a)
        ib = idx(val_b)
        total = pct_a + pct_b
        if total == 0:
            return scale[len(scale) // 2]
        wi = (ia * pct_a + ib * pct_b) / total
        return scale[round(wi)]

    # ------------------------------------------------------------------
    # Dominance overrides
    # ------------------------------------------------------------------

    def _apply_dominance(
        self,
        ancestry: dict[str, float],
        profiles: dict[str, dict],
        merged: dict[str, Any],
        recessive_threshold: int = 80,
    ) -> dict[str, Any]:
        result = copy.deepcopy(merged)

        for species, pct in ancestry.items():
            rules = self._lookup(self.rules, species)
            if not rules:
                continue
            profile = profiles[species]

            for trait_path, rule in rules.items():
                rtype = rule.get("type")

                if rtype == "dominant":
                    if pct >= 10:
                        value = rule.get("value") or self._get_nested(
                            profile, trait_path
                        )
                        if value is not None:
                            self._set_nested(result, trait_path, value)

                elif rtype == "recessive":
                    # Only express at >= recessive_threshold, or when all
                    # parents carry the same value for that trait.
                    value = rule.get("value") or self._get_nested(profile, trait_path)
                    if pct >= recessive_threshold:
                        if value is not None:
                            self._set_nested(result, trait_path, value)
                    else:
                        # Check if all parents share this trait value
                        if value is not None and all(
                            self._get_nested(profiles[sp], trait_path) == value
                            for sp in ancestry
                        ):
                            self._set_nested(result, trait_path, value)

                elif rtype == "semi_dominant":
                    # Shows in F1 crosses when above threshold %
                    threshold = rule.get("threshold", 30)
                    if pct >= threshold:
                        value = rule.get("value")
                        if value is not None:
                            self._set_nested(result, trait_path, value)

                elif rtype == "modifier":
                    effect = rule.get("effect", "")
                    if effect == "intensifies_lavender" and pct >= 15:
                        self._apply_lavender_intensifier(result, pct)

        return result

    @staticmethod
    def _apply_lavender_intensifier(traits: dict, dowiana_pct: float) -> None:
        """If the current colour blend contains lavender/pink, shift toward
        richer purple proportional to dowiana percentage.

        Exception: when the base colour is pale (white, cream, pale), dowiana's
        yellow influence shows through instead of intensifying to purple.
        This matches real-world crosses like Triumphans (dowiana × trianae).
        """
        color = traits.get("color", {})
        primary = (color.get("primary") or "").lower()
        lavender_keywords = ("lavender", "lilac", "rose", "mauve")
        exclude_keywords = ("olive", "green", "bronze", "brown", "orange", "yellow")
        pale_keywords = ("white", "pale", "cream")
        if any(kw in primary for kw in lavender_keywords) and not any(
            kw in primary for kw in exclude_keywords
        ):
            # If base is pale, dowiana yellow dominates — no purple shift
            if any(kw in primary for kw in pale_keywords):
                if dowiana_pct >= 50:
                    color["primary"] = "cream-yellow with golden tones"
                else:
                    color["primary"] = "warm ivory with golden flush"
            else:
                if dowiana_pct >= 75:
                    color["primary"] = "deep rich purple"
                elif dowiana_pct >= 50:
                    color["primary"] = "rich lavender-purple"
                else:
                    color["primary"] = "intensified lavender-purple"

    # ------------------------------------------------------------------
    # Stage 17 — transgressive segregation (off-parental-range novelty)
    # ------------------------------------------------------------------

    def _trait_divergence(
        self, ancestry: dict[str, float], profiles: dict[str, dict]
    ) -> float:
        """Parental divergence proxy in [0, 1] from the two dominant parents' pigment
        channels: mean normalized ordinal intensity distance across the three channels,
        plus a small hue-mismatch term. 0 when the two parents are pigment-identical (no
        transgression), rising with how different they are. Self-cross / single parent → 0.

        This is the trait-space analog of the Stage 16 embedding divergence δ that drove
        the off-chord transgression magnitude (Stage 17 Phase 0 confirmed the law survives
        into these axes); kept self-contained so the generator needs no embeddings.
        """
        ranked = sorted(ancestry.items(), key=lambda x: x[1], reverse=True)
        if len(ranked) < 2:
            return 0.0
        pa = profiles.get(ranked[0][0], {}).get("color", {}).get("pigments", {}) or {}
        pb = profiles.get(ranked[1][0], {}).get("color", {}).get("pigments", {}) or {}
        top = len(PIGMENT_INTENSITY_SCALE) - 1
        inten_d, hue_d = [], 0.0
        for ch in ("anthocyanin", "carotenoid", "co_pigment"):
            ca, cb = pa.get(ch, {}) or {}, pb.get(ch, {}) or {}
            ia = self._intensity_idx(ca.get("intensity", "none"))
            ib = self._intensity_idx(cb.get("intensity", "none"))
            inten_d.append(abs(ia - ib) / top)
            ha, hb = ca.get("hue"), cb.get("hue")
            if ha and hb and ha != hb:
                hue_d += 0.5 / 3.0
        base = sum(inten_d) / len(inten_d) if inten_d else 0.0
        return float(min(1.0, base + hue_d))

    @staticmethod
    def _intensity_idx(value: str) -> int:
        v = (value or "none").strip().lower()
        return PIGMENT_INTENSITY_SCALE.index(v) if v in PIGMENT_INTENSITY_SCALE else 0

    def _apply_transgression(
        self,
        ancestry: dict[str, float],
        profiles: dict[str, dict],
        merged: dict[str, Any],
        strength: float,
    ) -> dict[str, Any]:
        """Push pigment intensity (and saturation) BEYOND the parental range, scaled by
        ``m = clamp01(strength) * _trait_divergence``. Amplifies EXISTING (non-``none``)
        channels toward the extreme — does NOT invent novel channels (that, and the
        species-structured *direction*, is Phase 2). Divergence-gated (identical parents →
        no change), clamped to the scale ceiling, and a strict no-op when ``m == 0`` so
        ``strength=0`` reproduces prior behaviour byte-for-byte.
        """
        ranked = sorted(ancestry.items(), key=lambda x: x[1], reverse=True)
        parents = [profiles[s] for s, _ in ranked[:2] if s in profiles]
        if len(parents) < 2:
            return merged
        m = max(0.0, min(1.0, float(strength))) * self._trait_divergence(
            ancestry, profiles
        )
        if m <= 0.0:
            return merged

        result = copy.deepcopy(merged)
        top = len(PIGMENT_INTENSITY_SCALE) - 1
        pigs = result.get("color", {}).get("pigments")
        changed = False
        if pigs:
            for ch in ("anthocyanin", "carotenoid", "co_pigment"):
                cur = pigs.get(ch)
                if not cur:
                    continue
                cidx = self._intensity_idx(cur.get("intensity", "none"))
                if cidx == 0:  # don't invent a channel absent in the merge (Phase 2)
                    continue
                maxp = max(
                    self._intensity_idx(
                        (p.get("color", {}).get("pigments", {}).get(ch, {}) or {}).get(
                            "intensity", "none"
                        )
                    )
                    for p in parents
                )
                headroom = top - maxp
                if headroom <= 0:  # stronger parent already at scale ceiling
                    continue
                steps = min(headroom, int(m * (headroom + 1)))
                new = max(cidx, maxp + steps)
                if new != cidx:
                    cur["intensity"] = PIGMENT_INTENSITY_SCALE[new]
                    changed = True
            if changed:
                desc = self._pigments_to_description(pigs)
                if desc:
                    result.setdefault("color", {})["primary"] = desc

        # Saturation: amplify toward the high end (3-level scale), beyond the merge value.
        sat = result.get("color", {}).get("saturation")
        if sat in SATURATION_SCALE:
            sidx = SATURATION_SCALE.index(sat)
            shead = (len(SATURATION_SCALE) - 1) - sidx
            if shead > 0:
                result["color"]["saturation"] = SATURATION_SCALE[
                    min(len(SATURATION_SCALE) - 1, sidx + int(m * (shead + 1)))
                ]

        # Continuous, ceiling-free magnitude surface: a divergence-graded descriptive
        # intensifier on the dominant colour (the coarse 4-level pigment ordinal is usually
        # saturated for showy Cattleyas, so the push above rarely fires). MAGNITUDE = the
        # intensifier; DIRECTION (Phase 2) = a per-pair modality cue predicted from the
        # parents' off-chord direction table (intensity / hue / texture; identity heldout
        # cos +0.412). Missing table / unknown species → intensifier only. Idempotent.
        prim = result.get("color", {}).get("primary")
        if prim and isinstance(prim, str) and m >= 0.3:
            if not prim.lower().startswith(("intensely ", "vivid ")):
                prim = ("intensely " if m >= 0.6 else "vivid ") + prim
            recipe = None
            try:
                from orchid_clip_hybrid_g0.transgression_direction import predict_recipe

                recipe = predict_recipe(ranked[0][0], ranked[1][0])
            except Exception:
                recipe = None
            if recipe and m >= 0.5:
                mod = recipe.get("modality")
                if mod == "hue" and "unexpected" not in prim:
                    tone = recipe.get("tone")
                    prim += (
                        f", with an unexpected {tone + ' ' if tone else ''}"
                        "contrasting flush"
                    )
                elif mod == "texture" and "contrasting markings" not in prim:
                    prim += ", with bold contrasting markings"
            result["color"]["primary"] = prim
            result["color"]["transgression_emphasis"] = round(m, 3)
            if recipe:
                result["color"]["transgression_modality"] = recipe.get("modality")
        return result

    @staticmethod
    def _get_nested(d: dict, dotpath: str) -> Any:
        """Retrieve a value from a nested dict using a dot-separated path."""
        parts = dotpath.split(".")
        cur: Any = d
        for p in parts:
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                return None
        return cur

    @staticmethod
    def _set_nested(d: dict, dotpath: str, value: Any) -> None:
        parts = dotpath.split(".")
        cur = d
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value

    # ------------------------------------------------------------------
    # Natural-language composition
    # ------------------------------------------------------------------

    @staticmethod
    def _simplify_shape(shape: str) -> str:
        """Collapse verbose merged petal shapes into concise form."""
        if not shape:
            return ""
        # "broad, rounded to elongated rounded" -> keep dominant side
        if " to " in shape:
            shape = shape.split(" to ")[0].strip()
        # "broad rounded with some elongated variants" -> simplify
        for noise in ("with some", "with occasional"):
            if noise in shape:
                shape = shape.split(noise)[0].strip()
        # At most 2 comma-separated descriptors
        tokens = [t.strip() for t in shape.split(",")]
        if len(tokens) > 2:
            tokens = tokens[:2]
        return ", ".join(tokens)

    @staticmethod
    def _simplify_blend(text: str) -> str:
        """Shorten 'X blending with Y' — keep dominant parent's value only."""
        if not text:
            return ""
        if "blending with" in text:
            return text.split("blending with", 1)[0].strip()
        return text

    @staticmethod
    def _compose_description(traits: dict[str, Any], max_tokens: int = 75) -> str:
        """Assemble merged traits into a concise, CLIP-friendly string.

        Front-loads the most visually important traits within the ~77 token
        CLIP window.  Token order: medium anchor → size → lip (most distinctive
        Cattleya feature) → primary color + petal shape → distribution → secondary.
        Drops substance/texture which CLIP rarely sees due to truncation.
        """
        simplify_shape = PhenotypeEngine._simplify_shape
        simplify_blend = PhenotypeEngine._simplify_blend

        size = traits.get("size", "medium")
        color = traits.get("color", {})
        primary = simplify_blend(color.get("primary", ""))
        form = traits.get("form", {})
        petal_shape = simplify_shape(form.get("petal_shape", ""))

        # Emphasize warm colors to counteract LoRA purple bias
        warm_kw = (
            "orange",
            "red",
            "scarlet",
            "coral",
            "salmon",
            "burgundy",
            "crimson",
            "vermillion",
            "golden",
            "yellow",
            "amber",
        )
        cool_kw = ("lavender", "purple", "violet", "lilac", "mauve")
        primary_lower = primary.lower() if primary else ""
        is_warm = any(kw in primary_lower for kw in warm_kw)
        is_cool = any(kw in primary_lower for kw in cool_kw)
        display_primary = primary
        if is_warm and not is_cool and primary:
            display_primary = f"({primary}:1.3)"

        # --- Build lip fragment early (most distinctive Cattleya trait) ---
        lip_color = simplify_blend(color.get("lip", ""))
        lip_shape = simplify_blend(form.get("lip_shape", ""))
        lip_fragment = ""
        if lip_color:
            lip_words = lip_color.split()
            if len(lip_words) > 4:
                lip_color = " ".join(lip_words[:4])
                for suffix in (" with", " and", " to", " in"):
                    if lip_color.endswith(suffix):
                        lip_color = lip_color[: -len(suffix)]
            shape_prefix = ""
            if lip_shape:
                ls = lip_shape.lower()
                if "small" in ls or "tubular" in ls or "narrow" in ls:
                    shape_prefix = "small "
                elif "large" in ls or "frilled" in ls or "ruffled" in ls:
                    shape_prefix = "large ruffled "
            lip_fragment = f"{shape_prefix}{lip_color} lip"

        # --- Assemble parts: anchor → size → lip → petals → pattern → secondary ---
        parts: list[str] = []

        # Medium anchor + size
        parts.append(f"photograph of a {size} cattleya orchid plant in flower")

        # Lip first (most visually distinctive, must land within first ~40 tokens)
        if lip_fragment:
            parts.append(lip_fragment)

        # Primary color + petal shape
        if display_primary and petal_shape:
            parts.append(f"{display_primary} {petal_shape} petals")
        elif display_primary:
            parts.append(f"{display_primary} petals")

        # Distribution — only emit visually distinctive patterns
        distribution = color.get("distribution", "even")
        _skip = {"even", "veined", ""}
        if distribution and distribution not in _skip and "to" not in distribution:
            parts.append(f"{distribution} pattern")

        # Secondary colour / markings
        secondary = simplify_blend(color.get("secondary", "") or "")
        if secondary:
            parts.append(f"{secondary} markings")

        description = ", ".join(parts)

        # Token budget guard: rough estimate (words × 1.3 for BPE expansion)
        est_tokens = int(len(description.split()) * 1.3)
        if est_tokens > max_tokens:
            # Trim from the end (least important tokens)
            words = description.split()
            while int(len(words) * 1.3) > max_tokens and len(words) > 5:
                words.pop()
            description = " ".join(words)

        return description
