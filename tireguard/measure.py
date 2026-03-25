import numpy as np
import cv2


_SEVERITY = {"GOOD": 0, "WARNING": 1, "REPLACE": 2}
_INV_SEVERITY = {0: "GOOD", 1: "WARNING", 2: "REPLACE"}

DEFAULT_REPLACE_CHANNEL_FRAC = 0.006
DEFAULT_WARNING_CHANNEL_FRAC = 0.018
DEFAULT_REPLACE_SCORE = 0.035
DEFAULT_WARNING_SCORE = 0.065


def _safe_uint8(gray):
    arr = np.asarray(gray)
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def _build_dark_channel_mask(raw_gray):
    """Return a robust binary mask for potential groove shadow channels."""
    g = _safe_uint8(raw_gray)
    mean_b = float(np.mean(g))

    # Global darkness cue (legacy behavior baseline).
    global_thresh = max(1, int(mean_b * 0.72))
    global_mask = (g < global_thresh).astype(np.uint8)

    # Local darkness cue for uneven lighting.
    adapt = cv2.adaptiveThreshold(
        g,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )
    local_mask = (adapt > 0).astype(np.uint8)

    # Black-hat emphasizes dark valleys/channels on bright rubber background.
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, bh_kernel)
    bh_thresh = float(np.percentile(blackhat, 75.0))
    bh_mask = (blackhat >= bh_thresh).astype(np.uint8)

    # Fuse cues conservatively: always keep global, add local/blackhat support.
    fused = np.maximum(global_mask, np.minimum(local_mask, bh_mask))
    return fused, blackhat


def _groove_channel_fraction_multiscale(dark_mask, base_kpx):
    """Estimate groove channel area by erosion at multiple groove-width scales."""
    if dark_mask is None or dark_mask.size == 0:
        return 0.0, 0.0

    scales = (0.75, 1.0, 1.4)
    weights = (0.25, 0.5, 0.25)
    fractions = []
    unions = []

    for s in scales:
        kpx = max(5, int(round(base_kpx * s)))
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kpx, 1))
        k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kpx))
        eroded = np.maximum(cv2.erode(dark_mask, k_h), cv2.erode(dark_mask, k_v))
        unions.append(eroded)
        fractions.append(float(eroded.mean()))

    weighted_frac = float(sum(w * f for w, f in zip(weights, fractions)))
    # A stricter single-mask estimate useful for diagnostics/tuning.
    union_mask = np.maximum.reduce(unions)
    union_frac = float(union_mask.mean())
    return weighted_frac, union_frac

def combine_tread_and_quality_verdicts(tread_verdict, quality_verdict):
    """Combine tread-condition and capture-quality outcomes without hiding their causes."""
    v = (tread_verdict or "").upper()
    q = (quality_verdict or "OK").upper()
    if v not in _SEVERITY:
        return tread_verdict
    sev = _SEVERITY[v]
    if q == "WARNING":
        sev = max(sev, _SEVERITY["WARNING"])
    return _INV_SEVERITY[sev]


def groove_visibility_score(edges_binary, raw_gray=None, mm_per_px=None, mode="enhanced"):
    """
    Score groove visibility (0 = no grooves/worn, higher = deep grooves/good tread).

    When raw_gray (pre-CLAHE grayscale) is provided, uses morphological dark-channel
    analysis: real tire grooves are wide dark channels that survive morphological erosion,
    while surface cracks and wear texture are thin features that erode away.
    The final score is the geometric mean of the channel score and the edge score --
    both must be present for a high result, preventing worn/cracked surfaces from
    scoring falsely GOOD via edge density alone.
    """
    edges = (edges_binary > 0).astype(np.uint8)
    edge_density = float(edges.mean())

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
    areas = np.sort(areas)[::-1]
    top = areas[:10] if areas.size > 0 else np.array([0])
    continuity = float(np.mean(top))

    continuity_norm = continuity / (continuity + 2000.0)
    edge_score = 0.7 * edge_density + 0.3 * continuity_norm

    groove_channel_frac = None

    if raw_gray is not None and raw_gray.size > 0:
        # --- Morphological groove-channel detection ---
        # Grooves appear as genuinely dark, wide channels in the raw (pre-CLAHE) image.
        # Erosion with a groove-width kernel removes thin features (cracks, texture)
        # while preserving wide dark channels (actual grooves).
        MIN_GROOVE_MM = 1.5  # minimum real groove width to count
        raw_gray = _safe_uint8(raw_gray)
        if mm_per_px and mm_per_px > 0:
            kpx = int(MIN_GROOVE_MM / mm_per_px)
            # Cap at 20% of image width to avoid over-aggressive erosion
            kpx = max(5, min(raw_gray.shape[1] // 5, kpx))
        else:
            # Fallback: ~10% of image width
            kpx = max(5, raw_gray.shape[1] // 10)

        if str(mode).strip().lower() == "legacy":
            mean_b = float(np.mean(raw_gray))
            # Dark mask: pixels darker than 72% of the mean represent groove shadow floors
            dark_thresh = max(1, int(mean_b * 0.72))
            dark_mask = (raw_gray < dark_thresh).astype(np.uint8)

            # Erode along H and V axes; take union (grooves run in either direction)
            k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kpx, 1))
            k_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kpx))
            eroded = np.maximum(cv2.erode(dark_mask, k_h), cv2.erode(dark_mask, k_v))
            groove_channel_frac = float(eroded.mean())
            groove_channel_union_frac = float(eroded.mean())
            channel_contrast = None
        else:
            # Robust dark channel extraction under mixed illumination.
            dark_mask, blackhat = _build_dark_channel_mask(raw_gray)
            groove_channel_frac, groove_channel_union_frac = _groove_channel_fraction_multiscale(dark_mask, kpx)

            # Channel contrast reliability (0..1): weak valleys lower effective channel evidence.
            channel_mask = (dark_mask > 0)
            if np.any(channel_mask):
                bh_mean = float(np.mean(blackhat[channel_mask]))
                channel_contrast = float(max(0.0, min(1.0, bh_mean / 42.0)))
            else:
                channel_contrast = 0.0

            # Penalize channel fraction when channel contrast is weak.
            groove_channel_frac = float(groove_channel_frac * (0.55 + 0.45 * channel_contrast))

        # Geometric mean: both groove channels AND visible edges must be present.
        # A cracked-but-flat tire has edge_score > 0 but groove_channel_frac ≈ 0 → low score.
        # A well-grooved tire has both high → high score.
        score = float(np.sqrt(groove_channel_frac) * np.sqrt(edge_score))
    else:
        groove_channel_union_frac = None
        channel_contrast = None
        score = edge_score

    return {
        "edge_density": edge_density,
        "continuity": continuity,
        "edge_score": float(edge_score),
        "groove_channel_frac": groove_channel_frac,
        "groove_channel_union_frac": groove_channel_union_frac,
        "channel_contrast": channel_contrast,
        "score": float(score),
    }


def estimate_groove_channel_frac(score, edge_density=None, continuity=None):
    """Estimate groove-channel fraction from persisted scan metrics.

    Useful for API/database paths where raw ROI image is not available.
    """
    if edge_density is None or continuity is None:
        return None
    try:
        edge_density = float(edge_density)
        continuity = float(continuity)
        score = float(score)
    except (TypeError, ValueError):
        return None

    continuity_norm = continuity / (continuity + 2000.0)
    edge_score = 0.7 * edge_density + 0.3 * continuity_norm
    if edge_score <= 1e-8:
        return None
    groove = (score * score) / edge_score
    return float(max(0.0, min(1.0, groove)))


def apply_defect_guard(
    verdict,
    groove_channel_frac=None,
    quality_ok=True,
    score=None,
    replace_channel_frac=DEFAULT_REPLACE_CHANNEL_FRAC,
    warning_channel_frac=DEFAULT_WARNING_CHANNEL_FRAC,
    replace_score=DEFAULT_REPLACE_SCORE,
    warning_score=DEFAULT_WARNING_SCORE,
):
    """Downgrade optimistic verdicts when groove-channel evidence is weak."""
    v = (verdict or "").upper()
    if v not in _SEVERITY:
        return verdict

    sev = _SEVERITY[v]

    # Extremely weak dark-channel signal typically means worn/flat tread.
    if groove_channel_frac is not None and groove_channel_frac < replace_channel_frac:
        sev = max(sev, _SEVERITY["REPLACE"])
    # Borderline channel evidence should never remain GOOD.
    elif groove_channel_frac is not None and groove_channel_frac < warning_channel_frac:
        sev = max(sev, _SEVERITY["WARNING"])
    elif score is not None:
        # Fallback for paths without raw image channel estimate.
        try:
            s = float(score)
            if s < replace_score:
                sev = max(sev, _SEVERITY["REPLACE"])
            elif s < warning_score:
                sev = max(sev, _SEVERITY["WARNING"])
        except (TypeError, ValueError):
            pass

    # Poor capture quality can inflate edge-derived scores; avoid GOOD in this case.
    if not quality_ok:
        sev = max(sev, _SEVERITY["WARNING"])

    return _INV_SEVERITY[sev]


def pass_fail_from_score(
    score,
    good=0.10,
    warn=0.055,
    groove_channel_frac=None,
    quality_ok=True,
    replace_channel_frac=DEFAULT_REPLACE_CHANNEL_FRAC,
    warning_channel_frac=DEFAULT_WARNING_CHANNEL_FRAC,
    replace_score=DEFAULT_REPLACE_SCORE,
    warning_score=DEFAULT_WARNING_SCORE,
):
    """
    Classify a groove score into a tread verdict.
    Thresholds tuned for the geometric-mean formula used by groove_visibility_score.
    """
    if score >= good:
        verdict = "GOOD"
    elif score >= warn:
        verdict = "WARNING"
    else:
        verdict = "REPLACE"
    return apply_defect_guard(
        verdict,
        groove_channel_frac=groove_channel_frac,
        quality_ok=quality_ok,
        score=score,
        replace_channel_frac=replace_channel_frac,
        warning_channel_frac=warning_channel_frac,
        replace_score=replace_score,
        warning_score=warning_score,
    )
