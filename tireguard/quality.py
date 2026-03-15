import cv2
import numpy as np

def brightness_mean(gray):
    return float(np.mean(gray))

def glare_ratio(gray, bright_threshold=240):
    bright = np.sum(gray >= bright_threshold)
    return float(bright) / float(gray.size)

def sharpness_laplacian(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def run_quality_checks(gray, cfg):
    """
    Returns dict:
      ok: bool
      reasons: list[str]
      metrics: dict
    """
    b = brightness_mean(gray)
    g = glare_ratio(gray)
    s = sharpness_laplacian(gray)

    reasons = []
    if b < cfg.min_brightness:
        reasons.append(f"Too dark (brightness={b:.1f})")
    if b > cfg.max_brightness:
        reasons.append(f"Too bright (brightness={b:.1f})")
    if g > cfg.max_glare_ratio:
        reasons.append(f"Too much glare (glare_ratio={g:.3f})")
    if s < cfg.min_sharpness:
        reasons.append(f"Too blurry (sharpness={s:.1f})")

    return {
        "ok": len(reasons) == 0,
        "reasons": reasons,
        "metrics": {"brightness": b, "glare_ratio": g, "sharpness": s},
    }


def assess_capture_quality(gray, cfg, score=None, groove_channel_frac=None, tread_detect_meta=None):
    """Assess capture quality with a softer blur policy for strong tread scans."""
    quality = run_quality_checks(gray, cfg)
    reasons = list(quality.get("reasons") or [])
    metrics = dict(quality.get("metrics") or {})

    verdict = "OK"
    relaxed = False
    policy_note = None

    if reasons:
        verdict = "WARNING"
        blur_only = all(r.startswith("Too blurry") for r in reasons)
        sharpness = float(metrics.get("sharpness", 0.0) or 0.0)
        confidence = float((tread_detect_meta or {}).get("confidence", 0.0) or 0.0)
        relaxed_min_sharpness = float(getattr(cfg, "quality_relaxed_min_sharpness", 24.0))
        relaxed_score = float(getattr(cfg, "quality_relaxed_strong_score", 0.11))
        relaxed_channel_frac = float(getattr(cfg, "quality_relaxed_channel_frac", 0.02))
        relaxed_conf = float(getattr(cfg, "quality_relaxed_tread_confidence", 0.55))

        strong_score = score is not None and float(score) >= relaxed_score
        strong_channel = groove_channel_frac is not None and float(groove_channel_frac) >= relaxed_channel_frac
        strong_pattern = confidence >= relaxed_conf
        acceptable_blur = sharpness >= relaxed_min_sharpness

        if blur_only and acceptable_blur and (strong_score or strong_channel) and strong_pattern:
            verdict = "ADVISORY"
            relaxed = True
            policy_note = "Blur advisory relaxed by strong tread signal"

    return {
        **quality,
        "verdict": verdict,
        "relaxed": relaxed,
        "policy_note": policy_note,
    }
