from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _ensure_edges(gray: np.ndarray, edges: np.ndarray | None) -> np.ndarray:
    if edges is not None and edges.size > 0:
        return (edges > 0).astype(np.uint8)
    e = cv2.Canny(gray, 60, 140)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    return (e > 0).astype(np.uint8)


def _symmetry_score(edges01: np.ndarray) -> float:
    h, w = edges01.shape
    if h == 0 or w < 8:
        return 0.0
    half = w // 2
    left = edges01[:, :half]
    right = np.fliplr(edges01[:, w - half :])
    if left.size == 0 or right.size == 0:
        return 0.0
    diff = np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32)))
    return float(max(0.0, 1.0 - diff))


def _diagonal_ratio(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    if mag.size == 0:
        return 0.0
    thresh = float(np.percentile(mag, 75.0))
    strong = mag >= max(1.0, thresh)
    if not np.any(strong):
        return 0.0
    # 0 deg means mostly horizontal gradient, 90 mostly vertical gradient.
    # Mid-range angles indicate diagonal tread channels.
    ang = np.degrees(np.arctan2(np.abs(gy), np.abs(gx) + 1e-6))
    diag = (ang >= 22.5) & (ang <= 67.5) & strong
    return float(np.sum(diag) / max(1, int(np.sum(strong))))


def detect_tread_design(
    *,
    gray: np.ndarray,
    edges: np.ndarray | None = None,
) -> tuple[str, dict[str, Any]]:
    """Heuristic tread design classification from the current scan ROI.

    Returns one of: Symmetrical, Asymmetrical, Directional.
    """
    if gray is None or gray.size == 0:
        return "Asymmetrical", {"reason": "empty-gray"}

    edges01 = _ensure_edges(gray, edges)
    edge_density = float(edges01.mean())
    symmetry = _symmetry_score(edges01)
    diagonal = _diagonal_ratio(gray)

    # Conservative defaults: treat uncertain cases as asymmetrical.
    if symmetry >= 0.62 and diagonal < 0.43:
        label = "Symmetrical"
    elif diagonal >= 0.43 and symmetry < 0.66:
        label = "Directional"
    else:
        label = "Asymmetrical"

    confidence = 0.5
    if label == "Symmetrical":
        confidence = min(0.99, max(0.5, 0.5 + (symmetry - 0.62) * 1.2 + (0.43 - diagonal) * 0.3))
    elif label == "Directional":
        confidence = min(0.99, max(0.5, 0.5 + (diagonal - 0.43) * 1.4 + (0.66 - symmetry) * 0.2))
    else:
        center_bias = 1.0 - min(1.0, abs(symmetry - 0.5) * 2.0)
        confidence = min(0.95, max(0.5, 0.5 + center_bias * 0.35))

    return label, {
        "symmetry_score": float(symmetry),
        "diagonal_ratio": float(diagonal),
        "edge_density": float(edge_density),
        "confidence": float(confidence),
    }
