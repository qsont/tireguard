import numpy as np
import cv2

def groove_visibility_score(edges_binary):
    edges = (edges_binary > 0).astype(np.uint8)
    edge_density = float(edges.mean())

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
    areas = np.sort(areas)[::-1]
    top = areas[:10] if areas.size > 0 else np.array([0])
    continuity = float(np.mean(top))

    continuity_norm = continuity / (continuity + 2000.0)
    score = 0.7 * edge_density + 0.3 * continuity_norm

    return {"edge_density": edge_density, "continuity": continuity, "score": score}

def pass_fail_from_score(score, good=0.10, warn=0.07):
    if score >= good:
        return "GOOD"
    if score >= warn:
        return "WARNING"
    return "REPLACE"
