from typing import Dict, Any
import config


def classify_shot(shot_features: Dict[str, Any]) -> Dict[str, Any]:
    s = shot_features

    stability = s.get("visual_stability_score", 0.0)
    motion = s.get("intra_shot_motion_mean", 0.0)
    edge = s.get("avg_edge_density", 0.0)
    symmetry = s.get("avg_spatial_symmetry", 0.0)

    headline = s.get("avg_headline_text", 0.0)
    lower_third = s.get("avg_lower_third_text", 0.0)
    textcov = s.get("avg_text_coverage", 0.0)

    faces = s.get("detected_faces_count", 0)
    centered = s.get("presenter_centered", False)
    face_area = s.get("avg_face_area", 0.0)

    scores = {
        "Anchor Studio": 0.0,
        "Field Report": 0.0,
        "B-Roll": 0.0,
        "Title Card": 0.0,
        "Other": 0.0,
    }

    # Anchor Studio evidence
    if faces > 0:
        scores["Anchor Studio"] += config.ANCHOR_WEIGHT_SUBJECT
    if centered:
        scores["Anchor Studio"] += config.ANCHOR_WEIGHT_COMPOSITION
    if stability >= config.ANCHOR_MIN_STABILITY:
        scores["Anchor Studio"] += config.ANCHOR_WEIGHT_STABILITY
    if symmetry >= config.ANCHOR_MIN_SYMMETRY:
        scores["Anchor Studio"] += config.ANCHOR_WEIGHT_COMPOSITION
    if face_area >= config.ANCHOR_MIN_FACE_AREA:
        scores["Anchor Studio"] += config.ANCHOR_WEIGHT_SUBJECT

    # Field Report evidence
    if faces > 0:
        scores["Field Report"] += config.FIELD_WEIGHT_REPORTER
    if not centered or face_area < config.ANCHOR_MIN_FACE_AREA * 0.5 or symmetry < config.ANCHOR_MIN_SYMMETRY * 0.9:
        scores["Field Report"] += config.FIELD_WEIGHT_LOCATION
    if config.FIELD_MIN_MOTION <= motion <= config.FIELD_MAX_MOTION:
        scores["Field Report"] += config.FIELD_WEIGHT_MOTION

    # B-Roll evidence
    if faces == 0 and (motion >= config.BROLL_MIN_MOTION or edge >= config.BROLL_MIN_EDGE_DENSITY):
        scores["B-Roll"] += config.BROLL_WEIGHT_MOTION + config.BROLL_WEIGHT_TEXTURE
    elif faces > 0 and face_area < config.ANCHOR_MIN_FACE_AREA * 0.5 and (motion >= config.BROLL_MIN_MOTION or edge >= config.BROLL_MIN_EDGE_DENSITY or symmetry < config.ANCHOR_MIN_SYMMETRY * 0.9):
        scores["B-Roll"] += config.BROLL_WEIGHT_MOTION + config.BROLL_WEIGHT_TEXTURE
    if stability < config.BROLL_MAX_STABILITY:
        scores["B-Roll"] += config.BROLL_WEIGHT_NON_STUDIO

    # Title Card evidence
    if headline >= config.TITLE_MIN_HEADLINE_TEXT:
        scores["Title Card"] += config.TITLE_WEIGHT_TEXT_DOMINANCE
    if stability >= config.TITLE_MIN_STABILITY:
        scores["Title Card"] += config.TITLE_WEIGHT_STATIC_LAYOUT
    if textcov >= 0.25 or lower_third >= 0.01:
        scores["Title Card"] += config.TITLE_WEIGHT_HIGH_CONTRAST
    if faces == 0 or face_area < config.ANCHOR_MIN_FACE_AREA * 0.5:
        scores["Title Card"] += 1.0

    # Fallback if no evidence is strong
    if max(scores.values()) <= 1.0:
        scores["Other"] = 1.0

    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    if best_label == "Anchor Studio" and best_score < 4.0:
        best_label = "Other"

    confidence = round(min(95.0, max(40.0, 40.0 + best_score * 4.0)), 1)

    if best_label == "Anchor Studio":
        explanation = f"stable={stability:.2f}, symmetry={symmetry:.2f}, centered={centered}, faces={faces}, face_area={face_area:.4f}"
    elif best_label == "Field Report":
        explanation = f"motion={motion:.4f}, faces={faces}, centered={centered}, symmetry={symmetry:.2f}"
    elif best_label == "B-Roll":
        explanation = f"motion={motion:.4f}, edge={edge:.4f}, face_area={face_area:.4f}, symmetry={symmetry:.2f}"
    elif best_label == "Title Card":
        explanation = f"headline={headline:.4f}, textcov={textcov:.4f}, stability={stability:.2f}"
    else:
        explanation = "no strong category evidence"

    return {
        "shot_label": best_label,
        "confidence_score": confidence,
        "scores": scores,
        "rule_explanation": f"{best_label}: " + explanation
    }
