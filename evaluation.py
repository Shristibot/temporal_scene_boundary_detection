"""
Upgraded Evaluation Module.

Evaluates predicted taxonomy classifications against manually annotated ground truth labels
from JSON files.

Calculates overall accuracy, per-category accuracy, and formats a confusion matrix.
"""

import json
import os
from typing import List, Dict, Any, Tuple





def load_ground_truth_from_json(json_path: str) -> List[Dict[str, Any]]:
    """Load ground truth annotations from a JSON file."""
    if not os.path.exists(json_path):
        print(f"Error: JSON ground truth file does not exist: {json_path}")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "shots" in data:
        return data["shots"]
    return []


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluate predicted shot classifications against ground truth annotations.

    Args:
        predictions: List of dicts containing shot_id, classification / predicted_label
        ground_truth: List of dicts containing shot_id, true_label / ground_truth_label

    Returns:
        Dict containing overall_accuracy, per_category_accuracy, confusion_matrix, and total_evaluated.
    """
    # Map ground truth by (video_name, shot_id) or shot_id
    gt_map: Dict[Tuple[str, int], str] = {}
    gt_by_id: Dict[int, str] = {}

    for gt in ground_truth:
        shot_id = gt.get("shot_id")
        label = gt.get("true_label") or gt.get("ground_truth_label") or gt.get("classification")
        vname = gt.get("video_name", "")

        if shot_id is not None and label is not None:
            if vname:
                gt_map[(vname, int(shot_id))] = label
            gt_by_id[int(shot_id)] = label

    categories = ["Anchor Studio", "Field Report", "B-Roll", "Title Card", "Other"]

    correct_predictions: int = 0
    total_evaluated: int = 0

    category_correct: Dict[str, int] = {cat: 0 for cat in categories}
    category_total: Dict[str, int] = {cat: 0 for cat in categories}

    confusion_matrix: Dict[str, Dict[str, int]] = {
        true_cat: {pred_cat: 0 for pred_cat in categories}
        for true_cat in categories
    }

    for pred in predictions:
        shot_id = pred.get("shot_id")
        vname = pred.get("video_name", "")
        predicted_label = pred.get("classification") or pred.get("predicted_label")

        if shot_id is None or not predicted_label:
            continue

        true_label = None
        if (vname, int(shot_id)) in gt_map:
            true_label = gt_map[(vname, int(shot_id))]
        elif int(shot_id) in gt_by_id:
            true_label = gt_by_id[int(shot_id)]

        if true_label and true_label in categories and predicted_label in categories:
            total_evaluated += 1
            category_total[true_label] += 1
            confusion_matrix[true_label][predicted_label] += 1

            if predicted_label == true_label:
                correct_predictions += 1
                category_correct[true_label] += 1

    overall_accuracy = (correct_predictions / total_evaluated * 100.0) if total_evaluated > 0 else 0.0

    per_category_accuracy: Dict[str, float] = {}
    for cat in categories:
        tot = category_total[cat]
        acc = (category_correct[cat] / tot * 100.0) if tot > 0 else 0.0
        per_category_accuracy[cat] = round(acc, 2)

    return {
        "overall_accuracy": round(overall_accuracy, 2),
        "total_evaluated": total_evaluated,
        "correct_predictions": correct_predictions,
        "per_category_accuracy": per_category_accuracy,
        "category_totals": category_total,
        "confusion_matrix": confusion_matrix
    }


def format_confusion_matrix_text(confusion_matrix: Dict[str, Dict[str, int]]) -> str:
    """Format confusion matrix dict as readable text table."""
    categories = ["Anchor Studio", "Field Report", "B-Roll", "Title Card", "Other"]
    abbrev = ["Anchor", "Field", "B-Roll", "Title", "Other"]

    lines = []
    header_title = "True \\ Pred"
    header = f"{header_title:<15} | " + " | ".join(f"{name:>7}" for name in abbrev)
    lines.append(header)
    lines.append("-" * len(header))

    for cat_idx, true_cat in enumerate(categories):
        row_vals = [f"{confusion_matrix[true_cat][pred_cat]:>7}" for pred_cat in categories]
        row_str = f"{true_cat:<15} | " + " | ".join(row_vals)
        lines.append(row_str)

    return "\n".join(lines)
