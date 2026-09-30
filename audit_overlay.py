"""
Audit Overlay Module for Video Shot Classification Inspection.

Burns an on-screen text overlay into video frames showing:
- Segment number (e.g., "Segment 1")
- Ground Truth category (e.g., "Ground Truth: Anchor Studio" or "Ground Truth: N/A")
- Detected category (e.g., "Detected: Field Report")
- Visual Match / Mismatch status indicator (Green for Match, Red for Mismatch, Yellow for N/A)

Uses semi-transparent background box for maximum readability against bright footage.
"""

import cv2
import numpy as np
from typing import Optional


def draw_audit_overlay(
    frame: np.ndarray,
    shot_id: int,
    detected_label: str,
    ground_truth_label: Optional[str] = None
) -> np.ndarray:
    """
    Draw on-screen audit overlay on a single video frame.

    Args:
        frame: OpenCV image frame (BGR numpy array)
        shot_id: Segment/shot index number
        detected_label: Category string predicted by shot_classifier.py
        ground_truth_label: Ground truth category string from evaluation/JSON (or N/A)

    Returns:
        Modified frame with audit overlay burned in.
    """
    if frame is None or frame.size == 0:
        return frame

    # Standardize ground truth label text
    gt_text = ground_truth_label.strip() if (ground_truth_label and ground_truth_label.strip()) else "N/A"

    # Determine status & color coding
    if gt_text == "N/A":
        status_str = "UNVERIFIED"
        status_color = (0, 215, 255)     # Yellow / Amber
        header_color = (255, 255, 255)   # White
    elif gt_text == detected_label:
        status_str = "MATCH"
        status_color = (40, 220, 40)     # Vibrant Green
        header_color = (40, 255, 40)
    else:
        status_str = "MISMATCH"
        status_color = (50, 50, 255)     # Red
        header_color = (80, 80, 255)

    h, w = frame.shape[:2]

    # Calculate responsive font scale based on frame width (standardized for 1080p baseline)
    scale = max(0.5, (w / 1920.0) * 0.75)
    thickness = max(1, int(scale * 2))
    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = [
        f"Segment {shot_id}  [{status_str}]",
        f"Ground Truth: {gt_text}",
        f"Detected:     {detected_label}"
    ]

    # Compute bounding box dimensions for background overlay
    max_line_w = 0
    total_h = 0
    line_heights = []

    for line in lines:
        (lw, lh), baseline = cv2.getTextSize(line, font, scale, thickness)
        max_line_w = max(max_line_w, lw)
        line_height = lh + baseline + int(10 * scale)
        line_heights.append(line_height)
        total_h += line_height

    padding = int(16 * scale)
    box_w = max_line_w + (padding * 2) + int(10 * scale)
    box_h = total_h + (padding * 2)

    top_left_x = int(20 * scale)
    top_left_y = int(20 * scale)
    bottom_right_x = top_left_x + box_w
    bottom_right_y = top_left_y + box_h

    # Ensure box fits within image bounds
    bottom_right_x = min(w - 5, bottom_right_x)
    bottom_right_y = min(h - 5, bottom_right_y)

    # Copy frame to draw semi-transparent background box
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (top_left_x, top_left_y),
        (bottom_right_x, bottom_right_y),
        (15, 15, 20),  # Dark navy/grey background
        -1
    )

    # Apply alpha blending for semi-transparency (alpha = 0.65)
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    # Draw left border accent line in status color
    border_thick = max(3, int(6 * scale))
    cv2.rectangle(
        frame,
        (top_left_x, top_left_y),
        (top_left_x + border_thick, bottom_right_y),
        status_color,
        -1
    )

    # Draw text lines
    current_y = top_left_y + padding + line_heights[0] - int(6 * scale)
    text_x = top_left_x + border_thick + int(10 * scale)

    # Line 1: Segment & Status
    cv2.putText(
        frame,
        lines[0],
        (text_x, current_y),
        font,
        scale,
        header_color,
        thickness + 1,
        cv2.LINE_AA
    )

    # Line 2: Ground Truth
    current_y += line_heights[1]
    cv2.putText(
        frame,
        lines[1],
        (text_x, current_y),
        font,
        scale,
        (220, 220, 220),  # Light grey
        thickness,
        cv2.LINE_AA
    )

    # Line 3: Detected Category
    current_y += line_heights[2]
    cv2.putText(
        frame,
        lines[2],
        (text_x, current_y),
        font,
        scale,
        (255, 255, 255),  # Pure white
        thickness,
        cv2.LINE_AA
    )

    return frame
