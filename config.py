"""
Configuration Module for Temporal Scene-Boundary Detection and Visual Taxonomy System.

Contains ALL tunable thresholds, frame sampling parameters, shot detection parameters,
classification rule weights, hue supporting ranges, and redundancy parameters.
Centralized to prevent hard-coded magic numbers across modules.
"""

from typing import Optional

# =====================================================================
# 1. Frame Sampling Parameters
# =====================================================================
FRAME_SKIP: int = 2  # Process 1 frame every N frames (configurable via CLI --frame-skip)
MAX_FRAMES: Optional[int] = None        # Max frames to process (configurable via CLI --max-frames)
MAX_DURATION: Optional[float] = None    # Max duration in seconds to process (configurable via CLI --max-duration)
DEFAULT_URL_MAX_DURATION: float = 600.0 # Default max duration cap for online streaming URLs (10 mins) to prevent memory crash


# =====================================================================
# Audit Overlay Parameters
# =====================================================================
AUDIT_MODE: bool = False            # Enable on-screen audit overlay video generation
AUDIT_OUTPUT_DIR: str = "output"    # Output folder for audit videos

# =====================================================================
# 2. Shot Boundary Detection Parameters
# =====================================================================
SHOT_BOUNDARY_THRESHOLD: float = 0.22  # Combined frame & histogram difference threshold to trigger shot cut candidate
HISTOGRAM_THRESHOLD: float = 0.18      # HSV histogram dissimilarity threshold for the simplified detector
ADAPTIVE_PEAK_WINDOW: int = 5          # Sliding window size for local difference peak validation
ADAPTIVE_PEAK_RATIO: float = 1.25      # Local score ratio requirement over local window average
MINIMUM_SHOT_DURATION: float = 0.5     # Minimum duration in seconds to create a shot (prevents noise splits)
SHORT_SHOT_WARNING_THRESHOLD: float = 1.0 # Duration in seconds below which a shot triggers a warning log

# =====================================================================
# 3. Redundancy Analysis Parameters
# =====================================================================
REDUNDANCY_THRESHOLD: float = 0.85          # Keyframe structural similarity threshold for visual near-duplicates
REDUNDANCY_HISTOGRAM_WEIGHT: float = 0.6    # Weight of 3D HSV histogram correlation in redundancy score
REDUNDANCY_STRUCTURAL_GRID_WEIGHT: float = 0.25 # Weight of 3x3 spatial grid edge density similarity
REDUNDANCY_PROXIMITY_WEIGHT: float = 0.15   # Weight of brightness & symmetry proximity

# =====================================================================
# 4. Classification Decision & Margin Parameters
# =====================================================================
CLASSIFICATION_THRESHOLD: float = 3.0       # Minimum points required to assign a taxonomy label
CONFIDENCE_MARGIN_THRESHOLD: float = 1.0     # Required gap between top category score and second place before labeling 'Other'

# =====================================================================
# 5. Taxonomy Category Decision Thresholds & Scoring Weights
# =====================================================================

# --- Anchor Studio ---
ANCHOR_MIN_STABILITY: float = 0.75
ANCHOR_MIN_SYMMETRY: float = 0.80
ANCHOR_MIN_FACE_AREA: float = 0.01  # Studio anchors are usually prominent (> 1% of screen)
ANCHOR_WEIGHT_STABILITY: float = 2.5
ANCHOR_WEIGHT_COMPOSITION: float = 3.0
ANCHOR_WEIGHT_SUBJECT: float = 2.5
ANCHOR_WEIGHT_LOWER_THIRD: float = 1.5

# --- Field Report ---
FIELD_MIN_MOTION: float = 0.005
FIELD_MAX_MOTION: float = 0.06
FIELD_MAX_SYMMETRY: float = 0.78
FIELD_WEIGHT_REPORTER: float = 3.0
FIELD_WEIGHT_LOCATION: float = 2.5
FIELD_WEIGHT_MOTION: float = 1.5

# --- B-Roll ---
BROLL_MIN_MOTION: float = 0.02
BROLL_MAX_STABILITY: float = 0.75
BROLL_MIN_EDGE_DENSITY: float = 0.015
BROLL_WEIGHT_MOTION: float = 3.0
BROLL_WEIGHT_TEXTURE: float = 2.5
BROLL_WEIGHT_NON_STUDIO: float = 2.0

# --- Title Card ---
TITLE_MIN_HEADLINE_TEXT: float = 0.02
TITLE_MIN_STABILITY: float = 0.80
TITLE_WEIGHT_TEXT_DOMINANCE: float = 4.0
TITLE_WEIGHT_STATIC_LAYOUT: float = 2.5
TITLE_WEIGHT_HIGH_CONTRAST: float = 2.0
