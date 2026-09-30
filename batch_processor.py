"""
Batch Processor Module.

Iterates over input video files or web URLs, processes each video through the complete pipeline:
Video Reader -> Shot Detection -> Feature Extraction -> Classification -> Export Results -> (Optional) Audit Overlay Generation.

Gracefully handles invalid or corrupted video files without interrupting batch processing.
"""

import os
import csv
import json
from typing import List, Dict, Any, Optional

from video_reader import VideoReader, is_url
from shot_detector import detect_shot_boundaries
from feature_extractor import extract_shot_features
from shot_classifier import classify_shot
from audit_overlay import draw_audit_overlay
from evaluation import load_ground_truth_from_json
import config


def load_ground_truth_mapping(video_name: str, explicit_path: Optional[str] = None) -> Dict[int, str]:
    """
    Load ground truth label mapping (shot_id -> true_label) for a video.
    Checks explicit_path first, then auto-discovers in results/ or ground_truth/ folders.
    """
    gt_path = explicit_path

    if not gt_path:
        base_name = os.path.splitext(video_name)[0]
        candidate_paths = [
            os.path.join("results", f"{base_name}_ground_truth.json"),
            os.path.join("ground_truth", f"{base_name}_ground_truth.json"),
            os.path.join("ground_truth", f"{base_name}.json")
        ]
        for cp in candidate_paths:
            if os.path.exists(cp):
                gt_path = cp
                break

    if not gt_path or not os.path.exists(gt_path):
        return {}

    gt_data = load_ground_truth_from_json(gt_path)
    gt_map: Dict[int, str] = {}

    for item in gt_data:
        shot_id = item.get("shot_id")
        label = item.get("true_label") or item.get("ground_truth_label") or item.get("classification")
        if shot_id is not None and label:
            gt_map[int(shot_id)] = str(label)

    return gt_map


def process_single_video(
    video_path: str,
    output_dir: str = "results",
    frame_skip: int = config.FRAME_SKIP,
    max_frames: Optional[int] = config.MAX_FRAMES,
    max_duration: Optional[float] = config.MAX_DURATION,
    debug_classifier: bool = False,
    audit: bool = False,
    ground_truth_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single video file or online video URL end-to-end through the vision pipeline.

    Args:
        video_path: Path to local video file or web URL
        output_dir: Folder to save structured CSV and JSON outputs
        frame_skip: Sampling skip rate
        max_frames: Max frames limit to sample
        max_duration: Max duration limit in seconds
        debug_classifier: Print detailed classifier evidence per shot
        audit: Burn on-screen audit overlay video output
        ground_truth_path: Path to ground truth JSON file for audit overlay matching

    Returns:
        Dict containing video summary stats.
    """
    video_name = os.path.basename(video_path) if not is_url(video_path) else "online_video.mp4"
    print(f"\n--- Processing Video: {video_name} ---")

    # Safety default for online streams (e.g. 12-hour YouTube live streams) to prevent system OOM
    effective_max_duration = max_duration
    if max_duration is not None and max_duration <= 0:
        effective_max_duration = None

    if is_url(video_path) and (max_frames is None or max_frames <= 0) and (max_duration is None):
        effective_max_duration = config.DEFAULT_URL_MAX_DURATION
        print(f"[Online Stream Safety] Live stream detected without duration cap. Applying default streaming limit of {effective_max_duration:.0f}s (10 mins). Pass --max-duration <seconds> or pass --max-duration 0 for unlimited.")

    reader = VideoReader(video_path)
    if not reader.open():
        print(f"Failed to open video source: {video_path}")
        return {"video_name": video_name, "status": "Failed to open video"}

    metadata = reader.get_metadata()
    print(f"Metadata: {metadata['frame_width']}x{metadata['frame_height']} @ {metadata['frame_rate']} FPS, Duration: {metadata['video_duration']}s")

    # Step 1: Read Sampled Frames (Memory-safe downscaling and cap enabled)
    print("Reading and sampling frames...")
    sampled_frames = list(reader.read_sampled_frames(
        frame_skip=frame_skip,
        downscale_max_dim=480,
        max_frames=max_frames,
        max_duration=effective_max_duration,
        keep_raw_frames=audit,
        log_progress=True
    ))
    reader.close()

    if not sampled_frames:
        print(f"Warning: No frames read from {video_name}")
        return {"video_name": video_name, "status": "No readable frames"}

    # Step 2: Detect Shot Boundaries
    shots = detect_shot_boundaries(
        sampled_frames=sampled_frames,
        frame_rate=metadata["frame_rate"],
        histogram_threshold=config.HISTOGRAM_THRESHOLD,
        minimum_shot_duration=config.MINIMUM_SHOT_DURATION
    )
    print(f"Detected {len(shots)} shots.")

    # Step 3: Feature Extraction & Classification
    shot_results: List[Dict[str, Any]] = []
    category_counts: Dict[str, int] = {
        "Anchor Studio": 0,
        "Field Report": 0,
        "B-Roll": 0,
        "Title Card": 0,
        "Other": 0
    }

    for shot in shots:
        features = extract_shot_features(shot)
        classification_res = classify_shot(features)

        # Release frame memory buffer if audit mode is off
        if not audit:
            shot["sampled_frames"] = None

        label = classification_res["shot_label"]
        confidence = classification_res["confidence_score"]
        explanation = classification_res["rule_explanation"]

        if debug_classifier:
            print(f"\nShot {shot['shot_id']}")
            print("-" * 29)
            print(f"Prediction: {label} (Confidence: {confidence}%)")
            print("Scores:")
            for cat, score in classification_res["scores"].items():
                print(f"  {cat}: {score}")
            print("\nRule Explanation:\n{explanation}\n")

        if label in category_counts:
            category_counts[label] += 1
        else:
            category_counts["Other"] += 1

        shot_results.append({
            "video_name": video_name,
            "shot_id": shot["shot_id"],
            "start_time": shot["start_time"],
            "end_time": shot["end_time"],
            "duration": shot["shot_duration"],
            "classification": label,
            "confidence_score": confidence,
            "rule_explanation": explanation,
            "features": {
                k: v for k, v in features.items()
                if k not in ["keyframe", "mean_histogram_vector", "histogram_vector", "grid_edge_density", "shot_id", "start_time", "end_time", "shot_duration"]
            }
        })

    # Step 4: Redundancy Analysis (removed for simplicity)
    redundancy_stats = {
        "redundancy_percentage": 0.0,
        "redundant_shot_count": 0
    }

    # Step 5: Save Structured JSON & CSV Output
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(video_name)[0]

    json_path = os.path.join(output_dir, f"{base_filename}_results.json")
    json_output = {
        "video_metadata": metadata,
        "summary": {
            "total_shots": len(shots),
            "category_counts": category_counts,
            "redundancy_stats": redundancy_stats
        },
        "shots": shot_results
    }
    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump(json_output, f_json, indent=2)

    csv_path = os.path.join(output_dir, f"{base_filename}_results.csv")
    csv_headers = ["video_name", "shot_id", "start_time", "end_time", "duration", "classification", "confidence_score", "rule_explanation"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_headers, extrasaction="ignore")
        writer.writeheader()
        for shot_row in shot_results:
            writer.writerow(shot_row)

    print(f"Saved results to {json_path} and {csv_path}")

    # Step 6: On-Screen Audit Video Generation (if requested)
    should_audit = audit or config.AUDIT_MODE
    if should_audit:
        import cv2

        gt_map = load_ground_truth_mapping(video_name, explicit_path=ground_truth_path)
        if gt_map:
            print(f"Loaded ground truth annotations for {len(gt_map)} shots.")
        else:
            print("No ground truth annotations found. Audit overlay will display 'Ground Truth: N/A'.")

        audit_output_dir = config.AUDIT_OUTPUT_DIR
        os.makedirs(audit_output_dir, exist_ok=True)
        audit_video_path = os.path.join(audit_output_dir, f"{base_filename}_audit.mp4")

        print(f"\n--- Generating On-Screen Audit Video: {audit_video_path} ---")

        audit_reader = VideoReader(video_path)
        if audit_reader.open():
            w = metadata["frame_width"]
            h = metadata["frame_height"]
            fps = metadata["frame_rate"]

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(audit_video_path, fourcc, fps, (w, h))

            current_shot_idx = 0
            num_shots = len(shots)

            # Stream full-resolution frames, burn in audit overlay per-segment, and write to VideoWriter
            for frame_dict in audit_reader.read_sampled_frames(frame_skip=1, downscale_max_dim=None, log_progress=False):
                timestamp = frame_dict["timestamp"]
                frame_img = frame_dict["frame"]

                # Update active shot index based on current frame timestamp
                while current_shot_idx < num_shots - 1 and timestamp > shots[current_shot_idx]["end_time"]:
                    current_shot_idx += 1

                active_shot = shots[current_shot_idx]
                shot_id = active_shot["shot_id"]
                detected_label = shot_results[current_shot_idx]["classification"]
                gt_label = gt_map.get(shot_id, "N/A")

                # Burn overlay on frame
                rendered_frame = draw_audit_overlay(
                    frame=frame_img,
                    shot_id=shot_id,
                    detected_label=detected_label,
                    ground_truth_label=gt_label
                )
                writer.write(rendered_frame)

            writer.release()
            audit_reader.close()
            print(f"[Audit Complete] Rendered overlay video saved to: {audit_video_path}")
        else:
            print(f"Failed to re-open video for audit rendering: {video_path}")

    return {
        "video_name": video_name,
        "total_duration": metadata["video_duration"],
        "number_of_shots": len(shots),
        "anchor_shots": category_counts["Anchor Studio"],
        "field_report_shots": category_counts["Field Report"],
        "b_roll_shots": category_counts["B-Roll"],
        "title_card_shots": category_counts["Title Card"],
        "other_shots": category_counts["Other"],
        "status": "Success"
    }


def process_batch(
    input_folder: str,
    output_dir: str = "results",
    frame_skip: int = config.FRAME_SKIP,
    debug_classifier: bool = False,
    audit: bool = False,
    ground_truth_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Process all videos in input_folder, continuing safely even if individual videos fail.

    Args:
        input_folder: Path to directory containing video files
        output_dir: Path to directory for saving results
        frame_skip: Configured frame skip parameter
        debug_classifier: Print classifier evidence per shot
        audit: Enable on-screen audit overlay video generation
        ground_truth_path: Explicit path to ground truth JSON

    Returns:
        List of summary dicts for each video processed.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Input directory does not exist: {input_folder}")
        return []

    supported_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
    video_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(supported_extensions)
    ]

    print(f"Found {len(video_files)} videos in {input_folder}")

    batch_summaries: List[Dict[str, Any]] = []

    for idx, filename in enumerate(sorted(video_files), start=1):
        video_path = os.path.join(input_folder, filename)
        print(f"\nProgress [{idx}/{len(video_files)}]: {filename}")

        try:
            summary = process_single_video(
                video_path=video_path,
                output_dir=output_dir,
                frame_skip=frame_skip,
                debug_classifier=debug_classifier,
                audit=audit,
                ground_truth_path=ground_truth_path
            )
            batch_summaries.append(summary)
        except Exception as error:
            print(f"Error processing video {filename}: {error}. Continuing batch execution.")
            batch_summaries.append({
                "video_name": filename,
                "total_duration": 0.0,
                "number_of_shots": 0,
                "anchor_shots": 0,
                "field_report_shots": 0,
                "b_roll_shots": 0,
                "title_card_shots": 0,
                "other_shots": 0,
                "status": f"Error: {str(error)}"
            })

    # Export consolidated batch summary CSV & JSON
    if batch_summaries:
        os.makedirs(output_dir, exist_ok=True)
        batch_csv = os.path.join(output_dir, "batch_summary.csv")
        batch_json = os.path.join(output_dir, "batch_summary.json")

        csv_headers = [
            "video_name", "total_duration", "number_of_shots",
            "anchor_shots", "field_report_shots", "b_roll_shots",
            "title_card_shots", "other_shots", "status"
        ]
        with open(batch_csv, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=csv_headers)
            writer.writeheader()
            for row in batch_summaries:
                writer.writerow(row)

        with open(batch_json, "w", encoding="utf-8") as f_json:
            json.dump(batch_summaries, f_json, indent=2)

        print(f"\nBatch processing complete! Summary written to {batch_csv}")

    return batch_summaries
