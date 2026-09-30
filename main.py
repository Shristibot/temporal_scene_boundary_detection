"""
Main Entrypoint CLI for Temporal Scene-Boundary Detection & Visual Taxonomy System.

Usage:
1. Real-Video Inspection / Calibration Mode (Stops before classification):
   python3 main.py --inspect input_videos/rel_news.mp4

2. Process local single video:
   python3 main.py --video input_videos/rel_news.mp4

3. Process online video URL (no manual download):
   python3 main.py --url "https://www.youtube.com/watch?v=..."

4. Process with on-screen audit overlay:
   python3 main.py --video input_videos/rel_news.mp4 --audit --ground-truth results/rel_news_ground_truth.json

5. Process batch of videos in directory:
   python3 main.py --batch input_videos/

6. Evaluate predictions against ground truth (JSON):
   python3 main.py --evaluate results/rel_news_results.json --ground-truth results/rel_news_ground_truth.json
"""

import sys
import os
import csv
import json
import argparse
from typing import List, Dict, Any, Optional

from video_reader import VideoReader
from shot_detector import detect_shot_boundaries
from feature_extractor import extract_shot_features
from batch_processor import process_single_video, process_batch
from evaluation import evaluate_predictions, format_confusion_matrix_text, load_ground_truth_from_json
import config


def run_inspection_mode(
    video_path: str,
    frame_skip: int = config.FRAME_SKIP,
    max_frames: Optional[int] = None,
    max_duration: Optional[float] = None
) -> None:
    """
    Run Real-Video Inspection / Calibration Mode.

    1. Opens video using video_reader.py.
    2. Runs shot detection using shot_detector.py.
    3. Extracts raw visual features for every shot using feature_extractor.py.
    4. Prints raw feature values in a formatted terminal table.
    5. Saves results to results/<video_name>_inspection.csv.
    6. STOPS before classification.
    """
    reader = VideoReader(video_path)
    if not reader.open():
        print(f"Failed to open video: {video_path}")
        sys.exit(1)

    video_name = reader.get_metadata()["video_name"]
    base_filename = os.path.splitext(video_name)[0]

    print(f"\n================ REAL-VIDEO INSPECTION MODE ================")
    print(f"Inspecting Video: {video_name}")

    metadata = reader.get_metadata()
    print(f"Metadata: {metadata['frame_width']}x{metadata['frame_height']} @ {metadata['frame_rate']} FPS, Duration: {metadata['video_duration']}s, Total Frames: {metadata['total_frames']}")

    sampled_frames = list(reader.read_sampled_frames(
        frame_skip=frame_skip,
        downscale_max_dim=480,
        max_frames=max_frames,
        max_duration=max_duration
    ))
    reader.close()

    if not sampled_frames:
        print(f"Error: No readable frames extracted from {video_name}")
        sys.exit(1)

    shots = detect_shot_boundaries(
        sampled_frames=sampled_frames,
        frame_rate=metadata["frame_rate"],
        histogram_threshold=config.HISTOGRAM_THRESHOLD,
        minimum_shot_duration=config.MINIMUM_SHOT_DURATION
    )
    print(f"Detected {len(shots)} shots for feature inspection.\n")

    inspection_rows: List[Dict[str, Any]] = []

    for shot in shots:
        features = extract_shot_features(shot)

        row = {
            "shot_id": shot["shot_id"],
            "start_time": shot["start_time"],
            "end_time": shot["end_time"],
            "duration": shot["shot_duration"],
            "avg_hue": features.get("avg_hue", 0.0),
            "avg_saturation": features.get("avg_saturation", 0.0),
            "avg_brightness": features.get("avg_brightness", 0.0),
            "visual_stability_score": features.get("visual_stability_score", 0.0),
            "avg_edge_density": features.get("avg_edge_density", 0.0),
            "avg_text_coverage": features.get("avg_text_coverage", 0.0),
            "avg_headline_text": features.get("avg_headline_text", 0.0),
            "avg_lower_third_text": features.get("avg_lower_third_text", 0.0)
        }
        inspection_rows.append(row)

    # Print Formatted Readable Table to Terminal
    header = f"{'Shot':<5} | {'Start(s)':<8} | {'End(s)':<8} | {'Dur(s)':<6} | {'Hue':<6} | {'Bright':<7} | {'Stab':<6} | {'EdgeDens':<9} | {'TextCov':<8} | {'HeadText':<8} | {'LowThird':<8}"
    print(header)
    print("-" * len(header))

    for r in inspection_rows:
        line = (
            f"{r['shot_id']:<5} | {r['start_time']:<8.2f} | {r['end_time']:<8.2f} | {r['duration']:<6.2f} | "
            f"{r['avg_hue']:<6.1f} | {r['avg_brightness']:<7.1f} | {r['visual_stability_score']:<6.2f} | "
            f"{r['avg_edge_density']:<9.4f} | {r['avg_text_coverage']:<8.4f} | {r['avg_headline_text']:<8.4f} | {r['avg_lower_third_text']:<8.4f}"
        )
        print(line)

    # Save to CSV
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{base_filename}_inspection.csv")

    csv_headers = [
        "shot_id", "start_time", "end_time", "duration", "avg_hue", "avg_saturation",
        "avg_brightness", "visual_stability_score", "avg_edge_density",
        "avg_text_coverage", "avg_headline_text", "avg_lower_third_text"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_headers)
        writer.writeheader()
        for r in inspection_rows:
            writer.writerow(r)

    print(f"\nSaved raw inspection feature CSV to: {csv_path}")
    print("Inspection mode complete. Stopped before classification.")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Temporal Scene-Boundary Detection & Visual Taxonomy System"
    )

    parser.add_argument("--inspect", type=str, help="Run Real-Video Inspection Mode on a local file or URL")
    parser.add_argument("--video", type=str, help="Path to single input local video file")
    parser.add_argument("--url", type=str, help="Online video URL (HTTP/HTTPS link) to process directly via streaming")
    parser.add_argument("--audit", action="store_true", help="Generate on-screen audit overlay video for accuracy inspection")
    parser.add_argument("--batch", type=str, help="Directory containing multiple input videos to process")
    parser.add_argument("--evaluate", type=str, help="Path to predictions JSON file to evaluate")
    parser.add_argument("--ground-truth", type=str, help="Path to ground truth JSON file for evaluation/audit matching")
    parser.add_argument("--output", type=str, default="results", help="Directory to save results (default: results)")
    parser.add_argument("--frame-skip", type=int, default=config.FRAME_SKIP, help=f"Frame sampling rate (default: {config.FRAME_SKIP})")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of sampled frames to process")
    parser.add_argument("--max-duration", type=float, default=None, help="Maximum video duration in seconds to process (e.g. 600 for 10 mins)")
    parser.add_argument("--debug-classifier", action="store_true", help="Print detailed classifier scores and evidence for each shot")

    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_arguments()

    # Enable audit mode globally if flag passed
    if args.audit:
        config.AUDIT_MODE = True

    # Mode 1: Inspection Mode
    if args.inspect:
        run_inspection_mode(
            args.inspect,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
            max_duration=args.max_duration
        )
        return

    # Mode 2: Online Video URL Mode
    if args.url:
        summary = process_single_video(
            video_path=args.url,
            output_dir=args.output,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
            max_duration=args.max_duration,
            debug_classifier=args.debug_classifier,
            audit=args.audit,
            ground_truth_path=args.ground_truth
        )
        print("\nURL Process Summary:")
        print(json.dumps(summary, indent=2))
        return

    # Mode 3: Evaluation Mode
    if args.evaluate:
        if not args.ground_truth:
            print("Error: --ground-truth file path (.json) must be provided when running --evaluate.")
            sys.exit(1)

        if not os.path.exists(args.evaluate) or not os.path.exists(args.ground_truth):
            print(f"Error: Specified prediction or ground truth file does not exist.")
            sys.exit(1)

        with open(args.evaluate, "r", encoding="utf-8") as f_pred:
            pred_data = json.load(f_pred)
            predictions = pred_data.get("shots", [])

        ground_truth = load_ground_truth_from_json(args.ground_truth)

        eval_res = evaluate_predictions(predictions, ground_truth)

        print("\n================ EVALUATION RESULTS ================")
        print(f"Total Shots Evaluated: {eval_res['total_evaluated']}")
        print(f"Correct Predictions:  {eval_res['correct_predictions']}")
        print(f"Overall Accuracy:     {eval_res['overall_accuracy']}%\n")

        print("Per-Category Accuracy:")
        for cat, acc in eval_res['per_category_accuracy'].items():
            tot = eval_res['category_totals'][cat]
            print(f"  - {cat:<15}: {acc:>6.2f}% ({tot} shots)")

        print("\nConfusion Matrix:")
        print(format_confusion_matrix_text(eval_res['confusion_matrix']))
        print("====================================================\n")
        return

    # Mode 4: Local Single Video Processing Mode
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)

        summary = process_single_video(
            video_path=args.video,
            output_dir=args.output,
            frame_skip=args.frame_skip,
            max_frames=args.max_frames,
            max_duration=args.max_duration,
            debug_classifier=args.debug_classifier,
            audit=args.audit,
            ground_truth_path=args.ground_truth
        )
        print("\nProcess Summary:")
        print(json.dumps(summary, indent=2))
        return

    # Mode 5: Batch Processing Mode
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"Error: Batch input folder not found: {args.batch}")
            sys.exit(1)

        summaries = process_batch(
            input_folder=args.batch,
            output_dir=args.output,
            frame_skip=args.frame_skip,
            debug_classifier=args.debug_classifier,
            audit=args.audit,
            ground_truth_path=args.ground_truth
        )
        print(f"\nBatch processing complete for {len(summaries)} videos.")
        return

    # Default fallback: Run batch demonstration on input_videos/
    print("No mode specified. Running batch demonstration on input_videos/...")
    process_batch(
        input_folder="input_videos",
        output_dir="results",
        frame_skip=args.frame_skip,
        debug_classifier=args.debug_classifier,
        audit=args.audit,
        ground_truth_path=args.ground_truth
    )


if __name__ == "__main__":
    main()
