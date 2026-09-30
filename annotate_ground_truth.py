"""
Ground-Truth Annotation Tool — Streaming / Memory-Safe Version.

Usage:
    # Full video (no limit) — works even on 12-hour streams
    python3 annotate_ground_truth.py "https://www.youtube.com/live/Oa-Ah1Xz290?si=..."

    # Limit to first N seconds (optional)
    python3 annotate_ground_truth.py "https://www.youtube.com/live/..." --max-duration 3600

How it works (streaming architecture):
    - Reads and processes ONE frame at a time from the video/stream.
    - Detects shot boundaries by comparing adjacent frame HSV histograms.
    - When a shot boundary is found, the keyframe (center frame of the shot so far)
      is immediately saved to disk as a JPEG and its pixel data is discarded from RAM.
    - At any point, only a small sliding buffer of frames (one shot worth) is in RAM.
    - This allows processing of arbitrarily long videos (12 hours, 24 hours etc.)
      on a standard laptop without crashing.
"""

import sys
import os
import json
import argparse
import cv2
import numpy as np
from typing import List, Dict, Any, Optional

from video_reader import VideoReader, is_url
import config


def _compute_hist(frame: np.ndarray) -> np.ndarray:
    """Compute normalised 3-D HSV histogram for one frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def _hist_dissimilarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """Return dissimilarity score in [0, 1] between two histograms."""
    corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return float(max(0.0, min(1.0, (1.0 - corr) / 2.0)))


def generate_ground_truth_streaming(
    video_path: str,
    frame_skip: int = config.FRAME_SKIP,
    max_duration: Optional[float] = None,
    max_frames: Optional[int] = None,
    histogram_threshold: float = config.HISTOGRAM_THRESHOLD,
    minimum_shot_duration: float = config.MINIMUM_SHOT_DURATION,
    downscale_max_dim: int = 480,
) -> None:
    """
    Stream through the entire video one frame at a time.
    Detect shot boundaries on-the-fly and save one keyframe JPEG per shot
    to disk immediately, keeping RAM usage constant regardless of video length.
    """
    is_remote = is_url(video_path)
    if not is_remote and not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    print(f"\n--- Ground Truth Annotation Tool (Streaming Mode) ---")
    print(f"Source          : {video_path}")
    if max_duration:
        print(f"Duration limit  : {max_duration:.0f}s ({max_duration/3600:.2f} hrs)")
    else:
        print(f"Duration limit  : None - will process the FULL video")
    print(f"Frame skip      : every {frame_skip} frames")
    print(f"Shot threshold  : {histogram_threshold}")

    reader = VideoReader(video_path)
    if not reader.open():
        print(f"Failed to open video source: {video_path}")
        sys.exit(1)

    metadata = reader.get_metadata()
    video_name = metadata["video_name"]
    video_basename = os.path.splitext(video_name)[0]
    fps = metadata["frame_rate"]
    total_dur = metadata["video_duration"]

    print(f"Resolution      : {metadata['frame_width']}x{metadata['frame_height']}")
    print(f"Frame rate      : {fps} FPS")
    print(f"Total duration  : {total_dur:.1f}s ({total_dur/3600:.2f} hrs)")
    print()

    image_output_dir = os.path.join("ground_truth", video_basename)
    os.makedirs(image_output_dir, exist_ok=True)
    json_output_dir = "results"
    os.makedirs(json_output_dir, exist_ok=True)

    shot_id = 0
    shot_buffer: List[Dict] = []
    prev_hist: Optional[np.ndarray] = None
    shot_start_time: float = 0.0
    gt_shots_list: List[Dict[str, Any]] = []
    sampled_idx = 0
    frame_idx = 0

    import time as _time
    t_start = _time.time()
    t_last_log = t_start

    def _commit_shot(buf, s_time, e_time):
        nonlocal shot_id
        duration = e_time - s_time
        if duration < minimum_shot_duration and shot_id > 0:
            if gt_shots_list:
                gt_shots_list[-1]["end_time"] = e_time
            return False
        shot_id += 1
        center_idx = len(buf) // 2
        keyframe = buf[center_idx]["frame"]
        img_path = os.path.join(image_output_dir, f"shot_{shot_id:04d}.jpg")
        cv2.imwrite(img_path, keyframe)
        gt_shots_list.append({
            "shot_id": shot_id,
            "start_time": round(s_time, 3),
            "end_time": round(e_time, 3),
            "true_label": ""
        })
        return True

    print("Streaming frames and detecting shots...")
    print("(Keyframes are saved to disk immediately - memory stays constant)\n")

    cap = reader.video_capture

    while True:
        success, raw_frame = cap.read()
        if not success:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps if fps > 0 else 0.0

        if max_duration is not None and timestamp > max_duration:
            print(f"\n[Cap reached] Stopped at {timestamp:.1f}s (limit: {max_duration:.0f}s)")
            break
        if max_frames is not None and sampled_idx >= max_frames:
            print(f"\n[Cap reached] Stopped at {sampled_idx} sampled frames")
            break

        h, w = raw_frame.shape[:2]
        if max(h, w) > downscale_max_dim:
            scale = downscale_max_dim / float(max(h, w))
            raw_frame = cv2.resize(
                raw_frame,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA
            )

        curr_hist = _compute_hist(raw_frame)

        is_boundary = False
        if prev_hist is not None:
            diff = _hist_dissimilarity(prev_hist, curr_hist)
            if diff >= histogram_threshold:
                is_boundary = True

        if is_boundary and shot_buffer:
            end_time = shot_buffer[-1]["ts"]
            _commit_shot(shot_buffer, shot_start_time, end_time)
            shot_start_time = timestamp
            shot_buffer = [{"frame": raw_frame.copy(), "ts": timestamp}]
        else:
            shot_buffer.append({"frame": raw_frame.copy(), "ts": timestamp})

        prev_hist = curr_hist
        sampled_idx += 1
        frame_idx += 1

        now = _time.time()
        if now - t_last_log >= 10.0:
            t_last_log = now
            elapsed = now - t_start
            speed = sampled_idx / max(0.1, elapsed)
            pct = (timestamp / total_dur * 100.0) if total_dur > 0 else 0.0
            h_ = int(timestamp // 3600)
            m_ = int((timestamp % 3600) // 60)
            s_ = int(timestamp % 60)
            th_ = int(total_dur // 3600)
            tm_ = int((total_dur % 3600) // 60)
            ts_ = int(total_dur % 60)
            print(
                f"[Progress] {h_:02d}:{m_:02d}:{s_:02d} / {th_:02d}:{tm_:02d}:{ts_:02d}"
                f"  ({pct:.1f}%)  |  Shots found: {shot_id}"
                f"  |  Sampled frames: {sampled_idx}"
                f"  |  Speed: {speed:.0f} fps",
                flush=True
            )

    reader.close()

    if shot_buffer:
        end_time = shot_buffer[-1]["ts"]
        _commit_shot(shot_buffer, shot_start_time, end_time)
        shot_buffer = []

    gt_json_path = os.path.join(json_output_dir, f"{video_basename}_ground_truth.json")
    gt_json_data = {
        "video_name": video_name,
        "video_source": video_path,
        "shots": gt_shots_list
    }
    with open(gt_json_path, "w", encoding="utf-8") as f_gt:
        json.dump(gt_json_data, f_gt, indent=2)

    print(f"\n{'='*55}")
    print(f"  Ground truth annotation complete!")
    print(f"{'='*55}")
    print(f"  Total shots detected : {shot_id}")
    print(f"  Keyframes saved to   : {image_output_dir}/")
    print(f"  Ground truth JSON    : {gt_json_path}")
    print(f"\nNext Steps:")
    print(f"  1. Open '{image_output_dir}/' and inspect shot keyframe images.")
    print(f"  2. Open '{gt_json_path}' and fill in 'true_label' for each shot.")
    print(f"     Choices: Anchor Studio | Field Report | B-Roll | Title Card | Other")
    print(f"  3. Run full pipeline:")
    print(f'     python3 main.py --url "{video_path}"')
    print(f"  4. Evaluate predictions vs ground truth:")
    print(f"     python3 main.py --evaluate results/{video_basename}_results.json --ground-truth {gt_json_path}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Ground-Truth Annotation Tool (Streaming / Memory-Safe).\n"
            "Processes the full video on-the-fly without loading everything into RAM.\n"
            "Works on 12-hour+ YouTube live streams."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "video_path", type=str,
        help="Local video file path OR YouTube / web URL"
    )
    parser.add_argument(
        "--max-duration", type=float, default=None,
        help=(
            "Stop after this many seconds. "
            "Omit (or pass 0) for the full video. "
            "Example: --max-duration 3600 for the first 1 hour."
        )
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop after this many sampled frames (optional)."
    )
    parser.add_argument(
        "--frame-skip", type=int, default=config.FRAME_SKIP,
        help=f"Sample 1 frame every N frames (default: {config.FRAME_SKIP}). "
             f"Increase to 5-10 for faster processing on long videos."
    )
    parser.add_argument(
        "--threshold", type=float, default=config.HISTOGRAM_THRESHOLD,
        help=f"HSV histogram dissimilarity threshold for shot boundary detection "
             f"(default: {config.HISTOGRAM_THRESHOLD})."
    )

    args = parser.parse_args()
    max_dur = args.max_duration if (args.max_duration and args.max_duration > 0) else None

    generate_ground_truth_streaming(
        video_path=args.video_path,
        frame_skip=args.frame_skip,
        max_duration=max_dur,
        max_frames=args.max_frames,
        histogram_threshold=args.threshold,
    )
