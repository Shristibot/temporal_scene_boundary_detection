from typing import List, Dict, Any
import cv2
import numpy as np
import config

def calculate_histogram_difference(prev_item, curr_item) -> float:
    if isinstance(prev_item, dict) and "hist" in prev_item and "hist" in curr_item:
        hprev = prev_item["hist"]
        hcurr = curr_item["hist"]
    else:
        prev_frame = prev_item["frame"] if isinstance(prev_item, dict) else prev_item
        curr_frame = curr_item["frame"] if isinstance(curr_item, dict) else curr_item
        prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
        curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
        hprev = cv2.calcHist([prev_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        hcurr = cv2.calcHist([curr_hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hprev, hprev, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(hcurr, hcurr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Correlation: 1 means identical, convert to dissimilarity
    corr = cv2.compareHist(hprev, hcurr, cv2.HISTCMP_CORREL)
    dissimilarity = max(0.0, min(1.0, (1.0 - corr) / 2.0))
    return float(dissimilarity)


def detect_shot_boundaries(sampled_frames: List[Dict[str, Any]], frame_rate: float, histogram_threshold: float = config.HISTOGRAM_THRESHOLD,minimum_shot_duration: float = config.MINIMUM_SHOT_DURATION) -> List[Dict[str, Any]]:
    
    if not sampled_frames:
        return []

    if len(sampled_frames) == 1:
        f = sampled_frames[0]
        return [{
            "shot_id": 1,
            "start_frame": f["frame_number"],
            "end_frame": f["frame_number"],
            "start_time": f["timestamp"],
            "end_time": f["timestamp"],
            "shot_duration": 0.0,
            "sampled_frames": sampled_frames
        }]

    diffs = []
    for i in range(1, len(sampled_frames)):
        prev = sampled_frames[i - 1]
        curr = sampled_frames[i]
        d = calculate_histogram_difference(prev, curr)
        diffs.append(d)

    # Mark boundary at frame i when diff >= threshold
    boundaries = [0]
    for i, d in enumerate(diffs):
        # Use histogram dissimilarity threshold to mark boundaries
        if d >= histogram_threshold:
            boundaries.append(i + 1)

    # Build shots from boundaries
    shots = []
    total = len(sampled_frames)
    for idx in range(len(boundaries)):
        start = boundaries[idx]
        end = boundaries[idx + 1] - 1 if idx + 1 < len(boundaries) else total - 1
        shot_frames = sampled_frames[start:end + 1]
        start_time = shot_frames[0]["timestamp"]
        end_time = shot_frames[-1]["timestamp"]
        duration = round(end_time - start_time, 3)
        shots.append({
            "start_frame": shot_frames[0]["frame_number"],
            "end_frame": shot_frames[-1]["frame_number"],
            "start_time": start_time,
            "end_time": end_time,
            "shot_duration": duration,
            "sampled_frames": shot_frames
        })

    # Merge shots shorter than minimum_shot_duration into previous shot
    merged = []
    for s in shots:
        if not merged:
            merged.append(s)
        else:
            if s["shot_duration"] < minimum_shot_duration:
                prev = merged[-1]
                prev["end_frame"] = s["end_frame"]
                prev["end_time"] = s["end_time"]
                prev["shot_duration"] = round(prev["end_time"] - prev["start_time"], 3)
                prev["sampled_frames"].extend(s["sampled_frames"])
            else:
                merged.append(s)

    # Assign shot_id
    for i, s in enumerate(merged, start=1):
        s["shot_id"] = i
    return merged
