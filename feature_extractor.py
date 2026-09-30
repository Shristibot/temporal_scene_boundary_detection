import os
import cv2
import numpy as np

CASCADE_PATH = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH) if os.path.exists(CASCADE_PATH) else None


def extract_frame_features(frame):
    h, w = frame.shape[:2]
    total = h * w
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Brightness
    brightness = float(np.mean(gray))

    # Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges) / (total + 1e-6))

    # Simple left-right symmetry on edges (center column compares left vs right)
    mid = w // 2
    left_edges = np.count_nonzero(edges[:, :mid])
    right_edges = np.count_nonzero(edges[:, mid:])
    lr_diff = abs(left_edges - right_edges) / float(max(1, left_edges + right_edges))
    symmetry = float(max(0.0, 1.0 - lr_diff))

    # Text-like regions via simple Sobel + closing
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    _, tbin = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    text_mask = cv2.morphologyEx(tbin, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    headline_area = 0
    lower_third_area = 0
    total_text_area = 0
    lower_third_y = int(h * 0.78)
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if cw > 15 and ch > 6:
            total_text_area += area
            if y >= lower_third_y:
                lower_third_area += area
            else:
                # Large upper/center text
                if cw > w * 0.22 and ch > h * 0.04:
                    headline_area += area

    headline_ratio = float(headline_area / (total + 1e-6))
    lower_third_ratio = float(lower_third_area / (total + 1e-6))
    text_ratio = float(total_text_area / (total + 1e-6))

    # Face detection (fast, downscale)
    face_count = 0
    centered_face = False
    face_area = 0.0
    if FACE_CASCADE is not None and not FACE_CASCADE.empty():
        small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        faces = FACE_CASCADE.detectMultiScale(small, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        face_count = len(faces)
        for (fx, fy, fw, fh) in faces:
            # map back coordinates
            cx = (fx + fw / 2.0) * 2
            face_area += (fw * fh) * 4
            if w * 0.25 <= cx <= w * 0.75:
                centered_face = True
    face_area_ratio = float(face_area / (total + 1e-6))

    return {
        "brightness": brightness,
        "edge_density": edge_density,
        "symmetry": symmetry,
        "headline_ratio": headline_ratio,
        "lower_third_ratio": lower_third_ratio,
        "text_ratio": text_ratio,
        "face_count": face_count,
        "centered_face": centered_face,
        "face_area_ratio": face_area_ratio
    }


def extract_shot_features(shot):
    frames = shot.get("sampled_frames", [])
    if not frames:
        return {}

    feats = []
    diffs = []

    for i, f in enumerate(frames):
        if "features" in f:
            feats.append(f["features"])
        elif "frame" in f and f["frame"] is not None:
            feats.append(extract_frame_features(f["frame"]))

        if "motion_diff" in f:
            diffs.append(f["motion_diff"])
        elif i > 0 and "frame" in f and f["frame"] is not None and "frame" in frames[i-1] and frames[i-1]["frame"] is not None:
            g1 = cv2.cvtColor(frames[i - 1]["frame"], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(f["frame"], cv2.COLOR_BGR2GRAY)
            diffs.append(float(np.mean(cv2.absdiff(g1, g2)) / 255.0))

    if not feats:
        return {}

    motion = float(np.mean(diffs)) if diffs else 0.0

    avg_brightness = float(np.mean([f["brightness"] for f in feats]))
    avg_edge = float(np.mean([f["edge_density"] for f in feats]))
    avg_sym = float(np.mean([f["symmetry"] for f in feats]))
    avg_headline = float(np.mean([f["headline_ratio"] for f in feats]))
    avg_lower = float(np.mean([f["lower_third_ratio"] for f in feats]))
    avg_text = float(np.mean([f["text_ratio"] for f in feats]))
    face_count = max([f["face_count"] for f in feats])
    centered_face = any([f["centered_face"] for f in feats])
    avg_face_area = float(np.mean([f["face_area_ratio"] for f in feats]))

    stability = max(0.0, 1.0 - motion * 10.0)

    mid_idx = len(frames) // 2
    keyframe = frames[mid_idx].get("frame") if frames else None

    return {
        "shot_id": shot.get("shot_id"),
        "start_time": shot.get("start_time"),
        "end_time": shot.get("end_time"),
        "shot_duration": shot.get("shot_duration"),
        "visual_stability_score": round(stability, 3),
        "intra_shot_motion_mean": round(motion, 4),
        "avg_brightness": round(avg_brightness, 2),
        "avg_edge_density": round(avg_edge, 4),
        "avg_spatial_symmetry": round(avg_sym, 3),
        "avg_headline_text": round(avg_headline, 4),
        "avg_lower_third_text": round(avg_lower, 4),
        "avg_text_coverage": round(avg_text, 4),
        "detected_faces_count": int(face_count),
        "presenter_centered": bool(centered_face),
        "avg_face_area": round(avg_face_area, 4),
        "keyframe": keyframe
    }
