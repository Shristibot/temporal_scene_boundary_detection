# Temporal Scene-Boundary Detection & Rule-Based Visual Taxonomy System

A pure Python/OpenCV computer vision system for news bulletin videos that detects visual shot boundaries, extracts explainable visual features, classifies shots using transparent rule-based scoring, detects redundant visual content, and evaluates classification accuracy against ground truth annotations.

This system operates strictly using classical computer vision (Python + OpenCV + NumPy) without deep learning, neural networks, or cloud APIs.

---

## CLI Modes & Commands

Navigate to the project folder:
```bash
cd path/to/temporal_scene_boundary_detection
```

### 1. Real-Video Inspection / Calibration Mode (`--inspect`)
Runs video reading, shot detection, and feature extraction. Prints raw feature values in a formatted table and saves `results/<video_name>_inspection.csv`. **Stops before classification**.
```bash
python3 main.py --inspect input_videos/YOUR_VIDEO.mp4
```

### 2. Generate Ground-Truth Annotation Template (`annotate_ground_truth.py`)
Extracts keyframe JPEG images for every detected shot into `ground_truth/<video_name>/` and generates a blank `results/<video_name>_ground_truth.json` with `true_label: ""` for manual visual labeling.
```bash
python3 annotate_ground_truth.py input_videos/YOUR_VIDEO.mp4
```

### 3. Single Local Video Processing
Processes a single video through the complete pipeline (detection, features, rule classification, redundancy) and exports CSV and JSON reports to `results/`.
```bash
python3 main.py --video input_videos/YOUR_VIDEO.mp4
```

### 4. Online Video URL Processing
Streams and processes a YouTube or other online video directly without downloading first.
```bash
python3 main.py --url "https://www.youtube.com/watch?v=XC3WUX5NreI"
```

### 5. Download + Process YouTube URLs (Recommended for Full Videos)
Downloads the complete video locally via `yt-dlp`, then runs the full pipeline. Supports single URLs, multiple URLs, or a text file of URLs.
```bash
# Single URL
python3 download_and_process.py "https://www.youtube.com/watch?v=XC3WUX5NreI"

# Multiple URLs at once
python3 download_and_process.py "URL_1" "URL_2" "URL_3"

# URLs from a text file (one per line)
python3 download_and_process.py --file video_urls.txt
```
Optional flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--input-dir DIR` | `input_videos` | Where to save downloaded videos |
| `--output-dir DIR` | `results` | Where to save result JSON/CSV |
| `--gt-dir DIR` | `ground_truth` | Where to save ground-truth scaffolds |
| `--force-redownload` | off | Re-download even if cached locally |

### 6. Batch Video Processing
Processes all videos in a folder with error isolation (one failing video will not stop the batch):
```bash
python3 main.py --batch input_videos/
```

### 7. Evaluation Mode (JSON Ground Truth)
Compares system predictions against ground truth labels from a `.json` file:
```bash
python3 main.py --evaluate results/Stalin_Vs_Vijay__Tamil_Nadu_s_New_Battle_Over_Emergency_Legacy_results.json --ground-truth ground_truth/Stalin_Vs_Vijay__Tamil_Nadu_s_New_Battle_Over_Emergency_Legacy/Stalin_Vs_Vijay__Tamil_Nadu_s_New_Battle_Over_Emergency_Legacy_ground_truth.json
```

### 8. Audit Overlay Mode
Generates an annotated video with on-screen shot labels for visual inspection. Requires `--ground-truth` to match predictions:
```bash
python3 main.py --video input_videos/YOUR_VIDEO.mp4 --audit --ground-truth ground_truth/YOUR_VIDEO/YOUR_VIDEO_ground_truth.json
```

### Additional Flags

| Flag | Description |
|------|-------------|
| `--frame-skip N` | Sample every Nth frame (default from `config.py`) |
| `--max-frames N` | Cap the number of sampled frames to process |
| `--max-duration S` | Cap video processing at S seconds (e.g. `600` for 10 mins) |
| `--output DIR` | Override the results output directory (default: `results`) |
| `--debug-classifier` | Print detailed per-shot classifier scores and rule evidence |

---

## ⚙️ Calibration Workflow

To calibrate classification rules for new real-world broadcast sources:

1. **Run Inspection Mode**:
   ```bash
   python3 main.py --inspect input_videos/your_news_video.mp4
   ```
   Inspect the raw feature distributions in `results/your_news_video_inspection.csv` (stability, framing symmetry, edge density, text coverage).

2. **Generate Ground Truth Template**:
   ```bash
   python3 annotate_ground_truth.py input_videos/your_news_video.mp4
   ```
   View keyframe images in `ground_truth/your_news_video/` and manually fill in `true_label` inside `results/your_news_video_ground_truth.json`.

3. **Run Pipeline & Measure Accuracy**:
   ```bash
   python3 main.py --video input_videos/your_news_video.mp4
   python3 main.py --evaluate results/your_news_video_results.json --ground-truth results/your_news_video_ground_truth.json
   ```

4. **Tune Parameters in `config.py`**

---

## ⚠️ System Limitations

Since this system uses classical rule-based computer vision without deep learning or semantic models:

1. **Complex Motion Overlap**: Fast hand movements or rapid camera pans by a field reporter can mimic B-roll motion statistics.
2. **Graphic-Heavy Studio Desks**: Full-screen graphic animations placed over studio presenters may reduce spatial symmetry scores.
3. **Ambiguous Framing**: Shots where a presenter is framed at an angle in an outdoor studio setup may fall into the `Other` category due to balanced score margins between `Anchor Studio` and `Field Report`.
