# Temporal Scene-Boundary Detection & Rule-Based Visual Taxonomy System

A pure Python/OpenCV computer vision system for news bulletin videos that detects visual shot boundaries, extracts explainable visual features, classifies shots using transparent rule-based scoring, detects redundant visual content, and evaluates classification accuracy against ground truth annotations.

This system operates strictly using classical computer vision (Python + OpenCV + NumPy) without deep learning, neural networks, or cloud APIs.




## 🛠️ CLI Modes & Commands

Navigate to the project folder:
```bash
cd path/to/temporal_scene_boundary_detection
```

### 1. Real-Video Inspection / Calibration Mode (`--inspect`)
Runs video reading, shot detection, and feature extraction. Prints raw feature values in a formatted table and saves `results/<video_name>_inspection.csv`. **Stops before classification**.
```bash
python3 main.py --inspect "input_videos/news_bulletin_1.mp4"
```

### 2. Generate Ground-Truth Annotation Template (`annotate_ground_truth.py`)
Extracts keyframe JPEG images for every detected shot into `ground_truth/<video_name>/` and generates a blank `results/<video_name>_ground_truth.json` with `true_label: ""` for manual visual labeling.
```bash
python3 annotate_ground_truth.py "input_videos/news_bulletin_1.mp4"
```

### 3. Single Video Processing
Processes a single video through the complete pipeline (detection, features, rule classification, redundancy) and exports CSV and JSON reports to `results/`.
```bash
python3 main.py --video "input_videos/news_bulletin_1.mp4"
```

### 4. Batch Video Processing
Processes all videos in a folder with error isolation (one failing video will not stop the batch):
```bash
python3 main.py --batch "input_videos/"
```

### 5. Evaluation Mode (JSON Ground Truth)
Compares system predictions against ground truth labels from a `.json` file:
```bash
python3 main.py --evaluate "results/news_bulletin_1_results.json" --ground-truth "results/news_bulletin_1_ground_truth.json"

---

## ⚙️ Calibration Workflow

To calibrate classification rules for new real-world broadcast sources:

1. **Run Inspection Mode**:
   ```bash
   python3 main.py --inspect "input_videos/your_news_video.mp4"
   ```
   Inspect the raw feature distributions in `results/your_news_video_inspection.csv` (stability, framing symmetry, edge density, text coverage).

2. **Generate Ground Truth Template**:
   ```bash
   python3 annotate_ground_truth.py "input_videos/your_news_video.mp4"
   ```
   View keyframe images in `ground_truth/your_news_video/` and manually fill in `true_label` inside `results/your_news_video_ground_truth.json`.

3. **Run Pipeline & Measure Accuracy**:
   ```bash
   python3 main.py --video "input_videos/your_news_video.mp4"
   python3 main.py --evaluate "results/your_news_video_results.json" --ground-truth "results/your_news_video_ground_truth.json"
   ```

4. **Tune Parameters in `config.py`**:

---

## ⚠️ System Limitations

Since this system uses classical rule-based computer vision without deep learning or semantic models:

1. **Complex Motion Overlap**: Fast hand movements or rapid camera pans by a field reporter can mimic B-roll motion statistics.
2. **Graphic-Heavy Studio Desks**: Full-screen graphic animations placed over studio presenters may reduce spatial symmetry scores.
# Temporal Scene-Boundary Detection & Visual Taxonomy (Simplified)

This repository implements a compact, explainable pipeline using OpenCV and NumPy:

- Read video frames and sample at a fixed rate
- Detect shot boundaries using HSV histogram differences
- Extract simple features per shot (brightness, edge density, motion, text, face)
- Classify each shot with clear rule-based logic into: Anchor Studio, Field Report, B-Roll, Title Card, Other
- Save results and evaluate against JSON ground truth

Quick commands:

```bash
cd temporal_scene_boundary_detection

# Inspect features only (stops before classification)
python3 main.py --inspect input_videos/rel_news.mp4

# Process a single video end-to-end
python3 main.py --video input_videos/rel_news.mp4

# Evaluate predictions against ground truth JSON
python3 main.py --evaluate results/rel_news_results.json --ground-truth results/rel_news_ground_truth.json
```

The project is intentionally small so you can explain each step in a viva: what a frame is,
what FPS is, why HSV histograms are used for color comparison, why we sample frames,
and how simple visual features map to rule-based categories.
4. **Ambiguous Framing**: Shots where a presenter is framed at an angle in an outdoor studio setup may fall into the `Other` category due to balanced score margins between `Anchor Studio` and `Field Report`.
