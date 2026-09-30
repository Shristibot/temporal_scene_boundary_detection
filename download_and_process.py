"""
download_and_process.py
=======================
Generic video download + evaluation runner for
Temporal Scene-Boundary Detection & Rule-Based Visual Taxonomy Classification.

Usage - single URL:
    python3 download_and_process.py "https://www.youtube.com/watch?v=ABC123"

Usage - multiple URLs:
    python3 download_and_process.py "URL_1" "URL_2" "URL_3"

Usage - URLs from a text file (one URL per line):
    python3 download_and_process.py --file video_urls.txt

Options:
    --input-dir   DIR   Where to save downloaded videos  (default: input_videos)
    --output-dir  DIR   Where to save results            (default: results)
    --gt-dir      DIR   Where to save ground-truth JSON  (default: ground_truth)

Design principles:
  * NO hardcoded video URLs.
  * NO artificial duration cap on downloaded videos.
  * Feeds complete local file into the EXISTING pipeline (zero pipeline changes).
  * If a cached file looks truncated vs. the source, re-downloads it.
  * Continues past individual failures and reports everything at the end.

All shot-detection thresholds come from config.py and are NOT duplicated here.
"""

import argparse
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Directory defaults (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_INPUT_DIR:  str = "input_videos"
DEFAULT_OUTPUT_DIR: str = "results"
DEFAULT_GT_DIR:     str = "ground_truth"

# Tolerance used when comparing source vs. downloaded duration.
# If |downloaded - source| / source > this fraction, treat as truncated.
DURATION_TOLERANCE: float = 0.05   # 5 %


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fmt_seconds(s: Optional[float]) -> str:
    """Format a duration in seconds as HH:MM:SS, or 'N/A' if unknown."""
    if s is None:
        return "N/A"
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def print_sep(char: str = "=", width: int = 70) -> None:
    print(char * width, flush=True)


# ---------------------------------------------------------------------------
# URL collection
# ---------------------------------------------------------------------------

def collect_urls(raw_urls: List[str], file_path: Optional[str]) -> List[str]:
    """
    Merge URLs from command-line positional args and/or a text file.
    Blank lines and lines starting with '#' in the file are ignored.
    """
    urls: List[str] = list(raw_urls)

    if file_path:
        if not os.path.exists(file_path):
            log(f"ERROR: URL file not found: {file_path}")
            sys.exit(1)
        with open(file_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    urls.append(line)

    # Deduplicate while preserving order
    seen: set = set()
    deduped: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)

    return deduped


# ---------------------------------------------------------------------------
# yt-dlp helpers
# ---------------------------------------------------------------------------

def _get_yt_dlp():
    """Import yt_dlp or exit with a clear message."""
    try:
        import yt_dlp
        return yt_dlp
    except ImportError:
        log("ERROR: 'yt-dlp' is not installed.")
        log("       Install with:  pip install yt-dlp")
        sys.exit(1)


# Android player client resolves format 18 (360p combined mp4) without
# needing a GVS PO Token, which makes it the most reliable client for
# news/live-stream content that triggers bot-detection on the web client.
_EXTRACTOR_ARGS = {"youtube": {"player_client": ["android"]}}


def probe_url(url: str) -> Optional[Dict[str, Any]]:
    """
    Fetch metadata for *url* without downloading.
    Returns the yt-dlp info dict, or None on failure (with reason logged).
    """
    yt_dlp = _get_yt_dlp()

    probe_opts = {
        "quiet":         True,
        "no_warnings":   True,
        "skip_download": True,
        "extract_flat":  False,
        "extractor_args": _EXTRACTOR_ARGS,
    }
    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        return info
    except yt_dlp.utils.DownloadError as exc:
        log(f"  PROBE FAILED ({type(exc).__name__}): {exc}")
    except Exception as exc:
        log(f"  PROBE ERROR  ({type(exc).__name__}): {exc}")
    return None


def make_safe_filename(info: Dict[str, Any]) -> str:
    """
    Build a filesystem-safe filename from yt-dlp metadata.
    Safe chars: alphanumeric, space, underscore, hyphen.
    Collapses whitespace to underscores and caps at 80 chars.
    """
    title: str = info.get("title") or info.get("id") or "video"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-")
    cleaned = "".join(c if c in safe else "_" for c in title)
    cleaned = "_".join(cleaned.split())   # collapse whitespace
    cleaned = cleaned[:80]
    ext: str = info.get("ext") or "mp4"
    return f"{cleaned}.{ext}"


def get_local_video_duration(path: str) -> Optional[float]:
    """
    Read the duration of a local video file using OpenCV.
    Returns duration in seconds, or None if it cannot be read.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        fps   = cap.get(cv2.CAP_PROP_FPS)
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps and fps > 0 and total and total > 0:
            return total / fps
    except Exception:
        pass
    return None


def verify_download(
    local_path: str,
    source_duration_s: Optional[float],
) -> Tuple[bool, str]:
    """
    Verify a downloaded video file.

    Checks:
      1. File exists and is non-empty.
      2. File can be opened by OpenCV.
      3. If source_duration_s is known: downloaded duration is within
         DURATION_TOLERANCE of source duration (catches truncated files).

    Returns (ok: bool, reason: str).
    """
    if not os.path.exists(local_path):
        return False, "File does not exist after download"
    if os.path.getsize(local_path) == 0:
        return False, "File is empty (0 bytes)"

    dl_duration = get_local_video_duration(local_path)
    if dl_duration is None:
        return False, "Cannot open file with OpenCV - may be corrupt"

    if source_duration_s and source_duration_s > 0:
        ratio = abs(dl_duration - source_duration_s) / source_duration_s
        if ratio > DURATION_TOLERANCE:
            return (
                False,
                (
                    f"Duration mismatch: source={fmt_seconds(source_duration_s)} "
                    f"downloaded={fmt_seconds(dl_duration)} "
                    f"(deviation {ratio*100:.1f}% > tolerance {DURATION_TOLERANCE*100:.0f}%)"
                ),
            )

    return True, f"OK (downloaded duration: {fmt_seconds(dl_duration)})"


def download_video(
    url: str,
    output_dir: str,
    force_redownload: bool = False,
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Download *url* into *output_dir* as a complete mp4.

    Steps:
      1. Probe metadata to get title, extension, and source duration.
      2. If a matching local file already exists, verify it is complete:
           - If complete  -> reuse it (skip download).
           - If truncated -> delete and re-download.
      3. Download the FULL video (no duration cap, no download_ranges).
      4. Verify the downloaded file.

    Returns:
        (local_path, source_duration_s, downloaded_duration_s)
        local_path is None on failure.
    """
    yt_dlp = _get_yt_dlp()
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Probe
    # ------------------------------------------------------------------
    log("  Probing URL...")
    info = probe_url(url)
    if not info:
        return None, None, None

    source_duration_s: Optional[float] = info.get("duration")
    title = info.get("title") or info.get("id") or url
    log(f"  Title           : {title}")
    log(f"  Source duration : {fmt_seconds(source_duration_s)}")

    filename  = make_safe_filename(info)
    dest_path = os.path.join(output_dir, filename)

    # ------------------------------------------------------------------
    # Step 2: Check cached file
    # ------------------------------------------------------------------
    if (
        not force_redownload
        and os.path.exists(dest_path)
        and os.path.getsize(dest_path) > 0
    ):
        ok, reason = verify_download(dest_path, source_duration_s)
        dl_duration = get_local_video_duration(dest_path)
        if ok:
            log(f"  Cache verification: PASSED - {reason}")
            print(f"    Video               : {title}")
            print(f"    Source duration     : {fmt_seconds(source_duration_s)}")
            print(f"    Downloaded duration : {fmt_seconds(dl_duration)}")
            print(f"    Status              : COMPLETE")
            return dest_path, source_duration_s, dl_duration
        else:
            log(f"  Cache verification: FAILED - {reason}")
            print(f"    Video               : {title}")
            print(f"    Source duration     : {fmt_seconds(source_duration_s)}")
            print(f"    Downloaded duration : {fmt_seconds(dl_duration)}")
            print(f"    Status              : INVALID/TRUNCATED (re-downloading complete video)")
            log("  Deleting cached file and re-downloading complete video...")
            try:
                os.remove(dest_path)
            except OSError as err:
                log(f"  Warning: could not delete {dest_path}: {err}")

    # ------------------------------------------------------------------
    # Step 3: Download COMPLETE video - no duration cap whatsoever
    # ------------------------------------------------------------------
    log("  Downloading complete video (no cap)...")
    download_opts = {
        # format 18 = 360p combined mp4 - the only format available for
        # the android client without a GVS PO Token.
        # The fallback chain handles other sites / future YouTube changes.
        "format": "18/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": dest_path,
        "quiet": False,
        "no_warnings": False,
        "merge_output_format": "mp4",
        "writethumbnail": False,
        "writesubtitles": False,
        "writeautomaticsub": False,
        "noplaylist": True,
        "extractor_args": _EXTRACTOR_ARGS,
        # NO download_ranges, NO force_keyframes_at_cuts - full download.
    }

    try:
        with yt_dlp.YoutubeDL(download_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:
        log(f"  DOWNLOAD FAILED: {exc}")
        return None, source_duration_s, None
    except Exception as exc:
        log(f"  UNEXPECTED ERROR during download: {exc}")
        return None, source_duration_s, None

    # yt-dlp may adjust the extension or template path
    if not os.path.exists(dest_path):
        candidate = dest_path + ".mp4"
        if os.path.exists(candidate):
            dest_path = candidate
        else:
            base_cleaned = os.path.splitext(os.path.basename(dest_path))[0]
            matches = [
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if f.startswith(base_cleaned) and not f.endswith((".part", ".ytdl"))
            ]
            if matches:
                dest_path = matches[0]
            else:
                log(f"  ERROR: Expected file not found after download: {dest_path}")
                return None, source_duration_s, None

    # ------------------------------------------------------------------
    # Step 4: Verify
    # ------------------------------------------------------------------
    ok, reason = verify_download(dest_path, source_duration_s)
    size_mb    = os.path.getsize(dest_path) / (1024 * 1024)
    dl_duration = get_local_video_duration(dest_path)

    log(f"  File size       : {size_mb:.1f} MB")
    log(f"  Verification    : {'PASSED' if ok else 'FAILED'} - {reason}")
    print(f"    Video               : {title}")
    print(f"    Source duration     : {fmt_seconds(source_duration_s)}")
    print(f"    Downloaded duration : {fmt_seconds(dl_duration)}")
    print(f"    Status              : {'COMPLETE' if ok else 'INVALID/TRUNCATED'}")

    if not ok:
        return None, source_duration_s, dl_duration

    return dest_path, source_duration_s, dl_duration


# ---------------------------------------------------------------------------
# Ground-truth scaffolding
# ---------------------------------------------------------------------------

def write_unverified_ground_truth(
    video_name: str,
    shots: List[Dict[str, Any]],
    gt_dir: str = DEFAULT_GT_DIR,
) -> Optional[str]:
    """
    Write a ground-truth JSON scaffold for a newly processed video.
    Every shot is marked UNVERIFIED - no labels are assigned.
    Returns the path written, or None on error.
    """
    base    = os.path.splitext(video_name)[0]
    out_dir = os.path.join(gt_dir, base)
    os.makedirs(out_dir, exist_ok=True)

    gt_entries = [
        {
            "shot_id":             shot.get("shot_id"),
            "start_time":          shot.get("start_time"),
            "end_time":            shot.get("end_time"),
            "duration":            shot.get("duration"),
            "predicted_label":     shot.get("classification"),
            "true_label":          "",          # blank - fill in manually
            "verification_status": "UNVERIFIED",
        }
        for shot in shots
    ]

    gt_path = os.path.join(out_dir, f"{base}_ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as fh:
        json.dump(gt_entries, fh, indent=2)

    log(f"  Ground-truth scaffold -> {gt_path}")
    return gt_path


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="download_and_process.py",
        description=(
            "Download any supported video URL completely and evaluate it "
            "through the Temporal Scene-Boundary Detection pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 download_and_process.py "https://www.youtube.com/watch?v=ABC123"
  python3 download_and_process.py "URL_1" "URL_2" "URL_3"
  python3 download_and_process.py --file video_urls.txt
  python3 download_and_process.py --file video_urls.txt "URL_4"
        """,
    )
    parser.add_argument(
        "urls",
        nargs="*",
        metavar="URL",
        help="One or more video URLs to download and process.",
    )
    parser.add_argument(
        "--file", "-f",
        metavar="PATH",
        default=None,
        help="Text file with one video URL per line (lines starting with # are ignored).",
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        metavar="DIR",
        help=f"Directory for downloaded videos (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help=f"Directory for results JSON/CSV (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--gt-dir",
        default=DEFAULT_GT_DIR,
        metavar="DIR",
        help=f"Directory for ground-truth scaffolds (default: {DEFAULT_GT_DIR}).",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        default=False,
        help="Re-download even if a cached local file already exists.",
    )

    args = parser.parse_args()

    video_urls = collect_urls(args.urls, args.file)
    if not video_urls:
        parser.print_help()
        print(
            "\nERROR: No URLs provided. Pass URLs as arguments or use --file.",
            file=sys.stderr,
        )
        sys.exit(1)

    input_dir  = args.input_dir
    output_dir = args.output_dir
    gt_dir     = args.gt_dir

    # Import the existing pipeline - zero modifications to any existing module
    from batch_processor import process_single_video

    overall_start = time.time()

    print_sep()
    log("Temporal Scene-Boundary Detection - Batch Evaluation")
    log(f"Input  directory : {os.path.abspath(input_dir)}")
    log(f"Output directory : {os.path.abspath(output_dir)}")
    log(f"GT     directory : {os.path.abspath(gt_dir)}")
    log(f"URLs to process  : {len(video_urls)}")
    print_sep()

    # Accumulators
    download_results:  List[Dict[str, Any]] = []
    process_results:   List[Dict[str, Any]] = []
    failed_downloads:  List[str] = []
    failed_processing: List[str] = []

    for idx, url in enumerate(video_urls, start=1):
        print_sep("-")
        log(f"[{idx}/{len(video_urls)}] {url}")

        # ----------------------------------------------------------------
        # STEP 1 - Download complete video
        # ----------------------------------------------------------------
        log("  Step 1/2: Downloading...")
        t_dl_start = time.time()
        local_path, source_dur, dl_dur = download_video(
            url,
            input_dir,
            force_redownload=args.force_redownload,
        )
        t_dl = time.time() - t_dl_start

        if local_path is None:
            log("  Download failed - skipping processing for this URL.")
            failed_downloads.append(url)
            download_results.append({
                "url":               url,
                "status":            "DOWNLOAD_FAILED",
                "source_duration_s": source_dur,
                "download_time_s":   round(t_dl, 1),
            })
            continue

        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        log(
            f"  Download complete in {t_dl:.1f}s - "
            f"{os.path.basename(local_path)} ({size_mb:.1f} MB)"
        )
        log(f"    Source  : {fmt_seconds(source_dur)}")
        log(f"    On-disk : {fmt_seconds(dl_dur)}")

        download_results.append({
            "url":                   url,
            "status":                "OK",
            "local_path":            local_path,
            "size_mb":               round(size_mb, 1),
            "source_duration_s":     source_dur,
            "downloaded_duration_s": dl_dur,
            "download_time_s":       round(t_dl, 1),
        })

        # ----------------------------------------------------------------
        # STEP 2 - Run the existing pipeline on the COMPLETE local file.
        # process_single_video() treats local files differently from URLs:
        # it does NOT apply the DEFAULT_URL_MAX_DURATION cap, so the
        # entire video is processed from start to finish.
        # ----------------------------------------------------------------
        log("  Step 2/2: Running existing pipeline on complete video...")
        t_proc_start = time.time()
        try:
            summary = process_single_video(
                video_path=local_path,
                output_dir=output_dir,
                # frame_skip, max_frames, max_duration -> config.py defaults
            )
            t_proc = time.time() - t_proc_start

            summary["source_url"]            = url
            summary["source_duration_s"]     = source_dur
            summary["downloaded_duration_s"] = dl_dur
            summary["processing_time_s"]     = round(t_proc, 1)
            summary["download_time_s"]       = round(t_dl, 1)

            # Count classified vs. unclassified shots
            n_shots      = summary.get("number_of_shots", 0)
            n_classified = (
                summary.get("anchor_shots", 0)
                + summary.get("field_report_shots", 0)
                + summary.get("b_roll_shots", 0)
                + summary.get("title_card_shots", 0)
            )
            n_other        = summary.get("other_shots", 0)
            n_unclassified = n_shots - n_classified - n_other
            summary["classified_shots"]   = n_classified
            summary["other_shots_count"]  = n_other
            summary["unclassified_shots"] = max(n_unclassified, 0)

            process_results.append(summary)

            # ---- STEP 3: Ground-truth scaffold and metadata extraction ----
            result_json_path = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(local_path))[0] + "_results.json",
            )
            result_csv_path = os.path.join(
                output_dir,
                os.path.splitext(os.path.basename(local_path))[0] + "_results.csv",
            )
            summary["output_json"] = result_json_path
            summary["output_csv"] = result_csv_path

            if os.path.exists(result_json_path):
                with open(result_json_path, encoding="utf-8") as fh:
                    result_data = json.load(fh)
                shots_list = result_data.get("shots", [])
                meta = result_data.get("video_metadata", {})
                tot_frames = meta.get("total_frames", 0)
                summary["total_frames"] = tot_frames

                import config
                f_skip = getattr(config, "FRAME_SKIP", 2)
                summary["frames_processed"] = (tot_frames + f_skip - 1) // f_skip if tot_frames else 0

                gt_file = write_unverified_ground_truth(
                    os.path.basename(local_path), shots_list, gt_dir
                )
                summary["ground_truth_scaffold"] = gt_file

                # Report last shot end vs. video end for completeness check
                if shots_list:
                    last_shot_end = shots_list[-1].get("end_time", 0)
                    summary["last_shot_end_s"] = last_shot_end
                    log(f"  Last shot ends at : {fmt_seconds(last_shot_end)}")
                    log(f"  Video ends at     : {fmt_seconds(dl_dur)}")

        except Exception as exc:
            t_proc = time.time() - t_proc_start
            log(f"  PROCESSING ERROR: {exc}")
            failed_processing.append(local_path)
            process_results.append({
                "video_name":        os.path.basename(local_path),
                "source_url":        url,
                "status":            f"PROCESSING_ERROR: {exc}",
                "processing_time_s": round(t_proc, 1),
            })

    # -----------------------------------------------------------------------
    # Final batch report
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - overall_start

    successful_downloads  = [r for r in download_results if r["status"] == "OK"]
    successful_processing = [r for r in process_results  if r.get("status") == "Success"]
    total_shots           = sum(r.get("number_of_shots", 0) for r in successful_processing)

    print_sep()
    log("BATCH EVALUATION COMPLETE")
    print_sep()

    print(f"\n{'=':=<70}")
    print(f"  SUMMARY")
    print(f"{'=':=<70}")
    print(f"  Total URLs provided       : {len(video_urls)}")
    print(f"  Successfully downloaded   : {len(successful_downloads)}")
    print(f"  Successfully processed    : {len(successful_processing)}")
    print(f"  Failed downloads          : {len(failed_downloads)}")
    print(f"  Failed processing         : {len(failed_processing)}")
    print(f"  Total shots detected      : {total_shots}")
    print(f"  Total wall-clock time     : {total_elapsed:.1f}s")
    print(f"{'=':=<70}\n")

    if successful_processing:
        # Comparative summary table (Point 16 format)
        print(f"  BATCH SUMMARY TABLE")
        print(f"{'=':=<105}")
        print(f"  {'Video':<35} {'Source Dur':<12} {'Downloaded':<12} {'Frames':<9} {'Shots':<7} {'Classified':<11} {'Unclass':<8} {'Last Shot':<10}")
        print(f"  {'-'*101}")
        for r in successful_processing:
            v_name = r.get('video_name', '')[:34]
            s_dur = fmt_seconds(r.get('source_duration_s'))
            d_dur = fmt_seconds(r.get('downloaded_duration_s'))
            f_proc = str(r.get('frames_processed', 'N/A'))
            n_shots = str(r.get('number_of_shots', 0))
            n_class = str(r.get('classified_shots', 0) + r.get('other_shots_count', 0))
            n_uncl = str(r.get('unclassified_shots', 0))
            l_shot = fmt_seconds(r.get('last_shot_end_s'))
            print(f"  {v_name:<35} {s_dur:<12} {d_dur:<12} {f_proc:<9} {n_shots:<7} {n_class:<11} {n_uncl:<8} {l_shot:<10}")
        print(f"{'=':=<105}\n")

        print(f"  PER-VIDEO DETAILS")
        print(f"{'=':=<70}")
        for r in successful_processing:
            n          = r.get("number_of_shots", 0)
            classified = r.get("classified_shots", 0)
            other      = r.get("other_shots_count", 0)
            unclassif  = r.get("unclassified_shots", 0)
            base       = os.path.splitext(r.get("video_name", "video"))[0]

            print(f"\n  Video             : {r.get('video_name')}")
            print(f"  Source            : {r.get('source_url')}")
            print(f"  Source duration   : {fmt_seconds(r.get('source_duration_s'))}")
            print(f"  Downloaded dur.   : {fmt_seconds(r.get('downloaded_duration_s'))}")
            print(f"  Processed dur.    : {fmt_seconds(r.get('total_duration'))}")
            print(f"  Frames processed  : {r.get('frames_processed', 'N/A')}")
            print(f"  Last shot ends at : {fmt_seconds(r.get('last_shot_end_s'))}")
            print(f"  Total shots       : {n}")
            print(f"  Classified shots  : {classified + other}  ({classified} primary + {other} Other)")
            print(f"  Unclassified      : {unclassif}")
            print(f"  Classification distribution:")
            print(f"    Anchor Studio : {r.get('anchor_shots', 0)}")
            print(f"    Field Report  : {r.get('field_report_shots', 0)}")
            print(f"    B-Roll        : {r.get('b_roll_shots', 0)}")
            print(f"    Title Card    : {r.get('title_card_shots', 0)}")
            print(f"    Other         : {other}")
            print(f"  Processing time   : {r.get('processing_time_s', 'N/A')}s")
            print(f"  Output files:")
            print(f"    JSON: {os.path.join(output_dir, base + '_results.json')}")
            print(f"    CSV : {os.path.join(output_dir, base + '_results.csv')}")
            print(f"    GT  : {os.path.join(gt_dir, base, base + '_ground_truth.json')}")
        print(f"{'=':=<70}")

    if failed_downloads:
        print(f"\n  FAILED DOWNLOADS")
        print(f"{'=':=<70}")
        for u in failed_downloads:
            print(f"  - {u}")

    if failed_processing:
        print(f"\n  FAILED PROCESSING")
        print(f"{'=':=<70}")
        for p in failed_processing:
            print(f"  - {p}")

    # Machine-readable batch report
    batch_report = {
        "evaluation_timestamp":  datetime.datetime.now().isoformat(),
        "total_urls_provided":   len(video_urls),
        "successful_downloads":  len(successful_downloads),
        "successful_processing": len(successful_processing),
        "failed_downloads":      len(failed_downloads),
        "failed_processing":     len(failed_processing),
        "total_shots_detected":  total_shots,
        "total_wall_clock_s":    round(total_elapsed, 1),
        "download_results":      download_results,
        "process_results":       process_results,
    }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "new_videos_batch_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(batch_report, fh, indent=2, default=str)

    log(f"Batch report saved to: {report_path}")
    print_sep()


if __name__ == "__main__":
    main()
