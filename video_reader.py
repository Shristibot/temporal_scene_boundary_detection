import os
import sys
import time
from typing import Generator, Dict, Any, Tuple, Optional
import cv2
import numpy as np


def is_url(path: str) -> bool:
    """Check if the provided path string is an HTTP/HTTPS/RTSP web URL."""
    if not path:
        return False
    clean_path = path.strip().lower()
    return clean_path.startswith(("http://", "https://", "rtsp://", "rtmp://"))


class VideoReader:
    def __init__(self, video_path: str) -> None:
        """Initialize video reader with local file path or web URL."""
        self.video_path: str = video_path.strip()
        self.is_remote_url: bool = is_url(self.video_path)
        self.video_capture: Optional[cv2.VideoCapture] = None
        self.resolved_stream_url: Optional[str] = None
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.frame_rate: float = 0.0
        self.total_frames: int = 0
        self.video_duration: float = 0.0

    def open(self) -> bool:
        """
        Open video stream from local file or web URL (via yt-dlp).

        Returns:
            bool: True if video opened successfully, False otherwise.
        """
        target_source = self.video_path

        if self.is_remote_url:
            print(f"[Online Link] Fetching stream URL via yt-dlp for: {self.video_path}")
            try:
                import yt_dlp
                ydl_opts = {
                    'format': 'best[ext=mp4]/best',
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': False,
                    'extractor_args': {
                        'youtube': {
                            'player_client': ['mweb', 'android', 'ios', 'web']
                        }
                    }
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.video_path, download=False)
                    if not info:
                        print(f"Error: yt-dlp returned no information for URL: {self.video_path}")
                        return False
                    
                    stream_url = info.get('url')
                    if not stream_url and 'formats' in info and len(info['formats']) > 0:
                        stream_url = info['formats'][-1].get('url')

                    if not stream_url:
                        print(f"Error: Could not extract playable video stream URL from: {self.video_path}")
                        return False

                    self.resolved_stream_url = stream_url
                    target_source = stream_url
                    
                    # Pre-fill duration from metadata if available
                    duration_sec = info.get('duration')
                    if duration_sec and duration_sec > 0:
                        self.video_duration = float(duration_sec)

                    title = info.get('title', 'Online Video')
                    print(f"[Online Link Success] Stream resolved: '{title}' ({self.video_duration:.1f}s)")

            except ImportError:
                print("Error: 'yt-dlp' library is required to process online URLs. Install via: pip install yt-dlp")
                return False
            except Exception as error:
                print(f"[URL Warning] yt-dlp extraction warning for '{self.video_path}': {error}")
                # Fallback to direct OpenCV capture for direct MP4/M3U8 links
                if self.video_path.lower().endswith((".mp4", ".m3u8", ".mov", ".avi", ".ts", ".webm")):
                    print(f"[Online Link] Attempting direct OpenCV VideoCapture stream fallback for: {self.video_path}")
                    target_source = self.video_path
                else:
                    print(f"[URL Error] Could not open URL stream: {error}")
                    return False

        else:
            if not os.path.exists(self.video_path):
                print(f"Error: Local video file does not exist: {self.video_path}")
                return False

        try:
            if self.is_remote_url:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;60000000|reconnect;1|reconnect_streamed;1|reconnect_delay_max;5"
            self.video_capture = cv2.VideoCapture(target_source)
            if not self.video_capture.isOpened():
                print(f"Error: OpenCV could not open video source: {self.video_path}")
                return False

            self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_rate = float(self.video_capture.get(cv2.CAP_PROP_FPS))
            self.total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            # Fallback if FPS is invalid or zero
            if self.frame_rate <= 0:
                print(f"WARNING: Invalid or missing FPS ({self.frame_rate}). Defaulting to 25.0 FPS.")
                self.frame_rate = 25.0

            if self.total_frames > 0:
                calc_duration = self.total_frames / self.frame_rate
                if self.video_duration == 0.0 or calc_duration > self.video_duration:
                    self.video_duration = calc_duration
            elif self.video_duration > 0.0:
                self.total_frames = int(self.video_duration * self.frame_rate)
            else:
                print(f"WARNING: Unknown total frame count and duration for video stream.")

            if self.frame_width <= 0 or self.frame_height <= 0:
                print(f"Error: Invalid frame dimensions ({self.frame_width}x{self.frame_height})")
                return False

            return True

        except Exception as error:
            print(f"Exception encountered while opening video {self.video_path}: {error}")
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Return video metadata dictionary."""
        name = os.path.basename(self.video_path) if not self.is_remote_url else "online_video.mp4"
        return {
            "video_name": name,
            "video_path": self.video_path,
            "is_url": self.is_remote_url,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "frame_rate": round(self.frame_rate, 2),
            "total_frames": self.total_frames,
            "video_duration": round(self.video_duration, 2)
        }

    def read_sampled_frames(
        self,
        frame_skip: int = 1,
        downscale_max_dim: Optional[int] = 480,
        max_frames: Optional[int] = None,
        max_duration: Optional[float] = None,
        keep_raw_frames: bool = False,
        log_progress: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream sampled video frames from capture source.

        Args:
            frame_skip: Process 1 frame every N frames
            downscale_max_dim: Max frame height/width for memory efficiency (None to keep original)
            max_frames: Stop generator after yielding max_frames sampled frames
            max_duration: Stop generator after reaching max_duration seconds timestamp
            keep_raw_frames: Keep raw numpy array frames in dict (default False for RAM safety)
            log_progress: Print periodic progress logs for long jobs

        Yields:
            Dict containing frame_number, timestamp, hist, features, motion_diff, (and optional frame)
        """
        if self.video_capture is None or not self.video_capture.isOpened():
            print("Error: Attempted to read from an uninitialized or closed video capture.")
            return

        from feature_extractor import extract_frame_features

        current_frame_index: int = 0
        yielded_count: int = 0
        consecutive_failures: int = 0
        max_consecutive_failures: int = 50
        start_time = time.time()
        last_log_time = start_time
        prev_gray: Optional[np.ndarray] = None

        while True:
            success, frame = self.video_capture.read()
            if not success:
                consecutive_failures += 1
                if self.total_frames > 0 and current_frame_index < self.total_frames and consecutive_failures < max_consecutive_failures:
                    current_frame_index += 1
                    continue
                else:
                    break

            consecutive_failures = 0

            if current_frame_index % frame_skip == 0:
                timestamp: float = current_frame_index / self.frame_rate if self.frame_rate > 0 else 0.0

                if max_duration is not None and max_duration > 0 and timestamp > max_duration:
                    print(f"[Streaming Cap] Reached max duration limit of {max_duration:.1f}s.")
                    break

                # Memory Safety Downscaling: Resize frame if downscale_max_dim is set
                processed_frame = frame
                if downscale_max_dim is not None:
                    h, w = frame.shape[:2]
                    max_dim = max(h, w)
                    if max_dim > downscale_max_dim:
                        scale = downscale_max_dim / float(max_dim)
                        new_w = max(1, int(w * scale))
                        new_h = max(1, int(h * scale))
                        processed_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Pre-calculate features and HSV histogram on-the-fly
                frame_feats = extract_frame_features(processed_frame)
                
                hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                motion_diff = 0.0
                if prev_gray is not None:
                    motion_diff = float(np.mean(cv2.absdiff(prev_gray, gray)) / 255.0)
                prev_gray = gray

                # Periodic Progress Logging for long videos
                now = time.time()
                if log_progress and (now - last_log_time >= 10.0 or (self.total_frames > 0 and current_frame_index == self.total_frames - 1)):
                    last_log_time = now
                    elapsed = now - start_time
                    fps_speed = (current_frame_index + 1) / max(0.1, elapsed)

                    curr_mins = int(timestamp // 60)
                    curr_secs = int(timestamp % 60)
                    curr_hrs = curr_mins // 60
                    curr_mins = curr_mins % 60

                    time_str = f"{curr_hrs:02d}:{curr_mins:02d}:{curr_secs:02d}"

                    if self.video_duration > 0:
                        tot_mins = int(self.video_duration // 60)
                        tot_secs = int(self.video_duration % 60)
                        tot_hrs = tot_mins // 60
                        tot_mins = tot_mins % 60
                        tot_str = f"{tot_hrs:02d}:{tot_mins:02d}:{tot_secs:02d}"

                        pct = min(100.0, (timestamp / self.video_duration) * 100.0)
                        print(f"[Streaming Progress] Time: {time_str} / {tot_str} ({pct:.1f}%) | Frame {current_frame_index} | Speed: {fps_speed:.1f} FPS", flush=True)
                    else:
                        print(f"[Streaming Progress] Time: {time_str} | Frame {current_frame_index} | Speed: {fps_speed:.1f} FPS", flush=True)

                item = {
                    "frame_number": current_frame_index,
                    "timestamp": round(timestamp, 3),
                    "hist": hist,
                    "features": frame_feats,
                    "motion_diff": motion_diff
                }
                if keep_raw_frames:
                    item["frame"] = processed_frame

                yield item

                yielded_count += 1
                if max_frames is not None and max_frames > 0 and yielded_count >= max_frames:
                    print(f"[Streaming Cap] Reached max frames limit of {max_frames} sampled frames.")
                    break

            current_frame_index += 1

    def close(self) -> None:
        """Release video capture resources."""
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
