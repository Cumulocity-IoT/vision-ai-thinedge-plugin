"""
Raw Video Streaming with Detection Overlays for Vision AI Camera.

This module provides raw video streaming with AI detection overlays:
- Captures frames from the camera
- Draws bounding boxes, labels, keypoints, tracker IDs
- Encodes frames as H.264 via an ffmpeg subprocess (hardware-accelerated when available)
- Pushes an RTSP stream to go2rtc, which serves WebRTC/RTSP to clients

"""
from typing import Tuple
import subprocess
import time
import threading
import logging 
from typing import Callable, Optional, Any
import cv2
import numpy as np
from modlib.apps.annotate import Annotator, ColorPalette, Color
from modlib.models.results import Detections, Poses, Classifications


log = logging.getLogger(__name__)


class OverlayStreamServer:
    """
    Raw video streaming server with detection overlay support.

    Encodes BGR24 frames as H.264 via ffmpeg and pushes an RTSP stream to
    go2rtc. Uses h264_v4l2m2m (Raspberry Pi hardware encoder) whene available,
    falling back to libx264.
    """

    def __init__(
        self,
        frame_lock: threading.Lock,
        get_latest_frame: Callable[[], Optional[np.ndarray]],
        get_current_detections: Callable[[], Optional[Any]],
        labels: Optional[list],
        rtsp_url: str,
        encoder: str = "h264_v4l2m2m",
        draw_overlays: bool = True,
        size: Tuple = (640, 480),
        roi: Tuple = (0, 0, 1, 1)
    ):
        """
        Initialize raw video streaming server with overlay support.

        Args:
            frame_lock: Thread lock for accessing shared frame data
            get_latest_frame: Callable that returns the latest frame (numpy array)
            get_current_detections: Callable that returns detections object
            labels: List of label strings
            rtsp_url: RTSP URL to push the encoded stream to (e.g. go2rtc)
            encoder: ffmpeg video encoder to use (e.g. "h264_v4l2m2m" for Pi hardware, "libx264" for software)
            draw_overlays: Whether to draw detection overlays (default True)
            size: Output resolution; scaled down to fit within 1020x720
            roi: Region-of-interest as normalised (x1, y1, x2, y2)
        """
        max_width, max_height = 1020, 720
        if size[0] > max_width or size[1] > max_height:
            scale = min(max_width / size[0], max_height / size[1])
            self.width = int(size[0] * scale)
            self.height = int(size[1] * scale)
        else:
            self.width = size[0]
            self.height = size[1]
        self.fps = 15
        self.rtsp_url = rtsp_url
        self.encoder = encoder
        self.frame_lock = frame_lock
        self.get_latest_frame = get_latest_frame
        self.get_current_detections = get_current_detections
        self.labels = labels
        self.draw_overlays = draw_overlays
        self.ffmpeg_proc: Optional[subprocess.Popen] = None
        self.annotator: Optional[Any] = None
        self.roi = roi
        # Pre-allocated working buffer — reused every frame to avoid per-frame malloc
        self._frame_buf: Optional[np.ndarray] = None
        log.info(
            f"Raw video stream initialized: {self.width}x{self.height} @ {self.fps} fps, rtsp_url={rtsp_url}"
        )

    # ------------------------------------------------------------------
    # ffmpeg management
    # ------------------------------------------------------------------

    def _build_ffmpeg_cmd(self, encoder: str) -> list:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-c:v", encoder,
            "-b:v", "2000k",
        ]
        if encoder == "h264_v4l2m2m":
            # Hardware encoder only accepts YUV natively; force explicit conversion
            # so ffmpeg doesn't pick a wrong intermediate format automatically.
            # h264_mp4toannexb converts AVCC (length-prefixed) → Annex-B (start codes)
            # which dump_extra requires to inject SPS/PPS before every IDR frame.
            cmd += ["-vf", "format=yuv420p", "-bsf:v", "dump_extra"]
        elif encoder == "libx264":
            cmd += ["-vf", "format=yuv420p", "-preset", "ultrafast", "-tune", "zerolatency", "-bsf:v", "dump_extra"]
        # Keyframe every 2 seconds — clients need a keyframe to start decoding.
        cmd += ["-g", str(self.fps * 2)]
        cmd += ["-f", "rtsp", "-rtsp_transport", "tcp", self.rtsp_url]
        return cmd

    @staticmethod
    def _drain_stderr(proc: subprocess.Popen) -> None:
        """Read ffmpeg stderr line-by-line so the pipe never fills and blocks ffmpeg."""
        try:
            for raw in proc.stderr:  # type: ignore[union-attr]
                line = raw.decode(errors="replace").rstrip()
                if line:
                    log.debug("ffmpeg: %s", line)
        except Exception:
            pass

    def _start_ffmpeg(self, encoder: str) -> None:
        cmd = self._build_ffmpeg_cmd(encoder)
        log.info(f"Starting ffmpeg ({encoder}): {' '.join(cmd)}")
        self.ffmpeg_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        threading.Thread(
            target=self._drain_stderr,
            args=(self.ffmpeg_proc,),
            daemon=True,
            name="ffmpeg-stderr-drain",
        ).start()

    def _restart_ffmpeg(self, encoder: str, retry_delay: float = 2.0) -> None:
        if self.ffmpeg_proc is not None:
            try:
                self.ffmpeg_proc.stdin.close()
            except Exception:
                pass
            try:
                self.ffmpeg_proc.terminate()
            except Exception:
                pass
            try:
                self.ffmpeg_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self.ffmpeg_proc.kill()
                    self.ffmpeg_proc.wait()
                except Exception:
                    pass
            self.ffmpeg_proc = None
        log.info(f"Restarting ffmpeg in {retry_delay:.0f}s...")
        time.sleep(retry_delay)
        self._start_ffmpeg(encoder)

    # ------------------------------------------------------------------
    # Overlay annotation
    # ------------------------------------------------------------------

    def annotate_frame(
        self, frame: np.ndarray, detections: Detections|Poses|Classifications, labels: Optional[list], annotator: Any
    ) -> np.ndarray:
        """
        Draw detection boxes, labels, keypoints on frame.

        Args:
            frame: NumPy array (BGR format for OpenCV)
            detections: Detections/Poses/Classifications object from modlib
            labels: List of label strings
            annotator: Annotator instance for drawing

        Returns:
            Annotated frame (modified in-place)
        """
        if detections is None or not hasattr(detections, '__len__') or len(detections) == 0:
            return frame

        h, w, _ = frame.shape

        # Handle Poses - draw keypoints first
        if isinstance(detections, Poses):
            try:
                annotator.annotate_keypoints(frame, detections)
            except Exception as e:
                log.debug(f"Could not draw keypoints: {e}")
        if isinstance(detections, Classifications):
            return frame
        if self.roi != (0,0,1,1):
            detections.compensate_for_roi(self.roi)
        # Draw bounding boxes and labels
        for i in range(len(detections)):
            try:
                bbox = detections.bbox[i]
                x1, y1, x2, y2 = bbox

                # Rescale to frame size (bboxes are normalized 0-1)
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

                # Get class info
                if isinstance(detections, Detections):
                    class_id = (
                        int(detections.class_id[i])
                        if detections.class_id is not None
                        else None
                    )
                else:  # Poses
                    class_id = "Person"

                # Get tracker ID for color selection
                tracker_id = (
                    detections.tracker_id[i]
                    if hasattr(detections, "tracker_id")
                    and detections.tracker_id is not None
                    else None
                )
                idx = tracker_id if tracker_id is not None and tracker_id > 0 else i

                # Get color from annotator
                if isinstance(annotator.color, ColorPalette):
                    color = annotator.color.by_idx(idx)
                else:
                    color = annotator.color

                # Draw bounding box
                cv2.rectangle(
                    img=frame,
                    pt1=(x1, y1),
                    pt2=(x2, y2),
                    color=color.as_bgr(),
                    thickness=annotator.thickness,
                )

                # Prepare label text
                if labels is None or len(labels) == 0:
                    label_text = str(class_id)
                elif isinstance(class_id, int) and 0 <= class_id < len(labels):
                    label_text = labels[class_id]
                else:
                    label_text = str(class_id)

                # Add tracker ID if available
                if tracker_id is not None and tracker_id > 0:
                    label = f"{label_text} {tracker_id}"
                else:
                    label = label_text

                # Draw label
                annotator.set_label(
                    image=frame, x=x1, y=y1 - 20, color=color.as_bgr(), label=label
                )
            except Exception as e:
                log.debug(f"Error annotating detection {i}: {e}")
                continue

        return frame

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def serve_forever(self):
        """Start the raw video streaming server with overlays."""
        try:
            if self.draw_overlays:
                self.annotator = Annotator(
                    color=ColorPalette.default(),
                    thickness=2,
                    text_thickness=2,
                    text_scale=0.8,
                )
                log.info("Annotator created for detection overlay")

            self._start_ffmpeg(self.encoder)
            self._frame_buf = np.empty((self.height, self.width, 3), dtype=np.uint8)

            log.info(f"Raw video streaming started → {self.rtsp_url}")
            log.info(f"Stream settings: {self.width}x{self.height} @ {self.fps}fps (encoder: {self.encoder})")

            frame_interval = 1.0 / self.fps
            last_frame_time = time.time()
            frame_count = 0
            dropped_frames = 0
            last_frame_id = None

            while True:
                current_time = time.time()
                elapsed = current_time - last_frame_time

                # Frame rate limiting
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                    continue

                # Check if we're falling behind
                if elapsed > frame_interval * 2:
                    dropped_frames += 1
                    if dropped_frames % 10 == 0:
                        log.warning(f"Falling behind, dropped {dropped_frames} frames")

                last_frame_time = current_time

                # Restart ffmpeg if it crashed
                if self.ffmpeg_proc is None or self.ffmpeg_proc.poll() is not None:
                    log.warning("ffmpeg process died, restarting")
                    self._restart_ffmpeg(self.encoder)

                # Read shared state without locking.
                # Python reference reads are atomic under the GIL, so there is no
                # risk of a partial/corrupt read. The detection thread holds
                # frame_lock for 10 ms per frame (sleep inside the lock); acquiring
                # it here would starve that loop and stall the camera pipeline.
                latest_frame = self.get_latest_frame()
                detections = self.get_current_detections()
                labels = self.labels

                # Check if frame is available
                if latest_frame is None or not isinstance(latest_frame, np.ndarray):
                    # No frame available yet, send blank frame
                    blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    cv2.putText(
                        blank,
                        "Waiting for camera...",
                        (50, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )
                    latest_frame = blank
                else:
                    frame_id = id(latest_frame)
                    if frame_id == last_frame_id:
                        if frame_count > 0 and frame_count % 30 == 0:
                            log.debug("Reusing same frame (camera slower than stream FPS)")
                    last_frame_id = frame_id

                # Copy into the pre-allocated buffer to avoid a per-frame malloc.
                # cv2.resize with dst= and np.copyto both write into the existing array.
                if latest_frame.shape[0] != self.height or latest_frame.shape[1] != self.width:
                    cv2.resize(latest_frame, (self.width, self.height), dst=self._frame_buf)
                else:
                    np.copyto(self._frame_buf, latest_frame)

                frame_bgr = self._frame_buf

                if self.draw_overlays and self.roi != (0, 0, 1, 1):
                    pt1=(int(self.roi[0] * self.width),int(self.roi[1] * self.height))
                    pt2=(int((self.roi[0] + self.roi[2]) * self.width), int((self.roi[1] + self.roi[3]) * self.height))
                    cv2.rectangle(
                        img=frame_bgr,
                        pt1=pt1,
                        pt2=pt2,
                        color=Color.red().as_bgr(),
                        thickness=2,
                    )

                # Draw detection overlays if enabled
                if self.draw_overlays and detections is not None and self.annotator is not None:
                    try:
                        frame_bgr = self.annotate_frame(
                            frame_bgr, detections, labels, self.annotator
                        )
                    except Exception as e:
                        log.error(f"Error drawing detections: {e}", exc_info=True)

                # Write raw frame to ffmpeg stdin
                try:
                    assert self.ffmpeg_proc is not None and self.ffmpeg_proc.stdin is not None
                    self.ffmpeg_proc.stdin.write(self._frame_buf.data)
                    self.ffmpeg_proc.stdin.flush()
                    frame_count += 1

                    if frame_count % 100 == 0:
                        log.info(f"Streamed {frame_count} frames (dropped: {dropped_frames})")
                except BrokenPipeError:
                    log.warning("ffmpeg stdin broken (go2rtc disconnected?), restarting ffmpeg")
                    self._restart_ffmpeg(self.encoder)
                except OSError as e:
                    log.error(f"I/O error writing to ffmpeg: {e}")
                    break
                except Exception as e:
                    log.error(f"Error writing frame to ffmpeg: {e}")
                    break

        except Exception as e:
            log.error(f"Raw video streaming error: {e}", exc_info=True)
            raise
        finally:
            self.shutdown()

    def shutdown(self):
        """Stop the raw video streaming server gracefully."""
        log.info("Shutting down raw video stream...")

        if self.ffmpeg_proc is not None:
            try:
                self.ffmpeg_proc.stdin.close()
            except Exception:
                pass
            try:
                self.ffmpeg_proc.terminate()
                self.ffmpeg_proc.wait(timeout=5)
                log.info("ffmpeg process terminated")
            except Exception:
                try:
                    self.ffmpeg_proc.kill()
                    log.info("ffmpeg process killed")
                except Exception as e:
                    log.warning(f"Could not stop ffmpeg: {e}")

        log.info("Raw video stream stopped")