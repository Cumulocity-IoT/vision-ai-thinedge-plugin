"""
Raw Video Streaming with Detection Overlays for Vision AI Camera.

This module provides raw video streaming with AI detection overlays:
- Captures frames from the camera
- Draws bounding boxes, labels, keypoints, tracker IDs
- Writes raw BGR frames to a named pipe
- External ffmpeg process (via go2rtc) reads and encodes the stream

"""

import time
import threading
import logging
import os
from typing import Callable, Optional, Any
import stat
import cv2
import numpy as np
from modlib.apps.annotate import Annotator, ColorPalette
from modlib.models.results import Detections, Poses, Classifications


log = logging.getLogger(__name__)


class OverlayStreamServer:
    """
    Raw video streaming server with detection overlay support.

    Writes raw BGR24 frames to a named pipe for external encoding.
    Designed to be consumed by ffmpeg via go2rtc.
    """

    def __init__(
        self,
        frame_lock: threading.Lock,
        get_latest_frame: Callable[[], Optional[np.ndarray]],
        get_current_detections: Callable[[], Optional[Any]],
        labels: Optional[list],
        output_path: str,
        draw_overlays: bool = True,
    ):
        """
        Initialize raw video streaming server with overlay support.

        Args:

            frame_lock: Thread lock for accessing shared frame data
            get_latest_frame: Callable that returns the latest frame (numpy array)
            get_current_detections: Callable that returns detections object
            get_labels: Callable that returns labels list
            draw_overlays: Whether to draw detection overlays (default True)
        """
        self.width = 640
        self.height = 480
        self.fps = 15
        self.output_path = output_path
        self.frame_lock = frame_lock
        self.get_latest_frame = get_latest_frame
        self.get_current_detections = get_current_detections
        self.labels = labels
        self.draw_overlays = draw_overlays
        self.pipe_fd = None
        self.annotator: Optional[Any] = None

        log.info(
            f"Raw video stream initialized: {self.width}x{self.height} @ {self.fps} fps, output={output_path}"
        )

    def annotate_frame(
        self, frame: np.ndarray, detections: Any, labels: Optional[list], annotator: Any
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

    def _create_named_pipe(self):
        """Create named pipe for raw video output if it doesn't exist."""

        # Check if pipe already exists and is valid
        if os.path.exists(self.output_path):
            # Check if it's actually a named pipe
            if stat.S_ISFIFO(os.stat(self.output_path).st_mode):
                log.info(f"Named pipe already exists: {self.output_path}")
                return
            else:
                # It exists but is not a pipe, remove it
                try:
                    os.remove(self.output_path)
                    log.warning(f"Removed non-pipe file at: {self.output_path}")
                except Exception as e:
                    log.error(f"Could not remove existing file: {e}")
                    raise

        # Create the pipe
        try:
            os.mkfifo(self.output_path)
            log.info(f"Created named pipe: {self.output_path}")
        except FileExistsError:
            # Race condition - pipe was created between check and creation
            log.info(f"Named pipe already exists: {self.output_path}")
        except Exception as e:
            log.error(f"Failed to create named pipe: {e}")
            raise

    def _open_pipe(self):
        """
        Open the named pipe for writing.

        This will block until a reader (e.g., ffmpeg via go2rtc) opens the pipe.
        """
        log.info(f"Opening named pipe for writing: {self.output_path}")
        log.info("Note: This will block until a reader connects to the pipe")

        try:
            # Open in write-binary mode, unbuffered
            self.pipe_fd = open(self.output_path, 'wb', buffering=0)
            log.info("Named pipe opened successfully")
        except Exception as e:
            log.error(f"Failed to open named pipe: {e}")
            raise

    def serve_forever(self):
        """Start the raw video streaming server with overlays."""
        try:
            # Create named pipe
            self._create_named_pipe()

            # Create annotator for overlays
            if self.draw_overlays:
                    self.annotator = Annotator(
                        color=ColorPalette.default(),
                        thickness=2,
                        text_thickness=2,
                        text_scale=0.8,
                    )
                    log.info("Annotator created for detection overlay")

            # Open pipe (will block until reader connects)
            self._open_pipe()

            log.info(f"Raw video streaming started: {self.output_path}")
            log.info(f"Stream settings: {self.width}x{self.height} @ {self.fps}fps (format: raw BGR24)")
            log.info(f"Reader should use: ffmpeg -f rawvideo -pix_fmt bgr24 -s {self.width}x{self.height} -r {self.fps} -i {self.output_path}")

            frame_interval = 1.0 / self.fps
            last_frame_time = time.time()
            frame_count = 0
            dropped_frames = 0
            last_frame_id = None  # Track if we're getting new frames

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

                # Get frame, detections, labels (thread-safe)
                with self.frame_lock:
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
                    # Check if we're getting new frames
                    frame_id = id(latest_frame)
                    if frame_id == last_frame_id:
                        # Same frame as before - camera might be slow
                        if frame_count > 0 and frame_count % 30 == 0:
                            log.debug("Reusing same frame (camera slower than stream FPS)")
                    last_frame_id = frame_id

                # Make a copy to avoid modifying the original
                frame_copy = latest_frame.copy()

                # Resize if needed
                if frame_copy.shape[0] != self.height or frame_copy.shape[1] != self.width:
                    frame_copy = cv2.resize(frame_copy, (self.width, self.height))

                # Convert RGB to BGR if needed (OpenCV expects BGR)
                if len(frame_copy.shape) == 3 and frame_copy.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame_copy

                # Draw detection overlays if enabled
                if self.draw_overlays and detections is not None and self.annotator is not None:
                    try:
                        frame_bgr = self.annotate_frame(
                            frame_bgr, detections, labels, self.annotator
                        )
                    except Exception as e:
                        log.error(f"Error drawing detections: {e}", exc_info=True)

                # Write raw frame to pipe
                try:
                    if self.pipe_fd:
                        # Write frame data directly to pipe
                        frame_bytes = frame_bgr.tobytes()
                        self.pipe_fd.write(frame_bytes)
                        # No need to flush - buffering=0 means unbuffered
                        frame_count += 1

                        if frame_count % 100 == 0:
                            log.info(f"Streamed {frame_count} frames (dropped: {dropped_frames})")
                except BrokenPipeError:
                    log.warning("Pipe broken (reader disconnected), restarting stream")
                    if self.pipe_fd:
                        try:
                            self.pipe_fd.close()
                        except Exception as e:
                            log.warning(f"Error closing pipe: {e}")
                        self.pipe_fd = None
                        self._open_pipe()
                except IOError as e:
                    log.error(f"I/O error writing to pipe: {e}")
                    break
                except Exception as e:
                    log.error(f"Error writing frame to pipe: {e}")
                    break

        except Exception as e:
            log.error(f"Raw video streaming error: {e}", exc_info=True)
            raise
        finally:
            self.shutdown()

    def shutdown(self):
        """Stop the raw video streaming server gracefully."""
        log.info("Shutting down raw video stream...")

        # Close pipe file descriptor
        if self.pipe_fd:
            try:
                self.pipe_fd.close()
                log.info("Named pipe closed")
            except Exception as e:
                log.warning(f"Error closing pipe: {e}")

        # Don't remove the pipe - it may be reused by go2rtc or other processes
        # The pipe file will persist until explicitly removed
        log.info("Raw video stream stopped")
