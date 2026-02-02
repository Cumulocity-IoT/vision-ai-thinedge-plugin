"""
MJPEG HTTP Server for streaming video with AI detection overlays.

This module provides an MJPEG streaming server that:
- Serves video frames via HTTP with multipart/x-mixed-replace
- Draws AI detection overlays (bounding boxes, labels, confidence scores)
- Integrates with go2rtc for WebRTC streaming
- Runs in a separate thread for non-blocking operation

This handles all visualization logic that was previously in rpi_vision_ai_processor.py,
providing remote access via WebRTC instead of local OpenCV windows.
"""

import time
import threading
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable, Optional, Any
import cv2
import numpy as np

log = logging.getLogger(__name__)


class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """HTTP request handler for MJPEG streaming."""

    def log_message(self, msg_format, *args):
        """Override to use Python logging instead of stderr."""
        log.info("%s - %s", self.address_string(), msg_format % args)

    def do_GET(self):
        """Handle GET requests for MJPEG stream."""
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()

            try:
                for frame_data in self.server.generate_frames():  # type: ignore
                    self.wfile.write(frame_data)
            except (ConnectionResetError, BrokenPipeError):
                log.info("Client disconnected from stream")
        elif self.path == "/":
            # Serve simple info page
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            html = b"""
            <!DOCTYPE html>
            <html>
            <head><title>Vision AI MJPEG Stream</title></head>
            <body style="margin:0; padding:20px; background:#000; color:#fff; font-family:monospace;">
                <h1>Vision AI Camera Stream</h1>
                <p>This MJPEG stream is consumed by go2rtc for WebRTC conversion.</p>
                <p>Stream endpoint: <a href="/stream" style="color:#0f0;">/stream</a></p>
                <p>View via Cumulocity Device Management &gt; Webcam tab</p>
            </body>
            </html>
            """
            self.wfile.write(html)
        else:
            self.send_error(404)


class MJPEGServer:
    """
    MJPEG HTTP server that streams frames with AI detection overlays.

    This server replaces the show_preview functionality, providing the same
    visualization but accessible remotely via WebRTC (through go2rtc).
    """

    def __init__(
        self,
        port: int,
        quality: int,
        fps: int,
        frame_lock: threading.Lock,
        get_latest_frame: Callable[[], Optional[np.ndarray]],
        get_current_detections: Callable[[], Optional[Any]],
        get_labels: Callable[[], Optional[list]],
    ):
        """
        Initialize MJPEG server.

        Args:
            port: HTTP port to listen on
            quality: JPEG quality (1-100)
            fps: Target frames per second
            frame_lock: Thread lock for accessing shared frame data
            get_latest_frame: Callable that returns the latest frame (numpy array)
            get_current_detections: Callable that returns detections object
            get_labels: Callable that returns labels list
        """
        self.port = port
        self.quality = quality
        self.fps = fps
        self.frame_lock = frame_lock
        self.get_latest_frame = get_latest_frame
        self.get_current_detections = get_current_detections
        self.get_labels = get_labels
        self.server: Optional[HTTPServer] = None
        self._running = False
        self.annotator: Optional[Any] = None

        log.info(
            "MJPEG server initialized: port=%d, quality=%d, fps=%d",
            port,
            quality,
            fps,
        )

    def annotate_frame(
        self, frame: np.ndarray, detections: Any, labels: Optional[list], annotator: Any
    ) -> np.ndarray:
        """
        Draw detection boxes, labels, keypoints on frame.

        Works directly with numpy arrays for efficient streaming.

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

        # Import types locally to avoid import errors if modlib not available
        try:
            from modlib.models.results import Detections, Poses
            from modlib.apps.annotate import ColorPalette
        except ImportError:
            log.warning("modlib not available, skipping annotations")
            return frame

        h, w, _ = frame.shape

        # Handle Poses - draw keypoints first
        if isinstance(detections, Poses):
            try:
                # Use annotator's keypoint drawing if available
                # This creates a mock Frame-like object for compatibility
                class MockFrame:
                    """Mock Frame object for annotator compatibility."""
                    def __init__(self, img):
                        self.image = img

                mock_frame = MockFrame(frame)
                annotator.annotate_keypoints(mock_frame, detections)
            except Exception as e:
                log.debug("Could not draw keypoints: %s", e)

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
                        detections.class_id[i]
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
                log.debug("Error annotating detection %d: %s", i, e)
                continue

        return frame

    def generate_frames(self):
        """
        Generate MJPEG frames with detection overlays.

        Yields:
            MJPEG frame data (multipart format)
        """
        frame_interval = 1.0 / self.fps
        last_frame_time = 0

        # Create annotator instance for drawing (reused across frames)
        try:
            from modlib.apps.annotate import Annotator, ColorPalette
            self.annotator = Annotator(
                color=ColorPalette.default(), thickness=2, text_thickness=2, text_scale=0.8
            )
            log.info("Annotator created for detection overlay")
        except ImportError:
            log.warning("modlib not available, serving frames without overlays")
            self.annotator = None

        log.info("Starting MJPEG stream generation at %d FPS", self.fps)

        while self._running:
            current_time = time.time()

            # Frame rate limiting
            if current_time - last_frame_time < frame_interval:
                time.sleep(frame_interval - (current_time - last_frame_time))
                continue

            last_frame_time = current_time

            # Get frame, detections, labels (thread-safe)
            with self.frame_lock:
                latest_frame = self.get_latest_frame()
                detections = self.get_current_detections()
                labels = self.get_labels()

            # Check if frame is available
            if latest_frame is None or not isinstance(latest_frame, np.ndarray):
                # No frame available yet, send blank frame
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    blank,
                    "Waiting for camera...",
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                latest_frame = blank

            # Make a copy to avoid modifying the original
            frame_copy = latest_frame.copy()

            # Convert RGB to BGR if needed (OpenCV expects BGR)
            if frame_copy.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame_copy

            # Draw detection overlays if available
            if detections is not None and self.annotator is not None:
                try:
                    frame_bgr = self.annotate_frame(
                        frame_bgr, detections, labels, self.annotator
                    )
                except Exception as e:
                    log.error("Error drawing detections: %s", e, exc_info=True)

            # Encode frame as JPEG
            try:
                success, jpeg = cv2.imencode(
                    ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
                )
                if not success:
                    log.error("Failed to encode frame as JPEG")
                    continue

                # Yield multipart frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                )
            except Exception as e:
                log.error("Error encoding/sending frame: %s", e)
                time.sleep(0.1)

    def serve_forever(self):
        """Start the HTTP server and serve requests."""
        try:
            self.server = HTTPServer(("127.0.0.1", self.port), MJPEGStreamHandler)
            self.server.generate_frames = self.generate_frames  # type: ignore
            self._running = True
            log.info("MJPEG server started on http://127.0.0.1:%d/stream", self.port)
            self.server.serve_forever()
        except OSError as e:
            if e.errno == 98:  # Address already in use
                log.error("Port %d already in use", self.port)
                raise
            log.error("Failed to start MJPEG server: %s", e)
            raise
        except Exception as e:
            log.error("MJPEG server error: %s", e, exc_info=True)
            raise

    def shutdown(self):
        """Stop the HTTP server gracefully."""
        log.info("Shutting down MJPEG server...")
        self._running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            log.info("MJPEG server stopped")
