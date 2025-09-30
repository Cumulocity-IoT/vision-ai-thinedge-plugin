import argparse
import json
import logging.config
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
import paho.mqtt.client as mqtt
import yaml
import cv2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

sys.path.append("modlib")
from modlib.apps.tracker.byte_tracker import BYTETracker
from modlib.apps.annotate import ColorPalette, Annotator
from modlib.devices import AiCamera

from modlib.models.model import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.devices.frame import Frame, IMAGE_TYPE
from modlib.models.results import Classifications, Detections, Poses, Segments
from modlib.models.post_processors import (
    pp_od_bscn,
    pp_od_bcsn,
    pp_cls,
    pp_cls_softmax,
    pp_posenet,
    pp_yolo_pose_ultralytics,
)

# Configuration
DAEMON_SERVICE = (
    "rpi-vision-ai-processor.service"  # Change this to your daemon service name
)
# Default paths - can be overridden via environment variables or parameters
DEFAULT_LOG_DIR = "/var/log/tedge/vai-plugin"
DEFAULT_LOGGING_CONFIG = (
    "/etc/tedge/plugins/vai-plugin/rpi_vision_ai_processor_logging.conf"
)
DEFAULT_PLUGIN_CONFIG = "/etc/tedge/plugins/vai-plugin/plugin_config.yaml"
DEFAULT_CAMERA_CONFIG = "/etc/tedge/plugins/vai-plugin/camera_config.yaml"
DEFAULT_MODEL_BASE_PATH = "/etc/tedge/plugins/vai-plugin/model"

# Get configurable paths from environment variables or use defaults
LOG_DIR = os.getenv("VAI_LOG_DIR", DEFAULT_LOG_DIR)
LOGGING_CONFIG_FILE = os.getenv("VAI_PROCESSOR_LOGGING_CONFIG", DEFAULT_LOGGING_CONFIG)
PLUGIN_CONFIG_FILE = os.getenv("VAI_PLUGIN_CONFIG", DEFAULT_PLUGIN_CONFIG)
CAMERA_CONFIG_FILE = os.getenv("VAI_CAMERA_CONFIG", DEFAULT_CAMERA_CONFIG)
MODEL_BASE_PATH = os.getenv("VAI_MODEL_BASE_PATH", DEFAULT_MODEL_BASE_PATH)

DEFAULT_DURATION = 10
DEAFULT_FRAMERATE = 15


log = logging.getLogger(__name__)
np.set_printoptions(threshold=sys.maxsize)

mqtt_client: mqtt.Client

latest_frame = None
frame_lock = threading.Lock()
frame_counter: int = 0
show_preview: bool = False
is_fullscreen: bool = False


# Global variable to control recording
is_recording: bool = False
recording_thread: threading.Thread | None
video_writer: cv2.VideoWriter | None # Global for cv2.VideoWriter

recording_end_time: int = 0
video_path: str | None = None

current_config = {}

# current running recording operation, if any
video_op: str | None


def ensure_dir(path):
    """Ensure that a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)


def setup_logging(logging_config_path: str):
    """Setup logging configuration with parameterizable config file path."""
    ensure_dir(LOG_DIR)

    if os.path.exists(logging_config_path):
        logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
        log.info("Log rotation setup successfully!")
    else:
        # Fallback to basic logging if config file doesn't exist
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def publish_service_status(client, device_id, status, event_type):
    """
    Publish the status of the service to the MQTT broker.
    status (str): Status string (e.g., "start", "stop").
    """
    topic = f"te/device/{device_id}///e/{event_type}"

    payload = json.dumps({"text": status})

    client.publish(topic, payload)
    log.info(f"published status event: {payload} to topic:  {topic}")


def get_model_paths(config) -> tuple[str, Optional[str], Optional[str]]:
    """Get the paths for model file, labels file, and version from configuration."""
    model_path = config["metadata"]["model"]
    files = os.listdir(os.path.join(MODEL_BASE_PATH, model_path))

    # Get model file and labels file
    model_filename = next((f for f in files if f.endswith(".rpk")))
    model_labels = next((f for f in files if f.endswith(".txt")), None)
    version_file = next((f for f in files if f == "version"), None)
    version = None
    if version_file is not None:
        try:
            with open(
                os.path.join(MODEL_BASE_PATH, model_path, version_file), "r", encoding="utf-8"
            ) as vf:
                version = vf.readline().strip()
        except:
            log.warning(f"Could not read version file for model {model_path}")

    # Join paths
    model_name_path = os.path.join(MODEL_BASE_PATH, model_path, model_filename)
    model_labels_path = (
        os.path.join(MODEL_BASE_PATH, model_path, model_labels)
        if model_labels
        else None
    )

    return (model_name_path, model_labels_path, version)


def load_config(filename):
    """Load YAML configuration file from the current working directory."""
    file_path = os.path.join(os.getcwd(), filename)  # Get absolute path
    try:
        if not os.path.exists(file_path):
            log.info(
                f"?? Warning: {filename} not found in {os.getcwd()}. Using defaults."
            )
            return {}

        with open(file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            return config if config else {}

    except Exception as e:
        log.info(f"? Error loading {filename}: {e}")
        return {}


# check config file changes
class FileChangeHandler(FileSystemEventHandler):
    """Handler for monitoring configuration file changes."""

    def __init__(self, watched_files):
        """Initialize the handler with a list of files to watch."""
        self.watched_files = watched_files

    def on_any_event(self, event):
        """Handle file system events for watched configuration files."""
        path = event.src_path

        # Watch for specific config files
        if path in self.watched_files and event.event_type.upper() == "MODIFIED":
            log.info(f"[CONFIG FILE] {event.event_type.upper()} detected: {path}")
            publish_service_status(
                mqtt_client,
                "main",
                f"Config file : {os.path.basename(path)} updated on the Camera.",
                "config_deployed",
            )
            threading.Thread(target=restart_daemon, daemon=True).start()


def restart_daemon():
    """Restart the daemon service after configuration changes."""
    publish_service_status(
        mqtt_client, "main", "Camera Service Stopped.", "camera_service_stopped"
    )
    subprocess.run(["sudo", "systemctl", "restart", DAEMON_SERVICE], check=True)


def start_watcher(watched_files):
    """Start watching configuration files for changes."""
    event_handler = FileChangeHandler(watched_files)
    observer = Observer()

    log.info(f"files to watch: {watched_files}")

    for file_path in watched_files:
        if os.path.exists(file_path):
            observer.schedule(event_handler, path=file_path, recursive=False)
        else:
            log.info(f"Warning: {file_path} does not exist. Skipping...")

    observer.start()


def init_mqtt(config):
    """Initialize and connect the MQTT client."""
    global mqtt_client
    # Initialize MQTT client
    mqtt_client = mqtt.Client(config["mqtt"]["client_id"])
    mqtt_client.on_connect = on_connect
    mqtt_client.will_set(
        topic="te/device/main/service/rpi-vision-ai-processor/status/health",
        payload='{"status": "down"}',
        retain=True,
    )
    mqtt_client.on_message = on_message
    mqtt_client.connect(
        config["mqtt"]["broker"], config["mqtt"]["port"], config["mqtt"]["keepalive"]
    )
    mqtt_client.loop_start()
    log.info("MQTT client initialized and connected")


def subscribe_to_topics():
    """Subscribe to MQTT topics after all initialization is complete."""
    mqtt_client.subscribe("vai/+/image/+")
    mqtt_client.subscribe("vai/+/video/+")
    mqtt_client.subscribe("vai/+/model/activate/+")
    log.info("Subscribed to MQTT topics")


# Object Detection Logic


# Define MQTT callbacks
def on_connect(client, userdata, flags, rc):
    """Callback for when the client connects to the MQTT broker."""
    if rc > 0:
        log.error("Could not connect to MQTT broker")
        exit(1)
    log.info(
        "Connected to MQTT broker (subscriptions will be done after initialization)"
    )


def on_message(client, userdata, msg):
    """Callback for when a message is received from the MQTT broker."""
    log.info(f"Message received on topic {msg.topic}: {msg.payload.decode()}")
    payload = msg.payload.decode()
    if not payload.strip():
        return
    if model_match := re.match(r"vai/([^/]+)/model/activate/([^/]+)", msg.topic):
        device = model_match.group(1)
        op_id = model_match.group(2)
        model_name = msg.payload.decode()
        for camera in current_config["cameras"]:
            if current_config["cameras"][camera].get("id", "") == device:
                current_model = current_config["cameras"][camera]["metadata"]["model"]
                if current_model != model_name:
                    activate_model(device, model_name)
                else:
                    mqtt_client.publish(
                        f"vai/{device}/model/activate/{op_id}/result",
                        f'{{"status": "successful", "id": "{op_id}", "camera_id": "{device}", "result": "{model_name}"}}',
                    )
                    mqtt_client.publish(
                        f"vai/{device}/model/activate/{op_id}", "", retain=True
                    )
        return
    if image_match := re.match(r"vai/([^/]+)/image/([^/]+)", msg.topic):
        take_picture(image_match.group(1), image_match.group(2))
        return
    if video_match := re.match(r"vai/([^/]+)/video/([^/]+)", msg.topic):
        log.info(f" topic: {msg.topic}, payload: {payload}")
        duration = DEFAULT_DURATION
        framerate = DEAFULT_FRAMERATE
        try:
            data = json.loads(payload)
            duration = data.get("duration")
            framerate = data.get("framerate")
        except (ValueError, json.JSONDecodeError, TypeError):
            log.warning(
                "Invalid duration for video recording. Using default 10 seconds."
            )
        log.info(f"durations: {duration}")
        start_video_recording(
            video_match.group(1), video_match.group(2), duration, framerate
        )
        return


def activate_model(device, model_name):
    """Write the model name to the camera configuration file."""
    new_config = update_camera_config(device, model_name, True)

    with open(CAMERA_CONFIG_FILE, "w+", encoding="utf-8") as f:
        yaml.dump(new_config, f)


@dataclass
class BYTETrackerArgs:
    """Configuration arguments for the BYTE tracker."""

    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def detections_to_json_list(
    detections: Detections | Classifications | Poses, labels: list[str]
) -> list:
    """Convert detections to a JSON-serializable list format."""
    json_list = []

    for i in range(len(detections)):
        det: dict = {"confidence": float(detections.confidence[i])}
        if isinstance(detections, (Detections, Classifications)):
            class_id = int(detections.class_id[i])
            det["class_id"] = class_id
            det["label"] = labels[class_id]
        if isinstance(detections, (Detections, Poses)) and detections.bbox.size != 0:
            det["bbox"] = detections.bbox[i].tolist()
            det["tracker_id"] = (
                int(detections.tracker_id[i])
                if detections.tracker_id is not None
                else None
            )
        if isinstance(detections, Poses):
            xy = np.reshape(detections.keypoints[i], (-1, 2))
            s = detections.keypoint_scores[i]
            det["keypoints"] = [
                (label, float(x), float(y), float(s))
                for label, (x, y), s in zip(labels, xy, s)
            ]
        json_list.append(det)
    return json_list


def parse_detections(camera_config, device, labels, model_type):
    """Parse and process detections from the camera stream, publishing results to MQTT."""
    global latest_frame, frame_counter
    annotator = Annotator(
        color=ColorPalette.default(), thickness=2, text_thickness=2, text_scale=0.8
    )

    if model_type in ("object detection", "pose detection"):
        tracker = BYTETracker(
            BYTETrackerArgs(**camera_config.get("tracker", {}))
        )  # You can pass tracking config if needed
        bbox_normalization = camera_config["metadata"].get("bbox_normalization", False)
        bbox_order = camera_config["metadata"].get("bbox_order", "yx")
    with device as stream:
        frame: Frame
        for frame in stream:
            with frame_lock:
                latest_frame = frame.image
                if is_recording and time.time() < recording_end_time:
                    if video_writer is not None:
                        if latest_frame.shape[2] == 3:
                            frame_to_write = cv2.cvtColor(
                                latest_frame, cv2.COLOR_RGB2BGR
                            )
                        else:
                            frame_to_write = latest_frame
                        video_writer.write(frame_to_write)
                elif is_recording and time.time() >= recording_end_time:
                    stop_video_recording(camera_config["id"], "duration_elapsed")
                time.sleep(0.01)
            frame_counter = frame_counter + 1
            detections = frame.detections

            if frame.new_detection and isinstance(
                detections, (Detections, Classifications, Poses)
            ):
                max_detections = camera_config["metadata"].get("max_detections", 3)
                detections = detections[
                    detections.confidence > camera_config["metadata"]["threshold"]
                ][:max_detections]
                if isinstance(detections, (Detections, Poses)):
                    detections = tracker.update(
                        frame, detections
                    )  # this valid only for detection & Pose
                    if bbox_order == "xy":
                        detections.bbox = detections.bbox[:, [1, 0, 3, 2]]
                    if bbox_normalization:
                        _, w, _ = frame.image.shape
                        detections.bbox /= w
                        if hasattr(detections, "keypoints"):
                            detections.keypoints /= w
                if show_preview:
                    if isinstance(detections, Detections):
                        custom_annotate_boxes(
                            annotator, frame, detections=detections, labels=labels
                        )
                    elif isinstance(detections, Poses):
                        annotator.annotate_keypoints(frame, detections)
                        custom_annotate_boxes(annotator, frame, detections=detections)
                # Publish detected object event if conditions are met
                json_objects = detections_to_json_list(detections, labels)
                full_output = {
                    "frame_id": frame_counter,
                    "timestamp": frame.timestamp,
                    "detections": json_objects,
                }
                # Convert to JSON string
                json_string = json.dumps(full_output, indent=2)
                topic = current_config["mqtt"]["topics"]["detection"]
                if isinstance(detections, Classifications):
                    topic = current_config["mqtt"]["topics"]["classification"]
                # Publih to MQTT
                mqtt_client.publish(
                    "/".join(
                        [
                            current_config["mqtt"]["topics"]["base"],
                            camera_config["id"],
                            topic,
                            camera_config["metadata"]["model"],
                        ]
                    ),
                    json_string,
                )
                log.debug("Detection published to MQTT")
                if show_preview:
                    windowname = camera_config["metadata"].get("model", "Application")
                    frame.display(window_name=windowname)
                    if is_fullscreen:
                        cv2.setWindowProperty(
                            windowname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                        )


def get_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="RPI Vision AI Processor Service")
    parser.add_argument(
        "-p", "--plugin-config", help="Path to plugin configuration file"
    )
    parser.add_argument(
        "-c", "--camera-config", help="Path to camera configuration file"
    )
    parser.add_argument(
        "-l", "--logging-config", help="Path to logging configuration file"
    )
    parser.add_argument("-m", "--model-base-path", help="Base path for AI models")
    return parser.parse_args()


def update_camera_config(
    device: str, model_name: str, overwrite_from_model: bool
) -> dict:
    """Update camera configuration with model metadata and tracker settings."""
    updated_config = load_config(CAMERA_CONFIG_FILE)
    metadata_config = {}

    metadata_path = f"{MODEL_BASE_PATH}/{model_name}/metadata.yaml"

    if os.path.exists(metadata_path):
        metadata_config = load_config(metadata_path)
    else:
        log.info(f"Model metadata file not found: {metadata_path}")

    model_meta = metadata_config.get("metadata", {})
    model_tracker = metadata_config.get("tracker", {})

    for _, cam_data in updated_config.get("cameras", {}).items():
        if cam_data.get("id") != device:
            continue

        cam_metadata = cam_data.setdefault("metadata", {})
        cam_tracker = cam_data.setdefault("tracker", {})

        cam_metadata["model"] = model_name

        for key, value in model_meta.items():
            if key not in cam_metadata or overwrite_from_model:
                cam_metadata[key] = value

        for key, value in model_tracker.items():
            if key not in cam_tracker or overwrite_from_model:
                cam_tracker[key] = value

    log.info(f"Final Config: {updated_config}")
    return updated_config


def init_plugin_config():
    """Initialize and merge plugin and camera configurations."""
    global show_preview, is_fullscreen
    # Load both configurations
    plugin_config = load_config(PLUGIN_CONFIG_FILE)
    # TODO: handle multiple cameras/configs
    camera_config = load_config(CAMERA_CONFIG_FILE)

    config = {**plugin_config, **camera_config}
    show_preview = config.get("show_preview", False) or config.get(
        "show_fullscreen", False
    )
    is_fullscreen = config.get("show_fullscreen", False)
    return config


def start_health_checks():
    """Start periodic health check publishing to MQTT."""

    def publish_periodically():
        while True:
            n = datetime.now().timestamp()
            msg = f'{{"pid": {os.getpid()}, "status": "up", "time": {n} }}'
            mqtt_client.publish(
                "te/device/main/service/rpi-vision-ai-processor/status/health",
                payload=msg,
                retain=True,
            )
            time.sleep(30)

    # Start the publisher thread
    threading.Thread(target=publish_periodically, daemon=True).start()


class ClassificationModel(Model):
    """Model class for classification tasks."""

    def __init__(self, custom_model_file, labels, camera_config):
        """Initialize the classification model with file path, labels, and configuration."""
        super().__init__(
            model_file=custom_model_file,
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(labels, dtype=str, delimiter="\n", ndmin=1)
        self.camera_config = camera_config

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        """Pre-process input image before inference."""
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Classifications:
        """Post-process output tensors to generate classification results."""
        if self.camera_config["metadata"].get("softmax", None) is True:
            return pp_cls_softmax(output_tensors)
        return pp_cls(output_tensors)


class DetectionModel(Model):
    """Model class for object detection tasks."""

    def __init__(self, custom_model_file, labels, camera_config):
        """Initialize the detection model with file path, labels, and configuration."""
        super().__init__(
            model_file=custom_model_file,
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(labels, dtype=str, delimiter="\n", ndmin=1)
        self.camera_config = camera_config

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        """Pre-process input image before inference."""
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Detections:
        """Post-process output tensors to generate detection results."""
        detections: Detections
        if self.camera_config["metadata"].get("output_order", "bcsn") == "bscn":
            detections = pp_od_bscn(output_tensors)
        else:
            detections = pp_od_bcsn(output_tensors)
        return detections


class PoseEstimationModel(Model):
    """Model class for pose estimation tasks."""

    def __init__(self, custom_model_file, labels, camera_config):
        """Initialize the pose estimation model with file path, labels, and configuration."""
        super().__init__(
            model_file=custom_model_file,
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(labels, dtype=str, delimiter="\n", ndmin=1)
        self.camera_config = camera_config

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        """Pre-process input image before inference."""
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Poses:
        """Post-process output tensors to generate pose estimation results."""
        return pp_yolo_pose_ultralytics(output_tensors)
        # return pp_posenet(output_tensors)


def get_model_type(model_file, camera_config):
    """Determine the model type from configuration or intrinsics."""
    if camera_config["metadata"]["modeltype"] is not None:
        return camera_config["metadata"]["modeltype"]
    imx500 = IMX500(model_file)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    log.info(f"model Type: {intrinsics.task}")
    return intrinsics.task


def start_model(camera_config):
    """Initialize and start the AI model for a camera."""
    model = None
    # update model and labels
    model_file, label_file, version = get_model_paths(camera_config)

    model_type = get_model_type(model_file, camera_config)
    if model_type == "classification":
        model = ClassificationModel(
            custom_model_file=model_file, labels=label_file, camera_config=camera_config
        )
    elif model_type == "object detection":
        model = DetectionModel(
            custom_model_file=model_file, labels=label_file, camera_config=camera_config
        )
    elif model_type == "pose detection":
        model = PoseEstimationModel(
            custom_model_file=model_file, labels=label_file, camera_config=camera_config
        )
    else:
        raise NotImplementedError(f"Model type not supported: {model_type}")
    fps = camera_config["metadata"].get("framerate", 15)
    device = AiCamera(frame_rate=fps)
    device.deploy(model)

    imx_model = camera_config["metadata"]["model"]
    model_info = {"model": imx_model, "version": version}
    mqtt_client.publish(
        f'te/device/{camera_config["id"]}///twin/c8y_AICamera', json.dumps(model_info)
    )
    publish_service_status(
        mqtt_client,
        camera_config["id"],
        f"Camera Service Started with Model: {imx_model}.",
        "camera_service_started",
    )
    if hasattr(model, "labels"):
        labels = model.labels
    else:
        labels = []

    # Start parse_detections in a separate daemon thread
    # Daemon threads will be automatically terminated when the main process exits
    detection_thread = threading.Thread(
        target=parse_detections,
        args=(camera_config, device, labels, model_type),
        daemon=True,  # This ensures the thread is terminated when main process exits
        name=f"DetectionThread-{camera_config['id']}",
    )
    detection_thread.start()
    log.info(f"Started detection daemon thread for camera {camera_config['id']}")


def take_picture(camera_id, op_id):
    """Capture a frame and save it as an image, then publish the result to MQTT."""
    with frame_lock:
        if latest_frame is None:
            log.error("No frame available yet!")
            mqtt_client.publish(
                f"vai/{camera_id}/image/{op_id}/result",
                f'{{"status": "Failed", "id": "{op_id}", "camera_id": "{camera_id}", "failureReason": "No frame available yet!" }}',
            )
            mqtt_client.publish(f"vai/{camera_id}/image/{op_id}", "", retain=True)
            return
        frame_to_save = latest_frame.copy()
    now = datetime.now()
    formatted = now.strftime("%Y%m%d%H%M%S")
    filename = f"/tmp/tmp_{formatted}.jpg"
    cv2.imwrite(filename, frame_to_save)
    log.info("image captured")
    mqtt_client.publish(
        f"vai/{camera_id}/image/{op_id}/result",
        f'{{"status": "successful", "id": "{op_id}", "camera_id": "{camera_id}", "result": "{filename}" }}',
    )
    mqtt_client.publish(f"vai/{camera_id}/image/{op_id}", "", retain=True)


def start_video_recording(camera_id, local_op_id, duration, framerate):
    """Starts video recording using frames from the main stream."""
    global is_recording, video_writer, recording_end_time, video_path, video_op
    log.info(f"operation for camera : {camera_id}, op Id: {local_op_id}")
    if is_recording:
        log.info("Video recording is already in progress.")
        mqtt_client.publish(
            f"vai/{camera_id}/video/{local_op_id}/result",
            f'{{"status": "failed", "id": "{local_op_id}", "camera_id": "{camera_id}", "failure_reason": "Video recording already in progress"}}',
        )
        mqtt_client.publish(f"vai/{camera_id}/video/{local_op_id}", "", retain=True)
        return

    with frame_lock:
        if latest_frame is None:
            log.error("No frame available to determine video resolution.")
            mqtt_client.publish(
                f"vai/{camera_id}/video/{local_op_id}/result",
                f'{{"status": "failed", "id": "{local_op_id}", "camera_id": "{camera_id}", "failure_reason": "No frame available to determine video resolution"}}',
            )
            mqtt_client.publish(f"vai/{camera_id}/video/{local_op_id}", "", retain=True)
            return

        h, w, _ = latest_frame.shape  # Get frame dimensions

    # Define output directory
    output_dir = "/tmp"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{timestamp}.mp4"
    video_path = os.path.join(output_dir, video_filename)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = current_config.get("fps", framerate)  # Use FPS from config, default to 15

    try:
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        if not video_writer.isOpened():
            raise IOError("Could not open video writer.")

        is_recording = True
        video_op = local_op_id
        recording_end_time = time.time() + duration
        log.info(
            f"Started video recording for {duration} seconds. Saving to: {video_path}"
        )
    except Exception as e:
        log.error(f"Error starting video recording: {e}")
        mqtt_client.publish(
            f"vai/{camera_id}/video/{local_op_id}/result",
            f'{{"status": "failed", "id": "{local_op_id}", "camera_id": "{camera_id}", "failure_reason": "{str(e)}"}}',
        )
        mqtt_client.publish(f"vai/{camera_id}/video/{local_op_id}", "", retain=True)
        is_recording = False
        if video_writer is not None:
            video_writer.release()
        video_writer = None


def stop_video_recording(camera_id, reason="manual_stop"):
    """Stops the active video recording."""
    global is_recording, video_writer, video_path
    if is_recording:
        is_recording = False
        if video_writer is not None:
            video_writer.release()
            video_writer = None
            mqtt_client.publish(
                f"vai/{camera_id}/video/{video_op}/result",
                f'{{"status": "successful", "id": "{video_op}", "camera_id": "{camera_id}", "result": "{video_path}" }}',
            )
            mqtt_client.publish(f"vai/{camera_id}/video/{video_op}", "", retain=True)
            log.info(
                f"Video recording stopped due to: {reason}, video filepath: {video_path}"
            )
            video_path = None
        else:
            log.warning(
                "Attempted to stop recording, but video_writer was not initialized."
            )
            mqtt_client.publish(
                f"vai/{camera_id}/video/{video_op}/result",
                f'{{"status": "failed", "camera_id": "{camera_id}", "failure_reason": "Video writer not active"}}',
            )
            mqtt_client.publish(f"vai/{camera_id}/video/{video_op}", "", retain=True)
    else:
        log.info("No active video recording to stop.")
        mqtt_client.publish(
            f"vai/{camera_id}/video/{video_op}/result",
            f'{{"status": "failed", "camera_id": "{camera_id}", "failure_reason": "No active recording"}}',
        )
        mqtt_client.publish(f"vai/{camera_id}/video/{video_op}", "", retain=True)


def register_capabilities(config):
    """Register camera capabilities with the MQTT broker."""
    for cmd in ["activate_model", "image_capture", "video_capture"]:
        mqtt_client.publish(f'te/device/{config["id"]}///cmd/{cmd}', "{}")


def custom_annotate_boxes(
    annotator: Annotator,
    frame: Frame,
    detections: Detections,
    labels: Optional[List[str]] = None,
    skip_label: bool = False,
) -> np.ndarray:
    """
    Draws bounding boxes on the frame using the detections provided.

    Args:
        frame: The frame to annotate.
        detections: The detections for which the bounding boxes will be drawn
        labels: An optional list of labels corresponding to each detection.
            If `labels` are not provided, corresponding `class_id` will be used as label.
        skip_label: Is set to `True`, skips bounding box label annotation.

    Returns:
        The annotated frame.image with bounding boxes.
    """
    if (
        not isinstance(detections, Detections)
        and not isinstance(detections, Poses)
        and not isinstance(detections, Segments)
    ):
        raise ValueError(
            "Input `detections` should be of type Detections, Poses, or Segments"
        )

    # NOTE: Compensating for any introduced modified region of interest (ROI)
    # to ensure that detections are displayed correctly on top of the current `frame.image`.
    if frame.image_type != IMAGE_TYPE.INPUT_TENSOR:
        detections.compensate_for_roi(frame.roi)

    h, w, _ = frame.image.shape
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections.bbox[i]

        # Rescaling to frame size
        x1, y1, x2, y2 = (
            int(x1 * w),
            int(y1 * h),
            int(x2 * w),
            int(y2 * h),
        )
        if isinstance(detections, Detections):
            class_id = (
                detections.class_id[i] if detections.class_id is not None else None
            )
        else:  # Poses
            class_id, idx = "Person", 0

        tracker_id = (
            detections.tracker_id[i] if detections.tracker_id is not None else None
        )
        idx = tracker_id if tracker_id is not None and tracker_id > 0 else i
        color = (
            annotator.color.by_idx(idx)
            if isinstance(annotator.color, ColorPalette)
            else annotator.color
        )

        cv2.rectangle(
            img=frame.image,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=color.as_bgr(),
            thickness=annotator.thickness,
        )
        if skip_label:
            continue

        label = f"{class_id if (labels is None) else labels[class_id]} {tracker_id}"

        annotator.set_label(
            image=frame.image, x=x1, y=y1 - 20, color=color.as_bgr(), label=label
        )

    return frame.image


def main():
    """Main function with parameterizable config files for better testability."""
    global PLUGIN_CONFIG_FILE, CAMERA_CONFIG_FILE, LOGGING_CONFIG_FILE, MODEL_BASE_PATH, current_config
    args = get_args()
    plugin_config_file = args.plugin_config
    logging_config_file = args.logging_config
    camera_config_file = args.camera_config
    model_base_path = args.model_base_path
    # Override config files if provided
    if plugin_config_file:
        PLUGIN_CONFIG_FILE = plugin_config_file
    if camera_config_file:
        CAMERA_CONFIG_FILE = camera_config_file
    if logging_config_file:
        LOGGING_CONFIG_FILE = logging_config_file
    if model_base_path:
        MODEL_BASE_PATH = model_base_path

    setup_logging(LOGGING_CONFIG_FILE)

    # Initialize configuration and MQTT connection
    current_config = init_plugin_config()
    init_mqtt(current_config)

    # Start health checks
    start_health_checks()

    # Start detection threads for all cameras (these will run as daemon threads)
    for camera in current_config["cameras"]:
        camera_config = current_config["cameras"][camera]
        register_capabilities(camera_config)
        merged_config = update_camera_config(
            camera_config["id"], camera_config["metadata"]["model"], False
        )
        start_model(merged_config["cameras"][camera])

    start_watcher([PLUGIN_CONFIG_FILE, CAMERA_CONFIG_FILE])

    # Subscribe to MQTT topics after initialization
    if mqtt_client.is_connected():
        subscribe_to_topics()


if __name__ == "__main__":
    main()
    while True:
        time.sleep(1)
