"""
AI Model classes for Vision AI Camera.

Wraps modlib Model base class for classification, object detection,
and pose estimation tasks.
"""

import logging
from typing import List

import numpy as np

from modlib.models.model import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.results import Classifications, Detections, Poses
from modlib.models.post_processors import (
    pp_od_bscn,
    pp_od_bcsn,
    pp_cls,
    pp_cls_softmax,
    pp_yolo_pose_ultralytics,
)

log = logging.getLogger(__name__)


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
        if labels is None:
            log.warning("No labels file provided for classification model. Using empty labels.")
            self.labels = np.array([])
        else:
            try:
                self.labels = np.genfromtxt(labels, dtype=str, delimiter="\n", ndmin=1)
            except (OSError, ValueError) as e:
                log.error(f"Failed to load labels from {labels}: {e}")
                raise ValueError(f"Failed to load labels file: {e}")
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
        if labels is None:
            log.warning("No labels file provided for detection model. Using empty labels.")
            self.labels = np.array([])
        else:
            try:
                self.labels = np.genfromtxt(labels, dtype=str, delimiter="\n", ndmin=1)
            except (OSError, ValueError) as e:
                log.error(f"Failed to load labels from {labels}: {e}")
                raise ValueError(f"Failed to load labels file: {e}")
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
        if labels is None:
            log.warning("No labels file provided for pose estimation model. Using empty labels.")
            self.labels = np.array([])
        else:
            try:
                self.labels = np.genfromtxt(labels, dtype=str, delimiter="\n", ndmin=1)
            except (OSError, ValueError) as e:
                log.error(f"Failed to load labels from {labels}: {e}")
                raise ValueError(f"Failed to load labels file: {e}")
        self.camera_config = camera_config

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        """Pre-process input image before inference."""
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Poses:
        """Post-process output tensors to generate pose estimation results."""
        return pp_yolo_pose_ultralytics(output_tensors)
        # return pp_posenet(output_tensors)
