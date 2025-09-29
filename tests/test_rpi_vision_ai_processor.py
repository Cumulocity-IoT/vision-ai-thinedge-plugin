import sys
import types

import tempfile
import os
from unittest.mock import patch, MagicMock, mock_open

# --- Mocks for external modules ---
import logging.config
logging.config.fileConfig = MagicMock()

sys.modules['paho'] = MagicMock()
sys.modules['paho.mqtt'] = MagicMock()
sys.modules['paho.mqtt.client'] = MagicMock()

sys.modules['picamera2'] = types.ModuleType('picamera2')
sys.modules['picamera2.devices'] = types.ModuleType('picamera2.devices')
sys.modules['picamera2.devices.imx500'] = types.ModuleType('picamera2.devices.imx500')
sys.modules['picamera2.devices.imx500'].NetworkIntrinsics = type("NetworkIntrinsics", (), {})
sys.modules['picamera2.devices'].IMX500 = type("IMX500", (), {})

sys.modules['modlib'] = types.ModuleType('modlib')
sys.modules['modlib.apps'] = types.ModuleType('modlib.apps')
sys.modules['modlib.apps.tracker'] = types.ModuleType('modlib.apps.tracker')
sys.modules['modlib.apps.tracker.byte_tracker'] = types.ModuleType('modlib.apps.tracker.byte_tracker')
sys.modules['modlib.apps.tracker.byte_tracker'].BYTETracker = type("BYTETracker", (), {})
sys.modules['modlib.apps.annotate'] = types.ModuleType('modlib.apps.annotate')
sys.modules['modlib.apps.annotate'].ColorPalette = type("ColorPalette", (), {})
sys.modules['modlib.apps.annotate'].Annotator = type("Annotator", (), {})
sys.modules['modlib.devices'] = types.ModuleType('modlib.devices')
sys.modules['modlib.devices'].AiCamera = type("AiCamera", (), {})
sys.modules['modlib.models'] = types.ModuleType('modlib.models')
sys.modules['modlib.models.model'] = types.ModuleType('modlib.models.model')
sys.modules['modlib.models.model'].COLOR_FORMAT = object()
sys.modules['modlib.models.model'].MODEL_TYPE = object()
sys.modules['modlib.models.model'].Model = type("Model", (), {})
sys.modules['modlib.devices.frame'] = types.ModuleType('modlib.devices.frame')
sys.modules['modlib.devices.frame'].Frame = type("Frame", (), {})
sys.modules['modlib.devices.frame'].IMAGE_TYPE = object()
sys.modules['modlib.models.results'] = types.ModuleType('modlib.models.results')
sys.modules['modlib.models.results'].Classifications = type("Classifications", (), {})
sys.modules['modlib.models.results'].Detections = type("Detections", (), {})
sys.modules['modlib.models.results'].Poses = type("Poses", (), {})
sys.modules['modlib.models.results'].Segments = type("Segments", (), {})
sys.modules['modlib.models.post_processors'] = types.ModuleType('modlib.models.post_processors')
sys.modules['modlib.models.post_processors'].pp_od_bscn = MagicMock()
sys.modules['modlib.models.post_processors'].pp_od_bcsn = MagicMock()
sys.modules['modlib.models.post_processors'].pp_cls = MagicMock()
sys.modules['modlib.models.post_processors'].pp_cls_softmax = MagicMock()
sys.modules['modlib.models.post_processors'].pp_posenet = MagicMock()
sys.modules['modlib.models.post_processors'].pp_yolo_pose_ultralytics = MagicMock()
sys.modules['modlib.devices.sources'] = types.ModuleType('modlib.devices.sources')
sys.modules['modlib.devices.sources'].Images = type("Images", (), {})
sys.modules['modlib.devices.sources'].Video = type("Video", (), {})

sys.modules['watchdog'] = types.ModuleType('watchdog')
sys.modules['watchdog.events'] = types.ModuleType('watchdog.events')
sys.modules['watchdog.events'].FileSystemEventHandler = type("FileSystemEventHandler", (), {})
sys.modules['watchdog.observers'] = types.ModuleType('watchdog.observers')
sys.modules['watchdog.observers'].Observer = type("Observer", (), {})

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rpi-vision-ai-processor", "opt", "rpivisionai")))

# Set test environment variables before importing the module
os.environ['VAI_LOG_DIR'] = tempfile.mkdtemp()
os.environ['VAI_PROCESSOR_LOGGING_CONFIG'] = '/tmp/test_processor_logging.conf'
os.environ['VAI_PLUGIN_CONFIG'] = '/tmp/test_plugin_config.yaml'
os.environ['VAI_CAMERA_CONFIG'] = '/tmp/test_camera_config.yaml'
os.environ['VAI_MODEL_BASE_PATH'] = '/tmp/test_models'

import rpi_vision_ai_processor as processor


@patch("os.path.exists", return_value=False)
def test_load_config_file_not_exists(mock_exists):
    config = processor.load_config("nonexistent.yaml")
    assert config == {}


@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="mqtt:\n  broker: localhost\n  port: 1883\n  keepalive: 60")
@patch("yaml.safe_load", return_value={"mqtt": {"broker": "localhost", "port": 1883, "keepalive": 60}})
def test_load_config_file_exists(mock_yaml, mock_open_fn, mock_exists):
    config = processor.load_config("plugin_config.yaml")
    assert config == {"mqtt": {"broker": "localhost", "port": 1883, "keepalive": 60}}


@patch("os.listdir", return_value=["model.rpk", "labels.txt", "version"])
@patch("builtins.open", new_callable=mock_open, read_data="1.0.0")
def test_get_model_paths(mock_open_fn, mock_listdir):
    config = {"metadata": {"model": "test_model"}}
    with patch("os.path.join", side_effect=lambda *args: "/".join(args)):
        result = processor.get_model_paths(config)
        assert result[0].endswith("model.rpk")
        assert result[1].endswith("labels.txt")
        assert result[2] == "1.0.0"


@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data="cameras:\n  cam1:\n    id: cam1\n    metadata:\n      model: test_model")
@patch("yaml.safe_load", return_value={"cameras": {"cam1": {"id": "cam1", "metadata": {"model": "test_model"}}}})
def test_update_camera_config(mock_yaml, mock_open_fn, mock_exists):
    device = "cam1"
    model_name = "test_model"
    result = processor.update_camera_config(device, model_name, True)
    assert "cameras" in result
    assert device in result["cameras"]


def test_publish_service_status():
    mock_mqtt = MagicMock()
    device_id = "cam1"
    status = "start"
    type = "camera_service_started"
    with patch.object(processor, "log", MagicMock()):
        processor.publish_service_status(mock_mqtt, device_id, status, type)
        topic = f"te/device/{device_id}///e/{type}"
        payload = '{"text": "start"}'
        mock_mqtt.publish.assert_called_with(topic, payload)
