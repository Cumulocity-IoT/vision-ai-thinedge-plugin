import pytest
from unittest.mock import patch, MagicMock, call

import tempfile
import os

import sys
sys.modules['paho'] = MagicMock()
sys.modules['paho.mqtt'] = MagicMock()
sys.modules['paho.mqtt.client'] = MagicMock()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rpi-vision-ai-processor", "opt", "rpivisionai")))

test_resources = os.path.join(os.path.dirname(__file__), "..", "resources")
# Set test environment variables before importing the module
os.environ['VAI_LOG_DIR'] = tempfile.mkdtemp()
os.environ['VAI_MAPPER_LOGGING_CONFIG'] = os.path.join(test_resources, 'test_mapper_logging.conf')
os.environ['VAI_PLUGIN_CONFIG'] = os.path.join(test_resources, 'test_plugin_config.yaml')

import vision_ai_mapper as mapper


@pytest.fixture
def setup_mapper():
    """Setup fixture for each test."""
    mock_mqtt_client = MagicMock()
    mapper.mqtt_client = mock_mqtt_client
    mapper.operations_in_progress.clear()
    return mock_mqtt_client


def test_handle_image_success(setup_mapper):
    mock_mqtt_client = setup_mapper
    opid = "op123"
    camera_id = "cam1"
    filename = "test.jpg"
    mapper.operations_in_progress[opid] = {
        "status": "init",
        "externalSource": {"externalId": "dev123"}
    }
    with patch("subprocess.run") as mock_run, patch("os.remove") as mock_remove:
        mapper.handle_image_success(opid, camera_id, filename)
        assert mapper.operations_in_progress[opid]["status"] == "successful"
        mock_run.assert_called_once()
        mock_remove.assert_called_once_with(filename)
        mock_mqtt_client.publish.assert_called()


def test_handle_image_error(setup_mapper):
    mock_mqtt_client = setup_mapper
    opid = "op124"
    camera_id = "cam2"
    mapper.operations_in_progress[opid] = {"status": "init"}
    mapper.handle_image_error(opid, camera_id, "fail_reason")
    assert mapper.operations_in_progress[opid]["status"] == "failed"
    mock_mqtt_client.publish.assert_called()


def test_handle_activate_model_success(setup_mapper):
    mock_mqtt_client = setup_mapper
    opid = "op125"
    camera_id = "cam3"
    mapper.operations_in_progress[opid] = {"status": "init"}
    mapper.handle_activate_model_success(opid, camera_id, "modelX")
    assert mapper.operations_in_progress[opid]["status"] == "successful"
    mock_mqtt_client.publish.assert_called()


def test_handle_activate_model_error(setup_mapper):
    mock_mqtt_client = setup_mapper
    opid = "op126"
    camera_id = "cam4"
    mapper.operations_in_progress[opid] = {"status": "init"}
    mapper.handle_activate_model_error(opid, camera_id, "fail_reason")
    assert mapper.operations_in_progress[opid]["status"] == "failed"
    mock_mqtt_client.publish.assert_called()


def test_handle_video_success(setup_mapper):
    mock_mqtt_client = setup_mapper
    opid = "op127"
    camera_id = "cam5"
    filename = "video.mp4"
    mapper.operations_in_progress[opid] = {
        "status": "init",
        "externalSource": {"externalId": "dev456"}
    }
    with patch("subprocess.run") as mock_run, patch("os.remove") as mock_remove:
        mapper.handle_video_success(opid, camera_id, filename)
        assert mapper.operations_in_progress[opid]["status"] == "successful"
        mock_run.assert_called_once()
        mock_remove.assert_called_once_with(filename)
        mock_mqtt_client.publish.assert_called()


def test_handle_video_error(setup_mapper):
    mock_mqtt_client = setup_mapper
    opid = "op128"
    camera_id = "cam6"
    mapper.operations_in_progress[opid] = {"status": "init"}
    mapper.handle_video_error(opid, camera_id, "fail_reason")
    assert mapper.operations_in_progress[opid]["status"] == "failed"
    mock_mqtt_client.publish.assert_called()


def test_handle_video_error_unknown_op(setup_mapper):
    opid = "unknown"
    camera_id = "cam7"
    # Initialize the log variable since it's normally done in main()
    mapper.log = MagicMock()
    with patch.object(mapper.log, "error") as mock_log_error:
        mapper.handle_video_error(opid, camera_id, "fail_reason")
        mock_log_error.assert_called_with(f"handle_video_error: Unknown operation ID {opid}")


def test_clean_operation(setup_mapper):
    mock_mqtt_client = setup_mapper
    opid = "op129"
    topic = "te/device/cam8///cmd/image_capture/op129"
    mapper.operations_in_progress[opid] = {"status": "init"}
    mapper.clean_operation(opid, topic)
    assert opid not in mapper.operations_in_progress
    mock_mqtt_client.publish.assert_called_with(topic, '', retain=True)


def test_load_config_file_not_exists():
    # Initialize the log variable since it's normally done in main()
    mapper.log = MagicMock()
    with patch("os.path.exists", return_value=False):
        config = mapper.load_config("nonexistent.yaml")
        assert config == {}


def test_load_config_file_exists():
    fake_yaml = {"mqtt": {"broker": "localhost", "port": 1883, "keepalive": 60}}
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", MagicMock(return_value=MagicMock())), \
         patch("yaml.safe_load", return_value=fake_yaml):
        config = mapper.load_config("plugin_config.yaml")
        assert config == fake_yaml


def test_main_with_custom_config_files():
    """Test that main function accepts custom config file paths."""
    test_plugin_config = "/tmp/test_plugin.yaml"
    test_logging_config = "/tmp/test_logging.conf"
    
    with patch.object(mapper, 'load_config') as mock_load_config, \
         patch.object(mapper, 'init_mqtt') as mock_init_mqtt, \
         patch.object(mapper, 'start_health_checks') as mock_start_health, \
         patch('time.sleep', side_effect=KeyboardInterrupt):  # Stop the infinite loop
        
        mock_load_config.return_value = {"mqtt": {"broker": "localhost", "port": 1883, "keepalive": 60}}
        
        with pytest.raises(KeyboardInterrupt):
            mapper.main(plugin_config_file=test_plugin_config, logging_config_file=test_logging_config)
        
        # Verify that the custom config file path was used
        assert mapper.PLUGIN_CONFIG_FILE == test_plugin_config
        mock_load_config.assert_called_with(test_plugin_config)
        mock_init_mqtt.assert_called_once()
        mock_start_health.assert_called_once()
