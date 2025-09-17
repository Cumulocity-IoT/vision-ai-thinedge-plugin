import unittest
from unittest.mock import patch, MagicMock, call
import json

import sys
sys.modules['paho'] = MagicMock()
sys.modules['paho.mqtt'] = MagicMock()
sys.modules['paho.mqtt.client'] = MagicMock()

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import vision_ai_mapper as mapper

class TestVisionAiMapper(unittest.TestCase):

    def setUp(self):
        self.mock_mqtt_client = MagicMock()
        mapper.mqtt_client = self.mock_mqtt_client
        mapper.operations_in_progress.clear()

    def test_handle_image_success(self):
        opid = "op123"
        camera_id = "cam1"
        filename = "test.jpg"
        mapper.operations_in_progress[opid] = {
            "status": "init",
            "externalSource": {"externalId": "dev123"}
        }
        with patch("subprocess.run") as mock_run, patch("os.remove") as mock_remove:
            mapper.handle_image_success(opid, camera_id, filename)
            self.assertEqual(mapper.operations_in_progress[opid]["status"], "successful")
            mock_run.assert_called_once()
            mock_remove.assert_called_once_with(filename)
            self.mock_mqtt_client.publish.assert_called()

    def test_handle_image_error(self):
        opid = "op124"
        camera_id = "cam2"
        mapper.operations_in_progress[opid] = {"status": "init"}
        mapper.handle_image_error(opid, camera_id, "fail_reason")
        self.assertEqual(mapper.operations_in_progress[opid]["status"], "failed")
        self.mock_mqtt_client.publish.assert_called()

    def test_handle_activate_model_success(self):
        opid = "op125"
        camera_id = "cam3"
        mapper.operations_in_progress[opid] = {"status": "init"}
        mapper.handle_activate_model_success(opid, camera_id, "modelX")
        self.assertEqual(mapper.operations_in_progress[opid]["status"], "successful")
        self.mock_mqtt_client.publish.assert_called()

    def test_handle_activate_model_error(self):
        opid = "op126"
        camera_id = "cam4"
        mapper.operations_in_progress[opid] = {"status": "init"}
        mapper.handle_activate_model_error(opid, camera_id, "fail_reason")
        self.assertEqual(mapper.operations_in_progress[opid]["status"], "failed")
        self.mock_mqtt_client.publish.assert_called()

    def test_handle_video_success(self):
        opid = "op127"
        camera_id = "cam5"
        filename = "video.mp4"
        mapper.operations_in_progress[opid] = {
            "status": "init",
            "externalSource": {"externalId": "dev456"}
        }
        with patch("subprocess.run") as mock_run, patch("os.remove") as mock_remove:
            mapper.handle_video_success(opid, camera_id, filename)
            self.assertEqual(mapper.operations_in_progress[opid]["status"], "successful")
            mock_run.assert_called_once()
            mock_remove.assert_called_once_with(filename)
            self.mock_mqtt_client.publish.assert_called()

    def test_handle_video_error(self):
        opid = "op128"
        camera_id = "cam6"
        mapper.operations_in_progress[opid] = {"status": "init"}
        mapper.handle_video_error(opid, camera_id, "fail_reason")
        self.assertEqual(mapper.operations_in_progress[opid]["status"], "failed")
        self.mock_mqtt_client.publish.assert_called()

    def test_handle_video_error_unknown_op(self):
        opid = "unknown"
        camera_id = "cam7"
        with patch.object(mapper.log, "error") as mock_log_error:
            mapper.handle_video_error(opid, camera_id, "fail_reason")
            mock_log_error.assert_called_with(f"handle_video_error: Unknown operation ID {opid}")

    def test_clean_operation(self):
        opid = "op129"
        topic = "te/device/cam8///cmd/image_capture/op129"
        mapper.operations_in_progress[opid] = {"status": "init"}
        mapper.clean_operation(opid, topic)
        self.assertNotIn(opid, mapper.operations_in_progress)
        self.mock_mqtt_client.publish.assert_called_with(topic, '', retain=True)

    def test_load_config_file_not_exists(self):
        with patch("os.path.exists", return_value=False):
            config = mapper.load_config("nonexistent.yaml")
            self.assertEqual(config, {})

    def test_load_config_file_exists(self):
        fake_yaml = {"mqtt": {"broker": "localhost", "port": 1883, "keepalive": 60}}
        with patch("os.path.exists", return_value=True), \
             patch("builtins.open", unittest.mock.mock_open(read_data="mqtt:\n  broker: localhost\n  port: 1883\n  keepalive: 60")), \
             patch("yaml.safe_load", return_value=fake_yaml):
            config = mapper.load_config("plugin_config.yaml")
            self.assertEqual(config, fake_yaml)

if __name__ == "__main__":
    unittest.main()