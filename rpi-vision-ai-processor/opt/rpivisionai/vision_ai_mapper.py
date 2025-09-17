import json
import logging
import logging.config
import re
from typing import Any, Optional

import yaml
import os
import pwd
import grp
import threading
import time
from datetime import datetime
from dataclasses import dataclass
import subprocess

import paho.mqtt.client as mqtt

LOG_DIR = "/var/log/tedge/vai-plugin"
USER = "tedge"
GROUP = "tedge"

def ensure_dir(path, mode=0o755):
    os.makedirs(path, exist_ok=True)
    os.chmod(path, mode)

def ensure_keep_file(path, mode=0o644):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("This file is here to ensure the directory exists.\n")
    os.chmod(path, mode)

# Usage
ensure_dir(LOG_DIR)
ensure_keep_file(os.path.join(LOG_DIR, ".keep"))

logging.config.fileConfig('/etc/tedge/plugins/vai-plugin/vision_ai_mapper_logging.conf', disable_existing_loggers=False)
log = logging.getLogger(__name__)
log.info("Log rotation setup successfully!")

mqtt_client: mqtt.Client

operations_in_progress: dict[str, dict] = {}

PLUGIN_CONFIG_FILE= "/etc/tedge/plugins/vai-plugin/plugin_config.yaml"

@dataclass
class OperationResult:
    id: str
    status: str
    camera_id: str
    failure_reason: Optional[str] = None
    result: Optional[Any] = None

def init_mqtt(config):
    global mqtt_client
    # Initialize MQTT client
    mqtt_client = mqtt.Client("vision-ai-mapper")
    mqtt_client.on_connect = on_connect
    mqtt_client.will_set(topic='te/device/main/service/vision-ai-mapper/status/health',
                         payload=f'{{"status": "down"}}',
                         retain=True)
    mqtt_client.on_message = on_message
    mqtt_client.connect(config["mqtt"]["broker"], config["mqtt"]["port"], config["mqtt"]["keepalive"])
    mqtt_client.loop_start()
    log.info("MQTT client initialized")

def start_health_checks():
    def publish_periodically():
        while True:
            n = datetime.now().timestamp()
            msg = f'{{"pid": {os.getpid()}, "status": "up", "time": {n} }}'
            mqtt_client.publish('te/device/main/service/vision-ai-mapper/status/health', payload=msg, retain=True)
            time.sleep(30)

    # Start the publisher thread
    threading.Thread(target=publish_periodically, daemon=True).start()

# Define MQTT callbacks
def on_connect(client, userdata, flags, rc):
    if rc > 0:
        log.error("Could not connect to MQTT broker")
        exit(1)
    client.subscribe("te/device/+///cmd/activate_model/+")
    client.subscribe("te/device/+///cmd/image_capture/+")
    client.subscribe("te/device/+///cmd/video_capture/+")
    client.subscribe("vai/+/image/+/result")
    client.subscribe("vai/+/video/+/result")
    client.subscribe("vai/+/model/activate/+/result")
    log.info("Connected MQTT subscriber")

def on_message(client, userdata, msg):
    log.info(f"Message received on topic {msg.topic}: {msg.payload.decode()}")
    text = msg.payload.decode("utf-8")
    if not text.strip():
        return

    # handle thin-edge operation flow
    if te_match := re.match(r'te/device/([^/+]+)///cmd/([^/+]+)/([^/+]+)', msg.topic):
        camera_id, command, op_id = te_match.groups()
        payload = json.loads(msg.payload)
        if op_id not in operations_in_progress:
            log.warning(f"on_message: op_id {op_id} not in operations_in_progress")
            
        if payload["status"] == "init":
            operations_in_progress[op_id] = payload
            # trigger the actual actions by sending plugin-specific messages and set operation to "executing"
            if command == 'activate_model':
                model_name = payload["c8y_ActivateModel"]["model"]
                mqtt_client.publish(f'vai/{camera_id}/model/activate/{op_id}', model_name, retain=True)
            elif command == 'image_capture':
                mqtt_client.publish(f'vai/{camera_id}/image/{op_id}', json.dumps(payload["c8y_ImageCapture"]), retain=True)
            elif command == 'video_capture':
                mqtt_client.publish(f'vai/{camera_id}/video/{op_id}', json.dumps(payload["c8y_VideoCapture"]), retain=True)
            payload["status"] = "executing"
            mqtt_client.publish(msg.topic, json.dumps(payload))
        elif payload["status"] == "successful" or payload["status"] == "failed":
            # operation already handled, just clean it up
            clean_operation(op_id, msg.topic)

    # handle image results
    elif image_result_match := re.match(r'vai/([^/]+)/image/([^/]+)/result', msg.topic):
        camera_id, op_id = image_result_match.groups()
        try:
            payload = json.loads(msg.payload)
            operation_result = OperationResult(**payload)
            if operation_result.status == 'successful':
                handle_image_success(op_id, camera_id, filename=operation_result.result)
            else:
                handle_image_error(op_id, camera_id, operation_result.failure_reason)
        except Exception as e:
            log.error(e)
            handle_image_error(op_id, camera_id, "Could not send or parse image result message")

    # handle model activation results
    elif model_result_match := re.match(r'vai/([^/]+)/model/activate/([^/]+)/result', msg.topic):
        camera_id, op_id = model_result_match.groups()
        try:
            payload = json.loads(msg.payload)
            operation_result = OperationResult(**payload)
            if operation_result.status == 'successful':
                handle_activate_model_success(op_id, camera_id, operation_result.result)
            else:
                handle_activate_model_error(op_id, camera_id, operation_result.failure_reason)
        except Exception as e:
            log.error(e)
            handle_activate_model_error(op_id, camera_id, "Could not send or parse model activation message")
            
    # handle video results
    elif video_result_match := re.match(r'vai/([^/]+)/video/([^/]+)/result', msg.topic):
        camera_id, op_id = video_result_match.groups()
        try:
            payload = json.loads(msg.payload)
            operation_result = OperationResult(**payload)
            if operation_result.status == 'successful':
                handle_video_success(op_id, camera_id, filename=operation_result.result)
            else:
                handle_video_error(op_id, camera_id, operation_result.failure_reason)
        except Exception as e:
            log.error(e)
            handle_video_error(op_id, camera_id, "Could not send or parse video result message")

def handle_image_success(opid, camera_id, filename):
    original_op = operations_in_progress[opid]
    original_op["status"] = "successful"
    device_id = original_op["externalSource"]["externalId"]
    media_info = {
        "c8y_CameraOutput": {
            "type": "image",
            "format": "jpg"
        }
    }
    subprocess.run(["tedge", "upload", "c8y", "--file", filename, "--type", "c8y_CameraImage", "--device-id", device_id, "--json", json.dumps(media_info) ], check=True)
    os.remove(filename)
    mqtt_client.publish(f'te/device/{camera_id}///cmd/image_capture/{opid}', json.dumps(original_op))

def handle_image_error(opid, camera_id, failure_reason):
    original_op = operations_in_progress[opid]
    original_op["status"] = "failed"
    mqtt_client.publish(f'te/device/{camera_id}///cmd/image_capture/{opid}', json.dumps(original_op))

def handle_activate_model_success(opid, camera_id, model_name):
    original_op = operations_in_progress[opid]
    original_op["status"] = "successful"
    mqtt_client.publish(f'te/device/{camera_id}///cmd/activate_model/{opid}', json.dumps(original_op))

def handle_activate_model_error(opid, camera_id, failure_reason):
    original_op = operations_in_progress[opid]
    original_op["status"] = "failed"
    mqtt_client.publish(f'te/device/{camera_id}///cmd/activate_model/{opid}', json.dumps(original_op))
    
def handle_video_success(opid, camera_id, filename):
    original_op = operations_in_progress[opid]
    original_op["status"] = "successful"
    device_id = original_op["externalSource"]["externalId"]
    media_info = {
        "c8y_CameraOutput": {
            "type": "video",
            "format": "mp4"
        }
    }
    subprocess.run(["tedge", "upload", "c8y", "--file", filename, "--type", "c8y_CameraVideo", "--device-id", device_id, "--json", json.dumps(media_info)], check=True)
    os.remove(filename)
    mqtt_client.publish(f'te/device/{camera_id}///cmd/video_capture/{opid}', json.dumps(original_op))

def handle_video_error(opid, camera_id, failure_reason):
    original_op = operations_in_progress.get(opid)
    if original_op is None:
        log.error(f"handle_video_error: Unknown operation ID {opid}")
        return
    original_op["status"] = "failed"
    mqtt_client.publish(f'te/device/{camera_id}///cmd/video_capture/{opid}', json.dumps(original_op))

def clean_operation(op_id, topic):
    del operations_in_progress[op_id]
    mqtt_client.publish(topic, '', retain=True)
    pass

def load_config(filename):
    """Load YAML configuration file from the current working directory."""
    file_path = os.path.join(os.getcwd(), filename)  # Get absolute path
    try:
        if not os.path.exists(file_path):
            log.info(f"?? Warning: {filename} not found in {os.getcwd()}. Using defaults.")
            return {}

        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
            return config if config else {}

    except Exception as e:
        log.info(f"? Error loading {filename}: {e}")
        return {}

if __name__ == "__main__":
    plugin_config = load_config(PLUGIN_CONFIG_FILE)
    init_mqtt(plugin_config)
    start_health_checks()
    while True:
        time.sleep(0.5)