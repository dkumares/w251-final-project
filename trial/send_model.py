import paho.mqtt.client as mqtt
from model import CNNModel
import torch
import io
import base64

LOCAL_MQTT_HOST="mqtt_brkr"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="fed_ml/model"

local_mqttclient = mqtt.Client()
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)


# Read test model
model = CNNModel()
model.load_state_dict(torch.load('models/mnist_cnn.pt'))
buff = io.BytesIO()
torch.save(model.state_dict(), buff)

buff.seek(0) 

# Convert model to string for transmission
model_str = buff.getvalue()

local_mqttclient.publish(LOCAL_MQTT_TOPIC, payload=model_str, qos=0, retain=False)

local_mqttclient.loop_forever()
