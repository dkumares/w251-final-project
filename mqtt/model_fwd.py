import paho.mqtt.client as mqtt
import torch
import io
import pandas as pd
from model import CNNModel


LOCAL_MQTT_HOST="mqtt_brkr"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="fed_ml/model"

REMOTE_MQTT_HOST="52.53.246.158"
REMOTE_MQTT_PORT=1883
REMOTE_MQTT_TOPIC="fed_ml/model"

# remote_mqttclient = mqtt.Client()
# remote_mqttclient.connect(REMOTE_MQTT_HOST, LOCAL_MQTT_PORT, 60)

# Initialize Model
model = CNNModel()

def on_connect_local(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    client.subscribe(LOCAL_MQTT_TOPIC)
	
def on_message(client,userdata, msg):
  try:
    print("Model received!")
    print(msg.topic + ' ' + str(msg.payload))
    
    model_str = msg.payload
    buff = io.BytesIO(bytes(model_str))
    model.load_state_dict(torch.load(buff))
    
    print(model.state_dict())
    
    
    
    # if we wanted to re-publish this message, something like this should work
    msg = msg.payload
    #remote_mqttclient.publish(REMOTE_MQTT_TOPIC, payload=msg, qos=0, retain=False)
    #print("message re-published")
  except:
    print("Unexpected error:", sys.exc_info()[0])

local_mqttclient = mqtt.Client()
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_connect = on_connect_local
local_mqttclient.on_message = on_message

#local_mqttclient.publish(LOCAL_MQTT_TOPIC, payload="test msg6", qos=0, retain=False)

# go into a loop
local_mqttclient.loop_forever()
