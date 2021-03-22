import paho.mqtt.client as mqtt
import os
import io
import numpy as np
import sys
import time
import torch 
import pandas as pd
from model import CNNModel

LOCAL_MQTT_HOST="mqtt_brkr"
LOCAL_MQTT_PORT=1883
TRAINED_MODEL_TOPIC="fed_ml/+/model"

REMOTE_TRAINER_HOST="67.161.18.122"
REMOTE_MQTT_PORT=1883
REMOTE_TRAINER_TOPIC="fed_ml/coordinator/epoch_num/model"

# Connect to Jetson to send weights to trainers
remote_mqttclient = mqtt.Client()
remote_mqttclient.connect(REMOTE_TRAINER_HOST, REMOTE_MQTT_PORT, 60)

def send_initial_model():
    global_model = CNNModel()
    buff = io.BytesIO()
    torch.save(global_model.state_dict(), buff)
    buff.seek(0) 

    # Convert model to string for transmission
    model_str = buff.getvalue()

    # Testing the conversion
    #buff1 = io.BytesIO(bytes(model_str))
    #global_model.load_state_dict(torch.load(buff1))

    print('Sending initial model to trainers...')
    remote_mqttclient.publish(REMOTE_TRAINER_TOPIC, payload=model_str, qos=0, retain=False)

def on_connect_local(client, userdata, flags, rc):
    print("Connected to local broker with rc: " + str(rc))
    client.subscribe(TRAINED_MODEL_TOPIC)
	
def on_message(client,userdata, msg):
  try:
    print("Model from trainer received!")
    print('Topic: ', msg.topic)
    print('Message: ', msg.payload)
    
    #print("Sending Ack...")
    #remote_mqttclient.publish(REMOTE_TRAINER_TOPIC, payload="Model RECEIVED!", qos=0, retain=False)
  except:
    print("Unexpected error:", sys.exc_info())

# Connect to local broker to receive weights from trainers
local_mqttclient = mqtt.Client()
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_connect = on_connect_local
local_mqttclient.on_message = on_message

# Send initial random model to trainers.
send_initial_model()

# go into a loop
local_mqttclient.loop_forever()
