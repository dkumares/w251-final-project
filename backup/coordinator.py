import paho.mqtt.client as mqtt
import os
import io
import numpy as np
import sys
import time
import torch 
import pandas as pd
from model import CNNModel

# Load data for testing
df_test = pd.read_csv('data/mnist_test.csv')

df_test_features = df_test.iloc[:, 1:785]
df_test_label = df_test.iloc[:, 0]

X_test = df_test_features.to_numpy()
y_test = df_test_label.to_numpy()

sample = 10

img = X_test[sample] #shape (784,1)
img = img.reshape(1, 1, 28, 28) #shape (1,1,28,28)
img  = torch.from_numpy(img).float() #tensor

model = CNNModel()

LOCAL_MQTT_HOST="mqtt_brkr"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="fed_ml/trainer1/model"

REMOTE_TRAINER_HOST="67.161.18.122"
REMOTE_MQTT_PORT=1883
REMOTE_TRAINER_TOPIC="fed_ml/coordinator/model"

remote_mqttclient = mqtt.Client()
remote_mqttclient.connect(REMOTE_TRAINER_HOST, REMOTE_MQTT_PORT, 60)

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        client.subscribe(LOCAL_MQTT_TOPIC)
	
def on_message(client,userdata, msg):
  try:
    print("Model received!")
    print('Topic: ', msg.topic)

    model_str = msg.payload
    buff = io.BytesIO(bytes(model_str))
    model.load_state_dict(torch.load(buff))
    prediction = model(img).detach().numpy()[0].argmax()
    print('Prediction: ', prediction)
    
    print("Sending Ack...")
    remote_mqttclient.publish(REMOTE_TRAINER_TOPIC, payload="Model RECEIVED!", qos=0, retain=False)
  except:
    print("Unexpected error:", sys.exc_info())

local_mqttclient = mqtt.Client()
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_connect = on_connect_local
local_mqttclient.on_message = on_message


# go into a loop
local_mqttclient.loop_forever()
