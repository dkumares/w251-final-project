import paho.mqtt.client as mqtt
import os
import io
import numpy as np
import sys
import time
import copy
import torch 
import pandas as pd
from model import CNNModel
from utils import *

LOCAL_MQTT_HOST="mqtt_brkr"
LOCAL_MQTT_PORT=1883
TRAINED_MODEL_TOPIC="fed_ml/+/model"

#REMOTE_TRAINER_HOSTS=["67.161.18.122"]
#REMOTE_MQTT_PORT=1883
REMOTE_TRAINER_TOPIC="fed_ml/coordinator/epoch_num/model"

#NUM_TRAINERS = len(REMOTE_TRAINER_HOSTS)
NUM_TRAINERS = 1
BATCH_SIZE = 100
TOTAL_EPOCHS = 10

trainer_weights = []
remote_mqttclients = []
accuracies = []

# Connect to Jetson to send weights to trainers
#for remote_host in REMOTE_TRAINER_HOSTS:
#    remote_mqttclient = mqtt.Client()
#    remote_mqttclient.connect(remote_host, REMOTE_MQTT_PORT, 60)
#    remote_mqttclients.append(remote_mqttclient)

global_model = CNNModel()
current_epoch = 1

def get_test_dataloader():
    print('Loading test data...')
    df_test = pd.read_csv('data/mnist_test.csv')
    X_test = df_test.iloc[:, 1:]
    y_test = df_test.iloc[:, :1]
    X_test = np.array(X_test).reshape(X_test.shape[0], 784)
    
    X_test = np.array(X_test.reshape(X_test.shape[0], X_test.shape[1]))
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_test = torch.from_numpy(X_test).float()

    y_test = np.array(y_test)
    y_test = y_test.reshape(y_test.shape[0])
    y_test = torch.from_numpy(np.array(y_test))

    test = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)
    print('Completed loading test data.')
    return test_loader

def send_initial_model():
    #global_model = CNNModel()
    #buff = io.BytesIO()
    #torch.save(global_model.state_dict(), buff)
    #buff.seek(0) 

    # Convert model to string for transmission
    #model_str = buff.getvalue()
    model_str = encode_weights(global_model)

    # Testing the conversion
    #buff1 = io.BytesIO(bytes(model_str))
    #global_model.load_state_dict(torch.load(buff1))
    topic = REMOTE_TRAINER_TOPIC.replace('epoch_num', str(current_epoch))
    print('Sending initial model to trainers...')
    #for remote_mqttclient in remote_mqttclients:
    #    remote_mqttclient.publish(topic, payload=model_str, qos=0, retain=False)
    local_mqttclient.publish(topic, payload=model_str, qos=2, retain=False)

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def update_global_weights_and_send(weights):
    global_weights = average_weights(weights)
    
    # Load global model for evaluation
    global_model.load_state_dict(global_weights)
    global_model.eval()
    total = 0
    correct = 0
    for test_images, labels in test_loader:
        # Forward propagation
        outputs = global_model(test_images)
        # Get predictions from the maximum value
        predicted = torch.max(outputs.data, 1)[1]

        # Total number of labels
        total += len(labels)
        correct += (predicted == labels).sum()

    accuracy = 100 * correct / float(total)
    accuracies.append(accuracy)
    
    global current_epoch 

    print('Epoch: {} Accuracy: {} %'.format(current_epoch, accuracy))
    
    if current_epoch == TOTAL_EPOCHS:
        print('Sending EXIT to all trainers...')
        topic = REMOTE_TRAINER_TOPIC.replace('epoch_num', 'exit')
        #for remote_mqttclient in remote_mqttclients:
        #    remote_mqttclient.publish(topic, payload='bye', qos=0, retain=False)
        local_mqttclient.publish(topic, payload='bye', qos=2, retain=False)
        print('Training Complete!')
        os._exit(0)


    current_epoch = current_epoch + 1

    model_str = encode_weights(global_model)
    topic = REMOTE_TRAINER_TOPIC.replace('epoch_num', str(current_epoch))
    print('Sending updated model to trainers...')
    #for remote_mqttclient in remote_mqttclients:
    #    remote_mqttclient.publish(topic, payload=model_str, qos=0, retain=False)
    local_mqttclient.publish(topic, payload=model_str, qos=2, retain=False)
    # TODO: Add end condition here

    # TODO: Plot accuracies
    # TODO: accumulate losses from trainers

def on_connect_local(client, userdata, flags, rc):
    print("Connected to local broker with rc: " + str(rc))
    client.subscribe(TRAINED_MODEL_TOPIC)
	
def on_message(client,userdata, msg):
  try:
    print("Model from trainer received!")
    print('Topic: ', msg.topic)
    #print('Message: ', msg.payload)
    
    model_str = msg.payload
    buff = io.BytesIO(bytes(model_str))

    # Create a dummy model to read weights
    model = CNNModel()
    model.load_state_dict(torch.load(buff))
    
    global trainer_weights
    trainer_weights.append(copy.deepcopy(model.state_dict()))
    
    # Wait until we get trained weights from all trainers
    if len(trainer_weights) == NUM_TRAINERS:
        update_global_weights_and_send(trainer_weights)
        trainer_weights.clear()

  except:
    print("Unexpected error:", sys.exc_info())

# Connect to local broker to receive weights from trainers
local_mqttclient = mqtt.Client()
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_connect = on_connect_local
local_mqttclient.on_message = on_message

# Load test data
test_loader = get_test_dataloader()

# Send initial random model to trainers.
send_initial_model()

# go into a loop
local_mqttclient.loop_forever()
