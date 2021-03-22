import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import io
import time
import torch
import torch.nn as nn
import paho.mqtt.client as mqtt
#from torch.autograd import Variable
from model import CNNModel

LOCAL_MQTT_HOST="mqtt_brkr"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="fed_ml/trainer1/model"

REMOTE_MQTT_HOST="34.213.224.165"
REMOTE_MQTT_PORT=1883
REMOTE_COORDINATOR_TOPIC="fed_ml/coordinator/model"

batch_size = 100
data_file = 'data/mnist_train.csv'

remote_mqttclient = mqtt.Client()
remote_mqttclient.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, 60)

def get_data_loaders():
    print('Loading data...')
    df_train = pd.read_csv(data_file)
    df_features = df_train.iloc[:, 1:785]
    df_label = df_train.iloc[:, 0]
    X_train, X_valid, y_train, y_valid = train_test_split(df_features, df_label, 
                                                      test_size = 0.2,
                                                      random_state = 1234)
    X_train = np.array(X_train).reshape(X_train.shape[0], X_train.shape[1])
    X_valid = np.array(X_valid).reshape(X_valid.shape[0], X_valid.shape[1])
    
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    
    X_train  = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(np.array(y_train))
    X_valid = X_valid.reshape(X_valid.shape[0], 1, 28, 28)
    X_valid = torch.from_numpy(X_valid).float()

    y_valid = torch.from_numpy(np.array(y_valid))
    
    

    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(X_train, y_train)
    valid = torch.utils.data.TensorDataset(X_valid, y_valid)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size = batch_size, shuffle = False)
    
    return train_loader, valid_loader

def train_and_send(global_model_weights):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    model = CNNModel()
    model.load_state_dict(torch.load(global_model_weights))
    model.to(device)
    
    # Cross Entropy Loss 
    error = nn.CrossEntropyLoss().to(device)
    # SGD Optimizer
    learning_rate = 0.001
    # TODO: Try SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    
    start_time = time.time()
    for i, (images, labels) in enumerate(train_loader):
        train = images.to(device)
        labels = labels.to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train)
        # Calculate softmax and cross entropy loss
        loss = error(outputs, labels)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        
        count += 1
        if count % 100 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in valid_loader:
                valid = images.to(device)
                labels = labels.to(device)
                
                # Forward propagation
                outputs = model(valid)
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
        if count % 100 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))

    end_time = time.time()
    print('Time taken for epoch: ', str(end_time - start_time))
    remote_mqttclient.publish(REMOTE_MQTT_TOPIC, payload="Completed 1 epoch", qos=0, retain=False)
    
    
def on_connect_local(client, userdata, flags, rc):
    print("Connected to local broker with rc: " + str(rc))
    client.subscribe(REMOTE_COORDINATOR_TOPIC)
    
def on_message(client,userdata, msg):
  try:
    print("Model received from coordinator!")
    print(msg.topic)
    model_str = msg.payload
    global_model_weights_buff = io.BytesIO(bytes(model_str))
    train_and_send(global_model_weights_buff)    
  except:
    print("Unexpected error:", sys.exc_info()[0])

# Load the data
train_loader, valid_loader = get_data_loaders()

local_mqttclient = mqtt.Client()
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_connect = on_connect_local
local_mqttclient.on_message = on_message

# go into a loop
local_mqttclient.loop_forever()