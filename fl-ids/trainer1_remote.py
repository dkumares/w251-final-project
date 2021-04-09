import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import io
import re
import sys
import time
import torch
import torch.nn as nn
import paho.mqtt.client as mqtt
from model import MLP
from utils import *
from conf import RUN_TIME
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.insert(0, ROOT_DIR)

from util.set_up_logger import get_logger
logger = get_logger(os.path.splitext(os.path.basename(__file__))[0], write_logs_to_file=True, run_time=RUN_TIME)

TRAINED_MODEL_TOPIC="fed_ml/trainer1/model"

REMOTE_MQTT_HOST="18.237.219.233" # Change this IP address to the public IP Address of your EC2 instance that acts as a coordinator"
REMOTE_MQTT_PORT=1883
REMOTE_COORDINATOR_TOPIC="fed_ml/coordinator/+/model"

batch_size = 1000
data_file = 'data/test-02-16-2018-dos-slowhttp-hulk.csv'

def get_label(text):    
    if text == "Benign":
        return 0
    elif text == 'Infilteration':
        return 1
    elif text == 'DoS attacks-Slowloris':
        return 2
    elif text == 'SSH-Bruteforce':
        return 3
    elif text == 'DDOS attack-HOIC':
        return 4
    elif text == 'FTP-BruteForce':
        return 5
    elif text == 'DoS attacks-SlowHTTPTest':
        return 6
    elif text == 'Bot':
        return 7
    elif text == 'DoS attacks-Hulk':
        return 8
    elif text == 'DoS attacks-GoldenEye':
        return 9
    elif text == 'DDoS attacks-LOIC-HTTP':
        return 10
    elif text == 'DDOS attack-LOIC-UDP':
        return 11
    elif text == 'Brute Force -Web':
        return 12
    elif text == 'Brute Force -XSS':
        return 13
    elif text == 'SQL Injection':
        return 14

def get_data_loaders():
    logger.info('Loading data...')
    IDS_df = pd.read_csv(data_file)
    IDS_df = IDS_df.drop('timestamp', axis=1)
    
    # Finding the null values.
    logger.info(IDS_df.isin([np.nan, np.inf, -np.inf]).sum().sum())

    # logger.info shape after dropping NaN rows
    IDS_df = IDS_df.dropna()
    logger.info(IDS_df.shape)
    IDS_df = IDS_df.reset_index(drop=True)

    # Finding the null values.
    logger.info(IDS_df.isin([np.nan, np.inf, -np.inf]).sum().sum())

    IDS_df["label"] = IDS_df["label"].apply(get_label)

    # Convert all categorical features into numerical form:
    encodings_dictionary = dict()
    for c in IDS_df.columns:
     	if IDS_df[c].dtype == "object":
     	   encodings_dictionary[c] = LabelEncoder()
     	   IDS_df[c] = encodings_dictionary[c].fit_transform(IDS_df[c])

    IDS_df_normal = IDS_df[IDS_df["label"] == 0]
    IDS_df_abnormal = IDS_df[IDS_df["label"] != 0]
    y_normal = IDS_df_normal.pop("label").values
    X_normal = IDS_df_normal.values
    y_anomaly = IDS_df_abnormal.pop("label").values
    X_anomaly = IDS_df_abnormal.values

    # Train-test split the dataset:
    X_normal_train, X_normal_test, y_normal_train, y_normal_test = train_test_split(
    	X_normal, y_normal, test_size=0.2, random_state=11)

    X_anomaly_train, X_anomaly_test, y_anomaly_train, y_anomaly_test = train_test_split(
    	X_anomaly, y_anomaly, test_size=0.2, random_state=11, stratify=y_anomaly)

    X_train = np.concatenate((X_normal_train, X_anomaly_train))
    y_train = np.concatenate((y_normal_train, y_anomaly_train))
    X_test = np.concatenate((X_normal_test, X_anomaly_test))
    y_test = np.concatenate((y_normal_test, y_anomaly_test))

    # Pytorch
    X_train  = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train)

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test)

    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(X_train, y_train)
    valid = torch.utils.data.TensorDataset(X_test, y_test)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size = batch_size, shuffle = True)

    logger.info('Completed loading data')
    return train_loader, valid_loader

def train_and_send(global_model_weights, current_epoch):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    
    # Defining the DNN model
    input_size = train_loader.dataset.tensors[0].shape[1]
    model = MLP(input_size)
    model.load_state_dict(torch.load(global_model_weights))
    model.to(device)
    
    # Cross Entropy Loss 
    error = nn.CrossEntropyLoss().to(device)

    # Adam Optimizer
    learning_rate = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    
    logger.info('Start training...')
    start_time = time.time()
    for i, (data, labels) in enumerate(train_loader):
        train = data.to(device)
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
        
        if count % 500 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0

            # Iterate through test dataset
            for data, labels in valid_loader:
                valid = data.to(device)
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
        if count % 500 == 0:
            # logger.info Loss
            logger.info('Global Epoch:{} Iteration: {}  Loss: {}  Accuracy: {} %'.format(current_epoch, count, loss.data, accuracy))
        
        count += 1
        
    end_time = time.time()
    logger.info('Epoch completed. Time taken (seconds): ', str(end_time - start_time))
    
    # Encode model weights and send
    model.to('cpu')
    model_str = encode_weights(model)
    remote_mqttclient.publish(TRAINED_MODEL_TOPIC, payload=model_str, qos=2, retain=False)    
    
def on_connect_remote(client, userdata, flags, rc):
    logger.info("Connected to remote broker with rc: " + str(rc))
    client.subscribe(REMOTE_COORDINATOR_TOPIC)
    logger.info('Waiting for initial model from coordinator...')
    
def on_message(client,userdata, msg):
  try:
    logger.info("Model received from coordinator!")
    logger.info('Topic: ', msg.topic)
    #logger.info(msg.payload)
    epoch_num = re.search('coordinator/(.+)/model', msg.topic).group(1)
    if epoch_num == 'exit':
        logger.info('Got EXIT from coordinator. Exiting...')
        os._exit(0)
    
    current_epoch = int(epoch_num)
    
    # Decode the model weights
    model_str = msg.payload
    buff = io.BytesIO(bytes(model_str))
    
    #model.load_state_dict(torch.load(buff))
    #logger.info('Model loading complete!')
    train_and_send(buff, current_epoch)    
  except:
    logger.info("Unexpected error:", sys.exc_info())

# Load the data
train_loader, valid_loader = get_data_loaders()

remote_mqttclient = mqtt.Client()
remote_mqttclient.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, 60)
remote_mqttclient.on_connect = on_connect_remote
remote_mqttclient.on_message = on_message

# go into a loop
remote_mqttclient.loop_forever()