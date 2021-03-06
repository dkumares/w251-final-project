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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import copy

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

sys.path.insert(0, ROOT_DIR)

from util.set_up_logger import get_logger
logger = get_logger(os.path.splitext(os.path.basename(__file__))[0], write_logs_to_file=True, run_time=RUN_TIME)

LOCAL_MQTT_HOST="mqtt_brkr"
LOCAL_MQTT_PORT=1883
TRAINED_MODEL_TOPIC="fed_ml/+/model"
TRAINED_LOSS_TOPIC="fed_ml/+/loss"
REMOTE_TRAINER_TOPIC="fed_ml/coordinator/epoch_num/model"

#NUM_TRAINERS = len(REMOTE_TRAINER_HOSTS)
NUM_TRAINERS = 3
batch_size = 20000
TOTAL_EPOCHS = 5

trainer_weights = []
trainer_losses=[]
remote_mqttclients = []
accuracies = []
losses = []

data_file = 'data/TEST--DATA-IDS-2018-multiclass.csv'
model_file = 'data/global_model.pkl'

input_size=78
global_model = MLP(input_size)
current_epoch = 1

def get_label(text):    
    if text == "Benign":
        return 0
    elif text == 'Infilteration':
        return 1
    elif text == 'SSH-Bruteforce':
        return 2
    elif text == 'DDOS attack-HOIC':
        return 3
    elif text == 'FTP-BruteForce':
        return 4
    elif text == 'DoS attacks-SlowHTTPTest':
        return 5
    elif text == 'Bot':
        return 6
    elif text == 'DoS attacks-Hulk':
        return 7
    elif text == 'DoS attacks-GoldenEye':
        return 8
    elif text == 'DDoS attacks-LOIC-HTTP':
        return 9 
    else :
        return 10

def get_test_dataloader():
    logger.info('Loading test data...')
    IDS_df = pd.read_csv(data_file)
    #IDS_df = IDS_df.drop('timestamp', axis=1)

    '''    
    # Finding the null values.
    logger.info(IDS_df.isin([np.nan, np.inf, -np.inf]).sum().sum())

    # logger.info shape after dropping NaN rows
    IDS_df = IDS_df.dropna()
    logger.info(IDS_df.shape)
    IDS_df = IDS_df.reset_index(drop=True)

    # Finding the null values.
    logger.info(IDS_df.isin([np.nan, np.inf, -np.inf]).sum().sum())
    '''

    IDS_df["label"] = IDS_df["label"].apply(get_label)    

    # Convert all categorical features into numerical form:
    encodings_dictionary = dict()
    for c in IDS_df.columns:
        if IDS_df[c].dtype == "object":
            encodings_dictionary[c] = LabelEncoder()
            IDS_df[c] = encodings_dictionary[c].fit_transform(IDS_df[c])

    y_test = IDS_df.pop("label").values
    X_test = IDS_df.values

    # Pytorch
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test)

    # Pytorch train and test sets
    test = torch.utils.data.TensorDataset(X_test, y_test)

    # data loader
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)

    logger.info('Completed loading data')
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
    logger.info('Sending initial model to trainers...')

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

    # Aggregate all predictions in this torch tensor 
    all_labels = []
    all_predictions = []

    for test_data, labels in test_loader:
        # Forward propagation
        outputs = global_model(test_data)

        # Get predictions from the maximum value
        predicted = torch.max(outputs.data, 1)[1]

        # Total number of labels
        total += len(labels)
        correct += (predicted == labels).sum()

        all_predictions.append(predicted)
        all_labels.append(labels)

    accuracy = 100 * correct / float(total)
    accuracies.append(accuracy)
    
    global current_epoch 

    logger.info('Epoch: {} Accuracy: {} %'.format(current_epoch, accuracy))

    #f1_score_per_class = f1_score(all_labels, all_predictions, average=None)
    #logger.info('f1_score_per_class : {} %'.format(f1_score_per_class))

    #f1_score_weighted_average = f1_score(all_labels, all_predictions, average='weighted')
    #logger.info('f1_score_weighted_average : {} %'.format(f1_score_weighted_average))
    
    if current_epoch == TOTAL_EPOCHS:
        logger.info('Sending EXIT to all trainers...')
        topic = REMOTE_TRAINER_TOPIC.replace('epoch_num', 'exit')
        
        accuracies_df = pd.DataFrame(accuracies, columns=['accuracy'])
        accuracies_df.to_csv('data/accuracy.csv', index=False)
        
        losses_df = pd.DataFrame(losses, columns=['loss'])
        losses_df.to_csv('data/loss.csv', index=False)
        
        torch.save(global_model, model_file)
 
        local_mqttclient.publish(topic, payload='bye', qos=2, retain=False)
        logger.info('Training Complete!')
        os._exit(0)


    current_epoch = current_epoch + 1

    model_str = encode_weights(global_model)
    topic = REMOTE_TRAINER_TOPIC.replace('epoch_num', str(current_epoch))
    logger.info('Sending updated model to trainers...')

    local_mqttclient.publish(topic, payload=model_str, qos=2, retain=False)
    # TODO: Add end condition here

    # TODO: Plot accuracies
    # TODO: Plot loss
    # TODO: accumulate losses from trainers

def on_connect_local(client, userdata, flags, rc):
    logger.info("Connected to local broker with rc: " + str(rc))
    client.subscribe(TRAINED_MODEL_TOPIC)
    client.subscribe(TRAINED_LOSS_TOPIC)

def on_message(client,userdata, msg):
  try:
    if 'loss' in msg.topic:
        logger.info("Loss from trainer received!")
        #logger.info('Topic: ', msg.topic)
        logger.info(f'Topic: {str(msg.topic)}')
        global trainer_losses
        trainer_losses.append(float(msg.payload))
        
        if len(trainer_losses) == NUM_TRAINERS:
            losses.append(np.average(trainer_losses))
            trainer_losses.clear()
    else:
        logger.info("Model from trainer received!")
        #logger.info('Topic: ', msg.topic)
        #logger.info('Message: ', msg.payload)
        logger.info(f'Topic: {str(msg.topic)}')

        model_str = msg.payload
        buff = io.BytesIO(bytes(model_str))

        # Create a dummy model to read weights
        input_size=78
        model = MLP(input_size)
        model.load_state_dict(torch.load(buff))

        global trainer_weights
        trainer_weights.append(copy.deepcopy(model.state_dict()))

        # Wait until we get trained weights from all trainers
        if len(trainer_weights) == NUM_TRAINERS:
            update_global_weights_and_send(trainer_weights)
            trainer_weights.clear()

  except:
    logger.info(f"Unexpected error: {str(sys.exc_info())}")

# Connect to local broker to receive weights from trainers
local_mqttclient = mqtt.Client()
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 3600)
local_mqttclient.on_connect = on_connect_local
local_mqttclient.on_message = on_message

# Load test data
test_loader = get_test_dataloader()

# Send initial random model to trainers.
send_initial_model()

# go into a loop
local_mqttclient.loop_forever()
