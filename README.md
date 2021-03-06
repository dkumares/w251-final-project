
# W251 - Intrusion Detection with Federated Averaging

**Cybersecurity** has always been a field of cat-and-mouse games: defenders and attackers are constantly trying to stay one step ahead of the other in an effort to further their objectives, malicious or not. In the 21st century, there is no more a concept of "network perimeter" with organizations allowing Bring Your Own Data (BYOD), or hosting sensitive data on the cloud, which can be accessed from an authenticated computer or mobile device anywhere in the world. This immensely increases the attack surface, and to catch hackers in this domain, host and network intrusion detection using machine learning to identify anomalous behavior has proven to be very effective. 
  
The growth in the number of network connected devices (mobile phones, vehicles, IOT Appliances, etc.) has exponentially increased the volumes and sensitivity of network traffic data. This presents an additional challenge in building machine learning for anomaly detection which we aim to tackle using Federated Learning. 

**Federated learning (FL)** is a setting where many clients (e.g. edge devices like cell phones, Jetsons, or Amazon Echo)  collaboratively train a model under the orchestration of a central server, while keeping the training data decentralized. 

With Federated learning (FL), you download the current model to your edge device, compute an updated model at the device itself using local data and you upload the details back to the Global Model.

As such it addresses -

 - **Privacy**: As local device specific training data is never uploaded to the server. 
 - **Lower Latency**: Updated model can be used to make predictions on device. 
 - **Decentralized Learning**: Gain experience from a broad diverse range of datasets. 
 - **Reduced Communication Overheads**: Only model weights are sent to the server not raw data.

## Links
[Presentation Slides](files/W251-FINAL-IntrusionDetectionUsingFedML.pdf)  
[White Paper](files/Intrusion_Detection_Using_Federated_Learning.pdf)

## System Architecture
![system architecture](files/system_architecture.png)

## Model Architecture
![model_architecture](files/model_architecture.png)

## Dataset Characteristics
![dataset_characteristics](files/dataset_characteristics.png)

## EDA & Training Data Setup
![eda_training_setup](files/eda_training_setup.png)

## Setup
Pre-requisites: One EC2 instance for the coordinator and 'N' edge devices (e.g. Jetsons) for the trainers.

To run this setup:
1. Run [start_mqtt_brkr.sh](fl-ids/start_mqtt_brkr.sh) to start the MQTT broker on the EC2 instance.
2. Run [start_aws_coordinator.sh](fl-ids/start_aws_coordinator.sh) to start the coordinator on the EC2 instance.
3. On each edge device (Jetson NX), run [start_jetson_docker.sh](start_aws_coordinator.sh) to do the required setup.
4. Download the training data on each of the edge devices and the test data on the EC2 instance.
5. Modify corresponding trainer scripts [fl-ids/trainer1_remote.py](fl-ids/trainer1_remote.py) to point to the respective data locations on all four edge devices.
6. Inside the docker containers in each of the edge device, run `python3 fl-ids/trainer1_remote.py`
7. Once all the trainers display a message "Waiting for model from coordinator...", run the `python3 fl-ids/coordinator_local.py` on the coordinator container in the EC2 instance.

The model will train for the specified number of epochs specified in the coordinator and will exit.

## Training Demo

A video demo run on three NVIDIA Jetson devices using stub data can be found [here](files/fl_training_demo.mov). The video shows the end-to-end training process, from starting the broker and trainers to the shutdown call sent by the coordinator to signal the end of the training process. 

### Relevant Timestamps

* [0:07]: Starting the Paho MQTT broker on the cloud EC2 instance
* [0:22]: Starting trainers 1 through 3 on their corresponding Jetson devices
    * At this point, the trainers will weight for the coordinator to pass the initial common weights
* [0:36]: Starting the coordinator on the cloud EC2 instance
    * The coordinator initializes the common weights and passes the weights to the trainers via MQTT
* [0:58]: Trainers receive the weights and begins training for five global epochs
    * At the end of each epoch, the trainers pass the weights back to the coordinator, and will receive the new federated averaged weights to begin a new epoch of training
* [2:41]: The coordinate sends the exit signal to all trainers when the five global epochs are completed


## Experimental Results

**Loss Charts**

![loss_charts](files/loss_charts.png)

**Accuracy Charts**

![accuracy_charts](files/accuracy_charts.png)

## Contact:

Dhileeban Kumaresan - dkumares@berkeley.edu

Jocelyn Lu - jocelyn.r.lu@berkeley.edu

Nitin Pillai - nitin.pillai@berkeley.edu

Riyaz Kasmani - rkasmani@berkeley.edu
