import paho.mqtt.client as mqtt
import sys

LOCAL_MQTT_HOST="mqtt_brkr"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="fed_ml/trainer1/model"

REMOTE_MQTT_HOST="34.213.224.165"
REMOTE_MQTT_PORT=1883
REMOTE_MQTT_TOPIC="fed_ml/trainer1/model"

remote_mqttclient = mqtt.Client()
remote_mqttclient.connect(REMOTE_MQTT_HOST, LOCAL_MQTT_PORT, 60)

def on_connect_local(client, userdata, flags, rc):
    print("connected to local broker with rc: " + str(rc))
    client.subscribe(LOCAL_MQTT_TOPIC)
	
def on_message(client,userdata, msg):
  try:
    print("Model received!")
    print(msg.topic + ' ' + str(msg.payload))
    
    # re-publish message
    msg = msg.payload
    remote_mqttclient.publish(REMOTE_MQTT_TOPIC, payload=msg, qos=0, retain=False)
    print("Model sent to Coordinator!")
  except:
    print("Unexpected error:", sys.exc_info()[0])

local_mqttclient = mqtt.Client()
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_connect = on_connect_local
local_mqttclient.on_message = on_message

# go into a loop
local_mqttclient.loop_forever()
