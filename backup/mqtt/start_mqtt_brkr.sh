docker build -t fed-ml:mqtt_brkr -f Dockerfile.mqtt_brkr .
docker run --name mqtt_brkr --network fed-ml -p 1883:1883 -ti --rm --privileged fed-ml:mqtt_brkr mosquitto
