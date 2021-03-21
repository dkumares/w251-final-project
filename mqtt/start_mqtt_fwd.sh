docker build -t fed-ml:mqtt_fwd -f Dockerfile.mqtt_fwd .
docker run --name mqtt_fwd --network fed-ml -ti --rm --privileged fed-ml:mqtt_fwd python3 model_fwd.py
