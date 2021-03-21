docker build -t fed-ml:coord -f Dockerfile.ubuntu .
docker run --name fed-ml-coord --network fed-ml -ti --rm --privileged -v $(pwd):/final_project fed-ml:coord
