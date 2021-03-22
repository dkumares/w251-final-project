docker build -t pytorch-fl -f Dockerfile.pytorch .
docker run --network fed-ml --privileged --runtime nvidia --rm -it -p 8888:8888 -p 8889:8889 -v $(pwd):/final_project pytorch-fl
