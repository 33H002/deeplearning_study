#!/bin/bash

IMG_NAME='torch-cluster'
NETWORK_NAME='torch'

docker network create --driver=bridge $NETWORK_NAME
docker run -itd --name $IMG_NAME --net=$NETWORK_NAME -v $(pwd)/script:/workspace -p 8888:8888 --gpus all --restart=always $IMG_NAME

docker exec -it $IMG_NAME "/tmp/startJupyter.sh"
