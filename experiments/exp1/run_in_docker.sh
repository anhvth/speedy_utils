#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$DIR/../.." && pwd )"

IMAGE_NAME="speedy_utils_exp1"

echo "Building Docker image $IMAGE_NAME..."
docker build -t "$IMAGE_NAME" "$DIR"

echo "Running Docker container..."
docker run --rm -it \
    "$IMAGE_NAME"
