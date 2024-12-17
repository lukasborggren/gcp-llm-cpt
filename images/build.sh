#!/bin/bash
set -e

IMAGE_TYPE=$1
if [[ "$IMAGE_TYPE" != "spark" && "$IMAGE_TYPE" != "cuda" ]]; then
    echo "Error: IMAGE_TYPE must be either 'spark' or 'cuda'."
    exit 1
fi

BUILD_DIR=$(dirname "$0")
DOCKERFILE=$BUILD_DIR/Dockerfile.$IMAGE_TYPE

CONFIG=$(poetry run python -c "import src.config.gcp as c; print(c.PROJECT_ID, c.REGION, c.IMAGES['$IMAGE_TYPE'], c.PYTHON_VERSION, c.CUDA_VERSION)")
IFS=" " read -r PROJECT REGION IMAGE PYTHON_VERSION CUDA_VERSION <<< $CONFIG
SHORT_SHA=$(git rev-parse --short HEAD)

poetry export -o $BUILD_DIR/requirements.txt
trap "rm $BUILD_DIR/requirements.txt" EXIT

gcloud builds submit \
    --project $PROJECT \
    --region $REGION \
    --config $BUILD_DIR/cloudbuild.yaml \
    --ignore-file $BUILD_DIR/.gcloudignore \
    --substitutions SHORT_SHA=$SHORT_SHA,_IMAGE=$IMAGE,_DOCKERFILE=$DOCKERFILE,_PYTHON_VERSION=$PYTHON_VERSION,_CUDA_VERSION=$CUDA_VERSION \
    .