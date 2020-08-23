#!/bin/bash
set -e
docker run --rm --volume ${PWD}/${FOLDER_PATH}:/data --env RCLONE_CONFIG_COLAB_TYPE=drive --env RCLONE_CONFIG_COLAB_SCOPE=drive --env RCLONE_CONFIG_COLAB_CLIENT_ID=${RCLONE_CONFIG_COLAB_CLIENT_ID} --env RCLONE_CONFIG_COLAB_CLIENT_SECRET=${RCLONE_CONFIG_COLAB_CLIENT_SECRET} --env RCLONE_CONFIG_COLAB_TOKEN="${RCLONE_CONFIG_COLAB_TOKEN}" rclone/rclone -q copy /data colab:RBniCS-jupyter --include "*.ipynb"
