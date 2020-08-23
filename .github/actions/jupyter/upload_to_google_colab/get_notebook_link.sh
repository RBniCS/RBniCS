#!/bin/bash
set -e
src=${1?"Please provide input notebook as first argument"}
sub_src=$(dirname ${src#${FOLDER_PATH}/})/$(basename ${src})

function get_link() {
    set +e
    docker run --rm --env RCLONE_CONFIG_COLAB_TYPE=drive --env RCLONE_CONFIG_COLAB_SCOPE=drive --env RCLONE_CONFIG_COLAB_CLIENT_ID=${RCLONE_CONFIG_COLAB_CLIENT_ID} --env RCLONE_CONFIG_COLAB_CLIENT_SECRET=${RCLONE_CONFIG_COLAB_CLIENT_SECRET} --env RCLONE_CONFIG_COLAB_TOKEN="${RCLONE_CONFIG_COLAB_TOKEN}" rclone/rclone -q link colab:RBniCS-jupyter/${sub_src}
    set -e
}

function drive_to_colab() {
    drive="https://drive.google.com/open?id="
    colab="https://colab.research.google.com/drive/"
    echo "${1/${drive}/${colab}}"
}

function upload_notebook() {
    docker run --rm --volume ${PWD}/$(dirname ${src}):/data --env RCLONE_CONFIG_COLAB_TYPE=drive --env RCLONE_CONFIG_COLAB_SCOPE=drive --env RCLONE_CONFIG_COLAB_CLIENT_ID=${RCLONE_CONFIG_COLAB_CLIENT_ID} --env RCLONE_CONFIG_COLAB_CLIENT_SECRET=${RCLONE_CONFIG_COLAB_CLIENT_SECRET} --env RCLONE_CONFIG_COLAB_TOKEN="${RCLONE_CONFIG_COLAB_TOKEN}" rclone/rclone -q copy /data colab:RBniCS-jupyter/$(dirname ${sub_src}) --include "*.ipynb"
}

link=$(drive_to_colab $(get_link))
if [ -f "${src}.link" ]; then
    if [ "${link}" != "$(cat ${src}.link)" ]; then
        echo "ERROR: notebook ${sub_src} is currently served at ${link}, but used to served at $(cat ${src}.link)"
        exit 1
    else
        echo "Notebook ${sub_src} is currently served at ${link}"
    fi
else
    if [ "${link}" != "" ]; then
        echo ${link} > "${src}.link"  # for later use
        echo "Notebook ${sub_src} is currently served at ${link}"
    else
        # The file does not exists yet on colab: upload current notebook (even though it has not
        # been preprocessed by the patch script) to get the future link
        upload_notebook
        link=$(drive_to_colab $(get_link))
        if [ "${link}" != "" ]; then
            echo ${link} > "${src}.link"  # for later use
            echo "Notebook ${sub_src} will be served at ${link}"
        else
            echo "ERROR: cannot create notebook ${sub_src}"
            exit 1
        fi
    fi
fi
