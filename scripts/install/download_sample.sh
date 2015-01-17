#!/bin/sh
CURRENT_DIRECTORY=`cd "\`dirname \"$0\"\`";pwd`
DROPBOX_DOWNLOAD_PATH="https://www.dropbox.com/sh/bozf9hsbwy4vikd/AADu_qiPu9lSt61xjKzXgQTVa?dl=1"
TARGET_PATH="${CURRENT_DIRECTORY}/../../"

mkdir -p ${TARGET_PATH}
INSTALL_SCRIPT="curl --silent -L ${DROPBOX_DOWNLOAD_PATH} | tar -xz -C ${TARGET_PATH}"

echo ${INSTALL_SCRIPT}
eval ${INSTALL_SCRIPT}

