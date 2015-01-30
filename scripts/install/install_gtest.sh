#!/bin/sh
CURRENT_DIRECTORY=`cd "\`dirname \"$0\"\`";pwd`
GTEST_DOWNLOAD_PATH="https://googletest.googlecode.com/files/gtest-1.7.0.zip"
TARGET_PATH="${CURRENT_DIRECTORY}/../../dependency/gtest"

#check_exist_lib
if [ ! -d ${TARGET_PATH} ]; then

mkdir -p ${TARGET_PATH}
DOWNLOAD_SCRIPT="curl --silent -L ${GTEST_DOWNLOAD_PATH} | tar -xz --strip-components 1 -C ${TARGET_PATH}"

echo ${DOWNLOAD_SCRIPT}
eval ${DOWNLOAD_SCRIPT}

BUILD_SCRIPT="./configure --with-pthreads=no; make -j 8"
echo ${BUILD_SCRIPT}
eval "cd ${TARGET_PATH}; ${BUILD_SCRIPT}"

fi
