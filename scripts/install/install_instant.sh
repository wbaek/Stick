#!/bin/sh
CURRENT_DIRECTORY=`cd "\`dirname \"$0\"\`";pwd`
source ${CURRENT_DIRECTORY}/install_base.sh

DEVELOPER_NAME="wbaek"
PROJECT_NAME="instant"
TAG_NAME="v0.1.1"

SOURCE_PATH="https://github.com/${DEVELOPER_NAME}/${PROJECT_NAME}/archive/${TAG_NAME}.tar.gz"

DEPENDECY_PATH="${CURRENT_DIRECTORY}/../../dependency"
TARGET_PATH="${DEPENDECY_PATH}/${PROJECT_NAME}-${TAG_NAME}"
FINAL_PATH="${DEPENDECY_PATH}/${PROJECT_NAME}"

BUILD_SCRIPT="make install"

echo ""
printf "\033[1;32m"
printf "========== Get ${PROJECT_NAME} - ${TAG_NAME} ========="
printf "\033[0m"
echo ""

check_exist_libs ${PROJECT_NAME} ${DEPENDECY_PATH}

create_path ${DEPENDECY_PATH}
download_and_extract_source ${SOURCE_PATH} ${TARGET_PATH}
if [ ${?} == 1 ]; then
    build_source "${TARGET_PATH}" "${BUILD_SCRIPT}"
fi
add_target_link ${TARGET_PATH} ${FINAL_PATH}

printf "\033[1;32m"
printf "========== DONE. =========="
printf "\033[0m"
echo ""
