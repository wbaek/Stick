CONSOLE_RED="\033[0;31m"
CONSOLE_GREEN="\033[0;32m"
CONSOLE_YELLOW="\033[0;33m"
CONSOLE_NONE="\033[0m"

#args : project_name, dependency_path
check_exist_libs() {
    printf "${CONSOLE_GREEN}+${CONSOLE_NONE} check exist ${1} libs in ${2}\n"
    printf "${CONSOLE_YELLOW}"
    ls -alh ${2} | grep ${1}
    printf "${CONSOLE_NONE}"

    nums=`ls -alh ${2} | grep ${1} | wc -l`
    return ${nums}
}

#args : dependency_path
create_path() {
    if [ ! -d ${1} ]; then
        printf "${CONSOLE_GREEN}+${CONSOLE_NONE} make directory at ${1}\n"
        mkdir -p ${1}
    fi
}

#args : source_path, target_path
download_and_extract_source() {
    if [ ! -d "${2}" ]; then
        printf "${CONSOLE_GREEN}+${CONSOLE_NONE} download : ${1}\n"
        mkdir -p ${2}
        eval "curl --silent -L ${1} | tar -xz --strip-components 1 -C ${2}"
        return 1
    else
        printf "${CONSOLE_RED}warning${CONSOLE_NONE} "
        printf "already exist folder at ${2}\n"
        return 0
    fi
}

#args : "target_path", "build_script"
build_source() {
    if [ ${#2} -gt 0 ]; then
        printf "${CONSOLE_GREEN}+${CONSOLE_NONE} build source : ${2}\n"
        eval "cd ${1}; ${2}"
    fi
}

#args : target_path, final_path
add_target_link() {
    printf "${CONSOLE_GREEN}+${CONSOLE_NONE} add link : ${2} -> ${1}\n"
    rm -f ${2}
    ln -s ${1} ${2}
}

