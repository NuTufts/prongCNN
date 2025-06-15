#!/bin/bash

export PRONGCNN_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export LARPID_DIR=${PRONGCNN_DIR}/larpid
export LARPID_LIBDIR=${LARPID_DIR}/build/installed/lib
export LARPID_BINDIR=${LARPID_DIR}/build/installed/bin

[[ ":$LD_LIBRARY_PATH:" != *":${LARPID_LIBDIR}:"* ]] && LD_LIBRARY_PATH="${LARPID_LIBDIR}:${LD_LIBRARY_PATH}"
[[ ":$PATH:" != *":${LARPID_BINDIR}:"* ]] && PATH="${LARFLOW_BINDIR}:${PATH}"

# SET LIBTORCH CXX-11 ABI ENVIRONMENT
# This location is in the u20 container
LIBTORCH_DIR=/usr/local/libtorch1.9.0_cxx11abi/libtorch
LIBTORCH_LIBRARY_DIR=${LIBTORCH_DIR}/lib
LIBTORCH_BIN_DIR=${LIBTORCH_DIR}/bin

[[ ":$LD_LIBRARY_PATH:" != *":${LIBTORCH_LIBRARY_DIR}:"* ]] && LD_LIBRARY_PATH="${LIBTORCH_LIBRARY_DIR}:${LD_LIBRARY_PATH}"
[[ ":$PATH:" != *":${LIBTORCH_BIN_DIR}:"* ]] && PATH="${LIBTORCH_BIN_DIR}:${PATH}"

# Setup ubdl environment
# TODO: change to cluster location
# But for dev: set this to your UBDL folder
MY_UBDL_DIR=
if [ -z "${UBDL_BASEDIR}" ]; then
    echo "Also set up UBDL at ${MY_UBDL_DIR}"
    cd ${MY_UBDL_DIR}
    source setenv_py3_container.sh
    source configure_container.sh
fi

# Return to the script location
cd ${PRONGCNN_DIR}
