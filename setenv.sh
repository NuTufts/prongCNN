#!/bin/bash

export PRONGCNN_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/prongCNN
export LOCAL_DIR=/cluster/home/twongj01/.local
export PATH=${LOCAL_DIR}/bin:${PATH}

export LARPID_DIR=${PRONGCNN_DIR}/larpid
export LARPID_LIBDIR=${LARPID_DIR}/build/installed/lib
export LARPID_BINDIR=${LARPID_DIR}/build/installed/bin
export LD_LIBRARY_PATH=${LARPID_LIBDIR}:${LD_LIBRARY_PATH}
export PATH=${LARPID_BINDIR}:${PATH}

cd /cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/
source setenv_py3_container.sh
source configure_container.sh

cd $PRONGCNN_DIR
