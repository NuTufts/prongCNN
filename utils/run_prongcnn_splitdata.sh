#!/bin/bash

JOBSTARTDATE=$(date)

# Common parameters
export OMP_NUM_THREADS=16
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/prongCNN/
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/
FILELIST=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/prongCNN/prongcnndata_filtered_badfilesremoved_shuffled.txt
SCRIPTNAME=split_image_file_by_val_num.py

cudadev="cpu"

# LOCAL JOBDIR
local_jobdir=`printf /tmp/prongcnn_maketrainval_%04d ${SLURM_ARRAY_TASK_ID}`
rm -rf $local_jobdir
mkdir -p $local_jobdir

# comment out if running locally inside container
alias python=python3
cd ${UBDL_DIR}
source setenv_py3_container.sh
source configure_container.sh
export PYTHONPATH=${LARMATCH_DIR}:${PYTHONPATH}

cd $local_jobdir

cp ${WORKDIR}/preprocess/${SCRIPTNAME} .

CMD="python3 ${SCRIPTNAME} -l -f ${FILELIST}"
echo $CMD
$CMD

cp *.root ${WORKDIR}


JOBENDDATE=$(date)

echo "Job began at $JOBSTARTDATE"
echo "Job ended at $JOBENDDATE"

# clean-up
cd /tmp
#rm -r $local_jobdir

