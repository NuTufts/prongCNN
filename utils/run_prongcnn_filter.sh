#!/bin/bash

JOBSTARTDATE=$(date)

OFFSET=0
STRIDE=85
# for debug
#SLURM_ARRAY_TASK_ID=0
#SLURMD_NODENAME=p1cmp075

# Common parameters
export OMP_NUM_THREADS=16
WORKDIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/prongCNN/
UBDL_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/
OUTPUT_DIR=${WORKDIR}/output/
OUTPUT_LOGDIR=${WORKDIR}/logdir/
FILTER_DIR=${WORKDIR}/filtered_output/
FILELIST=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/prongCNN/utils/filepairs.txt
SCRIPTNAME=clean_reco_image_file_5class.py

mkdir -p ${FILTER_DIR}

# WE WANT TO RUN MULTIPLE FILES PER JOB IN ORDER TO BE GRID EFFICIENT
start_jobid=$(( ${OFFSET} + ${SLURM_ARRAY_TASK_ID}*${STRIDE}  ))

cudadev="cpu"
echo "JOB ARRAYID: ${SLURM_ARRAY_TASK_ID} : NODE = ${SLURMD_NODENAME}"

# LOCAL JOBDIR
local_jobdir=`printf /tmp/prongcnn_dataprep_%04d ${SLURM_ARRAY_TASK_ID}`
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

# run a loop
for ((i=0;i<${STRIDE};i++)); do

    jobid=$(( ${start_jobid} + ${i} ))
    echo "JOBID ${jobid}"
  
    # GET INPUT FILENAME
    let lineno=${jobid}+1
    #inputfile=`sed -n ${lineno}p ${FILELIST} | awk '{print $1}'`
    dlrecofile=`sed -n ${lineno}p ${FILELIST} | awk '{print $2}'`
    #baseinput=$(basename $inputfile)
    basereco=$(basename $dlrecofile)
    outname=`echo ${basereco} | sed 's|larflowreco|prongCNNdata|g' | sed 's|\_kpsrecomanagerana||g'`
    outpath=${OUTPUT_DIR}/${outname}

    echo "Copy input ${outpath}"
    cp ${outpath} .

    filteredname=`echo ${outname} | sed 's|\.root|\_cleaned\_minHit10\_noSecondaries\.root|g'`
    echo ${filteredname}
    
    CMD="python3 ${SCRIPTNAME} -l -f ${outname}"
    echo $CMD
    $CMD

    echo "copy over output: ${filteredname} to ${FILTER_DIR}"
    ls -lh
    cp ${filteredname} ${FILTER_DIR}/
done

#cp *.root ${FILTER_DIR}/

JOBENDDATE=$(date)

echo "Job began at $JOBSTARTDATE"
echo "Job ended at $JOBENDDATE"

# clean-up
cd /tmp
rm -r $local_jobdir

