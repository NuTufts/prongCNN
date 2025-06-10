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
FILELIST=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/prongCNN/utils/filepairs.txt
SCRIPTNAME=prepare_reco_images_cluster.py

mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_LOGDIR}


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
    inputfile=`sed -n ${lineno}p ${FILELIST} | awk '{print $1}'`
    dlrecofile=`sed -n ${lineno}p ${FILELIST} | awk '{print $2}'`
    baseinput=$(basename $inputfile)
    basereco=$(basename $dlrecofile)

    echo "copy over ${inputfile}"
    echo "copy over ${dlrecofile}"
    cp $inputfile $baseinput
    cp $dlrecofile $basereco
    chmod u+w $baseinput
    outname=`echo ${basereco} | sed 's|larflowreco|prongCNNdata|g' | sed 's|\_kpsrecomanagerana||g'`
    CMD="python3 ${SCRIPTNAME} -if ${basereco} -it ${baseinput} -ia ${jobid} -o ${outname}"
    echo $CMD
    $CMD

    echo "remove local copy: ${baseinput}"
    echo "remove local copy: ${basereco}"
    rm -f $baseinput
    rm -f $basereco

    echo "copy over output: ${outname} to ${OUTPUT_DIR}"
    cp $outname ${OUTPUT_DIR}/
done

JOBENDDATE=$(date)

echo "Job began at $JOBSTARTDATE"
echo "Job ended at $JOBENDDATE"

# copy log to logdir
#cp $local_logfile $OUTPUT_LOGDIR/

# clean-up
cd /tmp
rm -r $local_jobdir

