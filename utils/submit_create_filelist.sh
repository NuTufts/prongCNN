#!/bin/bash

# slurm submission script for running merged dlreco through larmatch and larflowreco
#SBATCH --job-name=prongcnndata
#SBATCH --mem-per-cpu=8000
#SBATCH --time=2-0:00:00
#SBATCH --array=0
#SBATCH --cpus-per-task=2
#SBATCH --partition=batch
##SBATCH --partition=wongjiradlab
##SBATCH --partition=preempt
##SBATCH --exclude=i2cmp006,s1cmp001,s1cmp002,s1cmp003,p1cmp041,c1cmp003,c1cmp004
##SBATCH --gres=gpu:p100:3
##SBATCH --partition ccgpu
##SBATCH --gres=gpu:a100:1
##SBATCH --nodelist=ccgpu01
#SBATCH --output=stdout_create_filelist.%j.%N.log
#SBATCH --error=griderr_create_filelist.%j.%N.log

container=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/u20.04_cu111_cudnn8_torch1.9.0_minkowski_npm.sif
BINDING=/cluster/tufts/wongjiradlabnu:/cluster/tufts/wongjiradlabnu,/cluster/tufts/wongjiradlab:/cluster/tufts/wongjiradlab,/cluster/home/twongj01:/cluster/home/twongj01
RUN_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/prongCNN/
FILELIST=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/prongCNN/filelist.txt
OUTPUTLIST=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/prongCNN/filelist_with_counts.txt

#module load singularity/3.5.3
module load apptainer/1.2.4-suid

CMD="python3 utils/create_filelist_with_counts.py --input ${FILELIST} --output ${OUTPUTLIST} --skip-missing"

# GPU MODE
#singularity exec --nv ${container} bash -c "cd ${RUN_DIR} && source run_batch_kps_larmatch.sh $OFFSET $STRIDE $SAMPLE_NAME ${INPUTFILE} ${INPUTSTEM} ${FILEIDLIST}"
# CPU MODE
ls /cluster/tufts/wongjiradlab/ > /dev/null
ls /cluster/tufts/wongjiradlabnu/ > /dev/null
apptainer exec --bind ${BINDING} ${container} bash -c "cd ${RUN_DIR} && source setenv.sh && ${CMD}"

