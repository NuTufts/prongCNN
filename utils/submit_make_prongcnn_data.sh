#!/bin/bash

# slurm submission script for running merged dlreco through larmatch and larflowreco
#SBATCH --job-name=prongcnndata
#SBATCH --mem-per-cpu=8000
#SBATCH --time=1-0:00:00
#SBATCH --array=0
#SBATCH --cpus-per-task=2
##SBATCH --partition=batch
#SBATCH --partition=wongjiradlab
##SBATCH --partition=preempt
##SBATCH --exclude=i2cmp006,s1cmp001,s1cmp002,s1cmp003,p1cmp041,c1cmp003,c1cmp004
##SBATCH --gres=gpu:p100:3
##SBATCH --partition ccgpu
##SBATCH --gres=gpu:a100:1
##SBATCH --nodelist=ccgpu01
#SBATCH --output=stdout_make_prongcnn_data_sub00.%j.%N.log
#SBATCH --error=griderr_make_prongcnn_data_sub00.%j.%N.log

#container=/cluster/tufts/wongjiradlabnu//larbys/larbys-container/lantern_v2_me_06_03_prod/
container=/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/u20.04_cu111_torch1.9.0_minkowski.sif
BINDING=/cluster/tufts/wongjiradlabnu:/cluster/tufts/wongjiradlabnu,/cluster/tufts/wongjiradlab:/cluster/tufts/wongjiradlab,/cluster/home/twongj01:/cluster/home/twongj01
RUN_DIR=/cluster/tufts/wongjiradlabnu/twongj01/gen2/prongCNN/utils/
STRIDE=20


module load singularity/3.5.3
# GPU MODE
#singularity exec --nv ${container} bash -c "cd ${RUN_DIR} && source run_batch_kps_larmatch.sh $OFFSET $STRIDE $SAMPLE_NAME ${INPUTFILE} ${INPUTSTEM} ${FILEIDLIST}"
# CPU MODE
cd /cluster/tufts/wongjiradlab/
cd /cluster/tufts/wongjiradlabnu/
singularity exec --bind ${BINDING} ${container} bash -c "cd ${RUN_DIR} && source run_prongcnn_dataprep.sh"

