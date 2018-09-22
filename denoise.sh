#!/bin/bash
#$ -l gpu=1
#$ -l h_rt=24:00:00
#$ -l h_vmem=40G
#$ -cwd
#$ -V
#$ -j y
#$ -o logs/output/
#Usage: ./vpython.sh master hashnet.py --lr 0.000001 etc
echo "$@"
source ~/virtual_enviroment/gpu-tensorflow-1.4/bin/activate
echo "Reserved GPU: $SGE_GPU"
export CUDA_VISIBLE_DEVICES=$SGE_GPU
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

BASE=/scratch_net/ofsoundof/yawli/hashnets_experiments
mkdir -p ${BASE}
TMPDIR=$(mktemp -d -p ${BASE} )
echo $TMPDIR
cd ${TMPDIR}

BRANCH=$1
shift 1
git clone -b ${BRANCH} ~/Documents/hashnets/ .

TEST_DIR=/scratch_net/ofsoundof/yawli/BSD68/
python -u "$@" --sge_gpu $SGE_GPU 
echo 'Training finished!'
echo

shift 2
python -u carn_denoise.py "$@" --sge_gpu $SGE_GPU --test_dir $TEST_DIR --test_score_compute --test_runtime_compute
echo 'Testing for BSD68 finished!'
echo 'First experiment finished!'
echo
echo
echo
<<'COMMENT'

#$ -l h="biwirender0[5-9]|biwirender[1-9][0-9]"
