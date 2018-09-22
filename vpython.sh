#!/bin/bash
#$ -l gpu=1
#$ -l h_rt=24:00:00
#$ -l h_vmem=20G
#$ -l h="biwirender0[5-9]|biwirender[1-9][0-9]"
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

TEST_DIR=/home/yawli/Documents/hashnets/test
python -u "$@" --sge_gpu $SGE_GPU 
echo 'Training finished!'
echo
<<'COMMENT'
shift 2
python -u hashnets.py "$@" --sge_gpu $SGE_GPU --test_dir "$TEST_DIR/Set5" 
echo 'Testing for Set5 finished!'
echo
python -u hashnets.py "$@" --sge_gpu $SGE_GPU --test_dir "$TEST_DIR/Set14"
echo 'Testing for Set14 finished!'
echo 'First experiment finished!'
echo
echo
echo

TEST_DIR=/home/yawli/Documents/hashnets/test
python -u "$@" --sge_gpu $SGE_GPU --regression 10 --crop_dim 64 --upscale 4 --dataset 'div2k'
echo 'Training finished!'
echo
python -u hashnets.py --test_dir "$TEST_DIR/Set5" --sge_gpu $SGE_GPU --regression 10 --crop_dim 64 --upscale 4 --dataset 'div2k'
echo 'Testing for Set5 finished!'
echo
python -u hashnets.py --test_dir "$TEST_DIR/Set14" --sge_gpu $SGE_GPU --regression 10 --crop_dim 64 --upscale 4 --dataset 'div2k'
echo 'Testing for Set14 finished!'
echo 'Second experiment finished!'
echo
echo
echo

TEST_DIR=/home/yawli/Documents/hashnets/test
python -u "$@" --sge_gpu $SGE_GPU --regression 10 --crop_dim 64 --upscale 2 --dataset 'div2k'
echo 'Training finished!'
echo
python -u hashnets.py --test_dir "$TEST_DIR/Set5" --sge_gpu $SGE_GPU --regression 10 --crop_dim 64 --upscale 2 --dataset 'div2k'
echo 'Testing for Set5 finished!'
echo
python -u hashnets.py --test_dir "$TEST_DIR/Set14" --sge_gpu $SGE_GPU --regression 10 --crop_dim 64 --upscale 2 --dataset 'div2k'
echo 'Testing for Set14 finished!'
echo 'Third experiment finished!'
echo
echo
echo

TEST_DIR=/home/yawli/Documents/hashnets/test
python -u "$@" --sge_gpu $SGE_GPU --regression 10 --crop_dim 64 --upscale 2 --dataset 'standard91'
echo 'Training finished!'
echo
python -u hashnets.py --test_dir "$TEST_DIR/Set5" --sge_gpu $SGE_GPU --regression 10 --crop_dim 64 --upscale 2 --dataset 'standard91'
echo 'Testing for Set5 finished!'
echo
python -u hashnets.py --test_dir "$TEST_DIR/Set14" --sge_gpu $SGE_GPU --regression 10 --crop_dim 64 --upscale 2 --dataset 'standard91'
echo 'Testing for Set14 finished!'
echo 'Fourth experiment finished!'
echo
echo
echo
COMMENT
# -l h="biwirender0[5-9]|biwirender[1-9][0-9]"

