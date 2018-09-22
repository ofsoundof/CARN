
TAG=CARN_DENOISE_Diff2

TF_SIGMA=25
TF_LAYER=7
TF_CHANNEL=16
TF_ANCHOR=16
TF_MAXITER=300000
TF_MODEL=3
TF_FEATURE_LAYER=3
CHECKPOINT=${TAG}_Sigma${TF_SIGMA}L${TF_LAYER}C${TF_CHANNEL}A${TF_ANCHOR}M${TF_MODEL}ITER_${TF_MAXITER}


#qsub -N $CHECKPOINT ./denoise.sh master carn_denoise.py --checkpoint $CHECKPOINT --max_iter $TF_MAXITER --summary_interval 1000 --checkpoint_interval 1000 --sigma $TF_SIGMA --deep_layer $TF_LAYER --deep_channel $TF_CHANNEL --deep_anchor $TF_ANCHOR --deep_kernel 3 --deep_feature_layer $TF_FEATURE_LAYER --model_selection $TF_MODEL 

#TF_SIGMA=75
#CHECKPOINT=${TAG}_Sigma${TF_SIGMA}L${TF_LAYER}C${TF_CHANNEL}A${TF_ANCHOR}M${TF_MODEL}ITER_${TF_MAXITER}
#qsub -N $CHECKPOINT ./denoise.sh master carn_denoise.py --checkpoint $CHECKPOINT --max_iter $TF_MAXITER --summary_interval 1000 --checkpoint_interval 1000 --sigma $TF_SIGMA --deep_layer $TF_LAYER --deep_channel $TF_CHANNEL --deep_anchor $TF_ANCHOR --deep_kernel 3 --deep_feature_layer $TF_FEATURE_LAYER --model_selection $TF_MODEL 

#TF_SIGMA=50
#CHECKPOINT=${TAG}_Sigma${TF_SIGMA}L${TF_LAYER}C${TF_CHANNEL}A${TF_ANCHOR}M${TF_MODEL}ITER_${TF_MAXITER}
#qsub -N $CHECKPOINT ./denoise.sh master carn_denoise.py --checkpoint $CHECKPOINT --max_iter $TF_MAXITER --summary_interval 1000 --checkpoint_interval 1000 --sigma $TF_SIGMA --deep_layer $TF_LAYER --deep_channel $TF_CHANNEL --deep_anchor $TF_ANCHOR --deep_kernel 3 --deep_feature_layer $TF_FEATURE_LAYER --model_selection $TF_MODEL 

TEST_DIR=/scratch_net/ofsoundof/yawli/BSD68/

TF_SIGMA=25
CHECKPOINT=/scratch_net/ofsoundof/yawli/logs/CARN_DENOISE_Diff2_Sigma25L7C16A16M3ITER_300000_2018-08-01-19-46-24
python carn_denoise.py --checkpoint $CHECKPOINT --deep_layer $TF_LAYER --deep_channel $TF_CHANNEL --deep_anchor $TF_ANCHOR --deep_kernel 3 --deep_feature_layer $TF_FEATURE_LAYER --model_selection $TF_MODEL --test_dir $TEST_DIR --sigma $TF_SIGMA --test_runtime_compute

TF_SIGMA=50
CHECKPOINT=/scratch_net/ofsoundof/yawli/logs/CARN_DENOISE_Diff2_Sigma50L7C16A16M3ITER_300000_2018-08-01-17-25-11
python carn_denoise.py --checkpoint $CHECKPOINT --deep_layer $TF_LAYER --deep_channel $TF_CHANNEL --deep_anchor $TF_ANCHOR --deep_kernel 3 --deep_feature_layer $TF_FEATURE_LAYER --model_selection $TF_MODEL --test_dir $TEST_DIR --sigma $TF_SIGMA --test_runtime_compute

TF_SIGMA=75
CHECKPOINT=/scratch_net/ofsoundof/yawli/logs/CARN_DENOISE_Diff2_Sigma75L7C16A16M3ITER_300000_2018-08-01-19-45-54
python carn_denoise.py --checkpoint $CHECKPOINT --deep_layer $TF_LAYER --deep_channel $TF_CHANNEL --deep_anchor $TF_ANCHOR --deep_kernel 3 --deep_feature_layer $TF_FEATURE_LAYER --model_selection $TF_MODEL --test_dir $TEST_DIR --sigma $TF_SIGMA --test_runtime_compute


