#Submit to GPU
#TAG=HASHNET5_5
#TAG=ARN_APLUS_ANCHORS_MORE_LAYERS32
#TAG=NOHASHNET32
#TAG=First_Kernel_64_Stride_2_Size_1
#TAG=Second_Kernel_16_Stride_2_Size_3
#TAG=test
#TAG=Arn_8_3_x3_imagenet
#TAG=hashnet_new_proposal_tuning_4layer4-4-4-10
#TAG=ARN_NEW_2
#TAG=ARN_norm_soft_256-3-3-k1-up3

#TAG=ARN_deepstack__k3_noneR-up3
#5-5-5-10
#TAG=Fourth_ARN_4
#arn_aplus_anchors_more_layers
#GLOBAL_REGRESSION_ANCHOR
#TF_H1=10
#TF_H2=10
#TF_LR=0.1
#TAG=VDSR
#TAG=ESPCN_Challenge_tanh
TAG=CARN_Challenge_GAN
TF_CROP_DIM=64
TF_UPSCALE=4
TF_LAYER=7
TF_CHANNEL=16
TF_ANCHOR=16
TF_MAXITER=600000
TF_MODEL=3
TF_FEATURE_LAYER=3
#CHECKPOINT=${TAG}
CHECKPOINT=${TAG}_UP${TF_UPSCALE}LM${TF_MODEL}ITER_${TF_MAXITER}
#CHECKPOINT=${TAG}_UP${TF_UPSCALE}M${TF_MODEL}F${TF_FEATURE_LAYER}ITER_${TF_MAXITER}
#CHECKPOINT=${TAG}_UP${TF_UPSCALE}M${TF_MODEL}ITER_${TF_MAXITER}

qsub -N $CHECKPOINT ./vpython_test_train.sh master hashnets_rgb.py --checkpoint $CHECKPOINT --max_iter $TF_MAXITER --summary_interval 1000 --checkpoint_interval 1000 --upscale $TF_UPSCALE --deep_layer $TF_LAYER --deep_channel $TF_CHANNEL --deep_anchor $TF_ANCHOR --deep_kernel 3 --model_selection $TF_MODEL --deep_feature_layer $TF_FEATURE_LAYER
#--dataset $TF_DATASET --upscale $TF_UPSCALE



<<'COMMENT'

TF_UPSCALE=4
TF_ANCHOR=16
TF_MAXITER=1000000
TF_MODEL=12
TF_FEATURE_LAYER=5

TF_LAYER=7
TF_CHANNEL=32
TF_CROP_DIM=64
CHECKPOINT=${TAG}_UP${TF_UPSCALE}L${TF_MODEL}ITER_${TF_MAXITER}


with open('check.txt', 'w') as txt_file:
	for i in range(84000, 189600, 200):
		txt_file.write("all_model_checkpoint_paths: \"/home/yawli/Documents/hashnets/logs/2018-05-04-21-36-45-ARN_deep_reproduce_noneR_noOutput_UP4L7C32A64M2ITER_200000/model-" + str(i) +"\"\n")
		
#import matplotlib.pyplot as plt
#plt.imshow(np.squeeze(sr_*255),cmap='gray')
#plt.show()
		
COMMENT
