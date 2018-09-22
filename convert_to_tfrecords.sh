#!/bin/bash

python convert_tfrecords_denoise.py --sigma 75 --crop_dim 72 --crop_overlap 8
#python convert_tfrecords_denoise.py --sigma 25 --crop_dim 72 --crop_overlap 8
#python convert_tfrecords_denoise.py --sigma 50 --crop_dim 72 --crop_overlap 8

#python convert_tfrecords.py --upscale 2 --crop_dim 64
#python convert_tfrecords.py --upscale 3 --crop_dim 66
#python convert_tfrecords.py --upscale 4 --crop_dim 64


