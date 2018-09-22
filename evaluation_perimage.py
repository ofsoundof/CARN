#!/usr/bin/env python
import sys
import os
import os.path
import random
import numpy as np

from PIL import Image
import scipy.misc
#from skimage.measure import structural_similarity as ssim
from myssim import compare_ssim as ssim


# as per the metadata file, input and output directories are the arguments
#[_, res_dir, ref_dir, scale] = sys.argv

#print("REF DIR")
#print(ref_dir)

#SCALE = int(scale)

def output_psnr_mse(img_orig, img_out):
    #compute psnr
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def _open_img(img_p, scale):
    #open image for the computation of psnr
    F = scipy.misc.fromimage(Image.open(img_p)).astype(float)/255.0
    boundarypixels = scale
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels]
    return F


def _open_img_ssim(img_p, scale):
    #open image for the computation of ssim
    F = scipy.misc.fromimage(Image.open(img_p))#.astype(float)
    boundarypixels = scale
    F = F[boundarypixels:-boundarypixels,boundarypixels:-boundarypixels]
    return F


def compute_psnr(ref_im, res_im, scale):
    #read image and compute psnr
    return output_psnr_mse(
        _open_img(os.path.join(ref_im), scale),
        _open_img(os.path.join(res_im), scale)
        )


def compute_mssim(ref_im, res_im, scale):
    #read image and compute ssim
    #there is only one channel, namely, the Y channel
    ref_img = _open_img_ssim(os.path.join(ref_im), scale)
    res_img = _open_img_ssim(os.path.join(res_im), scale)
    #channels = []
    #for i in range(3):
    #     channels.append(ssim(ref_img[:,:,i],res_img[:,:,i],
    #     gaussian_weights=True, use_sample_covariance=False))
    # return np.mean(channels)
    return ssim(ref_img, res_img, gaussian_weights=True, use_sample_covariance=False)


def compute_psnr_mssim(ref_pngs, res_pngs, scale):
    #ref_pngs = sorted([p for p in os.listdir(ref_dir) if p.lower().endswith('png')])
    #res_pngs = sorted([p for p in os.listdir(res_dir) if p.lower().endswith('png')])


    print("\nComputing PSNR")
    scores_psnr = []
    for (ref_im, res_im) in zip(ref_pngs, res_pngs):
        #print(ref_im,res_im)
        #sys.stdout.write('.'); sys.stdout.flush();
        if os.path.basename(ref_im) == 'bridge.png':
            image_ref = _open_img(os.path.join(ref_im), scale)*255
            image_res = _open_img(os.path.join(res_im), scale)*255
            image_ref = (image_ref-16)/219.859*256/255
            image_res = (image_res-16)/219.859*256/255
            scores_psnr.append(output_psnr_mse(image_ref, image_res))
        else:
            scores_psnr.append(
                compute_psnr(ref_im,res_im, scale)
            )
        print(scores_psnr[-1])
    psnr = np.mean(scores_psnr)

    print("\nComputing SSIM")
    scores_ssim = []
    for (ref_im, res_im) in zip(ref_pngs, res_pngs):
        #print(ref_im,res_im)
        #sys.stdout.write('.'); sys.stdout.flush();
        if os.path.basename(ref_im) == 'bridge.png':
            image_ref = _open_img_ssim(os.path.join(ref_im), scale).astype(float)
            image_res = _open_img_ssim(os.path.join(res_im), scale).astype(float)
            image_ref = (image_ref-16)/219.859*256
            image_res = (image_res-16)/219.859*256
            scores_ssim.append(ssim(image_ref, image_res, gaussian_weights=True, use_sample_covariance=False))
        else:
            scores_ssim.append(
                compute_mssim(ref_im,res_im, scale)
            )
        print(scores_ssim[-1])
    mssim = np.mean(scores_ssim)

    print("\nAverage PSNR:%f"%psnr)
    print("Average SSIM:%f"%mssim)

    scores_psnr.append(psnr)
    scores_ssim.append(mssim)
    scores = np.array([scores_psnr, scores_ssim])

    return scores

if __name__ == '__main__':
    [_, res_dir, ref_dir, scale] = sys.argv
    scale = int(scale)
    compute_psnr_mssim(ref_dir, res_dir, scale)

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--ref_dir',
    #     type=str,
    #     default='',
    #     help='Reference image (HR image) directory'
    # )
    # parser.add_argument(
    #     '--res_dir',
    #     type=str,
    #     default='',
    #     help='Resolved image (SR or interpolated image) directory'
    # )
    # parser.add_argument(
    #     '--scale',
    #     type=int,
    #     default=2,
    #     help='Upscale factor used to delete the borders of the images'
    # )
    # parsed, unparsed = parser.parse_known_args()

# the scores for the leaderboard must be in a file named "scores.txt"
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions

