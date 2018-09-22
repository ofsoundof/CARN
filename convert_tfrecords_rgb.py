__author__ = 'yawli'
# __author__ = 'yawli'
import glob
import argparse
from scipy import misc
import tensorflow as tf
import os
import numpy as np
# import imageio

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def read_interp_images(filename, upscale):
    image_high = misc.imread(filename)
    image_s = image_high.shape
    height = image_s[0]//(upscale * 2) * upscale * 2
    width = image_s[1]//(upscale * 2) * upscale * 2
    image_high = image_high[:height, :width, :]#.astype(np.float32)
    image_low = misc.imresize(image_high, 1.0/upscale, interp='bicubic')#.astype(np.float32)
    image_mid = misc.imresize(image_low, np.float32(upscale), interp='bicubic')#.astype(np.float32)
    shape = image_high.shape
    return image_low, image_mid, image_high, shape

# def write_to_tfrecords(crop_low, crop_mid, crop_high, dim, upscale, writer):
#     crop_low_raw = crop_low.tobytes()
#     crop_mid_raw = crop_mid.tobytes()
#     crop_high_raw = crop_high.tobytes()
#     example = tf.train.Example(features=tf.train.Features(feature={
#         'image_low': _bytes_feature(crop_low_raw),
#         'image_mid': _bytes_feature(crop_mid_raw),
#         'image_high': _bytes_feature(crop_high_raw),
#         'crop_dim': _int64_feature(dim),
#         'upscale': _int64_feature(upscale)}))
#     writer.write(example.SerializeToString())
#
# def main():
#     upscale = FLAGS.upscale
#     dim = FLAGS.crop_dim
#     image_path = sorted(glob.glob(('/scratch_net/ofsoundof/yawli/DIV2K_train_HR/GT/*.png')))
#     # image_paths_low = sorted(glob.glob('/home/yawli/Documents/hashnets/DIV2K_train_HR/low_x' + str(upscale) + '/*.png'))
#     # image_paths_mid = sorted(glob.glob('/home/yawli/Documents/hashnets/DIV2K_train_HR/mid_x' + str(upscale) + '/*.png'))
#     # image_paths_high = sorted(glob.glob('/home/yawli/Documents/hashnets/DIV2K_train_HR/high_x' + str(upscale) + '/*.png'))
#     out_dir = '/scratch_net/ofsoundof/yawli/DIV2K_train_HR/'
#     filename = out_dir + 'x' + str(upscale) + '.tfrecords'
#     writer = tf.python_io.TFRecordWriter(filename)
#     for i in range(len(image_path)):
#         image_low, image_mid, image_high, shape = read_interp_images(image_path[i], upscale)
#         print(image_path[i])
#         # from IPython import embed; embed(); exit()
#         hb = shape[0]//dim
#         wb = shape[1]//dim
#         dim_low = dim/upscale
#         for y in range(hb):
#             for x in range(wb):
#                 crop_low = image_low[y*dim_low:(y+1)*dim_low, x*dim_low:(x+1)*dim_low, :]
#                 crop_mid = image_mid[y*dim:(y+1)*dim, x*dim:(x+1)*dim, :]
#                 crop_high = image_high[y*dim:(y+1)*dim, x*dim:(x+1)*dim, :]
#                 # imageio.imwrite(os.path.dirname(filename) + '/i{}_y{}_x{}_low.png'.format(i, y, x), np.squeeze(crop_low))
#                 # imageio.imwrite(os.path.dirname(filename) + '/i{}_y{}_x{}_mid.png'.format(i, y, x), np.squeeze(crop_mid))
#                 # imageio.imwrite(os.path.dirname(filename) + '/i{}_y{}_x{}_high.png'.format(i, y, x), np.squeeze(crop_high))
#                 write_to_tfrecords(crop_low, crop_mid, crop_high, dim, upscale, writer)
#         # import matplotlib.pyplot as plt
#         # fig=plt.figure()
#         # fig.add_subplot(1,2,1)
#         # plt.imshow(np.uint8(crop_high))
#         # fig.add_subplot(1,2,2)
#         # plt.imshow(np.uint8(crop_high[:,:,0]-crop_mid[:,:,0]-np.min(crop_high[:,:,0]-crop_mid[:,:,0])),cmap='gray')
#         # plt.show()
#         # from IPython import embed; embed();
#     writer.close()

def write_to_tfrecords(crop_low, crop_mid, crop_high, upscale, shape, writer):
    crop_low_raw = crop_low.tobytes()
    crop_mid_raw = crop_mid.tobytes()
    crop_high_raw = crop_high.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_low': _bytes_feature(crop_low_raw),
        'image_mid': _bytes_feature(crop_mid_raw),
        'image_high': _bytes_feature(crop_high_raw),
        'height': _int64_feature(shape[0]),
        'width': _int64_feature(shape[1]),
        'upscale': _int64_feature(upscale)}))
    writer.write(example.SerializeToString())

def main():
    upscale = FLAGS.upscale
    image_path = sorted(glob.glob(('/scratch_net/ofsoundof/yawli/DIV2K_valid_HR/GT/*.png')))
    # image_paths_low = sorted(glob.glob('/home/yawli/Documents/hashnets/DIV2K_train_HR/low_x' + str(upscale) + '/*.png'))
    # image_paths_mid = sorted(glob.glob('/home/yawli/Documents/hashnets/DIV2K_train_HR/mid_x' + str(upscale) + '/*.png'))
    # image_paths_high = sorted(glob.glob('/home/yawli/Documents/hashnets/DIV2K_train_HR/high_x' + str(upscale) + '/*.png'))
    out_dir = '/scratch_net/ofsoundof/yawli/DIV2K_valid_HR/'
    filename = out_dir + 'x' + str(upscale) + '_8.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(image_path)):
        image_low, image_mid, image_high, shape = read_interp_images(image_path[i], upscale)
        print(image_path[i])
        write_to_tfrecords(image_low, image_mid, image_high, upscale, shape, writer)
        # import matplotlib.pyplot as plt
        # fig=plt.figure()
        # fig.add_subplot(1,2,1)
        # plt.imshow(np.uint8(crop_high))
        # fig.add_subplot(1,2,2)
        # plt.imshow(np.uint8(crop_high[:,:,0]-crop_mid[:,:,0]-np.min(crop_high[:,:,0]-crop_mid[:,:,0])),cmap='gray')
        # plt.show()
        # from IPython import embed; embed();
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('upscale', type=int, default='4', help='Upscaling factor')
    FLAGS, unparsed = parser.parse_known_args()
    main()

# import matplotlib.pyplot as plt
# fig=plt.figure()
# fig.add_subplot(1,3,1)
# plt.imshow(np.squeeze(image_low))
# fig.add_subplot(1,3,2)
# plt.imshow(np.squeeze(image_mid))
# fig.add_subplot(1,3,3)
# plt.imshow(np.squeeze(image_high))
# plt.show()