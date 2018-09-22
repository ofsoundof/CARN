# __author__ = 'yawli'
import glob
import imageio
import argparse
import tensorflow as tf
import os
import scipy.io as sio
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def read_images(filename_gt, filename_n):
    # image_gt = np.expand_dims(imageio.imread(filename_gt).astype(np.float32), 2)
    mat_gt = sio.loadmat(filename_gt, struct_as_record=False)
    image_gt = np.expand_dims(mat_gt['img_o'].astype(np.float32), 2)
    mat_n = sio.loadmat(filename_n, struct_as_record=False)
    image_n = np.expand_dims(mat_n['img_n'].astype(np.float32), 2)
    # image_n = np.expand_dims(imageio.imread(filename_n), 2)
    shape = image_n.shape
    return image_gt, image_n, shape

def write_to_tfrecords(crop_gt, crop_n, dim, sigma, writer):
    crop_gt_raw = crop_gt.tobytes()
    crop_n_raw = crop_n.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_gt': _bytes_feature(crop_gt_raw),
        'image_n': _bytes_feature(crop_n_raw),
        'crop_dim': _int64_feature(dim),
        'sigma': _int64_feature(sigma)}))
    writer.write(example.SerializeToString())

def main():
    sigma = FLAGS.sigma
    dim = FLAGS.crop_dim
    overlap = FLAGS.crop_overlap
    image_paths_gt = sorted(glob.glob('/scratch_net/ofsoundof/yawli/Train400/GT/*.mat'))
    image_paths_n = sorted(glob.glob('/scratch_net/ofsoundof/yawli/Train400/Sig' + str(sigma) + '_diff/*.mat'))
    filename = '/scratch_net/ofsoundof/yawli/Train400/Sig' + str(sigma) + '_diff.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(image_paths_gt)):
        image_gt, image_n, shape = read_images(image_paths_gt[i], image_paths_n[i])
        #from IPython import embed; embed(); #exit()
        # print(image_paths_gt[i],image_paths_n[i], image_paths_high[i])
        # from IPython import embed; embed(); exit()
        for y in range(0, shape[0] - dim + 1, overlap):
            for x in range(0, shape[0] - dim + 1, overlap):
                crop_gt = image_gt[y: y + dim, x: x + dim, 0]
                crop_n = image_n[y: y + dim, x: x + dim, 0]
                # from IPython import embed; embed(); exit()
                # imageio.imwrite(os.path.dirname(filename) + '/i{}_y{}_x{}_low.png'.format(i, y, x), np.squeeze(crop_low))
                # imageio.imwrite(os.path.dirname(filename) + '/i{}_y{}_x{}_mid.png'.format(i, y, x), np.squeeze(crop_mid))
                # imageio.imwrite(os.path.dirname(filename) + '/i{}_y{}_x{}_high.png'.format(i, y, x), np.squeeze(crop_high))
                write_to_tfrecords(crop_gt, crop_n, dim, sigma, writer)
                # from IPython import embed; embed(); exit()
    writer.close()

# def read_images(filename_gt, filename_n):
#     # image_gt = np.expand_dims(imageio.imread(filename_gt).astype(np.float32), 2)
#     mat_gt = sio.loadmat(filename_gt, struct_as_record=False)
#     image_gt = np.expand_dims(mat_gt['img_o'].astype(np.float32), 2)
#     mat_n = sio.loadmat(filename_n, struct_as_record=False)
#     image_n = np.expand_dims(mat_n['img_n'].astype(np.float32), 2)
#     image_gt = image_gt[:-1, :-1, :]
#     image_n = image_n[:-1, :-1, :]
#     # image_n = np.expand_dims(imageio.imread(filename_n), 2)
#     shape = image_n.shape
#     return image_gt, image_n, shape
#
# def write_to_tfrecords(image_gt, image_n, shape, sigma, writer):
#     image_gt_raw = image_gt.tobytes()
#     image_n_raw = image_n.tobytes()
#     example = tf.train.Example(features=tf.train.Features(feature={
#         'image_gt': _bytes_feature(image_gt_raw),
#         'image_n': _bytes_feature(image_n_raw),
#         'height': _int64_feature(shape[0]),
#         'width': _int64_feature(shape[1]),
#         'sigma': _int64_feature(sigma)}))
#     writer.write(example.SerializeToString())
#
# def main():
#     sigma = FLAGS.sigma
#     image_paths_gt = sorted(glob.glob('/scratch_net/ofsoundof/yawli/BSD68/GT/*.mat'))
#     image_paths_n = sorted(glob.glob('/scratch_net/ofsoundof/yawli/BSD68/Sig' + str(sigma) + '/*.mat'))
#     filename = '/scratch_net/ofsoundof/yawli/BSD68/Sig' + str(sigma) + '.tfrecords'
#     writer = tf.python_io.TFRecordWriter(filename)
#     for i in range(len(image_paths_gt)):
#         image_gt, image_n, shape = read_images(image_paths_gt[i], image_paths_n[i])
#         write_to_tfrecords(image_gt, image_n, shape, sigma, writer)
#         # from IPython import embed; embed(); exit()
#     writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sigma', type=int, default='15', help='Sigma')
    parser.add_argument('crop_dim', type=int, default='72', help='Cropping dimension')
    parser.add_argument('crop_overlap', type=int, default='8', help='Overlapping')
    FLAGS, unparsed = parser.parse_known_args()
    main()
# import matplotlib.pyplot as plt
# fig=plt.figure()
# fig.add_subplot(1,3,1)
# plt.imshow(np.squeeze(image_gt),cmap='gray')
# fig.add_subplot(1,3,2)
# plt.imshow(np.squeeze(image_n),cmap='gray')
# fig.add_subplot(1,3,3)
# plt.imshow(np.squeeze(image_n-image_gt), cmap='gray')
# plt.show()
#
# import matplotlib.pyplot as plt
# fig=plt.figure()
# fig.add_subplot(1,3,1)
# plt.imshow(np.squeeze(crop_gt),cmap='gray')
# fig.add_subplot(1,3,2)
# plt.imshow(np.squeeze(crop_n),cmap='gray')
# fig.add_subplot(1,3,3)
# plt.imshow(np.squeeze(crop_n-crop_gt), cmap='gray')
# plt.show()
