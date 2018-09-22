# __author__ = 'yawli'
import glob
import imageio
import argparse
import tensorflow as tf
import os
import numpy as np

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def read_images(filename_low, filename_mid, filename_high):
    image_low = np.expand_dims(imageio.imread(filename_low), 2)
    image_mid = np.expand_dims(imageio.imread(filename_mid), 2)
    image_high = np.expand_dims(imageio.imread(filename_high), 2)
    shape = image_high.shape
    return image_low, image_mid, image_high, shape

def write_to_tfrecords(crop_low, crop_mid, crop_high, dim, upscale, writer):
    crop_low_raw = crop_low.tobytes()
    crop_mid_raw = crop_mid.tobytes()
    crop_high_raw = crop_high.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_low': _bytes_feature(crop_low_raw),
        'image_mid': _bytes_feature(crop_mid_raw),
        'image_high': _bytes_feature(crop_high_raw),
        'crop_dim': _int64_feature(dim),
        'upscale': _int64_feature(upscale)}))
    writer.write(example.SerializeToString())

def main():
    upscale = FLAGS.upscale
    dim = FLAGS.crop_dim
    image_paths_low = sorted(glob.glob('/home/yawli/Documents/hashnets/DIV2K_train_HR/low_x' + str(upscale) + '/*.png'))
    image_paths_mid = sorted(glob.glob('/home/yawli/Documents/hashnets/DIV2K_train_HR/mid_x' + str(upscale) + '/*.png'))
    image_paths_high = sorted(glob.glob('/home/yawli/Documents/hashnets/DIV2K_train_HR/high_x' + str(upscale) + '/*.png'))
    filename = '/home/yawli/Documents/hashnets/DIV2K_train_HR/x' + str(upscale) + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(image_paths_low)):
        image_low, image_mid, image_high, shape = read_images(image_paths_low[i], image_paths_mid[i], image_paths_high[i])
        # print(image_paths_low[i],image_paths_mid[i], image_paths_high[i])
        # from IPython import embed; embed(); exit()
        hb = shape[0]//dim
        wb = shape[1]//dim
        dim_low = dim/upscale
        for y in range(hb):
            for x in range(wb):
                crop_low = image_low[y*dim_low:(y+1)*dim_low, x*dim_low:(x+1)*dim_low, 0]
                crop_mid = image_mid[y*dim:(y+1)*dim, x*dim:(x+1)*dim, 0]
                crop_high = image_high[y*dim:(y+1)*dim, x*dim:(x+1)*dim, 0]
                # imageio.imwrite(os.path.dirname(filename) + '/i{}_y{}_x{}_low.png'.format(i, y, x), np.squeeze(crop_low))
                # imageio.imwrite(os.path.dirname(filename) + '/i{}_y{}_x{}_mid.png'.format(i, y, x), np.squeeze(crop_mid))
                # imageio.imwrite(os.path.dirname(filename) + '/i{}_y{}_x{}_high.png'.format(i, y, x), np.squeeze(crop_high))
                write_to_tfrecords(crop_low, crop_mid, crop_high, dim, upscale, writer)
        # from IPython import embed; embed(); exit()
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('upscale', type=int, default='2', help='Upscaling factor')
    parser.add_argument('crop_dim', type=int, default='64', help='Cropping dimension')
    FLAGS, unparsed = parser.parse_known_args()
    main()

