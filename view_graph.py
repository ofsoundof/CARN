__author__ = 'yawli'


import datetime
import time
import argparse
import sys
from model import *
from utils import *
import os
import glob
MAX_RGB = 1.0
mean_RGB = tf.constant([123.680, 116.779, 103.939], tf.float32)
mean_Y = 0 #111.6804
DATASETS = {'div2k': load_div2k, 'standard91': load_standard91}

def run_training():

    with tf.device('/cpu:0'):
        crop_dim = FLAGS.crop_dim
        num_crops = FLAGS.num_crop
        upscale = FLAGS.upscale
        image_paths_low = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/DIV2K_train_HR/low_x' + str(upscale) + '/*.png'), dtype=tf.string)
        image_paths_mid = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/DIV2K_train_HR/mid_x' + str(upscale) + '/*.png'), dtype=tf.string)
        image_paths_high = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/DIV2K_train_HR/high_x' + str(upscale) + '/*.png'), dtype=tf.string)
        image_list = glob.glob('/home/yawli/Documents/hashnets/DIV2K_train_HR/low_x' + str(upscale) + '/*.png')
        image_list_low = [os.path.join('/home/yawli/Documents/hashnets/DIV2K_train_HR', os.path.splitext(os.path.basename(i))[0] + '_LR.png') for i in image_list]
        image_list_mid = [os.path.join('/home/yawli/Documents/hashnets/DIV2K_train_HR', os.path.splitext(os.path.basename(i))[0] + '_MR.png') for i in image_list]
        image_list_high = [os.path.join('/home/yawli/Documents/hashnets/DIV2K_train_HR', os.path.splitext(os.path.basename(i))[0] + '_HR.png') for i in image_list]


        [image_queue_low, image_queue_mid, image_queue_high, image_out_low, image_out_mid, image_out_high] = \
            tf.train.slice_input_producer([image_paths_low, image_paths_mid, image_paths_high, image_list_low, image_list_mid, image_list_high], shuffle=True, capacity=FLAGS.q_capacity)

        image_content_low = tf.read_file(image_queue_low)
        image_decoded_low = tf.image.decode_png(image_content_low, channels=1)
        image_low = tf.squeeze(tf.cast(image_decoded_low, tf.float32))

        image_content_mid = tf.read_file(image_queue_mid)
        image_decoded_mid = tf.image.decode_png(image_content_mid, channels=1)
        image_mid = tf.squeeze(tf.cast(image_decoded_mid, tf.float32))

        image_content_high = tf.read_file(image_queue_high)
        image_decoded_high = tf.image.decode_png(image_content_high, channels=1)
        image_high = tf.squeeze(tf.cast(image_decoded_high, tf.float32))

        size_low = tf.cast(tf.shape(image_low), tf.float32)
        offset_h = tf.cast(tf.floor(tf.random_uniform([num_crops, 1], 0.0, size_low[0] - crop_dim/upscale)), dtype=tf.int32)
        offset_w = tf.cast(tf.floor(tf.random_uniform([num_crops, 1], 0.0, size_low[1] - crop_dim/upscale)), dtype=tf.int32)

        def index_gen(offset_seed_h, offset_seed_w, dim):
            mask_low = tf.ones([dim, dim], dtype=tf.int32)
            index_offset_h = tf.reshape(kf.utils.kronecker_product(offset_seed_h, mask_low), [num_crops, dim, dim, 1])
            index_offset_w = tf.reshape(kf.utils.kronecker_product(offset_seed_w, mask_low), [num_crops, dim, dim, 1])
            index_offset = tf.concat([index_offset_h, index_offset_w], axis=3)
            index_base_range = range(1, dim + 1)
            index_base_w, index_base_h = tf.meshgrid(index_base_range, index_base_range)
            index_base = tf.tile(tf.expand_dims(tf.stack([index_base_h, index_base_w], axis=2), axis=0), [num_crops, 1, 1, 1])
            index = index_base + index_offset
            return index

        index_low = index_gen(offset_h, offset_w, crop_dim/upscale)
        index_high = index_gen(offset_h*upscale, offset_w*upscale, crop_dim)
        batch_input_low = tf.gather_nd(image_low, index_low)
        batch_input_mid = tf.gather_nd(image_mid, index_high)
        batch_input_high = tf.gather_nd(image_high, index_high)
        batch_input_low.set_shape([num_crops, crop_dim/upscale, crop_dim/upscale])
        batch_input_mid.set_shape([num_crops, crop_dim, crop_dim])
        batch_input_high.set_shape([num_crops, crop_dim, crop_dim])
        batch_input_low = tf.squeeze(tf.expand_dims(batch_input_low, axis=3), axis=0)
        batch_input_mid = tf.squeeze(tf.expand_dims(batch_input_mid, axis=3), axis=0)
        batch_input_high = tf.squeeze(tf.expand_dims(batch_input_high, axis=3), axis=0)

        low = tf.cast(batch_input_low, tf.uint8)
        low_encoded = tf.image.encode_png(low)
        save_op_low = tf.write_file(image_out_low, low_encoded)
        mid = tf.cast(batch_input_mid, tf.uint8)
        mid_encoded = tf.image.encode_png(mid)
        save_op_mid = tf.write_file(image_out_mid, mid_encoded)
        high = tf.cast(batch_input_high, tf.uint8)
        high_encoded = tf.image.encode_png(high)
        save_op_high = tf.write_file(image_out_high, high_encoded)

    #Create the sess, and use some options for better using gpu
    print("Create session")
    sess = tf.Session()

    print("Create checkpoint directory")
    FLAGS.checkpoint = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '-' + FLAGS.checkpoint
    with open('checkpoint.txt', 'w') as text_file: #save at the current directory, used for testing
        text_file.write(FLAGS.checkpoint)
    LOG_DIR = os.path.join('/home/yawli/Documents/hashnets/logs', FLAGS.checkpoint )
    assert (not os.path.exists(LOG_DIR)), 'LOG_DIR %s already exists'%LOG_DIR

    print("Create summary file writer")
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
    train_writer.close()

    print("Initialization")
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer()) #required by tf.train.match_filenames_once
    print("Create coordinator")
    coord = tf.train.Coordinator()
    print("Start queue runner")
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(50):
        sess.run([save_op_low,save_op_mid,save_op_high])
        # image_low, image_mid, image_high = sess.run([batch_input_low, batch_input_mid, batch_input_high])
        # import matplotlib.pyplot as plt
        # # plt.imshow(im)
        # from IPython import embed; embed(); exit()
    print("Start iteration")
    print("Request queue stop")
    coord.request_stop()
    coord.join(threads)
    print("Queue stops!")

def main(_):
    run_training()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--max_iter', type=int, default=100000, help='Number of iter to run trainer.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--dataset', type=str, default='div2k', help='Dataset for experiment')
    parser.add_argument('--checkpoint', type=str, default='', help='Unique checkpoint name')
    parser.add_argument('--test_dir', type=str, default='', help='Test directory')
    parser.add_argument('--summary_interval', type=int, default=1, help='Summary interval')
    parser.add_argument('--checkpoint_interval', type=int, default=10000, help='Checkpoint interval')
    parser.add_argument('--upscale', type=int, default=2, help='Upscaling factor')
    parser.add_argument('--regression', type=int, default=1, help='Number of regressors')
    parser.add_argument('--sge_gpu_all', type=str, default='0', help='All available gpus')
    parser.add_argument('--sge_gpu', type=str, default='0', help='Currently used gpu')
    parser.add_argument("--debug", type="bool", nargs="?", const=True, default=False, help="Use debugger to track down bad values during training")
    parser.add_argument('--q_capacity', type=int, default=20, help='Queue capacity')
    parser.add_argument('--q_min', type=int, default=5, help='Minimum number of elements after dequeue')
    parser.add_argument('--crop_dim', type=int, default=800, help='Crop dimension')
    parser.add_argument('--num_crop', type=int, default=1, help='Number of crops per image')
    FLAGS, unparsed = parser.parse_known_args()
    print("FLAGS: {}".format(FLAGS))
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

num_crops = 10
crop_dim = 10
upscale = 2
offset_h = tf.cast(tf.floor(tf.random_uniform([num_crops, 1], 0.0, 30 - crop_dim/upscale)), dtype=tf.int32)
offset_w = tf.cast(tf.floor(tf.random_uniform([num_crops, 1], 0.0, 60 - crop_dim/upscale)), dtype=tf.int32)

def index_gen(offset_seed_h, offset_seed_w, dim):
    mask = tf.ones([dim, dim], dtype=tf.int32)
    index_offset_h = tf.reshape(kf.utils.kronecker_product(offset_seed_h, mask), [num_crops, dim, dim, 1])
    index_offset_w = tf.reshape(kf.utils.kronecker_product(offset_seed_w, mask), [num_crops, dim, dim, 1])
    index_offset = tf.concat([index_offset_h, index_offset_w], axis=3)
    index_base_range = range(1, dim + 1)
    index_base_w, index_base_h = tf.meshgrid(index_base_range, index_base_range)
    index_base = tf.tile(tf.expand_dims(tf.stack([index_base_h, index_base_w], axis=2), axis=0), [num_crops, 1, 1, 1])
    index = index_base + index_offset
    ones = tf.ones([num_crops, dim, dim, 1], dtype=tf.int32)
    index = tf.expand_dims(tf.concat([index, ones], axis=3), axis=3)
    return index
