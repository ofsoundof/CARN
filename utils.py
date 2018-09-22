__author__ = 'yawli'
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import numpy as np
import glob
from myssim import compare_ssim as ssim
from tensorflow.python.debug.lib import debug_data
from math import sqrt
import tensorflow.contrib.kfac as kf

def training(loss, FLAGS, global_step):

    # Create a variable to track the global step.

    # Create optimizer.
    # boundaries = [10000]
    # boundaries = [2000, 4000, 20000]
    # boundaries = [10000, 30000, 70000]
    boundaries = [100000, 200000, 300000, 400000]
    #boundaries = [200000, 300000, 350000]
    #boundaries = [150000, 200000, 250000]
    # boundaries = [60000, 120000, 180000]
    # boundaries = [200000, 400000]
    # boundaries = [4000, 8000, 12000, 16000]
    #values = [20.0, 10.0, 5.0, 2.5, 1.25]
    # values = [1.0, 0.5, 0.25, 0.125]
    # values = [0.1, 0.05, 0.025, 0.0125]
    # values = [0.001, 0.0001, 0.00005, 0.000025, 0.00001]
    #values = [0.0001, 0.00005, 0.000025, 0.00001]
    values = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    FLAGS.lr = tf.train.piecewise_constant(global_step, boundaries, values, name='learning_rate')
    # FLAGS.lr = tf.train.exponential_decay(learning_rate=FLAGS.lr, global_step=global_step, decay_steps=4000, decay_rate=0.9)
    # FLAGS.lr = 0.1
    #FLAGS.lr = tf.train.exponential_decay(FLAGS.lr, global_step, 100000, 0.96)#, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)

    # # # Use the optimizer to apply the gradients that minimize the loss
    # # # (and also increment the global step counter) as a single training step.
    # with tf.name_scope("compute_gradients"):
    # grad_var = optimizer.compute_gradients(loss)
    # # gradients = tf.identity(gradients, name="show_gradients") Error
    # grad_var_clip = [(tf.clip_by_norm(gv[0], 1), gv[1]) for gv in grad_var]
    # train_op = optimizer.apply_gradients(grads_and_vars=grad_var_clip, global_step=global_step, name='apply_gradients')
    #clip = 1 if FLAGS.lr == 0.001 else 10
    train_op = slim.learning.create_train_op(loss, optimizer, global_step, clip_gradient_norm=0, summarize_gradients=True)

    return train_op, global_step

def psnr_ssim(image_list_high, image_list_mid, image_list_sr, FLAGS, model):

    from evaluation_perimage import compute_psnr_mssim
    #name of the images
    image_name = [os.path.splitext(os.path.basename(p))[0] for p in image_list_high]
    image_name.append('average')
    #compute psnr/ssim for interpolated (mid-resolution) images
    scores_mid = compute_psnr_mssim(image_list_high, image_list_mid, FLAGS.upscale)
    #compute psnr/ssim for sr images
    scores_sr = compute_psnr_mssim(image_list_high, image_list_sr, FLAGS.upscale)
    scores_gain = scores_sr - scores_mid
    # checkpoint = os.path.join(FLAGS.checkpoint, 'psnr_ssim.csv')
    # with open(checkpoint, 'a') as csv_file:
    #     csv_file.write('PSNR(dB) for ' + os.path.basename(FLAGS.test_dir) + ', Scale ' + str(FLAGS.upscale) + ', ' +model + '\n')
    #     content = '{:<12}  {:<8}  {:<8}  {:<8}\n'.format('Image', 'Bicubic', 'SR', 'Gain')
    #     csv_file.write(content)
    #     for i in range(len(image_list_high)+1):
    #         content = '{:<12}  {:<8.4f}  {:<8.4f}  {:<8.4f}\n'.format(image_name[i], scores_mid[0][i], scores_sr[0][i], scores_gain[0][i])
    #         csv_file.write(content)
    #
    #     csv_file.write('SSIM for ' + os.path.basename(FLAGS.test_dir) + ', Scale ' + str(FLAGS.upscale) + ', ' +model + '\n')
    #     content = '{:<12}  {:<8}  {:<8}  {:<8}\n'.format('Image', 'Bicubic', 'SR', 'Gain')
    #     csv_file.write(content)
    #     for i in range(len(image_list_high)+1):
    #         content = '{:<12}   {:<7.4f}   {:<7.4f}  {:<8.4f}\n'.format(image_name[i], scores_mid[1][i], scores_sr[1][i], scores_gain[1][i])
    #         csv_file.write(content)
    #     csv_file.write('\n\n')
    return scores_mid, scores_sr, scores_gain

def comp_mse_psnr(net,target, MAX_RGB):
    MSE = tf.reduce_mean(tf.squared_difference(target,net))
    PSNR_from_MSE = lambda x: 10.0*tf.log(MAX_RGB**2/x)/tf.log(10.0)
    #log(x+y) >= log(x) + log(y), more data inside the log gives an upper bound
    PSNR = PSNR_from_MSE(MSE)
    return MSE, PSNR

# Functions for use Dataset
def _parse_train(example_proto):
    """
    Parse examples in training dataset from tfrecords
    """
    features = {"image_low": tf.FixedLenFeature((), tf.string),
              "image_mid": tf.FixedLenFeature((), tf.string),
              "image_high": tf.FixedLenFeature((), tf.string),
              "crop_dim": tf.FixedLenFeature((), tf.int64),
              "upscale": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    dim = tf.cast(parsed_features['crop_dim'], tf.int64) #whether this is needed?
    upscale = tf.cast(parsed_features['upscale'], tf.int64)

    # from IPython import embed; embed(); exit()
    image_low = tf.decode_raw(parsed_features['image_low'], tf.uint8)
    image_mid = tf.decode_raw(parsed_features['image_mid'], tf.uint8)
    image_high = tf.decode_raw(parsed_features['image_high'], tf.uint8)
    image_low = tf.cast(tf.reshape(image_low, tf.stack([dim/upscale, dim/upscale, 1])), tf.float32)
    image_mid = tf.cast(tf.reshape(image_mid, tf.stack([dim, dim, 1])), tf.float32)
    image_high = tf.cast(tf.reshape(image_high, tf.stack([dim, dim, 1])), tf.float32)
    decision = tf.random_uniform([2], 0, 1)
    image_low = random_flip(image_low, decision[0])
    image_mid = random_flip(image_mid, decision[0])
    image_high = random_flip(image_high, decision[0])
    return image_low, image_mid, image_high

def _parse_test(filename_low, filename_mid, filename_high):
    """
    Parse examples to form testing dataset.
    """
    image_low = tf.read_file(filename_low)
    image_low = tf.image.decode_png(image_low, channels=1)
    image_low = tf.expand_dims(tf.cast(image_low, tf.float32), axis=0)
    image_mid = tf.read_file(filename_mid)
    image_mid = tf.image.decode_png(image_mid, channels=1)
    image_mid = tf.expand_dims(tf.cast(image_mid, tf.float32), axis=0)
    image_high = tf.read_file(filename_high)
    image_high = tf.image.decode_png(image_high, channels=1)
    image_high = tf.expand_dims(tf.cast(image_high, tf.float32), axis=0)
    return image_low, image_mid, image_high

def train_dataset_prepare(FLAGS):
    """
    Prepare for training dataset
    """
    filenames = os.path.join(FLAGS.storage_path, 'Datasets/DIV2K_train_HR/x' + str(FLAGS.upscale) + '.tfrecords')
    train_dataset = tf.data.TFRecordDataset(filenames)
    train_dataset = train_dataset.shuffle(buffer_size=5000).repeat()  # Repeat the input indefinitely.
    # train_dataset = train_dataset.apply(tf.contrib.data.map_and_batch(map_func=_parse_train, batch_size=FLAGS.batch_size, num_parallel_batches=8))
    train_dataset = train_dataset.map(map_func=_parse_train, num_parallel_calls=20)  # Parse the record into tensors.
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.prefetch(256)
    return train_dataset

def test_dataset_prepare(path_fun, test_dir):
    """
    Prepare for test dataset
    """
    image_list_low, image_list_mid, image_list_high = path_fun(test_dir)
    test_dataset = tf.data.Dataset.from_tensor_slices((image_list_low, image_list_mid, image_list_high))
    test_dataset = test_dataset.map(map_func=_parse_test)
    return test_dataset

def test_path_prepare(test_dir):
    """
    Prepare for paths of test image
    """
    set5_test_dir_low = test_dir + '_lowY/'
    set5_test_dir_mid = test_dir + '_midY/'
    set5_test_dir_high = test_dir + '_highY/'
    image_list_low = sorted(glob.glob(set5_test_dir_low + '*.png'))
    image_list_mid = sorted(glob.glob(set5_test_dir_mid + '*.png'))
    image_list_high = sorted(glob.glob(set5_test_dir_high + '*.png'))

    return image_list_low, image_list_mid, image_list_high

#used to crop the boundaries, compute psnr and ssim
def image_converter(image_sr_array, image_mid_array, image_high_array, flag, boundarypixels, MAX_RGB):
    """
    make image conversions and return the subjective score, i.e., psnr and ssim of SR and bicubic image
    :param image_sr_array:
    :param image_mid_array:
    :param image_high_array:
    :param flag:
    :param boundarypixels:
    :return:
    """
    image_sr_array = image_sr_array[boundarypixels:-boundarypixels, boundarypixels:-boundarypixels]
    image_mid_array = image_mid_array[boundarypixels:-boundarypixels, boundarypixels:-boundarypixels]
    image_high_array = image_high_array[boundarypixels:-boundarypixels, boundarypixels:-boundarypixels]
    if flag:
        image_sr_array = (image_sr_array*255/MAX_RGB-16)/219.859*256/255*MAX_RGB
        image_mid_array = (image_mid_array*255/MAX_RGB-16)/219.859*256/255*MAX_RGB
        image_high_array = (image_high_array*255/MAX_RGB-16)/219.859*256/255*MAX_RGB
    psnr_sr = 10 * np.log10(MAX_RGB ** 2 / (np.mean(np.square(image_high_array - image_sr_array))))
    psnr_itp = 10 * np.log10(MAX_RGB ** 2 / (np.mean(np.square(image_high_array - image_mid_array))))
    # from IPython import embed; embed(); exit()
    ssim_sr = ssim(np.uint8(image_sr_array*255/MAX_RGB), np.uint8(image_high_array*255/MAX_RGB), gaussian_weights=True, use_sample_covariance=False)
    ssim_itp = ssim(np.uint8(image_mid_array*255/MAX_RGB), np.uint8(image_high_array*255/MAX_RGB), gaussian_weights=True, use_sample_covariance=False)
    score = [psnr_itp, psnr_sr, ssim_itp, ssim_sr]
    return score

#functions for file writer
def file_writer(filename, mode, content):
    with open(filename, mode) as opened_file:
        opened_file.write(content)

def content_generator_one_line(descriptor, header, header_fmt, written_content, written_content_fmt):
    content = descriptor + header_fmt.format(header)
    for i in range(len(written_content)):
        content = content + written_content_fmt.format(written_content[i])
    content += '\n'
    return content

def content_generator_mul_line(descriptor_mul, written_content_mul, written_content_fmt_mul, header_mul, header_fmt='{:<10}\t'):
    content = ''
    for l in range(len(header_mul)):
        #from IPython import embed; embed(); exit()
        content += content_generator_one_line(descriptor_mul[l], header_mul[l], header_fmt, written_content_mul[l, :], written_content_fmt_mul[l])
    return content






########################################################################################################################
# The following should be used according to the additional aims of the procedure or will not be used but show some
# possibilities of designing the algorithm

def training_alternate(loss, FLAGS):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    boundaries = [15000, 30000, 40000]
    # boundaries = [12000, 24000, 36000]
    values = [0.001, 0.0001, 0.00001, 0.000001]
    FLAGS.lr = tf.train.piecewise_constant(global_step, boundaries, values, name='learning_rate')
    optimizer_feature_regression = tf.train.MomentumOptimizer(learning_rate=FLAGS.lr, momentum=0.9)
    optimizer_similarity = tf.train.MomentumOptimizer(learning_rate=FLAGS.lr, momentum=0.9)
    feature_regression_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='feature') \
                              + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='regression')

    train_op_feature_regression = slim.learning.create_train_op(loss, optimizer_feature_regression, variables_to_train=feature_regression_vars, clip_gradient_norm=10, summarize_gradients=True)
    # control_dependencies, execute the following codes when the dependencies have been executed.
    with tf.control_dependencies([train_op_feature_regression]+ tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        similarity_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='similarity')
        train_op_similarity = slim.learning.create_train_op(loss, optimizer_similarity, variables_to_train=similarity_vars, clip_gradient_norm=10, summarize_gradients=True)
    return train_op_similarity, global_step

def training_separate(loss, FLAGS):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    boundaries = [20000, 40000, 60000]
    values = [0.001, 0.0001, 0.00001, 0.000001]
    FLAGS.lr = tf.train.piecewise_constant(global_step, boundaries, values, name='learning_rate')
    optimizer_feature_regression = tf.train.MomentumOptimizer(learning_rate=FLAGS.lr, momentum=0.9)
    optimizer_similarity = tf.train.MomentumOptimizer(learning_rate=FLAGS.lr, momentum=0.9)

    feature_regression_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='feature') \
                              + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='regression')
    train_op_feature_regression = slim.learning.create_train_op(loss, optimizer_feature_regression, variables_to_train=feature_regression_vars, clip_gradient_norm=10, summarize_gradients=True)

    similarity_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='similarity')
    train_op_similarity = slim.learning.create_train_op(loss, optimizer_similarity, variables_to_train=similarity_vars, clip_gradient_norm=10, summarize_gradients=True)
    return train_op_feature_regression, train_op_similarity, global_step

def random_flip(input_image, decision):
    f1 = tf.identity(input_image)
    f2 = tf.image.flip_left_right(input_image)
    output_image = tf.cond(tf.less(decision, 0.5), lambda: f2, lambda: f1)
    return output_image

# Load the input using queues.
def load_standard91(FLAGS):
    CROPPED_DIM = FLAGS.crop_dim
    # image_paths_low = tf.convert_to_tensor(tf.train.match_filenames_once('./standard91/low_x2/*.png'), dtype=tf.string)
    # image_paths_mid = tf.convert_to_tensor(tf.train.match_filenames_once('./standard91/mid_x2/*.png'), dtype=tf.string)
    # image_paths_high = tf.convert_to_tensor(tf.train.match_filenames_once('./standard91/high/*.png'), dtype=tf.string)
    # image_paths_low = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/ImageNetTiny/low_x2/*.png'), dtype=tf.string)
    # image_paths_mid = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/ImageNetTiny/mid_x2/*.png'), dtype=tf.string)
    # image_paths_high = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/ImageNetTiny/high_x2/*.png'), dtype=tf.string)
    # image_paths_low = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/DIV2K_train_HR/low_x2/*.png'), dtype=tf.string)
    # image_paths_mid = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/DIV2K_train_HR/mid_x2/*.png'), dtype=tf.string)
    # image_paths_high = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/DIV2K_train_HR/high_x2/*.png'), dtype=tf.string)
    # image_paths_low = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/standard91/low_x' + str(FLAGS.upscale) + '/*.png'), dtype=tf.string)
    # image_paths_mid = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/standard91/mid_x' + str(FLAGS.upscale) + '/*.png'), dtype=tf.string)
    # image_paths_high = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/standard91/high_x' + str(FLAGS.upscale) + '/*.png'), dtype=tf.string)
    image_paths_low = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/test/Set5_X' + str(FLAGS.upscale) + '_lowRGB/*.png'), dtype=tf.string)
    image_paths_mid = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/test/Set5_X' + str(FLAGS.upscale) + '_midRGB/*.png'), dtype=tf.string)
    image_paths_high = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/test/Set5_X' + str(FLAGS.upscale) + '_highRGB/*.png'), dtype=tf.string)

    [image_queue_low, image_queue_mid, image_queue_high] = \
        tf.train.slice_input_producer([image_paths_low, image_paths_mid, image_paths_high], shuffle=True, capacity=FLAGS.q_capacity)

    image_content_low = tf.read_file(image_queue_low)
    image_decoded_low = tf.image.decode_png(image_content_low, channels=3)
    image_low = tf.cast(image_decoded_low, tf.float32)

    image_content_mid = tf.read_file(image_queue_mid)
    image_decoded_mid = tf.image.decode_png(image_content_mid, channels=3)
    image_mid = tf.cast(image_decoded_mid, tf.float32)

    image_content_high = tf.read_file(image_queue_high)
    image_decoded_high = tf.image.decode_png(image_content_high, channels=3)
    image_high = tf.cast(image_decoded_high, tf.float32)

    size_low = tf.shape(image_low)
    offset_h = tf.cast(tf.floor(tf.random_uniform([], 0, tf.cast(size_low[0], tf.float32) - CROPPED_DIM/FLAGS.upscale)), dtype=tf.int32)
    offset_w = tf.cast(tf.floor(tf.random_uniform([], 0, tf.cast(size_low[1], tf.float32) - CROPPED_DIM/FLAGS.upscale)), dtype=tf.int32)
    image_crop_low = tf.image.crop_to_bounding_box(image_low, offset_h, offset_w, CROPPED_DIM/FLAGS.upscale, CROPPED_DIM/FLAGS.upscale)
    image_crop_mid = tf.image.crop_to_bounding_box(image_mid, offset_h*FLAGS.upscale, offset_w*FLAGS.upscale, CROPPED_DIM, CROPPED_DIM)
    image_crop_high = tf.image.crop_to_bounding_box(image_high, offset_h*FLAGS.upscale, offset_w*FLAGS.upscale, CROPPED_DIM, CROPPED_DIM)

    decision = tf.random_uniform([], 0, 1, dtype=tf.float32)
    image_flip_low = random_flip(image_crop_low, decision)
    image_flip_mid = random_flip(image_crop_mid, decision)
    image_flip_high = random_flip(image_crop_high, decision)

    image_flip_low.set_shape([CROPPED_DIM/FLAGS.upscale, CROPPED_DIM/FLAGS.upscale, 3])
    image_flip_mid.set_shape([CROPPED_DIM, CROPPED_DIM, 3])
    image_flip_high.set_shape([CROPPED_DIM, CROPPED_DIM, 3])

    [batch_low, batch_mid, batch_high] = tf.train.batch(
        [image_flip_low, image_flip_mid, image_flip_high],
        batch_size=FLAGS.batch_size,
        capacity=FLAGS.q_capacity,
        num_threads=8
    )
    return batch_low, batch_mid, batch_high

# Since the dimensions of DIV2K images are very large, it could be very inefficient to crop only one image patch from
# one image. Because in that case, most of the time will be spent on reading and decoding the images. Thus, multiple
# patches should be cropped from one image. The following three functions realize the function in different ways.

# Use for loop to crop multiple images and then combine them into a batch
def load_div2k_loop(FLAGS):
    CROPPED_DIM = 64
    num_crops = 100
    image_paths_low = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/DIV2K_train_HR/low_x2/*.png'), dtype=tf.string)
    image_paths_mid = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/DIV2K_train_HR/mid_x2/*.png'), dtype=tf.string)
    image_paths_high = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/DIV2K_train_HR/high_x2/*.png'), dtype=tf.string)
    # image_paths_low = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/standard91/low_x2/*.png'), dtype=tf.string)
    # image_paths_mid = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/standard91/mid_x2/*.png'), dtype=tf.string)
    # image_paths_high = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/standard91/high_x2/*.png'), dtype=tf.string)

    [image_queue_low, image_queue_mid, image_queue_high] = \
        tf.train.slice_input_producer([image_paths_low, image_paths_mid, image_paths_high], shuffle=True, capacity=FLAGS.q_capacity)

    image_content_low = tf.read_file(image_queue_low)
    image_decoded_low = tf.image.decode_png(image_content_low, channels=1)
    image_low = tf.cast(image_decoded_low, tf.float32)

    image_content_mid = tf.read_file(image_queue_mid)
    image_decoded_mid = tf.image.decode_png(image_content_mid, channels=1)
    image_mid = tf.cast(image_decoded_mid, tf.float32)

    image_content_high = tf.read_file(image_queue_high)
    image_decoded_high = tf.image.decode_png(image_content_high, channels=1)
    image_high = tf.cast(image_decoded_high, tf.float32)

    batch_input_low = []
    batch_input_mid = []
    batch_input_high = []
    size_low = tf.shape(image_low)
    for i in range(num_crops):
        offset_h = tf.cast(tf.floor(tf.random_uniform([], 0, tf.cast(size_low[0], tf.float32) - CROPPED_DIM/2)), dtype=tf.int32)
        offset_w = tf.cast(tf.floor(tf.random_uniform([], 0, tf.cast(size_low[1], tf.float32) - CROPPED_DIM/2)), dtype=tf.int32)
        image_flip_low = tf.image.crop_to_bounding_box(image_low, offset_h, offset_w, CROPPED_DIM/2, CROPPED_DIM/2)
        image_flip_mid = tf.image.crop_to_bounding_box(image_mid, offset_h*2, offset_w*2, CROPPED_DIM, CROPPED_DIM)
        image_flip_high = tf.image.crop_to_bounding_box(image_high, offset_h*2, offset_w*2, CROPPED_DIM, CROPPED_DIM)

        # decision = tf.random_uniform([], 0, 1, dtype=tf.float32)
        # image_flip_low = random_flip(image_crop_low, decision)
        # image_flip_mid = random_flip(image_crop_mid, decision)
        # image_flip_high = random_flip(image_crop_high, decision)

        image_flip_low.set_shape([CROPPED_DIM/2, CROPPED_DIM/2, 1])
        image_flip_mid.set_shape([CROPPED_DIM, CROPPED_DIM, 1])
        image_flip_high.set_shape([CROPPED_DIM, CROPPED_DIM, 1])

        batch_input_low.append(image_flip_low)
        batch_input_mid.append(image_flip_mid)
        batch_input_high.append(image_flip_high)

    batch_input_low = tf.stack(batch_input_low)
    batch_input_mid = tf.stack(batch_input_mid)
    batch_input_high = tf.stack(batch_input_high)
    [batch_low, batch_mid, batch_high] = tf.train.shuffle_batch(
        [batch_input_low, batch_input_mid, batch_input_high],
        batch_size=FLAGS.batch_size,
        capacity=FLAGS.q_capacity,
        min_after_dequeue=FLAGS.q_min,
        num_threads=4,
        enqueue_many=True
    )
    return batch_low, batch_mid, batch_high

# Use tf.gather_nd to gather multiple crops. And use numpy ops to create the indices for tf.gather_nd. But in this way,
# the dimensions of the images should be known in numpy format. Can it be done?
def load_div2k_numpy(FLAGS):
    """
    Use numpy to compute the indices. But the dimension of the image needs to be known, which is impossible in the queue enviroment.
    :param FLAGS:
    :return:
    """
    CROPPED_DIM = 32
    num_crops = 128
    image_paths_low = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/DIV2K_train_HR/low_x2/*.png'), dtype=tf.string)
    image_paths_mid = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/DIV2K_train_HR/mid_x2/*.png'), dtype=tf.string)
    image_paths_high = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/DIV2K_train_HR/high_x2/*.png'), dtype=tf.string)
    # image_paths_low = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/standard91/low_x2/*.png'), dtype=tf.string)
    # image_paths_mid = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/standard91/mid_x2/*.png'), dtype=tf.string)
    # image_paths_high = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/standard91/high_x2/*.png'), dtype=tf.string)

    [image_queue_low, image_queue_mid, image_queue_high] = \
        tf.train.slice_input_producer([image_paths_low, image_paths_mid, image_paths_high], shuffle=True, capacity=FLAGS.q_capacity)

    image_content_low = tf.read_file(image_queue_low)
    image_decoded_low = tf.image.decode_png(image_content_low, channels=1)
    image_low = tf.cast(image_decoded_low, tf.float32)

    image_content_mid = tf.read_file(image_queue_mid)
    image_decoded_mid = tf.image.decode_png(image_content_mid, channels=1)
    image_mid = tf.cast(image_decoded_mid, tf.float32)

    image_content_high = tf.read_file(image_queue_high)
    image_decoded_high = tf.image.decode_png(image_content_high, channels=1)
    image_high = tf.cast(image_decoded_high, tf.float32)

    # with tf.Session() as sess:
    #     size_low = sess.run(tf.shape(image_low))
    #Don't know the dimension of the image.
    offset_h = np.floor(np.random.uniform(0, 100 - CROPPED_DIM/2, num_crops))
    offset_w = np.floor(np.random.uniform(0, 100 - CROPPED_DIM/2, num_crops))
    indices_low = [[[[y + offset_h[num], x + offset_w[num]] for x in range(CROPPED_DIM/2)] for y in range(CROPPED_DIM/2)] for num in range(num_crops)]
    indices_high = [[[[y + offset_h[num]*2, x + offset_w[num]*2] for x in range(CROPPED_DIM)] for y in range(CROPPED_DIM)] for num in range(num_crops)]

    indices_low = tf.cast(indices_low, tf.int32)
    indices_high = tf.cast(indices_high, tf.int32)
    batch_input_low = tf.gather_nd(image_low, indices_low)
    batch_input_mid = tf.gather_nd(image_mid, indices_high)
    batch_input_high = tf.gather_nd(image_high, indices_high)

    [batch_low, batch_mid, batch_high] = tf.train.batch(
        [batch_input_low, batch_input_mid, batch_input_high],
        batch_size=FLAGS.batch_size,
        capacity=FLAGS.q_capacity,
        num_threads=4,
        enqueue_many=True
    )
    return batch_low, batch_mid, batch_high

# Use tensorflow ops to create the indices for tf.gather_nd. However, even in this way, considerable amount of time is
# spent on reading, decoding images, and tf.gather_nd operation. Use Dataset will make the input pipeline much faster.
def load_div2k(FLAGS):
    crop_dim = FLAGS.crop_dim
    num_crops = FLAGS.num_crop
    upscale = FLAGS.upscale
    image_paths_low = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/DIV2K_train_HR/low_x' + str(upscale) + '/*.png'), dtype=tf.string)
    image_paths_mid = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/DIV2K_train_HR/mid_x' + str(upscale) + '/*.png'), dtype=tf.string)
    image_paths_high = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/DIV2K_train_HR/high_x' + str(upscale) + '/*.png'), dtype=tf.string)

    # image_paths_low = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/standard91/low_x2/*.png'), dtype=tf.string)
    # image_paths_mid = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/standard91/mid_x2/*.png'), dtype=tf.string)
    # image_paths_high = tf.convert_to_tensor(tf.train.match_filenames_once('/home/yawli/Documents/hashnets/standard91/high_x2/*.png'), dtype=tf.string)

    [image_queue_low, image_queue_mid, image_queue_high] = \
        tf.train.slice_input_producer([image_paths_low, image_paths_mid, image_paths_high], shuffle=True, capacity=FLAGS.q_capacity)

    image_content_low = tf.read_file(image_queue_low)
    image_decoded_low = tf.image.decode_png(image_content_low, channels=1)
    image_low = tf.cast(image_decoded_low, tf.float32)

    image_content_mid = tf.read_file(image_queue_mid)
    image_decoded_mid = tf.image.decode_png(image_content_mid, channels=1)
    image_mid = tf.cast(image_decoded_mid, tf.float32)

    image_content_high = tf.read_file(image_queue_high)
    image_decoded_high = tf.image.decode_png(image_content_high, channels=1)
    image_high = tf.cast(image_decoded_high, tf.float32)

    size_low = tf.cast(tf.shape(image_low), tf.float32)
    offset_h = tf.cast(tf.floor(tf.random_uniform([num_crops, 1], 0.0, size_low[0] + 1 - crop_dim/upscale)), dtype=tf.int32)
    offset_w = tf.cast(tf.floor(tf.random_uniform([num_crops, 1], 0.0, size_low[1] + 1 - crop_dim/upscale)), dtype=tf.int32)

    # The index consist of two parts including the base and the offset. Generate the two parts separately and then sum them up
    def index_gen(offset_seed_h, offset_seed_w, dim):
        mask = tf.ones([dim, dim], dtype=tf.int32)
        index_offset_h = tf.reshape(kf.utils.kronecker_product(offset_seed_h, mask), [num_crops, dim, dim, 1])
        index_offset_w = tf.reshape(kf.utils.kronecker_product(offset_seed_w, mask), [num_crops, dim, dim, 1])
        index_offset = tf.concat([index_offset_h, index_offset_w], axis=3)
        index_base_range = range(0, dim)
        index_base_w, index_base_h = tf.meshgrid(index_base_range, index_base_range)
        index_base = tf.tile(tf.expand_dims(tf.stack([index_base_h, index_base_w], axis=2), axis=0), [num_crops, 1, 1, 1])
        index = index_base + index_offset
        ones = tf.zeros([num_crops, dim, dim, 1], dtype=tf.int32)
        index = tf.expand_dims(tf.concat([index, ones], axis=3), axis=3)
        return index

    index_low = index_gen(offset_h, offset_w, crop_dim/upscale)
    index_high = index_gen(offset_h*upscale, offset_w*upscale, crop_dim)
    batch_input_low = tf.gather_nd(image_low, index_low)
    batch_input_mid = tf.gather_nd(image_mid, index_high)
    batch_input_high = tf.gather_nd(image_high, index_high)
    batch_input_low.set_shape([num_crops, crop_dim/upscale, crop_dim/upscale, 1])
    batch_input_mid.set_shape([num_crops, crop_dim, crop_dim, 1])
    batch_input_high.set_shape([num_crops, crop_dim, crop_dim, 1])
    # batch_input_low = tf.expand_dims(batch_input_low, axis=3)
    # batch_input_mid = tf.expand_dims(batch_input_mid, axis=3)
    # batch_input_high = tf.expand_dims(batch_input_high, axis=3)

    [batch_low, batch_mid, batch_high] = tf.train.shuffle_batch(
        [batch_input_low, batch_input_mid, batch_input_high],
        batch_size=FLAGS.batch_size,
        capacity=FLAGS.q_capacity,
        min_after_dequeue=FLAGS.q_min,
        num_threads=4,
        enqueue_many=True
    )
    return batch_low, batch_mid, batch_high

def put_kernels_on_grid (kernel, pad = 1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.
  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: print('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tf.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x

def my_filter_callable(datum, tensor):
    # A filter that detects zero-valued scalars.
    if isinstance(tensor, debug_data.InconvertibleTensorProto):
        # Uninitialized tensor doesn't have bad numerical values.
        # Also return False for data types that cannot be represented as numpy
        # arrays.
        return False
    elif (np.issubdtype(tensor.dtype, np.float) or
        np.issubdtype(tensor.dtype, np.integer)):
        return np.any(tensor >= 65536)# or np.any(np.isinf(tensor))
    else:
        return False
        # return np.issubdtype(tensor.dtype, np.float32) and np.any(tensor >= 65536)

# def rgb_to_ycbcr(image_rgb): #batch x H x W x C
#     image_r = tf.squeeze(tf.slice(image_rgb, [0, 0, 0, 0], [-1, -1, -1, 1]), axis=3)
#     image_g = tf.squeeze(tf.slice(image_rgb, [0, 0, 0, 1], [-1, -1, -1, 1]), axis=3)
#     image_b = tf.squeeze(tf.slice(image_rgb, [0, 0, 0, 2], [-1, -1, -1, 1]), axis=3)
#     image_y = 16 + (65.738 * image_r + 129.057 * image_g + 25.064 * image_b)/256
#     image_cb = 128 + (-37.945 * image_r - 74.494 * image_g + 112.439 * image_b)/256
#     image_cr = 128 + (112.439 * image_r - 94.154 * image_g - 18.285 * image_b)/256
#     image_ycbcr = tf.stack([image_y, image_cb, image_cr], axis=3)
#     return image_ycbcr
#
# def ycbcr_to_rgb(image_ycbcr):
#     image_y = tf.squeeze(tf.slice(image_ycbcr, [0, 0, 0, 0], [-1, -1, -1, 1]), axis=3)
#     image_cb = tf.squeeze(tf.slice(image_ycbcr, [0, 0, 0, 1], [-1, -1, -1, 1]), axis=3)
#     image_cr = tf.squeeze(tf.slice(image_ycbcr, [0, 0, 0, 2], [-1, -1, -1, 1]), axis=3)
#     image_r = (298.082 * image_y + 408.583 * image_cr)/256 - 222.921
#     image_g = (298.082 * image_y - 100.291 * image_cb - 208.120 * image_cr)/256 + 135.576
#     image_b = (298.082 * image_y + 516.412 * image_cb)/256 - 276.836
#
#     image_r = tf.maximum(0, tf.minimum(255, image_r))
#     image_g = tf.maximum(0, tf.minimum(255, image_g))
#     image_b = tf.maximum(0, tf.minimum(255, image_b))
#
#     image_rgb = tf.stack([image_r, image_g, image_b], axis=3)
#     return image_rgb

def load_cdvl(FLAGS, mean_RGB):
    CROPPED_DIM = 128
    filename_queue = tf.train.string_input_producer(
        glob.glob('cdvl/crops/*.tfrecords') )

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
        'crop1' :tf.FixedLenFeature([], tf.string),
        'crop2': tf.FixedLenFeature([], tf.string),
    })
    crop1_decoded = tf.image.decode_png(features['crop1'])
    crop1_decoded.set_shape([CROPPED_DIM,CROPPED_DIM,3])
    crop1_decoded = tf.cast(crop1_decoded,tf.float32)-mean_RGB

    crop1_batch  = tf.train.shuffle_batch(
        [crop1_decoded],
        batch_size=FLAGS.batch_size,
        capacity=FLAGS.q_capacity,
        min_after_dequeue=FLAGS.q_min
    )
    target = crop1_batch
    data = tf.image.resize_bilinear(target,[CROPPED_DIM/2,CROPPED_DIM/2])
    FLAGS.cdvl = locals()
    return data,target
