__author__ = 'yawli'

import scipy.misc
from PIL import Image
import datetime
import time
import argparse
import sys
import pickle
#from tensorflow.python import debug as tf_debug
from model import *
from utils import *
import os

import tensorflow as tf
from tensorflow.python.client import timeline
import json
MAX_RGB = 1.0
# mean_RGB = tf.constant([123.680, 116.779, 103.939], tf.float32)
# mean_Y = tf.constant(111.8274, tf.float32)
mean_Y = 0 #111.6804 / 255 * MAX_RGB
DATASETS = {'div2k': load_div2k, 'standard91': load_standard91}

# Functions for use Dataset
def _parse_train_rgb(example_proto):
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
    image_low = tf.cast(tf.reshape(image_low, tf.stack([dim/upscale, dim/upscale, 3])), tf.float32)
    image_mid = tf.cast(tf.reshape(image_mid, tf.stack([dim, dim, 3])), tf.float32)
    image_high = tf.cast(tf.reshape(image_high, tf.stack([dim, dim, 3])), tf.float32)
    decision = tf.random_uniform([2], 0, 1)
    image_low = random_flip(image_low, decision[0])
    image_mid = random_flip(image_mid, decision[0])
    image_high = random_flip(image_high, decision[0])
    return image_low, image_mid, image_high

def _parse_test_rgb(filename_low, filename_mid, filename_high):
    """
    Parse examples to form testing dataset.
    """
    image_low = tf.read_file(filename_low)
    image_low = tf.image.decode_png(image_low, channels=3)
    image_low = tf.expand_dims(tf.cast(image_low, tf.float32), axis=0)
    image_mid = tf.read_file(filename_mid)
    image_mid = tf.image.decode_png(image_mid, channels=3)
    image_mid = tf.expand_dims(tf.cast(image_mid, tf.float32), axis=0)
    image_high = tf.read_file(filename_high)
    image_high = tf.image.decode_png(image_high, channels=3)
    image_high = tf.expand_dims(tf.cast(image_high, tf.float32), axis=0)
    return image_low, image_mid, image_high

def _parse_test_div2k(example_proto):
    """
    Parse examples to form testing dataset.
    """
    features = {"image_low": tf.FixedLenFeature((), tf.string),
                "image_mid": tf.FixedLenFeature((), tf.string),
                "image_high": tf.FixedLenFeature((), tf.string),
                "height": tf.FixedLenFeature((), tf.int64),
                "width": tf.FixedLenFeature((), tf.int64),
                "upscale": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    height = tf.cast(parsed_features['height'], tf.int64) #whether this is needed?
    width = tf.cast(parsed_features['width'], tf.int64)
    upscale = tf.cast(parsed_features['upscale'], tf.int64)

    # from IPython import embed; embed(); exit()
    image_low = tf.decode_raw(parsed_features['image_low'], tf.uint8)
    image_mid = tf.decode_raw(parsed_features['image_mid'], tf.uint8)
    image_high = tf.decode_raw(parsed_features['image_high'], tf.uint8)
    image_low = tf.cast(tf.reshape(image_low, tf.stack([1, height/upscale, width/upscale, 3])), tf.float32)
    image_mid = tf.cast(tf.reshape(image_mid, tf.stack([1, height, width, 3])), tf.float32)
    image_high = tf.cast(tf.reshape(image_high, tf.stack([1, height, width, 3])), tf.float32)
    return image_low, image_mid, image_high

def train_dataset_prepare_rgb(FLAGS):
    """
    Prepare for training dataset
    """
    filenames = '/scratch_net/ofsoundof/yawli/DIV2K_train_HR/x' + str(FLAGS.upscale) + '_chall.tfrecords'
    train_dataset = tf.data.TFRecordDataset(filenames)
    train_dataset = train_dataset.shuffle(buffer_size=5000).repeat()  # Repeat the input indefinitely.
    # train_dataset = train_dataset.apply(tf.contrib.data.map_and_batch(map_func=_parse_train, batch_size=FLAGS.batch_size, num_parallel_batches=8))
    train_dataset = train_dataset.map(map_func=_parse_train_rgb, num_parallel_calls=20)  # Parse the record into tensors.
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.prefetch(256)
    return train_dataset

def test_dataset_prepare_div2k(upscale):
    """
    Prepare for test dataset
    """

    filenames = '/scratch_net/ofsoundof/yawli/DIV2K_valid_HR/x' + str(upscale) + '.tfrecords'
    train_dataset = tf.data.TFRecordDataset(filenames)
    train_dataset = train_dataset.map(map_func=_parse_test_div2k)  # Parse the record into tensors.
    return train_dataset

def test_dataset_prepare_rgb(path_fun, test_dir):
    """
    Prepare for test dataset
    """
    image_list_low, image_list_mid, image_list_high = path_fun(test_dir)
    test_dataset = tf.data.Dataset.from_tensor_slices((image_list_low, image_list_mid, image_list_high))
    test_dataset = test_dataset.map(map_func=_parse_test_rgb)
    return test_dataset

def test_path_prepare_rgb(test_dir):
    """
    Prepare for paths of test image
    """
    set5_test_dir_low = test_dir + '_lowRGB/'
    set5_test_dir_mid = test_dir + '_midRGB/'
    set5_test_dir_high = test_dir + '_highRGB/'
    image_list_low = sorted(glob.glob(set5_test_dir_low + '*.png'))
    image_list_mid = sorted(glob.glob(set5_test_dir_mid + '*.png'))
    image_list_high = sorted(glob.glob(set5_test_dir_high + '*.png'))

    return image_list_low, image_list_mid, image_list_high

#used to crop the boundaries, compute psnr and ssim
def image_converter_rgb(image_sr_array, image_mid_array, image_high_array, flag, boundarypixels, MAX_RGB):
    """
    make image conversions and return the subjective score, i.e., psnr and ssim of SR and bicubic image
    :param image_sr_array:
    :param image_mid_array:
    :param image_high_array:
    :param flag:
    :param boundarypixels:
    :return:
    """
    image_sr_array = image_sr_array[boundarypixels:-boundarypixels, boundarypixels:-boundarypixels, :]
    image_mid_array = image_mid_array[boundarypixels:-boundarypixels, boundarypixels:-boundarypixels, :]
    image_high_array = image_high_array[boundarypixels:-boundarypixels, boundarypixels:-boundarypixels, :]
    if flag:
        image_sr_array = (image_sr_array*255/MAX_RGB-16)/219.859*256/255*MAX_RGB
        image_mid_array = (image_mid_array*255/MAX_RGB-16)/219.859*256/255*MAX_RGB
        image_high_array = (image_high_array*255/MAX_RGB-16)/219.859*256/255*MAX_RGB
    psnr_sr = 10 * np.log10(MAX_RGB ** 2 / (np.mean(np.square(image_high_array - image_sr_array))))
    psnr_itp = 10 * np.log10(MAX_RGB ** 2 / (np.mean(np.square(image_high_array - image_mid_array))))
    # from IPython import embed; embed(); exit()
    ssim_sr = ssim(np.uint8(image_sr_array*255/MAX_RGB), np.uint8(image_high_array*255/MAX_RGB), multichannel=True, gaussian_weights=True, use_sample_covariance=False)
    ssim_itp = ssim(np.uint8(image_mid_array*255/MAX_RGB), np.uint8(image_high_array*255/MAX_RGB), multichannel=True, gaussian_weights=True, use_sample_covariance=False)
    score = [psnr_itp, psnr_sr, ssim_itp, ssim_sr]
    return score

def run_test_train_dataset(selected_model):

    #Create the sess, and use some options for better using gpu
    print("Create session")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    config.allow_soft_placement = True  # If a operation is not define it the default device, let it execute in another.
    sess = tf.Session(config=config)

    with tf.device('/cpu:0'):
        #input setup
        #train dataset setup
        train_dataset = train_dataset_prepare_rgb(FLAGS)
        train_iterator = train_dataset.make_one_shot_iterator()
        #test dataset setup
        test_dir = '/home/yawli/Documents/hashnets/test/'
        test_set5_iterator = test_dataset_prepare_div2k(FLAGS.upscale).make_initializable_iterator()
        #create dataset handle and iterator
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        low, mid, high = iterator.get_next()

    with tf.device('/device:GPU:' + FLAGS.sge_gpu):
        low = (low-mean_Y)/255*MAX_RGB
        mid = mid/255*MAX_RGB
        high = high/255*MAX_RGB

        global_step = tf.Variable(1, name='global_step', trainable=False)

        # mid_ycbcr = rgb_to_ycbcr(mid*255)
        # mid_y = tf.slice(mid_ycbcr, [0, 0, 0, 0], [-1, -1, -1, 1])/255
        # high_ycbcr = rgb_to_ycbcr(high*255)
        # high_y = tf.slice(high_ycbcr, [0, 0, 0, 0], [-1, -1, -1, 1])/255
        #
        # sr_y = carn_sr_y(mid_y)
        # sr_ycbcr = tf.concat([sr_y*255, tf.slice(mid_ycbcr, [0, 0, 0, 1], [-1, -1, -1, -1])], axis=3)
        # sr = ycbcr_to_rgb(sr_ycbcr)/255
        # loss = tf.reduce_sum(tf.squared_difference(high_y, sr_y)) + tf.reduce_sum(tf.losses.get_regularization_losses())

        sr = carn_sr_rgb(mid)
        loss = tf.reduce_sum(tf.squared_difference(high, sr)) + tf.reduce_sum(tf.losses.get_regularization_losses())


        MSE_sr, PSNR_sr = comp_mse_psnr(sr, high, MAX_RGB)
        MSE_interp, PSNR_interp = comp_mse_psnr(mid, high, MAX_RGB)
        SSIM_sr = tf.image.ssim_multiscale(sr, high, max_val=1.0)
        SSIM_interp = tf.image.ssim_multiscale(mid, high, max_val=1.0)
        PSNR_gain = PSNR_sr - PSNR_interp
        train_op, global_step = training(loss, FLAGS, global_step)

    #train summary
    tf.summary.scalar('MSE_sr', MSE_sr)
    tf.summary.scalar('PSNR_sr', PSNR_sr)
    tf.summary.scalar('MSE_interp', MSE_interp)
    tf.summary.scalar('PSNR_interp', PSNR_interp)
    tf.summary.scalar('PSNR_gain', PSNR_gain)
    slim.summarize_collection(collection=tf.GraphKeys.TRAINABLE_VARIABLES)
    #test summaries
    psnr_test_train_set5 = tf.placeholder(tf.float32)
    # psnr_test_train_set14 = tf.placeholder(tf.float32)
    tf.summary.scalar('psnr_test_train_set5', psnr_test_train_set5)
    # tf.summary.scalar('psnr_test_train_set14', psnr_test_train_set14)
    merged = tf.summary.merge_all()


    #Get the dataset handle
    train_handle = sess.run(train_iterator.string_handle())
    test_set5_handle = sess.run(test_set5_iterator.string_handle())

    print("Create checkpoint directory")
    # if not FLAGS.restore:
    FLAGS.checkpoint = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '-' + FLAGS.checkpoint
    with open('checkpoint.txt', 'w') as text_file: #save at the current directory, used for testing
        text_file.write(FLAGS.checkpoint)
    LOG_DIR = os.path.join('/scratch_net/ofsoundof/yawli/logs', FLAGS.checkpoint )
    # LOG_DIR = os.path.join('/home/yawli/Documents/hashnets/logs', FLAGS.checkpoint )
    # assert (not os.path.exists(LOG_DIR)), 'LOG_DIR %s already exists'%LOG_DIR

    print("Create summary file writer")
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)

    print("Create saver")
    saver = tf.train.Saver(max_to_keep=200)

    #Always init, then optionally restore
    print("Initialization")
    sess.run(tf.global_variables_initializer())

    if FLAGS.restore:
        print('Restore model from checkpoint {}'.format(tf.train.latest_checkpoint(LOG_DIR)))
        saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))
        checkpoint_filename = 'checkpoint'
    else:
        checkpoint_filename = 'checkpoint'
    # checkpoint_filename = 'checkpoint'
    score5_all = []
    model_index = []
    path_set5 = sorted(glob.glob(('/scratch_net/ofsoundof/yawli/DIV2K_valid_HR/GT/*.png')))
    path_set5 = [os.path.join(LOG_DIR, os.path.basename(i)) for i in path_set5]
    i = sess.run(global_step) + 1
    print("Start iteration")
    while i <= FLAGS.max_iter:
        if i % FLAGS.summary_interval == 0:
            # test set5
            sess.run(test_set5_iterator.initializer)
            score5 = np.zeros((4, 101))
            for j in range(100):
                _sr, _mid, _high, _PSNR_interp, _PSNR_sr, _SSIM_interp, _SSIM_sr = sess.run([sr, mid, high, PSNR_interp, PSNR_sr, SSIM_interp, SSIM_sr], feed_dict={handle: test_set5_handle})
                scipy.misc.toimage(np.squeeze(_sr)*255/MAX_RGB, cmin=0, cmax=255).save(path_set5[j])
                # The current scipy version started to normalize all images so that min(data) become black and max(data)
                # become white. This is unwanted if the data should be exact grey levels or exact RGB channels.
                # The solution: scipy.misc.toimage(image_array, cmin=0.0, cmax=255).save('...') or use imageio library
                #from IPython import embed; embed();
                # score = image_converter_rgb(np.squeeze(_sr), np.squeeze(_mid), np.squeeze(_high), False, FLAGS.upscale, MAX_RGB)
                score = [_PSNR_interp, _PSNR_sr, _SSIM_interp, _SSIM_sr]
                score5[:, j] = score
            score5[:, -1] = np.mean(score5[:, :-1], axis=1)
            print('PSNR results for DIV2K: SR {}, Bicubic {}'.format(score5[1, -1], score5[0, -1]))
            time_start = time.time()
            [_, i, l, mse_sr, mse_interp, psnr_sr, psnr_interp, psnr_gain, summary] =\
                sess.run([train_op, global_step, loss, MSE_sr, MSE_interp, PSNR_sr, PSNR_interp, PSNR_gain, merged],
                         feed_dict={handle: train_handle, psnr_test_train_set5: score5[1, -1]})#, options=options, run_metadata=run_metadata)
            duration = time.time() - time_start
            train_writer.add_summary(summary, i)
            print("Iter %d, total loss %.5f; sr (mse %.5f, psnr %.5f); interp (mse %.5f, psnr %.5f); psnr_gain:%f" % (i-1, l, mse_sr, psnr_sr, mse_interp, psnr_interp, psnr_gain))
            print('Training time for one iteration: {}'.format(duration))

            if (i-1) % FLAGS.checkpoint_interval == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'model'), global_step=i, latest_filename=checkpoint_filename, write_meta_graph=True)
                print("Saved checkpoint to {}".format(save_path))
                print("Flushing train writer")
                train_writer.flush()

            model_index.append(i)
            score5_all.append([score5[0, -1], score5[1, -1], score5[1, -1]-score5[0, -1], score5[2, -1], score5[3, -1], score5[3, -1]-score5[2, -1]])
        else:
            [_, i, _] = sess.run([train_op, global_step, loss], feed_dict={handle: train_handle})
            #print sl
            #print sm
            #print ssr
            #from IPython import embed; embed();


    #write to csv files
    descriptor5 = 'Average PSNR (dB)/SSIM for DIV2K, Scale ' + str(FLAGS.upscale) + ', different model parameters during training' + '\n'
    # descriptor14 = 'Average PSNR (dB)/SSIM for Set14, Scale ' + str(FLAGS.upscale) + ', different model parameters during training' + '\n'
    descriptor = [descriptor5, '', '', '', '', '', '']#, descriptor14, '', '', '', '', '', '']
    header_mul = ['Iteration'] + ['Bicu', 'SR', 'Gain']
    model_index_array = np.expand_dims(np.array(model_index), axis=0)
    score5_all_array = np.transpose(np.array(score5_all))
    # score14_all_array = np.transpose(np.array(score14_all))
    written_content = np.concatenate((model_index_array, score5_all_array), axis=0)
    written_content_fmt = ['{:<8}'] + ['{:<8.4f}', '{:<8.4f}', '{:<8.4f}'] * 2
    content = content_generator_mul_line(descriptor, written_content, written_content_fmt, header_mul)
    file_writer(os.path.join(LOG_DIR, 'psnr_ssim.csv'), 'a', content)


    #write to pickle files
    variables = {'index': np.array(model_index), 'bicubic': score5_all_array[0, :], 'sr': score5_all_array[1, :], 'gain': score5_all_array[2, :]}
    pickle_out = open(os.path.join(LOG_DIR, 'psnr_DIV2K.pkl'), 'wb')
    pickle.dump(variables, pickle_out)
    pickle_out.close()

    # variables = {'index': np.array(model_index), 'bicubic': score14_all_array[0, :], 'sr': score14_all_array[1, :], 'gain': score14_all_array[2, :]}
    # pickle_out = open(os.path.join(LOG_DIR, 'psnr_set14.pkl'), 'wb')
    # pickle.dump(variables, pickle_out)
    # pickle_out.close()

def test(selected_model):
    # test_dir = '/home/yawli/Documents/hashnets/test/Set5'

    #input path
    if FLAGS.model_selection == 9:
        channel = 'RGB'
        test_dir_low = FLAGS.test_dir + '_X' + str(FLAGS.upscale) + '_low' + channel + '/'
        test_dir_mid = FLAGS.test_dir + '_X' + str(FLAGS.upscale) + '_mid' + channel + '/'
        test_dir_high = FLAGS.test_dir + '_X' + str(FLAGS.upscale) + '_high' + channel + '/'
        image_list_low = sorted(glob.glob(test_dir_low + '*.png'))
        image_list_mid = sorted(glob.glob(test_dir_mid + '*.png'))
        image_list_high = sorted(glob.glob(test_dir_high + '*.png'))
    else:
        image_list_low, image_list_mid, image_list_high = test_path_prepare(FLAGS.test_dir + '_X' + str(FLAGS.upscale))
    if len(FLAGS.checkpoint) == 0:
        with open('checkpoint.txt', 'r') as text_file:
            lines = text_file.readlines()
            FLAGS.checkpoint = os.path.join('/scratch_net/ofsoundof/yawli/logs', lines[0])
    #output path
    image_list_sr = [os.path.join(FLAGS.checkpoint, os.path.basename(img_name)) for img_name in image_list_low]
    print(image_list_sr)

    #read input
    image_low = []
    image_mid = []
    image_high = []
    for i in range(len(image_list_low)):
        if FLAGS.model_selection == 9:
            image_low.append(np.expand_dims((scipy.misc.fromimage(Image.open(image_list_low[i])).astype(np.float32) - mean_Y)/255*MAX_RGB, axis=0))
            image_mid.append(np.expand_dims(scipy.misc.fromimage(Image.open(image_list_mid[i])).astype(np.float32)/255*MAX_RGB, axis=0))
            image_high.append(scipy.misc.fromimage(Image.open(image_list_high[i])).astype(np.float32)/255*MAX_RGB)
        else:
            image_low.append(np.expand_dims(np.expand_dims((scipy.misc.fromimage(Image.open(image_list_low[i])).astype(np.float32) - mean_Y)/255*MAX_RGB, axis=0), axis=3))
            image_mid.append(np.expand_dims(np.expand_dims((scipy.misc.fromimage(Image.open(image_list_mid[i])).astype(np.float32))/255*MAX_RGB, axis=0), axis=3))
            image_high.append(scipy.misc.fromimage(Image.open(image_list_high[i])).astype(np.float32)/255*MAX_RGB)

    #build the model
    print('The used GPU is /device:GPU:'+FLAGS.sge_gpu)
    l = 3 if FLAGS.model_selection == 9 else 1
    with tf.device('/device:GPU:'+FLAGS.sge_gpu):
        image_input_low = tf.placeholder(dtype=tf.float32, shape=(1, None, None, l))
        image_input_mid = tf.placeholder(dtype=tf.float32, shape=(1, None, None, l))
        global_step = tf.Variable(1, name='global_step', trainable=False)
        image_sr, annealing_factor = selected_model(image_input_low, image_input_mid, FLAGS, global_step)

    print('Config')
    config = tf.ConfigProto()
    print("Allow growth")
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    print("Allow soft placement")
    config.allow_soft_placement = True  # If a operation is not define it the default device, let it execute in another.
    # config.log_device_placement = True
    print("Create session")
    sess = tf.Session(config=config)

    print("Load checkpoint")
    assert FLAGS.checkpoint != ''
    checkpoint_to_resume = FLAGS.checkpoint
    print("Resuming from checkpoint {}".format(checkpoint_to_resume))
    all_checkpoints = glob.glob(checkpoint_to_resume + '/model-*')
    all_checkpoints = [os.path.splitext(x)[0] for x in all_checkpoints if x.endswith('index')]
    all_checkpoints = sorted(all_checkpoints, key=lambda x: int(os.path.basename(x)[6:-1]))
    # all_checkpoints = tf.train.get_checkpoint_state(checkpoint_to_resume).all_model_checkpoint_paths
    ckpt_num = 2#len(all_checkpoints) #50 #search from the last ckpt_num checkpoints

    saver = tf.train.Saver()
    #from IPython import embed; embed(); exit()
    if FLAGS.test_score_compute:
        print("\nTest for every saved checkpoint and search the best PSNR")
        model_index = np.zeros((1, ckpt_num))
        score_mean = np.zeros((4, ckpt_num))
        # start from the last ckpt_num checkpoints
        for i in range(-ckpt_num, 0, 1):
            current_checkpoint = all_checkpoints[i]
            print("\nCurrent checkpoint: {}".format(current_checkpoint))
            # from IPython import embed; embed(); exit()
            assert current_checkpoint is not None

            saver.restore(sess, current_checkpoint)
            score_all = np.zeros((4, len(image_list_low)+1))

            #image_list_output = []
            start_time = time.time()
            for image in range(len(image_list_low)):
                image_sr_ = sess.run(image_sr, feed_dict={image_input_low: image_low[image], image_input_mid: image_mid[image]})
                scipy.misc.toimage(np.squeeze(image_sr_)*255/MAX_RGB, cmin=0, cmax=255).save(image_list_sr[image])

                flag = image == 2 and len(image_list_low) == 14
                score = image_converter(np.squeeze(image_sr_), np.squeeze(image_mid[image]), image_high[image], flag, FLAGS.upscale, MAX_RGB)
                score_all[:, image] = score
            score_all[:, -1] = np.mean(score_all[:, :-1], axis=1)
            duration = time.time() - start_time
            print('The execution time is {}'.format(duration))
            print('Compute PSNR and SSIM')
            print('PSNR Bi Ave. {:<8.4f}, per image {}.'.format(score_all[0, -1], np.round(score_all[0, 0:-1], decimals=2)))
            print('PSNR SR Ave. {:<8.4f}, per image {}.'.format(score_all[1, -1], np.round(score_all[1, 0:-1], decimals=2)))
            print('SSIM Bi Ave. {:<8.4f}, per image {}.'.format(score_all[2, -1], np.round(score_all[2, 0:-1], decimals=4)))
            print('SSIM SR Ave. {:<8.4f}, per image {}.'.format(score_all[3, -1], np.round(score_all[3, 0:-1], decimals=4)))

            model_name = os.path.basename(str(current_checkpoint))
            model_index[0, i] = int(model_name[6:])
            score_mean[:, i] = score_all[:, -1]
        print('\nWrite the average PSNR for all of the checkpoints to csv files')
        descriptor_psnr = 'Average PSNR (dB) for ' + os.path.basename(FLAGS.test_dir) + ', Scale ' + str(FLAGS.upscale) + ', different model parameters during training' + '\n'
        descriptor_ssim = 'Average SSIM for ' + os.path.basename(FLAGS.test_dir) + ', Scale ' + str(FLAGS.upscale) + ', different model parameters during training' + '\n'
        descriptor_mul = [descriptor_psnr, '', '', descriptor_ssim, '']
        header_mul = ['Iteration', 'Bicubic', 'SR', 'Bicubic', 'SR']
        written_content_fmt = ['{:<8} ', '{:<8.2f} ', '{:<8.2f} ', '{:<8.3f} ', '{:<8.3f} ']
        written_content = np.concatenate((model_index, score_mean), axis=0)
        content = content_generator_mul_line(descriptor_mul, written_content, written_content_fmt, header_mul)
        #write the average psnr and ssim values for all of the checkpoints in the current log directory
        file_writer(os.path.join(FLAGS.checkpoint, 'psnr_ssim.csv'), 'a', content)
        #write to pkl files
        variables = {'index': model_index, 'bicubic': score_mean[0, :], 'sr': score_mean[1, :], 'bicubic_ssim': score_mean[2, :], 'sr_ssim': score_mean[3, :]}
        pickle_out = open(os.path.join(FLAGS.checkpoint, 'psnr_ssim_test_' + os.path.basename(FLAGS.test_dir) + '.pkl'), 'wb')
        pickle.dump(variables, pickle_out)
        pickle_out.close()

        print('Find the best PSNR')
        value_psnr = score_mean[1, :]
        value_ssim = score_mean[3, :]
        index = model_index
        i_max = np.argmax(value_psnr)
        index_max = int(index[0, i_max])
        value_psnr_max = value_psnr[i_max]
        value_ssim_max = value_ssim[i_max]
        print('({}, {}, {}): the maximum PSNR {} appears at {}th iteration'.format(index_max, value_psnr_max, value_ssim_max, value_psnr_max, index_max))
        content = "Index {}, Checkpoint {}, PSNR {}, SSIM {} \n".format(i_max, index_max, value_psnr_max, value_ssim_max)
        #write the maximum psnr and index in the current log directory
        file_writer(os.path.join(FLAGS.checkpoint, 'psnr_ssim.csv'), 'a', content)

        # the checkpoints should be counted from the end of the list
        current_checkpoint = all_checkpoints[i_max - ckpt_num]
        print('Test for the best checkpoint {}'.format(current_checkpoint))
        assert current_checkpoint is not None
        saver.restore(sess, current_checkpoint)

        #get the best psnr and save the image using parameters corresponding to best psnr
        score_best = np.zeros([4, len(image_list_low)+1])
        for image in range(len(image_list_low)):
            image_sr_ = sess.run(image_sr, feed_dict={image_input_low: image_low[image], image_input_mid: image_mid[image]})
            #save image
            scipy.misc.toimage(np.squeeze(image_sr_)*255/MAX_RGB, cmin=0, cmax=255).save(image_list_sr[image])
            flag = image == 2 and len(image_list_low) == 14
            score = image_converter(np.squeeze(image_sr_), np.squeeze(image_mid[image]), image_high[image], flag, FLAGS.upscale, MAX_RGB)
            score_best[:, image] = score
        score_best[:, -1] = np.mean(score_best[:, :-1], axis=1)

        print('Save the best PSNR')
        model_name = os.path.basename(current_checkpoint)[6:]
        content = 'Dataset {:<8}, Model {:<4} '.format(os.path.basename(FLAGS.test_dir), FLAGS.model_selection)\
                  + '{:<8} {:<8.2f} {:<8.2f} {:<8.4f} {:<8.4f} '.format(model_name, score_best[0][-1], score_best[1][-1], score_best[2][-1], score_best[3][-1]) + '\n'
        #write the best average psnr in a shared file under the project directory
        file_writer('/home/yawli/Documents/hashnets/final_test_results_average.csv', 'a', content)

        descriptor_mul = ['Dataset {}, Model {} \n'.format(os.path.basename(FLAGS.test_dir), FLAGS.model_selection), '', '', '', '']
        header_mul = ['Image', 'psnr_bicu', 'psnr_sr', 'ssim_bicu', 'ssim_sr']
        written_content_fmt = ['{:<8} ', '{:<8} ', '{:<8} ', '{:<8} ', '{:<8} ']
        image_name = [os.path.basename(image)[4:7] if image[-6:] == 'HR.png' else os.path.splitext(os.path.basename(image))[0] for image in image_list_low]
        score_best[:2, :] = np.round(score_best[:2, :], decimals=2)
        score_best[2:, :] = np.round(score_best[2:, :], decimals=3)
        written_content = np.concatenate((np.expand_dims(np.array(image_name), axis=0), score_best[:, :-1]), axis=0)
        content = content_generator_mul_line(descriptor_mul, written_content, written_content_fmt, header_mul)
        #write the best per image psnr in a shared file under the project directory
        file_writer('/home/yawli/Documents/hashnets/final_test_results_perimage.csv', 'a', content)

    if FLAGS.test_runtime_compute:
        print('\nGet the runtime')
        current_checkpoint = all_checkpoints[-1]
        assert current_checkpoint is not None
        saver.restore(sess, current_checkpoint)

        #report runtime
        # import matplotlib.pyplot as plt
        test_iter = 100
        duration = np.zeros(len(image_list_low)+2)
        #profiler = tf.profiler.Profiler(sess.graph)
        #builder = tf.profiler.ProfileOptionBuilder
        for image in range(len(image_list_low)):
            print(image_list_low[image])
            per_image_start = time.time()
            for i in range(test_iter):
                #run_meta = tf.RunMetadata()
                sess.run(image_sr, feed_dict={image_input_low: image_low[image], image_input_mid: image_mid[image]})#,
                         #options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_meta)
                #profiler.add_step(image, run_meta)
                # Profile the parameters of your model.
                #profiler.profile_name_scope(options=(builder.trainable_variables_parameter()))

                # Or profile the timing of your model operations.
                #opts = builder.time_and_memory()
                #profiler.profile_operations(options=opts)

                # Or you can generate a timeline:
                #opts = (builder(builder.time_and_memory()).with_step(i).with_timeline_output('profile').build())
                #profiler.profile_graph(options=opts)
                # from IPython import embed; embed()
                #Used for get the runtime
                # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                # many_runs_timeline.update_timeline(chrome_trace)
                #
                # sr_shape = sr_.shape
                # index.append(np.reshape(sr_, sr_shape[1]*sr_shape[2]))
                # plt.figure(image+1)
                # plt.hist(index[image], bins=512)
                # plt.title('Image {}'.format(image))
                # plt.show()
            duration[image+2] = time.time() - per_image_start
        duration[1] = np.sum(duration[2:])
        duration[0] = duration[1]/len(image_list_low)
        duration = duration/test_iter*1000
        # # from IPython import embed; embed(); exit()
        # index1 = np.concatenate(index)
        # plt.hist(index1, bins=512)  # arguments are passed to np.histogram
        # plt.title("Total")
        # plt.show()
        # many_runs_timeline.save(os.path.join(FLAGS.checkpoint, 'timeline_merged_{}_runs.json'.format(os.path.basename(FLAGS.test_dir))))
        print('The average and per image execution time is {}'.format(duration))
        #write the total and average runtime in the current log directory
        content = '\n\nAve. test time {}, Dataset {} \n'.format(duration[0], os.path.basename(FLAGS.test_dir))
        file_writer(os.path.join(FLAGS.checkpoint, 'psnr_ssim.csv'), 'a', content)

        # descriptor_mul = ['', '']
        # header_mul = ['Model', FLAGS.model_selection]
        # written_content_fmt = ['{:<8}\t', '{:<8}\t']
        # image_name = [os.path.basename(image)[4:7] if image[-6:] == 'HR.png' else os.path.splitext(os.path.basename(image))[0] for image in image_list_low]
        # time_identity = ['Ave.', 'Total'] + image_name
        # written_content = np.stack((np.array(time_identity), np.round(duration, decimals=3)), axis=0)

        descriptor_mul = ['']
        header_mul = [FLAGS.model_selection]
        written_content_fmt = ['{:<8.3f}\t']
        written_content = np.expand_dims(np.round(duration, decimals=3), axis=0)

        content = content_generator_mul_line(descriptor_mul, written_content, written_content_fmt, header_mul)
        time_dir = os.path.join('/home/yawli/Documents/hashnets/time_' + os.path.basename(FLAGS.test_dir) + '.csv')
        #write all of the runtime information in a shared file under the project directory
        file_writer(time_dir, 'a', content)

def main(_):
    """
    run_train
    run_test_train
    run_test_train_dataset

    test
    test_restore      Test fast architecture
    test_restore_gpu  To get gpu/cpu test time

    """
    if FLAGS.model_selection == 1:
        selected_model = fast_hashnet
    elif FLAGS.model_selection == 2:
        selected_model = fast_hashnet_restore
    elif FLAGS.model_selection == 3:
        selected_model = deep_hashnet
    elif FLAGS.model_selection == 4:
        selected_model = deep_hashnet_stack
    elif FLAGS.model_selection == 5:
        selected_model = vdsr
    elif FLAGS.model_selection == 6:
        selected_model = srcnn
    elif FLAGS.model_selection == 7:
        selected_model = fsrcnn
    elif FLAGS.model_selection == 8:
        selected_model = espcn
    elif FLAGS.model_selection == 9:
        selected_model = srresnet
    elif FLAGS.model_selection == 10:
        selected_model = espcn_comp
    elif FLAGS.model_selection == 11:
        selected_model = espcn_chal
    else:
        selected_model = deep_hashnet_more
    # print('Selected model is {}'.format(selected_model))
    if len(FLAGS.test_dir):
        print(FLAGS.test_dir)
        if FLAGS.test_procedure == 1:
            test(selected_model)
        elif FLAGS.test_procedure == 2:
            test_restore_gpu(selected_model)
        elif FLAGS.test_procedure == 3:
            test_restore_rgb(selected_model)
        else:
            test_restore()

    else:
        if FLAGS.model_selection == 9:
            run_train(selected_model)
        else:
            run_test_train_dataset(selected_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")

    #frequently used training parameters
    parser.add_argument('--max_iter', type=int, default=100000, help='Number of iter to run trainer.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--dataset', type=str, default='div2k', help='Dataset for experiment')
    parser.add_argument('--checkpoint', type=str, default='', help='Unique checkpoint name')
    parser.add_argument('--summary_interval', type=int, default=1, help='Summary interval')
    parser.add_argument('--checkpoint_interval', type=int, default=10000, help='Checkpoint interval')
    parser.add_argument('--upscale', type=int, default=2, help='Upscaling factor')
    parser.add_argument('--sge_gpu_all', type=str, default='0', help='All available gpus')
    parser.add_argument('--sge_gpu', type=str, default='0', help='Currently used gpu')

    #procedure selection
    parser.add_argument('--test_dir', type=str, default='', help='Test directory, when not empty, start testing')
    parser.add_argument("--test_procedure", type=int, default=0, help="test or test_restore or test_restore_gpu")
    parser.add_argument("--test_score_compute", help="When used, compute the best PSNR/SSIM values", action="store_true")
    parser.add_argument("--test_runtime_compute", help="When used compute the runtime", action="store_true")

    #model parameters
    parser.add_argument('--deep_feature_layer', type=int, default=3, help='Number of feature layers in the deep architecture')
    parser.add_argument('--deep_layer', type=int, default=7, help='Number of layers in the deep architecture')
    parser.add_argument('--deep_channel', type=int, default=2 ** 2, help='Number of channels of the regression output except the last regresion layer for which the output channel is always upscale**2')
    parser.add_argument('--deep_anchor', type=int, default=16, help='Number of anchors in the deep architecture')
    parser.add_argument('--deep_kernel', type=int, default=3, help='Kernel size in the regression layers of the deep architecture')
    parser.add_argument('--model_selection', type=int, default=2, help='Select the use model')
    parser.add_argument("--restore", help="Restore from saved checkpoint and continue training", action="store_true")

    #possibly deprecated parameters
    parser.add_argument("--output", type=int, default=0, help="whether to use output layer")
    parser.add_argument("--skip_conv", type=int, default=0, help="whether use convolution in skip connection")
    parser.add_argument("--activation_regression", type=int, default=0, help="activation function in the regression layer")
    parser.add_argument("--biases_add_regression", type=int, default=1, help="biases_add in the similarity layer")
    parser.add_argument("--dense_connection", type=int, default=0, help="dense connection, should be used together with skip_conv")

    parser.add_argument('--regression', type=int, default=1, help='Number of regressors')
    parser.add_argument("--debug", type="bool", nargs="?", const=True, default=False, help="Use debugger to track down bad values during training")

    #parameters used for queue input
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--q_capacity', type=int, default=40000, help='Queue capacity')
    parser.add_argument('--q_min', type=int, default=10000, help='Minimum number of elements after dequeue')
    parser.add_argument('--crop_dim', type=int, default=64, help='Crop dimension')
    parser.add_argument('--num_crop', type=int, default=100, help='Number of crops per image')

    FLAGS, unparsed = parser.parse_known_args()
    print("FLAGS: {}".format(FLAGS))
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
