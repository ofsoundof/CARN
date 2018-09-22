__author__ = 'yawli'

import scipy.misc
from PIL import Image
import datetime
import time
import argparse
import sys
import pickle
from utils import training, random_flip, comp_mse_psnr, file_writer, content_generator_mul_line
import os
import glob
import tensorflow.contrib.slim as slim
import numpy as np
from myssim import compare_ssim as ssim
import tensorflow as tf
import scipy.io as sio
# import imageio
MAX_RGB = 1
# Functions for use Dataset
def _parse_train(example_proto):
    """
    Parse examples in training dataset from tfrecords
    """
    features = {"image_gt": tf.FixedLenFeature((), tf.string),
              "image_n": tf.FixedLenFeature((), tf.string),
              "crop_dim": tf.FixedLenFeature((), tf.int64),
              "sigma": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    dim = tf.cast(parsed_features['crop_dim'], tf.int64) #whether this is needed?
    sigma = tf.cast(parsed_features['sigma'], tf.int64)

    # from IPython import embed; embed(); exit()
    image_gt = tf.decode_raw(parsed_features['image_gt'], tf.float32)
    image_n = tf.decode_raw(parsed_features['image_n'], tf.float32)
    image_gt = tf.cast(tf.reshape(image_gt, tf.stack([dim, dim, 1])), tf.float32)
    image_n = tf.cast(tf.reshape(image_n, tf.stack([dim, dim, 1])), tf.float32)
    decision = tf.random_uniform([2], 0, 1)
    image_gt = random_flip(image_gt, decision[0])
    image_n = random_flip(image_n, decision[0])
    return image_gt, image_n

def _parse_test(example_proto):
    """
    Parse examples in training dataset from tfrecords
    """
    features = {"image_gt": tf.FixedLenFeature((), tf.string),
              "image_n": tf.FixedLenFeature((), tf.string),
              "height": tf.FixedLenFeature((), tf.int64),
              "width": tf.FixedLenFeature((), tf.int64),
              "sigma": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    height = tf.cast(parsed_features['height'], tf.int64)
    width = tf.cast(parsed_features['width'], tf.int64)
    sigma = tf.cast(parsed_features['sigma'], tf.int64)

    # from IPython import embed; embed(); exit()
    image_gt = tf.decode_raw(parsed_features['image_gt'], tf.float32)
    image_n = tf.decode_raw(parsed_features['image_n'], tf.float32)
    image_gt = tf.cast(tf.reshape(image_gt, tf.stack([height, width, 1])), tf.float32)
    image_n = tf.cast(tf.reshape(image_n, tf.stack([height, width, 1])), tf.float32)
    image_gt = tf.expand_dims(image_gt, axis=0)
    image_n = tf.expand_dims(image_n, axis=0)
    return image_gt, image_n

# def _parse_test(image_gt, image_n):
#     """
#     Parse examples to form testing dataset.
#     """
#     image_gt = tf.expand_dims(tf.expand_dims(image_gt, axis=0), axis=3)
#     image_n = tf.expand_dims(tf.expand_dims(image_n, axis=0), axis=3)
#     # image_gt = tf.read_file(filename_gt)
#     # image_gt = tf.image.decode_png(image_gt, channels=1)
#     # image_gt = tf.expand_dims(tf.cast(image_gt, tf.float32), axis=0)
#     # mat = sio.loadmat(filename_n, struct_as_record=False)
#     # image_n = mat['img_n'].astype(np.float32)
#     # image_n = tf.constant(image_n, tf.float32)
#     # image_n = tf.constant(image_n, tf.float32)
#     # image_n = tf.expand_dims(tf.expand_dims(image_n, axis=0), axis=2)
#     # image_n = tf.read_file(filename_n)
#     # image_n = tf.image.decode_png(image_n, channels=1)
#     # image_n = tf.expand_dims(tf.cast(image_n, tf.float32), axis=0)
#     return image_gt, image_n

def train_dataset_prepare(FLAGS):
    """
    Prepare for training dataset
    """
    filenames = '/scratch_net/ofsoundof/yawli/Train400/Sig' + str(FLAGS.sigma) + '_diff.tfrecords'
    train_dataset = tf.data.TFRecordDataset(filenames)
    train_dataset = train_dataset.shuffle(buffer_size=5000).repeat()  # Repeat the input indefinitely.
    # train_dataset = train_dataset.apply(tf.contrib.data.map_and_batch(map_func=_parse_train, batch_size=FLAGS.batch_size, num_parallel_batches=8))
    train_dataset = train_dataset.map(map_func=_parse_train, num_parallel_calls=20)  # Parse the record into tensors.
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.prefetch(256)
    return train_dataset

def test_dataset_prepare(path_fun, test_dir, sigma):
    """
    Prepare for test dataset
    """
    # image_list_gt, image_list_n = path_fun(test_dir, sigma)
    # image_gt_a = []
    # image_n_a = []
    # for i in range(len(image_list_gt)):
    #     image_gt = imageio.imread(image_list_gt[i]).astype(np.float32)
    #     mat = sio.loadmat(image_list_n[i], struct_as_record=False)
    #     image_n = mat['img_n'].astype(np.float32)
    #     image_gt_a.append(tf.convert_to_tensor(image_gt))
    #     image_n_a.append(tf.convert_to_tensor(image_n))
    # test_dataset = tf.data.Dataset.from_tensors((image_gt_a, image_n_a))
    # test_dataset = test_dataset.map(map_func=_parse_test)
    # return test_dataset

    filenames = '/scratch_net/ofsoundof/yawli/BSD68/Sig' + str(FLAGS.sigma) + '.tfrecords'
    train_dataset = tf.data.TFRecordDataset(filenames)
    train_dataset = train_dataset.map(map_func=_parse_test)  # Parse the record into tensors.
    return train_dataset

def test_path_prepare(test_dir, sigma):
    """
    Prepare for paths of test image
    """
    test_dir_gt = test_dir + 'GT/'
    test_dir_n = test_dir + 'Sig' + str(sigma) + '/'
    image_list_gt = sorted(glob.glob(test_dir_gt + '*.png'))
    image_list_n = sorted(glob.glob(test_dir_n + '*.mat'))

    return image_list_gt, image_list_n

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
    psnr_sr = 10 * np.log10(MAX_RGB ** 2 / (np.mean(np.square(image_high_array - image_sr_array))))
    psnr_itp = 10 * np.log10(MAX_RGB ** 2 / (np.mean(np.square(image_high_array - image_mid_array))))
    # from IPython import embed; embed(); exit()
    ssim_sr = ssim(np.uint8(image_sr_array*255/MAX_RGB), np.uint8(image_high_array*255/MAX_RGB), gaussian_weights=True, use_sample_covariance=False)
    ssim_itp = ssim(np.uint8(image_mid_array*255/MAX_RGB), np.uint8(image_high_array*255/MAX_RGB), gaussian_weights=True, use_sample_covariance=False)
    score = [psnr_itp, psnr_sr, ssim_itp, ssim_sr]
    return score

def carn_denoise(data, FLAGS):
    num_anchor = FLAGS.deep_anchor
    inner_channel = FLAGS.deep_channel
    activation = tf.keras.layers.PReLU(shared_axes=[1, 2]) #if FLAGS.activation_regression == 1 else None
    biases_add = tf.zeros_initializer() #if FLAGS.biases_add_regression == 1 else None

    with slim.arg_scope([slim.conv2d], stride=1,
                        weights_initializer=tf.keras.initializers.he_normal(),
                        weights_regularizer=slim.l2_regularizer(0.0001), reuse=tf.AUTO_REUSE):
        # feature = slim.stack(data, slim.conv2d, [(64, [3, 3]), (64, [3, 3]), (inner_channel, [3, 3])], scope='feature_layer')
        feature = data
        # for i in range(1, FLAGS.deep_feature_layer):
        #     feature = slim.conv2d(feature, 64, [3, 3], scope='feature_layer' + str(i), activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
        feature = slim.conv2d(feature, 64, [3, 3], scope='feature_layer' + str(1), activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
        feature = slim.conv2d(feature, 64, [3, 3], stride=1, scope='feature_layer' + str(2), activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
        # feature = slim.conv2d(feature, 64, [3, 3], scope='feature_layer2', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
        feature = slim.conv2d(feature, inner_channel, [3, 3], stride=2, scope='feature_layer' + str(FLAGS.deep_feature_layer), activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
        reshape_size = tf.shape(feature)
        kernel_size = FLAGS.deep_kernel

        def regression_layer(input_feature, r, k, dim, num, flag_shortcut, l):
            with tf.name_scope('regression_layer' + l):
                result = slim.conv2d(input_feature, num*dim, [k, k], scope='regression_' + l, activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x 2^Rxs^2  filter k_size x k_size x Cin x 2^Rxs^2
                result = tf.reshape(result, [r[0], r[1], r[2], num, dim])   # B x H x W 2^R x s^2
                alpha = slim.conv2d(input_feature, num, [k, k], scope='alpha_' + l, activation_fn=tf.nn.softmax, biases_initializer=tf.zeros_initializer())  # B x H x W x R  filter k_size x k_size x Cin x R
                alpha = tf.expand_dims(alpha, 4)
                output_feature = tf.reduce_sum(result * alpha, axis=3)
                if flag_shortcut:
                    if FLAGS.skip_conv == 1:
                        skip_connection = slim.conv2d(input_feature, dim, [3, 3], scope='skip_connection_' + l, activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))
                        return output_feature + skip_connection
                    else:
                        return output_feature + input_feature
                else:
                    return output_feature
        if FLAGS.deep_layer == 1:
            regression = regression_layer(feature, reshape_size, kernel_size, 1, num_anchor, inner_channel==1, '1')
        else:
            regression = regression_layer(feature, reshape_size, kernel_size, inner_channel, num_anchor, True, '1')
            for i in range(2, FLAGS.deep_layer):
                regression = regression_layer(regression, reshape_size, kernel_size, inner_channel, num_anchor, True, str(i))
            regression = regression_layer(regression, reshape_size, kernel_size, 4, num_anchor, inner_channel == 4, str(FLAGS.deep_layer))
    sr_space = tf.depth_to_space(regression, 2, name='sr_space')
    # sr_space = pixelShuffler(regression, FLAGS.upscale)
    return sr_space

def simple_arch(data, FLAGS):
    #data_shape = tf.shape(data)
    #bilinear = tf.image.resize_bilinear(data,[data_shape[1]*FLAGS.upscale,data_shape[2]*FLAGS.upscale])
    net = slim.conv2d(data, 64, [3, 3], stride=1, scope='conv1')
    net = slim.batch_norm(net)
    net = slim.conv2d(net, 64, [3, 3], stride=1, scope='conv2')
    net = slim.batch_norm(net)
    net = slim.conv2d(net, 1, [3, 3], stride=1, scope='conv3', activation_fn=None)
    return net

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
        train_dataset = train_dataset_prepare(FLAGS)
        train_iterator = train_dataset.make_one_shot_iterator()
        #test dataset setup
        test_dir = '/scratch_net/ofsoundof/yawli/BSD68/'
        test_iterator = test_dataset_prepare(test_path_prepare, test_dir, FLAGS.sigma).make_initializable_iterator()
        #create dataset handle and iterator
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        image_gt, image_n = iterator.get_next()

    with tf.device('/device:GPU:' + FLAGS.sge_gpu):
        image_gt = image_gt/255*MAX_RGB
        image_n = image_n/255*MAX_RGB

        global_step = tf.Variable(1, name='global_step', trainable=False)
        image_dn = selected_model(image_n, FLAGS)
        MSE_dn, PSNR_dn = comp_mse_psnr(image_dn, image_gt, MAX_RGB)
        MSE_n, PSNR_n = comp_mse_psnr(image_n, image_gt, MAX_RGB)
        PSNR_gain = PSNR_dn - PSNR_n
        loss = tf.reduce_sum(tf.squared_difference(image_gt, image_dn)) + tf.reduce_sum(tf.losses.get_regularization_losses())
        train_op, global_step = training(loss, FLAGS, global_step)

    #train summary
    tf.summary.scalar('MSE_dn', MSE_dn)
    tf.summary.scalar('PSNR_dn', PSNR_dn)
    tf.summary.scalar('MSE_n', MSE_n)
    tf.summary.scalar('PSNR_n', PSNR_n)
    tf.summary.scalar('PSNR_gain', PSNR_gain)
    slim.summarize_collection(collection=tf.GraphKeys.TRAINABLE_VARIABLES)
    #test summaries
    psnr_validate = tf.placeholder(tf.float32)
    tf.summary.scalar('psnr_validate', psnr_validate)
    merged = tf.summary.merge_all()


    #Get the dataset handle
    train_handle = sess.run(train_iterator.string_handle())
    test_handle = sess.run(test_iterator.string_handle())

    print("Create checkpoint directory")
    # if not FLAGS.restore:
    FLAGS.checkpoint = FLAGS.checkpoint + '_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    with open('checkpoint.txt', 'w') as text_file: #save at the current directory, used for testing
        text_file.write(FLAGS.checkpoint)
    LOG_DIR = os.path.join('/scratch_net/ofsoundof/yawli/logs', FLAGS.checkpoint )
    # LOG_DIR = os.path.join('/home/yawli/Documents/hashnets/logs', FLAGS.checkpoint )
    # assert (not os.path.exists(LOG_DIR)), 'LOG_DIR %s already exists'%LOG_DIR

    print("Create summary file writer")
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)

    print("Create saver")
    saver = tf.train.Saver(max_to_keep=100)

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
    score_all = []
    model_index = []
    image_path_dn, _ = test_path_prepare(test_dir, FLAGS.sigma)
    image_path_dn = [os.path.join(LOG_DIR, os.path.basename(i)) for i in image_path_dn]
    i = sess.run(global_step) + 1
    print("Start iteration")
    while i <= FLAGS.max_iter:
        if i % FLAGS.summary_interval == 0:
            # test set5
            sess.run(test_iterator.initializer)
            score_per = np.zeros((4, 69))
            for j in range(68):

                img_dn, img_n, img_gt = sess.run([image_dn, image_n, image_gt], feed_dict={handle: test_handle})
                scipy.misc.toimage(np.squeeze(img_dn)*255/MAX_RGB, cmin=0, cmax=255).save(image_path_dn[j])
                # The current scipy version started to normalize all images so that min(data) become black and max(data)
                # become white. This is unwanted if the data should be exact grey levels or exact RGB channels.
                # The solution: scipy.misc.toimage(image_array, cmin=0.0, cmax=255).save('...') or use imageio library
                # from IPython import embed; embed();
                score = image_converter(np.squeeze(img_dn), np.squeeze(img_n), np.squeeze(img_gt), False, 0, MAX_RGB)
                score_per[:, j] = score
            score_per[:, -1] = np.mean(score_per[:, :-1], axis=1)
            print('PSNR results for Set5: SR {}, Bicubic {}'.format(score_per[1, -1], score_per[0, -1]))


            time_start = time.time()
            [_, i, l, mse_dn, mse_n, psnr_dn, psnr_n, psnr_gain, summary] =\
                sess.run([train_op, global_step, loss, MSE_dn, MSE_n, PSNR_dn, PSNR_n, PSNR_gain, merged],
                         feed_dict={handle: train_handle, psnr_validate: score_per[1, -1]})#, options=options, run_metadata=run_metadata)
            # from IPython import embed; embed(); exit()
            duration = time.time() - time_start
            train_writer.add_summary(summary, i)
            print("Iter %d, total loss %.5f; denoise (mse %.5f, psnr %.5f); noise (mse %.5f, psnr %.5f); psnr_gain:%f" % (i-1, l, mse_dn, psnr_dn, mse_n, psnr_n, psnr_gain))
            print('Training time for one iteration: {}'.format(duration))

            if (i-1) % FLAGS.checkpoint_interval == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'model'), global_step=i, latest_filename=checkpoint_filename, write_meta_graph=True)
                print("Saved checkpoint to {}".format(save_path))
                print("Flushing train writer")
                train_writer.flush()

            model_index.append(i)
            score_all.append([score_per[0, -1], score_per[1, -1], score_per[1, -1]-score_per[0, -1], score_per[2, -1], score_per[3, -1], score_per[3, -1]-score_per[2, -1]])
        else:
            [_, i, _] = sess.run([train_op, global_step, loss], feed_dict={handle: train_handle})

    #write to csv files
    descriptor5 = 'Average PSNR (dB)/SSIM for Set5, Scale ' + str(FLAGS.upscale) + ', different model parameters during training' + '\n'
    descriptor = [descriptor5, '', '', '', '', '', '']
    header_mul = ['Iteration'] + ['N', 'DN', 'Gain'] * 2
    model_index_array = np.expand_dims(np.array(model_index), axis=0)
    score_all_array = np.transpose(np.array(score_all))
    written_content = np.concatenate((model_index_array, score_all_array), axis=0)
    written_content_fmt = ['{:<8}'] + ['{:<8.4f}', '{:<8.4f}', '{:<8.4f}'] * 2
    content = content_generator_mul_line(descriptor, written_content, written_content_fmt, header_mul)
    file_writer(os.path.join(LOG_DIR, 'psnr_ssim.csv'), 'a', content)


    #write to pickle files
    variables = {'index': np.array(model_index), 'noise': score_all_array[0, :], 'denoise': score_all_array[1, :], 'gain': score_all_array[2, :]}
    pickle_out = open(os.path.join(LOG_DIR, 'psnr_bsd68.pkl'), 'wb')
    pickle.dump(variables, pickle_out)
    pickle_out.close()

def test(selected_model):
    # test_dir = '/home/yawli/Documents/hashnets/test/Set5'

    #input path
    test_dir = FLAGS.test_dir #'/home/yawli/Documents/hashnets/BSD68/'
    image_list_gt, image_list_n = test_path_prepare(test_dir, FLAGS.sigma)

    if len(FLAGS.checkpoint) == 0:
        with open('checkpoint.txt', 'r') as text_file:
            lines = text_file.readlines()
            FLAGS.checkpoint = os.path.join('/scratch_net/ofsoundof/yawli/logs', lines[0])
    #output path
    image_list_dn = [os.path.join(FLAGS.checkpoint, os.path.basename(img_name)) for img_name in image_list_gt]
    print(image_list_dn)

    #read input
    image_gt = []
    image_n = []
    for i in range(len(image_list_gt)):
        # image_n.append(np.expand_dims(np.expand_dims((scipy.misc.fromimage(Image.open(image_list_n[i])).astype(np.float32))/255*MAX_RGB, axis=0), axis=3))
        mat = sio.loadmat(image_list_n[i], struct_as_record=False)
        image_n_p = mat['img_n'].astype(np.float32)/255*MAX_RGB
        image_n.append(np.expand_dims(np.expand_dims(image_n_p[:-1, :-1], axis=0), axis=3))
        image_gt_p = scipy.misc.fromimage(Image.open(image_list_gt[i])).astype(np.float32)/255*MAX_RGB
        image_gt.append(image_gt_p[:-1, :-1])

    #build the model
    print('The used GPU is /device:GPU:'+FLAGS.sge_gpu)
    with tf.device('/device:GPU:'+FLAGS.sge_gpu):
        image_input_n = tf.placeholder(dtype=tf.float32, shape=(1, None, None, 1))
        global_step = tf.Variable(1, name='global_step', trainable=False)
        image_dn = selected_model(image_input_n, FLAGS)

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
    ckpt_num = len(all_checkpoints) #50 #search from the last ckpt_num checkpoints

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
            score_all = np.zeros((4, len(image_list_gt)+1))

            #image_list_output = []
            start_time = time.time()
            for image in range(len(image_list_gt)):
                image_dn_ = sess.run(image_dn, feed_dict={image_input_n: image_n[image]})
                scipy.misc.toimage(np.squeeze(image_dn_)*255/MAX_RGB, cmin=0, cmax=255).save(image_list_dn[image])

                flag = image == 2 and len(image_list_gt) == 14
                score = image_converter(np.squeeze(image_dn_), np.squeeze(image_n[image]), image_gt[image], flag, 0, MAX_RGB)
                score_all[:, image] = score
            score_all[:, -1] = np.mean(score_all[:, :-1], axis=1)
            duration = time.time() - start_time
            print('The execution time is {}'.format(duration))
            print('Compute PSNR and SSIM')
            print('PSNR N Ave. {:<8.4f}, per image {}.'.format(score_all[0, -1], np.round(score_all[0, 0:-1], decimals=2)))
            print('PSNR DN Ave. {:<8.4f}, per image {}.'.format(score_all[1, -1], np.round(score_all[1, 0:-1], decimals=2)))
            print('SSIM N Ave. {:<8.4f}, per image {}.'.format(score_all[2, -1], np.round(score_all[2, 0:-1], decimals=4)))
            print('SSIM DN Ave. {:<8.4f}, per image {}.'.format(score_all[3, -1], np.round(score_all[3, 0:-1], decimals=4)))

            model_name = os.path.basename(str(current_checkpoint))
            model_index[0, i] = int(model_name[6:])
            score_mean[:, i] = score_all[:, -1]
        print('\nWrite the average PSNR for all of the checkpoints to csv files')
        descriptor_psnr = 'Average PSNR (dB) for ' + os.path.basename(FLAGS.test_dir) + ', Sigma ' + str(FLAGS.sigma) + ', different model parameters during training' + '\n'
        descriptor_ssim = 'Average SSIM for ' + os.path.basename(FLAGS.test_dir) + ', Sigma ' + str(FLAGS.sigma) + ', different model parameters during training' + '\n'
        descriptor_mul = [descriptor_psnr, '', '', descriptor_ssim, '']
        header_mul = ['Iteration', 'N', 'DN', 'N', 'DN']
        written_content_fmt = ['{:<8} ', '{:<8.2f} ', '{:<8.2f} ', '{:<8.3f} ', '{:<8.3f} ']
        written_content = np.concatenate((model_index, score_mean), axis=0)
        content = content_generator_mul_line(descriptor_mul, written_content, written_content_fmt, header_mul)
        #write the average psnr and ssim values for all of the checkpoints in the current log directory
        file_writer(os.path.join(FLAGS.checkpoint, 'psnr_ssim.csv'), 'a', content)
        #write to pkl files
        variables = {'index': model_index, 'N_psnr': score_mean[0, :], 'DN_psnr': score_mean[1, :], 'N_ssim': score_mean[2, :], 'DN_ssim': score_mean[3, :]}
        pickle_out = open(os.path.join(FLAGS.checkpoint, 'psnr_ssim_test_BSD68.pkl'), 'wb')
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
        score_best = np.zeros([4, len(image_list_gt)+1])
        for image in range(len(image_list_gt)):
            image_dn_ = sess.run(image_dn, feed_dict={image_input_n: image_n[image]})
            #save image
            scipy.misc.toimage(np.squeeze(image_dn_)*255/MAX_RGB, cmin=0, cmax=255).save(image_list_dn[image])
            flag = image == 2 and len(image_list_gt) == 14
            score = image_converter(np.squeeze(image_dn_), np.squeeze(image_n[image]), image_gt[image], flag, 0, MAX_RGB)
            score_best[:, image] = score
        score_best[:, -1] = np.mean(score_best[:, :-1], axis=1)

        print('Save the best PSNR')
        model_name = os.path.basename(current_checkpoint)[6:]
        content = 'Dataset {:<8}, Model {:<4} '.format(os.path.basename(FLAGS.test_dir), FLAGS.model_selection)\
                  + '{:<8} {:<8.2f} {:<8.2f} {:<8.4f} {:<8.4f} '.format(model_name, score_best[0][-1], score_best[1][-1], score_best[2][-1], score_best[3][-1]) + '\n'
        #write the best average psnr in a shared file under the project directory
        file_writer('/home/yawli/Documents/hashnets/final_test_results_average_denoise.csv', 'a', content)

        descriptor_mul = ['Dataset {}, Model {} \n'.format(os.path.basename(FLAGS.test_dir), FLAGS.model_selection), '', '', '', '']
        header_mul = ['Image', 'psnr_n', 'psnr_dn', 'ssim_n', 'ssim_dn']
        written_content_fmt = ['{:<8} ', '{:<8} ', '{:<8} ', '{:<8} ', '{:<8} ']
        image_name = [os.path.basename(image)[4:7] if image[-6:] == 'HR.png' else os.path.splitext(os.path.basename(image))[0] for image in image_list_gt]
        score_best[:2, :] = np.round(score_best[:2, :], decimals=2)
        score_best[2:, :] = np.round(score_best[2:, :], decimals=3)
        written_content = np.concatenate((np.expand_dims(np.array(image_name), axis=0), score_best[:, :-1]), axis=0)
        content = content_generator_mul_line(descriptor_mul, written_content, written_content_fmt, header_mul)
        #write the best per image psnr in a shared file under the project directory
        file_writer('/home/yawli/Documents/hashnets/final_test_results_perimage_denoise.csv', 'a', content)

    if FLAGS.test_runtime_compute:
        print('\nGet the runtime')
        current_checkpoint = all_checkpoints[-1]
        assert current_checkpoint is not None
        saver.restore(sess, current_checkpoint)

        #report runtime
        # import matplotlib.pyplot as plt
        test_iter = 1
        duration = np.zeros(len(image_list_gt)+2)
        for image in range(len(image_list_gt)):
            per_image_start = time.time()
            for i in range(test_iter):
                sess.run(image_dn, {image_input_n: image_n[image]})#, options=options, run_metadata=run_metadata)
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
        duration[0] = duration[1]/len(image_list_gt)
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
        # image_name = [os.path.basename(image)[4:7] if image[-6:] == 'HR.png' else os.path.splitext(os.path.basename(image))[0] for image in image_list_gt]
        # time_identity = ['Ave.', 'Total'] + image_name
        # written_content = np.stack((np.array(time_identity), np.round(duration, decimals=3)), axis=0)

        descriptor_mul = ['']
        header_mul = [FLAGS.model_selection]
        written_content_fmt = ['{:<8.3f}\t']
        written_content = np.expand_dims(np.round(duration, decimals=3), axis=0)

        content = content_generator_mul_line(descriptor_mul, written_content, written_content_fmt, header_mul)
        time_dir = os.path.join('/home/yawli/Documents/hashnets/time_BSD68_denoise.csv')
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
    # if FLAGS.model_selection == 1:
    #     selected_model = fast_hashnet
    # elif FLAGS.model_selection == 2:
    #     selected_model = fast_hashnet_restore
    # elif FLAGS.model_selection == 3:
    #     selected_model = deep_hashnet
    # elif FLAGS.model_selection == 4:
    #     selected_model = deep_hashnet_stack
    # elif FLAGS.model_selection == 5:
    #     selected_model = vdsr
    # elif FLAGS.model_selection == 6:
    #     selected_model = srcnn
    # elif FLAGS.model_selection == 7:
    #     selected_model = fsrcnn
    # elif FLAGS.model_selection == 8:
    #     selected_model = espcn
    # elif FLAGS.model_selection == 9:
    #     selected_model = srresnet
    # elif FLAGS.model_selection == 10:
    #     selected_model = espcn_comp
    # elif FLAGS.model_selection == 11:
    #     selected_model = carn_comp
    # else:
    #     selected_model = deep_hashnet_more
    selected_model = carn_denoise
    # print('Selected model is {}'.format(selected_model))
    if len(FLAGS.test_dir):
        print(FLAGS.test_dir)
        test(selected_model)

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
    parser.add_argument('--sigma', type=int, default=15, help='Upscaling factor')
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
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--q_capacity', type=int, default=40000, help='Queue capacity')
    parser.add_argument('--q_min', type=int, default=10000, help='Minimum number of elements after dequeue')
    parser.add_argument('--crop_dim', type=int, default=64, help='Crop dimension')
    parser.add_argument('--num_crop', type=int, default=100, help='Number of crops per image')

    FLAGS, unparsed = parser.parse_known_args()
    print("FLAGS: {}".format(FLAGS))
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
