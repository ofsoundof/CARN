import scipy.misc
from PIL import Image
import datetime
import time
import argparse
import sys
import pickle
import os
import tensorflow as tf
import json
from model import *
from utils import *
# from tensorflow.python.client import timeline
# from tensorflow.python import debug as tf_debug

MAX_RGB = 1.0
mean_Y = 0 #111.6804 / 255 * MAX_RGB
DATASETS = {'div2k': load_div2k, 'standard91': load_standard91}

# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
# print('Cuda_visible_devices: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
# print(type(os.environ["CUDA_VISIBLE_DEVICES"]))
# cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

def run_test_train_dataset(selected_model):

    #Create the sess, and use some options for better using gpu
    print("Create session")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    config.allow_soft_placement = True  # If a operation is not define it the default device, let it execute in another.
    sess = tf.Session(config=config)

    with tf.device('/cpu:0'):
        #train dataset setup
        train_dataset = train_dataset_prepare(FLAGS)
        train_iterator = train_dataset.make_one_shot_iterator()
        #test dataset setup
        test_dir = os.path.join(FLAGS.storage_path, 'Datasets/Test')
        test_set5_iterator = test_dataset_prepare(test_path_prepare, test_dir + 'Set5_X' + str(FLAGS.upscale)).make_initializable_iterator()
        test_set14_iterator = test_dataset_prepare(test_path_prepare, test_dir + 'Set14_X' + str(FLAGS.upscale)).make_initializable_iterator()
        #create dataset handle and iterator
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        low, mid, high = iterator.get_next()

    with tf.device('/device:GPU:' + FLAGS.sge_gpu):
        low = (low-mean_Y)/255*MAX_RGB
        mid = mid/255*MAX_RGB
        high = high/255*MAX_RGB

        global_step = tf.Variable(1, name='global_step', trainable=False)
        if FLAGS.restore and FLAGS.model_selection == 10:
            with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                saver = tf.train.import_meta_graph('/scratch_net/ofsoundof/yawli/logs/2018-05-11-13-21-55-ESPCN_COMP_UP4M10F20ITER_500000/model-510000.meta')
                saver.restore(sess, tf.train.latest_checkpoint('/scratch_net/ofsoundof/yawli/logs/2018-05-11-13-21-55-ESPCN_COMP_UP4M10F20ITER_500000'))
                graph = tf.get_default_graph()
                # from IPython import embed; embed(); exit()
                feature_19 = tf.stop_gradient(graph.get_tensor_by_name('feature_layer19/p_re_lu/add:0'))

                with slim.arg_scope([slim.conv2d], stride=1,
                            weights_initializer=tf.keras.initializers.he_normal(),
                            weights_regularizer=slim.l2_regularizer(0.0001)):

                    for i in range(20, FLAGS.deep_feature_layer):
                        net = slim.conv2d(feature_19, 32, [3, 3], scope='feature_layer' + str(i), activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))
                    net = slim.conv2d(net, FLAGS.upscale ** 2, [3, 3], scope='output_layer', activation_fn=None)
                sr_space = tf.depth_to_space(net, FLAGS.upscale, 'space')
                sr = sr_space + mid
                annealing_factor = tf.shape(sr)[0]
                MSE_sr, PSNR_sr = comp_mse_psnr(sr, high, MAX_RGB)
                MSE_interp, PSNR_interp = comp_mse_psnr(mid, high, MAX_RGB)
                PSNR_gain = PSNR_sr - PSNR_interp
                loss = tf.reduce_sum(tf.squared_difference(high, sr)) + tf.reduce_sum(tf.losses.get_regularization_losses())
                train_op, global_step = training(loss, FLAGS, global_step)

        else:
            sr, annealing_factor = selected_model(low, mid, FLAGS, global_step)
            #annealing_factor = global_step
            MSE_sr, PSNR_sr = comp_mse_psnr(sr, high, MAX_RGB)
            MSE_interp, PSNR_interp = comp_mse_psnr(mid, high, MAX_RGB)
            PSNR_gain = PSNR_sr - PSNR_interp
            loss = tf.reduce_sum(tf.squared_difference(high, sr)) + tf.reduce_sum(tf.losses.get_regularization_losses())
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
    psnr_test_train_set14 = tf.placeholder(tf.float32)
    tf.summary.scalar('psnr_test_train_set5', psnr_test_train_set5)
    tf.summary.scalar('psnr_test_train_set14', psnr_test_train_set14)
    merged = tf.summary.merge_all()


    #Get the dataset handle
    train_handle = sess.run(train_iterator.string_handle())
    test_set5_handle = sess.run(test_set5_iterator.string_handle())
    test_set14_handle = sess.run(test_set14_iterator.string_handle())

    print("Create checkpoint directory")
    # if not FLAGS.restore:
    # FLAGS.checkpoint = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '-' + FLAGS.checkpoint
    with open('checkpoint.txt', 'w') as text_file: #save at the current directory, used for testing
        text_file.write(FLAGS.checkpoint)
    LOG_DIR = os.path.join(FLAGS.storage_path, 'logs_carn', FLAGS.model_flag, FLAGS.checkpoint)
    assert (not os.path.exists(LOG_DIR)), 'LOG_DIR %s already exists'%LOG_DIR

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
    score5_all = []
    score14_all = []
    model_index = []
    _, _, path_set5 = test_path_prepare(test_dir + 'Set5_X' + str(FLAGS.upscale))
    path_set5 = [os.path.join(LOG_DIR, os.path.basename(i)) for i in path_set5]
    _, _, path_set14 = test_path_prepare(test_dir + 'Set14_X' + str(FLAGS.upscale))
    path_set14 = [os.path.join(LOG_DIR, os.path.basename(i)) for i in path_set14]
    i = sess.run(global_step) + 1
    print("Start iteration")
    while i <= FLAGS.max_iter:
        if i % FLAGS.summary_interval == 0:
            # test set5
            sess.run(test_set5_iterator.initializer)
            score5 = np.zeros((4, 6))
            for j in range(5):
                _sr, _mid, _high = sess.run([sr, mid, high], feed_dict={handle: test_set5_handle})
                scipy.misc.toimage(np.squeeze(_sr)*255/MAX_RGB, cmin=0, cmax=255).save(path_set5[j])
                # The current scipy version started to normalize all images so that min(data) become black and max(data)
                # become white. This is unwanted if the data should be exact grey levels or exact RGB channels.
                # The solution: scipy.misc.toimage(image_array, cmin=0.0, cmax=255).save('...') or use imageio library
                score = image_converter(np.squeeze(_sr), np.squeeze(_mid), np.squeeze(_high), False, FLAGS.upscale, MAX_RGB)
                score5[:, j] = score
            score5[:, -1] = np.mean(score5[:, :-1], axis=1)
            print('PSNR results for Set5: SR {}, Bicubic {}'.format(score5[1, -1], score5[0, -1]))

            #test set14
            sess.run(test_set14_iterator.initializer)
            score14 = np.zeros((4, 15))
            for j in range(14):
                _sr, _mid, _high = sess.run([sr, mid, high], feed_dict={handle: test_set14_handle})
                scipy.misc.toimage(np.squeeze(_sr)*255/MAX_RGB, cmin=0, cmax=255).save(path_set14[j])
                score = image_converter(np.squeeze(_sr), np.squeeze(_mid), np.squeeze(_high), j == 2, FLAGS.upscale, MAX_RGB)
                score14[:, j] = score
            score14[:, -1] = np.mean(score14[:, :-1], axis=1)
            print('PSNR results for Set14: SR {}, Bicubic {}'.format(score14[1, -1], score14[0, -1]))

            time_start = time.time()
            [_, i, af, l, mse_sr, mse_interp, psnr_sr, psnr_interp, psnr_gain, summary] =\
                sess.run([train_op, global_step, annealing_factor, loss, MSE_sr, MSE_interp, PSNR_sr, PSNR_interp, PSNR_gain, merged],
                         feed_dict={handle: train_handle, psnr_test_train_set5: score5[1, -1], psnr_test_train_set14: score14[1, -1]})#, options=options, run_metadata=run_metadata)
            duration = time.time() - time_start
            train_writer.add_summary(summary, i)
            print("Iter %d, annealing factor %d, total loss %.5f; sr (mse %.5f, psnr %.5f); interp (mse %.5f, psnr %.5f); psnr_gain:%f" % (i-1, af, l, mse_sr, psnr_sr, mse_interp, psnr_interp, psnr_gain))
            print('Training time for one iteration: {}'.format(duration))

            if (i-1) % FLAGS.checkpoint_interval == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'model'), global_step=i, latest_filename=checkpoint_filename, write_meta_graph=True)
                print("Saved checkpoint to {}".format(save_path))
                print("Flushing train writer")
                train_writer.flush()

            model_index.append(i)
            score5_all.append([score5[0, -1], score5[1, -1], score5[1, -1]-score5[0, -1], score5[2, -1], score5[3, -1], score5[3, -1]-score5[2, -1]])
            score14_all.append([score14[0, -1], score14[1, -1], score14[1, -1]-score14[0, -1], score14[2, -1], score14[3, -1], score14[3, -1]-score14[2, -1]])
        else:
            [_, i, _] = sess.run([train_op, global_step, loss], feed_dict={handle: train_handle})

    #write to csv files
    descriptor5 = 'Average PSNR (dB)/SSIM for Set5, Scale ' + str(FLAGS.upscale) + ', different model parameters during training' + '\n'
    descriptor14 = 'Average PSNR (dB)/SSIM for Set14, Scale ' + str(FLAGS.upscale) + ', different model parameters during training' + '\n'
    descriptor = [descriptor5, '', '', '', '', '', '', descriptor14, '', '', '', '', '', '']
    header_mul = (['Iteration'] + ['Bicu', 'SR', 'Gain'] * 2) * 2
    model_index_array = np.expand_dims(np.array(model_index), axis=0)
    score5_all_array = np.transpose(np.array(score5_all))
    score14_all_array = np.transpose(np.array(score14_all))
    written_content = np.concatenate((model_index_array, score5_all_array, model_index_array, score14_all_array), axis=0)
    written_content_fmt = (['{:<8}'] + ['{:<8.4f}', '{:<8.4f}', '{:<8.4f}'] * 2) * 2
    content = content_generator_mul_line(descriptor, written_content, written_content_fmt, header_mul)
    file_writer(os.path.join(LOG_DIR, 'psnr_ssim.csv'), 'a', content)


    #write to pickle files
    variables = {'index': np.array(model_index), 'bicubic': score5_all_array[0, :], 'sr': score5_all_array[1, :], 'gain': score5_all_array[2, :]}
    pickle_out = open(os.path.join(LOG_DIR, 'psnr_set5.pkl'), 'wb')
    pickle.dump(variables, pickle_out)
    pickle_out.close()

    variables = {'index': np.array(model_index), 'bicubic': score14_all_array[0, :], 'sr': score14_all_array[1, :], 'gain': score14_all_array[2, :]}
    pickle_out = open(os.path.join(LOG_DIR, 'psnr_set14.pkl'), 'wb')
    pickle.dump(variables, pickle_out)
    pickle_out.close()

def test(selected_model):
    # test_dir = '/home/yawli/Documents/hashnets/test/Set5'

    # #input path
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

    # test_dir_low = FLAGS.test_dir + 'y2/'
    # test_dir_mid = FLAGS.test_dir + 'y2_mid/'
    # test_dir_high = FLAGS.test_dir + 'y1/'
    # image_list_low = sorted(glob.glob(test_dir_low + '*.png'))
    # image_list_mid = sorted(glob.glob(test_dir_mid + '*.png'))
    # image_list_high = sorted(glob.glob(test_dir_high + '*.png'))

    if len(FLAGS.checkpoint) == 0:
        with open('checkpoint.txt', 'r') as text_file:
            lines = text_file.readlines()
            FLAGS.checkpoint = os.path.join(FLAGS.storage_path, 'logs_carn', FLAGS.model_flag, lines[0])
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



# Use queue to load the input.
def run_train(selected_model):
    #Don't understand the following lines of code. But I'm sure that I should be careful about it since it once caused
    #an annoying bug that confuses me for several days.
    #from tensorflow.python.client import device_lib
    #def get_available_gpus():
    #    local_device_protos = device_lib.list_local_devices()
    #    return [x.name for x in local_device_protos if x.device_type == 'GPU']
    #print("GPU TEST")
    #print(get_available_gpus())

    #Build the model. Note that when working on the SGE batch system, the GPU should be specified as the reserved ones
    #allocated by the qsub or qrsh command if you don't want your program always use the default GPU0, which may occupy
    #other's resources. Thus, I debug it with the hint of Gu Shuhang, transfer the parameter of reserved gpu ids from
    #bash script into the python&tensorflow code here using the argparse module.
    # print(FLAGS.sge_gpu_all[0])
    # print(FLAGS.sge_gpu_all[1])
    # from IPython import embed; embed(); exit()
    with tf.device('/cpu:0'):
        if FLAGS.model_selection == 9:
            low, mid, high = load_standard91(FLAGS)
        else:
            low, mid, high = DATASETS[FLAGS.dataset](FLAGS)

    with tf.device('/device:GPU:' + FLAGS.sge_gpu):
        low = (low-mean_Y)/255*MAX_RGB
        mid = mid/255*MAX_RGB
        high = high/255*MAX_RGB

        global_step = tf.Variable(1, name='global_step', trainable=False)
        sr, annealing_factor = selected_model(low, mid, FLAGS, global_step)
        MSE_sr, PSNR_sr = comp_mse_psnr(sr, high, MAX_RGB)
        MSE_interp, PSNR_interp = comp_mse_psnr(mid, high, MAX_RGB)
        PSNR_gain = PSNR_sr - PSNR_interp
        loss = tf.reduce_sum(tf.squared_difference(high, sr)) + tf.reduce_sum(tf.losses.get_regularization_losses())
        train_op, global_step = training(loss, FLAGS, global_step)

    #Add summary
    tf.summary.scalar('MSE_sr', MSE_sr)
    tf.summary.scalar('PSNR_sr', PSNR_sr)
    tf.summary.scalar('MSE_interp', MSE_interp)
    tf.summary.scalar('PSNR_interp', PSNR_interp)
    tf.summary.scalar('PSNR_gain', PSNR_gain)
    slim.summarize_collection(collection=tf.GraphKeys.TRAINABLE_VARIABLES)
    # from IPython import embed; embed(); exit()

    #visualize filters
    # training_vars = tf.trainable_variables()
    # feature_matrix1 = None
    # feature_matrix2 = None
    # for v in training_vars:
    #     if v.name.startswith('feature_layer1/weights'):
    #         feature_matrix1 = v # 5 x 5 x 1 x 64
    #     if v.name.startswith('feature_layer2/weights'):
    #         feature_matrix2 = v # 3 x 3 x 64 x 28
    # feature_on_grid1 = put_kernels_on_grid(feature_matrix1, pad=1) # 1 x 8x5 x 8x5 x 1
    # feature_on_grid2 = put_kernels_on_grid(feature_matrix2, pad=1) # 1 x 4x3 x 7x3 x 64
    # feature_on_grid2 = tf.transpose(feature_on_grid2, [1, 2, 0, 3]) # 4x3 x 7x3 x 1 x 64
    # feature_on_grid2 = put_kernels_on_grid(feature_on_grid2, pad=4)
    # tf.summary.image('feature_layer1_matrix', feature_on_grid1, max_outputs=10)
    # tf.summary.image('feature_layer2_matrix', feature_on_grid2, max_outputs=10)
    merged = tf.summary.merge_all()

    #Create the sess, and use some options for better using gpu
    print("Create session")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    config.allow_soft_placement = True  # If a operation is not define it the default device, let it execute in another.
    sess = tf.Session(config=config)

    print("Create checkpoint directory")
    if not FLAGS.restore:
        FLAGS.checkpoint = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '-' + FLAGS.checkpoint
        with open('checkpoint.txt', 'w') as text_file: #save at the current directory, used for testing
            text_file.write(FLAGS.checkpoint)
    LOG_DIR = os.path.join('/home/yawli/Documents/hashnets/logs', FLAGS.checkpoint )
    # assert (not os.path.exists(LOG_DIR)), 'LOG_DIR %s already exists'%LOG_DIR

    print("Create summary file writer")
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)

    print("Create saver")
    saver = tf.train.Saver(max_to_keep=2000)

    #Always init, then optionally restore
    print("Initialization")
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer()) #required by tf.train.match_filenames_once
    #Coordinator
    print("Create coordinator")
    coord = tf.train.Coordinator()
    print("Start queue runner")
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if FLAGS.restore:
        print('Restore model from checkpoint {}'.format(tf.train.latest_checkpoint(LOG_DIR)))
        saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))
        checkpoint_filename = 'checkpoint'
    else:
        checkpoint_filename = 'checkpoint'

    i = sess.run(global_step) + 1
    print("Start iteration")
    while i <= FLAGS.max_iter:
        if i%FLAGS.summary_interval == 0:
            if i%FLAGS.checkpoint_interval == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'model'), global_step=i, latest_filename=checkpoint_filename, write_meta_graph=False)
                print("Saved checkpoint to {}".format(save_path))
                print("Flushing train writer")
                train_writer.flush()
            time_start = time.time()
            # from IPython import embed; embed(); exit()
            [_, i, af, l, mse_sr, mse_interp, psnr_sr, psnr_interp, psnr_gain, summary] = sess.run([train_op, global_step, annealing_factor, loss, MSE_sr, MSE_interp, PSNR_sr, PSNR_interp, PSNR_gain, merged])#, options=options, run_metadata=run_metadata)
            duration = time.time() - time_start
            train_writer.add_summary(summary, i)
            print("Iter %d, annealing factor %d, total loss %.5f; sr (mse %.5f, psnr %.5f); interp (mse %.5f, psnr %.5f); psnr_gain:%f" % (i-1, af, l, mse_sr, psnr_sr, mse_interp, psnr_interp, psnr_gain))
            print('Training time for one iteration: {}'.format(duration))
        else:
            [_, i, _] = sess.run([train_op, global_step, loss])
    print("Request queue stop")
    coord.request_stop()
    coord.join(threads)
    print("Queue stops!")

def test_restore():
    # test_dir = '/home/yawli/Documents/hashnets/test/Set5'
    test_dir_low_Set5 = FLAGS.test_dir + '/Set5_X' + str(FLAGS.upscale) + '_lowY/'
    test_dir_mid_Set5 = FLAGS.test_dir + '/Set5_X' + str(FLAGS.upscale) + '_midY/'
    test_dir_high_Set5 = FLAGS.test_dir + '/Set5_X' + str(FLAGS.upscale) + '_highY/'
    test_dir_low_Set14 = FLAGS.test_dir + '/Set14_X' + str(FLAGS.upscale) + '_lowY/'
    test_dir_mid_Set14 = FLAGS.test_dir + '/Set14_X' + str(FLAGS.upscale) + '_midY/'
    test_dir_high_Set14 = FLAGS.test_dir + '/Set14_X' + str(FLAGS.upscale) + '_highY/'
    image_list_low = sorted(glob.glob(test_dir_low_Set5 +'*.png')) + sorted(glob.glob(test_dir_low_Set14 +'*.png'))
    print(image_list_low)
    image_list_mid = sorted(glob.glob(test_dir_mid_Set5 +'*.png')) + sorted(glob.glob(test_dir_mid_Set14 +'*.png'))
    print(image_list_mid)
    image_list_high = sorted(glob.glob(test_dir_high_Set5 +'*.png')) + sorted(glob.glob(test_dir_high_Set14 +'*.png'))
    print(image_list_high)

    image_list_sr = [os.path.join(FLAGS.checkpoint, os.path.basename(img_name)) for img_name in image_list_low]
    print(image_list_sr)
    image_low = []
    image_mid = []

    for i in range(len(image_list_low)):
        image_low.append(np.expand_dims(np.expand_dims((scipy.misc.fromimage(Image.open(image_list_low[i])) - mean_Y)/255*MAX_RGB, axis=0), axis=3))
        image_mid.append(np.expand_dims(np.expand_dims(scipy.misc.fromimage(Image.open(image_list_mid[i]))/255*MAX_RGB, axis=0), axis=3))

    if len(FLAGS.checkpoint) == 0:
        with open('checkpoint.txt', 'r') as text_file:
            lines = text_file.readlines()
            FLAGS.checkpoint = os.path.join(os.path.dirname(os.path.dirname(FLAGS.test_dir)), 'logs', lines[0])

    # super-resolve images using trained network and save the derived sr images
    image_input_low = tf.placeholder(dtype=tf.float32, shape=(1, None, None, 1))
    image_input_mid = tf.placeholder(dtype=tf.float32, shape=(1, None, None, 1))
    sr = fast_hashnet_restore(image_input_low, image_input_mid, FLAGS)*255/MAX_RGB

    print("Create session")
    config = tf.ConfigProto()
    sess = tf.Session(config=config)

    print("Load checkpoint")
    assert FLAGS.checkpoint != ''
    checkpoint_to_resume = '/home/yawli/Documents/hashnets/logs/' + FLAGS.checkpoint
    print("Resuming from checkpoint {}".format(checkpoint_to_resume))
    all_checkpoints = tf.train.get_checkpoint_state(checkpoint_to_resume).all_model_checkpoint_paths
    ckpt_num = len(all_checkpoints)
    print("\nTest for every saved checkpoint")

    #Used for getting the runtime
    class TimeLiner:
        _timeline_dict = None

        def update_timeline(self, chrome_trace):
            # convert crome trace to python dict
            chrome_trace_dict = json.loads(chrome_trace)
            # for first run store full trace
            if self._timeline_dict is None:
                self._timeline_dict = chrome_trace_dict
            # for other - update only time consumption, not definitions
            else:
                for event in chrome_trace_dict['traceEvents']:
                    # events time consumption started with 'ts' prefix
                    if 'ts' in event:
                        self._timeline_dict['traceEvents'].append(event)

        def save(self, f_name):
            with open(f_name, 'w') as f:
                json.dump(self._timeline_dict, f)
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    many_runs_timeline = TimeLiner()

    for i in [ckpt_num-1]:
        current_checkpoint = all_checkpoints[i]
        print("\n\nCurrent checkpoint: {}".format(current_checkpoint))
        assert current_checkpoint is not None
        saver = tf.train.Saver()
        saver.restore(sess, current_checkpoint)

        # import matplotlib.pyplot as plt
        start_time = time.time()
        # index = []
        for iter in range(100):
            for image in range(len(image_list_low)):
                sr_ = sess.run(sr, {image_input_low: image_low[image], image_input_mid: image_mid[image]})#, options=options, run_metadata=run_metadata)

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
        duration = time.time() - start_time
        # # from IPython import embed; embed(); exit()
        # index1 = np.concatenate(index)
        # plt.hist(index1, bins=512)  # arguments are passed to np.histogram
        # plt.title("Total")
        # plt.show()
        # many_runs_timeline.save(os.path.join(FLAGS.checkpoint, 'timeline_merged_%d_runs_1.5.json' % len(image_list_low)))
        print('The execution time is {}'.format(duration))

def test_restore_rgb(selected_model):
    """
    restore rgb images
    """
    # test_dir = '/home/yawli/Documents/hashnets/test/Set5'
    test_dir_low = FLAGS.test_dir + '_X' + str(FLAGS.upscale) + '_lowY/'
    test_dir_mid = FLAGS.test_dir + '_X' + str(FLAGS.upscale) + '_midY/'
    test_dir_high = FLAGS.test_dir + '_X' + str(FLAGS.upscale) + '_highY/'
    test_dir_mid_rgb = FLAGS.test_dir + '_X' + str(FLAGS.upscale) + '_midRGB/'
    image_list_low = sorted(glob.glob(test_dir_low + '*.png'))
    print(image_list_low)
    image_list_mid = sorted(glob.glob(test_dir_mid + '*.png'))
    print(image_list_mid)
    image_list_high = sorted(glob.glob(test_dir_high + '*.png'))
    print(image_list_high)
    image_list_mid_rgb = sorted(glob.glob(test_dir_mid_rgb + '*.png'))
    image_list_sr = [os.path.join(FLAGS.checkpoint, os.path.basename(img_name)) for img_name in image_list_low]
    print(image_list_sr)
    image_low = []
    image_mid = []
    image_high = []
    image_mid_rgb = []

    for i in range(len(image_list_low)):
        image_low.append(np.expand_dims(np.expand_dims((scipy.misc.fromimage(Image.open(image_list_low[i])).astype(float) - mean_Y)/255*MAX_RGB, axis=0), axis=3))
        image_mid.append(np.expand_dims(np.expand_dims(scipy.misc.fromimage(Image.open(image_list_mid[i])).astype(float)/255*MAX_RGB, axis=0), axis=3))
        image_high.append(np.expand_dims(np.expand_dims(scipy.misc.fromimage(Image.open(image_list_high[i])).astype(float)/255*MAX_RGB, axis=0), axis=3))
        image_mid_rgb.append(scipy.misc.fromimage(Image.open(image_list_mid_rgb[i])).astype(float))
    # from IPython import embed; embed()
    if len(FLAGS.checkpoint) == 0:
        with open('checkpoint.txt', 'r') as text_file:
            lines = text_file.readlines()
            FLAGS.checkpoint = os.path.join(os.path.dirname(os.path.dirname(FLAGS.test_dir)), 'logs', lines[0])
    with tf.device('/device:GPU:' + FLAGS.sge_gpu):
        # super-resolve images using trained network and save the derived sr images
        global_step = tf.Variable(1, name='global_step', trainable=False)
        image_input_low = tf.placeholder(dtype=tf.float32, shape=(1, None, None, 1))
        image_input_mid = tf.placeholder(dtype=tf.float32, shape=(1, None, None, 1))
        sr,annealing_factor = selected_model(image_input_low, image_input_mid, FLAGS, global_step)
        # sr = sr
    print("Create session")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    config.allow_soft_placement = True  # If a operation is not define it the default device, let it execute in another.
    sess = tf.Session(config=config)

    print("Load checkpoint")
    assert FLAGS.checkpoint != ''
    checkpoint_to_resume = FLAGS.checkpoint
    print("Resuming from checkpoint {}".format(checkpoint_to_resume))

    # checkpoint_to_resume = '/home/yawli/Documents/hashnets/logs/2018-05-09-20-47-14-ARN_deep_reproduce_noneR_noOutput_feature2_UP4L1C16A16M3ITER_500000'
    all_checkpoints = glob.glob(checkpoint_to_resume + '/model-*')
    all_checkpoints = [os.path.splitext(x)[0] for x in all_checkpoints if x.endswith('index')]
    all_checkpoints = sorted(all_checkpoints, key=lambda x: int(os.path.basename(x)[6:]))
    print(all_checkpoints[0], all_checkpoints[-1])
    # all_checkpoints = tf.train.get_checkpoint_state(checkpoint_to_resume).all_model_checkpoint_paths
    ckpt_num = len(all_checkpoints)
    print("\nTest for every saved checkpoint")
    saver = tf.train.Saver()
    boundarypixels = FLAGS.upscale
    psnr_mean_sr_gather = []
    model_index = []
    for i in range(ckpt_num):#[ckpt_num-1]:#
        current_checkpoint = all_checkpoints[i]
        print("\n\nCurrent checkpoint: {}".format(current_checkpoint))
        assert current_checkpoint is not None
        saver.restore(sess, current_checkpoint)
        psnr_set_sr = []
        for image in range(len(image_list_low)):
            _sr = sess.run(sr, {image_input_low: image_low[image], image_input_mid: image_mid[image]})#, options=options, run_metadata=run_metadata)
            _sr = _sr[0, boundarypixels:-boundarypixels, boundarypixels:-boundarypixels, 0]
            _high = image_high[image][0, boundarypixels:-boundarypixels, boundarypixels:-boundarypixels, 0]
            if image == 2 and len(image_list_low) == 14:
                _sr = (_sr*255/MAX_RGB-16)/219.859*256/255*MAX_RGB
                _high = (_high*255/MAX_RGB-16)/219.859*256/255*MAX_RGB
            psnr_sr = 10 * np.log10(MAX_RGB ** 2 / (np.mean(np.square(_high - _sr))))
            psnr_set_sr.append(psnr_sr)
        psnr_mean_sr_gather.append(np.mean(psnr_set_sr))
        model_index.append(int(os.path.basename(current_checkpoint)[6:]))
    with open(os.path.join(FLAGS.checkpoint, 'psnr_ssim.csv'), 'a') as csv_file:
        csv_file.write('\n\nPSNR for saved {} checkpoints\n'.format(ckpt_num))
        content = '{:<10} '.format('Model:')
        for i in range(len(model_index)):
            content = content + '{:<8} '.format(model_index[i])
        content += '\n'
        csv_file.write(content)
        content = '{:<10} '.format('SR:')
        for i in range(len(model_index)):
            content = content + "{:<8.4f} ".format(psnr_mean_sr_gather[i])
        content += '\n'
        csv_file.write(content)
    value = np.array(psnr_mean_sr_gather)
    index = np.array(model_index)
    i_max = np.argmax(value)
    index_max = index[i_max]
    value_max = value[i_max]
    print('({}, {}): the maximum PSNR {} appears at {}th iteration'.format(value_max, index_max, value_max, index_max))
    current_checkpoint = all_checkpoints[i_max]
    print("\n\nCheckpoints with largest PSNR: {}".format(current_checkpoint))
    assert current_checkpoint is not None
    saver.restore(sess, current_checkpoint)
    for image in range(len(image_list_low)):
        _sr = sess.run(sr, {image_input_low: image_low[image], image_input_mid: image_mid[image]})#, options=options, run_metadata=run_metadata)
        _sr = _sr*255/MAX_RGB
        scipy.misc.toimage(np.squeeze(_sr), cmin=0, cmax=255).save(image_list_sr[image])
        # rgb_mid = image_mid_rgb[image]
        # ycbcr_mid = rgbtoycbcr(rgb_mid)
        # ycbcr_sr = np.concatenate([np.squeeze(_sr, axis=0), ycbcr_mid[:, :, 1:3]], axis=2)
        # rgb_sr = ycbcrtorgb(ycbcr_sr)
        # if image == 2 and len(image_list_low) == 14:
        #     scipy.misc.toimage(np.squeeze(rgb_sr[:,:,0]), cmin=0, cmax=255).save(image_list_sr[image])
        # else:
        #     scipy.misc.toimage(rgb_sr, cmin=0, cmax=255).save(image_list_sr[image])

    #     content = '{:<8} '.format('') + '{:<8} '.format('Total') + '{:<8}\n'.format('Average')
    #     csv_file.write(content)
    #     content = '{:<8} '.format(os.path.basename(FLAGS.test_dir)) + "{:<8.4f} ".format(duration) + "{:<8.4f}\n".format(duration/test_iter/len(image_list_low))
    #     csv_file.write(content)
    #     # test_dir = '/home/yawli/Documents/hashnets/test/Set5'
    # time_dir = os.path.join(os.path.dirname(os.path.dirname(FLAGS.test_dir)), 'time_' + os.path.basename(FLAGS.test_dir) + '.csv')
    # with open(time_dir, 'a') as csv_file:
    #     content = '{:<8}\t '.format(FLAGS.model_selection) + "{:<8.4f}\t".format(duration) + "{:<8.4f}\n".format(duration/test_iter/len(image_list_low))
    #     csv_file.write(content)

def test_restore_gpu(selected_model):
    # test_dir = '/home/yawli/Documents/hashnets/test/Set5'
    channel = 'RGB' if FLAGS.model_selection == 9 else 'Y'
    test_dir_low = FLAGS.test_dir + '_X' + str(FLAGS.upscale) + '_low' + channel + '/'
    test_dir_mid = FLAGS.test_dir + '_X' + str(FLAGS.upscale) + '_mid' + channel + '/'
    test_dir_high = FLAGS.test_dir + '_X' + str(FLAGS.upscale) + '_high' + channel + '/'
    image_list_low = sorted(glob.glob(test_dir_low + '*.png'))
    print(image_list_low)
    image_list_mid = sorted(glob.glob(test_dir_mid + '*.png'))
    print(image_list_mid)
    image_list_high = sorted(glob.glob(test_dir_high + '*.png'))
    print(image_list_high)

    image_list_sr = [os.path.join(FLAGS.checkpoint, os.path.basename(img_name)) for img_name in image_list_low]
    print(image_list_sr)
    image_low = []
    image_mid = []


    for i in range(len(image_list_low)):
        if FLAGS.model_selection == 9:
            image_low.append(np.expand_dims((scipy.misc.fromimage(Image.open(image_list_low[i])).astype(float) - mean_Y)/255*MAX_RGB, axis=0))
            image_mid.append(np.expand_dims(scipy.misc.fromimage(Image.open(image_list_mid[i])).astype(float)/255*MAX_RGB, axis=0))
        else:
            image_low.append(np.expand_dims(np.expand_dims((scipy.misc.fromimage(Image.open(image_list_low[i])).astype(float) - mean_Y)/255*MAX_RGB, axis=0), axis=3))
            image_mid.append(np.expand_dims(np.expand_dims(scipy.misc.fromimage(Image.open(image_list_mid[i])).astype(float)/255*MAX_RGB, axis=0), axis=3))
    # from IPython import embed; embed()
    if len(FLAGS.checkpoint) == 0:
        with open('checkpoint.txt', 'r') as text_file:
            lines = text_file.readlines()
            FLAGS.checkpoint = os.path.join('/scratch_net/ofsoundof/yawli/logs', lines[0])

            # FLAGS.checkpoint = os.path.join(os.path.dirname(os.path.dirname(FLAGS.test_dir)), 'logs', lines[0])
    l = 3 if FLAGS.model_selection == 9 else 1
    with tf.device('/device:GPU:' + FLAGS.sge_gpu):
        # super-resolve images using trained network and save the derived sr images
        global_step = tf.Variable(1, name='global_step', trainable=False)
        image_input_low = tf.placeholder(dtype=tf.float32, shape=(1, None, None, l))
        image_input_mid = tf.placeholder(dtype=tf.float32, shape=(1, None, None, l))
        sr,annealing_factor = selected_model(image_input_low, image_input_mid, FLAGS, global_step)
        sr = sr*255/MAX_RGB
    print("Create session")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    config.allow_soft_placement = True  # If a operation is not define it the default device, let it execute in another.
    sess = tf.Session(config=config)

    print("Load checkpoint")
    assert FLAGS.checkpoint != ''
    checkpoint_to_resume = FLAGS.checkpoint
    print("Resuming from checkpoint {}".format(checkpoint_to_resume))

    # checkpoint_to_resume = '/home/yawli/Documents/hashnets/logs/2018-05-09-20-47-14-ARN_deep_reproduce_noneR_noOutput_feature2_UP4L1C16A16M3ITER_500000'
    all_checkpoints = glob.glob(checkpoint_to_resume + '/model-*')
    all_checkpoints = [os.path.splitext(x)[0] for x in all_checkpoints if x.endswith('index')]
    all_checkpoints = sorted(all_checkpoints, key=lambda x: int(os.path.basename(x)[6:-1]))

    print(all_checkpoints[0], all_checkpoints[-1])

    # all_checkpoints = tf.train.get_checkpoint_state(checkpoint_to_resume).all_model_checkpoint_paths
    ckpt_num = len(all_checkpoints)
    print("\nTest for every saved checkpoint")

    #Used for getting the runtime
    class TimeLiner:
        _timeline_dict = None

        def update_timeline(self, chrome_trace):
            # convert crome trace to python dict
            chrome_trace_dict = json.loads(chrome_trace)
            # for first run store full trace
            if self._timeline_dict is None:
                self._timeline_dict = chrome_trace_dict
            # for other - update only time consumption, not definitions
            else:
                for event in chrome_trace_dict['traceEvents']:
                    # events time consumption started with 'ts' prefix
                    if 'ts' in event:
                        self._timeline_dict['traceEvents'].append(event)

        def save(self, f_name):
            with open(f_name, 'w') as f:
                json.dump(self._timeline_dict, f)
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    many_runs_timeline = TimeLiner()

    for i in [ckpt_num-100]:
        current_checkpoint = all_checkpoints[i]
        print("\n\nCurrent checkpoint: {}".format(current_checkpoint))
        assert current_checkpoint is not None
        saver = tf.train.Saver()
        saver.restore(sess, current_checkpoint)

        # import matplotlib.pyplot as plt

        # index = []
        test_iter = 10
        start_time = time.time()
        per_image_duration = []
        for image in range(len(image_list_low)):
            per_image = time.time()
            for iter in range(test_iter):
                sr_ = sess.run(sr, {image_input_low: image_low[image], image_input_mid: image_mid[image]})#, options=options, run_metadata=run_metadata)
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
            per_image_duration.append(time.time() - per_image)
        duration = np.sum(per_image_duration)
        # # from IPython import embed; embed(); exit()
        # index1 = np.concatenate(index)
        # plt.hist(index1, bins=512)  # arguments are passed to np.histogram
        # plt.title("Total")
        # plt.show()
        # many_runs_timeline.save(os.path.join(FLAGS.checkpoint, 'timeline_merged_{}_runs.json'.format(os.path.basename(FLAGS.test_dir))))
        print('The execution time for {} iterations is {}'.format(test_iter, duration))
        with open(os.path.join(FLAGS.checkpoint, 'psnr_ssim.csv'), 'a') as csv_file:
            csv_file.write('\n\nThe test time for {} iterations\n'.format(test_iter))
            content = '{:<8} '.format('') + '{:<8} '.format('Total') + '{:<8}\n'.format('Average')
            csv_file.write(content)
            content = '{:<8} '.format(os.path.basename(FLAGS.test_dir)) + "{:<8.4f} ".format(duration) + "{:<8.4f}\n".format(duration/test_iter/len(image_list_low))
            csv_file.write(content)
            # test_dir = '/home/yawli/Documents/hashnets/test/Set5'
        time_dir = os.path.join('/home/yawli/Documents/hashnets/time_' + os.path.basename(FLAGS.test_dir) + '.csv')
        with open(time_dir, 'a') as csv_file:
            content = '{:<8}\t '.format(FLAGS.model_selection) + "{:<8.4f}\t".format(duration) + "{:<8.4f}\t".format(duration/test_iter/len(image_list_low))
            for image in range(len(image_list_low)):
                content += "{:<8.4f}\t".format(per_image_duration[image]/test_iter)
            content += '\n'
            csv_file.write(content)

def rgbtoycbcr(image_rgb): #batch x H x W x C
    image_r = image_rgb[:, :, 0]
    image_g = image_rgb[:, :, 1]
    image_b = image_rgb[:, :, 2]
    image_y = 16 + (65.738 * image_r + 129.057 * image_g + 25.064 * image_b)/256
    image_cb = 128 + (-37.945 * image_r - 74.494 * image_g + 112.439 * image_b)/256
    image_cr = 128 + (112.439 * image_r - 94.154 * image_g - 18.285 * image_b)/256
    image_ycbcr = np.stack([image_y, image_cb, image_cr], axis=2)
    return image_ycbcr

def ycbcrtorgb(image_ycbcr):
    image_y = image_ycbcr[:, :, 0]
    image_cb = image_ycbcr[:, :, 1]
    image_cr = image_ycbcr[:, :, 2]
    image_r = (298.082 * image_y + 408.583 * image_cr)/256 - 222.921
    image_g = (298.082 * image_y - 100.291 * image_cb - 208.120 * image_cr)/256 + 135.576
    image_b = (298.082 * image_y + 516.412 * image_cb)/256 - 276.836

    image_r = tf.maximum(0, tf.minimum(255, image_r))
    image_g = tf.maximum(0, tf.minimum(255, image_r))
    image_b = tf.maximum(0, tf.minimum(255, image_r))

    image_rgb = np.stack([image_r, image_g, image_b], axis=2)
    return image_rgb

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
        selected_model = carn
    elif FLAGS.model_selection == 4:
        selected_model = vdsr
    elif FLAGS.model_selection == 6:
        selected_model = srcnn
    elif FLAGS.model_selection == 7:
        selected_model = fsrcnn
    elif FLAGS.model_selection == 8:
        selected_model = espcn
    elif FLAGS.model_selection == 9:
        selected_model = srresnet
    else:
        selected_model = espcn_comp

    if len(FLAGS.test_dir):
        print("The test dir is {}".format(FLAGS.test_dir))
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
    parser.add_argument('--storage_path', type=str, default='/scratch_net/ofsoundof/yawli/', help='Location where the input train and test dataset and the output logs are stored')

    #procedure selection
    parser.add_argument('--test_dir', type=str, default='', help='Test directory, when not empty, start testing')
    parser.add_argument("--test_procedure", type=int, default=1, help="test or test_restore or test_restore_gpu")
    parser.add_argument("--test_score_compute", help="When used, compute the best PSNR/SSIM values", action="store_true")
    parser.add_argument("--test_runtime_compute", help="When used compute the runtime", action="store_true")

    #model parameters
    parser.add_argument('--deep_feature_layer', type=int, default=3, help='Number of feature layers in the deep architecture')
    parser.add_argument('--deep_layer', type=int, default=7, help='Number of layers in the deep architecture')
    parser.add_argument('--deep_channel', type=int, default=2 ** 2, help='Number of channels of the regression output except the last regresion layer for which the output channel is always upscale**2')
    parser.add_argument('--deep_anchor', type=int, default=16, help='Number of anchors in the deep architecture')
    parser.add_argument('--deep_kernel', type=int, default=3, help='Kernel size in the regression layers of the deep architecture')
    parser.add_argument('--model_selection', type=int, default=2, help='Select the use model')
    parser.add_argument('--model_flag', type=str, default='CARN', help='Model flag used for identifying the log path, could be CARN, SRCNN, VDSR, ESPCN etc.')
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




# Try to use multiple GPUs
PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]

# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the default device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device
    return _assign

# https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
        over the devices. The inner list ranges over the different variables.
    Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(grads, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def tower_loss(model_fn, input_fn, global_step, scope):
    low, mid, high = input_fn()
    low = (low-mean_Y)/255*MAX_RGB
    mid = mid/255*MAX_RGB
    high = high/255*MAX_RGB
    sr, annealing_factor = model_fn(low, mid, FLAGS, global_step)
    loss = tf.reduce_sum(tf.squared_difference(high, sr)) + tf.reduce_sum(tf.losses.get_regularization_losses())
    _, psnr_sr = comp_mse_psnr(high, sr, MAX_RGB)
    _, psnr_inter = comp_mse_psnr(high, mid, MAX_RGB)
    psnr_gain = psnr_sr - psnr_inter
    #return loss and psnr gain for every tower. There is no need to add summaries here. Add summaries outside tower scope.
    return loss, psnr_gain

def create_parallel_optimization(model_fn, input_fn, optimizer, devices, controller="/cpu:0"):
    global_step = tf.train.get_or_create_global_step()
    tower_grads = []
    losses = []
    psnr_gains = []

    # Get the current variable scope so we can reuse all variables we need once we get
    # to the second iteration of the loop below
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id in enumerate(devices):
            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the controller.
            with tf.device(assign_to_device(id, controller)), tf.name_scope(name):

                # Compute loss and gradients, but don't apply them yet
                loss, psnr_gain = tower_loss(model_fn, input_fn, global_step, outer_scope)
                print('outer_scope is {}'.format(outer_scope))
                with tf.name_scope("compute_gradients"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    grads = optimizer.compute_gradients(loss)
                    grads = [(tf.clip_by_norm(gv[0], 2), gv[1]) for gv in grads]
                    tower_grads.append(grads)

                losses.append(loss)
                psnr_gains.append(psnr_gain)
                outer_scope.reuse_variables()

    # Apply the gradients on the controlling device
    with tf.name_scope("apply_gradients"), tf.device(controller):
        gradients = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step)

    return gradients, apply_gradient_op, losses, psnr_gains

def test_during_training(model_fn, input_fn, device, name):
    """
    Make sure that variable_scope and name_scope are the same with that in create_parallel_optimization. If
    variable_scope is different, then new variables are created. What if name_scope is different?
    Make sure that variable reuse mode is activated.
    """
    global_step = tf.train.get_or_create_global_step()
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        with tf.device(device), tf.name_scope(name):
            low, mid, high = input_fn()
            low = (low-mean_Y)/255*MAX_RGB
            mid = mid/255*MAX_RGB
            high = high/255*MAX_RGB
            sr, _ = model_fn(low, mid, FLAGS, global_step)
            _, psnr_sr = comp_mse_psnr(high, sr, MAX_RGB)
            _, psnr_itp = comp_mse_psnr(high, mid, MAX_RGB)
            # psnr_inter is used to check whether the benchmark is implemented correctly
    return psnr_sr, psnr_itp

def training_op(FLAGS, global_step):
    boundaries = [5000, 70000]
    values = [0.001, 0.0001, 0.00001]
    FLAGS.lr = tf.train.piecewise_constant(global_step, boundaries, values, name='learning_rate')
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
    return optimizer

def run_test_train(selected_model):

    #input setup
    #train dataset setup
    train_dataset = train_dataset_prepare(FLAGS)
    train_iterator = train_dataset.make_one_shot_iterator()
    #test dataset setup
    test_dir = '/home/yawli/Documents/hashnets/test/'
    test_set5_iterator = test_dataset_prepare(test_path_prepare, test_dir + 'Set5_X' + str(FLAGS.upscale)).make_initializable_iterator()
    test_set14_iterator = test_dataset_prepare(test_path_prepare, test_dir + 'Set14_X' + str(FLAGS.upscale)).make_initializable_iterator()
    #create dataset handle and iterator
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

    def input_fn():
        """Input function defined here but used later. Return the next batch from iterator.
        """
        with tf.device(None):
            # remove any device specifications for the input data
            return iterator.get_next()

    #global_step is used by optimizer and fast_hashnet. Thus, it is created beforehand.
    global_step = tf.get_variable('global_step', initializer=1, trainable=False)
    optimizer = training_op(FLAGS, global_step)
    devices = get_available_gpus()
    print('The GPU devices are {}'.format(devices))
    #define training ops
    gradients, apply_gradient_op, losses, psnr_gains = create_parallel_optimization(selected_model, input_fn, optimizer, devices, controller="/cpu:0")

    #define testing ops
    psnr5_sr, psnr5_itp = test_during_training(selected_model, input_fn, devices[0], 'tower_0')
    n = 1 if len(get_available_gpus()) > 1 else 0
    psnr14_sr, psnr14_itp = test_during_training(selected_model, input_fn, devices[n], 'tower_{}'.format(n))

    #define summaries
    summaries = []
    #train summaries
    summaries.append(tf.summary.scalar('psnr_gain_test', psnr_gains[0]))
    summaries.append(tf.summary.scalar('total_loss_test', losses[0]))
    #gradient summaries
    for grad, var in gradients:
        if grad is not None:
            summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
    #trainable variable summaries
    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))
    #test summaries
    psnr_test_train_set5 = tf.placeholder(tf.float32)
    psnr_test_train_set14 = tf.placeholder(tf.float32)
    summaries.append(tf.summary.scalar('psnr_test_train_set5', psnr_test_train_set5))
    summaries.append(tf.summary.scalar('psnr_test_train_set14', psnr_test_train_set14))
    summary_op = tf.summary.merge(summaries)

    #Create the sess, and use some options for better using gpu
    print("Create session")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
    config.allow_soft_placement = True  # If a operation is not define it the default device, let it execute in another.
    sess = tf.Session(config=config)

    print("Create checkpoint directory")
    if not FLAGS.restore:
        FLAGS.checkpoint = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '-' + FLAGS.checkpoint
        with open('checkpoint.txt', 'w') as text_file: #save at the current directory, used for testing
            text_file.write(FLAGS.checkpoint)

    LOG_DIR = os.path.join('/scratch_net/ofsoundof/yawli/logs', FLAGS.checkpoint )
    # LOG_DIR = os.path.join('/home/yawli/Documents/hashnets/logs', FLAGS.checkpoint )
    # assert (not os.path.exists(LOG_DIR)), 'LOG_DIR %s already exists'%LOG_DIR

    print("Create summary file writer")
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)

    print("Create saver")
    saver = tf.train.Saver(max_to_keep=1000)

    #Always init, then optionally restore
    print("Initialization of variables and dataset handle")
    sess.run(tf.global_variables_initializer())
    # sess.run(tf.local_variables_initializer()) #required by tf.train.match_filenames_once

    #Get the dataset handle
    train_handle = sess.run(train_iterator.string_handle())
    test_set5_handle = sess.run(test_set5_iterator.string_handle())
    test_set14_handle = sess.run(test_set14_iterator.string_handle())

    #restore from trained network
    if FLAGS.restore:
        print('Restore model from checkpoint {}'.format(tf.train.latest_checkpoint(LOG_DIR)))
        saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))
        checkpoint_filename = 'checkpoint'
    else:
        checkpoint_filename = 'checkpoint'

    i = sess.run(global_step) + 1
    print("Start iteration")
    while i <= FLAGS.max_iter:
        if i % FLAGS.summary_interval == 0:
            # test set5
            sess.run(test_set5_iterator.initializer)
            _psnr5_sr = []
            _psnr5_itp = []
            for j in range(5):
                psnr_sr, psnr_itp = sess.run([psnr5_sr, psnr5_itp], feed_dict={handle: test_set5_handle})
                _psnr5_sr.append(psnr_sr)
                _psnr5_itp.append(psnr_itp)
            psnr_mean_set5_sr = np.mean(_psnr5_sr)
            psnr_mean_set5_itp = np.mean(_psnr5_itp)
            print('PSNR results for Set5: SR {}, Bicubic{}'.format(psnr_mean_set5_sr, psnr_mean_set5_itp))

            #test set14
            sess.run(test_set14_iterator.initializer)
            _psnr14_sr = []
            _psnr14_itp = []
            for j in range(14):
                psnr_sr, psnr_itp = sess.run([psnr14_sr, psnr14_itp], feed_dict={handle: test_set14_handle})
                _psnr14_sr.append(psnr_sr)
                _psnr14_itp.append(psnr_itp)
            psnr_mean_set14_sr = np.mean(_psnr14_sr)
            psnr_mean_set14_itp = np.mean(_psnr14_itp)
            print('PSNR results for Set14: SR {}, Bicubic{}'.format(psnr_mean_set14_sr, psnr_mean_set14_itp))

            #save checkpoint
            if i % FLAGS.checkpoint_interval == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, 'model'), global_step=i, latest_filename=checkpoint_filename, write_meta_graph=False)
                print("Saved checkpoint to {}".format(save_path))
                train_writer.flush()

            #train
            time_start = time.time()
            [_, i, _losses, _psnr_gains, summary] = sess.run([apply_gradient_op, global_step, losses, psnr_gains, summary_op],
                                                             feed_dict={handle: train_handle, psnr_test_train_set5: psnr_mean_set5_sr, psnr_test_train_set14: psnr_mean_set14_sr})#, options=options, run_metadata=run_metadata)
            duration = time.time() - time_start
            train_writer.add_summary(summary, i)
            print("Iter {}, losses {}, psnr_gains {} for all the towers".format(i-1, _losses, _psnr_gains))
            print('Training time for one iteration: {}'.format(duration))
        else:
            [_, i] = sess.run([apply_gradient_op, global_step], feed_dict={handle: train_handle})



# def _parse_train(example_proto):
#     """
#     Parse examples in training dataset from tfrecords
#     """
#     features = {"image_gt": tf.FixedLenFeature((), tf.string),
#               "image_n": tf.FixedLenFeature((), tf.string),
#               "crop_dim": tf.FixedLenFeature((), tf.int64),
#               "sigma": tf.FixedLenFeature((), tf.int64)}
#     parsed_features = tf.parse_single_example(example_proto, features)
#     dim = tf.cast(parsed_features['crop_dim'], tf.int64) #whether this is needed?
#     sigma = tf.cast(parsed_features['sigma'], tf.int64)
#
#     # from IPython import embed; embed(); exit()
#     image_gt = tf.decode_raw(parsed_features['image_gt'], tf.uint8)
#     image_n = tf.decode_raw(parsed_features['image_n'], tf.uint8)
#     image_gt = tf.cast(tf.reshape(image_gt, tf.stack([dim, dim, 1])), tf.float32)
#     image_n = tf.cast(tf.reshape(image_n, tf.stack([dim, dim, 1])), tf.float32)
#     decision = tf.random_uniform([2], 0, 1)
#     image_gt = random_flip(image_gt, decision[0])
#     image_n = random_flip(image_n, decision[0])
#     return image_gt, image_n
#
# def _parse_test(filename_gt, filename_n):
#     """
#     Parse examples to form testing dataset.
#     """
#     image_gt = tf.read_file(filename_gt)
#     image_gt = tf.image.decode_png(image_gt, channels=1)
#     image_gt = tf.expand_dims(tf.cast(image_gt, tf.float32), axis=0)
#     image_n = tf.read_file(filename_n)
#     image_n = tf.image.decode_png(image_n, channels=1)
#     image_n = tf.expand_dims(tf.cast(image_n, tf.float32), axis=0)
#     return image_gt, image_n
#
# def train_dataset_prepare(FLAGS):
#     """
#     Prepare for training dataset
#     """
#     filenames = '/home/yawli/Documents/hashnets/Train400/Sig' + str(FLAGS.sigma) + '.tfrecords'
#     train_dataset = tf.data.TFRecordDataset(filenames)
#     train_dataset = train_dataset.shuffle(buffer_size=5000).repeat()  # Repeat the input indefinitely.
#     # train_dataset = train_dataset.apply(tf.contrib.data.map_and_batch(map_func=_parse_train, batch_size=FLAGS.batch_size, num_parallel_batches=8))
#     train_dataset = train_dataset.map(map_func=_parse_train, num_parallel_calls=20)  # Parse the record into tensors.
#     train_dataset = train_dataset.batch(FLAGS.batch_size)
#     train_dataset = train_dataset.prefetch(256)
#     return train_dataset
#
# def test_dataset_prepare(path_fun, test_dir, sigma):
#     """
#     Prepare for test dataset
#     """
#     image_list_gt, image_list_n = path_fun(test_dir, sigma)
#     test_dataset = tf.data.Dataset.from_tensor_slices((image_list_gt, image_list_n))
#     test_dataset = test_dataset.map(map_func=_parse_test)
#     return test_dataset
#
# def test_path_prepare(test_dir, sigma):
#     """
#     Prepare for paths of test image
#     """
#     test_dir_gt = test_dir + 'GT/'
#     test_dir_n = test_dir + 'Sig' + str(sigma) + '/'
#     image_list_gt = sorted(glob.glob(test_dir_gt + '*.png'))
#     image_list_n = sorted(glob.glob(test_dir_n + '*.png'))
#
#     return image_list_gt, image_list_n
#
# def image_converter(image_sr_array, image_mid_array, image_high_array, flag, boundarypixels, MAX_RGB):
#     """
#     make image conversions and return the subjective score, i.e., psnr and ssim of SR and bicubic image
#     :param image_sr_array:
#     :param image_mid_array:
#     :param image_high_array:
#     :param flag:
#     :param boundarypixels:
#     :return:
#     """
#     psnr_sr = 10 * np.log10(MAX_RGB ** 2 / (np.mean(np.square(image_high_array - image_sr_array))))
#     psnr_itp = 10 * np.log10(MAX_RGB ** 2 / (np.mean(np.square(image_high_array - image_mid_array))))
#     # from IPython import embed; embed(); exit()
#     ssim_sr = ssim(np.uint8(image_sr_array*255/MAX_RGB), np.uint8(image_high_array*255/MAX_RGB), gaussian_weights=True, use_sample_covariance=False)
#     ssim_itp = ssim(np.uint8(image_mid_array*255/MAX_RGB), np.uint8(image_high_array*255/MAX_RGB), gaussian_weights=True, use_sample_covariance=False)
#     score = [psnr_itp, psnr_sr, ssim_itp, ssim_sr]
#     return score
# def carn_denoise(data, FLAGS):
#     num_anchor = FLAGS.deep_anchor
#     inner_channel = FLAGS.deep_channel
#     activation = tf.keras.layers.PReLU(shared_axes=[1, 2]) if FLAGS.activation_regression == 1 else None
#     biases_add = tf.zeros_initializer() if FLAGS.biases_add_regression == 1 else None
#
#     with slim.arg_scope([slim.conv2d], stride=1,
#                         weights_initializer=tf.keras.initializers.he_normal(),
#                         weights_regularizer=slim.l2_regularizer(0.0001), reuse=tf.AUTO_REUSE):
#         # feature = slim.stack(data, slim.conv2d, [(64, [3, 3]), (64, [3, 3]), (inner_channel, [3, 3])], scope='feature_layer')
#         feature = data
#         for i in range(1, FLAGS.deep_feature_layer):
#             feature = slim.conv2d(feature, 64, [3, 3], scope='feature_layer' + str(i), activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
#         # feature = slim.conv2d(feature, 64, [3, 3], scope='feature_layer2', activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
#         feature = slim.conv2d(feature, inner_channel, [3, 3], scope='feature_layer' + str(FLAGS.deep_feature_layer), activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))  # B x H x W x f1
#         reshape_size = tf.shape(feature)
#         kernel_size = FLAGS.deep_kernel
#
#         def regression_layer(input_feature, r, k, dim, num, flag_shortcut, l):
#             with tf.name_scope('regression_layer' + l):
#                 result = slim.conv2d(input_feature, num*dim, [k, k], scope='regression_' + l, activation_fn=activation)  # B x H x W x 2^Rxs^2  filter k_size x k_size x Cin x 2^Rxs^2
#                 result = tf.reshape(result, [r[0], r[1], r[2], num, dim])   # B x H x W 2^R x s^2
#                 alpha = slim.conv2d(input_feature, num, [k, k], scope='alpha_' + l, activation_fn=tf.nn.softmax, biases_initializer=biases_add)  # B x H x W x R  filter k_size x k_size x Cin x R
#                 alpha = tf.expand_dims(alpha, 4)
#                 output_feature = tf.reduce_sum(result * alpha, axis=3)
#                 if flag_shortcut:
#                     if FLAGS.skip_conv == 1:
#                         skip_connection = slim.conv2d(input_feature, dim, [3, 3], scope='skip_connection_' + l, activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))
#                         return output_feature + skip_connection
#                     else:
#                         return output_feature + input_feature
#                 else:
#                     return output_feature
#         if FLAGS.deep_layer == 1:
#             regression = regression_layer(feature, reshape_size, kernel_size, 1, num_anchor, inner_channel==1, '1')
#         else:
#             regression = regression_layer(feature, reshape_size, kernel_size, inner_channel, num_anchor, True, '1')
#             for i in range(2, FLAGS.deep_layer):
#                 regression = regression_layer(regression, reshape_size, kernel_size, inner_channel, num_anchor, True, str(i))
#             regression = regression_layer(regression, reshape_size, kernel_size, 1, num_anchor, inner_channel == 1, str(FLAGS.deep_layer))
#     return regression
#
# def run_test_train_dataset(selected_model):
#
#     #Create the sess, and use some options for better using gpu
#     print("Create session")
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True  # Do not assign whole gpu memory, just use it on the go
#     config.allow_soft_placement = True  # If a operation is not define it the default device, let it execute in another.
#     sess = tf.Session(config=config)
#
#     with tf.device('/cpu:0'):
#         #input setup
#         #train dataset setup
#         train_dataset = train_dataset_prepare(FLAGS)
#         train_iterator = train_dataset.make_one_shot_iterator()
#         #test dataset setup
#         test_dir = '/home/yawli/Documents/hashnets/test/'
#         # test_set5_iterator = test_dataset_prepare(test_path_prepare, test_dir + 'Set5_X' + str(FLAGS.upscale)).make_initializable_iterator()
#         # test_set14_iterator = test_dataset_prepare(test_path_prepare, test_dir + 'Set14_X' + str(FLAGS.upscale)).make_initializable_iterator()
#         #create dataset handle and iterator
#         handle = tf.placeholder(tf.string, shape=[])
#         iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
#         high, low = iterator.get_next()
#
#     with tf.device('/device:GPU:' + FLAGS.sge_gpu):
#         low = (low-mean_Y)/255*MAX_RGB
#         mid = low/255*MAX_RGB
#         high = high/255*MAX_RGB
#
#         global_step = tf.Variable(1, name='global_step', trainable=False)
#         if FLAGS.restore and FLAGS.model_selection == 10:
#             with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
#                 saver = tf.train.import_meta_graph('/scratch_net/ofsoundof/yawli/logs/2018-05-11-13-21-55-ESPCN_COMP_UP4M10F20ITER_500000/model-510000.meta')
#                 saver.restore(sess, tf.train.latest_checkpoint('/scratch_net/ofsoundof/yawli/logs/2018-05-11-13-21-55-ESPCN_COMP_UP4M10F20ITER_500000'))
#                 graph = tf.get_default_graph()
#                 # from IPython import embed; embed(); exit()
#                 feature_19 = tf.stop_gradient(graph.get_tensor_by_name('feature_layer19/p_re_lu/add:0'))
#
#                 with slim.arg_scope([slim.conv2d], stride=1,
#                             weights_initializer=tf.keras.initializers.he_normal(),
#                             weights_regularizer=slim.l2_regularizer(0.0001)):
#
#                     for i in range(20, FLAGS.deep_feature_layer):
#                         net = slim.conv2d(feature_19, 32, [3, 3], scope='feature_layer' + str(i), activation_fn=tf.keras.layers.PReLU(shared_axes=[1, 2]))
#                     net = slim.conv2d(net, FLAGS.upscale ** 2, [3, 3], scope='output_layer', activation_fn=None)
#                 sr_space = tf.depth_to_space(net, FLAGS.upscale, 'space')
#                 sr = sr_space + mid
#                 annealing_factor = tf.shape(sr)[0]
#                 MSE_sr, PSNR_sr = comp_mse_psnr(sr, high, MAX_RGB)
#                 MSE_interp, PSNR_interp = comp_mse_psnr(mid, high, MAX_RGB)
#                 PSNR_gain = PSNR_sr - PSNR_interp
#                 loss = tf.reduce_sum(tf.squared_difference(high, sr)) + tf.reduce_sum(tf.losses.get_regularization_losses())
#                 train_op, global_step = training(loss, FLAGS, global_step)
#
#         else:
#             sr = carn_denoise(low, FLAGS)
#             annealing_factor = tf.Variable(1, trainable=False)
#             MSE_sr, PSNR_sr = comp_mse_psnr(sr, high, MAX_RGB)
#             MSE_interp, PSNR_interp = comp_mse_psnr(low, high, MAX_RGB)
#             PSNR_gain = PSNR_sr - PSNR_interp
#             loss = tf.reduce_sum(tf.squared_difference(high, sr)) + tf.reduce_sum(tf.losses.get_regularization_losses())
#             train_op, global_step = training(loss, FLAGS, global_step)
#
#     #train summary
#     tf.summary.scalar('MSE_sr', MSE_sr)
#     tf.summary.scalar('PSNR_sr', PSNR_sr)
#     tf.summary.scalar('MSE_interp', MSE_interp)
#     tf.summary.scalar('PSNR_interp', PSNR_interp)
#     tf.summary.scalar('PSNR_gain', PSNR_gain)
#     slim.summarize_collection(collection=tf.GraphKeys.TRAINABLE_VARIABLES)
#     #test summaries
#     # psnr_test_train_set5 = tf.placeholder(tf.float32)
#     # psnr_test_train_set14 = tf.placeholder(tf.float32)
#     # tf.summary.scalar('psnr_test_train_set5', psnr_test_train_set5)
#     # tf.summary.scalar('psnr_test_train_set14', psnr_test_train_set14)
#     merged = tf.summary.merge_all()
#
#
#     #Get the dataset handle
#     train_handle = sess.run(train_iterator.string_handle())
#     # test_set5_handle = sess.run(test_set5_iterator.string_handle())
#     # test_set14_handle = sess.run(test_set14_iterator.string_handle())
#
#     print("Create checkpoint directory")
#     # if not FLAGS.restore:
#     FLAGS.checkpoint = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '-' + FLAGS.checkpoint
#     with open('checkpoint.txt', 'w') as text_file: #save at the current directory, used for testing
#         text_file.write(FLAGS.checkpoint)
#     LOG_DIR = os.path.join('/scratch_net/ofsoundof/yawli/logs', FLAGS.checkpoint )
#     # LOG_DIR = os.path.join('/home/yawli/Documents/hashnets/logs', FLAGS.checkpoint )
#     # assert (not os.path.exists(LOG_DIR)), 'LOG_DIR %s already exists'%LOG_DIR
#
#     print("Create summary file writer")
#     train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
#
#     print("Create saver")
#     saver = tf.train.Saver(max_to_keep=100)
#
#     #Always init, then optionally restore
#     print("Initialization")
#     sess.run(tf.global_variables_initializer())
#
#     if FLAGS.restore:
#         print('Restore model from checkpoint {}'.format(tf.train.latest_checkpoint(LOG_DIR)))
#         saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))
#         checkpoint_filename = 'checkpoint'
#     else:
#         checkpoint_filename = 'checkpoint'
#     # checkpoint_filename = 'checkpoint'
#     # score5_all = []
#     # score14_all = []
#     # model_index = []
#     # _, _, path_set5 = test_path_prepare(test_dir + 'Set5_X' + str(FLAGS.upscale))
#     # path_set5 = [os.path.join(LOG_DIR, os.path.basename(i)) for i in path_set5]
#     # _, _, path_set14 = test_path_prepare(test_dir + 'Set14_X' + str(FLAGS.upscale))
#     # path_set14 = [os.path.join(LOG_DIR, os.path.basename(i)) for i in path_set14]
#     i = sess.run(global_step) + 1
#     print("Start iteration")
#     while i <= FLAGS.max_iter:
#         if i % FLAGS.summary_interval == 0:
#             # test set5
#             # sess.run(test_set5_iterator.initializer)
#             # score5 = np.zeros((4, 6))
#             # for j in range(5):
#             #     _sr, _mid, _high = sess.run([sr, mid, high], feed_dict={handle: test_set5_handle})
#             #     scipy.misc.toimage(np.squeeze(_sr)*255/MAX_RGB, cmin=0, cmax=255).save(path_set5[j])
#             #     # The current scipy version started to normalize all images so that min(data) become black and max(data)
#             #     # become white. This is unwanted if the data should be exact grey levels or exact RGB channels.
#             #     # The solution: scipy.misc.toimage(image_array, cmin=0.0, cmax=255).save('...') or use imageio library
#             #     score = image_converter(np.squeeze(_sr), np.squeeze(_mid), np.squeeze(_high), False, FLAGS.upscale, MAX_RGB)
#             #     score5[:, j] = score
#             # score5[:, -1] = np.mean(score5[:, :-1], axis=1)
#             # print('PSNR results for Set5: SR {}, Bicubic {}'.format(score5[1, -1], score5[0, -1]))
#             #
#             # #test set14
#             # sess.run(test_set14_iterator.initializer)
#             # score14 = np.zeros((4, 15))
#             # for j in range(14):
#             #     _sr, _mid, _high = sess.run([sr, mid, high], feed_dict={handle: test_set14_handle})
#             #     scipy.misc.toimage(np.squeeze(_sr)*255/MAX_RGB, cmin=0, cmax=255).save(path_set14[j])
#             #     score = image_converter(np.squeeze(_sr), np.squeeze(_mid), np.squeeze(_high), j == 2, FLAGS.upscale, MAX_RGB)
#             #     score14[:, j] = score
#             # score14[:, -1] = np.mean(score14[:, :-1], axis=1)
#             # print('PSNR results for Set14: SR {}, Bicubic {}'.format(score14[1, -1], score14[0, -1]))
#
#             time_start = time.time()
#             [_, i, af, l, mse_sr, mse_interp, psnr_sr, psnr_interp, psnr_gain, summary] =\
#                 sess.run([train_op, global_step, annealing_factor, loss, MSE_sr, MSE_interp, PSNR_sr, PSNR_interp, PSNR_gain, merged],
#                          feed_dict={handle: train_handle})#, options=options, run_metadata=run_metadata)
#             duration = time.time() - time_start
#             train_writer.add_summary(summary, i)
#             print("Iter %d, annealing factor %d, total loss %.5f; sr (mse %.5f, psnr %.5f); interp (mse %.5f, psnr %.5f); psnr_gain:%f" % (i-1, af, l, mse_sr, psnr_sr, mse_interp, psnr_interp, psnr_gain))
#             print('Training time for one iteration: {}'.format(duration))
#
#             if (i-1) % FLAGS.checkpoint_interval == 0:
#                 save_path = saver.save(sess, os.path.join(LOG_DIR, 'model'), global_step=i, latest_filename=checkpoint_filename, write_meta_graph=True)
#                 print("Saved checkpoint to {}".format(save_path))
#                 print("Flushing train writer")
#                 train_writer.flush()
#             #
#             # model_index.append(i)
#             # score5_all.append([score5[0, -1], score5[1, -1], score5[1, -1]-score5[0, -1], score5[2, -1], score5[3, -1], score5[3, -1]-score5[2, -1]])
#             # score14_all.append([score14[0, -1], score14[1, -1], score14[1, -1]-score14[0, -1], score14[2, -1], score14[3, -1], score14[3, -1]-score14[2, -1]])
#         else:
#             [_, i, _] = sess.run([train_op, global_step, loss], feed_dict={handle: train_handle})
#
#     # #write to csv files
#     # descriptor5 = 'Average PSNR (dB)/SSIM for Set5, Scale ' + str(FLAGS.upscale) + ', different model parameters during training' + '\n'
#     # descriptor14 = 'Average PSNR (dB)/SSIM for Set14, Scale ' + str(FLAGS.upscale) + ', different model parameters during training' + '\n'
#     # descriptor = [descriptor5, '', '', '', '', '', '', descriptor14, '', '', '', '', '', '']
#     # header_mul = (['Iteration'] + ['Bicu', 'SR', 'Gain'] * 2) * 2
#     # model_index_array = np.expand_dims(np.array(model_index), axis=0)
#     # score5_all_array = np.transpose(np.array(score5_all))
#     # score14_all_array = np.transpose(np.array(score14_all))
#     # written_content = np.concatenate((model_index_array, score5_all_array, model_index_array, score14_all_array), axis=0)
#     # written_content_fmt = (['{:<8}'] + ['{:<8.4f}', '{:<8.4f}', '{:<8.4f}'] * 2) * 2
#     # content = content_generator_mul_line(descriptor, written_content, written_content_fmt, header_mul)
#     # file_writer(os.path.join(LOG_DIR, 'psnr_ssim.csv'), 'a', content)
#     #
#     #
#     # #write to pickle files
#     # variables = {'index': np.array(model_index), 'bicubic': score5_all_array[0, :], 'sr': score5_all_array[1, :], 'gain': score5_all_array[2, :]}
#     # pickle_out = open(os.path.join(LOG_DIR, 'psnr_set5.pkl'), 'wb')
#     # pickle.dump(variables, pickle_out)
#     # pickle_out.close()
#     #
#     # variables = {'index': np.array(model_index), 'bicubic': score14_all_array[0, :], 'sr': score14_all_array[1, :], 'gain': score14_all_array[2, :]}
#     # pickle_out = open(os.path.join(LOG_DIR, 'psnr_set14.pkl'), 'wb')
#     # pickle.dump(variables, pickle_out)
#     # pickle_out.close()
