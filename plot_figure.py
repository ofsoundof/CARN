__author__ = 'yawli'

import pickle
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
def plot_figure(variables, FLAGS):
    path = FLAGS.checkpoint
    test_set = os.path.splitext(FLAGS.filename)[0][5:]
    step = [int(index) for index in variables['index']]
    fig = plt.figure(1)
    ax = plt.subplot(211)
    ax.plot(step, variables['bicubic'], 'k', label='Bicubic')
    ax.plot(step, variables['sr'], 'r', label='SR')
    ax.legend()
    #plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR result for ' + test_set)
    plt.grid()
    ay = plt.subplot(212)
    ay.plot(step, variables['gain'], label='Gain')
    ay.legend()
    plt.xlabel('Iteration')
    plt.ylabel('PSNR (dB)')
    plt.grid()
    axes = plt.gca()
    axes.set_ylim([max(variables['gain']) - 0.4, max(variables['gain'])+0.1])
    fig.savefig(os.path.join('logs', path, 'psnr_' + test_set + '.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tmpdir',
        type=str,
        default='',
        help='The temp dir'
    )
    parser.add_argument(
        '--filename',
        type=str,
        default='',
        help='The filename'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help='Checkpoint'
    )
    FLAGS, unparsed = parser.parse_known_args()
    if len(FLAGS.checkpoint) == 0:
        with open(os.path.join(FLAGS.tmpdir, 'checkpoint.txt'), 'r') as text_file:
            lines = text_file.readlines()
            FLAGS.checkpoint = os.path.join(os.getcwd(), 'logs', lines[0])

    pickle_in = open(os.path.join(FLAGS.checkpoint, FLAGS.filename), 'rb')
    variables = pickle.load(pickle_in)
    pickle_in.close()
    plot_figure(variables, FLAGS)
    #np.array(variables['sr'][4000:5000])
    #value = np.array(variables['sr'])
    #index = np.array(variables['index'])
    value = np.array(variables['sr'])
    index = np.array(variables['index'])
    i_max = np.argmax(value)
    index_max = index[i_max]
    value_max = value[i_max]
    print('({}, {}): the maximum PSNR {} appears at {}th iteration'.format(value_max, index_max, value_max, index_max))
    #from IPython import embed; embed(); exit()
    with open('/home/yawli/Documents/hashnets/final_test_results_highest.csv', 'a') as csv_file:
        content = "{:<8.4f} \t".format(value_max) + "{:<8.2f} \t".format(np.round(value_max, 2)) + "{:<8.4f} \n".format(index_max)
        csv_file.write(content)