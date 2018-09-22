from __future__ import division
import numpy as np
import scipy.misc
import os
import random
import tensorflow as tf

def frames_to_tuples(frames):
    X = []
    for i in range(0,len(frames)-1):
        X.append((frames[i], frames[i+1]))
    return X

IMAGE_PATH = 'Standard91'
def training_data():
    images = sorted(os.listdir(VIDEO_PATH))
    X = []
    for img in images:
        frames = [os.path.join(video, frame) for frame in sorted(os.listdir(VIDEO_PATH + video))]
        Xv = frames_to_tuples(frames)
        X.extend(Xv)
    np.random.shuffle(X)
    return X

CROPPED_DIM = 128
def process_frames(im1, im2):
    assert im1.shape == im2.shape
    #im12 = np.concatenate((im1,im2),axis=2)
    for i in range(0,im1.shape[0] - CROPPED_DIM,CROPPED_DIM // 2):
        for j in range(0, im1.shape[1] - CROPPED_DIM, CROPPED_DIM // 2):
            #print("Cropping (%d,%d)-(%d,%d) from %dx%d"%(i,j,i+CROPPED_DIM,j+CROPPED_DIM,im1.shape[0],im1.shape[1]))
            yield im1[i:i + CROPPED_DIM, j:j + CROPPED_DIM, :],im2[i:i + CROPPED_DIM, j:j + CROPPED_DIM, :]




X = training_data()
sess = tf.Session()

crop_input = tf.placeholder(tf.uint8, shape=[CROPPED_DIM,CROPPED_DIM,3],name='crop_input')
crop_encoded = tf.image.encode_png(crop_input)

count = 0
NUM_PER_SHARD = 100000
if not os.path.exists('cdvl/crops/'):
    os.makedirs('cdvl/crops/')
for image1,image2 in X:
    print(count)
    im1 = (scipy.misc.imread(VIDEO_PATH + image1, mode='RGB'))  # HxWxC RGB
    im2 = (scipy.misc.imread(VIDEO_PATH + image2, mode='RGB'))
    for (x1,x2) in process_frames(im1,im2):
        if count % NUM_PER_SHARD == 0:
            if count > 0:
                writer.close()
            writer = tf.python_io.TFRecordWriter('cdvl/crops/crop_data_%03d.tfrecords' % (count // NUM_PER_SHARD))
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'crop1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sess.run(crop_encoded, feed_dict={crop_input: x1})])),
                    'crop2': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sess.run(crop_encoded, feed_dict={crop_input: x2})])),
                }
        ))
        writer.write(example.SerializeToString())
        count += 1
        # example_parsed = tf.parse_single_example(example.SerializeToString(),features={
        #     'crop1' :tf.FixedLenFeature([], tf.string),
        #     'crop2': tf.FixedLenFeature([], tf.string),
        # })
if count%NUM_PER_SHARD != 0:
    writer.close()
