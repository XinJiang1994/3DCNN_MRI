import os
import time

import tensorflow as tf

from input_data_flirt import load_data, load_data_with_val
from model import CNN_MRI
from utils import pp
import numpy as np
import nibabel as nib
import random
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 40, "The size of batch images [64]")
flags.DEFINE_string("dataset", "brain", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "./samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("pretrain", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("zoom_rate", 32, "zoom_rate from original image,e.g. 60 means 60%")
flags.DEFINE_string("modal", "flirt", "data modal")
FLAGS = flags.FLAGS


def main(_):
    pp.pprint(flags.FLAGS.__flags)
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    gpu_options.allow_growth = True
    run_config = tf.ConfigProto(allow_soft_placement=False,gpu_options=gpu_options,log_device_placement=False)
    run_config.gpu_options.allow_growth = True

    tf.reset_default_graph()
    input_shape=[FLAGS.batch_size,182,218,182,1]
    # input_shape = [FLAGS.batch_size, 145, 174, 145, 1]
    modal=FLAGS.modal
    seqs = [0,1,2,3,4,5,6,7,8,9]
    # seqs=[0,3]
    t = len(seqs)
    start_t=time.time()
    for iter in range(1):
        acc=[0]*t
        acc_val = [0] * t
        for acc_id, dataset_seq in enumerate(seqs):
            tf.reset_default_graph()
            if FLAGS.train:
                with tf.Session(config=run_config) as sess:

                    next_batch,next_batch_v,next_batch_test=load_data_with_val(sess,batch_size=FLAGS.batch_size,zoom_rate=FLAGS.zoom_rate,cross=dataset_seq,modal='flirt')

                    cnn_mri=CNN_MRI(
                        sess,
                        input_shape=input_shape,
                        config=FLAGS,
                        batch_size=FLAGS.batch_size,
                        zoom_rate=FLAGS.zoom_rate,
                        y_dim1=2,
                        y_dim2=4,
                        stride=[1,1,1,1,1],
                        padding='SAME',
                        checkpoint_dir=FLAGS.checkpoint_dir,
                        model_name='CNN_{}_{}'.format(modal,dataset_seq),
                        isTrain=True
                    )

                    cnn_mri.train(next_batch,next_batch_v,FLAGS,dataset_seq)
                    n1 = 0
                    r1 = 0
                    for i in range(1):
                        data, label = sess.run(next_batch_v)
                        label = np.squeeze(label)
                        label = np.squeeze(label)
                        g1 = cnn_mri.cnn_correct(data, label, FLAGS)
                        n1 = n1 + FLAGS.batch_size
                        r1 = r1 + g1
                    a_t_g = r1 / n1
                    print('validation set 100 elements accuracy:{} '.format(a_t_g))
            tf.reset_default_graph()
            loaded_graph = tf.Graph()
            with tf.Session(graph=loaded_graph) as sess:
                _, next_batch_v,next_batch_test = load_data_with_val(sess, batch_size=FLAGS.batch_size, zoom_rate=FLAGS.zoom_rate,
                                                              cross=dataset_seq)
                cnn_mri = CNN_MRI(
                    sess,
                    config=FLAGS,
                    input_shape=input_shape,
                    batch_size=FLAGS.batch_size,
                    zoom_rate=FLAGS.zoom_rate,
                    y_dim1=2,
                    y_dim2=4,
                    stride=[1, 1, 1, 1, 1],
                    padding='SAME',
                    checkpoint_dir=FLAGS.checkpoint_dir,
                    model_name='CNN_{}_{}'.format(modal,dataset_seq),
                    isTrain=False
                )
                cnn_mri.load(FLAGS.checkpoint_dir)
                n1 = 0
                r1 = 0
                for i in range(100):
                    data, label = sess.run(next_batch_v)
                    label = np.squeeze(label)
                    g1 = cnn_mri.cnn_correct(data, label, FLAGS)
                    n1 = n1 + FLAGS.batch_size
                    r1 = r1 + g1
                a_v_g = r1 / n1
                acc_val[acc_id]=a_v_g
                print('validation set 100 elements accuracy:{} '.format(a_v_g))
                n1 = 0
                r1 = 0
                for i in range(100):
                    data, label = sess.run(next_batch_test)
                    label = np.squeeze(label)
                    label = np.squeeze(label)
                    g1 = cnn_mri.cnn_correct(data, label, FLAGS)
                    n1 = n1 + FLAGS.batch_size
                    r1 = r1 + g1
                a_v_g = r1 / n1
                acc[acc_id]=a_v_g
                print('test set 100 elements accuracy:{} '.format(a_v_g))

        print('acc_val:{}'.format(acc_val))
        print('acc:{}'.format(acc))
        print('avg_acc:{}'.format(sum(acc)/t))

    print('time cost:{}'.format(time.time()-start_t)    )

if __name__ == '__main__':
    tf.app.run()
