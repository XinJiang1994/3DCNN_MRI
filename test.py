import os
import time

import tensorflow as tf

from input_data_test import load_data_with_val
from model import CNN_MRI
from utils import pp
import numpy as np
import nibabel as nib
import random
import matplotlib.pyplot as plt
import matplotlib

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 40, "The size of batch images [64]")
flags.DEFINE_string("dataset", "brain", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "./samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("pretrain", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("zoom_rate", 40, "zoom_rate from original image,e.g. 60 means 60%")
flags.DEFINE_string("modal", "test", "data modal")
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
    #input_shape=[FLAGS.batch_size,182,218,182,1]
    input_shape = [FLAGS.batch_size, 145, 174, 145, 1]
    modal=FLAGS.modal
    # seqs = [0,1,2,3,4,5,6,7,8,9]
    seqs=[1]
    t = len(seqs)
    start_t=time.time()
    for iter in range(1):
        acc=[0]*t
        acc_val = [0] * t
        for acc_id, dataset_seq in enumerate(seqs):
            tf.reset_default_graph()
            if FLAGS.train:
                with tf.Session(config=run_config) as sess:

                    next_batch,next_batch_v,next_batch_test=load_data_with_val(sess,batch_size=FLAGS.batch_size,zoom_rate=FLAGS.zoom_rate,cross=dataset_seq,test_pos=10)

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
                _, next_batch_v,next_batch_test = load_data_with_val(sess, batch_size=65, zoom_rate=FLAGS.zoom_rate,
                                                              cross=dataset_seq,test_pos=10)
                input_shape[0]=65
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
                for i in range(1):
                    data, label = sess.run(next_batch_v)
                    label = np.squeeze(label)
                    g1 = cnn_mri.cnn_correct(data, label, FLAGS)
                    n1 = n1 + input_shape[0]
                    r1 = r1 + g1
                a_v_g = r1 / n1
                acc_val[acc_id]=a_v_g
                print('validation set 100 elements accuracy:{} '.format(a_v_g))
                n1 = 0
                r1 = 0
                for i in range(1):
                    data, label = sess.run(next_batch_test)
                    label = np.squeeze(label)
                    label = np.squeeze(label)
                    g1 = cnn_mri.cnn_correct(data, label, FLAGS)
                    n1 = n1 + input_shape[0]
                    r1 = r1 + g1
                a_v_g = r1 / n1
                acc[acc_id]=a_v_g
                print('test set 100 elements accuracy:{} '.format(a_v_g))
       #     print('acc_val:{}'.format(acc_val))
        print('acc:{}'.format(acc))
        print('avg_acc:{}'.format(sum(acc)/t))


    # visualization
    tf.reset_default_graph()
    loaded_graph = tf.Graph()
    dataset_seq = 1
    with tf.Session(graph=loaded_graph) as sess:
        cnn_mri = CNN_MRI(
            sess,
            config=FLAGS,
            input_shape=[100, input_shape[1], input_shape[2], input_shape[3], input_shape[4]],
            batch_size=FLAGS.batch_size,
            zoom_rate=FLAGS.zoom_rate,
            y_dim1=2,
            y_dim2=4,
            stride=[1, 1, 1, 1, 1],
            padding='SAME',
            checkpoint_dir=FLAGS.checkpoint_dir,
            model_name='CNN_{}_{}'.format(modal, dataset_seq),
            isTrain=False
        )
        cnn_mri.load(FLAGS.checkpoint_dir)
        latent_dir = './latent'
        if not os.path.isdir(latent_dir):
            os.mkdir(latent_dir)
        if FLAGS.visualize:
            sample_n = 5
            next_batch, _,_ = load_data_with_val(sess, batch_size=100, zoom_rate=FLAGS.zoom_rate,
                                               cross=dataset_seq,test_pos=10)
            # get the first layer's latent
            layers = [1, 2, 3]
            data, label = sess.run(next_batch)
            label = np.squeeze(label)
            for l in layers:
                latent = cnn_mri.get_latent(data, FLAGS, l)
                shape = list(latent.shape)
                print(shape)
                imgs_m = []
                imgs_m_main = []
                imgs_f = []
                imgs_f_main = []
                count = 0
                for i in range(shape[0]):
                    if label[i][0] == 1:
                        continue
                    if count == sample_n:
                        break
                    count += 1
                    for j in range(shape[4]):
                        arr = latent[i, :, :, :, j]
                        # amin,amax=arr.min(),arr.max()
                        # arr=255*(arr-amin)/(amax-amin)
                        # arr=arr.astype(np.int32)
                        # save_name=os.path.join(latent_dir,'latent_L_{}_batch_{}_FM_{}.nii.gz'.format(l,i,j))
                        # affine=np.eye(4,4)
                        # nib.save(nib.Nifti1Image(arr, affine), save_name)

                        axial_middle = arr.shape[2] // 2
                        imgs_m.append(arr[:, :, axial_middle].T)
                        axial_middle = arr.shape[1] // 2
                        imgs_m_main.append(arr[:, axial_middle, :].T)
                count = 0
                for i in range(shape[0]):
                    if label[i][0] != 1:
                        continue
                    if count == sample_n:
                        break
                    count += 1
                    for j in range(shape[4]):
                        arr = latent[i, :, :, :, j]
                        # amin, amax = arr.min(), arr.max()
                        # arr = 255 * (arr - amin) / (amax - amin)
                        # arr = arr.astype(np.int32)
                        # save_name = os.path.join(latent_dir, 'latent_L_{}_batch_{}_FM_{}.nii.gz'.format(l, i, j))
                        # affine = np.eye(4, 4)
                        # nib.save(nib.Nifti1Image(arr, affine), save_name)

                        axial_middle = arr.shape[2] // 2
                        imgs_f.append(arr[:, :, axial_middle].T)
                        matplotlib.image.imsave('latent/latent_L_{}_batch_{}_FM_{}.png'.format(l, i, j), arr[:, :, axial_middle].T)
                        org_data=data[i]
                        org_data=np.squeeze(org_data)
                        print('org_data shape : {}'.format(org_data.shape))
                        midd=org_data.shape[2] // 2
                        matplotlib.image.imsave('latent/org_data_batch_{}.png'.format(i),
                                                org_data[:, :, midd].T)

                        axial_middle = arr.shape[1] // 2
                        imgs_f_main.append(arr[:, axial_middle, :].T)
                print(len(imgs_m))
                imgs_m_main[len(imgs_m_main):len(imgs_m_main)] = imgs_f_main
                imgs_m[len(imgs_m):len(imgs_m)] = imgs_f
                imgs = imgs_m
                v = 'overview'
                if l == 1:
                    pic_name = os.path.join(latent_dir, 'seq{}_Latent{}_{}.png'.format(dataset_seq,l, v))
                    fig, axes = plt.subplots(nrows=sample_n * 2, ncols=32, sharex=True, sharey=True,
                                             figsize=(48, 20))
                    inc = 32
                    s = [(x + 1) * inc for x in range(20)]
                    for image, rows in zip(
                            [imgs[0:s[0]], imgs[s[0]:s[1]], imgs[s[1]:s[2]], imgs[s[2]:s[3]], imgs[s[3]:s[4]],
                             imgs[s[4]:s[5]], imgs[s[5]:s[6]], imgs[s[6]:s[7]], imgs[s[7]:s[8]], imgs[s[8]:s[9]]]
                            , axes):
                        for img, ax in zip(image, rows):
                            ax.imshow(img)
                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)
                    fig.tight_layout(pad=0)
                    plt.savefig(pic_name, bbox_inches='tight')
                    plt.close(fig)
                if l == 2:
                    pic_name = os.path.join(latent_dir, 'seq{}_Latent{}_{}.png'.format(dataset_seq, l, v))
                    fig, axes = plt.subplots(nrows=sample_n * 2, ncols=64, sharex=True, sharey=True,
                                             figsize=(72, 14))
                    inc = 64
                    s = [(x + 1) * inc for x in range(20)]
                    for image, rows in zip(
                            [imgs[0:s[0]], imgs[s[0]:s[1]], imgs[s[1]:s[2]], imgs[s[2]:s[3]], imgs[s[3]:s[4]],
                             imgs[s[4]:s[5]], imgs[s[5]:s[6]], imgs[s[6]:s[7]], imgs[s[7]:s[8]], imgs[s[8]:s[9]]]
                            , axes):
                        for img, ax in zip(image, rows):
                            ax.imshow(img)
                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)
                    fig.tight_layout(pad=0)
                    plt.savefig(pic_name, bbox_inches='tight')
                    plt.close(fig)
                if l == 3:
                    print('latent3 ##############')
                    # sex=''
                    # if label[i][0]==1:
                    #     sex='male'
                    # else:41
                    #     sex='female'
                    pic_name = os.path.join(latent_dir, 'seq{}_Latent{}_{}.png'.format(dataset_seq, l, v))
                    fig, axes = plt.subplots(nrows=sample_n * 2, ncols=128, sharex=True, sharey=True,
                                             figsize=(96, 10))
                    inc = 128
                    s = [(x + 1) * inc for x in range(20)]
                    for image, rows in zip(
                            [imgs[0:s[0]], imgs[s[0]:s[1]], imgs[s[1]:s[2]], imgs[s[2]:s[3]], imgs[s[3]:s[4]],
                             imgs[s[4]:s[5]], imgs[s[5]:s[6]], imgs[s[6]:s[7]], imgs[s[7]:s[8]], imgs[s[8]:s[9]]]
                            , axes):
                        for img, ax in zip(image, rows):
                            ax.imshow(img)
                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)
                    print(imgs[0])
                    fig.tight_layout(pad=0.0)
                    plt.savefig(pic_name, bbox_inches='tight')
                    plt.close(fig)
    print('time cost:{}'.format(time.time()-start_t)    )

if __name__ == '__main__':
    tf.app.run()
