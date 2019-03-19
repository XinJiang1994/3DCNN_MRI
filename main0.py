import csv
import os
import time

import tensorflow as tf

from input_data import load_data, load_data_with_val
from model import CNN_MRI
from utils import pp
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

flags = tf.app.flags
flags.DEFINE_integer("epoch", 60, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 45, "The size of batch images [64]")
flags.DEFINE_string("dataset", "brain", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "./samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("pretrain", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("zoom_rate", 40, "zoom_rate from original image,e.g. 60 means 60%")
flags.DEFINE_string("modal", "raw10_10", "data modal")
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

                    next_batch,next_batch_v,next_batch_test=load_data_with_val(sess,batch_size=FLAGS.batch_size,zoom_rate=FLAGS.zoom_rate,testset_id=dataset_seq)
                    input_shape[0] = FLAGS.batch_size
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
                        g1,results = cnn_mri.cnn_correct(data, label, FLAGS)
                        n1 = n1 + FLAGS.batch_size
                        r1 = r1 + g1
                    a_t_g = r1 / n1
                    print('validation set 100 elements accuracy:{} '.format(a_t_g))
            tf.reset_default_graph()
            loaded_graph = tf.Graph()
            with tf.Session(graph=loaded_graph) as sess:
                _, next_batch_v,next_batch_test = load_data_with_val(sess, batch_size=100, zoom_rate=FLAGS.zoom_rate,
                                                                     testset_id=dataset_seq)
                input_shape[0]=100
                cnn_mri = CNN_MRI(
                    sess,
                    config=FLAGS,
                    input_shape=input_shape,
                    batch_size=100,
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
                    g1,results = cnn_mri.cnn_correct(data, label, FLAGS)
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
                    g1,reults = cnn_mri.cnn_correct(data, label, FLAGS)
                    n1 = n1 + input_shape[0]
                    r1 = r1 + g1
                    
                    l=label[:,0:2]
                    
                    pre_y=cnn_mri.predict_y(data,label,FLAGS)
                    pre_y=np.array(pre_y)
                    pre_y = np.squeeze(pre_y)
                    pre_y=np.argmax(pre_y,1)
                    shape = list(pre_y.shape)
                    y_l=np.argmax(l,1)
                    
                    m=0
                    f=0
                    m_r=0
                    f_r=0
                    print('&&&&&&&&&&&&Label shape:',y_l.shape)
                    print('&&&&&&&&&&&&output shape:',shape)
                    print('y_l:',y_l)
                    print('pre_y:',pre_y)
                    for ix_ in range(shape[0]):
                        if y_l[ix_]==0:
                            m=m+1
                            if y_l[ix_]==pre_y[ix_]:
                                m_r=m_r+1
                        if y_l[ix_]==1:
                            f=f+1
                            if y_l[ix_]==pre_y[ix_]:
                                f_r=f_r+1
                    print('----------prediction results for men and women--- men:{}/{},women:{}/{}--------'.format(m_r,m,f_r,f))
                    with open("{}_men_women_acc.csv".format(FLAGS.modal), "a") as fw:
                        writer = csv.writer(fw)
                        writer.writerow([m,m_r,f,f_r,])
                a_v_g = r1 / n1
                acc[acc_id]=a_v_g
                print('test set 100 elements accuracy:{} '.format(a_v_g))
       #     print('acc_val:{}'.format(acc_val))
        print('acc:{}'.format(acc))
        print('avg_acc:{}'.format(sum(acc)/t))


#    ########### visualization
#    tf.reset_default_graph()
#    loaded_graph = tf.Graph()
#    dataset_seq = 9
#    with tf.Session(graph=loaded_graph) as sess:
#        cnn_mri = CNN_MRI(
#            sess,
#            config=FLAGS,
#            input_shape=[100, input_shape[1], input_shape[2], input_shape[3], input_shape[4]],
#            batch_size=100,
#            zoom_rate=FLAGS.zoom_rate,
#            y_dim1=2,
#            y_dim2=4,
#            stride=[1, 1, 1, 1, 1],
#            padding='SAME',
#            checkpoint_dir=FLAGS.checkpoint_dir,
#            model_name='CNN_{}_{}'.format(modal, dataset_seq),
#            isTrain=False
#        )
#        cnn_mri.load(FLAGS.checkpoint_dir)
        
#        ##########  calculate entropy of features   and  calculate mean features ############
#        mean_features_m=np.zeros((32,27,33,27))
#        mean_features_f=np.zeros((32,27,33,27))
#        count_m=0
#        count_f=0
#        for i in range(8):
#            next_batch, _,_ = load_data_with_val(sess, batch_size=100, zoom_rate=FLAGS.zoom_rate,
#                                                 testset_id=dataset_seq)
#            data, label = sess.run(next_batch)
#            label = np.squeeze(label)
#            #shape = list(data.shape)
##            for ix in range(shape[0]):
##                d=data[ix,:,:,:]
##                l=label[ix]
#                
#            latent = cnn_mri.get_latent(data, FLAGS, 1)
#            L_shape = list(latent.shape)
#            
#            for n in range(L_shape[0]):
#                mean_fa=[]
#                H_n=[]
#                H_n.append(1-label[n][0])
#                mean_fa.append(1-label[n][0])
#                for c in range(L_shape[4]):
#                    d = latent[n, :, :, :, c]
#                    ##calculate mean fa
#                    cover_pos=d>0
#                    meanfa=np.sum(d[cover_pos])/np.sum(cover_pos)
#                    mean_fa.append(meanfa)
#                    
#                    ###calculate mean_features
#                    if label[n][0]==1 and count_m<337:
#                        mean_features_m[c,:,:,:]=mean_features_m[c,:,:,:]+d
#                        count_m=count_m+1
#                    if label[n][0]==0 and count_f<337:
#                        mean_features_f[c,:,:,:]=mean_features_f[c,:,:,:]+d
#                        count_f=count_f+1
#                    print('d shape:',d.shape)  ##[27 33 27]
#                    ##calculate entropy
#                    dd=np.reshape(d,[1,-1])
#                    dd=np.squeeze(dd)
#                    ddmin, ddmax = dd.min(), dd.max()
#                    dd =255 * (dd-ddmin)/(ddmax-ddmin)
#                    dd=np.around(dd)
#                    dd=dd.astype(np.int64)
#                    counts=np.bincount(dd)
#                    counts=counts/np.sum(counts)
#                    pos=counts!=0
#                    Pi=counts[pos]
#                    H=-np.sum(Pi*np.log(Pi))
#                    H_n.append(H)
#                with open("entropy.csv", "a") as fw:
#                    writer = csv.writer(fw)
#                    writer.writerow(H_n)
#                with open("feature_mean_fa.csv", "a") as fw:
#                    writer = csv.writer(fw)
#                    writer.writerow(mean_fa)

# #######  Mean Features  ##################
#        mean_features_m=mean_features_m/800
#        mean_features_f=mean_features_f/800
#        s_m=list(mean_features_m.shape)
#        s_f=list(mean_features_f.shape)
#        latent_dir = './latent/mean_features'
#        if not os.path.isdir(latent_dir):
#            os.mkdir(latent_dir)
#        for i in range(s_m[0]):
#            save_name='m_feature_{}'.format(i)
#            save_name=os.path.join(latent_dir,save_name)
#            arr=mean_features_m[i,:,:,:]
#            affine=np.eye(4,4)
#            nib.save(nib.Nifti1Image(arr, affine), save_name)
#        for i in range(s_f[0]):
#            save_name='f_feature_{}'.format(i)
#            save_name=os.path.join(latent_dir,save_name)
#            arr=mean_features_f[i,:,:,:]
#            affine=np.eye(4,4)
#            nib.save(nib.Nifti1Image(arr, affine), save_name)

        
#        latent_dir = './latent'
#        if not os.path.isdir(latent_dir):
#            os.mkdir(latent_dir)
#        if FLAGS.visualize:
#            sample_n = 10
#            next_batch, _,_ = load_data_with_val(sess, batch_size=100, zoom_rate=FLAGS.zoom_rate,
#                                                 testset_id=dataset_seq)
#            # get the first layer's latent
#            layers = [1, 2, 3]
#            data, label = sess.run(next_batch)
#            label = np.squeeze(label)
#            for l in layers:
#                latent = cnn_mri.get_latent(data, FLAGS, l)
#                shape = list(latent.shape)
#                #print(shape)
#                imgs_m = []
#                imgs_m_main = []
#                imgs_f = []
#                imgs_f_main = []
#                count = 0
#                for i in range(shape[0]):
#                    if label[i][0] == 1:
#                        continue
#                    if count == sample_n:
#                        break
#                    count += 1
#                    for j in range(shape[4]):
#                        arr = latent[i, :, :, :, j]
#                        # amin,amax=arr.min(),arr.max()
#                        # arr=255*(arr-amin)/(amax-amin)
#                        # arr=arr.astype(np.int32)
#                        # save_name=os.path.join(latent_dir,'latent_L_{}_batch_{}_FM_{}.nii.gz'.format(l,i,j))
#                        # affine=np.eye(4,4)
#                        # nib.save(nib.Nifti1Image(arr, affine), save_name)

#                        axial_middle = arr.shape[2] // 2
#                        imgs_m.append(arr[:, :, axial_middle].T)
#                        axial_middle = arr.shape[1] // 2
#                        imgs_m_main.append(arr[:, axial_middle, :].T)
#                count = 0
#                for i in range(shape[0]):
#                    if label[i][0] != 1:
#                        continue
#                    if count == sample_n:
#                        break
#                    count += 1
#                    for j in range(shape[4]):
#                        arr = latent[i, :, :, :, j]
#                        # amin, amax = arr.min(), arr.max()
#                        # arr = 255 * (arr - amin) / (amax - amin)
#                        # arr = arr.astype(np.int32)
#                        # save_name = os.path.join(latent_dir, 'latent_L_{}_batch_{}_FM_{}.nii.gz'.format(l, i, j))
#                        # affine = np.eye(4, 4)
#                        # nib.save(nib.Nifti1Image(arr, affine), save_name)

#                        axial_middle = arr.shape[2] // 2
#                        imgs_f.append(arr[:, :, axial_middle].T)
#                        axial_middle = arr.shape[1] // 2
#                        imgs_f_main.append(arr[:, axial_middle, :].T)
#                #print(len(imgs_m))
#                imgs_m_main[len(imgs_m_main):len(imgs_m_main)] = imgs_f_main
#                imgs_m[len(imgs_m):len(imgs_m)] = imgs_f
#                imgs = imgs_m
#                v = 'overview'
#                if l == 1:
#                    pic_name = os.path.join(latent_dir, 'seq{}_Latent{}_{}.png'.format(dataset_seq,l, v))
#                    fig, axes = plt.subplots(nrows=sample_n * 2, ncols=32, sharex=True, sharey=True,
#                                             figsize=(48, 20))
#                    inc = 32
#                    s = [(x + 1) * inc for x in range(20)]
#                    for image, rows in zip(
#                            [imgs[0:s[0]], imgs[s[0]:s[1]], imgs[s[1]:s[2]], imgs[s[2]:s[3]], imgs[s[3]:s[4]],
#                             imgs[s[4]:s[5]], imgs[s[5]:s[6]], imgs[s[6]:s[7]], imgs[s[7]:s[8]], imgs[s[8]:s[9]],
#                             imgs[s[9]:s[10]], imgs[s[10]:s[11]], imgs[s[11]:s[12]], imgs[s[12]:s[13]], imgs[s[13]:s[14]],
#                             imgs[s[14]:s[15]], imgs[s[15]:s[16]], imgs[s[16]:s[17]], imgs[s[17]:s[18]], imgs[s[18]:s[19]]]
#                            , axes):
#                        for img, ax in zip(image, rows):
#                            ax.imshow(img)
#                            ax.get_xaxis().set_visible(False)
#                            ax.get_yaxis().set_visible(False)
#                    fig.tight_layout(pad=0)
#                    plt.savefig(pic_name, bbox_inches='tight')
#                    plt.close(fig)
                    
                    
#                if l == 2:
#                    pic_name = os.path.join(latent_dir, 'seq{}_Latent{}_{}.png'.format(dataset_seq, l, v))
#                    fig, axes = plt.subplots(nrows=sample_n * 2, ncols=64, sharex=True, sharey=True,
#                                             figsize=(72, 14))
#                    inc = 64
#                    s = [(x + 1) * inc for x in range(20)]
#                    for image, rows in zip(
#                            [imgs[0:s[0]], imgs[s[0]:s[1]], imgs[s[1]:s[2]], imgs[s[2]:s[3]], imgs[s[3]:s[4]],
#                             imgs[s[4]:s[5]], imgs[s[5]:s[6]], imgs[s[6]:s[7]], imgs[s[7]:s[8]], imgs[s[8]:s[9]]]
#                            , axes):
#                        for img, ax in zip(image, rows):
#                            ax.imshow(img)
#                            ax.get_xaxis().set_visible(False)
#                            ax.get_yaxis().set_visible(False)
#                    fig.tight_layout(pad=0)
#                    plt.savefig(pic_name, bbox_inches='tight')
#                    plt.close(fig)
#                if l == 3:
#                    #print('latent3 ##############')
#                    # sex=''
#                    # if label[i][0]==1:
#                    #     sex='male'
#                    # else:
#                    #     sex='female'
#                    pic_name = os.path.join(latent_dir, 'seq{}_Latent{}_{}.png'.format(dataset_seq, l, v))
#                    fig, axes = plt.subplots(nrows=sample_n * 2, ncols=128, sharex=True, sharey=True,
#                                             figsize=(96, 10))
#                    inc = 128
#                    s = [(x + 1) * inc for x in range(20)]
#                    for image, rows in zip(
#                            [imgs[0:s[0]], imgs[s[0]:s[1]], imgs[s[1]:s[2]], imgs[s[2]:s[3]], imgs[s[3]:s[4]],
#                             imgs[s[4]:s[5]], imgs[s[5]:s[6]], imgs[s[6]:s[7]], imgs[s[7]:s[8]], imgs[s[8]:s[9]]]
#                            , axes):
#                        for img, ax in zip(image, rows):
#                            ax.imshow(img)
#                            ax.get_xaxis().set_visible(False)
#                            ax.get_yaxis().set_visible(False)
#                    #print(imgs[0])
#                    fig.tight_layout(pad=0.0)
#                    plt.savefig(pic_name, bbox_inches='tight')
#                    plt.close(fig)
    print('time cost:{}'.format(time.time()-start_t)    )

if __name__ == '__main__':
    tf.app.run()
