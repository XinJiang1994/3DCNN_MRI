import os
import tensorflow as tf

from model import CNN_MRI
from utils import pp
import numpy as np
import nibabel as nib
import random

#raw_size=[182,218,182]
raw_size=[55,55,55]
z=1

def _parse_function_full(example_proto):
  features = {"label": tf.FixedLenFeature([6], tf.int64),
              "img_raw": tf.FixedLenFeature([], tf.string)}
  parsed_features = tf.parse_single_example(example_proto, features)
  img = tf.decode_raw(parsed_features['img_raw'], tf.float32)
  print("$$$$$$$$$$$$$img shape:",parsed_features['img_raw'])
  shape = [55,55,55,1]
  img = tf.reshape(img, shape)
  # img=cent_crop(img)
  img = tf.cast(img, tf.float32)
  print(parsed_features['label'])
  print('img shape~~~~~~~~~~~~~~~~:{}'.format(img.get_shape()))
  label = tf.cast(parsed_features['label'], tf.int64)
  print(label)
  label=tf.reshape(label,[1,6])
  return img, label


def load_data(sess,filename,batch_size,zoom_rate,shuffle_buffer=None):
    dataset = tf.data.TFRecordDataset(filename)
    _parse_function = _parse_function_full
    dataset = dataset.map(_parse_function)
    if shuffle_buffer is not None:
        dataset=dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()
    sess.run(iterator.initializer)
    return next_batch
#
# def load_trainset(sess,batch_size,zoom_rate,shuffle_buffer=None):
#     data_train_files=''
#     if zoom_rate==60:
#         data_train_files = '/root/userfolder/Dataset/FA/train_zoomed_r0.6_0.tfrecord'
#     elif zoom_rate==50:
#         data_train_files = '/root/userfolder/Dataset/FA/train_zoomed_r0.5_0.tfrecord'
#     elif zoom_rate==40:
#         data_train_files = '/root/userfolder/Dataset/FA/train_zoomed_r0.4_0.tfrecord'
#     else:
#         data_train_files = '/root/userfolder/Dataset/FA/train_zoomed_r0.5_0.tfrecord'
#     next_batch = load_data(sess, filename=data_train_files, batch_size=batch_size, zoom_rate=zoom_rate,shuffle_buffer=shuffle_buffer)
#     return next_batch
# def load_valset(sess,batch_size,zoom_rate,shuffle_buffer=None):
#     data_validition_files = ''
#     if zoom_rate == 60:
#         data_validition_files = '/root/userfolder/Dataset/FA/validition_zoomed_r0.6_0.tfrecord'
#     elif zoom_rate == 50:
#         data_validition_files = '/root/userfolder/Dataset/FA/validition_zoomed_r0.5_0.tfrecord'
#     elif zoom_rate == 40:
#         data_validition_files = '/root/userfolder/Dataset/FA/validition_zoomed_r0.4_0.tfrecord'
#     elif zoom_rate == 30:
#         data_validition_files = '/root/userfolder/Dataset/FA/validition_zoomed_r0.4_0.tfrecord'
#     else:
#         data_validition_files = '/root/userfolder/Dataset/FA/validition_zoomed_r0.3_0.tfrecord'
#     next_batch_v = load_data(sess, filename=data_validition_files, batch_size=batch_size,
#                              zoom_rate=zoom_rate,shuffle_buffer=shuffle_buffer)
#     return next_batch_v
def load_data_with_val(sess,batch_size,zoom_rate,shuffle_buffer=None,brain_area=None,testset_id=0):
    valset_id = np.mod(testset_id + 1, 10)
    data_dir = '/root/commonfile/Dataset/FA/multi_files/'
    file_list=[]
    area=''
    if brain_area is not None:
        area='area_i_10/area_{}/0/'.format(brain_area)
    for i in range(10):
        p = os.path.join(data_dir, '{}train_zoomed_r{}_{}.tfrecord'.format(area,z, i))
        file_list.append(p)
    print(file_list)

    val_file = file_list[valset_id]
    test_file = file_list[testset_id]
    train_files=[file for file in file_list if file!=val_file and file!=test_file]
    next_batch_t=load_data(sess, filename=train_files, batch_size=batch_size,
              zoom_rate=zoom_rate, shuffle_buffer=shuffle_buffer)
    next_batch_v = load_data(sess, filename=val_file, batch_size=batch_size,
                             zoom_rate=zoom_rate, shuffle_buffer=shuffle_buffer)
    next_batch_test = load_data(sess, filename=test_file, batch_size=batch_size,
                             zoom_rate=zoom_rate, shuffle_buffer=shuffle_buffer)
    return next_batch_t,next_batch_v,next_batch_test

# def test():
#     with tf.Session() as sess:
#         next_bath_t,next_batch_v=load_data_with_val(sess,32,0.5,seq=0)
#         for i in range(2000):
#             try:
#                 data,label=sess.run(next_bath_t)
#                 print (i)
#             except tf.errors.OutOfRangeError:
#                 print ('End of dataset')
#                 break;
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"
# test()


