import os
import tensorflow as tf

from model import CNN_MRI
from utils import pp
import numpy as np
import nibabel as nib
import random

#raw_size=[182,218,182]
raw_size=[145,174,145]
z=0.40
# registered='/registered'
#data_dir='/root/userfolder/Dataset/FA/multi_files/'
#data_dir='/root/userfolder/Dataset/FA/multi_files/tbss_mask'
# data_dir='/home/tclab/Documents/Dataset/FA/multi_files'
# data_dir='/root/userfolder/Dataset/FA/multi_files/preproccessed'
# data_dir='/root/userfolder/Dataset/FA/multi_files/registered'

# def cent_crop(data):
#     shape=[143,172,143,1]
#     d = (145-shape[0])//2
#     h = (174-shape[1])//2
#     w = (145-shape[2])//2
#     # data_croped = data[d - shape[0]:d, h - shape[1]:h, w - shape[2]:w]
#     begin = [d, h, w,0]
#     data_croped = tf.slice(data, begin, shape)
#     return data_croped

def _parse_function_full(example_proto):
  features = {"label": tf.FixedLenFeature([6], tf.int64),
              "img_raw": tf.FixedLenFeature([], tf.string)}
  parsed_features = tf.parse_single_example(example_proto, features)
  img = tf.decode_raw(parsed_features['img_raw'], tf.float32)
  img = tf.reshape(img, [72, 87, 72, 1])
  # img=cent_crop(img)
  img = tf.cast(img, tf.float32)
  print(parsed_features['label'])
  print('img shape~~~~~~~~~~~~~~~~:{}'.format(img.get_shape()))
  label = tf.cast(parsed_features['label'], tf.int64)
  print(label)
  # label=tf.cond(label < 1, lambda: tf.constant([1, 0]), lambda: tf.constant([0, 1]))
  label=tf.reshape(label,[1,6])
  return img, label

def _parse_function_30(example_proto):
  features = {"label": tf.FixedLenFeature([6], tf.int64),
              "img_raw": tf.FixedLenFeature([], tf.string)}
  parsed_features = tf.parse_single_example(example_proto, features)
  img = tf.decode_raw(parsed_features['img_raw'], tf.float32)
  shape = [round(raw_size[0] *30/100),round(raw_size[1] *30/100),round(raw_size[2] *30/100),1]
  img = tf.reshape(img, shape)
  # img=cent_crop(img)
  img = tf.cast(img, tf.float32)
  print(parsed_features['label'])
  print('img shape~~~~~~~~~~~~~~~~:{}'.format(img.get_shape()))
  label = tf.cast(parsed_features['label'], tf.int64)
  print(label)
  # label=tf.cond(label < 1, lambda: tf.constant([1, 0]), lambda: tf.constant([0, 1]))
  label=tf.reshape(label,[1,6])
  return img, label
def _parse_function_32(example_proto):
  features = {"label": tf.FixedLenFeature([6], tf.int64),
              "img_raw": tf.FixedLenFeature([], tf.string)}
  parsed_features = tf.parse_single_example(example_proto, features)
  img = tf.decode_raw(parsed_features['img_raw'], tf.float32)
  shape = [round(raw_size[0] *32/100),round(raw_size[1] *32/100),round(raw_size[2] *32/100),1]
  img = tf.reshape(img, shape)
  # img=cent_crop(img)
  img = tf.cast(img, tf.float32)
  print(parsed_features['label'])
  print('img shape~~~~~~~~~~~~~~~~:{}'.format(img.get_shape()))
  label = tf.cast(parsed_features['label'], tf.int64)
  print(label)
  # label=tf.cond(label < 1, lambda: tf.constant([1, 0]), lambda: tf.constant([0, 1]))
  label=tf.reshape(label,[1,6])
  return img, label



def _parse_function_40(example_proto):
  features = {"label": tf.FixedLenFeature([6], tf.int64),
              "img_raw": tf.FixedLenFeature([], tf.string)}
  parsed_features = tf.parse_single_example(example_proto, features)
  img = tf.decode_raw(parsed_features['img_raw'], tf.float32)
  shape = [round(raw_size[0] *40/100),round(raw_size[1] *40/100),round(raw_size[2] *40/100),1]
  img = tf.reshape(img, shape)
  # img=cent_crop(img)
  img = tf.cast(img, tf.float32)
  print(parsed_features['label'])
  print('img shape~~~~~~~~~~~~~~~~:{}'.format(img.get_shape()))
  label = tf.cast(parsed_features['label'], tf.int64)
  print(label)
  # label=tf.cond(label < 1, lambda: tf.constant([1, 0]), lambda: tf.constant([0, 1]))
  label=tf.reshape(label,[1,6])
  return img, label



def _parse_function_50(example_proto):
  features = {"label": tf.FixedLenFeature([6], tf.int64),
              "img_raw": tf.FixedLenFeature([], tf.string)}
  parsed_features = tf.parse_single_example(example_proto, features)
  img = tf.decode_raw(parsed_features['img_raw'], tf.float32)
  shape = [round(raw_size[0] *50//100),round(raw_size[1] *50//100),round(raw_size[2] *50//100),1]
  img = tf.reshape(img, shape)
  # img=cent_crop(img)
  img = tf.cast(img, tf.float32)
  print(parsed_features['label'])
  print('img shape~~~~~~~~~~~~~~~~:{}'.format(img.get_shape()))
  label = tf.cast(parsed_features['label'], tf.int64)
  print(label)
  # label=tf.cond(label < 1, lambda: tf.constant([1, 0]), lambda: tf.constant([0, 1]))
  label=tf.reshape(label,[1,6])
  return img, label

def _parse_function_60(example_proto):
  features = {"label": tf.FixedLenFeature([6], tf.int64),
              "img_raw": tf.FixedLenFeature([], tf.string)}
  parsed_features = tf.parse_single_example(example_proto, features)
  img = tf.decode_raw(parsed_features['img_raw'], tf.float32)
  img = tf.reshape(img, [87, 104, 87, 1])
  # img=cent_crop(img)
  img = tf.cast(img, tf.float32)
  print(parsed_features['label'])
  print('img shape~~~~~~~~~~~~~~~~:{}'.format(img.get_shape()))
  label = tf.cast(parsed_features['label'], tf.int64)
  print(label)
  # label=tf.cond(label < 1, lambda: tf.constant([1, 0]), lambda: tf.constant([0, 1]))
  label=tf.reshape(label,[1,6])
  return img, label

def _parse_function_croped(example_proto):
  print(example_proto)
  features = {"img_raw": tf.FixedLenFeature([], tf.string),
      "label": tf.FixedLenFeature([6], tf.int64)}
  parsed_features = tf.parse_single_example(example_proto, features)
  img = tf.decode_raw(parsed_features['img_raw'], tf.float32)
  img = tf.reshape(img, [145, 174, 145, 1])
  # img=crop(img)

  # r=random.random()
  # if r>0.5:
  #     img=tf.transpose(img,[2,1,0,3])
  img = tf.cast(img, tf.float32)
  print('img shape~~~~~~~~~~~~~~~~:{}'.format(img.get_shape()))
  print(parsed_features['label'])
  label = tf.cast(parsed_features['label'], tf.int64)
  print(label)
  # label=tf.cond(label < 1, lambda: tf.constant([1, 0]), lambda: tf.constant([0, 1]))
  label=tf.reshape(label,[1,6])
  return img, label

def load_data(sess,filename,batch_size,zoom_rate,shuffle_buffer=None):
    dataset = tf.data.TFRecordDataset(filename)
    if zoom_rate==60:
        _parse_function=_parse_function_60
    elif zoom_rate==50:
        _parse_function = _parse_function_50
    elif zoom_rate==40:
        _parse_function = _parse_function_40
    elif zoom_rate==30:
        _parse_function = _parse_function_30
    elif zoom_rate==32:
        _parse_function = _parse_function_32
    else:
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
def load_data_with_val(sess,batch_size,zoom_rate,shuffle_buffer=None,cross=0,brain_area=None,modal='native',test_pos=0):
    data_dir = '/root/userfolder/Dataset/FA/multi_files/'
    if modal=='native':
        data_dir = os.path.join(data_dir, 'nativeFA/{}'.format(cross))
    else:
        data_dir = os.path.join(data_dir, 'flirt/{}'.format(cross))
    file_list = []
    area=''
    if brain_area is not None:
        area='covered/area_{}/'.format(brain_area)
    for i in range(11):
        p = os.path.join(data_dir, '{}train_zoomed_r{}_{}.tfrecord'.format(area,z, i))
        file_list.append(p)
    print(file_list)

    val_file=file_list[test_pos]
    test_file=file_list[test_pos]
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


