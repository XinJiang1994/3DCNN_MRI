import time
from ops import *
import nibabel as nib
import matplotlib.pyplot as plt



class CNN_MRI():
    def __init__(self,sess,config,input_shape,batch_size,zoom_rate=100,y_dim1=2,y_dim2=4,stride=[1,1,1,1,1],padding='SAME',checkpoint_dir='./checkpoint',model_name='CNN',isTrain=True):
        self.sess=sess
        self.input_shape=input_shape
        self.batch_size=batch_size
        self.zoom_rate=zoom_rate
        self.y_dim1=y_dim1
        self.y_dim2=y_dim2
        self.stride=stride
        self.padding=padding
        self.model_name=model_name
        self.checkpoint_dir=checkpoint_dir
        self.isTrain=isTrain
        self.keep_prob = tf.placeholder(tf.float32)
        self.config=config
        self.build_model()
        try:
            tf.global_variables_initializer().run()
        except:
            tf.global_variables_initializer().run()
        show_all_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name and 'bn_' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name and 'bn_' in g.name]
        var_list = self.t_vars + bn_moving_vars
        # self.saver = tf.train.Saver(var_list)
        self.saver = tf.train.Saver()

    def build_model(self):
        devices=['/device:GPU:0','/device:GPU:0','/device:GPU:0']
        self.latent=[]
        with tf.variable_scope("cnn") as scope:
            def weight_variable(shape,name):
                initial = tf.truncated_normal(shape, stddev=0.1)
                return tf.Variable(initial,name=name)

            def bias_variable(shape,name):
                initial = tf.constant(0.1, shape=shape)
                return tf.Variable(initial,name=name)
            def conv3d(x, W,stride,padding='SAME'):
                return tf.nn.conv3d(x, W, strides=stride, padding=padding)

            def sigmoid_cross_entropy_with_logits(x, y):
                try:
                    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
                except:
                    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)
            def softmax_cross_entropy_with_logits(x, y):
                try:
                    return tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=y)
                except:
                    return tf.nn.softmax_cross_entropy_with_logits(logits=x, targets=y)
            input_size=[self.input_shape[0],round(self.input_shape[1]*self.zoom_rate/100),
                        round(self.input_shape[2]*self.zoom_rate/100),round(self.input_shape[3]*self.zoom_rate/100),
                        self.input_shape[4]]
            self.x = tf.placeholder(tf.float32, input_size, 'input')
            self.label=tf.placeholder(tf.float32, [self.input_shape[0],6], 'input')

            # x_image=max_pool_2x2(self.x)
            with tf.device(devices[1]):
                x_image=self.x
                # x_image = tf.nn.max_pool3d(x_image, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='VALID',
                #                            name='Max_pooling_conv0')
                # self.latent.append(x_image)

                #The 1st convn layer
                W_conv1 = weight_variable([3, 3, 3, self.input_shape[4], 32],name='cnn_W_conv1')
                b_conv1 = bias_variable([32], name='cnn_b_conv1')
                h_conv1 = conv3d(x_image, W_conv1, stride=[1, 1, 1, 1, 1], padding='VALID')
                h_conv1 = tf.nn.bias_add(h_conv1, b_conv1)

                # W_conv1_1 = weight_variable([3, 3, 3, 32, 32], name='cnn_W_conv1_1')
                # b_conv1_1 = bias_variable([32], name='cnn_b_conv1_1')
                # h_conv1 = conv3d(h_conv1, W_conv1_1, stride=[1, 1, 1, 1, 1], padding='SAME')
                # h_conv1 = tf.nn.bias_add(h_conv1, b_conv1_1)
            with tf.device(devices[2]):
                bn1=batch_norm(name='bn_1')
                h_conv1=bn1(h_conv1,train=self.isTrain)
                h_conv1=lrelu(h_conv1)
                #self.latent.append(h_conv1)
                h_pool1 = tf.nn.max_pool3d(h_conv1,ksize=[1,3,3,3,1],strides=[1,2,2,2,1],padding='VALID',name='Max_pooling_conv1')
                self.latent.append(h_pool1)
            with tf.device(devices[1]):
                #the 2nd convn layer412557

                W_conv2 = weight_variable([3, 3, 3, 32, 64], name='cnn_W_conv2')
                b_conv2 = bias_variable([64], name='cnn_b_conv2')
                h_conv2 = conv3d(h_pool1, W_conv2,stride=self.stride)
                h_conv2=tf.nn.bias_add(h_conv2,b_conv2)

                # W_conv2_2 = weight_variable([3, 3, 3, 64, 64], name='cnn_W_conv2_2')
                # b_conv2_2 = bias_variable([64], name='cnn_b_conv2_2')
                # h_conv2 = conv3d(h_conv2, W_conv2_2, stride=self.stride)
                # h_conv2 = tf.nn.bias_add(h_conv2, b_conv2_2)

                bn2 = batch_norm(name='bn_2')
                h_conv2 = bn2(h_conv2,train=self.isTrain)
                h_conv2=lrelu(h_conv2)
                h_pool2 = tf.nn.max_pool3d(h_conv2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2 , 1], padding='VALID')
                self.latent.append(h_pool2)

                # # the 3rd convn layer
                W_conv3 = weight_variable([3, 3, 3, 64, 128], name='cnn_W_conv3')
                b_conv3 = bias_variable([128], name='cnn_b_conv3')
                h_conv3 = conv3d(h_pool2, W_conv3, stride=self.stride)
                h_conv3=tf.nn.bias_add(h_conv3,b_conv3)
                bn3 = batch_norm(name='bn_3')
                h_conv3 = bn3(h_conv3,train=self.isTrain)
                h_conv3=tf.nn.max_pool3d(h_conv3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2 , 1], padding='SAME')
                self.latent.append(h_conv3)

            #fc1
                h_pool_shape = h_conv3.get_shape().as_list()
            d = h_pool_shape[1]
            h = h_pool_shape[2]
            w = h_pool_shape[3]
            c = h_pool_shape[4]
            print('d,h,w,c --------:{} {} {} {}'.format(d, h, w, c))


            with tf.device(devices[1]):
                self.h_pool3_flat = tf.reshape(h_conv3, [-1, h * w * d * c])

                W_fc1 = weight_variable([h * w * d * c, self.y_dim1], name='g_W_fc1')
                b_fc1 = bias_variable([self.y_dim1], name='g_b_fc1')

                h_fc1_logits_= tf.matmul(self.h_pool3_flat, W_fc1)
                self.y_logits_=tf.nn.bias_add(h_fc1_logits_,b_fc1)
                self.y_=tf.nn.softmax(self.y_logits_)
            batch_s=self.input_shape[0]
            label_g=tf.slice(self.label,[0,0],[batch_s,2])
            self.g_loss=tf.reduce_mean(softmax_cross_entropy_with_logits(self.y_logits_,label_g))
            self.g_loss_sum=tf.summary.scalar('g_loss',self.g_loss)

            gender_l=tf.argmax(label_g,1)
            gender_l=tf.reshape(gender_l, [batch_s, 1])

            gender_p=tf.argmax(self.y_,1)
            gender_p=tf.reshape(gender_p, [batch_s, 1])

            self.predicted_value =gender_p
            self.g_correct_prediction = tf.equal(gender_p, gender_l)
            self.g_accuracy = tf.reduce_mean(tf.cast(self.g_correct_prediction, tf.float32))

            t_vars=tf.trainable_variables()
            self.t_vars=t_vars
            self.g_vars=[var for var in t_vars if 'g_' in var.name or 'cnn_' in var.name]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.g_optim = tf.train.AdamOptimizer(self.config.learning_rate, beta1=self.config.beta1).minimize(self.g_loss,
                                                                                                    var_list=self.t_vars)


    def train(self,next_batch,next_batch_v,config,data_seq=None):

        g_sum=merge_summary([self.g_loss_sum])
        self.writer = SummaryWriter("./logs/{}".format(data_seq), self.sess.graph)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS, checkpoint_counter:{}".format(checkpoint_counter))
        else:
            print(" [!] Load failed...")

        show_all_variables()
        batch_idx=20
        max_acc = self.val_acc(next_batch_v,config)
        second_acc=0
        third_acc=0

        for iter in xrange(config.epoch):
            for id in range(batch_idx):
                data,label=self.sess.run(next_batch)
                label = np.squeeze(label)
                _, g_loss, sum_str = self.sess.run([self.g_optim, self.g_loss, g_sum],
                                                 feed_dict={self.x: data, self.label: label, self.keep_prob: 0.8})
                self.writer.add_summary(sum_str, counter)
                print("train_cnn {} {} / {} / {} epoch,g_loss : {} ,time: {} ".format(self.model_name,id,iter + 1, config.epoch,
                                                                              g_loss,
                                                                              time.time() - start_time))
                counter+=1
                if np.mod(counter, 5) ==0:
                    n1 = 0
                    r1 = 0
                    for i in range(3):
                        data, label = self.sess.run(next_batch_v)
                        label = np.squeeze(label)
                        label = np.squeeze(label)
                        g1,result = self.cnn_correct(data, label, config)
                        n1 = n1 + config.batch_size
                        r1 = r1 + g1
                    a_v_g = r1 / n1
                    if a_v_g > max_acc:
                        max_acc=a_v_g
                        self.save(config.checkpoint_dir, counter)
                    print('Validation accuracy:{}'.format(a_v_g))
                # if np.mod(counter, 10) == 0:
                #     self.save(config.checkpoint_dir, counter)



    def cnn_test(self,next_batch,config):
        self.load(config.checkpoint_dir)
        data,label = self.sess.run(next_batch)
        label = np.squeeze(label)
        g_accuracy,age_accurcy = self.sess.run([self.g_accuracy,self.age_accuracy],feed_dict={self.x: data,self.label:label,self.keep_prob:1})
        return g_accuracy,age_accurcy
    # def cnn_predict_logits(self,data,label,config):
    #     self.load(config.checkpoint_dir)
    #     predicted_logits = self.sess.run(self.predicted_value, feed_dict={self.x: data, self.keep_prob: 1})
    #     return predicted_logits
    def val_acc(self,next_batch_v,config):
        n1 = 0
        r1 = 0
        for i in range(100):
            data, label = self.sess.run(next_batch_v)
            label = np.squeeze(label)
            g1,result = self.cnn_correct(data, label, config)
            n1 = n1 + config.batch_size
            r1 = r1 + g1
        a_v_g = r1 / n1
        return a_v_g
    def predict_y(self,data,label,config):
        # self.load(config.checkpoint_dir)
        g = self.sess.run([self.y_], feed_dict={self.x: data, self.keep_prob: 1})
        #gender=np.argmax(g, 1)
        return g
    def cnn_correct(self,data,label,config):
        # self.load(config.checkpoint_dir)
        g_correct_prediction = self.sess.run([self.g_correct_prediction], feed_dict={self.x: data, self.label:label,self.keep_prob: 1})
        g=np.sum(g_correct_prediction)
        return g,g_correct_prediction
    def get_latent(self,data,config,l):
        self.load(config.checkpoint_dir)
        latent=self.sess.run(self.latent[l-1],feed_dict={self.x: data,self.keep_prob:1})
        return latent

    @property
    def model_dir(self):
        return "{}".format(
            self.model_name)

    def save(self, checkpoint_dir, step):
        model_name = "CNN_MRI.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
    def visualize(self,next_batch,config):
        latent_dir = './latent'
        if not os.path.isdir(latent_dir):
            os.mkdir(latent_dir)
        # get the first layer's latent
        layers = [1, 2, 3]
        data, label = self.sess.run(next_batch)
        data=data[0:10]
        label=label[0:10]
        label = np.squeeze(label)
        for l in layers:
            latent = self.get_latent(data, config, l)
            shape = list(latent.shape)
            print(shape)
            for i in range(shape[0]):
                imgs = []
                for j in range(shape[4]):
                    arr = latent[i, :, :, :, j]
                    amin, amax = arr.min(), arr.max()
                    arr = 255 * (arr - amin) / (amax - amin)
                    arr = arr.astype(np.int32)
                    save_name = os.path.join(latent_dir, 'latent_L_{}_batch_{}_FM_{}.nii.gz'.format(l, i, j))
                    affine = np.eye(4, 4)
                    nib.save(nib.Nifti1Image(arr, affine), save_name)

                    axial_middle = arr.shape[2] // 2
                    imgs.append(arr[:, :, axial_middle].T)

                if l == 1:
                    sex = ''
                    if label[i][0] == 1:
                        sex = 'male'
                    else:
                        sex = 'female'
                    pic_name = os.path.join(latent_dir, 'Latent{}_sample{}_{}.png'.format(l, i, sex))
                    fig, axes = plt.subplots(nrows=4, ncols=8, sharex=True, sharey=True, figsize=(12.8, 8))
                    for image, row in zip([imgs[:8], imgs[8:16], imgs[16:24], imgs[24:32]], axes):
                        for img, ax in zip(image, row):
                            ax.imshow(img)
                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)
                    fig.tight_layout(pad=0)
                    plt.savefig(pic_name, bbox_inches='tight')
                    plt.close(fig)
                if l == 2:
                    sex = ''
                    if label[i][0] == 1:
                        sex = 'male'
                    else:
                        sex = 'female'
                    pic_name = os.path.join(latent_dir, 'Latent{}_sample{}_{}.png'.format(l, i, sex))
                    fig, axes = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True, figsize=(6.4, 8))
                    for image, row in zip(
                            [imgs[:8], imgs[8:16], imgs[16:24], imgs[24:32], imgs[32:40], imgs[40:48], imgs[48:56],
                             imgs[56:64]], axes):
                        for img, ax in zip(image, row):
                            ax.imshow(img)
                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)
                    fig.tight_layout(pad=0.0)
                    plt.savefig(pic_name, bbox_inches='tight')
                    plt.close(fig)
                if l == 3:
                    print('latent3 ##############')
                    sex = ''
                    if label[i][0] == 1:
                        sex = 'male'
                    else:
                        sex = 'female'
                    pic_name = os.path.join(latent_dir, 'Latent{}_sample{}_{}.png'.format(l, i, sex))
                    fig, axes = plt.subplots(nrows=8, ncols=16, sharex=True, sharey=True, figsize=(6, 4))
                    for image, row in zip(
                            [imgs[:16], imgs[16:32], imgs[32:48], imgs[48:64], imgs[64:80], imgs[80:96], imgs[96:112],
                             imgs[112:128]], axes):
                        for img, ax in zip(image, row):
                            ax.imshow(img)
                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)
                    print(imgs[0])
                    fig.tight_layout(pad=0.0)
                    plt.savefig(pic_name, bbox_inches='tight')
                    plt.close(fig)
