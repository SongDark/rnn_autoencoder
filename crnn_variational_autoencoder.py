# coding:utf-8
from utils import *
from ops import *
from datamanager import AirWriting
from datamanager_ct import CT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class AutoEncoder(BasicBlock):
    def __init__(self,
            data_dim=6,
            len_latent=64, 
            rnn_type='rnn', 
            encoder_hidden_units=[128],
            decoder_hidden_units=[128],
            name='AE'):
        super(AutoEncoder, self).__init__(name)
        
        self.data_dim = data_dim
        self.len_latent = len_latent

        self.rnn_type = rnn_type
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = decoder_hidden_units
    
    def encode(self, x, lens=None, is_training=True, reuse=False):
        with tf.variable_scope(self.name + '_encoder', reuse=reuse):
            batch_size = x.get_shape().as_list()[0] # x [n, T, d, 1]

            # convolutions
            net = lrelu(conv2d(x, 16, 9, 1, 2, 1, padding='SAME', name='c1'), name='l1')
            net = lrelu(bn(conv2d(net, 32, 9, 1, 2, 1, padding='SAME', name='c2'), is_training, name='bn2'), name='l2')
            net = lrelu(bn(conv2d(net, 64, 9, self.data_dim, 2, self.data_dim, padding='SAME', name='c3'), is_training, name='bn3'), name='l3')

            net = tf.reshape(net, (batch_size, -1, 64)) # [n, T//8, 64]

            conv_lens = tf.floor(tf.divide(lens, 8))

            # rnn encoding
            multi_cells = gen_rnn_cells(self.encoder_hidden_units, self.rnn_type)
            outs, states = tf.nn.dynamic_rnn(
                cell=multi_cells,
                inputs=net,
                sequence_length=conv_lens,
                time_major=False,
                dtype=tf.float32 
            )

            # output embedding
            if self.rnn_type == 'lstm':
                emb = states[-1].c
            else:
                emb = states[-1]
            
            mean_code = dense(emb, self.len_latent, name='fc_mean')
            std_code = dense(emb, self.len_latent, name='fc_std')

        return mean_code, std_code, conv_lens
    
    def convert_noise(self, mean_code, std_code):
        gaussian_noise = tf.random_normal(tf.shape(mean_code), 0.0, 1.0, tf.float32)
        return mean_code + tf.exp(std_code) * gaussian_noise
    
    def decode(self, noise, conv_lens, is_training=True, reuse=False):
        with tf.variable_scope(self.name + '_decoder', reuse=reuse):
            
            maxlen = tf.to_int32(tf.reduce_max(conv_lens))

            # input copying
            batch_size = noise.get_shape().as_list()[0]
            
            inputs = tf.reshape(noise, (batch_size, 1, -1))
            inputs = tf.tile(inputs, (1, maxlen, 1))

            # rnn decoding
            multi_cells = gen_rnn_cells(self.decoder_hidden_units, self.rnn_type)
            outs, states = tf.nn.dynamic_rnn(
                cell=multi_cells,
                inputs=inputs,
                sequence_length=conv_lens,
                time_major=False,
                dtype=tf.float32
            )
            outs = tf.expand_dims(outs, 2) # [n,T//8,1,128]

            net = lrelu(bn(deconv2d(outs, 64, 9, self.data_dim, 2, self.data_dim, padding='SAME', name='dc1'), is_training, name='bn1'), name='l1')
            net = lrelu(bn(deconv2d(net, 32, 9, 1, 2, 1, padding='SAME', name='dc2'), is_training, name='bn2'), name='l2')
            net = lrelu(bn(deconv2d(net, 16, 9, 1, 2, 1, padding='SAME', name='dc3'), is_training, name='bn3'), name='l3')
            net = conv2d(net, 1, 9, 1, 1, 1, padding='SAME', name='c1')

        return net

    


# # X = tf.random_normal((5, 256, 6), 0.0, 1.0, tf.float32)
# X = tf.placeholder(shape=(5, None, 6), dtype=tf.float32)
# L = tf.placeholder(shape=(5), dtype=tf.int32)
# AE = AutoEncoder(10)

# m,s = AE.encode(X)
# n = AE.convert_noise(m,s)
# Y = AE.decode(n, lens=L)

# t = tf.random_normal((5,), 0.0, 1.0, tf.float32)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print sess.run(Y, feed_dict={X:np.ones((5,100,6)), L:[45,45,56,78,91]}).shape



class Trainer(BasicTrainFramework):
    def __init__(self, batch_size, datamanager, autoencoder, version='trainer', save_path='', gpu='1'):
        super(Trainer, self).__init__(batch_size, version=version, save_path=save_path, gpu=gpu)

        self.autoencoder = autoencoder

        self.data = datamanager
        self.data_key = datamanager.data_key
        self.data_dim = datamanager.data_dim
        self.autoencoder.data_dim = self.data_dim
        self.sample_data = self.data(self.batch_size, phase='test', maxlen=None, var_list=[self.data_key, 'lens'])

        self.build_placeholder()
        self.build_network()
        self.build_seq_reconstruction()

        self.build_sess()
        self.build_dirs()
        
    def build_placeholder(self):
        self.source = tf.placeholder(shape=(self.batch_size, None, self.data_dim, 1), dtype=tf.float32)
        self.length = tf.placeholder(shape=(self.batch_size), dtype=tf.int32)
        self.target = tf.placeholder(shape=(self.batch_size, None, self.data_dim, 1), dtype=tf.float32)
    
    def build_network(self):
        self.mean_code, self.std_code, conv_lens = self.autoencoder.encode(self.source, self.length, True, False)
        self.embedding = self.autoencoder.convert_noise(self.mean_code, self.std_code)
        self.reconstruct = self.autoencoder.decode(self.embedding, conv_lens, True, False)

    def build_seq_reconstruction(self):
        self.cycloss = tf.reduce_mean(tf.squared_difference(self.reconstruct, self.target))
        self.latent_KL = 0.5 * tf.reduce_sum(tf.square(self.mean_code) + tf.exp(self.std_code) - self.std_code - 1.0)
        self.loss = self.cycloss + self.latent_KL

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.cycsolver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.9).minimize(self.loss, var_list=self.autoencoder.vars)
    
    def sample(self, epoch):
        mean_code_test, std_code_test, conv_lens_test = self.autoencoder.encode(self.source, self.length, False, True)
        embedding_test = self.autoencoder.convert_noise(mean_code_test, std_code_test)
        reconstruct_test = self.autoencoder.decode(embedding_test, conv_lens_test, False, True)

        feed_dict = {
            self.source : self.sample_data[self.data_key],
            self.length : self.sample_data['lens']
        }
        cyc = self.sess.run(reconstruct_test, feed_dict=feed_dict)

        plot2d(cyc, self.sample_data['lens'], None, os.path.join(self.fig_dir, "cyc_{}.png".format(epoch)))
        # plot(cyc, [180] * self.batch_size, os.path.join(self.fig_dir, "cyc_{}.png".format(epoch)))

        if epoch==0:
            plot2d(self.sample_data[self.data_key], self.sample_data['lens'], None, os.path.join(self.fig_dir, "ground_truth.png"))
            # plot(self.sample_data[self.data_key], [180] * self.batch_size, os.path.join(self.fig_dir, "ground_truth.png"))

    def cal_loss(self):
        train_cycloss = 0.0
        for _ in range(self.data.train_num // self.batch_size):
            data = self.data(self.batch_size, 'train', var_list=[self.data_key,'lens'])
            feed_dict = {
                self.source : data[self.data_key],
                self.length : data['lens'],
                self.target : data[self.data_key]
            }
            train_cycloss += self.sess.run(self.cycloss, feed_dict=feed_dict) * self.batch_size
        test_cycloss = 0.0
        for _ in range(self.data.test_num // self.batch_size):
            data = self.data(self.batch_size, 'test', var_list=[self.data_key,'lens'])
            feed_dict = {
                self.source : data[self.data_key],
                self.length : data['lens'],
                self.target : data[self.data_key]
            }
            test_cycloss += self.sess.run(self.cycloss, feed_dict=feed_dict) * self.batch_size
        
        print train_cycloss / float(self.data.train_num)
        print test_cycloss / float(self.data.test_num)

    def train(self, epoches=1):
        batches_per_epoch = self.data.train_num // self.batch_size

        for epoch in range(epoches):
            self.data.shuffle_train(seed=epoch)

            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx

                data = self.data(self.batch_size, 'train', maxlen=None, var_list=[self.data_key, 'lens'])

                feed_dict_train = {
                    self.source : data[self.data_key],
                    self.length : data['lens'],
                    self.target : data[self.data_key]
                }

                self.sess.run(self.cycsolver, feed_dict=feed_dict_train)

                if cnt % 25 == 0:
                    cycloss_train, KL_train = self.sess.run([self.cycloss, self.latent_KL], feed_dict=feed_dict_train)

                    print "Epoch [%3d/%3d] Iter [%3d/%3d] cycloss=%.4f KL=%.4f" % (epoch, epoches, idx, batches_per_epoch, cycloss_train, KL_train)

            if epoch % 50 == 0:
                self.sample(epoch)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'))

def rnn_test():
    datamanager = CT('CT_seq', train_ratio=0.8, expand_dim=3, seed=0)
    autoencoder = AutoEncoder(2, 64, 'rnn', [128], [128], 'AE')
    trainer = Trainer(64, datamanager, autoencoder, version='vae', save_path='save/crnn_vae/rnn/', gpu='0')
    trainer.train(301)

def gru_test():
    datamanager = CT('CT_seq', train_ratio=0.8, expand_dim=3, seed=0)
    autoencoder = AutoEncoder(2, 64, 'gru', [128], [128], 'AE')
    trainer = Trainer(64, datamanager, autoencoder, version='vae', save_path='save/crnn_vae/gru/', gpu='0')
    trainer.train(301)
    # trainer.load_model()
    # trainer.cal_loss()

def lstm_test():
    datamanager = CT('CT_seq', train_ratio=0.8, expand_dim=3, seed=0)
    autoencoder = AutoEncoder(2, 64, 'lstm', [128], [128], 'AE')
    trainer = Trainer(64, datamanager, autoencoder, version='vae', save_path='save/crnn_vae/lstm/', gpu='1')
    trainer.train(301)

gru_test()