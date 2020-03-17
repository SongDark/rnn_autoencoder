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
            # input embedding
            x_dim = x.get_shape().as_list()[2]
            W = tf.get_variable('input_weights', [x_dim, self.encoder_hidden_units[0]], tf.float32, tf.random_normal_initializer(stddev=0.02))
            b = tf.get_variable('input_biases', [self.encoder_hidden_units[0]], initializer=tf.constant_initializer(0.0))
            x_in = tf.einsum("ntd,dh->nth", x, W) + b

            # rnn encoding
            multi_cells = gen_rnn_cells(self.encoder_hidden_units, self.rnn_type)
            outs, states = tf.nn.dynamic_rnn(
                cell=multi_cells,
                inputs=x_in,
                sequence_length=lens,
                time_major=False,
                dtype=tf.float32 
            )

            # output embedding
            if self.rnn_type == 'lstm':
                emb = tf.nn.rnn_cell.LSTMStateTuple(
                    c=dense(states[-1].c, self.len_latent, name='fc_c'), 
                    h=dense(states[-1].h, self.len_latent, name='fc_h')
                )
            else:
                emb = dense(states[-1], self.len_latent, name='fc_latent')

        return emb
    
    def decode(self, emb, lens, is_training=True, reuse=False):
        with tf.variable_scope(self.name + '_decoder', reuse=reuse):
            
            inputs_dim = 1
            maxlen = tf.reduce_max(lens)

            # input embedding
            if self.rnn_type == 'lstm':
                batch_size = emb.c.get_shape().as_list()[0]
                emb = tf.nn.rnn_cell.LSTMStateTuple(
                    c=dense(emb.c, self.decoder_hidden_units[0], name='fc_input_c'),
                    h=dense(emb.h, self.decoder_hidden_units[0], name='fc_input_h'),
                )
            else:
                batch_size = emb.get_shape().as_list()[0]
                emb = dense(emb, self.decoder_hidden_units[0], name='fc_input')

            # rnn decoding
            multi_cells = gen_rnn_cells(self.decoder_hidden_units, self.rnn_type)
            outs, states = tf.nn.dynamic_rnn(
                cell=multi_cells,
                inputs=tf.zeros(shape=(batch_size, maxlen, inputs_dim), dtype=tf.float32),
                sequence_length=lens,
                time_major=False,
                initial_state=tuple([emb]),
                dtype=tf.float32
            )

            W = tf.get_variable('output_weights', [self.decoder_hidden_units[-1], self.data_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))
            b = tf.get_variable('output_biases', [self.data_dim], initializer=tf.constant_initializer(0.0))
            outs = tf.einsum("ntd,dh->nth", outs, W) + b

        return outs
    
    def decode_1(self, emb, lens, is_training=True, reuse=False):
        with tf.variable_scope(self.name + '_decoder', reuse=reuse):
            
            maxlen = tf.reduce_max(lens)

            # input copying
            if self.rnn_type == 'lstm':
                emb = emb.c
            batch_size = emb.get_shape().as_list()[0]
            
            inputs = tf.reshape(emb, (batch_size, 1, -1))
            inputs = tf.tile(inputs, (1, maxlen, 1))

            # rnn decoding
            multi_cells = gen_rnn_cells(self.decoder_hidden_units, self.rnn_type)
            outs, states = tf.nn.dynamic_rnn(
                cell=multi_cells,
                inputs=inputs,
                sequence_length=lens,
                time_major=False,
                dtype=tf.float32
            )

            W = tf.get_variable('output_weights', [self.decoder_hidden_units[-1], self.data_dim], tf.float32, tf.random_normal_initializer(stddev=0.02))
            b = tf.get_variable('output_biases', [self.data_dim], initializer=tf.constant_initializer(0.0))
            outs = tf.einsum("ntd,dh->nth", outs, W) + b

        return outs

    


# # X = tf.random_normal((5, 256, 6), 0.0, 1.0, tf.float32)
# X = tf.placeholder(shape=(5, None, 6), dtype=tf.float32)
# L = tf.placeholder(shape=(5), dtype=tf.int32)
# AE = AutoEncoder(10)

# E = AE.encode(X)
# Y = AE.decode(E, lens=L)

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
        print self.sample_data['data'].shape
        print self.sample_data['lens']

        self.build_placeholder()
        self.build_network()
        self.build_seq_reconstruction()

        self.build_sess()
        self.build_dirs()
        
    def build_placeholder(self):
        self.source = tf.placeholder(shape=(self.batch_size, None, self.data_dim), dtype=tf.float32)
        self.length = tf.placeholder(shape=(self.batch_size), dtype=tf.int32)
        self.target = tf.placeholder(shape=(self.batch_size, None, self.data_dim), dtype=tf.float32)
    
    def build_network(self):
        self.embedding = self.autoencoder.encode(self.source, self.length, True, False)
        self.reconstruct = self.autoencoder.decode_1(self.embedding, self.length, True, False)

    def build_seq_reconstruction(self):
        self.cycloss = tf.reduce_mean(tf.squared_difference(self.reconstruct, self.target))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.cycsolver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.9).minimize(self.cycloss, var_list=self.autoencoder.vars)
    
    def sample(self, epoch):
        embedding_test = self.autoencoder.encode(self.source, self.length, False, True)
        reconstruct_test = self.autoencoder.decode_1(embedding_test, self.length, False, True)

        feed_dict = {
            self.source : self.sample_data[self.data_key],
            self.length : self.sample_data['lens']
        }
        cyc = self.sess.run(reconstruct_test, feed_dict=feed_dict)

        plot2d(cyc, self.sample_data['lens'], os.path.join(self.fig_dir, "cyc_{}.png".format(epoch)))
        # plot(cyc, [180] * self.batch_size, os.path.join(self.fig_dir, "cyc_{}.png".format(epoch)))

        if epoch==0:
            plot2d(self.sample_data[self.data_key], self.sample_data['lens'], os.path.join(self.fig_dir, "ground_truth.png"))
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
                    cycloss_train = self.sess.run(self.cycloss, feed_dict=feed_dict_train)

                    print "Epoch [%3d/%3d] Iter [%3d/%3d] cycloss=%.4f" % (epoch, epoches, idx, batches_per_epoch, cycloss_train)

            if epoch % 50 == 0:
                self.sample(epoch)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'))
    

    
def rnn_test():
    datamanager = CT('CT_seq', train_ratio=0.8, expand_dim=None, seed=0)
    autoencoder = AutoEncoder(2, 64, 'rnn', [128], [128], 'AE')
    trainer = Trainer(64, datamanager, autoencoder, version='test', save_path='save/rnn_ae/rnn/', gpu='2')
    # trainer.train(301)
    trainer.load_model()
    trainer.cal_loss()

def lstm_test():
    datamanager = CT('CT_seq', train_ratio=0.8, expand_dim=None, seed=0)
    autoencoder = AutoEncoder(2, 64, 'lstm', [128], [128], 'AE')
    trainer = Trainer(64, datamanager, autoencoder, version='test', save_path='save/rnn_ae/lstm/', gpu='1')
    # trainer.train(301) # 
    trainer.load_model()
    trainer.cal_loss()

def gru_test():
    datamanager = CT('CT_seq', train_ratio=0.8, expand_dim=None, seed=0)
    autoencoder = AutoEncoder(2, 64, 'gru', [128], [128], 'AE')
    trainer = Trainer(64, datamanager, autoencoder, version='test', save_path='save/rnn_ae/gru/', gpu='1')
    # trainer.train(301)
    trainer.load_model()
    trainer.cal_loss()

# rnn_test()
# lstm_test()
gru_test()