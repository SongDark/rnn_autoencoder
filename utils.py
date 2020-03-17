# coding:utf-8

import tensorflow as tf 
import numpy as np
import os 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class BasicBlock(object):
    def __init__(self, name):
        self.name = name
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
    @property
    def encoder_vars(self):
        vs = self.vars 
        return [v for v in vs if 'encode' in v.name]
    @property
    def decoder_vars(self):
        vs = self.vars 
        return [v for v in vs if 'decode' in v.name]

class BasicTrainFramework(object):
	def __init__(self, batch_size, version, save_path, gpu='0'):
		self.batch_size = batch_size
		self.version = version
		self.save_path = save_path
		os.environ["CUDA_VISIBLE_DEVICES"] = gpu

	def build_dirs(self):
		self.log_dir = os.path.join(self.save_path, 'logs') 
		self.model_dir = os.path.join(self.save_path, 'checkpoints')
		self.fig_dir = os.path.join(self.save_path, 'figs')
		for d in [self.log_dir, self.model_dir, self.fig_dir]:
			if (d is not None) and (not os.path.exists(d)):
				print "mkdir " + d
				os.makedirs(d)
	
	def build_sess(self):
		gpu_options = tf.GPUOptions(allow_growth=True)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver()

	def load_model(self, checkpoint_dir=None, ckpt_name=None):
		import re 
		print "load checkpoints ..."
		checkpoint_dir = checkpoint_dir or self.model_dir
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = ckpt_name or os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			# counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print "Success to read {}".format(ckpt_name)
			return True, 0
		else:
			print "Failed to find a checkpoint"
			return False, 0

def check_folder(des_dir):
    if not os.path.exists(des_dir):
        os.makedirs(des_dir)
        return False
    return True

def event_reader(event_path, event_name=None, names=[]):
    # get the newest event file
    if event_name is None:
        fs = os.listdir(event_path)
        fs.sort(key=lambda fn:os.path.getmtime(os.path.join(event_path, fn)))
        event_name = fs[-1]
    print "load from event:", os.path.join(event_path, event_name)
    res = {}
    for n in names:
        res[n] = []
    for e in tf.train.summary_iterator(os.path.join(event_path, event_name)):
        for v in e.summary.value:
            for n in names:
                if n == v.tag:
                    res[n].append(float(v.simple_value))
    return res

def one_hot_encode(ys, max_class):
    '''one-hot encoding'''
    res = np.zeros((len(ys), max_class), dtype=np.float32)
    for i in range(len(ys)):
        res[i][ys[i]] = 1.0
    return res

def shuffle_in_unison_scary(*args, **kwargs):
	np.random.seed(kwargs['seed'])
	rng_state = np.random.get_state()
	for i in range(len(args)):
		np.random.shuffle(args[i])
		np.random.set_state(rng_state)

def conv_cond_concat(x, y):
    # x: [N, H, W, C]
    # y: [N, 1, 1, d]
    # x_shapes, y_shapes = x.get_shape(), y.get_shape()
    x_shapes, y_shapes = tf.shape(x), tf.shape(y)
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def imshow(imgs, save_path, n=5):
    tmp = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            tmp[i].append(imgs[i*n+j])
        tmp[i] = np.concatenate(tmp[i], 1)
    tmp = np.concatenate(tmp, 0)
    plt.imshow(tmp[:,:,0], cmap=plt.cm.gray)
    plt.savefig(save_path)
    plt.clf()

def plot2d(seqs, lens, titles, save_path, n=5):
    for i in range(n):
        for j in range(n):
            idx = i*n + j 
            plt.subplot(n, n, idx+1)
            plt.plot(seqs[idx][:lens[idx], 0], seqs[idx][:lens[idx], 1], linewidth=2)
            plt.xticks([])
            plt.yticks([])
            if titles is not None:
                plt.title(titles[idx])
    plt.savefig(save_path)
    plt.clf()

def plot(seqs, lens, save_path, n=5):
    for i in range(n):
        for j in range(n):
            idx = i*n + j 
            plt.subplot(n, n, idx+1)
            plt.plot(seqs[idx][:lens[idx], :], linewidth=2)
            plt.xticks([])
            plt.yticks([])
    plt.savefig(save_path)
    plt.clf()

def scatter(xs, ys, cs, save_path):
    plt.scatter(xs, ys, c=cs)
    plt.colorbar()
    # plt.savefig(save_path)
    # plt.clf()

def tsne(embs):
    from sklearn.manifold import TSNE

    model = TSNE(n_components=2, random_state=0)
    embs = model.fit_transform(embs)

    return embs


