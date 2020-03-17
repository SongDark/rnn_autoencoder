# coding:utf-8

import tensorflow as tf 

# spectral norm
def spectral_norm(w, iteration=10, name="sn"):
	'''
	Ref: https://github.com/taki0112/Spectral_Normalization-Tensorflow/blob/65218e8cc6916d24b49504c337981548685e1be1/spectral_norm.py
	'''
	w_shape = w.shape.as_list() # [KH, KW, Cin, Cout] or [H, W]
	w = tf.reshape(w, [-1, w_shape[-1]]) # [KH*KW*Cin, Cout] or [H, W]

	u = tf.get_variable(name+"_u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)
	s = tf.get_variable(name+"_sigma", [1, ], initializer=tf.random_normal_initializer(), trainable=False)

	u_hat = u # [1, Cout] or [1, W]
	v_hat = None 

	for _ in range(iteration):
		v_hat = tf.nn.l2_normalize(tf.matmul(u_hat, tf.transpose(w))) # [1, KH*KW*Cin] or [1, H]
		u_hat = tf.nn.l2_normalize(tf.matmul(v_hat, w)) # [1, Cout] or [1, W]
		
	u_hat = tf.stop_gradient(u_hat)
	v_hat = tf.stop_gradient(v_hat)

	sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat)) # [1,1]
	sigma = tf.reshape(sigma, (1,))

	with tf.control_dependencies([u.assign(u_hat), s.assign(sigma)]):
		# ops here run after u.assign(u_hat)
		w_norm = w / sigma 
		w_norm = tf.reshape(w_norm, w_shape)
	
	return w_norm

# batch norm
def bn(x, is_training, name):
	return tf.contrib.layers.batch_norm(x, decay=0.999, updates_collections=None, epsilon=0.001, scale=True,fused=False,is_training=is_training,scope=name)


def conv2d(x, channel, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, sn=False, padding="VALID", bias=True, name='conv2d'):
	with tf.variable_scope(name):
		w = tf.get_variable('weights', [k_h, k_w, x.get_shape()[-1], channel], initializer=tf.truncated_normal_initializer(stddev=stddev))
		if sn:
			conv = tf.nn.conv2d(x, spectral_norm(w, name="sn"), strides=[1, d_h, d_w, 1], padding=padding)
		else:
			conv = tf.nn.conv2d(x, w, strides=[1, d_h, d_w, 1], padding=padding)
		if bias:
			biases = tf.get_variable('biases', shape=[channel], initializer=tf.zeros_initializer())
			conv = tf.nn.bias_add(conv, biases)
	return conv

def deconv2d(x, channel, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, sn=False, padding='VALID', name='deconv2d'):
    '''deconvolution 2d'''
    def get_deconv_lens(H, k, d):
		if padding == "VALID":
			# return tf.multiply(H, d) + k - 1
			return H * d + k - 1
		elif padding == "SAME":
			# return tf.multiply(H, d)
			return H * d
    shape = tf.shape(x)
    H, W = shape[1], shape[2]
    N, _, _, C = x.get_shape().as_list()
    with tf.variable_scope(name):
		w = tf.get_variable('weights', [k_h, k_w, channel, x.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
		biases = tf.get_variable('biases', shape=[channel], initializer=tf.zeros_initializer())
		if sn:
			w = spectral_norm(w, name="sn")
    
    # N, H, W, C = x.get_shape().as_list() # ???
    H1 = get_deconv_lens(H, k_h, d_h)
    W1 = get_deconv_lens(W, k_w, d_w)
    deconv = tf.nn.conv2d_transpose(x, w, output_shape=[N, H1, W1, channel], strides=[1, d_h, d_w, 1], padding=padding)
    deconv = tf.nn.bias_add(deconv, biases)
    
    return deconv

def dense(x, output_size, sn=False, activation=None, reuse=False, name='dense'):
    '''dense layer'''
    shape = x.get_shape().as_list()
    with tf.variable_scope(name, reuse=reuse):
        W = tf.get_variable(
			'weights', [shape[1], output_size], 
			tf.float32, 
			tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable(
			'biases', [output_size], 
			initializer=tf.constant_initializer(0.0))
        if sn:
			W = spectral_norm(W, name="sn")
	out = tf.matmul(x, W) + bias 
	if activation is not None:
		out = activation(out)
	return out

def lrelu(x, leak=0.2, name='leaky_relu'):
	return tf.maximum(x, leak*x, name=name) 

def resblock(x, channel, is_training=True, name='resnet'):
    with tf.variable_scope(name):
        net = tf.pad(x, [[0,0], [4,4], [0,0], [0,0]], 'REFLECT')
        net = tf.nn.relu(bn(conv2d(net, channel, 9,1,1,1, padding='VALID', name=name+'_c1'), is_training, name=name+'_l1'))
        net = tf.pad(net, [[0,0], [4,4], [0,0], [0,0]], 'REFLECT')
        net = bn(conv2d(net, channel, 9,1,1,1, padding='VALID', name=name+'_c2'), is_training, name=name+'_l2')
        net = tf.nn.relu(x + net)
    return net

def gen_rnn_cells(hidden_units=[128], rnn_type='rnn'):
	if rnn_type == 'rnn':
		rnn_layers = [tf.nn.rnn_cell.BasicRNNCell(n) for n in hidden_units]
	elif rnn_type == 'lstm':
		rnn_layers = [tf.nn.rnn_cell.LSTMCell(n, use_peepholes=True) for n in hidden_units]
	elif rnn_type == 'gru':
		rnn_layers = [tf.nn.rnn_cell.GRUCell(n) for n in hidden_units]
	return tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
