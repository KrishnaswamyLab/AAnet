
import tensorflow as tf
import tensorflow.layers as layers
import numpy as np

def bn(x, name=None):
    return tf.layers.batch_normalization(x, training=tf.keras.backend.learning_phase(), momentum=.9, scale=True, fused=True, name=name)

class Encoder(object):
    def __init__(self, num_at, z_dim=[256,64], act_at=None):
        self.z_dim = z_dim
        self.num_at = num_at
        self.name = 'ae/enc_net'
        self.layers_dim = self.z_dim + [self.num_at-1]
        self.act_at = act_at

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = x
            for idx,out_dim in enumerate(self.layers_dim):
                if idx == len(self.layers_dim)-1:
                    act_fun = self.act_at
                else:
                    act_fun = tf.nn.leaky_relu
                fc = layers.dense(
                    fc, out_dim,
                    activation=act_fun,
                    kernel_initializer=tf.keras.initializers.glorot_normal()
                )
            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class Decoder(object):
    def __init__(self, x_dim, noise_z_std=0.0, z_dim=[256,64], act_out=tf.nn.tanh, act_at=None):
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.name = 'ae/dec_net'
        self.noise_z_std = noise_z_std
        self.layers_dim = self.z_dim[::-1] + [x_dim]
        self.act_out = act_out
        self.act_at = act_at

    def __call__(self, z, reuse=True, c=None):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = z
            fc = tf.keras.layers.GaussianNoise(self.noise_z_std)(fc)
            if c != None:
                fc = tf.concat([fc, z], axis=1)
            for idx,out_dim in enumerate(self.layers_dim):
                if idx == len(self.layers_dim)-1:
                    act_fun = self.act_out
                elif idx == 0:
                    act_fun = self.act_at
                else:
                    act_fun = tf.nn.leaky_relu
                fc = layers.dense(
                    fc, out_dim,
                    activation=act_fun,
                    kernel_initializer=tf.keras.initializers.glorot_normal()
                )
            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class FreyConvEncoder(object):
    def __init__(self, num_at, z_dim=None):
        self.name = 'frey/convae/enc_net'
        self.nfilt = 32
        self.k = 4
        self.s = 2
        self.num_at = num_at

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = x
            fc = tf.reshape(fc, shape=[-1, 28, 20, 1])
            fc = layers.conv2d(fc, filters=self.nfilt, kernel_initializer=tf.keras.initializers.glorot_normal(), kernel_size=self.k,
                padding='same', strides=[self.s,self.s], activation=None, name='h1')
            fc = bn(fc, 'h1')
            fc = tf.nn.leaky_relu(fc)
            fc = layers.conv2d(fc, filters=self.nfilt*2, kernel_initializer=tf.keras.initializers.glorot_normal(), kernel_size=self.k,
                padding='same', strides=[self.s,self.s], activation=None, name='h2')
            fc = bn(fc, 'h2')
            fc = tf.nn.leaky_relu(fc)
            fc = layers.flatten(fc)
            fc = layers.dense(
                fc, self.num_at-1,
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_normal()
            )
            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class FreyConvDecoder(object):
    def __init__(self, x_dim, noise_z_std=0.0, act_out=tf.nn.tanh):
        self.x_dim = x_dim
        self.name = 'frey/convae/dec_net'
        self.noise_z_std = noise_z_std
        self.act_out = act_out  
        self.x_dim = x_dim
        self.x_shape = [28, 20, 1]
        self.nfilt = 32
        self.k = 4
        self.s = 2

    def __call__(self, z, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = z
            fc = tf.keras.layers.GaussianNoise(self.noise_z_std)(fc)
            fc = layers.dense(
                fc, 7*5*self.nfilt*2,
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_normal()
            )
            fc = tf.reshape(fc, [tf.shape(z)[0], 7, 5, self.nfilt*2])
            fc = tf.layers.conv2d_transpose(fc, filters=self.nfilt, kernel_initializer=tf.keras.initializers.glorot_normal(), kernel_size=self.k,
                padding='same', strides=[self.s,self.s], activation=None, name='d1')
            #fc = bn(fc, 'd11')
            fc = tf.nn.leaky_relu(fc)
            fc = tf.layers.conv2d_transpose(fc, filters=1, kernel_initializer=tf.keras.initializers.glorot_normal(), kernel_size=self.k,
                padding='same', strides=[self.s,self.s], activation=self.act_out, name='recon')
            fc = tf.reshape(fc, shape=[-1, self.x_dim])
            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class CelebA28ConvEncoder(object):
    def __init__(self, num_at, act_at=None):
        self.name = 'ae/enc_net'
        self.nfilt = 64
        self.k = 4
        self.s = 2
        self.num_at = num_at
        self.act_at = act_at

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = x
            fc = tf.reshape(fc, shape=[-1, 28, 28, 3])
            fc = layers.conv2d(fc, filters=self.nfilt, kernel_initializer=tf.keras.initializers.glorot_normal(), kernel_size=self.k,
                padding='same', strides=[self.s,self.s], activation=None, name='h1')
            fc = bn(fc, 'eb1')
            fc = tf.nn.leaky_relu(fc)
            fc = layers.conv2d(fc, filters=self.nfilt*2, kernel_initializer=tf.keras.initializers.glorot_normal(), kernel_size=self.k,
                padding='same', strides=[self.s,self.s], activation=None, name='h2')
            fc = bn(fc, 'eb2')
            fc = tf.nn.leaky_relu(fc)
            fc = layers.flatten(fc)
            fc = layers.dense(
                fc, self.num_at-1,
                activation=self.act_at,
                kernel_initializer=tf.keras.initializers.glorot_normal()
            )
            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class CelebA28ConvDecoder(object):
    def __init__(self, noise_z_std=0.0, act_out=tf.nn.tanh, act_at=None):
        self.name = 'ae/dec_net'
        self.noise_z_std = noise_z_std
        self.act_out = act_out
        self.nfilt = 64
        self.k = 4
        self.s = 2
        self.x_dim = 28*28*3
        self.act_at = act_at

    def __call__(self, z, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = z
            fc = tf.keras.layers.GaussianNoise(self.noise_z_std)(fc)
            fc = layers.dense(
                fc, 7*7*self.nfilt*2,
                activation=self.act_at,
                kernel_initializer=tf.keras.initializers.glorot_normal()
            )
            fc = tf.reshape(fc, [tf.shape(z)[0], 7, 7, self.nfilt*2])
            fc = tf.layers.conv2d_transpose(fc, filters=self.nfilt, kernel_initializer=tf.keras.initializers.glorot_normal(), kernel_size=self.k,
                padding='same', strides=[self.s,self.s], activation=None, name='d1')
            #fc = bn(fc, 'db1')
            fc = tf.nn.leaky_relu(fc)
            fc = tf.layers.conv2d_transpose(fc, filters=3, kernel_initializer=tf.keras.initializers.glorot_normal(), kernel_size=self.k,
                padding='same', strides=[self.s,self.s], activation=None, name='recon')
            #fc = bn(fc, 'db2')
            if self.act_out != None:
                fc = self.act_out(fc)
            fc = tf.reshape(fc, shape=[-1, self.x_dim])
            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class CelebA56ConvEncoder(object):
    def __init__(self, num_at, act_at=None, nfilt=16):
        self.name = 'ae/enc_net'
        self.nfilt = nfilt
        self.k = 4
        self.s = 2
        self.num_at = num_at
        self.act_at = act_at

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = x
            fc = tf.reshape(fc, shape=[-1, 56, 56, 3])
            fc = layers.conv2d(fc, filters=self.nfilt, kernel_initializer=tf.keras.initializers.glorot_normal(), kernel_size=self.k,
                padding='same', strides=[self.s,self.s], activation=None, name='h1')
            #fc = bn(fc, 'eb1')
            fc = tf.nn.leaky_relu(fc)
            fc = layers.conv2d(fc, filters=self.nfilt*2, kernel_initializer=tf.keras.initializers.glorot_normal(), kernel_size=self.k,
                padding='same', strides=[self.s,self.s], activation=None, name='h2')
            #fc = bn(fc, 'eb2')
            fc = tf.nn.leaky_relu(fc)
            fc = layers.conv2d(fc, filters=self.nfilt*4, kernel_initializer=tf.keras.initializers.glorot_normal(), kernel_size=self.k,
                padding='same', strides=[self.s,self.s], activation=None, name='h3')
            #fc = bn(fc, 'eb3')
            fc = tf.nn.leaky_relu(fc)
            fc = layers.flatten(fc)
            fc = layers.dense(
                fc, self.num_at-1,
                activation=self.act_at,
                kernel_initializer=tf.keras.initializers.glorot_normal()
            )
            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class CelebA56ConvDecoder(object):
    def __init__(self, noise_z_std=0.0, act_out=tf.nn.tanh, act_at=None, nfilt=16):
        self.name = 'ae/dec_net'
        self.noise_z_std = noise_z_std
        self.act_out = act_out
        self.nfilt = nfilt
        self.k = 4
        self.s = 2
        self.x_dim = 56*56*3
        self.act_at = act_at

    def __call__(self, z, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            fc = z
            fc = tf.keras.layers.GaussianNoise(self.noise_z_std)(fc)
            fc = layers.dense(
                fc, 7*7*self.nfilt*4,
                activation=self.act_at,
                kernel_initializer=tf.keras.initializers.glorot_normal()
            )
            fc = tf.reshape(fc, [-1, 7, 7, self.nfilt*4])
            fc = tf.layers.conv2d_transpose(fc, filters=self.nfilt*2, kernel_initializer=tf.keras.initializers.glorot_normal(), kernel_size=self.k,
                padding='same', strides=[self.s,self.s], activation=None, name='d1')
            #fc = bn(fc, 'db1')
            fc = tf.nn.leaky_relu(fc)
            fc = tf.layers.conv2d_transpose(fc, filters=self.nfilt, kernel_initializer=tf.keras.initializers.glorot_normal(), kernel_size=self.k,
                padding='same', strides=[self.s,self.s], activation=None, name='d2')
            #fc = bn(fc, 'db2')
            fc = tf.nn.leaky_relu(fc)
            fc = tf.layers.conv2d_transpose(fc, filters=3, kernel_initializer=tf.keras.initializers.glorot_normal(), kernel_size=self.k,
                padding='same', strides=[self.s,self.s], activation=None, name='recon')
            #fc = bn(fc, 'db3')
            if self.act_out != None:
                fc = self.act_out(fc)
            fc = tf.reshape(fc, shape=[-1, self.x_dim])
            return fc

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]





