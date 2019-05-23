# Submitted to 33rd Conference on Neural Information Processing Systems (NeurIPS 2019). Do not distribute
import time
import tensorflow as tf
import numpy as np
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class AAnet(object):
    """Main Class for running AAnet.

    Usage:

    ```
    ##############
    # MODEL PARAMS
    ##############

    noise_z_std = 0.5 # This is the noise added to the latent space during training
    z_dim = [1024, 512, 256, 128] # Layer dimensions for encoder and decoder (reversed)
    act_out = tf.nn.tanh # Activation for the final layer
    input_dim = data.shape[1]
    n_batches = 40000 # Batches for training

    # Create encoder
    enc_net = network.Encoder(num_at=n_archetypes, z_dim=z_dim)
    # Create decoder
    dec_net = network.Decoder(x_dim=input_dim, noise_z_std=noise_z_std, z_dim=z_dim, act_out=act_out)
    # Assemble model
    model = AAnet.AAnet(enc_net, dec_net)

    ##########
    # TRAINING
    ##########

    model.train(data, batch_size=256, num_batches=n_batches)

    ###################
    # GETTING OUTPUT
    ###################

    # Get convex archetypal mixtures for input data. Output is data x archetypes. Rows sum to 1.
    # This is the 'archetypal space'
    new_archetypal_coords = model.data2at(data)
    # Get archetypes in the feature space
    new_archetypes = model.get_ats_x()

    ```

    """
    def __init__(self, enc_net, dec_net, gamma_mse=1.0, gamma_nn=1.0, gamma_convex=1.0, learning_rate=1e-3, rseed=42, gpu_mem=0.4):

        tf.reset_default_graph()

        self.gamma_mse = gamma_mse
        self.gamma_nn = gamma_nn
        self.gamma_convex = gamma_convex
        self.enc_net = enc_net
        self.dec_net = dec_net
        self.num_at = self.enc_net.num_at
        self.x_dim = self.dec_net.x_dim
        self.z_dim = self.num_at-1
        self.learning_rate = learning_rate
        self.rseed = rseed
        self.is_training = tf.keras.backend.learning_phase()
        self.gpu_mem = gpu_mem

        # tensors
        self.x = tf.placeholder(tf.float32, [None, self.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        # network
        self.z_ = self.enc_net(self.x, reuse=False) # encoder
        self.x_ = self.dec_net(self.z_, reuse=False) # decoder
        self.z_01 = (self.z_ + 1) / 2
        self.z_01_full = tf.concat([self.z_01, tf.reshape(1-tf.reduce_sum(self.z_01, axis=1), (-1,1))], 1) # add virtual archetype
        self.x__ = self.dec_net(self.z) # decoding from z

        # loss
        self.mse_loss = tf.reduce_mean(tf.square(self.x - self.x_))
        self.convex_loss = tf.reduce_mean(tf.maximum(tf.reduce_sum(self.z_01, axis=1) - 1, 0))
        self.nn_loss = -1 * tf.reduce_mean(tf.reduce_sum(tf.minimum(self.z_01, 0), axis=1))
        mu, sigma = tf.nn.moments(self.z_01_full, axes=[0])

        self.loss = self.gamma_mse * self.mse_loss
        self.loss += self.gamma_convex * self.convex_loss
        self.loss += self.gamma_nn * self.nn_loss

        # optimizer
        self.ae_adam = None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.ae_adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9)\
                .minimize(self.loss, var_list=self.enc_net.vars+self.dec_net.vars)

        # set gpu config
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        conf.gpu_options.per_process_gpu_memory_fraction = self.gpu_mem
        #conf.log_device_placement=True
        self.sess = tf.Session(config=conf)
        np.random.seed(self.rseed)
        tf.set_random_seed(self.rseed)
        tf.logging.set_verbosity('ERROR')
        self.init_vars()

    def init_vars(self):
        self.sess.run(tf.global_variables_initializer())

    def sample_x(self, data, n):
        '''Get some of the input data'''
        idx = np.random.randint(data.shape[0], size=n)
        return data[idx,:], idx

    def data2at(self, data):
        '''Runs data through the encoder to recover the correct archetypes.
        Adds in the kth archetype as 1 - sum(archetypes[:k-1])'''
        return self.sess.run(self.z_01_full, feed_dict={self.x: data})

    def data2z(self, data):
        '''Runs data through the encoder to get data in the latent space. Note,
        the kth archetype is not returned'''
        return self.sess.run(self.z_, feed_dict={self.x: data})

    def z2data(self, z):
        '''Runs points from the latent space, z, through the decoder and returns
        those points in the feature space.'''
        return self.sess.run(self.x__, feed_dict={self.z: z})

    def at2data(self, z):
        '''Takes points as a mixture of archetypes and decodes
        back to the feature space.'''
        z = (z * 2) - 1
        z = z[:,:-1]
        return self.z2data(z)

    def get_ats(self):
        '''Returns the archetypes in the latent space (i.e.) single activations
        the nodes + 1'''
        return np.eye(self.num_at)

    def get_ats_x(self):
        '''Returns the archetypes in the feature space'''
        return self.at2data(self.get_ats())

    def plot_at_mds(self, data, c=None):
        '''Method for visualizing the latent archetypal space. Algorithm is:
        1. MDS is performed on the archetypes in the feature space to provide
           a frame for the data.
        2. Points represented as a mixture of archetypes are interpolated between
           the calculated coordinates for each at
        This could also be achieved by running MDS on the latent space + archetypes. This
        interpolation approach is faster and yeilds similar results.'''

        data_at = self.data2at(data)
        embedding = MDS(n_components=2)
        Y_mds_ats = embedding.fit_transform(self.get_ats_x())
        Y_mds_data = data_at @ Y_mds_ats

        fig, ax = plt.subplots(1, figsize=(8,6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('MDS1')
        ax.set_ylabel('MDS2')
        ax.scatter(Y_mds_data[:,0], Y_mds_data[:,1], s=1, alpha=0.5, c=c)
        ax.scatter(Y_mds_ats[:,0], Y_mds_ats[:,1], s=200, c='r', zorder=3)
        for i in range(Y_mds_ats.shape[0]):
            ax.text(Y_mds_ats[i,0], Y_mds_ats[i,1], i+1, horizontalalignment='center', verticalalignment='center', fontdict={'color': 'white','size':10,'weight':'bold'}, zorder=4)
        return ax

    def plot_pca_data_ats(self, data, c=None):
        '''Similar to above, but with PCA'''
        pca = PCA(n_components=2)
        Z_at = np.eye(self.num_at)
        Z_at_m11 = (Z_at * 2) - 1
        Z_at_m11 = Z_at_m11[:,:-1]
        at_recon = self.sess.run(self.x__, feed_dict={self.z: Z_at_m11})
        Y_pca_z = pca.fit_transform(data)
        Y_pca = pca.transform(at_recon)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(Y_pca_z[:,0], Y_pca_z[:,1], s=1, alpha=0.5, c=c)
        plt.scatter(Y_pca[:,0], Y_pca[:,1], s=200, c='r')
        for i in range(Y_pca.shape[0]):
            plt.text(Y_pca[i,0], Y_pca[i,1], i+1, horizontalalignment='center', verticalalignment='center', fontdict={'color': 'white','size':10,'weight':'bold'})

    def plot_pca_ats_data(self, data, c=None):
        pca = PCA(n_components=2)
        Z_at = np.eye(self.num_at)
        Z_at_m11 = (Z_at * 2) - 1
        Z_at_m11 = Z_at_m11[:,:-1]
        at_recon = self.sess.run(self.x__, feed_dict={self.z: Z_at_m11})
        Y_pca = pca.fit_transform(at_recon)
        Y_pca_z = pca.transform(data)
        fig, ax = plt.subplots(1, figsize=(8,6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.scatter(Y_pca_z[:,0], Y_pca_z[:,1], s=1, alpha=0.5, c=c)
        ax.scatter(Y_pca[:,0], Y_pca[:,1], s=200, c='r')
        for i in range(Y_pca.shape[0]):
            ax.text(Y_pca[i,0], Y_pca[i,1], i+1, horizontalalignment='center', verticalalignment='center', fontdict={'color': 'white','size':10,'weight':'bold'})

    def plot_pca_data_recon_ats(self, data, c=None):
        pca = PCA(n_components=2)
        Z_at = np.eye(self.num_at)
        Z_at_m11 = (Z_at * 2) - 1
        Z_at_m11 = Z_at_m11[:,:-1]
        data_recon = self.sess.run(self.x_, feed_dict={self.x: data})
        at_recon = self.sess.run(self.x__, feed_dict={self.z: Z_at_m11})
        Y_pca_z = pca.fit_transform(data_recon)
        Y_pca = pca.transform(at_recon)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(Y_pca_z[:,0], Y_pca_z[:,1], s=1, alpha=0.5, c=c)
        plt.scatter(Y_pca[:,0], Y_pca[:,1], s=200, c='r')
        for i in range(Y_pca.shape[0]):
            plt.text(Y_pca[i,0], Y_pca[i,1], i+1, horizontalalignment='center', verticalalignment='center', fontdict={'color': 'white','size':10,'weight':'bold'})

    def plot_pca_ats_data_recon(self, data, c=None):
        pca = PCA(n_components=2)
        Z_at = np.eye(self.num_at)
        Z_at_m11 = (Z_at * 2) - 1
        Z_at_m11 = Z_at_m11[:,:-1]
        data_recon = self.sess.run(self.x_, feed_dict={self.x: data})
        at_recon = self.sess.run(self.x__, feed_dict={self.z: Z_at_m11})
        Y_pca = pca.fit_transform(at_recon)
        Y_pca_z = pca.transform(data_recon)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(Y_pca_z[:,0], Y_pca_z[:,1], s=1, alpha=0.5, c=c)
        plt.scatter(Y_pca[:,0], Y_pca[:,1], s=200, c='r')
        for i in range(Y_pca.shape[0]):
            plt.text(Y_pca[i,0], Y_pca[i,1], i+1, horizontalalignment='center', verticalalignment='center', fontdict={'color': 'white','size':10,'weight':'bold'})

    def plot_at_pca_single(self, data, c=None):
        pca = PCA(n_components=2)
        Z_at = np.eye(self.num_at)
        samples_Z = self.sess.run(self.z_01_full, feed_dict={self.x: data})
        Y_pca_z = pca.fit_transform(samples_Z)
        Y_pca = pca.transform(Z_at)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(Y_pca_z[:,0], Y_pca_z[:,1], s=1, alpha=0.5, c=c)
        plt.scatter(Y_pca[:,0], Y_pca[:,1], s=200, c='r')
        for i in range(Y_pca.shape[0]):
            plt.text(Y_pca[i,0], Y_pca[i,1], i+1, horizontalalignment='center', verticalalignment='center', fontdict={'color': 'white','size':10,'weight':'bold'})

    def plot_at(self, data, n_plot=None, c=None):
        if n_plot == None:
            n_plot = self.num_at
        Z_at = np.eye(self.num_at)
        samples_Z = self.sess.run(self.z_01_full, feed_dict={self.x: data})
        f, axarr = plt.subplots(n_plot, n_plot, figsize=(4*n_plot, 3*n_plot))
        for i in range(n_plot):
            for j in range(n_plot):
                axarr[i, j].scatter(samples_Z[:,i], samples_Z[:,j], s=1, alpha=0.5, c=c)
                axarr[i, j].scatter(Z_at[:,i], Z_at[:,j], s=20, c='r')

    def plot_at_pca(self, data, n_plot=None, c=None):
        if n_plot == None:
            n_plot = self.num_at
        Z_at = np.eye(self.num_at)
        samples_Z = self.sess.run(self.z_01_full, feed_dict={self.x: data})
        pca = PCA(n_components=n_plot)
        Y = pca.fit_transform(samples_Z)
        Y_at = pca.transform(Z_at)
        f, axarr = plt.subplots(n_plot, n_plot, figsize=(4*n_plot, 3*n_plot))
        for i in range(n_plot):
            for j in range(n_plot):
                axarr[i, j].scatter(Y[:,i], Y[:,j], s=.5, alpha=0.5, c=c)
                axarr[i, j].scatter(Y_at[:,i], Y_at[:,j], s=20, c='r')


    def at_scan(self, n, nplot=99999):
        '''Interpolate between archetypes'''

        up = np.linspace(0, 1, num=n)
        dn = np.linspace(1, 0, num=n)
        q = np.zeros([0,self.num_at])
        for i in range(np.min([nplot, self.num_at])):
            for j in range(i):
                z = np.zeros([n,self.num_at])
                z[:,i] = up
                z[:,j] = dn
                q = np.concatenate([q,z])
        return q

    def sample_at_uniform(self, n):
        ''' Method for uniformly sampling from a simplex'''
        u = np.random.uniform(0,1,[n,self.num_at])
        e = -np.log(u)
        x = e / np.sum(e, axis=1, keepdims=True)
        return x

    def sample_boundary_uniform(self, n):
        '''Sample the points along the boundary of a simplex'''
        x_all = []
        for i in range(num_at):
            x = self.sample_at_uniform(self.num_at-1, n)
            x = np.insert(x, i, 0, axis=1)
            x_all.append(x)
        return np.concatenate(x_all, axis=0)

    def sample_z_uniform(self, n):
        '''Sample the latent space uniformly'''
        z = self.sample_at_uniform(n)
        z = (z * 2) - 1
        z = z[:,:-1]
        return z

    def dist_to_closest_at(self, data):
        '''Calculate distance to the closest AT for a data point'''

        samples_Z = self.sess.run(self.z_01_full, feed_dict={self.x: data})
        d = tf.reduce_max(samples_Z, axis=1).eval(session=self.sess)
        d = 1 - d
        d = d / (1 - 1/self.num_at)
        return d

    def dist_to_hull(self, data):
        '''Calculate distance to the outside of the simplex for a data point'''

        samples_Z = self.sess.run(self.z_01_full, feed_dict={self.x: data})
        return np.maximum(np.max(samples_Z-1, axis=1),0) + np.maximum(np.max(-samples_Z, axis=1),0)

    def close_sess(self):
        self.sess.close()

    def train(self, data, batch_size=128, num_batches=20000, verbose=True):
        '''Train the model'''

        start_time = time.time()

        for t in range(0, num_batches):
            bx, idx = self.sample_x(data, batch_size)
            self.sess.run(self.ae_adam, feed_dict={self.x: bx, self.is_training: 1})

            if verbose and (t % 500 == 0 or t+1 == num_batches):
                bx, idx = self.sample_x(data, np.min([data.shape[0],batch_size*10]))
                loss = self.sess.run(
                    self.loss, feed_dict={self.x: bx}
                )
                #loss = self.compute_loss(data) # compute loss on all data
                print('Iter [%8d] Time [%5.4f] loss [%.4f]' %
                            (t, time.time() - start_time, loss))

        if verbose:
            print('done.')

    def compute_loss(self, data):
        return self.sess.run(self.loss, feed_dict={self.x: data})

    def compute_mse_loss(self, data):
        return self.sess.run(self.mse_loss, feed_dict={self.x: data})
