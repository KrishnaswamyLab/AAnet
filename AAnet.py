
import time
import tensorflow as tf
import numpy as np
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class AAnet(object):
    def __init__(self, enc_net, dec_net, gamma_mse=1.0, gamma_nn=1.0, gamma_convex=1.0, learning_rate=1e-3, rseed=42):

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
        self.convex_loss = tf.reduce_mean(tf.maximum(tf.reduce_sum(self.z_01, axis=1) - 1, 0)) # for P-1
        self.nn_loss = -1 * tf.reduce_mean(tf.reduce_sum(tf.minimum(self.z_01, 0), axis=1)) # for P-1
        self.loss = self.gamma_mse * self.mse_loss
        self.loss += self.gamma_convex * self.convex_loss
        self.loss += self.gamma_nn * self.nn_loss

        # optimizer
        self.ae_adam = None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.ae_adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5, beta2=0.9)\
                .minimize(self.loss, var_list=self.enc_net.vars+self.dec_net.vars)
        
        conf = tf.ConfigProto()                                                                                                                                                                                                                  
        conf.gpu_options.allow_growth=True
        conf.gpu_options.per_process_gpu_memory_fraction=0.4
        #conf.log_device_placement=True
        self.sess = tf.Session(config=conf)
        np.random.seed(self.rseed)
        tf.set_random_seed(self.rseed)
        tf.logging.set_verbosity('ERROR')
        self.init_vars()

    def init_vars(self):
        self.sess.run(tf.global_variables_initializer())

    def sample_x(self, data, n):
        idx = np.random.randint(data.shape[0], size=n)
        return data[idx,:]

    def data2at(self, data):
        return self.sess.run(self.z_01_full, feed_dict={self.x: data})

    def data2z(self, data):
        return self.sess.run(self.z_, feed_dict={self.x: data})

    def z2data(self, z):
        return self.sess.run(self.x__, feed_dict={self.z: z})

    def at2data(self, z):
        z = (z * 2) - 1
        z = z[:,:-1]
        return self.z2data(z)

    def get_ats(self):
        return np.eye(self.num_at)

    def get_ats_x(self):
        Z_at = self.get_ats()
        Z_at_m11 = (Z_at * 2) - 1
        Z_at_m11 = Z_at_m11[:,:-1]
        at_recon = self.sess.run(self.x__, feed_dict={self.z: Z_at_m11})
        return at_recon

    def plot_at_mds(self, data):
        Z_at = np.eye(self.num_at)
        Z_at_m11 = (Z_at * 2) - 1
        Z_at_m11 = Z_at_m11[:,:-1]
        at_recon = self.sess.run(self.x__, feed_dict={self.z: Z_at_m11})
        samples_Z = self.sess.run(self.z_01_full, feed_dict={self.x: data})
        mds = MDS(n_components=2)
        Y_mds = mds.fit_transform(at_recon)
        Y_mds_z = samples_Z @ Y_mds
        nbrs = NearestNeighbors(n_neighbors=3).fit(Y_mds)
        _, indices = nbrs.kneighbors(Y_mds)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(Y_mds_z[:,0], Y_mds_z[:,1], s=1, alpha=0.5)
        for i in range(indices.shape[0]):
            plt.plot(Y_mds[indices[i,[0,1]],0], Y_mds[indices[i,[0,1]],1], 'grey', linewidth=0.5)
            plt.plot(Y_mds[indices[i,[0,2]],0], Y_mds[indices[i,[0,2]],1], 'grey', linewidth=0.5)
        plt.scatter(Y_mds[:,0], Y_mds[:,1], s=200, c='r', zorder=3)
        for i in range(Y_mds.shape[0]):
            plt.text(Y_mds[i,0], Y_mds[i,1], i+1, horizontalalignment='center', verticalalignment='center', fontdict={'color': 'white','size':10,'weight':'bold'}, zorder=4)

    def plot_pca_data_ats(self, data):
        pca = PCA(n_components=2)
        Z_at = np.eye(self.num_at)
        Z_at_m11 = (Z_at * 2) - 1
        Z_at_m11 = Z_at_m11[:,:-1]
        at_recon = self.sess.run(self.x__, feed_dict={self.z: Z_at_m11})
        Y_pca_z = pca.fit_transform(data)
        Y_pca = pca.transform(at_recon)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(Y_pca_z[:,0], Y_pca_z[:,1], s=1, alpha=0.5)
        plt.scatter(Y_pca[:,0], Y_pca[:,1], s=200, c='r')
        for i in range(Y_pca.shape[0]):
            plt.text(Y_pca[i,0], Y_pca[i,1], i+1, horizontalalignment='center', verticalalignment='center', fontdict={'color': 'white','size':10,'weight':'bold'})

    def plot_pca_ats_data(self, data):
        pca = PCA(n_components=2)
        Z_at = np.eye(self.num_at)
        Z_at_m11 = (Z_at * 2) - 1
        Z_at_m11 = Z_at_m11[:,:-1]
        at_recon = self.sess.run(self.x__, feed_dict={self.z: Z_at_m11})
        Y_pca = pca.fit_transform(at_recon)
        Y_pca_z = pca.transform(data)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(Y_pca_z[:,0], Y_pca_z[:,1], s=1, alpha=0.5)
        plt.scatter(Y_pca[:,0], Y_pca[:,1], s=200, c='r')
        for i in range(Y_pca.shape[0]):
            plt.text(Y_pca[i,0], Y_pca[i,1], i+1, horizontalalignment='center', verticalalignment='center', fontdict={'color': 'white','size':10,'weight':'bold'})

    def plot_pca_data_recon_ats(self, data):
        pca = PCA(n_components=2)
        Z_at = np.eye(self.num_at)
        Z_at_m11 = (Z_at * 2) - 1
        Z_at_m11 = Z_at_m11[:,:-1]
        data_recon = self.sess.run(self.x_, feed_dict={self.x: data})
        at_recon = self.sess.run(self.x__, feed_dict={self.z: Z_at_m11})
        Y_pca_z = pca.fit_transform(data_recon)
        Y_pca = pca.transform(at_recon)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(Y_pca_z[:,0], Y_pca_z[:,1], s=1, alpha=0.5)
        plt.scatter(Y_pca[:,0], Y_pca[:,1], s=200, c='r')
        for i in range(Y_pca.shape[0]):
            plt.text(Y_pca[i,0], Y_pca[i,1], i+1, horizontalalignment='center', verticalalignment='center', fontdict={'color': 'white','size':10,'weight':'bold'})

    def plot_pca_ats_data_recon(self, data):
        pca = PCA(n_components=2)
        Z_at = np.eye(self.num_at)
        Z_at_m11 = (Z_at * 2) - 1
        Z_at_m11 = Z_at_m11[:,:-1]
        data_recon = self.sess.run(self.x_, feed_dict={self.x: data})
        at_recon = self.sess.run(self.x__, feed_dict={self.z: Z_at_m11})
        Y_pca = pca.fit_transform(at_recon)
        Y_pca_z = pca.transform(data_recon)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(Y_pca_z[:,0], Y_pca_z[:,1], s=1, alpha=0.5)
        plt.scatter(Y_pca[:,0], Y_pca[:,1], s=200, c='r')
        for i in range(Y_pca.shape[0]):
            plt.text(Y_pca[i,0], Y_pca[i,1], i+1, horizontalalignment='center', verticalalignment='center', fontdict={'color': 'white','size':10,'weight':'bold'})

    def plot_at_pca_single(self, data):
        pca = PCA(n_components=2)
        Z_at = np.eye(self.num_at)
        samples_Z = self.sess.run(self.z_01_full, feed_dict={self.x: data})
        Y_pca_z = pca.fit_transform(samples_Z)
        Y_pca = pca.transform(Z_at)
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(Y_pca_z[:,0], Y_pca_z[:,1], s=1, alpha=0.5)
        plt.scatter(Y_pca[:,0], Y_pca[:,1], s=200, c='r')
        for i in range(Y_pca.shape[0]):
            plt.text(Y_pca[i,0], Y_pca[i,1], i+1, horizontalalignment='center', verticalalignment='center', fontdict={'color': 'white','size':10,'weight':'bold'})

    def plot_at(self, data, n_plot=None):
        if n_plot == None:
            n_plot = self.num_at
        Z_at = np.eye(self.num_at)
        samples_Z = self.sess.run(self.z_01_full, feed_dict={self.x: data})
        f, axarr = plt.subplots(n_plot, n_plot, figsize=(4*n_plot, 3*n_plot))
        for i in range(n_plot):
            for j in range(n_plot):   
                axarr[i, j].scatter(samples_Z[:,i], samples_Z[:,j], s=1, alpha=0.5)
                axarr[i, j].scatter(Z_at[:,i], Z_at[:,j], s=20, c='r')

    def plot_at_pca(self, data, n_plot=None):
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
                axarr[i, j].scatter(Y[:,i], Y[:,j], s=1, alpha=0.5)
                axarr[i, j].scatter(Y_at[:,i], Y_at[:,j], s=20, c='r')


    def at_scan(self, n, nplot=99999):
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
        u = np.random.uniform(0,1,[n,self.num_at])
        e = -np.log(u)
        x = e / np.sum(e, axis=1, keepdims=True)
        return x

    def sample_z_uniform(self, n):
        z = self.sample_at_uniform(n)
        z = (z * 2) - 1
        z = z[:,:-1]
        return z

    def dist_to_closest_at(self, data):
        samples_Z = self.sess.run(self.z_01_full, feed_dict={self.x: data})
        d = tf.reduce_max(samples_Z, axis=1).eval(session=self.sess)
        d = 1 - d
        d = d / (1 - 1/self.num_at)
        return d

    def close_sess(self):
        self.sess.close()

    def train(self, data, batch_size=128, num_batches=20000, verbose=True):
        start_time = time.time()
        # todo: per epoch
        for t in range(0, num_batches):
            bx = self.sample_x(data, batch_size)
            self.sess.run(self.ae_adam, feed_dict={self.x: bx, self.is_training: 1})

            if verbose and (t % 500 == 0 or t+1 == num_batches):
                bx = self.sample_x(data, batch_size)
                loss = self.sess.run(
                    self.loss, feed_dict={self.x: bx}
                )
                print('Iter [%8d] Time [%5.4f] loss [%.4f]' %
                            (t, time.time() - start_time, loss))

            if verbose and t == num_batches-1:
                print('done.')

    def compute_loss(self, data):
        return self.sess.run(self.loss, feed_dict={self.x: data})

    def compute_mse_loss(self, data):
        return self.sess.run(self.mse_loss, feed_dict={self.x: data})

