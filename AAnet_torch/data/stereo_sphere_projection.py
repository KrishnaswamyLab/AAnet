from torch.utils.data import Dataset
import numpy as np
import torch


class StereoSphereProjection(Dataset):
    def __init__(
        self,
        n_components=3,
        simplex_radius=1.0,
        n_obs=1000,
    ):
        super().__init__()
        self.n_components = n_components
        self.simplex_radius = simplex_radius
        self.n_obs = n_obs
        self.data = self._sample_from_stereo_sphere_simplex(
                        self.n_components,
                        self.simplex_radius,
                        self.n_obs,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

    def _barycentric_to_euclidean(self, X_barycentric, simplex_radius):
        '''
        Converts an M x (N + 1) array of barycentric coordinates into an M x N array
        of the corresponding Euclidean coordinates.
        '''

        n_components = X_barycentric.shape[1] - 1
        coord_transform = np.zeros((n_components, n_components + 1))

        # Construct the N x (N + 1) coordinate transformation matrix
        # See the 'Simplex' Wiki page for a detailed explanation
        np.fill_diagonal(coord_transform, 1 / np.sqrt(2))
        for i in range(n_components):
            for j in range(n_components):
                coord_transform[i, j] -= 1 / (n_components * np.sqrt(2)) * (1 + 1 / np.sqrt(n_components + 1))
        for i in range(n_components):
            coord_transform[i, n_components] = 1 / (2 * np.sqrt(n_components + 1))

        # Transform the simplex coordinates into points in R^n
        X_euclidean = simplex_radius * coord_transform @ X_barycentric.T
        X_euclidean = X_euclidean.T

        return X_euclidean

    def _sample_from_n_simplex(self, n_components, simplex_radius, n_obs):
        '''
        Samples Eucliean points uniformly from an N-dimensional simplex of a
        specified radius.
        '''

        exp_samples = np.random.exponential(size=(n_components, n_obs))
        X_barycentric = exp_samples / np.linalg.norm(exp_samples, ord=1, axis=0)
        self.X_barycentric = X_barycentric.T

        X_euclidean = self._barycentric_to_euclidean(self.X_barycentric, simplex_radius)

        return X_euclidean

    def _stereographic_inverse(self, X_hyperplane):
        '''
        Computes the inverse of a stereographic projection, mapping points in N-dimensional
        Euclidean space to points in (N + 1)-dimensional Euclidean space on the unit hypersphere.
        '''

        X_hyperplane = X_hyperplane.T
        X_sphere = np.zeros((X_hyperplane.shape[0] + 1, X_hyperplane.shape[1]))

        norms = np.linalg.norm(X_hyperplane, axis=0)
        X_sphere[0] = (norms**2 - 1) / (norms**2 + 1)
        for i in range(1, X_sphere.shape[0]):
            X_sphere[i] = 2 * X_hyperplane[i - 1] / (norms**2 + 1)

        return X_sphere.T

    def _sample_from_stereo_sphere_simplex(self, n_components, simplex_radius, n_obs):
        '''
        Samples non-uniformly from an N-dimensional simplex projected onto an
        N-dimensional sphere using a stereographic projection.

        (Note that a 'stereographic spherical simplex' is _not_ the same thing as a
        spherical simplex defined using the submanifold geometry the N-dimensional sphere
        inherets from (N + 1)-dimensional Euclidean space.)

        Parameters
        ----------
        n_components : int, optional, default: 3
            dimension of the sphere the points are sampled from (note that the
            sphere is a 2-dimensional space)

        simplex_radius : float, optional, default: 1
            radius of the simplex in Euclidean space, before it is projected onto
            the sphere

        n_obs : int, optional, default: 1000
            number of points sampled from the spherical simplex

        Returns
        -------
        A numpy array from shape (n_obs, n_components + 1)
        '''
        # Get data from hyperplane
        self.X_hyperplane = self._sample_from_n_simplex(n_components + 1, simplex_radius, n_obs)
        self.data = self._stereographic_inverse(self.X_hyperplane)
        # Append vertices to data object
        self.vertices_barycentric = np.eye(n_components+1)
        self.vertices = self._stereographic_inverse(
                            self._barycentric_to_euclidean(
                                self.vertices_barycentric, simplex_radius)
                           )
        return self.data
