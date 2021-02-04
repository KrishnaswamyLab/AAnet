from torch.utils.data import Dataset
import numpy as np
import torch

class SimplexSphereProjection(Dataset):
    def __init__(self, n_obs=1000, radius=1):
        super().__init__()
        self.n_obs = n_obs
        self.radius = radius
        self.data, self.vertices = self._generate_data_on_sphere(self.n_obs, self.radius)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]

    def _get_n_simplex(self, n=2, scale=1, device=None):
        '''
        Returns an n-simplex centered at the origin
        '''

        nth = 1/(n-1)*(1-np.sqrt(n)) * np.ones(n-1)
        D = np.vstack([np.eye(n-1), nth]) * scale
        return torch.tensor(D - np.mean(D, axis=0), device=device)


    def _generate_data_on_sphere(self, n_obs=1000, radius=1):
        '''
        Generates data on that lies on a sphere of `radius`.

        TODO: not convinced that this is the correct way to project points onto sphere
              also doesn't scale to more than 3 dimensions :(
        '''
        np.random.seed(0)

        n_dim = 2
        n_archetypes = 3

        # Get coordinates of archetypes in R2
        archetypes = self._get_n_simplex(n=n_archetypes).numpy()

        # Uniformly sample from a simplex
        data = np.random.uniform(0, 1, size=[n_obs, n_archetypes])
        data = -np.log(data)
        data = data / np.sum(data, axis=1, keepdims=True)

        # Project the points to appear within the archetypal coordiantes
        data_atype = data @ archetypes

        # 2D points on simplex to points on a sphere
        data_sphere  = self._latlong_to_xyz(data_atype, radius=radius)
        atype_sphere = self._latlong_to_xyz(archetypes, radius=radius)

        # Center the data
        data_center = data_sphere.mean(axis=0, keepdims=True)
        data_sphere -= data_center
        atype_sphere -= data_center

        return data_sphere, atype_sphere


    def _latlong_to_xyz(self, X, radius=1):
        '''
        Takes 2D input as lat/long coordinates and returns coordinates in R3
        '''
        longitude = X[:,0] / radius
        latitude = 2 * np.arctan(np.exp(X[:,1]/radius)) - np.pi/2
        x3 = radius * np.cos(latitude) * np.cos(longitude)
        y3 = radius * np.cos(latitude) * np.sin(longitude)
        z3 = radius * np.sin(latitude)
        #x3 = (x3 - R) / 10
        return np.vstack([x3, y3, z3]).T
