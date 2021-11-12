from ..types_ import *
import torch
from torch import nn
from abc import abstractmethod
import numpy as np

class BaseAAnet(nn.Module):
    '''
    Base class for AAnet variants. Implements functions to calculate barycentric
    coordinates, translating between the archetypal space and the feature space, etc.
    '''
    def __init__(self) -> None:
        super(BaseAAnet, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError


    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


    def get_archetypes_latent(self):
        return torch.tensor(
            np.vstack(
                [torch.eye(self.n_archetypes - 1), np.zeros(n_archetypes-1)]
            )
        )

    def get_n_simplex(self, n=2, scale=1):
        '''
        Returns an n-simplex centered at the origin in the feature space
        '''
        nth = 1/(n-1)*(1-np.sqrt(n)) * np.ones(n-1)
        D = np.vstack([np.eye(n-1), nth]) * scale
        return torch.tensor(D - np.mean(D, axis=0), dtype=torch.float, device=self.device)

    def get_archetypes_data(self):
        '''Returns archetypes in the feature domain'''
        return self.decode(self.get_n_simplex(self.n_archetypes, self.simplex_scale))

    def euclidean_to_barycentric(self, X):
        '''
        Converts euclidean coordinates to barycentric coordinates wrt a regular simplex
        centered the origin scaled by `scale`.

        Requires `self.archetypal_simplex` to be fit.
        '''
        simplex = self.archetypal_simplex

        T = torch.zeros((X.shape[1], X.shape[1])).to(self.device)
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                T[i,j] = simplex[i,j] - simplex[-1,j]

        T_inv =  torch.inverse(T).type(torch.float).to(self.device)
        X_bary = torch.einsum('ij,bj->bi', T_inv, X - simplex[-1]).to(self.device)
        X_bary = torch.cat([X_bary, (1-torch.sum(X_bary, axis=1, keepdim=True))], axis=1).to(self.device)
        return X_bary

    def is_in_simplex(self, X_bary):
        '''Is True for points that are inside the simplex and False otherise'''
        all_non_negative = torch.sum(X_bary >= 0, axis=1) == X_bary.shape[1]
        all_convex = torch.sum(X_bary <= 1, axis=1) == X_bary.shape[1]
        return  all_non_negative & all_convex

    def dist_to_simplex(self, X_bary):
        '''
        Sums all negative values outside the simplex

        TODO: this results in lower loss values on the boundaries of the voronoi regions outside the simplex
        '''
        return torch.sum(
                 torch.where((X_bary < 0),
                   torch.abs(X_bary),
                   torch.zeros(X_bary.shape, dtype=torch.float).to(self.device)
                   ),
                 axis=1).to(self.device)

    def calc_archetypal_loss(self, archetypal_embedding):
        '''
        Returns MSE archetypal loss (sum of negative values inside the simplex)
        '''
        X_bary = self.euclidean_to_barycentric(archetypal_embedding)
        return torch.mean(self.dist_to_simplex(X_bary) ** 2)
    
    def calc_diffusion_extrema_loss(self, archetypal_embedding):
        '''
        Returns MSE diffusion extrema loss (minimize MSE between diffusion extrema and archetypes)
        Diffusion extrema are concatenated to the beginning of each batch as first n_archetypes samples
        '''
        X_bary = self.euclidean_to_barycentric(archetypal_embedding)
        return torch.mean((X_bary[:self.n_archetypes,:] - torch.eye(self.n_archetypes).to(self.device)) ** 2)
