import torch
from torch import nn, optim
import numpy as np
import sklearn

from .utils import get_n_simplex

class AAnet(nn.Module):
    def __init__(
        self,
        input_shape,
        n_archetypes=4,
        noise=0,
        layer_widths=[128, 128],
        activation_out="tanh",
        simplex_scale=1,
        device=None,
        **kwargs
    ):
        super().__init__()
        self.n_archetypes = n_archetypes
        self.noise = noise
        self.layer_widths = layer_widths
        self.activation_out = activation_out
        self.simplex_scale = simplex_scale
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.encoder_layers = []
        # Instantiate encoder
        for i, width in enumerate(layer_widths):
            if i == 0:
                # Instantiate first layer
                self.encoder_layers.append(
                    nn.Linear(in_features=input_shape, out_features=width)
                )
            else:
                # Middle layers
                self.encoder_layers.append(
                    nn.Linear(in_features=layer_widths[i - 1], out_features=width)
                )

        # Last encoder layer
        self.encoder_layers.append(
            nn.Linear(in_features=layer_widths[-1], out_features=n_archetypes - 1)
        )

        # Instantiate decoder
        self.decoder_layers = []
        decoder_widths = layer_widths[::-1]
        for i, width in enumerate(decoder_widths):
            if i == 0:
                # Instantiate first layer
                self.decoder_layers.append(
                    nn.Linear(in_features=n_archetypes - 1, out_features=width)
                )
            else:
                # Middle layers
                self.decoder_layers.append(
                    nn.Linear(in_features=decoder_widths[i - 1], out_features=width)
                )

        # Last decoder layer
        self.decoder_layers.append(
            nn.Linear(in_features=decoder_widths[-1], out_features=input_shape)
        )
        self.encoder_layers = nn.ModuleList(self.encoder_layers)
        self.decoder_layers = nn.ModuleList(self.decoder_layers)

        self.to(device)

    def encode(self, features):
        # Encode data
        for i, layer in enumerate(self.encoder_layers[:-1]):
            if i == 0:
                # First layer takes features
                activation = torch.relu(layer(features))
            else:
                activation = torch.relu(layer(activation))
        # Embedding layer no activation
        activation = self.encoder_layers[-1](activation)
        return activation

    def decode(self, activation):
        # Decode embedding
        for layer in self.decoder_layers[:-1]:
            activation = layer(activation)
            activation = torch.relu(activation)

        # Final layer no activation
        activation = self.decoder_layers[-1](activation)
        return activation

    def forward(self, features):
        # Encode data
        activation = self.encode(features)

        # Save archetypal_embedding
        archetypal_embedding = activation.clone()

        # Add noise
        if self.noise > 0:
            activation += torch.normal(mean=0., std=self.noise, size=activation.shape)

        # Decode embedding
        reconstructed = self.decode(activation)
        return reconstructed, archetypal_embedding

    def get_archetypes_latent(self):
        return torch.tensor(np.vstack([torch.eye(self.n_archetypes - 1),
                       np.zeros(n_archetypes-1)]))



    def get_n_simplex(self, n=2, scale=1):
        '''
        Returns an n-simplex centered at the origin
        '''


        nth = 1/(n-1)*(1-np.sqrt(n)) * np.ones(n-1)
        D = np.vstack([np.eye(n-1), nth]) * scale
        return torch.tensor(D - np.mean(D, axis=0)).to(self.device)

    def get_archetypes_data(self):
        return self.get_n_simplex(self.n_archetypes)


    def euclidean_to_barycentric(self, X):
        '''
        Converts euclidean coordinates to barycentric coordinates wrt a regular simplex
        centered the origin scaled by `scale`
        '''


        simplex = self.get_n_simplex(self.n_archetypes, scale=self.simplex_scale).to(self.device)

        T = torch.zeros((X.shape[1], X.shape[1])).to(self.device)
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                T[i,j] = simplex[i,j] - simplex[-1,j]

        T_inv =  torch.inverse(T).type(torch.double).to(self.device)
        X_bary = torch.einsum('ij,bj->bi', T_inv, X - simplex[-1]).to(self.device)
        X_bary = torch.cat([X_bary, (1-torch.sum(X_bary, axis=1, keepdim=True))], axis=1).to(self.device)
        return X_bary

    def is_in_simplex(self, X_bary):
        return (torch.sum(X_bary >= 0, axis=1) == 3) & (torch.sum(X_bary <= 1, axis=1) == 3)

    def dist_to_simplex(self, X_bary):
        '''
        Sums all negative values for
        '''
        return torch.sum(
                 torch.where((X_bary < 0),
                   torch.abs(X_bary),
                   torch.zeros(X_bary.shape, dtype=torch.double).to(self.device)
                   ),
                 axis=1).to(self.device)

    def calc_archetypal_loss(self, archetypal_embedding):
        '''
        Returns MSE archetypal loss (sum of negative values inside the simplex)
        '''
        X_bary = self.euclidean_to_barycentric(archetypal_embedding)
        return torch.mean(self.dist_to_simplex(X_bary) ** 2)
