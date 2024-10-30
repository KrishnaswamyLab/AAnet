import torch
from torch import nn, optim
import numpy as np
import sklearn
from torch.nn import functional as F

from . import BaseAAnet



class AAnet_vanilla(BaseAAnet):
    def __init__(
        self,
        input_shape,
        n_archetypes=4,
        noise=0,
        layer_widths=[128, 128],
        activation_out="tanh",
        simplex_scale=1,
        device=None,
        diffusion_extrema=None,
        archetypal_weight=1,
        **kwargs
    ):
        super().__init__()
        self.input_shape = input_shape
        self.n_archetypes = n_archetypes
        self.noise = noise
        self.layer_widths = layer_widths
        self.activation_out = activation_out
        self.simplex_scale = simplex_scale
        self.diffusion_extrema = diffusion_extrema
        self.archetypal_weight = archetypal_weight
        
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

        self.archetypal_simplex = self.get_n_simplex(self.n_archetypes, scale=self.simplex_scale)

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

    def forward(self, input) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
            Returns:
                recons: reconstructed data
                input: original data
                archetypal_embedding: archetypal embedding of the data
        '''
        # Encode data
        activation = self.encode(input)

        # Save archetypal_embedding
        archetypal_embedding = activation.clone()

        # Add noise
        if self.noise > 0:
            activation += torch.normal(mean=0., std=self.noise, size=activation.shape)

        # Decode embedding
        return self.decode(activation), input, archetypal_embedding

    def loss_function(self,
                      recons,
                      input,
                      archetypal_embedding,
                      mu=None,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons_loss = F.mse_loss(recons, input)

        archetypal_loss = self.calc_archetypal_loss(archetypal_embedding)

        loss = recons_loss + self.archetypal_weight * archetypal_loss

        # TODO:KL divergence loss
        kld_loss = 0

        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD': kld_loss, 'Archetypal_Loss':archetypal_loss}
