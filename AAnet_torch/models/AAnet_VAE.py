import torch
from torch import nn, optim
import numpy as np
import sklearn

# VAE imports
from torch.nn import functional as F
from ..types_ import *
from . import BaseAAnet

class AAnet_VAE(BaseAAnet):
    def __init__(
        self,
        input_shape,
        n_archetypes=4,
        layer_widths=[128, 128],
        activation_out="tanh",
        simplex_scale=1,
        device=None,
        **kwargs
    ):
        super().__init__()

        self.input_shape = input_shape
        self.n_archetypes = n_archetypes
        self.layer_widths = layer_widths
        self.activation_out = activation_out
        self.simplex_scale = simplex_scale
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        layers = []
        # Instantiate encoder
        for i, width in enumerate(layer_widths):

            layers.append(
                nn.Sequential(
                    nn.Linear(in_features=input_shape, out_features=width),
                    nn.ReLU(),
                )
            )
            input_shape = width

        self.encoder = nn.Sequential(*layers)
        # Latent code layer
        self.fc_mu = nn.Linear(layer_widths[-1], n_archetypes - 1)
        self.fc_var = nn.Linear(layer_widths[-1], n_archetypes - 1)

        # Instantiate decoder
        self.decoder_input = nn.Linear(n_archetypes - 1, layer_widths[-1])

        layers = []
        layer_widths.reverse()
        for i in range(len(layer_widths) - 1):
            # Instantiate first layer
            layers.append(
                nn.Sequential(
                    nn.Linear(in_features=layer_widths[i], out_features=layer_widths[i+1]),
                    nn.ReLU(),
                    )
            )

        self.decoder = nn.Sequential(*layers)

        # Last decoder layer
        self.final_layer = nn.Sequential(
                            nn.Linear(layer_widths[-1],
                                      self.input_shape),
                            )

        self.archetypal_simplex = self.get_n_simplex(self.n_archetypes, scale=self.simplex_scale)

        self.to(device)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        #result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        archetypal_embedding = mu.clone()
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, archetypal_embedding, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        #TODO what's up with this M_N?
        kld_weight = 1 #kwargs['M_N'] # Account for the minibatch samples from the dataset


        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(torch.sum((1 - log_var.exp()) ** 2, dim = 1), dim = 0)

        archetypal_loss = self.calc_archetypal_loss(mu)

        loss = recons_loss + kld_weight * kld_loss + archetypal_loss

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD': kld_loss, 'Archetypal_Loss':archetypal_loss}
