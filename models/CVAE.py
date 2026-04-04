#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Conditional Variational Autoencoder (CVAE) for FedMTL.
Encoder: (x, c) -> (mu, logvar)
Decoder: (z, c) -> x_recon
Loss: Reconstruction + KL(q(z|x,c) || p(z|c))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_img_shape(args):
    """Get image shape based on dataset."""
    if args.dataset in ['fmnist', 'mnist']:
        return (1, 28, 28)
    elif args.dataset in ['cifar10', 'cifar100', 'cinic10']:
        return (3, 32, 32)
    elif args.dataset == 'tinyimagenet':
        return (3, 64, 64)
    elif args.dataset == 'imagenet100':
        return (3, 224, 224)
    elif args.dataset == 'speechcommands':
        return (1, 32, 32)  # spectrogram shape
    return (3, 32, 32)  # default


class CVAE(nn.Module):
    """
    Conditional VAE for FedMTL.
    Captures local data distribution conditioned on class labels.
    """

    def __init__(self, args, latent_dim=64, num_classes=None):
        super(CVAE, self).__init__()
        self.img_shape = get_img_shape(args)
        self.num_classes = num_classes or args.num_classes
        self.latent_dim = latent_dim
        self.input_dim = int(np.prod(self.img_shape))

        # Encoder: x + c -> mu, logvar
        enc_input = self.input_dim + self.num_classes
        self.encoder = nn.Sequential(
            nn.Linear(enc_input, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder: z + c -> x_recon
        dec_input = latent_dim + self.num_classes
        self.decoder = nn.Sequential(
            nn.Linear(dec_input, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim),
            nn.Tanh(),
        )

    def encode(self, x, c):
        """Encode (x, c) -> (mu, logvar)."""
        x_flat = x.view(x.size(0), -1)
        h = torch.cat([x_flat, c], dim=1)
        h = self.encoder(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        """Decode (z, c) -> x_recon."""
        h = torch.cat([z, c], dim=1)
        x_recon = self.decoder(h)
        x_recon = x_recon.view(x_recon.size(0), *self.img_shape)
        return x_recon

    def forward(self, x, c):
        """Forward pass: (x, c) -> (x_recon, mu, logvar)."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, c)
        return x_recon, mu, logvar

    def generate(self, c, num_samples=1, device=None):
        """Generate samples given class condition c (one-hot)."""
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        x = self.decode(z, c)
        return x
