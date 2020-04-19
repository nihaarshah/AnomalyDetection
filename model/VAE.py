# Aspects of code borrowed from github.com/rtqichen/ffjord/

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(
        self, input_size, encoder_sizes, decoder_sizes, z_size, batch_norm=False, dropout=0
    ):
        super().__init__()
        self.input_size = input_size
        self.z_size = z_size
        self.encoder_sizes = [input_size] + encoder_sizes
        self.decoder_sizes = [self.z_size] + decoder_sizes
        self.batch_norm = batch_norm
        self.dropout = dropout
        self._set_encoder()
        self._set_decoder()
        self.weight_init()

        self.FloatTensor = (
            torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        )

        # log-det-jacobian = 0 without flows
        self.log_det_j = self.FloatTensor(1).zero_()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.normal_(m.bias, 0, 0.01)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _set_encoder(self):
        """Set encoder layers and z_mu, z_var """

        layers = []
        for l in range(1, len(self.encoder_sizes)):
            layers.append(nn.Linear(self.encoder_sizes[l - 1], self.encoder_sizes[l]))
            layers.append(nn.LeakyReLU(.1))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.encoder_sizes[l]))
            layers.append(nn.Dropout(self.dropout))

        self.encoder = nn.Sequential(*layers)
        self.z_mu = nn.Linear(self.encoder_sizes[-1], self.z_size)
        self.z_var = nn.Sequential(nn.Linear(self.encoder_sizes[-1], self.z_size), nn.Softplus())

    def _set_decoder(self):
        """Set decoder layers """

        layers = []
        for l in range(1, len(self.decoder_sizes)):
            layers.append(nn.Linear(self.decoder_sizes[l - 1], self.decoder_sizes[l]))
            layers.append(nn.ReLU())
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.decoder_sizes[l]))
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(self.decoder_sizes[-1], self.input_size))
        self.decoder = nn.Sequential(*layers)

    def reparameterize(self, mu, var):
        """Reparameterization trick (sample z via a standard normal)"""

        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_()
        z = eps.mul(std).add_(mu)

        return z

    def encode(self, x):

        x = self.encoder(x)
        mu = self.z_mu(x)
        var = self.z_var(x)

        return mu, var

    def decode(self, z):

        return self.decoder(z)

    def forward(self, x):

        z_mu, z_var = self.encode(x)
        z = self.reparameterize(z_mu, z_var)

        # Normalizing flows here

        output = self.decode(z)
        # Return more here with flows like in https://github.com/rtqichen/ffjord/blob/bce4d2def767f2b9a3288ae0b5d43781ad4dc6b1/vae_lib/models/VAE.py#L170
        return output, z_mu, z_var, z
