
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class MaskedLinear(nn.Module):
    """
    Adopted from https://github.com/rtqichen/ffjord
    Creates masked linear layer for MLP MADE.
    For input (x) to hidden (h) or hidden to hidden layers choose diagonal_zeros = False.
    For hidden to output (y) layers:
    If output depends on input through y_i = f(x_{<i}) set diagonal_zeros = True.
    Else if output depends on input through y_i = f(x_{<=i}) set diagonal_zeros = False.
    """

    def __init__(self, in_features, out_features, diagonal_zeros=False, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diagonal_zeros = diagonal_zeros
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        mask = torch.from_numpy(self.build_mask())
        if torch.cuda.is_available():
            mask = mask.cuda()
        self.mask = torch.autograd.Variable(mask, requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal(self.weight)
        if self.bias is not None:
            self.bias.data.zero_()

    def build_mask(self):
        n_in, n_out = self.in_features, self.out_features
        assert n_in % n_out == 0 or n_out % n_in == 0

        mask = np.ones((n_in, n_out), dtype=np.float32)
        if n_out >= n_in:
            k = n_out // n_in
            for i in range(n_in):
                mask[i + 1 :, i * k : (i + 1) * k] = 0
                if self.diagonal_zeros:
                    mask[i : i + 1, i * k : (i + 1) * k] = 0
        else:
            k = n_in // n_out
            for i in range(n_out):
                mask[(i + 1) * k :, i : i + 1] = 0
                if self.diagonal_zeros:
                    mask[i * k : (i + 1) * k :, i : i + 1] = 0
        return mask

    def forward(self, x):
        output = x.mm(self.mask * self.weight)

        if self.bias is not None:
            return output.add(self.bias.expand_as(output))
        else:
            return output

    def __repr__(self):
        if self.bias is not None:
            bias = True
        else:
            bias = False
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ", diagonal_zeros="
            + str(self.diagonal_zeros)
            + ", bias="
            + str(bias)
            + ")"
        )
