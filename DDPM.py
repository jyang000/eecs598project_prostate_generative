# DDPM model
# reference: todo

# author: jiayao

import torch
import torch.nn as nn


# DDPM model
# reference: https://github.com/yang-song/score_sde_pytorch/tree/main


class DDPM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Add layers
        self.activation = None

        # Downsampling blocks

        # Upsampling blocks

    def forward(self,x):

        h = x

        # Passing through Downsampling blocks

        # Passing through Upsampling blocks

        return x