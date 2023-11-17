# the model

import torch
import torch.nn as nn

import unet_parts

# only for test
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    


# U-Net
# reference: https://github.com/milesial/Pytorch-UNet/tree/master



class UNet(nn.Module):
    def __init__(self,n_channels,n_classes,bilinear=False):
        super(UNet,self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Add layers info
        self.inc = (unet_parts.DoubleConv(n_channels, 64))
        self.down1 = (unet_parts.Down(64, 128))
        self.down2 = (unet_parts.Down(128, 256))
        self.down3 = (unet_parts.Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (unet_parts.Down(512, 1024 // factor))
        self.up1 = (unet_parts.Up(1024, 512 // factor, bilinear))
        self.up2 = (unet_parts.Up(512, 256 // factor, bilinear))
        self.up3 = (unet_parts.Up(256, 128 // factor, bilinear))
        self.up4 = (unet_parts.Up(128, 64, bilinear))
        self.outc = (unet_parts.OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    
    def use_checkpointing(self):
        # TODO: the use of this part
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
    


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