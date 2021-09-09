import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
import numpy as np
import os
from .stylegan_layers import D_basic
from .model_settings import MODEL_POOL

def get_discriminator(type_name, gan_model="stylegan_ffhq", type_num=100, isinit=False, weight_path=None):
    if type_name == 'resnet18':
        net = Resnet18(type_num = type_num)
    elif type_name == 'Resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif type_name == 'Resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif type_name == 'Stylegan_D':
        # load weights
        model_settings = MODEL_POOL[gan_model]
        net = D_basic(
        num_channels        = 3,            # Number of input color channels. Overridden based on dataset.
        resolution          = model_settings["resolution"],           # Input resolution. Overridden based on dataset.
        fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
        fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
        fmap_max            = 512,          # Maximum number of feature maps in any layer.
        nonlinearity        = 'lrelu',      # Activation function: 'relu', 'lrelu',
        use_wscale          = True,         # Enable equalized learning rate?
        mbstd_group_size    = 4,            # Group size for the minibatch standard deviation layer, 0 = disable.
        mbstd_num_features  = 1,            # Number of features for the minibatch standard deviation layer.
        )
        if isinit == True:
            weight_path = model_settings["weight_D_path"]
    else:
        raise Exception("There is no corresponding discriminator")

    if isinit == False:
        print("No pre-trained discriminator weights will be loaded!")
        return net
    if os.path.isfile(weight_path):
        net.load_state_dict(torch.load(weight_path, map_location='cpu'))
        print("Suceefully load discriminator weight")
    else:
        print(f"{weight_path} is not exists.No pre-trained discriminator weights will be loaded!")
    return net

def save_hook(module, input, output):
    setattr(module, 'output', output)


class Resnet18(nn.Module):
    """docstring for Resnet"""
    def __init__(self, type_num):
        super(Resnet18, self).__init__()
        self.features_extractor = resnet18(pretrained=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)

        self.type_estimator = nn.Linear(512, np.product(type_num))

    def forward(self, x):
        batch_size = x.shape[0]
        self.features_extractor(x)
        features = self.features.output.view([batch_size, -1])
        logits = self.type_estimator(features)

        return features, logits
