import torch.nn as nn
import torch
from .stylegan_generator import StyleGANGenerator

import sys 
sys.path.append("..")
from utils.util import parse_indices

def get_generator(model_name, is_mani=False, mani_layers=[]):
    model = model_name.split("_")[0]
    data_type = model_name.split("_")[1]
    if model == "stylegan":
        net = StyleGAN(model_name, is_mani, mani_layers)
    return net

class StyleGAN(nn.Module):
    def __init__(self, model_name, is_mani=False, mani_layers=[]):
        super(StyleGAN, self).__init__()
        self.stylegan = StyleGANGenerator(model_name)
        self.is_mani = is_mani
        self.mani_layers = mani_layers
        self.in_size = (512,)
        
    def forward(self, latent, boundary=None, steps=None, is_edit=True):
        w = self.stylegan.net.mapping(latent)
        wp = self.stylegan.net.truncation(w)
        if is_edit:
            wp = layers_manipulate(self.stylegan.net.num_layers, wp, boundary, steps, self.mani_layers)
        imgs = self.stylegan.net.synthesis(wp)
        return imgs

def layers_manipulate(num_layers, latent, boundary, steps, mani_layers=False):
    '''
    latent: [bs, num_layers, dim]
    boundary: [1, dim]
    steps: [bs]
    '''
    if steps.shape[0] != latent.shape[0]:
        raise ValueError(f'steps batch size shape {steps.shape} and '
            f'’latent batch size shape {latent.shape} mismatch')
    
    result = latent

    #if isinstance(boundary, list) and len(boundary) == 2:
    #    boundary = boundary[0]
    #else:
    #    raise ValueError(f"boundary should be tensor while now is {len(boundary)}")

    boundary = boundary.repeat(num_layers, 1)
    if boundary.shape[0] != num_layers:
        raise ValueError(f'Boundary should be with shape [num_layers, '
                   f'*code_shape], where `num_layers` equals to '
                   f'{num_layers}, but {boundary.shape} is received!')

    layer_indices = parse_indices(
        mani_layers, min_val=0, max_val=num_layers - 1)
    if len(layer_indices) == 0:
        layer_indices = list(range(num_layers))
    is_manipulatable = torch.zeros(result.shape, dtype=bool).cuda()
    is_manipulatable[:, layer_indices, :] = True

    if latent.shape[1:] != boundary.shape:
        raise ValueError(f'Latent code shape {x.shape} and boundary shape '
                     f'{boundary.shape} mismatch!')

    bs = steps.shape[0]
    boundary = boundary.unsqueeze(0)
    boundary = boundary.repeat(bs, 1, 1)

    steps = steps.unsqueeze(-1)
    steps = steps.unsqueeze(-1)

    trans_result = latent + boundary*steps
    #normalize
    latent_norm = torch.norm(latent, dim=2, keepdim=True)
    trans_norm = (latent_norm / torch.norm(trans_result, dim=2, keepdim=True)) * trans_result

    wp = torch.where(is_manipulatable, trans_norm, result)
    return wp

