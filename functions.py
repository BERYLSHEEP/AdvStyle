import torch
import random
from skimage import io as img
from skimage import color, morphology, filters
from PIL import Image
import numpy as np
import torch.nn as nn
import os
import math

def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:0")
    opt.niter_init = opt.niter
    # opt.noise_amp_init = opt.noise_amp
    opt.nfc_init = opt.nfc
    opt.min_nfc_init = opt.min_nfc
    opt.scale_factor_init = opt.scale_factor
    opt.out_ = 'TrainedModels/%s/alpha_ori=%d,alpha_edit=%d' % (opt.input_name, opt.alpha_ori, opt.alpha_edit)
    opt.log_size = int(math.log(opt.resolution, 2))
    opt.wp_layer = (opt.log_size-1)*2
    opt.scale = 256/opt.resolution
    # if opt.mode == 'SR':
    #     opt.alpha = 100

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt

def generate_dir2save(opt):
    dir2save = None
    if (opt.mode == 'train'):
        dir2save = 'TrainedModels/%s/alpha_ori=%d,alpha_edit=%d' % (opt.input_name,opt.alpha_ori, opt.alpha_edit)
    return dir2save

def exist_img(opt):
    path = '%s/Images/%s.png' % (opt.input_dir,opt.input_name)
    if os.path.exists(path):
        return path
    else:
        path = '%s/Images/%s.jpg' % (opt.input_dir,opt.input_name)
        if os.path.exists(path):
            return path
        else:
            return None

def read_image_path(opt, path=None):
    if path == None:
        path = f"{opt.input_dir}/{opt.input_name}/Images/"
    if os.path.exists(path):
        images_name = os.listdir(path)
        images_name.sort()
        #print(images_name)
        images_file = list(os.path.join(path,name) for name in images_name)
        return images_file
    # path = exist_img(opt)
def read_image(paths, opt):
    images = None
    for idx,path in enumerate(paths):
        if os.path.exists(path):
            x = img.imread(path)
            x = np2torch(x,opt)
            x = x[:,0:3,:,:]
            if idx == 0:
                images = x
            else:
                images = torch.cat((images, x), 0)
    return images


def read_latent(opt, path=None, space="w"):
    latents = None
    if path == None:
        path = f"{opt.input_dir}/{opt.input_name}/Latent/"
    if os.path.exists(path):
        latents_name = os.listdir(path)
        latents_name.sort()
        print(latents_name)
        latent_file = list(os.path.join(path,name) for name in latents_name)
    
        for idx,path in enumerate(latent_file):
            latent = np.load(path)
            latent = torch.from_numpy(latent)
            if not opt.not_cuda:
                latent = move_to_gpu(latent)
            if space == "wp" and len(latent.shape) == 2:
                latent = latent.unsqueeze(0)
            if idx == 0:
                latents = latent
            else:
                latents = torch.cat((latents, latent), 0)
    return latents

def read_image_dir(dir,opt):
    x = img.imread('%s' % (dir))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x

def reset_grads(model,require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

def np2torch(x,opt):
    if opt.nc_im == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor)
    x = norm(x)
    return x

def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.detach().cpu().numpy()
    x = x.astype(np.uint8)
    return x

def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    else:
        print("Can not move to gpu.")
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t

def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))
        # mean = np.array([x/255.0 for x in [125.3,123.0,113.9]])
        # std = np.array([x/255.0 for x in [63.0,62.1,66.7]])

    inp = np.clip(inp,0,1)
    return inp

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

def make_image(tensor):
    return tensor.detach().clamp_(min=-1, max=1).add(1).div_(2).mul(255).type(torch.uint8).permute(0, 2, 3, 1).to('cpu').numpy().squeeze(0)


def save_img(im, path,is_torch=True, is_map=False, trans_type=None):
    # utils.save_image(
    #     img.squeeze(0).detach().cpu(),
    #     path,
    #     nrow=5,
    #     normalize=True,
    #     range=(-1, 1))
    if is_torch:
        im = make_image(im)

    if is_map:
        # 是否将img的值【min，max】映射到【0，255】上
        min_val = np.unravel_index(np.argmin(im,axis=None), im.shape)
        max_val = np.unravel_index(np.argmax(im,axis=None), im.shape)
        # print(f"min_val:{min_val}, max_val:{max_val}")
        im = np.round(255*((im-im[min_val])/(im[max_val]-im[min_val])))

    im = Image.fromarray(im.astype(np.uint8))

    if trans_type == "L":
        im = im.convert("L")

    im.save(path)

def save_networks(netEnc,epoch,opt):
    torch.save(netEnc.state_dict(), '%s/netEnc_%d.pth' % (opt.outf,epoch))

def create_reals_pyramid(real, opt, **kwarg):
    reals = []
    reals.append(real)
    for i in range(2,opt.log_size,1):
        real = imresize(real, 0.5, opt)
        # resolution = pow(2,opt.log_size-i+1)
        # if is_save:
        #     img = Image.fromarray(torch2uint8(real))
        #     im.save(f"{opt.out_}/reals/{resolution}x{resolution}.png")
        reals.append(real)
    return reals

def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise, size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise

def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def torch_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def parse_indices(obj, min_val=None, max_val=None):
    """Parses indices.

    If the input is a list or tuple, this function has no effect.

    The input can also be a string, which is either a comma separated list of
    numbers 'a, b, c', or a dash separated range 'a - c'. Space in the string will
    be ignored.

    Args:
    obj: The input object to parse indices from.
    min_val: If not `None`, this function will check that all indices are equal
      to or larger than this value. (default: None)
    max_val: If not `None`, this function will check that all indices are equal
      to or smaller than this field. (default: None)

    Returns:
    A list of integers.

    Raises:
    If the input is invalid, i.e., neither a list or tuple, nor a string.
    """
    if obj is None or obj == '':
        indices = []
    elif isinstance(obj, int):
        indices = [obj]
    elif isinstance(obj, (list, tuple, np.ndarray)):
        indices = list(obj)
    elif isinstance(obj, str):
        indices = []
        splits = obj.replace(' ', '').split(',')
        for split in splits:
            numbers = list(map(int, split.split('-')))
            if len(numbers) == 1:
                indices.append(numbers[0])
            elif len(numbers) == 2:
                indices.extend(list(range(numbers[0], numbers[1] + 1)))
    else:
        raise ValueError(f'Invalid type of input: {type(obj)}!')

    assert isinstance(indices, list)
    indices = sorted(list(set(indices)))
    for idx in indices:
        assert isinstance(idx, int)
        if min_val is not None:
            assert idx >= min_val, f'{idx} is smaller than min val `{min_val}`!'
        if max_val is not None:
            assert idx <= max_val, f'{idx} is larger than max val `{max_val}`!'

    return indices

def torch_mani(latent_origin,
                latent_edit, 
                manipulation_layers=None,
                num_layers=18,
                is_code_layerwise=False
    ):
    assert num_layers > 0
    if not is_code_layerwise:
        x = latent_origin.unsqueeze(1).repeat(1, num_layers, 1)
    else:
        x = latent_origin
    if x.shape[1] != num_layers:
        raise ValueError(f'Latent codes should be with shape [num, num_layers, '
                       f'*code_shape], where `num_layers` equals to '
                       f'{num_layers}, but {x.shape} is received!')
    # if not (len(latent_edit.shape) == 2):
    #     raise ValueError(f'latent edit should be with shape [edit_layers_num, *code_shape]'
    #         f'but {latent_edit.shape} is received')
    layer_indices = parse_indices(
      manipulation_layers, min_val=0, max_val=num_layers - 1)
    if not layer_indices:
        layer_indices = list(range(num_layers))
        if latent_edit.shape[0] == 1 and len(latent_edit.shape) == 2:
            latent_edit = latent_edit.repeat(num_layers, 1)
        elif latent_edit.shape[1] != num_layers and len(latent_edit.shape) == 3:
            raise ValueError(f'latent_edit should be with shape [edit_layers_num, *code_shape]'
                f', where edit_layers_num equals to 1 when manipulation_layers is None'
                f'but {latent_edit.shape} is received') 

    # if latent_edit.shape[0] != len(layer_indices):
    #     raise ValueError(f'latent_edit shape [0] should be equals to the lenght of manipulation_layers'
    #         f'while latent_edit shape is {latent_edit.shape} and layer_indices length is {len(layer_indices)}')

    # is_mani_edit = torch.zeros(x.shape, dtype=bool).cuda()
    # is_mani_edit[:, layer_indices, :] = True

    # is_mani_ori = torch.ones(x.shape, dtype=bool).cuda()
    # is_mani_ori[:, layer_indices, :] = False

    x[:, layer_indices, :] = latent_edit
    #result = torch.where(is_mani_edit, latent_edit, x)

    return x

def manipulation(latent_origin,
                latent_edit, 
                manipulation_layers=None,
                num_layers=18,
                is_code_layerwise=False):

    latent_origin = torch_to_numpy(latent_origin)
    latent_edit = torch_to_numpy(latent_edit)

    assert num_layers > 0
    if not is_code_layerwise:
        x = latent_origin[:, np.newaxis]
        x = np.tile(x, [num_layers if axis == 1 else 1 for axis in range(x.ndim)])
    else:
        x = latent_origin
    
    if x.shape[1] != num_layers:
        raise ValueError(f'Latent codes should be with shape [num, num_layers, '
                       f'*code_shape], where `num_layers` equals to '
                       f'{num_layers}, but {x.shape} is received!')

    if not (len(latent_edit.shape) == 2):
        raise ValueError(f'latent edit should be with shape [edit_layers_num, *code_shape]'
            f'but {latent_edit.shape} is received')


    layer_indices = parse_indices(
      manipulation_layers, min_val=0, max_val=num_layers - 1)
    if not layer_indices:
        layer_indices = list(range(num_layers))
        if latent_edit.shape[0] == 1:
            latent_edit = np.tile(latent_edit, [num_layers if axis == 0 else 1 for axis in range(latent_edit.ndim)])
        else:
            raise ValueError(f'latent_edit should be with shape [edit_layers_num, *code_shape]'
                f', where edit_layers_num equals to 1 when manipulation_layers is None'
                f'but {latent_edit.shape} is received') 

    if latent_edit.shape[0] != len(layer_indices):
        raise ValueError(f'latent_edit shape [0] should be equals to the lenght of manipulation_layers'
            f'while latent_edit shape is {latent_edit.shape} and layer_indices length is {len(layer_indices)}')

    is_manipulatable = np.zeros(x.shape, dtype=bool)
    is_manipulatable[:, :, layer_indices] = True

    x = np.where(is_manipulatable, latent_edit, x)
    x = torch.from_numpy(x).cuda()

    return x


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

class MeanTracker(object):
    def __init__(self, name):
        self.values = []
        self.name = name

    def add(self, val):
        self.values.append(float(val))

    def mean(self):
        return np.mean(self.values)

    def flush(self):
        mean = self.mean()
        self.values = []
        return self.name, mean

def save_comparison(origin_imgs, edit_imgs, epoch, opt):
    scale = 256/opt.resolution
    canvas = Image.new('RGB', (int(opt.resolution*scale*len(origin_imgs)), int(opt.resolution*2*scale)), 'white')
    for col in range(len(origin_imgs)):
        canvas.paste(Image.fromarray(make_image(origin_imgs[col]), mode='RGB'), (int(col*opt.resolution*scale), 0))
        canvas.paste(Image.fromarray(make_image(edit_imgs[col]), mode='RGB'), (int(col*opt.resolution*scale), int(opt.resolution*scale)))
    canvas.save(f'{opt.outf}/mix_{epoch}.png')