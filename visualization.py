import torch
import PIL.Image
import numpy as np
import io
from torchvision.utils import save_image 
from utils.util import torch_to_numpy, manipulation
import time


def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return Image.open(buf)

def convert_array_to_images(np_array):
  """Converts numpy array to images with data type `uint8`.

  This function assumes the input numpy array is with range [-1, 1], as well as
  with shape [batch_size, channel, height, width]. Here, `channel = 3` for color
  image and `channel = 1` for gray image.

  The return images are with data type `uint8`, lying in range [0, 255]. In
  addition, the return images are with shape [batch_size, height, width,
  channel]. NOTE: the channel order will be the same as input.

  Inputs:
    np_array: The numpy array to convert.

  Returns:
    The converted images.

  Raises:
    ValueError: If this input is with wrong shape.
  """
  input_shape = np_array.shape
  if len(input_shape) != 4 or input_shape[1] not in [1, 3]:
    raise ValueError('Input `np_array` should be with shape [batch_size, '
                     'channel, height, width], where channel equals to 1 or 3. '
                     'But {} is received!'.format(input_shape))

  images = (np_array + 1.0) * 127.5
  images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
  images = images.transpose(0, 2, 3, 1)
  return images

@torch.no_grad()
def make_interpolation_chart(G, z=None, direction=None, shifts_r=10, shifts_count=5, resolution=512, canvas=None ,y_axis=0 ,boundary=False, name="",**kwargs):
    '''
    get z+direction image.
    
    Arguments:
        G {[type]} -- [fixed]
        kwargs** :dpi    
    Keyword Arguments:
        z {[tensor]} -- [noise input] (n, 512) 
        direction {[tensor]} -- [trained] dir (n, m, 512) + alpha(n, m, 1)  --m:direction number
        shifts_r {number} -- [shift range] (default: {10})
        shifts_count {number} -- [shift number] (default: {5})
    '''
    assert z.shape[0] == 1, 'just for batch size is one'
    #z = z.expand([shifts_count, z.shape[1]])
    #solo_image_dir = "/home/beryl/Documents/github/multi-direction-code/paper_test_image"

    if isinstance(shifts_r, int):
      shifts_r = [shifts_r]

    l = torch.linspace(0, 2, shifts_count).reshape((shifts_count,1)).expand(shifts_count, len(shifts_r))
    shifts_r = torch.tensor(shifts_r, dtype=torch.float32)
    shifts = ((l-1)* shifts_r).cuda()
    fake_imgs = []
    for index, shift in enumerate(shifts):
      fake_img = G(z, direction, shift, is_edit=True)
      #solo_canvas = PIL.Image.new('RGB',(resolution, resolution), 'white')
      #numpy_img = convert_array_to_images(fake_img.detach().cpu().numpy()).squeeze()
      #solo_canvas.paste(PIL.Image.fromarray(numpy_img, mode='RGB'), (0, 0))
      #solo_canvas.save(solo_image_dir+ "/{}_{:.2}.png".format(name,(shift.detach().cpu().numpy()[0])))
      fake_imgs.append(fake_img)
    fake_imgs = torch.stack(fake_imgs, dim=1).squeeze()

    fake_img = fake_imgs.cpu().detach().numpy()
    #fake_img = fake_img[::-1]
    fake_img = convert_array_to_images(fake_img)
    #绘画
    if canvas == None: 
      canvas = PIL.Image.new('RGB', (resolution*shifts_count, resolution*(y_axis+1)), 'white')
    for col, image in enumerate(list(fake_img)):
      canvas.paste(PIL.Image.fromarray(image, mode='RGB'), (col*resolution, y_axis*resolution))
    return canvas

def adjust_pixel_range(images, min_val=-1.0, max_val=1.0, channel_order='NCHW'):
  """Adjusts the pixel range of the input images.

  This function assumes the input array (image batch) is with shape [batch_size,
  channel, height, width] if `channel_order = NCHW`, or with shape [batch_size,
  height, width] if `channel_order = NHWC`. The returned images are with shape
  [batch_size, height, width, channel] and pixel range [0, 255].

  NOTE: The channel order of output images will remain the same as the input.

  Args:
    images: Input images to adjust pixel range.
    min_val: Min value of the input images. (default: -1.0)
    max_val: Max value of the input images. (default: 1.0)
    channel_order: Channel order of the input array. (default: NCHW)

  Returns:
    The postprocessed images with dtype `numpy.uint8` and range [0, 255].

  Raises:
    ValueError: If the input `images` are not with type `numpy.ndarray` or the
      shape is invalid according to `channel_order`.
  """
  if not isinstance(images, np.ndarray):
    raise ValueError(f'Images should be with type `numpy.ndarray`!')

  channel_order = channel_order.upper()
  if channel_order not in ['NCHW', 'NHWC']:
    raise ValueError(f'Invalid channel order `{channel_order}`!')

  if images.ndim != 4:
    raise ValueError(f'Input images are expected to be with shape `NCHW` or '
                     f'`NHWC`, but `{images.shape}` is received!')
  if channel_order == 'NCHW' and images.shape[1] not in [1, 3]:
    raise ValueError(f'Input images should have 1 or 3 channels under `NCHW` '
                     f'channel order!')
  if channel_order == 'NHWC' and images.shape[3] not in [1, 3]:
    raise ValueError(f'Input images should have 1 or 3 channels under `NHWC` '
                     f'channel order!')

  images = images.astype(np.float32)
  images = (images - min_val) * 255 / (max_val - min_val)
  images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
  if channel_order == 'NCHW':
    images = images.transpose(0, 2, 3, 1)

  return images



@torch.no_grad()
def output_interpolation_chart(G, ws=None, direction=None, shifts_r=10, mani_layers=None, shifts_vec=[], attr=1, shifts_count=5, resolution=512, canvas=None ,y_axis=0 ,**kwargs):
    '''
    get z+direction image.

    Arguments:
        G {[type]} -- [fixed]
        kwargs** :dpi
    Keyword Arguments:
        z {[tensor]} -- [noise input] (n, 512)
        direction {[tensor]} -- [trained] dir (n, m, 512) + alpha(n, m, 1)  --m:direction number
        shifts_r {number} -- [shift range] (default: {10})
        shifts_count {number} -- [shift number] (default: {5})
    '''
    assert ws.shape[0] == 1, 'just for batch size is one'
    #z = z.expand([shifts_count, z.shape[1]])

    if isinstance(shifts_r, int):
      shifts_r = [shifts_r]
    l1 = np.array(shifts_vec, dtype=np.float32) / 100.0
    l1 = np.expand_dims(l1,0)
    l1 = l1.repeat(shifts_count, axis=0)

    l2 = np.linspace(0, 1, shifts_count).reshape((shifts_count,1))
    l1[:,attr-1:attr] = l2

    l = torch.from_numpy(l1)

    shifts_r = torch.tensor(shifts_r, dtype=torch.float32)

    shifts = (l * shifts_r).cuda()
    fake_imgs = []


    wp_np = torch_to_numpy(ws)

    for shift in shifts:
      wp_mani = manipulation(latent_codes=wp_np,
              boundary=direction,
              start_distance=np.asarray([0,0,0,0,0,0,0,0,0,0,0]),
              end_distance=shift,
              steps=1,
              layerwise_manipulation=True,
              num_layers=16,
              manipulation_layers=mani_layers,
              is_code_layerwise=True,
              is_boundary_layerwise=False)
      test_torch = torch.from_numpy(wp_mani[:,0,:,:])
      test_torch = test_torch.type(torch.FloatTensor).cuda()
      fake_img = G.net.synthesis(test_torch)
      fake_imgs.append(fake_img)
    fake_imgs = torch.stack(fake_imgs, dim=1).squeeze()

    fake_img = fake_imgs.cpu().detach().numpy()
    fake_img = convert_array_to_images(fake_img)
    #绘画
    return list(fake_img)
