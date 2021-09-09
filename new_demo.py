import os
import argparse
import torch
import sys
import numpy as np
import math
from tqdm import tqdm
import cv2
from functions import save_img, read_image_path
from visualization import adjust_pixel_range
from models.stylegan_generator import StyleGANGenerator
from utils.util import torch_to_numpy, manipulation


def parse_attr(attribute):
    attr_list = attribute.split(",")
    return attr_list

def manipulate_test(attribute, output_dir, noise_path, resolution, gan_model, latent_type):
    attr_list = parse_attr(attribute)
    boundary = []
    manipulate_layers = []
    shift_range = []

    for attr in attr_list:
        direction_path = os.path.join("./boundaries", f"{attr}.npy")
        direction_dict = np.load(direction_path, allow_pickle=True)[()]
        boundary.append(direction_dict["boundary"])
        # direction vector
        manipulate_layers.append(direction_dict["manipulate_layers"])
        # specific operation layers
        shift_range.append(direction_dict["shift_range"])

        '''
        recommand range
        direction_dict["shift_range"]: [-10, 10] 
        represents that the negative direction step is -10 and the positive direction step is 10
        '''


    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/{attr_list[0]}", exist_ok=True)

    step = 7
    gan = StyleGANGenerator(gan_model)
    gan.net.eval()

    num_layers = int((math.log(resolution, 2) -1)*2)
    
    if noise_path is None:
        noise_paths = []
        for attr in attr_list:
            dir_name = f"./noise/{attr}"
            noise_path = read_image_path(None, dir_name)
            noise_paths += noise_path
    else:
        if not os.path.exists(noise_path):
            raise ValueError(f"noise path is not exist: {noise_path}")
        noise_paths = [noise_path]


    with tqdm(total = len(noise_paths)) as pbar:
        for noise_path in noise_paths:
            name = os.path.basename(noise_path).split(".")[0]
            latent = np.load(noise_path)
            noise_torch = torch.from_numpy(latent).float().cuda()
            
            #noise_torch = torch.randn((1,512)).cuda()
            #np.save("./result/test.npy", noise_torch.detach().cpu())
            #w = gan.net.mapping(noise_torch)
            #ws = gan.net.truncation(w)
            #image = gan.net.synthesis(ws)
            #save_img(image, "./result/test.png",is_torch=True, is_map=False, trans_type=None)


            if latent_type == "ws":
                ws = noise_torch
            elif latent_type == "z":
                w = gan.net.mapping(noise_torch)
                ws = gan.net.truncation(w)
            if latent_type == "w":
                ws = gan.net.truncation(noise_torch)


            output_images = []
            #bdary = torch.from_numpy(boundary[0].T).cuda()
            #w = w - torch.matmul(w, bdary)*torch.from_numpy(boundary[0]).cuda()
            wp_np = torch_to_numpy(ws)
            shift_range = np.array(shift_range)
            
            wp_mani = manipulation(latent_codes=wp_np,
                    boundary=boundary,
                    start_distance=shift_range[:,0],
                    end_distance=shift_range[:,1],
                    steps=step,
                    layerwise_manipulation=True,
                    num_layers=num_layers,
                    manipulation_layers=manipulate_layers,
                    is_code_layerwise=True,
                    is_boundary_layerwise=False)
            '''
            When generating one image,
            please set step to 1,
            set end_distance to x,
            where shift_range[:,0] <= x <= shift_range[:,1] is recommended,
            set start_distance randomly.

            when generating multi images(multi steps),
            please set end_distance to shift_range[:,1],
            set start_distance to shift_range[:,0]
            
            wp_np shape: [batch_size, steps, num_layers, *code_shape]
            '''
            for step_idx in range(step):
                test_torch = torch.from_numpy(wp_mani[:,step_idx,:,:])
                test_torch = test_torch.type(torch.FloatTensor).cuda()
                images = gan.net.synthesis(test_torch)
                
                save_img(images, f"./{output_dir}/{attr_list[0]}/{name}_{step_idx}.png",is_torch=True, is_map=False, trans_type=None)

            pbar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    p = subparsers.add_parser("manipulate_test")
    p.add_argument("attribute", help="manipulate attribute name")
    p.add_argument("--output_dir", default="./result")
    p.add_argument("--noise_path", help="the path of input noise file", default=None)
    p.add_argument("--resolution", default=1024, type=int)
    p.add_argument("--gan_model", default="stylegan_ffhq", type=str)
    # 示例： python new_demo.py manipulate_test maruko,comic   
    # manipulate "maruko, comic" attribute simultaneously using default maximum steps.
    p.add_argument("--latent_type", default="ws", type=str)

    args = parser.parse_args(sys.argv[1:])
    func = globals()[args.command]
    del args.command
    func(**vars(args))