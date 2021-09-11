# -*- coding: utf-8 -*-
from flask import *
from flask_bootstrap import Bootstrap
from urllib.parse import *
import time
import os
import argparse
import torch
import numpy as np
import random
import PIL.Image
from visualization import output_interpolation_chart
from models.stylegan_generator import StyleGANGenerator

app = Flask(__name__)
app.secret_key = 'dev'
ALLOWED_EXTENSIONS = set(['jpeg', 'jpg', 'png', 'gif'])
ATTRIBUTES_ = [
    {'name':'black-hair',
     'boundary':'black',
     'range':15},
    {'name':'pink-hair',
     'boundary':'pink',
     'range':15},
    {'name':'blonde-hair',
     'boundary':'blonde',
     'range':10},
    {'name':'open-mouth',
     'boundary':'open_mouth',
     'range':10},
    {'name':'comic-style',
     'boundary':'comic',
     'range':15},
    {'name':'realness-style',
     'boundary':'realness',
     'range':15},
    {'name':'itomugi-kun-style',
     'boundary':'itomugi_kun',
     'range':15},
    {'name':'chibi-maruko-style',
     'boundary':'maruko',
     'range':10},
    {'name':'hair_length',
     'boundary':'short_hair',
     'range':15},
    {'name':'blunt_bangs',
     'boundary':'blunt_bangs',
     'range':15}
]
SHIFTS_COUNT = 101
parser = argparse.ArgumentParser(description='demo')
parser.add_argument('-i', '--boundaries_dir',
                default='./boundaries',
                help='where the boundaries are located')
parser.add_argument('-d', '--device', default=0)

# Keep the following parameters by default
parser.add_argument('--composing_type', default='w',
                    help='composing type, "mid" for composing at middle of GAN. choices: z, w, w+, mid')
parser.add_argument('--direction_type', default='w',
                    help='where direction is belong to, "mid_space" for direction belong to feature space of GAN')
parser.add_argument('--w_start', type=int, default=0)
parser.add_argument('--w_end', type=int, default=7)

args, other_args = parser.parse_known_args()
torch.cuda.set_device(0)
out_dir = 'static/results'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

gan = StyleGANGenerator("stylegan_anime")
gan.net.eval()

Bootstrap(app)

direction_list = []
shifts_r_list = []
manipulate_layers_list = []
for i in range(10):
    boundary_name = ATTRIBUTES_[i]['boundary']
    direction_path = os.path.join(args.boundaries_dir, f"{boundary_name}.npy")
    direction_dict = np.load(direction_path, allow_pickle=True)[()]
    direction_list.append(direction_dict["boundary"])
    # direction vector
    manipulate_layers_list.append(direction_dict["manipulate_layers"])
    # specific operation layers
    shifts_r_list.append(direction_dict["shift_range"][1])

def edit(shifts, attr, composing_type="z"):
    noise_torch = torch.from_numpy(np.load('noise/38_w+.npy')).cuda()
    noise_torch = noise_torch.unsqueeze(0)
    if composing_type == "w+":
        ws = noise_torch
    elif composing_type == "z":
        w = gan.net.mapping(noise_torch)
        ws = gan.net.truncation(w)
    if composing_type == "w":
        ws = gan.net.truncation(noise_torch)

    fake_img = output_interpolation_chart(gan, ws=ws, direction=direction_list, shifts_r=shifts_r_list, mani_layers=manipulate_layers_list, shifts_vec=shifts, attr=attr, shifts_count=SHIFTS_COUNT)
    os.makedirs(out_dir, exist_ok=True)
    for i,image in enumerate(fake_img):
        img = PIL.Image.fromarray(image, mode='RGB')
        img.save(os.path.join(out_dir,str(i)+'.jpg'))


@app.route("/randomize")
def randomize():
    z = np.load('noise/maruko/16.npy')
    attr = 1
    steps = [0 for i in range(10)]
    edit(steps, attr, args.composing_type)
    _steps = []
    for i,step in enumerate(steps):
        if i == attr-1:
            _steps.append(str(0))
        else:
            _steps.append(str(step))
    imgSrc = os.path.join(out_dir, "loading.gif")
    return render_template('index.html', imgSrc=imgSrc, steps=_steps, attr=str(attr))


@app.route('/pos', methods=[ "POST"])
def rootpost():
    if request.method == "POST":
        attr = int(request.form['fname'])
        steps = []
        steps.append(int(request.form['black-hair-step']))
        steps.append(int(request.form['pink-hair-step']))
        steps.append(int(request.form['blonde-hair-step']))
        steps.append(int(request.form['open-mouth-step']))
        steps.append(int(request.form['comic-step']))
        steps.append(int(request.form['ffhq-step']))
        steps.append(int(request.form['ikun-step']))
        steps.append(int(request.form['chibi-step']))
        steps.append(int(request.form['hairlength-step']))
        steps.append(int(request.form['bluntbangs-step']))
        edit(steps, attr, args.composing_type)
        _steps = []
        for i,step in enumerate(steps):
            if i == attr-1:
                _steps.append(str(0))
            else:
                _steps.append(str(step))
        imgSrc = 'static/results/0.jpg?ran='+str(random.random())
        return render_template('index.html', imgSrc=imgSrc, steps=_steps, attr=str(attr))

@app.route('/')
def root():
    z = np.load('noise/maruko/16.npy')
    attr = 1
    steps = [0 for i in range(10)]
    edit(steps, attr,args.composing_type)
    _steps = []
    for i,step in enumerate(steps):
        if i == attr-1:
            _steps.append(str(0))
        else:
            _steps.append(str(step))
    imgSrc = 'static/results/0.jpg?ran='+str(random.random())
    return render_template('index.html', imgSrc=imgSrc, steps=_steps, attr=str(attr))

if __name__=="__main__":
    app.run(debug=True)
