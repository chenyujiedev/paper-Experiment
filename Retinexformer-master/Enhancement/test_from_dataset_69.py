# Retinexformer inference-only script for custom low-light images (Mac / MPS / CPU)
# python setup.py develop --no_cuda_ext
import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte

from basicsr.models import create_model
from basicsr.utils.options import parse


def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x

    outs = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                outs.append(forward_transformed(x, hflip, vflip, rot, model))
    outs = torch.stack(outs)
    return torch.mean(outs, dim=0)


parser = argparse.ArgumentParser(description='Retinexformer inference on custom low-light images')
parser.add_argument('--input_dir', default='../data/LoLv1/Test/input', type=str, help='Directory of low-light input images')
parser.add_argument('--result_dir', default='results_data/', type=str, help='Directory for output results')
parser.add_argument('--opt', type=str, default='../Options/result_retinex_v1.yml', help='Path to option YAML file')
parser.add_argument('--weights', default='../pretrained_weights/LOL_v1.pth', type=str, help='Path to weights')
parser.add_argument('--self_ensemble', action='store_true', help='Use self-ensemble for better results')
args = parser.parse_args()

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print("Using device:", device)
print("Input dir:", args.input_dir)

# parse option
opt = parse(args.opt, is_train=False)
opt['dist'] = False
opt['num_gpu'] = 0  # force no cuda data parallel

# create model
model_restoration = create_model(opt).net_g

checkpoint = torch.load(args.weights, map_location=device)
try:

    model_restoration.load_state_dict(checkpoint['params'])
except Exception:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===> Testing using weights:", args.weights)
model_restoration = model_restoration.to(device)
model_restoration.eval()

factor = 4
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, config, checkpoint_name)
os.makedirs(result_dir, exist_ok=True)

input_paths = natsorted(
    glob(os.path.join(args.input_dir, '*.png')) +
    glob(os.path.join(args.input_dir, '*.jpg')) +
    glob(os.path.join(args.input_dir, '*.jpeg')) +
    glob(os.path.join(args.input_dir, '*.bmp'))
)

print(f'Found {len(input_paths)} images')

with torch.inference_mode():
    for inp_path in tqdm(input_paths):
        if device.type == 'cuda':
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

        img = np.float32(utils.load_img(inp_path)) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).to(device)

        b, c, h, w = input_.shape
        H = ((h + factor) // factor) * factor
        W = ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

        if h < 3000 and w < 3000:
            if args.self_ensemble:
                restored = self_ensemble(input_, model_restoration)
            else:
                restored = model_restoration(input_)
        else:
            input_1 = input_[:, :, :, 1::2]
            input_2 = input_[:, :, :, 0::2]
            if args.self_ensemble:
                restored_1 = self_ensemble(input_1, model_restoration)
                restored_2 = self_ensemble(input_2, model_restoration)
            else:
                restored_1 = model_restoration(input_1)
                restored_2 = model_restoration(input_2)
            restored = torch.zeros_like(input_)
            restored[:, :, :, 1::2] = restored_1
            restored[:, :, :, 0::2] = restored_2

        restored = restored[:, :, :h, :w]
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        save_name = os.path.splitext(os.path.basename(inp_path))[0] + '.png'
        save_path = os.path.join(result_dir, save_name)
        utils.save_img(save_path, img_as_ubyte(restored))

print("Done. Results saved to:", result_dir)