import sys
import os
import numpy as np
import torch
import argparse
import torch.utils
from PIL import Image
from model import Network
from multi_read_data import MemoryFriendlyLoader

parser = argparse.ArgumentParser("ruas")
parser.add_argument('--data_path', type=str, default='./data/LOLv1/test/input',
                    help='folder path of test images')
parser.add_argument('--save_path', type=str, default='./result_ruas_v1',
                    help='folder path to save results')
parser.add_argument('--model', type=str, default='upe',
                    help='checkpoint name: upe / lol / dark')
parser.add_argument('--seed', type=int, default=2,
                    help='random seed')

args = parser.parse_args()

# Mac: 优先 MPS，不行就 CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

# 让 loader 去读整个目录，由 loader 自己匹配 png/jpg/jpeg
test_low_data_names = args.data_path

TestDataset = MemoryFriendlyLoader(img_dir=test_low_data_names, task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset,
    batch_size=1,
    pin_memory=False,
    num_workers=0
)


def save_images(tensor, path):
    image_numpy = tensor[0].detach().cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy = np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8')
    im = Image.fromarray(image_numpy)
    im.save(path, 'png')


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if device.type == "cuda":
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        cudnn.enabled = True
        torch.cuda.manual_seed(args.seed)

    print("args =", args)

    model = Network().to(device)

    ckpt_path = './ckpt/' + args.model + '.pt'
    model_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(model_dict)
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    with torch.no_grad():
        for _, (input_tensor, image_name) in enumerate(test_queue):
            input_tensor = input_tensor.to(device)
            image_name = image_name[0].split('.')[0]
            u_list, r_list = model(input_tensor)

            u_name = f'{image_name}.png'
            u_path = os.path.join(save_path, u_name)
            print('processing {}'.format(u_name))

            if args.model == 'lol':
                save_images(u_list[-1], u_path)
            elif args.model in ['upe', 'dark']:
                save_images(u_list[-2], u_path)
            else:
                save_images(u_list[-1], u_path)


if __name__ == '__main__':
    main()