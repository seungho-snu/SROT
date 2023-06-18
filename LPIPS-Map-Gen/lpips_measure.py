import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import util
import cv2
from setuptools import glob

from lpips import lpips

import torch

def fiFindByWildcard(wildcard):
    return glob.glob(os.path.expanduser(wildcard), recursive=True)


def dprint(d):
    out = []
    for k, v in d.items():
        out.append(f"{k}: {v:0.4f}")
    print(", ".join(out))


def t(array):
    return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def imread(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img[:, :, [2, 1, 0]]


# def lpips_analysis(gt_path, sr_path, scale, img_idx):
def lpips_analysis(gt_path, sr_path):

    gt = imread(gt_path)
    sr = imread(sr_path)

    h_gt, w_gt, _ = gt.shape
    h_sr, w_sr, _ = sr.shape
    h = min(h_gt, h_sr)
    w = min(w_gt, w_sr)

    gt = gt[:h, :w]
    sr = sr[:h, :w]

    lpips_sp = loss_fn_alex_sp(2 * t(sr) - 1, 2 * t(gt) - 1)

    # save images
    sr_path_folder, file_name = os.path.split(sr_path)
    sr_lpips_path = sr_path_folder + '_LPIPS'

    LPIPS_Map = util.tensor2img(lpips_sp.detach())
    lpips_map_dir = os.path.expanduser(sr_lpips_path)
    os.makedirs(lpips_map_dir, exist_ok=True)

    gt_file_name = os.path.basename(gt_path)

    lpips_map_path = os.path.join(sr_lpips_path, gt_file_name)
    util.save_img(LPIPS_Map, lpips_map_path)

    return lpips_sp


gt_dir, srs_dir = sys.argv[1:]

gt_dir = os.path.expanduser(gt_dir)
srs_dir = os.path.expanduser(srs_dir)

########################################
# Get Paths
########################################

gt_file_path_list = fiFindByWildcard(os.path.join(gt_dir, '*.png'))
n_imgs = len(gt_file_path_list)

loss_fn_alex_sp = lpips.LPIPS(spatial=True)

lpipses_sp = []

for img_idx in range(n_imgs):
    gt = gt_file_path_list[img_idx]
    gt_file_name = os.path.basename(gt)
    sr = os.path.join(srs_dir, gt_file_name)
    lpips_sp = lpips_analysis(gt, sr)
    lpips_gl = lpips_sp.mean().item()
    print(f'[{img_idx+1}/{n_imgs}] {gt_file_name} - LPIPS: {lpips_gl:.4f}')