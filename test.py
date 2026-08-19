import os
import argparse

import torch
import cv2
import numpy as np
import yaml
import lpips
from models import model_rrdb, model_swinir
from datasets import srdata_test
from torch.utils import data

import logging
from utils import utils_logger, util_calculate_psnr_ssim
import codes.PerceptualSimilarity.models as models
from torchvision.transforms.functional import normalize
def img2tensor(imgs, bgr2rgb=True, float32=True):


    
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)
def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--opt', type=str, default='path/to/test_rrdb_P+FSGAN.yml')
    parser.add_argument('--output_path', type=str, default='path/to/result')
    #Urban100 Set5 Set14 Manga109 BSD100 DIV
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    # Initialization
    with open(args.opt, 'r') as f:
        opt = yaml.safe_load(f)
    opt['name'] = opt['name'].replace('RRDB', opt['model_type'])
    print(opt)

    ckpt_path = opt['ckpt_path']

    weight = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    weight = weight['model']

    # Models
    if opt['model_type'].lower() == 'rrdb':
        model = model_rrdb.RRDBNet(**opt['model']['rrdb']).to('cuda')
    else:
        raise ValueError(f"Model {opt['model_type']} is currently unsupported!")

    model.load_state_dict(weight, strict=False)
    model = model.cuda()

    # Datasets
    testset = srdata_test.Test(**opt['test'])
    data_loader_test = data.DataLoader(
        testset,
        **opt['dataloader']['test'],
        shuffle=False,
    )

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if opt['test']['use_hr']:
        logger_name = opt['stage']
        utils_logger.logger_info(logger_name, os.path.join(args.output_path, logger_name+'.log'), mode='w')
        logger = logging.getLogger(logger_name)
        p = 0
        s = 0
        count = 0

    # Start testing
    model.eval()
    for batch in data_loader_test:
        lr = batch['lr']
        fn = batch['fn'][0]
        if opt['test']['use_hr']:
            hr = batch['hr']

        lr = lr.to('cuda')
        with torch.no_grad():
            sr = model(lr)


        sr = sr.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
        sr = sr * 255.
        sr = np.clip(sr.round(), 0, 255).astype(np.uint8)
        sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(args.output_path, fn), sr)

        if opt['test']['use_hr']:
            hr = hr.squeeze(0).numpy().transpose(1, 2, 0)
            hr = hr * 255.
            hr = np.clip(hr.round(), 0, 255).astype(np.uint8)
            hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)

            psnr = util_calculate_psnr_ssim.calculate_psnr(sr, hr, crop_border=4, test_y_channel=True)
            ssim = util_calculate_psnr_ssim.calculate_ssim(sr, hr, crop_border=4, test_y_channel=True)

            p += psnr
            s += ssim

            count += 1

            logger.info('{}: PSNR: {}, SSIM: {}'.format(fn, psnr, ssim))
    if opt['test']['use_hr']:
        p /= count
        s /= count
        logger.info(
            "Avg psnr: {}. ssim: {}. count: {}".format(p, s, count))

    print('Testing finished!')

if __name__ == '__main__':
    main()
