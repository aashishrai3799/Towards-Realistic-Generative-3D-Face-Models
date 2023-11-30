## Copyright © 2023 Human Sensing Lab @ Carnegie Mellon University ##

import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch
import torchvision

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points
import img_2_tex
from stylegan2.model import Generator


def main(args):
    
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    ## Initialize StyleGAN2

    stylegan2_gen = Generator(1024, 512, 8, channel_multiplier=2)
    stylegan2_gen.to(args.device)
    ckpt_path = 'stylegan2/training_logs/checkpoint/stylegan2_ffhq.pt'
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    stylegan2_gen.load_state_dict(ckpt["g_ema"], strict=True)
    # mean_latent = stylegan2_gen.mean_latent(4096)


    ## Initialize AlbedoGAN

    albedogan = Generator(1024, 512, 8, channel_multiplier=2)
    albedogan = torch.nn.DataParallel(albedogan)
    albedogan.to(args.device)
    ckpt_path = 'stylegan2/training_logs/checkpoint/albedogan.pt'
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    albedogan.load_state_dict(ckpt["g"], strict=True)

    ## Initialize Displacement Generator

    disp_GAN = Generator(256, 512, 8, channel_multiplier=2)
    disp_GAN = torch.nn.DataParallel(disp_GAN)
    disp_GAN.to(args.device)
    ckpt_path = 'stylegan2/training_logs/checkpoint/disp_GAN.pt'
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    disp_GAN.load_state_dict(ckpt["g"], strict=True)


    ## Initialize DECA

    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device, D_detail_flag=True, detail_GAN=disp_GAN)

    iscrop = True
    detector = 'fan'
    sample_step = 10

    get_cropped_img = datasets.TestData_stylegan(stylegan2_gen =None, iscrop=iscrop, face_detector=detector, sample_step=sample_step, crop_size=1024)


    ## Generate Face Images using StyleGAN2

    num_imgs = 100

    with torch.no_grad():

        for count in tqdm(range(num_imgs)):

            name = str(count).zfill(6)

            sample_z = torch.randn(1, 512, device=args.device)
            sample_z = stylegan2_gen.get_latent(sample_z)
            img_gt, latents = stylegan2_gen([sample_z], truncation=0.5, truncation_latent=mean_latent, input_is_latent=True)
            img_gt = (img_gt+1)/2
            sample_w = latents

            data_list = get_cropped_img.__getitem__(img_gt[0]*255)
            # img_cropped = data_list['image'].to('cuda')[None,...]
            
            # albedo_pred, _ = albedogan([sample_w], input_is_latent=True)
            
            images = data_list['image'].to(device)[None,...]

            # with torch.no_grad():

            codedict = deca.encode(torchvision.transforms.Resize(224)(images))
            codedict['images'] = images
            codedict['latent'] = sample_w
            opdict, visdict = deca.decode(codedict, name) #tensor

            if args.render_orig:

                tform = data_list['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1,2).to(device)
                original_image = data_list['original_image'][None, ...].to(device)
                _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)    
                orig_visdict['inputs'] = original_image            

            if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
                os.makedirs(os.path.join(savefolder, name), exist_ok=True)

            # -- save results

            if args.saveDepth:
                depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
                visdict['depth_images'] = depth_image
                cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))

            if args.saveKpt:
                np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
                np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())

            if args.saveObj:
                deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict, codedict)
                # cv2.imwrite(os.path.join(savefolder, name, name + '_pred.png'), util.tensor2image(albedo_pred[0]))

            if args.saveMat:
                opdict = util.dict_tensor2npy(opdict)
                savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
            if args.saveVis:
                cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
                if args.render_orig:
                    cv2.imwrite(os.path.join(savefolder, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))

            if args.saveImages:
                for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
                    if vis_name not in visdict.keys():
                        continue
                    image = util.tensor2image(visdict[vis_name][0])
                    cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name +'.jpg'), util.tensor2image(visdict[vis_name][0]))
                    if args.render_orig:
                        image = util.tensor2image(orig_visdict[vis_name][0])
                        cv2.imwrite(os.path.join(savefolder, name, 'orig_' + name + '_' + vis_name +'.jpg'), util.tensor2image(orig_visdict[vis_name][0]))
        print(f'-- please check the results in {savefolder}')


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Towards Realistic 3D Face Synthesis')


    parser.add_argument('-i', '--inputpath', default='inference_test/in_data', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='inference_test/out_data', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--sample_step', default=10, type=int,
                        help='sample images from video data for every step' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    parser.add_argument('--render_orig', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to render results in original image size, currently only works when rasterizer_type=standard')
    # save
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())
