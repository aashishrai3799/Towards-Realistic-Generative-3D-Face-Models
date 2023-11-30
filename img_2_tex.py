## Copyright © 2023 Human Sensing Lab @ Carnegie Mellon University ##

import os
import torchvision
import torch
from tqdm import tqdm
import math
import numpy as np
import cv2

run_this_file=0

if run_this_file == 1:
    # from decalib.utils.renderer import SRenderY, set_rasterizer
    from decalib.datasets import datasets 
    # from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg
    from decalib.deca import DECA


    image_size = 1024
    # topology_path = '/media/aashish/HDD2/stylegan3d/DECA_new/DECA/data/head_template.obj'
    uv_size = 1024
    rasterizer_type = 'pytorch3d'
    device = 'cuda'
    # savefolder = 'pose_tex/out_data/'
    savefolder = '/media/aashish/HDD/texture_train_images/'
    # inputpath = '/media/aashish/HDD2/stylegan3d/stylegan2_pytorch_dr/1000_samples/'
    # inputpath = 'pose_tex/in_data/'
    inputpath = '/media/aashish/HDD2/stylegan3d/stylegan2_pytorch_dr/1000_samples/'
    iscrop = True
    detector = 'fan'
    sample_step = 10
    useTex = True
    extractTex = True


    savefolder = savefolder
    device = device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(inputpath, iscrop=iscrop, face_detector=detector, sample_step=sample_step, crop_size=1024)

    # run DECA
    deca_cfg.model.use_tex = useTex
    deca_cfg.rasterizer_type = rasterizer_type
    deca_cfg.model.extract_tex = extractTex
    deca = DECA(config = deca_cfg, device=device)

    # render = SRenderY(image_size, obj_filename=topology_path, uv_size=uv_size, rasterizer_type=rasterizer_type).to(device)



def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def get_normal(p1, p2, p3):
	return np.cross(p2-p1, p3-p1)

def mesh_angle(vertices, vertex_ids):

	normal = get_normal(np.array(vertices[vertex_ids[0]]), 
								np.array(vertices[vertex_ids[1]]), 
								np.array(vertices[vertex_ids[2]]))

	ang = int(angle(normal, [1,0,1])*360/math.pi)

	return ang

def tex_correction(uv_texture, angle):

    if angle < 0:
        max_pixel = 512
        arr = np.array(range(max_pixel))/max_pixel
        arr_flip = np.flip(arr, 0)
        uv_texture[200:,:max_pixel,:] = torch.flip(uv_texture, (1,))[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture[200:,:max_pixel,:] * arr[None,...,None]
        uv_texture[200:,:max_pixel,:] = torch.flip(uv_texture, (1,))[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture[200:,:max_pixel,:] * arr[None,...,None]
        uv_texture[200:,:max_pixel,:] = torch.flip(uv_texture, (1,))[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture[200:,:max_pixel,:] * arr[None,...,None]
        uv_texture[200:,:max_pixel,:] = torch.flip(uv_texture, (1,))[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture[200:,:max_pixel,:] * arr[None,...,None]

    else:
        max_pixel = -512
        arr = np.array(range(abs(max_pixel)))/abs(max_pixel)
        arr_flip = np.flip(arr, 0)
        uv_texture[200:,max_pixel:,:] = torch.flip(uv_texture, (1,))[200:,max_pixel:,:] * arr[None,...,None] + uv_texture[200:,max_pixel:,:] * arr_flip[None,...,None]
        uv_texture[200:,max_pixel:,:] = torch.flip(uv_texture, (1,))[200:,max_pixel:,:] * arr[None,...,None] + uv_texture[200:,max_pixel:,:] * arr_flip[None,...,None]
        uv_texture[200:,max_pixel:,:] = torch.flip(uv_texture, (1,))[200:,max_pixel:,:] * arr[None,...,None] + uv_texture[200:,max_pixel:,:] * arr_flip[None,...,None]
        uv_texture[200:,max_pixel:,:] = torch.flip(uv_texture, (1,))[200:,max_pixel:,:] * arr[None,...,None] + uv_texture[200:,max_pixel:,:] * arr_flip[None,...,None]

    return uv_texture

def tex_correction_eye(uv_texture, angle):

    if angle < 0:
        max_pixel = 512
        eye = 1
        arr = np.array(range(max_pixel))/max_pixel
        arr_flip = np.flip(arr, 0)
        uv_texture[:,:max_pixel,:] = torch.flip(uv_texture, (1,))[:,:max_pixel,:]
        uv_texture[:200,:200,:] = eye

    else:
        max_pixel = -512
        eye = uv_texture[:200,-200:,:].clone()
        arr = np.array(range(abs(max_pixel)))/abs(max_pixel)
        arr_flip = np.flip(arr, 0)
        uv_texture[:,max_pixel:,:] = torch.flip(uv_texture, (1,))[:,max_pixel:,:]
        uv_texture[:200,-200:,:] = eye

    return uv_texture


def tex_merge(uv_texture_r, uv_texture_c, uv_texture_l):

    max_pixel = 512
    arr = np.array(range(max_pixel))/max_pixel
    arr_flip = np.flip(arr, 0)
    uv_texture_c[200:,:max_pixel,:] = uv_texture_l[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture_c[200:,:max_pixel,:] * arr[None,...,None]
    uv_texture_c[200:,:max_pixel,:] = uv_texture_l[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture_c[200:,:max_pixel,:] * arr[None,...,None]
    uv_texture_c[200:,:max_pixel,:] = uv_texture_l[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture_c[200:,:max_pixel,:] * arr[None,...,None]
    uv_texture_c[200:,:max_pixel,:] = uv_texture_l[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture_c[200:,:max_pixel,:] * arr[None,...,None]
    uv_texture_c[200:,:max_pixel,:] = uv_texture_l[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture_c[200:,:max_pixel,:] * arr[None,...,None]
    # uv_texture[200:,:max_pixel,:] = torch.flip(uv_texture, (1,))[200:,:max_pixel,:] * arr_flip[None,...,None] + uv_texture[200:,:max_pixel,:] * arr[None,...,None]

    max_pixel = -512
    arr = np.array(range(abs(max_pixel)))/abs(max_pixel)
    arr_flip = np.flip(arr, 0)
    uv_texture_c[200:,max_pixel:,:] = uv_texture_r[200:,max_pixel:,:] * arr[None,...,None] + uv_texture_c[200:,max_pixel:,:] * arr_flip[None,...,None]
    uv_texture_c[200:,max_pixel:,:] = uv_texture_r[200:,max_pixel:,:] * arr[None,...,None] + uv_texture_c[200:,max_pixel:,:] * arr_flip[None,...,None]
    uv_texture_c[200:,max_pixel:,:] = uv_texture_r[200:,max_pixel:,:] * arr[None,...,None] + uv_texture_c[200:,max_pixel:,:] * arr_flip[None,...,None]
    uv_texture_c[200:,max_pixel:,:] = uv_texture_r[200:,max_pixel:,:] * arr[None,...,None] + uv_texture_c[200:,max_pixel:,:] * arr_flip[None,...,None]
    uv_texture_c[200:,max_pixel:,:] = uv_texture_r[200:,max_pixel:,:] * arr[None,...,None] + uv_texture_c[200:,max_pixel:,:] * arr_flip[None,...,None]
    # uv_texture[200:,max_pixel:,:] = torch.flip(uv_texture, (1,))[200:,max_pixel:,:] * arr[None,...,None] + uv_texture[200:,max_pixel:,:] * arr_flip[None,...,None]

    return uv_texture_c


def main():
    
    for i in tqdm(range(len(testdata))):

        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None,...]

        with torch.no_grad():

            codedict = deca.encode(torchvision.transforms.Resize(224)(images))
            ## images: [0, 1]
            codedict['images'] = images
            uv_tex, vertices = deca.decode_tex(codedict)

            angle1 = mesh_angle(vertices[0].detach().cpu().numpy(), [3572,3555,2205])
            angle2 = mesh_angle(vertices[0].detach().cpu().numpy(), [3572,723,3555])

            avg_ang = int((angle1+angle2)/2)

            avg_ang = 90-(360-avg_ang)

            print(name, avg_ang)

            correct_tex = tex_correction(uv_tex[0].permute(1,2,0).detach().cpu(), avg_ang)
            # correct_tex = uv_tex[0].permute(1,2,0).detach().cpu()

            # TODO Perlin Noise, Blending

            cv2.imwrite(os.path.join(savefolder,name+'.png'), cv2.cvtColor(correct_tex.numpy()*255, cv2.COLOR_BGR2RGB))
            # print('Done')

def get_tex_from_img(images, get_cropped_img, deca):
    
    textures = torch.zeros_like(images).to('cuda')
    count=0

    for img in images:

        # name = testdata[i]['imagename']
        # images = testdata[i]['image'].to(device)[None,...]
        data_list = get_cropped_img.__getitem__(img*255)
        img_cropped = data_list['image'].to('cuda')[None,...]

        with torch.no_grad():

            codedict = deca.encode(torchvision.transforms.Resize(224)(img_cropped))
            codedict['images'] = img_cropped
            uv_tex, vertices, uv_face_eye_mask, uv_texture = deca.decode_tex(codedict)

            angle1 = mesh_angle(vertices[0].detach().cpu().numpy(), [3572,3555,2205])
            angle2 = mesh_angle(vertices[0].detach().cpu().numpy(), [3572,723,3555])

            avg_ang = int((angle1+angle2)/2)
            avg_ang = 90-(360-avg_ang)

            correct_tex = tex_correction(uv_tex[0].permute(1,2,0).detach().cpu(), avg_ang)
            correct_tex = correct_tex.permute(2,0,1)[None,...].to('cuda')
            correct_tex = correct_tex[:,:3,:,:]*uv_face_eye_mask + (uv_texture[:,:3,:,:]*(1-uv_face_eye_mask))
            textures[count] = correct_tex
            count+=1
            # print('done')
            # correct_tex = uv_tex[0].permute(1,2,0).detach().cpu()

            # TODO Perlin Noise, Blending

    return textures


if run_this_file == 1:
    main()            
