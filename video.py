## Copyright © 2023 Human Sensing Lab @ Carnegie Mellon University ##

import os
import torch
import matplotlib.pyplot as plt
from typing import NamedTuple, Sequence
from pytorch3d.io import load_objs_as_meshes, load_obj
import re
import numpy as np
from pytorch3d.structures import Meshes
import cv2

from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (look_at_view_transform,
    FoVPerspectiveCameras, PerspectiveCameras, PointLights, DirectionalLights, Materials, BlendParams, HardPhongShader,
    RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesUV, TexturesVertex)

from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.renderer.mesh.textures import Textures

import sys
# from plot_image_grid import image_grid

def load_objs_as_meshes_custom(obj_filename, device):
    verts, faces, aux = load_obj(obj_filename)
    verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
    faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
    tex_maps = aux.texture_images

    # tex_maps is a dictionary of {material name: texture image}.
    # Take the first image:
    texture_image = list(tex_maps.values())[0]
    texture_image = texture_image[None, ...]  # (1, H, W, 3)

    # Create a textures object
    tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

    # Initialise the mesh with textures
    meshes = Meshes(verts=[verts*120], faces=[faces.verts_idx], textures=tex)

    return meshes.to(device)

def load_objs_as_meshes_custom_sh_light(obj_filename, device):
    verts, faces, aux = load_obj(obj_filename)
    verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
    faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
    tex_maps = aux.texture_images

    # tex_maps is a dictionary of {material name: texture image}.
    # Take the first image:
    texture_image = list(tex_maps.values())[0]
    texture_image = texture_image[None, ...]  # (1, H, W, 3)

    # Create a textures object
    # tex = Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

    # Initialise the mesh with textures
    meshes = Meshes(verts=[verts*120], faces=[faces.verts_idx], textures=tex)

    # return meshes.to(device)
    return [verts*120], [faces.verts_idx], verts_uvs, faces_uvs, texture_image


def render_mesh_tex(mesh, device, distance=10, theta=0, light=(0,0,0)):

    # mesh = load_objs_as_meshes([obj_filename], device=device)
    # mesh = load_objs_as_meshes_custom(obj_filename, device=device)

    R, T = look_at_view_transform(distance, -10, theta)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(image_size=1024, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=True)

    blend_params=BlendParams(gamma=1, background_color=(1.0, 1.0, 1.0))

    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(device=device, cameras=cameras, blend_params=blend_params))

    images = renderer(mesh,cameras=cameras,lights=DirectionalLights(device=device, direction=(light,)))

    return images


def render_mesh_tex_ortho(obj_filename, device, theta):

	mesh = load_objs_as_meshes([obj_filename], device=device)

	R, T = look_at_view_transform(165, 0, theta)
	cameras = PerspectiveCameras(device=device, R=R, T=T)

	raster_settings = RasterizationSettings(image_size=1024, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=True)

	blend_params=BlendParams(gamma=1, background_color=(1.0, 1.0, 1.0))

	renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
			shader=HardPhongShader(device=device, cameras=cameras, blend_params=blend_params))

	images = renderer(mesh,cameras=cameras,lights=DirectionalLights(device=device, direction=((0,0,0),)))
	
	return images


def check_folder(path):
	if not os.path.exists(path):
		os.mkdir(path)

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

print('Device:', device)


in_path = 'paper_samples/disp_gan/out_data_sg/'
out_path = 'paper_samples/disp_gan/videos_sg/'

size = (1024,1024)


img_nos = os.listdir(in_path)
img_nos.sort()


for img_no in img_nos:

    if '.png' not in img_no and '.jpg' not in img_no:

        # img_no = str(img_no).zfill(6)
        obj_file_path = os.path.join(in_path, img_no, img_no+'.obj') 

        out = cv2.VideoWriter(os.path.join(out_path, img_no+'.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), 90, size)

        mesh = load_objs_as_meshes_custom(obj_file_path, device=device)
        print(img_no, 'Generating video...')

        for angle in range(-150,360):
            # print(angle, end='\r')
            image = render_mesh_tex(mesh, device, distance=40, theta=angle/10, light=(0,0,0))*255
            out.write(cv2.cvtColor(image[0,:,:,:3].to(torch.uint8).cpu().numpy(), cv2.COLOR_BGR2RGB))

        for angle in reversed(range(-150,360)):
            # print(angle, end='\r')
            image = render_mesh_tex(mesh, device, distance=40, theta=angle/10, light=(0,0,0))*255
            out.write(cv2.cvtColor(image[0,:,:,:3].to(torch.uint8).cpu().numpy(), cv2.COLOR_BGR2RGB))

        for angle in range(-150,360):
            # print(angle, end='\r')
            image = render_mesh_tex(mesh, device, distance=40, theta=angle/10, light=(0,0,0))*255
            out.write(cv2.cvtColor(image[0,:,:,:3].to(torch.uint8).cpu().numpy(), cv2.COLOR_BGR2RGB))

        for angle in reversed(range(-150,360)):
            # print(angle, end='\r')
            image = render_mesh_tex(mesh, device, distance=40, theta=angle/10, light=(0,0,0))*255
            out.write(cv2.cvtColor(image[0,:,:,:3].to(torch.uint8).cpu().numpy(), cv2.COLOR_BGR2RGB))

        out.release()

        # print(img_no, 'Done!')



