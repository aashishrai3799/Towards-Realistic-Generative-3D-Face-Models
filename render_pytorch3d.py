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


# in_path = 'interpol_id/'
# out_path = 'rendered_id_interpolation/'


# folders = os.listdir(in_path)
# folders.sort()


# obj_file_path = 'flame_reconstruct/000001/000001.obj'
in_path = 'paper_samples/supplementary/out_data/'
out_path = 'paper_samples/supplementary/videos/'

size = (1024,1024)


'''
# img_nos = [21,75,328,341,355,369,372,405,415,470,559,598,644,646,673,707,720,750,785,790,803,850,856,874,891,905,936,949,957,961,962,969,972]


# for img_no in img_nos:

#     img_no = str(img_no).zfill(6)
#     obj_filename = os.path.join(in_path, img_no, img_no+'.obj') 

#     image_front = render_mesh_tex(obj_filename, device, distance=50, theta=0, light=(0,0,1))*255
#     image_left = render_mesh_tex(obj_filename, device, distance=50, theta=0, light=(-1,0,1))*255
#     image_right = render_mesh_tex(obj_filename, device, distance=50, theta=0, light=(1,0,1))*255
#     image_top = render_mesh_tex(obj_filename, device, distance=50, theta=0, light=(0,1,1))*255
#     image_bottom = render_mesh_tex(obj_filename, device, distance=50, theta=0, light=(0,-1,1))*255

#     render_img_path = os.path.join(out_path, img_no+'_front_.png')
#     cv2.imwrite(render_img_path, cv2.cvtColor(image_front[0].cpu().numpy(), cv2.COLOR_BGR2RGB))

#     render_img_path = os.path.join(out_path, img_no+'_left_.png')
#     cv2.imwrite(render_img_path, cv2.cvtColor(image_left[0].cpu().numpy(), cv2.COLOR_BGR2RGB))

#     render_img_path = os.path.join(out_path, img_no+'_right_.png')
#     cv2.imwrite(render_img_path, cv2.cvtColor(image_right[0].cpu().numpy(), cv2.COLOR_BGR2RGB))

#     render_img_path = os.path.join(out_path, img_no+'_top_.png')
#     cv2.imwrite(render_img_path, cv2.cvtColor(image_top[0].cpu().numpy(), cv2.COLOR_BGR2RGB))

#     render_img_path = os.path.join(out_path, img_no+'_bottom_.png')
#     cv2.imwrite(render_img_path, cv2.cvtColor(image_bottom[0].cpu().numpy(), cv2.COLOR_BGR2RGB))

'''


# img_nos = [21,75,328,341,355,369,372,405,415,470,559,598,644,646,673,707,720,750,785,790,803,850,856,874,891,905,936,949,957,961,962,969,972]

img_nos = [341]


for img_no in img_nos:

    img_no = str(img_no).zfill(6)
    obj_file_path = os.path.join(in_path, img_no, img_no+'.obj') 

    out = cv2.VideoWriter(os.path.join(out_path, img_no+'text_.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), 90, size)


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

    print(img_no, 'Done!')

#for folder in ['sample_14','sample_19']: #folders[9:]:
# for folder in folders[10:]:
	
# 	check_folder(os.path.join(out_path, folder))
# 	exps = os.listdir(os.path.join(in_path,folder))
# 	exps.sort()
	
# 	#for exp in exps:
# 	for exp in ['17_eye_closed', '1_smile', '0_neutral', '2_mouth_stretch']:
		
# 		check_folder(os.path.join(out_path, folder, exp))
# 		files = os.listdir(os.path.join(in_path, folder, exp))
# 		files.sort()

# 		#-----------------------------------------------------#
		
# 		#files = os.listdir(in_path)
# 		objs = []
# 		files=sorted_alphanumeric(files)
# 		for f in files:
# 			if f[-4:] == '.obj':
# 				objs.append(f)

# 		#objs.sort()

# 		print('Objs found:', len(objs))
		
# 		angle = -14

# 		for obj in objs[:15]:

# 			obj_filename = os.path.join(in_path, folder, exp, obj)
			
# 			images = render_mesh_tex(obj_filename, device, theta=0)  ## RENDER FUNCTION ##
			
# 			angle+=2
			
# 			plt.figure(figsize=(10, 10))
# 			imgg = images[0, ..., :3].cpu().numpy()
			
# 			#print(imgg[:10,0,0])
			
# 			plt.imshow(imgg)
# 			plt.axis("off");
# 			plt.savefig(os.path.join(out_path, folder, exp, obj[:-4]+'.jpg'))
# 			#plt.show()

# 			torch.cuda.empty_cache()






































