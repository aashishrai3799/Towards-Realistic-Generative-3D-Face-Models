## Copyright © 2023 Human Sensing Lab @ Carnegie Mellon University ##

import os
import numpy as np
import torch
import cv2
import numpy as np
import nvdiffrast.torch as dr
from mesh_obj import mesh_obj

def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, uv, uv_idx, tex, resolution, enable_mip, max_mip_level):
    pos_clip = transform_pos(mtx, pos)
    rast_out, rast_out_db = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])

    if enable_mip:
        texc, texd = dr.interpolate(uv[None, ...], rast_out, uv_idx, rast_db=rast_out_db, diff_attrs='all')
        color = dr.texture(tex[None, ...], texc, texd, filter_mode='linear-mipmap-linear', max_mip_level=max_mip_level)
    else:
        texc, _ = dr.interpolate(uv[None, ...], rast_out, uv_idx)
        color = dr.texture(tex[None, ...], texc, filter_mode='linear')

    color = color * torch.clamp(rast_out[..., -1:], 0, 1) # Mask out background.

    # Put white background

    color = color + torch.logical_not(torch.clamp(rast_out[..., -1:], 0, 1))*255

    return color
    
def render2(glctx, mtx, pos, pos_idx, vtx_col, col_idx, resolution: int):
    pos_clip    = transform_pos(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    color, _    = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    return color

def get_mesh_params(mesh):
	faces_v = np.zeros((len(mesh.faces),3))
	faces_vt = np.zeros((len(mesh.faces),3))

	for i in range(len(mesh.faces)):
		faces_v[i] = mesh.faces[i][0]
		faces_vt[i] = mesh.faces[i][2]


	vertices = np.asarray(mesh.vertices)
	texcoords = np.asarray(mesh.texcoords)

	print(vertices.shape, texcoords.shape, faces_v.shape, faces_vt.shape)          # (26317, 3)

	return vertices, texcoords, faces_v, faces_vt

def prep_data(vertices, texcoords, faces_v, faces_vt):

	vtx_pos = torch.from_numpy(vertices.astype(np.float32)).cuda()
	vtx_uv  = torch.from_numpy(texcoords.astype(np.float32)).cuda()
	pos_idx = torch.from_numpy(faces_v.astype(np.int32)).cuda()
	uv_idx  = torch.from_numpy(faces_vt.astype(np.int32)).cuda()

	return vtx_pos, vtx_uv, pos_idx, uv_idx

def get_r_rot(distance, theta):

    R = [[-1, 0, theta],       # A, 0, yaw
                [ 0,  -1, 0],
                [ 0, 0, -1]]

    T = [0.0, 0.0, distance]
    
    r_rot = np.array([[R[0][0], R[0][1], R[0][2], 0],
                        [R[1][0], R[1][1], R[1][2], 0],
                        [R[2][0], R[2][1], R[2][2], 0.5],
                        [0,       T[0],    T[1],    T[2]]], 
                        dtype=np.float32)
                                        
    return r_rot

in_path = 'paper_samples/supplementary/out_data/'
out_path = 'paper_samples/supplementary/out_data_multipose/'
img_nos = [21,75,328,341,355,369,372,405,415,470,559,598,644,646,673,707,720,750,785,790,803,850,856,874,891,905,936,949,957,961,962,969,972]


for img_no in img_nos:

    img_no = str(img_no).zfill(6)
    obj_filename = os.path.join(in_path, img_no, img_no+'.obj') 
    tex_filename = os.path.join(in_path, img_no, img_no+'.png') 

    mesh = mesh_obj(obj_filename)

    vertices, texcoords, faces_v, faces_vt = get_mesh_params(mesh)

    glctx = dr.RasterizeGLContext()
    ref_res = 1024
    max_mip_level = 9

    vtx_pos, vtx_uv, pos_idx, uv_idx = prep_data(vertices, texcoords, faces_v, faces_vt)

    tex = torch.from_numpy(cv2.flip(cv2.imread(tex_filename),0).astype(np.float32)).cuda()

    angles = [0, 15, 30, -30, -15]

    for angle in angles:

        r_rot = get_r_rot(distance=20, theta=angle*torch.pi/180)

        color = render(glctx, r_rot, vtx_pos*150, pos_idx-1, vtx_uv, uv_idx-1, tex, ref_res, True, max_mip_level)

        render_img_path = os.path.join(out_path, img_no+'_'+str(angle)+'_.png')
        # cv2.imwrite(render_img_path, color[0].cpu().numpy())

    # cv2.imwrite(obj_filename_detail.replace('200_detail.obj', 'test.png'), tex_opt.detach().cpu().numpy())
    # cv2.imwrite('test.png', color[0].detach().cpu().numpy())








