## This file has been taken from MICA ##

import os
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import torch
from insightface.app.common import Face
from insightface.utils import face_align
from numpy.lib import math

input_mean = 127.5
input_std = 127.5


def create_folders(folders):
    if not type(folders) is list:
        folders = folders.split('/')
    parents = '/'
    for folder in folders:
        parents = os.path.join(parents, folder)
        if os.path.exists(parents):
            continue
        Path(parents).mkdir(exist_ok=True)


def get_arcface_input(face, img):
    aimg = face_align.norm_crop(img, landmark=face.kps)
    blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
    return blob[0], aimg


def get_arcface_input_from_full_face(full_face):
    # In cases when we fail to detect bbox, use full_face to process arcface_inputs
    aimg = cv2.resize(full_face, (112, 112))
    blob = cv2.dnn.blobFromImages([aimg], 1.0 / input_std, (112, 112), (input_mean, input_mean, input_mean), swapRB=True)
    return blob[0], aimg

def get_image(name, to_rgb=False):
    images_dir = osp.join(Path(__file__).parent.absolute(), '../images')
    ext_names = ['.jpg', '.png', '.jpeg']
    image_file = None
    for ext_name in ext_names:
        _image_file = osp.join(images_dir, "%s%s" % (name, ext_name))
        if osp.exists(_image_file):
            image_file = _image_file
            break
    assert image_file is not None, '%s not found' % name
    img = cv2.imread(image_file)
    if to_rgb:
        img = img[:, :, ::-1]
    return img


def dist(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def get_center(bboxes, img):
    img_center = img.shape[0] // 2, img.shape[1] // 2
    size = bboxes.shape[0]
    distance = np.Inf
    j = 0
    for i in range(size):
        x1, y1, x2, y2 = bboxes[i, 0:4]
        dx = abs(x2 - x1) / 2.0
        dy = abs(y2 - y1) / 2.0
        current = dist((x1 + dx, y1 + dy), img_center)
        if current < distance:
            distance = current
            j = i

    return j


def bbox2point(left, right, top, bottom, type='bbox'):
    if type == 'kpt68':
        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    elif type == 'bbox':
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
    else:
        raise NotImplementedError
    return old_size, center


def mica_preprocess(img, app):
    # assumes img to be in range(0, 1)
    img = img.astype('float32') * 255.0
    bboxes, kpss = app.det_model.detect(img, max_num=0, metric='default')
    if len(bboxes) == 0:
        # Handling cases where box detector is failing over here
        arface_inp, _ = get_arcface_input_from_full_face(img)
        bbox = [0, 0, img.shape[0] - 1, img.shape[1] - 1]
    else:
        i = get_center(bboxes, img)
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = None
        if kpss is not None:
            kps = kpss[i]
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        arface_inp, _ = get_arcface_input(face, img)
    return torch.tensor(arface_inp), torch.tensor(bbox)


if __name__ == "__main__":
    from insightface.app import FaceAnalysis
    mica_face_detector = FaceAnalysis(name='antelopev2', providers=['CUDAExecutionProvider'])
    mica_face_detector.prepare(ctx_id=0, det_size=(224, 224))

    img = np.zeros((224, 224, 3))

    arcface_inp, bbx = mica_preprocess(img, mica_face_detector)