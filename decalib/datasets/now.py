import os
import torch
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision
from decalib.datasets import detectors
from decalib.utils import util

# Mica specific libraries
from insightface.app import FaceAnalysis
from mica_util import mica_preprocess


class NoWDataset(Dataset):
    def __init__(self, ring_elements=6, image_size=224, scale=1.6,
                 dataset_path="/media/exx/8TB1/hiresh/Capstone/NoW_Dataset", eval_set="val",
                 use_mica=False):
        if eval_set == "val":
            self.data_path = os.path.join(dataset_path, 'imagepathsvalidation.txt')
        else:
            self.data_path = os.path.join(dataset_path, 'imagepathstest.txt')

        with open(self.data_path) as f:
            self.data_lines = f.readlines()

        self.imagefolder = os.path.join(dataset_path, 'final_release_version', 'iphone_pictures')
        self.bbxfolder = os.path.join(dataset_path, 'final_release_version', 'detected_face')

        # self.data_path = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/test_image_paths_ring_6_elements.npy'
        # self.imagepath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/iphone_pictures/'
        # self.bbxpath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/detected_face/'
        self.image_size = image_size
        self.scale = scale

        self.use_mica = use_mica
        if self.use_mica:
            self.mica_face_detector = FaceAnalysis(name='antelopev2', root='insightface/', providers=['CUDAExecutionProvider'])
            self.mica_face_detector.prepare(ctx_id=0, det_size=(224, 224))

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, index):
        imagepath = os.path.join(self.imagefolder, self.data_lines[index].strip())  # + '.jpg'
        bbx_path = os.path.join(self.bbxfolder, self.data_lines[index].strip().replace('.jpg', '.npy'))
        bbx_data = np.load(bbx_path, allow_pickle=True, encoding='latin1').item()
        # box = np.array([[bbx_data['left'], bbx_data['top']], [bbx_data['right'], bbx_data['bottom']]]).astype('float32')
        left = bbx_data['left']
        right = bbx_data['right']
        top = bbx_data['top']
        bottom = bbx_data['bottom']

        image = imread(imagepath)[:, :, :3]

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * self.scale)

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image / 255.
        dst_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        dst_image = dst_image.transpose(2, 0, 1)

        data_dict = {'image': torch.tensor(dst_image).float(),
                     'imagename': self.data_lines[index].strip().replace('.jpg', ''),
                     'imagepath': imagepath
                     # 'tform': tform,
                     # 'original_image': torch.tensor(image.transpose(2, 0, 1)).float(),
                     }
        if self.use_mica:
            arcface_inp, bbox = mica_preprocess(image.copy(), self.mica_face_detector)
            data_dict['arcface_inp'] = arcface_inp

        kpt_path = imagepath.replace(".jpg", "_landmarks.txt")
        if os.path.exists(kpt_path):
            kpt = np.loadtxt(kpt_path)
            cropped_kpt = np.dot(tform.params,
                                 np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1
            kpt_array = torch.from_numpy(cropped_kpt).type(dtype=torch.float32)
            data_dict['landmark'] = kpt_array
        else:
            print("Keypoint not found:", imagepath)

        return data_dict


def plot_tensor(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

def save_tensor_to_path(img, path):
    npimg = img.cpu().numpy()
    rgb_img = np.transpose(npimg, (1, 2, 0))
    imsave(path, rgb_img)


if __name__ == "__main__":
    scale_min, scale_max = 1.4, 1.8
    now_path = "/home/secret-user/now_benchmark_data/NoW_Dataset"
    dataset = NoWDataset(scale=(scale_min + scale_max) / 2, use_mica=True, dataset_path=now_path)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=False)

    tmp_fold = "./tmp"
    os.makedirs(tmp_fold, exist_ok=True)

    for i, data in enumerate(dataloader):
        images = data['image']
        print(data['image'].shape)
        grid_img = torchvision.utils.make_grid(data['image'], nrow=2)
        plot_tensor(grid_img)
        plt.savefig(f"{tmp_fold}/now_transformed_{i}.jpg")
        landmarks = data['landmark']
        landmark_img_tensor = util.tensor_vis_landmarks(images, landmarks, gt_landmarks=landmarks)
        grid_img = torchvision.utils.make_grid(landmark_img_tensor, nrow=2)
        save_tensor_to_path(grid_img, f"{tmp_fold}/now_lmk_{i}.jpg")
        break

'''
import os, sys
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class NoWDataset(Dataset):
    def __init__(self, ring_elements=6, crop_size=224, scale=1.6):
        folder = 'Now_Dataset'
        self.data_path = os.path.join(folder, 'imagepathsvalidation.txt')
        with open(self.data_path) as f:
            self.data_lines = f.readlines()

        self.imagefolder = os.path.join(folder, 'final_release_version', 'iphone_pictures')
        self.bbxfolder = os.path.join(folder, 'final_release_version', 'detected_face')

        # self.data_path = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/test_image_paths_ring_6_elements.npy'
        # self.imagepath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/iphone_pictures/'
        # self.bbxpath = '/ps/scratch/face2d3d/ringnetpp/eccv/test_data/evaluation/NoW_Dataset/final_release_version/detected_face/'
        self.crop_size = crop_size
        self.scale = scale
            
    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, index):
        imagepath = os.path.join(self.imagefolder, self.data_lines[index].strip()) #+ '.jpg'
        bbx_path = os.path.join(self.bbxfolder, self.data_lines[index].strip().replace('.jpg', '.npy'))
        bbx_data = np.load(bbx_path, allow_pickle=True, encoding='latin1').item()
        # box = np.array([[bbx_data['left'], bbx_data['top']], [bbx_data['right'], bbx_data['bottom']]]).astype('float32')
        left = bbx_data['left']; right = bbx_data['right']
        top = bbx_data['top']; bottom = bbx_data['bottom']

        imagename = imagepath.split('/')[-1].split('.')[0]
        image = imread(imagepath)[:,:,:3]

        h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        size = int(old_size*self.scale)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.crop_size - 1], [self.crop_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.
        dst_image = warp(image, tform.inverse, output_shape=(self.crop_size, self.crop_size))
        dst_image = dst_image.transpose(2,0,1)
        return {'image': torch.tensor(dst_image).float(),
                'imagename': self.data_lines[index].strip().replace('.jpg', ''),
                # 'tform': tform,
                # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }
                '''