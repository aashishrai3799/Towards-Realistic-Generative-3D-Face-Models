import imp
import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
from decalib.utils import util

# Mica specific libraries

from insightface.app import FaceAnalysis


class CelebAHQDataset(Dataset):
    def __init__(self, image_size=224, scale=1.4, trans_scale=0, isEval=False,
                 data_dir="/media/exx/8TB1/ayush/training_data/CelebA/aligned_dataset/img_align_celeba", use_mica=False):
        '''
        # 53877 faces
        K must be less than 6
        '''
        self.image_size = image_size
        self.imagefolder = f'{data_dir}/images'
        self.kptfolder = f'{data_dir}/landmarks'
        self.segfolder = f'{data_dir}/mask'

        self.kptpath_list = os.listdir(self.kptfolder)
        self.scale = scale  # [scale_min, scale_max]
        self.trans_scale = trans_scale  # 0.5?
        self.use_mica = use_mica
        if self.use_mica:
            self.mica_face_detector = FaceAnalysis(name='antelopev2', root='insightface/', providers=['CUDAExecutionProvider'])
            self.mica_face_detector.prepare(ctx_id=0, det_size=(224, 224))

    def __len__(self):
        return len(self.kptpath_list)

    def __getitem__(self, idx):
        while (100):
            kptname = self.kptpath_list[idx]
            name = kptname.split('.')[0]
            image_path = os.path.join(self.imagefolder, name + '.png')
            kpt_path = os.path.join(self.kptfolder, kptname)
            seg_path = (os.path.join(self.segfolder, name + '.npy'))

            kpt = np.loadtxt(kpt_path)
            if len(kpt.shape) != 2:
                idx = np.random.randint(low=0, high=len(self.kptpath_list))
                continue
            # print(kpt_path, kpt.shape)
            # kpt = kpt[:,:2]

            image = imread(image_path) / 255.
            if image is None:
                print("path:", image_path)
            if len(image.shape) < 3:
                image = np.tile(image[:, :, None], 3)

            mask = self.load_mask(seg_path, self.image_size, self.image_size)

            ### crop information
            tform = self.crop(image, kpt)
            ## crop
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params,
                                 np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T  # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.image_size * 2 - 1
            # cropped_kpt[:, 1] = -cropped_kpt[:, 1]

            ###
            images_array = torch.from_numpy(cropped_image.transpose(2, 0, 1)).type(dtype=torch.float32)
            kpt_array = torch.from_numpy(cropped_kpt).type(dtype=torch.float32)
            mask_array = torch.from_numpy(mask).unsqueeze(0).type(dtype=torch.float32)
            data_dict = {
                'image': images_array,
                'landmark': kpt_array,
                'mask': mask_array
            }
            if image is None:
                print("**image_path:", image_path)

            return data_dict

    def crop(self, image, kpt):
        left = np.min(kpt[:, 0])
        right = np.max(kpt[:, 0])
        top = np.min(kpt[:, 1])
        bottom = np.max(kpt[:, 1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])
        trans_scale = (np.random.rand(2) * 2 - 1) * self.trans_scale
        center = center + trans_scale * old_size  # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size * scale)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform

    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno > 0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask


def plot_tensor(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


def save_tensor_to_path(img, path):
    npimg = img.cpu().numpy()
    rgb_img = np.transpose(npimg, (1, 2, 0))
    imsave(path, rgb_img)


if __name__ == "__main__":
    scale_min, scale_max = 1.4, 1.8
    celebA_local = "/home/secret-user/Capstone/celebA_mini/img_align_celeba"
    dataset = CelebAHQDataset(image_size=224, scale=(scale_min, scale_max), data_dir=celebA_local, use_mica=True)

    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    tmp_fold = "./tmp2"
    os.makedirs(tmp_fold, exist_ok=True)
    for i, data in enumerate(dataloader):
        images = data['image']
        landmarks = data['landmark']
        print(data['image'].shape, data['mask'].shape, data['landmark'].shape, data['arcface_inp'].shape)
        grid_img = torchvision.utils.make_grid(images, nrow=2)
        plot_tensor(grid_img)
        plt.savefig(f"{tmp_fold}/celebA_orig_{i}.jpg")

        landmark_img_tensor = util.tensor_vis_landmarks(images, landmarks, gt_landmarks=landmarks)

        grid_img = torchvision.utils.make_grid(landmark_img_tensor, nrow=2)

        save_tensor_to_path(grid_img, f"{tmp_fold}/celeA_lmk_{i}.jpg")
