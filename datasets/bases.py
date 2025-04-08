from PIL import Image, ImageFile
import cv2
from torch.utils.data import Dataset
from .preprocessing import generate_occlusion_mask
import os.path as osp
import random
import torch
import os
from utils.transforms import get_affine_transform
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, crop_transform=None, eraser_transform=None,eraser_transform1=None):
        self.dataset = dataset
        self.transform = transform
        self.crop_transform = crop_transform
        self.eraser_transform = eraser_transform
        self.eraser_transform1 = eraser_transform1
        self.input_size = [512, 512]
        self.aspect_ratio = self.input_size[1] * 1.0 / self.input_size[0]
        self.input_size = np.asarray(self.input_size)
    def __len__(self):
        return len(self.dataset)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w, h], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img_name = os.path.basename(img_path)
        img = read_image(img_path)
        img_pro = np.array(img)

        if self.transform is not None:
            img1 = self.transform(img)

            if self.crop_transform is not None and self.eraser_transform is not None and self.eraser_transform1 is not None:
                img2 = self.crop_transform(img)
                img3 = self.eraser_transform(img)
                img4= self.eraser_transform1(img)



                return img1, img2, img3, img4, pid, camid, trackid, img_path.split('/')[-1]
            else:
                return img1, img1, img1, img1, pid, camid, trackid, img_path.split('/')[-1]
# class ImageDataset(Dataset):
#     def __init__(self, dataset, transform=None, crop_transform=None, eraser_transform=None):
#         self.dataset = dataset
#         self.transform = transform
#         self.crop_transform = crop_transform
#         self.eraser_transform = eraser_transform
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, index):
#         img_path, pid, camid, trackid = self.dataset[index]
#         img = read_image(img_path)
#
#         if self.transform is not None:
#             img1 = self.transform(img)
#             if self.crop_transform is not None and self.eraser_transform is not None:
#                 img2 = self.crop_transform(img)
#                 img3 = self.eraser_transform(img)
#                 return img1, img2, img3, pid, camid, trackid,img_path.split('/')[-1]
#             else:
#                 return img1, img1, img1, pid, camid, trackid,img_path.split('/')[-1]