# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn.functional as F
import numpy as np
import cv2
import torch

class BRG2Tensor_transform(object):
    def __call__(self, pic):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

class BGR2RGB_transform(object):
    def __call__(self, tensor):
        return tensor[[2,1,0],:,:]

def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, input_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def transform_parsing(pred, center, scale, width, height, input_size):

    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    target_pred = cv2.warpAffine(
            pred,
            trans,
            (int(width), int(height)), #(int(width), int(height)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0))

    return target_pred

def transform_logits(logits, center, scale, width, height, input_size):

    trans = get_affine_transform(center, scale, 0, input_size, inv=1)
    channel = logits.shape[2]
    target_logits = []
    for i in range(channel):
        target_logit = cv2.warpAffine(
            logits[:,:,i],
            trans,
            (int(width), int(height)), #(int(width), int(height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0))
        target_logits.append(target_logit)
    target_logits = np.stack(target_logits,axis=2)

    return target_logits
def resize_feature_map(x, output_size=(22, 11), return_numpy=False):
    # 检查x是否为NumPy数组并将其转换为PyTorch张量
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    # 确保x的格式是 [B, H, W, C]
    if x.dim() == 4:
        x = x.permute(0, 3, 1, 2)  # 转换为 [B, C, H, W]

    # 使用双线性插值进行缩放
    x_resized = F.interpolate(x, size=output_size, mode='bilinear', align_corners=False)

    # 将结果转换回 [B, H, W, C] 格式
    x_resized = x_resized.permute(0, 2, 3, 1)  # 转换为 [B, H, W, C]

    # 根据参数决定返回 NumPy 数组还是 PyTorch 张量
    if return_numpy:
        return x_resized.cpu().numpy()
    return x_resized

def transform_logits_batch(logits_batch, center, scale, width, height, input_size):
    # logits_batch: (N, H, W, C), N是batch大小
    batch_size = logits_batch.shape[0]
    channel = logits_batch.shape[3]
    transformed_batch = []

    # 遍历批次中的每个样本
    for n in range(batch_size):
        logits = logits_batch[n]
        # 将当前样本的 center 和 scale 传入 get_affine_transform
        trans = get_affine_transform(center[n], scale[n], 0, input_size, inv=1)
        target_logits = []

        # 对每个通道进行仿射变换
        for i in range(channel):
            target_logit = cv2.warpAffine(
                logits[:, :, i],
                trans,
                (85, 220),  # 统一为 input_size 的宽高
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0))
            target_logits.append(target_logit)

        # 将变换后的通道堆叠起来
        target_logits = np.stack(target_logits, axis=2)
        transformed_batch.append(target_logits)

    # 将批次中的每个样本堆叠为一个新的四维张量
    transformed_batch = np.stack(transformed_batch, axis=0)

    return transformed_batch




def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w-1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w-1) * 0.5, (dst_h-1) * 0.5]
    dst[1, :] = np.array([(dst_w-1) * 0.5, (dst_h-1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[1]), int(output_size[0])),
                             flags=cv2.INTER_LINEAR)

    return dst_img
