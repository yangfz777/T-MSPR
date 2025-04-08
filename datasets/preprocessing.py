import random
import math
import os
from PIL import Image
import torchvision.transforms as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class RandomErasing_Background(object):
    def __init__(self, EPSILON=0.5, root=None):
        self.EPSILON = EPSILON
        self.root = root
        self.occ_imgs = os.listdir(self.root)

        for img in self.occ_imgs:
            if not img.endswith('.jpg'):
                self.occ_imgs.remove(img)

        self.len = len(self.occ_imgs)

    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        index = random.randint(0, self.len - 1)

        occ_img = self.occ_imgs[index]
        occ_img = Image.open(os.path.join(self.root, occ_img)).convert('RGB')

        h, w = img.size()[1], img.size()[2]
        h_, w_ = occ_img.height, occ_img.width

        ratio = h_ / w_
        if ratio > 2:
            # re_size = (random.randint(h//2, h), random.randint(w//4, w//2))
            re_size = (h, random.randint(w // 4, w // 2))
            function = T.Compose([
                T.Resize(re_size, interpolation=3),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ])
            occ_img = function(occ_img)
        else:
            # re_size = (random.randint(h//4, h//2), random.randint(w//2, w))
            re_size = (random.randint(h // 4, h // 2), w)
            function = T.Compose([
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
                T.Resize(re_size, interpolation=3),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ])
            occ_img = function(occ_img)
        # self.save_occ_img(occ_img)
        h_, w_ = re_size[0], re_size[1]

        index_ = random.randint(0, 3)
        # points = [(0, 0), (0, w), (h, 0), (h, w)]
        top = random.randint(0, h - h_)
        left = random.randint(0, w - w_)

        # 覆盖图像
        img[:, top:top + h_, left:left + w_] = occ_img

        self.save_occ_img(img)
        return img

    # def save_occ_img(self, occ_img):
    #         # 将张量转换回PIL图像进行保存
    #         self.save_path = "/opt/data/private/yfz/PADE/logs_occ_duke/temp2"
    #         occ_img = T.ToPILImage()(occ_img)
    #         if not os.path.exists(self.save_path):
    #             os.makedirs(self.save_path)
    #         occ_img_save_path = os.path.join(self.save_path, f"occ_img_{random.randint(0, 99999)}.jpg")
    #         occ_img.save(occ_img_save_path)
        # if index_ == 0:
        #     img[:, 0:h_, 0:w_] = occ_img
        # elif index_ == 1:
        #     img[:, 0:h_, w - w_:w] = occ_img
        # elif index_ == 2:
        #     img[:, h - h_:h, 0:w_] = occ_img
        # else:
        #     img[:, h - h_:h, w - w_:w] = occ_img
        #
        # return img


class AddLocalSaltNoise(object):
    def __init__(self, prob=0.01, area_ratio=0.1):
        self.prob = prob
        self.area_ratio = area_ratio

    def __call__(self, tensor):
        c, h, w = tensor.shape
        area = h * w
        noise_area = int(area * self.area_ratio)

        noise_h = int(noise_area ** 0.5)
        noise_w = int(noise_area ** 0.5)

        if noise_h > h or noise_w > w:
            noise_h, noise_w = h, w

        top = random.randint(0, h - noise_h)
        left = random.randint(0, w - noise_w)

        num_salt = int(self.prob * noise_h * noise_w)
        for _ in range(num_salt):
            y = random.randint(top, top + noise_h - 1)
            x = random.randint(left, left + noise_w - 1)
            tensor[:, y, x] = 1

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(prob={0}, area_ratio={1})'.format(self.prob, self.area_ratio)


class RandomErasing_Backgroundpro1(object):
    def __init__(self, EPSILON=0.5, root=None):
        self.EPSILON = EPSILON
        self.root = root
        # 只保留.jpg格式的图片
        self.occ_imgs = [img for img in os.listdir(self.root) if img.endswith('.jpg')]
        self.len = len(self.occ_imgs)

    def __call__(self, img):
        # 随机决定是否应用遮挡
        if random.uniform(0, 1) > self.EPSILON:
            return img

        # 随机选择一个遮挡图片
        index = random.randint(0, self.len - 1)
        occ_img = self.occ_imgs[index]
        occ_img = Image.open(os.path.join(self.root, occ_img)).convert('RGB')

        h, w = img.size()[1], img.size()[2]  # 假设img是格式为(C, H, W)的张量
        occ_h, occ_w = occ_img.height, occ_img.width

        # 将遮挡图片和目标图片划分为4个水平区域
        img_regions = [
            (0, h // 4), (h // 4, h // 2), (h // 2, 3 * h // 4), (3 * h // 4, h)
        ]
        occ_regions = [
            (0, occ_h // 4), (occ_h // 4, occ_h // 2), (occ_h // 2, 3 * occ_h // 4), (3 * occ_h // 4, occ_h)
        ]

        # 按照不同概率选择遮挡方式
        prob = random.uniform(0, 1)
        if prob <= 0.7:
            # 0.7 的概率用遮挡图片的 2、3 区域覆盖目标图片的 3、4 区域
            occ_region_indices = [1, 2]
            img_region_indices = [2, 3]
        elif prob <= 0.8:
            # 0.1 的概率用遮挡图片的 2 区域覆盖目标图片的 4 区域
            occ_region_indices = [1]
            img_region_indices = [3]
        elif prob <= 0.9:
            # 0.1 的概率用遮挡图片的 1、2 区域覆盖目标图片的 3、4 区域
            occ_region_indices = [0, 1]
            img_region_indices = [2, 3]
        else:
            # 0.1 的概率用遮挡图片的 3 区域覆盖目标图片的 1 区域
            occ_region_indices = [2]
            img_region_indices = [0]

        # 遍历选择的区域并执行遮挡
        for occ_idx, img_idx in zip(occ_region_indices, img_region_indices):
            occ_top, occ_bottom = occ_regions[occ_idx]
            img_top, img_bottom = img_regions[img_idx]

            # 裁剪出遮挡区域
            occ_crop = occ_img.crop((0, occ_top, occ_w, occ_bottom))

            # 调整遮挡图片大小以适应目标图像对应区域
            function = T.Compose([
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
                T.Resize((img_bottom - img_top, w), interpolation=3),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ])
            occ_crop = function(occ_crop)

            # 将裁剪区域覆盖到原图的相应位置
            img[:, img_top:img_bottom, :] = occ_crop

        return img
class RandomErasing_Backgroundpro(object):
    def __init__(self, EPSILON=0.5, root=None):
        self.EPSILON = EPSILON
        self.root = root
        # 只保留.jpg格式的图片
        self.occ_imgs = [img for img in os.listdir(self.root) if img.endswith('.jpg')]
        self.len = len(self.occ_imgs)

    def __call__(self, img):
        # 随机决定是否应用遮挡
        if random.uniform(0, 1) > self.EPSILON:
            return img

        # 随机选择一个遮挡图片
        index = random.randint(0, self.len - 1)
        occ_img = self.occ_imgs[index]
        occ_img = Image.open(os.path.join(self.root, occ_img)).convert('RGB')

        h, w = img.size()[1], img.size()[2]  # 假设img是格式为(C, H, W)的张量
        occ_h, occ_w = occ_img.height, occ_img.width

        # 随机选择遮挡图片的一部分
        crop_h = random.randint(occ_h // 4, occ_h // 2)
        crop_w = random.randint(occ_w // 4, occ_w // 2)
        crop_top = random.randint(0, occ_h - crop_h)
        crop_left = random.randint(0, occ_w - crop_w)

        occ_img = occ_img.crop((crop_left, crop_top, crop_left + crop_w, crop_top + crop_h))

        # 调整遮挡图片大小以适应目标图像
        re_size = (crop_h, crop_w)
        if crop_h / crop_w > 2:
            re_size = (h, random.randint(w // 4, w // 2))
        else:
            re_size = (random.randint(h // 4, h // 2), w)

        function = T.Compose([
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
            T.Resize(re_size, interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
        ])
        occ_img = function(occ_img)

        h_, w_ = re_size[0], re_size[1]

        # 保存遮挡图片
        # self.save_occ_img(occ_img)

        # 随机确定遮挡的位置
        top = random.randint(0, h - h_)
        left = random.randint(0, w - w_)

        # 将遮挡图片覆盖到原图上
        img[:, top:top + h_, left:left + w_] = occ_img
        # 保存遮挡图片
        # self.save_occ_img(img)
        return img


    # def visualize_occ_img(self, occ_img):
    #     # 将张量转换回PIL图像进行可视化
    #     occ_img = T.ToPILImage()(occ_img)
    #     plt.imshow(occ_img)
    #     plt.axis('off')  # 关闭坐标轴
    #     plt.show()
    # def save_occ_img(self, occ_img):
    #     # 将张量转换回PIL图像进行保存
    #     self.save_path = "/opt/data/private/yfz/PADE/logs_occ_duke/temp"
    #     occ_img = T.ToPILImage()(occ_img)
    #     if not os.path.exists(self.save_path):
    #         os.makedirs(self.save_path)
    #     occ_img_save_path = os.path.join(self.save_path, f"occ_img_{random.randint(0, 99999)}.jpg")
    #     occ_img.save(occ_img_save_path)


class RandomZoom(object):
    def __init__(self, EPSILON=0.5, root=None):
        self.EPSILON = EPSILON
        self.root = root
        self.occ_imgs = os.listdir(self.root)

        for img in self.occ_imgs:
            if not img.endswith('.jpg'):
                self.occ_imgs.remove(img)

        self.len = len(self.occ_imgs)

    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        index = random.randint(0, self.len - 1)

        occ_img = self.occ_imgs[index]
        occ_img = Image.open(os.path.join(self.root, occ_img)).convert('RGB')
        # print("img的尺寸", img.size)
        # h, w = img.size()[1], img.size()[2]
        # h_, w_ = occ_img.height, occ_img.width
        # print('h_*****w_', h_, w_)
        # print('h******w', h, w)


        # re_size = (random.randint(h//4, h//2), random.randint(w//2, w))
        re_size = (256,128)
        function = T.Compose([
                T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
                T.Resize([256, 128], interpolation=3),
                T.ToTensor(),
            ])
        a = random.uniform(0.8, 1.0)
        new_width = int(a * 256)
        new_height = int(a * 128)

        # 使用双三次插值法进行缩放
        img = img.resize((new_height, new_width), resample=Image.BICUBIC)
        # print("新的img的尺寸", img.size)

        function1 = T.Compose([

            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        occ_img = function(occ_img)
        img = function1(img)
        # self.save_occ_img(occ_img)
        # print("img的shape", img.shape)
        # exit()
        h, w = img.size()[1], img.size()[2]
        h_, w_ = re_size[0], re_size[1]

        # index_ = random.randint(0, 3)
        # points = [(0, 0), (0, w), (h, 0), (h, w)]
        top = random.randint(0, h_ - h)
        left = random.randint(0, w_ - w)

        # 覆盖图像
        occ_img[:, top:top + h, left:left + w] = img

        # self.save_occ_img(img)
        return occ_img


class RandomResizeAndPad(object):
    def __init__(self, target_size, scale_range=(0.8, 1.0)):
        """
        Args:
            target_size (tuple): Desired output size (height, width).
            scale_range (tuple): Range of scaling factor.
        """
        self.target_size = target_size
        self.scale_range = scale_range

    def __call__(self, img):
        # Convert PIL image to numpy array
        img = np.array(img)

        # Get original image size
        h, w = img.shape[:2]

        # Random scale factor
        scale = random.uniform(*self.scale_range)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image while maintaining aspect ratio
        resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Calculate padding to reach target size
        target_h, target_w = self.target_size
        delta_w = max(0, target_w - new_w)
        delta_h = max(0, target_h - new_h)
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # Pad the resized image with edge replication
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_REPLICATE)

        # Convert back to PIL image
        return Image.fromarray(padded_image)


import cv2
import numpy as np


def generate_occlusion_mask(full_image, occluded_image, threshold=30):
    """
    生成遮挡掩码，标记遮挡图片中的遮挡区域。

    参数:
    - full_image: 完整的行人图片 (无遮挡).
    - occluded_image: 带有遮挡的图片.
    - threshold: 像素差异的阈值, 超过该值则认为是遮挡区域.

    返回:
    - mask: 遮挡掩码，1 表示遮挡区域，0 表示非遮挡区域.
    """

    # 如果输入是 PyTorch Tensor，需要转换为 NumPy 数组
    if isinstance(full_image, torch.Tensor):
        full_image = full_image.cpu().numpy().transpose(1, 2, 0)  # 转换为 NumPy 数组，并调整维度顺序
    if isinstance(occluded_image, torch.Tensor):
        occluded_image = occluded_image.cpu().numpy().transpose(1, 2, 0)

    # 将图像转换为灰度图，减少计算复杂度
    gray_full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
    gray_occluded_image = cv2.cvtColor(occluded_image, cv2.COLOR_BGR2GRAY)

    # 计算两张图像的绝对差异
    diff = cv2.absdiff(gray_full_image, gray_occluded_image)

    # 根据阈值生成掩码，超过阈值的区域认为是遮挡区域
    _, mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)

    return mask

# 加载完整图片和遮挡图片
# full_image_path = 'full_image.jpg'  # 完整行人图片路径
# occluded_image_path = 'occluded_image.jpg'  # 带有遮挡的图片路径
#
# full_image = cv2.imread(full_image_path)
# occluded_image = cv2.imread(occluded_image_path)
#
# # 生成遮挡掩码
# mask = generate_occlusion_mask(full_image, occluded_image)
#
# # 可视化遮挡掩码 (将其放大到 [0, 255] 范围以便显示)
# mask_visual = mask * 255
# cv2.imshow('Occlusion Mask', mask_visual)
#
# # 保存遮挡掩码
# cv2.imwrite('occlusion_mask.png', mask_visual)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import random
# import math
# import os
# from PIL import Image
# import torchvision.transforms as T
# import torch
# class RandomErasing(object):
#     """ Randomly selects a rectangle region in an image and erases its pixels.
#         'Random Erasing Data Augmentation' by Zhong et al.
#         See https://arxiv.org/pdf/1708.04896.pdf
#     Args:
#          probability: The probability that the Random Erasing operation will be performed.
#          sl: Minimum proportion of erased area against input image.
#          sh: Maximum proportion of erased area against input image.
#          r1: Minimum aspect ratio of erased area.
#          mean: Erasing value.
#     """
#
#     def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
#         self.probability = probability
#         self.mean = mean
#         self.sl = sl
#         self.sh = sh
#         self.r1 = r1
#
#     def __call__(self, img):
#
#         if random.uniform(0, 1) >= self.probability:
#             return img
#
#         for attempt in range(100):
#             area = img.size()[1] * img.size()[2]
#
#             target_area = random.uniform(self.sl, self.sh) * area
#             aspect_ratio = random.uniform(self.r1, 1 / self.r1)
#
#             h = int(round(math.sqrt(target_area * aspect_ratio)))
#             w = int(round(math.sqrt(target_area / aspect_ratio)))
#
#             if w < img.size()[2] and h < img.size()[1]:
#                 x1 = random.randint(0, img.size()[1] - h)
#                 y1 = random.randint(0, img.size()[2] - w)
#                 if img.size()[0] == 3:
#                     img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
#                     img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
#                     img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
#                 else:
#                     img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
#                 return img
#
#         return img
#
#
# class RandomErasing_Background(object):
#     def __init__(self, EPSILON=0.5, root=None):
#         self.EPSILON = EPSILON
#         self.root = root
#         self.occ_imgs = os.listdir(self.root)
#
#         for img in self.occ_imgs:
#             if not img.endswith('.jpg'):
#                 self.occ_imgs.remove(img)
#
#         self.len = len(self.occ_imgs)
#
#     def __call__(self, img):
#
#         if random.uniform(0, 1) > self.EPSILON:
#             return img
#
#         index = random.randint(0, self.len - 1)
#
#         occ_img = self.occ_imgs[index]
#         occ_img = Image.open(os.path.join(self.root, occ_img)).convert('RGB')
#
#         h, w = img.size()[1], img.size()[2]
#         h_, w_ = occ_img.height, occ_img.width
#
#         ratio = h_ / w_
#         if ratio > 2:
#             # re_size = (random.randint(h//2, h), random.randint(w//4, w//2))
#             re_size = (h, random.randint(w // 4, w // 2))
#             function = T.Compose([
#                 T.Resize(re_size, interpolation=3),
#                 T.RandomHorizontalFlip(p=0.5),
#                 T.ToTensor(),
#             ])
#             occ_img = function(occ_img)
#         else:
#             # re_size = (random.randint(h//4, h//2), random.randint(w//2, w))
#             re_size = (random.randint(h // 4, h // 2), w)
#             function = T.Compose([
#                 T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
#                 T.Resize(re_size, interpolation=3),
#                 T.RandomHorizontalFlip(p=0.5),
#                 T.ToTensor(),
#             ])
#             occ_img = function(occ_img)
#
#         h_, w_ = re_size[0], re_size[1]
#
#         index_ = random.randint(0, 3)
#         # points = [(0, 0), (0, w), (h, 0), (h, w)]
#         top = random.randint(0, h - h_)
#         left = random.randint(0, w - w_)
#
#         # 覆盖图像
#         img[:, top:top + h_, left:left + w_] = occ_img
#
#         return img
#         # if index_ == 0:
#         #     img[:, 0:h_, 0:w_] = occ_img
#         # elif index_ == 1:
#         #     img[:, 0:h_, w - w_:w] = occ_img
#         # elif index_ == 2:
#         #     img[:, h - h_:h, 0:w_] = occ_img
#         # else:
#         #     img[:, h - h_:h, w - w_:w] = occ_img
#         #
#         # return img
#
#
# class AddLocalSaltNoise(object):
#     def __init__(self, prob=0.01, area_ratio=0.1):
#         self.prob = prob
#         self.area_ratio = area_ratio
#
#     def __call__(self, tensor):
#         c, h, w = tensor.shape
#         area = h * w
#         noise_area = int(area * self.area_ratio)
#
#         noise_h = int(noise_area ** 0.5)
#         noise_w = int(noise_area ** 0.5)
#
#         if noise_h > h or noise_w > w:
#             noise_h, noise_w = h, w
#
#         top = random.randint(0, h - noise_h)
#         left = random.randint(0, w - noise_w)
#
#         num_salt = int(self.prob * noise_h * noise_w)
#         for _ in range(num_salt):
#             y = random.randint(top, top + noise_h - 1)
#             x = random.randint(left, left + noise_w - 1)
#             tensor[:, y, x] = 1
#
#         return tensor
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(prob={0}, area_ratio={1})'.format(self.prob, self.area_ratio)