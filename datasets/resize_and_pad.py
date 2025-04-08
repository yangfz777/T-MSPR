# import cv2
# import numpy as np
# import random
#
# def resize_and_pad(image, target_size):
#     # 获取原始图像尺寸
#     h, w = image.shape[:2]
#     target_w, target_h = target_size
#
#     # 随机缩放比例在0.8到1.0之间
#     scale = random.uniform(0.8, 1.0)
#     new_w = int(w * scale)
#     new_h = int(h * scale)
#
#     # 缩放图像
#     resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#
#     # 计算填充大小
#     delta_w = target_w - new_w
#     delta_h = target_h - new_h
#     top, bottom = delta_h // 2, delta_h - (delta_h // 2)
#     left, right = delta_w // 2, delta_w - (delta_w // 2)
#
#     # 使用边缘信息进行填充
#     padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_REPLICATE)
#
#     return padded_image
#
# # 示例使用
# image = cv2.imread('/opt/data/private/yfz/PADE-main/mini-data/0013_c5_f0058070.jpg')
# target_size = (128, 256)
# output_image = resize_and_pad(image, target_size)
#
# # 使用 Matplotlib 显示图像
# import matplotlib.pyplot as plt
#
# plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
# plt.title('Resized and Padded Image')
# plt.axis('off')  # 隐藏坐标轴
# plt.show()
#
# # 保存图像
# cv2.imwrite('resized_and_padded_image.jpg', output_image)
# print("图像已保存到 resized_and_padded_image.jpg")
#
import random
import cv2
import numpy as np
from PIL import Image
import math

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

# Example usage
if __name__ == "__main__":
    from torchvision import transforms
    import matplotlib.pyplot as plt

    # Define the transformation pipeline
    transform = transforms.Compose([
        RandomResizeAndPad(target_size=(256,128), scale_range=(0.8, 1.0)),
        transforms.ToTensor(),
    ])

    # Load an image using PIL
    img = Image.open('/opt/data/private/yfz/PADE-main/mini-data/0013_c5_f0058070.jpg')

    # Apply the transformation
    img_transformed = transform(img)

    # Display the transformed image using matplotlib
    plt.imshow(img_transformed.permute(1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
    plt.title('Resized and Padded Image')
    plt.axis('off')  # Hide axis
    plt.show()