import os
import random
import shutil

# 设置源路径
# occluded_dataset_path = '/opt/data/private/yfz/PADE-main-single-3090/data/OccludedREID/query'
#
# # 检查目录是否存在
# if not os.path.exists(occluded_dataset_path):
#     print(f"目录不存在: {occluded_dataset_path}")
# else:
#     print(f"目录存在: {occluded_dataset_path}")
#
# # 遍历文件夹中的文件
# for img in os.listdir(occluded_dataset_path):
#     if img.endswith('.jpg'):  # 只处理以 .jpg 结尾的文件
#         # 获取文件前缀作为新的文件夹名
#         prefix = img.split('_')[0]
#
#         # 打印出文件前缀，帮助调试
#         print(f"处理文件: {img}, 前缀: {prefix}")
#
#         # 为每个不同的前缀创建一个子目录
#         prefix_dir = os.path.join(occluded_dataset_path, prefix)
#         if not os.path.exists(prefix_dir):
#             os.makedirs(prefix_dir)  # 如果目录不存在，则创建目录
#             print(f"创建目录: {prefix_dir}")
#
#         # 生成新的文件路径
#         src = os.path.join(occluded_dataset_path, img)
#         dst = os.path.join(prefix_dir, img)
#
#         # 移动文件到对应的文件夹中
#         shutil.move(src, dst)
#         print(f"移动文件: {src} -> {dst}")
#
# print("文件已成功重新组织！")
occluded_dataset_path = os.path.join('/opt/data/private/yfz/PADE-main-single-3090/data/OccludedREID', 'query')
whole_dataset_path = os.path.join('/opt/data/private/yfz/PADE-main-single-3090/data/OccludedREID', 'gallery')
new_dataset_path = r'/opt/data/private/yfz/PADE-main-single-3090/data/OccludedREID/new'
for files in os.listdir(occluded_dataset_path):
    for img in os.listdir(os.path.join(occluded_dataset_path, files)):
        new_name = img.split('.')[0] + '_01' + '.jpg'
        shutil.copy(os.path.join(occluded_dataset_path, files, img), os.path.join(new_dataset_path, new_name))

for files in os.listdir(whole_dataset_path):
    for img in os.listdir(os.path.join(whole_dataset_path, files)):
        new_name = img.split('.')[0] + '_02' + '.jpg'
        shutil.copy(os.path.join(whole_dataset_path, files, img), os.path.join(new_dataset_path, new_name))

dataset_path = '/opt/data/private/yfz/PADE-main-single-3090/data/OccludedREID/new'
imgs = os.listdir(dataset_path)
for i in range(10):
    new_dataset_path = '/opt/data/private/yfz/PADE-main-single-3090/data/OccludedREID/new' + str(i)
    pids = list(range(1, 201))
    train_pids = random.sample(pids, int(len(pids) / 2))
    os.makedirs(os.path.join('/opt/data/private/yfz/PADE-main-single-3090/data/OccludedREID/new' + str(i), 'bounding_box_train'))
    os.makedirs(os.path.join('/opt/data/private/yfz/PADE-main-single-3090/data/OccludedREID/new' + str(i), 'query'))
    os.makedirs(os.path.join('/opt/data/private/yfz/PADE-main-single-3090/data/OccludedREID/new' + str(i), 'bounding_box_test'))
    for img in imgs:
        pid, cid = img.split('_')[0], img.split('_')[2].split('.')[0]
        if int(pid) in train_pids:
            shutil.copy(os.path.join(dataset_path, img), os.path.join(new_dataset_path, 'bounding_box_train', img))
        else:
            if int(cid) == 1:
                shutil.copy(os.path.join(dataset_path, img), os.path.join(new_dataset_path, 'query', img))
            else:
                shutil.copy(os.path.join(dataset_path, img), os.path.join(new_dataset_path, 'bounding_box_test', img))