import logging
import os
import time
import torch
import numpy as np
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
from PIL import Image
import torch.nn.functional as F
from torch.nn.functional import interpolate
import torch.distributed as dist
from utils.transforms import transform_logits, transform_logits_batch, resize_feature_map
import argparse
from collections import OrderedDict
from loss.triplet_loss import TripletLoss
from loss.MSEloss import compute_multichannel_loss
import pickle
import matplotlib.pyplot as plt
from datasets.occ_duke import OCC_DukeMTMCreID
__factory = {

    'occluded_dukemtmc': OCC_DukeMTMCreID,

}
dataset_settings = {
    'lip': {
        'input_size': [473, 473],
        'num_classes': 20,
        'label': ['Background', 'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 'Coat',
                  'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm', 'Right-arm',
                  'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']
        # 'num_classes': 5,
        # 'label': ['Background', 'head', 'body', 'legs', 'feet']
    },
    'lip_merged': {
        'input_size': [473, 473],
        'num_classes': 6,  # 合并后的类别数
        'label': ['Background', 'Head', 'Body', 'Accessory', 'Legs', 'Feet'],
        'merge_map': {  # 定义原始类别到合并后类别的映射,背景信息没有使用
            0: 0,  # Background -> Background
            1: 3, 2: 1, 3: 3, 4: 3, 5: 2, 6: 2, 7: 2,
            8: 5, 9: 4, 10: 2, 11: 3, 12: 4, 13: 1,
            14: 2, 15: 2, 16: 4, 17: 4, 18: 5, 19: 5
        }
    },
    'atr': {
        'input_size': [512, 512],
        'num_classes': 18,
        'label': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf']
    },
    'pascal': {
        'input_size': [512, 512],
        'num_classes': 7,
        'label': ['Background', 'Head', 'Torso', 'Upper Arms', 'Lower Arms', 'Upper Legs', 'Lower Legs'],
    }
}

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette



def model_size(model):
    param_size = sum(param.numel() for param in model.parameters()) * 4  # 每个浮点数占4字节
    return param_size / (1024 ** 2)
def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):

    num_classes = dataset_settings[cfg.dataset]['num_classes']
    input_size = dataset_settings[cfg.dataset]['input_size']
    label = dataset_settings[cfg.dataset]['label']
    merge_map = dataset_settings[cfg.dataset].get('merge_map', None)  # 获取合并映射

    print("Evaluating total class number {} with {}".format(num_classes, label))

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    top = [0, 0, 0]
    linear_layer = nn.Linear(768, 6).to('cuda')
    triplet = TripletLoss()
    mse_loss = nn.MSELoss()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()  # 记录当前时间以计算训练时间
        loss_meter.reset()  # 重置损失计量器
        acc_meter.reset()  # 重置准确率计量器
        evaluator.reset()  # 重置评估器
        scheduler.step(epoch)  # 更新学习率调度器
        model.train()  # 将模型设置为训练模式


        for n_iter, (img1, img2, img3, img4, vid, target_cam, target_view, img_path) in enumerate(train_loader):
            img_path_list = []
            optimizer.zero_grad()  # 清除优化器的梯度
            optimizer_center.zero_grad()  # 清除中心优化器的梯度

            # 将输入数据转移到设备（GPU或CPU）
            img1, img2, img3, img4, target, target_cam, target_view = [x.to(device) for x in
                                                            [img1, img2, img3, img4, vid, target_cam, target_view]]
            img_path_list.extend(img_path)
            # 存放 .npy 文件的文件夹路径
            npy_folder = "/opt/data/private/yfz/PADE-main-single-3090/data/new_lablepro"

            stacked_array = []
            for image_name in img_path_list:
                # 替换文件扩展名为 .npy
                npy_name = os.path.splitext(image_name)[0] + '.npy'
                npy_path = os.path.join(npy_folder, npy_name)

                if os.path.exists(npy_path):
                    # 加载 .npy 文件
                    data = np.load(npy_path)  # 每个文件形状为 [1, 22, 11, 6]
                    if data.shape[0] == 1:  # 确保是 [1, 22, 11, 6]，去掉第一个维度
                        data = data[0]  # 去掉第0维，变成 [22, 11, 6]
                    stacked_array.append(data)

            # 将所有数组堆叠到一个大数组中（沿第0维堆叠）
            if stacked_array:
                result_array = np.stack(stacked_array, axis=0)
            else:
                print("未找到任何匹配的 .npy 文件")
            # 使用混合精度训练
            with amp.autocast(enabled=True):
                score, feat = model(img1, img2, img3, img4, target,  cam_label=target_cam, view_label=target_view)
                if isinstance(result_array, np.ndarray):
                    result_array = torch.tensor(result_array).to('cuda')
                l1_loss = torch.abs(feat[2] - feat[3]).mean(dim=2)
                l1_loss = l1_loss.mean()
                loss1 = 0.25 * compute_multichannel_loss(feat[1],result_array,loss_type="mse")
                loss =  loss_fn(score, feat, target, target_cam) + loss1 +   l1_loss
            scaler.scale(loss).backward()  # 使用混合精度反向传播
            scaler.step(optimizer)  # 更新优化器
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img1.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))


        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if (epoch % eval_period == 0):
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img1, img2, img3, img4, vid, camid, camids, target_view ,img_path  ) in enumerate(val_loader):
                        with torch.no_grad():
                            img1 = img1.to(device)
                            img2 = img2.to(device)
                            img3 = img3.to(device)
                            img4 = img4.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img1, img2, img3, img4, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    if mAP > top[1]:
                        top[1] = mAP; top[0] = epoch; top[2] = cmc[0]
                        torch.save(model.state_dict(),
                            os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))
                        logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img1, img2, img3, img4, vid, camid, camids, target_view ,img_path  ) in enumerate(val_loader):
                    with torch.no_grad():
                        img1 = img1.to(device)
                        img2 = img2.to(device)
                        img3 = img3.to(device)
                        img4 = img4.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img1, img2, img3, img4, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                if mAP > top[1]:
                    top[1] = mAP; top[0] = epoch; top[2] = cmc[0]
                    torch.save(model.state_dict(),
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_best.pth'))

                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                logger.info('Best result: {} {:.1%} {:.1%}' .format(top[0], top[1], top[2]))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img1, img2, img3, img4, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img1 = img1.to(device)
            img2 = img2.to(device)
            img3 = img3.to(device)
            img4 = img4.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img1, img2, img3, img4, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    if cfg.TEST.VISUALIZE ==True:
        dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

        evaluator.visualize(img_path_list, dataset.query_dir, dataset.gallery_dir, pids=cfg.TEST.PID_VISUAL, save_dir=cfg.TEST.VISUAL_DIR,visual_mode = cfg.TEST.VISUAL_MODE)

    return cmc[0], cmc[4]


