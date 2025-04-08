import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .preprocessing import RandomErasing_Background,AddLocalSaltNoise,RandomErasing_Backgroundpro,RandomZoom,RandomResizeAndPad
from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
import os
# from .occ_reid import Occ_ReID
# from .partial_reid import Partial_REID

# __factory = {
#     'market1501': Market1501,
#     'dukemtmc': DukeMTMCreID,
#     'msmt17': MSMT17,
#     'occluded_dukemtmc': OCC_DukeMTMCreID,
#     'veri': VeRi,
#     'VehicleID': VehicleID,
#     # 'partial_reid': Partial_REID,
#     # 'occ_reid': Occ_ReID
# }
#
# def train_collate_fn(batch):
#     """
#     # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
#     """
#     imgs1, imgs2, imgs3, pids, camids, viewids , _ = zip(*batch)
#     pids = torch.tensor(pids, dtype=torch.int64)
#     viewids = torch.tensor(viewids, dtype=torch.int64)
#     camids = torch.tensor(camids, dtype=torch.int64)
#     return torch.stack(imgs1, dim=0), torch.stack(imgs2, dim=0), torch.stack(imgs3, dim=0), pids, camids, viewids,
#
# def val_collate_fn(batch):
#     imgs1, imgs2, imgs3, pids, camids, viewids, img_paths = zip(*batch)
#     viewids = torch.tensor(viewids, dtype=torch.int64)
#     camids_batch = torch.tensor(camids, dtype=torch.int64)
#     return torch.stack(imgs1, dim=0), torch.stack(imgs2, dim=0), torch.stack(imgs3, dim=0), pids, camids, camids_batch, viewids, img_paths
#
#
# def make_dataloader(cfg):
#     train_transforms = T.Compose([
#
#             T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
#             # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
#             # T.Pad(cfg.INPUT.PADDING),
#             # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
#             T.ToTensor(),
#             T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
#             # T.RandomResizedCrop(size=(256, 128)),
#             # RandomErasing_Background(EPSILON=1, root=os.path.join('/opt/data/private/yfz/PADE-main', 'crop_backgrounds')),
#             # RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
#             # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
#         ])
#     # crop_transforms = T.Compose([
#     #         T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
#     #         # T.Resize(cfg.INPUT.Zoom, interpolation=3),
#     #         # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
#     #         T.Pad(30),
#     #         T.ToTensor(),
#     #         # AddLocalSaltNoise(prob=0.01, area_ratio=0.1),
#     #         T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
#     #         # RandomZoom(EPSILON=1,root=os.path.join('/opt/data/private/yfz/NFormer-main/data/occluded_dukemtmc',
#     #         #                                           'bounding_box_train')),
#     #         # T.RandomResizedCrop(size=(256, 128), scale=(0.3, 0.6)),
#     #         T.RandomResizedCrop(size=(256, 128)),
#     #     ])
#     crop_transforms = T.Compose([
#         T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
#         # T.Resize(cfg.INPUT.Zoom, interpolation=3),
#         # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
#         # T.Pad(30),
#         # T.ToTensor(),
#         # AddLocalSaltNoise(prob=0.01, area_ratio=0.1),
#         # T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
#         # RandomZoom(EPSILON=1, root=os.path.join('/opt/data/private/yfz/NFormer-main/data/occluded_dukemtmc',
#         #                                         'bounding_box_train')),
#         RandomResizeAndPad(cfg.INPUT.SIZE_TRAIN, scale_range=(0.8, 1.0)),
#         T.ToTensor(),
#         T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
#         # T.RandomResizedCrop(size=(256, 128), scale=(0.3, 0.6)),
#         # T.RandomResizedCrop(size=(256, 128)),
#     ])
#     eraser_transforms = T.Compose([
#             T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
#             # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
#             T.ToTensor(),
#             T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
#             # RandomErasing_Background(EPSILON=1, root=os.path.join('/opt/data/private/yfz/PADE-main', 'crop_backgrounds')),
#             # RandomErasing_Backgroundpro(EPSILON=1, root=os.path.join('/opt/data/private/yfz/NFormer-main/data/occluded_dukemtmc', 'bounding_box_train')),
#             RandomErasing(probability=1, mode='pixel', max_count=1, device='cpu'),
#         ])
#
#
#
#     val_transforms = T.Compose([
#         T.Resize(cfg.INPUT.SIZE_TEST),
#         T.ToTensor(),
#         T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
#     ])
#
#     num_workers = cfg.DATALOADER.NUM_WORKERS
#
#     dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
#
#     train_set = ImageDataset(dataset.train, train_transforms, crop_transform=crop_transforms, eraser_transform=eraser_transforms)
#     train_set_normal = ImageDataset(dataset.train, val_transforms)
#     num_classes = dataset.num_train_pids
#     cam_num = dataset.num_train_cams
#     view_num = dataset.num_train_vids
#
#     if 'triplet' in cfg.DATALOADER.SAMPLER:
#         if cfg.MODEL.DIST_TRAIN:
#             print('DIST_TRAIN START')
#             mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
#             data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
#             batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
#             train_loader = torch.utils.data.DataLoader(
#                 train_set,
#                 num_workers=num_workers,
#                 batch_sampler=batch_sampler,
#                 collate_fn=train_collate_fn,
#                 pin_memory=True,
#             )
#         else:
#             train_loader = DataLoader(
#                 train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
#                 sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
#                 num_workers=num_workers, collate_fn=train_collate_fn
#             )
#     elif cfg.DATALOADER.SAMPLER == 'softmax':
#         print('using softmax sampler')
#         train_loader = DataLoader(
#             train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
#             collate_fn=train_collate_fn
#         )
#     else:
#         print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))
#
#     val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
#
#     val_loader = DataLoader(
#         val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
#         collate_fn=val_collate_fn
#     )
#     train_loader_normal = DataLoader(
#         train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
#         collate_fn=val_collate_fn
#     )
#     return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num

# import torch
# import torchvision.transforms as T
# from torch.utils.data import DataLoader
# from .preprocessing import RandomErasing_Background,AddLocalSaltNoise,R
# from .bases import ImageDataset
# from timm.data.random_erasing import RandomErasing
# from .sampler import RandomIdentitySampler
# from .dukemtmcreid import DukeMTMCreID
# from .market1501 import Market1501
# from .msmt17 import MSMT17
# from .sampler_ddp import RandomIdentitySampler_DDP
# import torch.distributed as dist
# from .occ_duke import OCC_DukeMTMCreID
# from .vehicleid import VehicleID
# from .veri import VeRi
# import os
from .occ_reid import Occ_ReID
from .partial_reid import Partial_REID
#
__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occluded_dukemtmc': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'partial_reid': Partial_REID,
    'occ_reid': Occ_ReID
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs1, imgs2, imgs3, imgs4, pids, camids, viewids, img_path = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs1, dim=0), torch.stack(imgs2, dim=0), torch.stack(imgs3, dim=0), torch.stack(imgs4, dim=0), pids, camids, viewids, img_path

def val_collate_fn(batch):
    imgs1, imgs2, imgs3, imgs4, pids, camids, viewids, img_path = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs1, dim=0), torch.stack(imgs2, dim=0), torch.stack(imgs3, dim=0), torch.stack(imgs4, dim=0), pids, camids, camids_batch, viewids ,img_path


def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            # T.Pad(cfg.INPUT.PADDING),
            # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    crop_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(30),
            T.ToTensor(),
            # AddLocalSaltNoise(prob=0.01, area_ratio=0.1),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            # T.RandomResizedCrop(size=(256, 128), scale=(0.3, 0.6)),
            T.RandomResizedCrop(size=(256, 128)),
        ])
    eraser_transforms = T.Compose([
                    T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                    # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                    T.ToTensor(),
                    T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                    # RandomErasing_Background(EPSILON=1, root=os.path.join('/opt/data/private/yfz/PADE-main', 'crop_backgrounds')),
                    RandomErasing_Backgroundpro(EPSILON=1, root=os.path.join('/opt/data/private/yfz/NFormer-main/data/occluded_dukemtmc', 'bounding_box_train')),
                    RandomErasing(probability=1, mode='pixel', max_count=1, device='cpu'),
                ])
    eraser_transforms1 = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
                # T.Resize(cfg.INPUT.Zoom, interpolation=3),
                # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
                # T.Pad(30),
                # T.ToTensor(),
                # AddLocalSaltNoise(prob=0.01, area_ratio=0.1),
                # T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
                # RandomZoom(EPSILON=1, root=os.path.join('/opt/data/private/yfz/NFormer-main/data/occluded_dukemtmc',
                #                                         'bounding_box_train')),
                RandomResizeAndPad(cfg.INPUT.SIZE_TRAIN, scale_range=(0.8, 1.0)),
                T.ToTensor(),
                T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),

                # T.RandomResizedCrop(size=(256, 128), scale=(0.3, 0.6)),
                # T.RandomResizedCrop(size=(256, 128)),
    ])



    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set = ImageDataset(dataset.train, train_transforms, crop_transform=crop_transforms, eraser_transform=eraser_transforms,eraser_transform1=eraser_transforms1)
    train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()
            data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader = torch.utils.data.DataLoader(
                train_set,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_collate_fn
            )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    train_loader_normal = DataLoader(
        train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, train_loader_normal, val_loader, len(dataset.query), num_classes, cam_num, view_num
