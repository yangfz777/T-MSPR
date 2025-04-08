# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .GlobalTripletLoss import GlobalTripletLoss
from loss.MSEloss import compute_multichannel_loss
from .GlobalTripletLosspro import TripletLosspro

def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            triplet1 = GlobalTripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[4:8]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS1 = [F.cross_entropy(scor, target) for scor in score[8:]]
                        ID_LOSS1 = sum(ID_LOSS1) * 0.6
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * ID_LOSS1 + 0.5 * ((F.cross_entropy(score[0], target) + F.cross_entropy(score[1], target)  + F.cross_entropy(score[2], target) + F.cross_entropy(score[3], target) )/4)
                        Q = F.softmax(score[0], dim=-1)  # 转化为概率分布
                        P = F.softmax(score[4], dim=-1)
                        # P2 = F.softmax(score[2], dim=-1)
                        # P3 = F.softmax(score[3], dim=-1)
                        # P4 = F.softmax(score[7], dim=-1)
                        # P5 = F.softmax(score[5], dim=-1)
                        # P6 = F.softmax(score[6], dim=-1)
                        # P7 = F.softmax(score[7], dim=-1)
                         # P8 = F.softmax(score[8], dim=-1)
                        # P9 = F.softmax(score[9], dim=-1)
                        # P10 = F.softmax(score[10], dim=-1)


                        kl_loss =  F.kl_div(P.log(), Q,
                                                        reduction='batchmean')
                        # kl_loss2 =  F.kl_div(P2.log(), Q,
                        #                                 reduction='batchmean')
                        # kl_loss3 =  F.kl_div(P3.log(), Q,
                        #                                 reduction='batchmean')
                        # kl_loss4 =  F.kl_div(P4.log(), Q,
                        #                                 reduction='batchmean')
                        # kl_loss5 = F.kl_div(P5.log(), Q,
                        #                     reduction='batchmean')
                        # kl_loss6 = F.kl_div(P6.log(), Q,
                        #                     reduction='batchmean')
                        # kl_loss7 = F.kl_div(P7.log(), Q,
                        #                     reduction='batchmean')
                        # kl_loss8 = F.kl_div(P8.log(), Q,
                        #                     reduction='batchmean')
                        # kl_loss9 = F.kl_div(P9.log(), Q,
                        #                     reduction='batchmean')
                        # kl_loss10 = F.kl_div(P10.log(), Q,
                        #                     reduction='batchmean')
                    else:
                        ID_LOSS = F.cross_entropy(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[7:11]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS1 = [triplet(feats, target)[0] for feats in feat[11:]]
                            TRI_LOSS1 = sum(TRI_LOSS1) * 0.6
                            # TRI_LOSS = (0.5 * TRI_LOSS + 0.5 * (
                            #             triplet(feat[0], target)[0] + triplet(feat[1], target)[0] +
                            #             triplet(feat[2], target)[0]) + triplet1(feat[0], feat[2], target)[0] +
                            #             triplet1(feat[0], feat[1], target)[0])

                            TRI_LOSS = (0.5 * TRI_LOSS  + 0.5 * TRI_LOSS1 + 0.5 * (triplet(feat[0], target)[0] ) + triplet(feat[4], target)[0] + triplet(feat[5], target)[0] + triplet(feat[6], target)[0])
                            # TRI_LOSS = (0.5 * TRI_LOSS + 0.5 * TRI_LOSS1 + 0.5 * (triplet(feat[0], target)[0]) +
                            #             triplet(feat[2], target)[0] + triplet(feat[3], target)[0] )
                            # TRI_LOSS = 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + \
                               cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion
#
#
