# encoding: utf-8
"""
this file is based on:
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com

several modifications are applied
"""
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss


def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER

    l1loss = torch.nn.SmoothL1Loss()
    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        if cfg.MODEL.NO_MARGIN:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
            print("using triplet loss with margin:{}".format(cfg.SOLVER.MARGIN))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))


    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        label_smooth = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    id_loss_rotation = [label_smooth(scor, target) for scor in score[1:]]
                    id_loss_rotation = sum(id_loss_rotation) / len(id_loss_rotation)
                    id_loss_origin = label_smooth(score[0], target)
                    id_loss = cfg.MODEL.ROTATION_WEIGHT * id_loss_rotation + cfg.MODEL.ORIGINAL_WEIGHT * id_loss_origin
                else:
                    id_loss_rotation = [F.cross_entropy(scor, target) for scor in score[1:]]
                    id_loss_rotation = sum(id_loss_rotation) / len(id_loss_rotation)
                    id_loss_origin = F.cross_entropy(score[0], target)
                    id_loss = cfg.MODEL.ROTATION_WEIGHT*id_loss_rotation + cfg.MODEL.ORIGINAL_WEIGHT*id_loss_origin

                mean = [scor for scor in score[1:]]
                mean = sum(mean) / len(mean)
                constraint_id = l1loss(score[0], mean)

                tri_loss_rotation = [triplet(feats, target)[0] for feats in feat[1:]]
                tri_loss_rotation = sum(tri_loss_rotation) / len(tri_loss_rotation)
                tri_loss_origin = triplet(feat[0], target)[0]
                tri_loss = cfg.MODEL.ROTATION_WEIGHT*tri_loss_rotation + cfg.MODEL.ORIGINAL_WEIGHT*tri_loss_origin

                mean = [feats for feats in feat[1:]]
                mean = sum(mean) / len(mean)
                constraint_tr = l1loss(feat[0], mean)

                return id_loss + tri_loss + constraint_tr + constraint_id
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    else:
        print('expected sampler should be softmax or softmax_triplet'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


