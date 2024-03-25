import random
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import copy
from .backbones.vit_pytorch_submit import vit_base_patch16_224_RotatedTransformer

def rotation(features, H, W):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Patch Rotation Operation
    x = features.transpose(1, 2).view(batchsize, dim, H, W)
    degree = random.randint(-10, 10)
    x = TF.rotate(img=x, angle=degree)
    x = x.flatten(2).transpose(1, 2)
    return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)



class build_transformer_local(nn.Module):
    def __init__(self, num_classes, cfg, factory):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN,
                                                        stride_size=cfg.MODEL.STRIDE_SIZE,
                                                        drop_path_rate=cfg.MODEL.DROP_PATH)

        self.num_x = self.base.num_x
        self.num_y = self.base.num_y

        self.rotation_numbers = cfg.MODEL.NUMBERS

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.b2_1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2_2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2_3 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2_4 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2_5 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.classifier_a = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_a.apply(weights_init_classifier)

        self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)
        self.classifier_5 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_5.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)
        self.bottleneck_5 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_5.bias.requires_grad_(False)
        self.bottleneck_5.apply(weights_init_kaiming)

    def forward(self, x):

        features = self.base(x)

        # original feature
        b1_feat = self.b1(features)
        global_feat = b1_feat[:, 0]  # cls token

        # Feature Level Rotation

        token = features[:, 0:1]
        x = features[:, 1:]
        parts = []

        for i in range(self.rotation_numbers):
            rot = rotation(x, self.num_x, self.num_y)  # simulate rotation
            parts.append(rot)

        rotated_features = []

        for index in range(self.rotation_numbers):
            rotated_feat = parts[index]
            if index == 0:
                rotated_feat = self.b2_1(torch.cat((token, rotated_feat), dim=1))
            if index == 1:
                rotated_feat = self.b2_2(torch.cat((token, rotated_feat), dim=1))
            if index == 2:
                rotated_feat = self.b2_3(torch.cat((token, rotated_feat), dim=1))
            if index == 3:
                rotated_feat = self.b2_4(torch.cat((token, rotated_feat), dim=1))
            if index == 4:
                rotated_feat = self.b2_5(torch.cat((token, rotated_feat), dim=1))
            rotated_feat_1 = rotated_feat[:, 0]
            rotated_features.append(rotated_feat_1)

        feat = self.bottleneck(global_feat)

        cls_scores = []
        rotated_features_bn = []
        global_score = self.classifier(feat)

        cls_scores.append(global_score)

        count = 1
        for f in rotated_features:
            if count == 1:
                rotated_feat_bn = self.bottleneck_1(f)
                cls_score_1 = self.classifier_1(rotated_feat_bn)
            if count == 2:
                rotated_feat_bn = self.bottleneck_2(f)
                cls_score_1 = self.classifier_2(rotated_feat_bn)
            if count == 3:
                rotated_feat_bn = self.bottleneck_3(f)
                cls_score_1 = self.classifier_3(rotated_feat_bn)
            if count == 4:
                rotated_feat_bn = self.bottleneck_4(f)
                cls_score_1 = self.classifier_4(rotated_feat_bn)
            if count == 5:
                rotated_feat_bn = self.bottleneck_5(f)
                cls_score_1 = self.classifier_5(rotated_feat_bn)
            rotated_features_bn.append(rotated_feat_bn)
            cls_scores.append(cls_score_1)
            count += 1

        if self.training:
            feat_train = [global_feat]
            for i in rotated_features:
                feat_train.append(i)
            return cls_scores, feat_train
        else:
            if self.neck_feat == 'bn':
                feat_test = [feat]
                return torch.cat(feat_test, dim=1)
            else:
                feat_test = [global_feat]
                return torch.cat(feat_test, dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_RotatedTransformer': vit_base_patch16_224_RotatedTransformer
}


def make_model(cfg, num_class):
    model = build_transformer_local(num_class, cfg, __factory_T_type)
    print('Rotated Vision Transformer')
    return model
