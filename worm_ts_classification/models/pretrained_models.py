#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ..path import get_base_dir

import os

basedir = get_base_dir()
pretrained_path = os.path.join(basedir, 'pretrained_models')
os.environ['TORCH_MODEL_ZOO'] = pretrained_path

from .models import Flatten

from torch import nn
import torchvision

#.models import squeezenet1_1, squeezenet1_0, resnet18, resnet34, resnet50, resnet101, resnet152


_squeezenets = {
        'squeezenet10' : torchvision.models.squeezenet1_0,
        'squeezenet11' : torchvision.models.squeezenet1_1
        }


_resnets = {
        'resnet18' : torchvision.models.resnet18,
        'resnet34' : torchvision.models.resnet34,
        'resnet50' : torchvision.models.resnet50
        }

_vggs = {'vgg11bn' :torchvision.models.vgg11_bn,
         'vgg13bn' :torchvision.models.vgg13_bn,
         'vgg16bn' :torchvision.models.vgg16_bn,
         'vgg19bn' :torchvision.models.vgg19_bn,
        }

class BasePretrained(nn.Module):
    def __init__(self, features_module, num_classes, n_layers2freeze,  num_feat_maps, dropout_fc = 0.5):
        super().__init__()
        self.features = features_module
        
        if not n_layers2freeze is None:
            layers2freeze = self.features[:n_layers2freeze].parameters()
            for param in layers2freeze:
                param.requires_grad = False
        
        self.fc_clf = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  
                nn.Dropout(dropout_fc),
                nn.Conv2d(num_feat_maps, num_classes, kernel_size=1,
                                stride=1, padding=0, bias=True),
                Flatten()
                )
    
    def forward(self, x_in):
        x_in = x_in.repeat(1,3,1,1)
        feats = self.features(x_in)
        x_out = self.fc_clf(feats)
        return x_out

class SqueezeNetP(BasePretrained):
    def __init__(self, model_name, num_classes, n_layers2freeze = None, pretrained=True, **argwks):
        num_feat_maps = 512
        model_base = _squeezenets[model_name](pretrained=pretrained)
        features_module = model_base.features
        super().__init__(features_module, num_classes, n_layers2freeze, num_feat_maps, **argwks)
    
    


class ResNetP(BasePretrained):
    def __init__(self, model_name, num_classes, n_layers2freeze = None, pretrained=True, **argwks):
        
        num_feat_maps = 512
        model_base = _resnets[model_name](pretrained=pretrained)
        features_module = nn.Sequential(
                model_base.conv1,
                model_base.bn1,
                model_base.relu,
                model_base.maxpool,
        
                model_base.layer1,
                model_base.layer2,
                model_base.layer3,
                model_base.layer4
                )
        
        super().__init__(features_module, num_classes, n_layers2freeze, num_feat_maps, **argwks)
        
class VGGP(BasePretrained):
    def __init__(self, model_name, num_classes, n_layers2freeze = None, pretrained=True, **argwks):
        
        num_feat_maps = 512
        model_base = _vggs[model_name](pretrained=pretrained)
        features_module = model_base.features
        super().__init__(features_module, num_classes, n_layers2freeze, num_feat_maps, **argwks)

if __name__ == '_main__':
    pass