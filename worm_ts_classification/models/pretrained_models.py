#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

pretrained_path = str(Path.home() / 'workspace/pytorch/pretrained_models/')
os.environ['TORCH_MODEL_ZOO'] = pretrained_path

from .models import Flatten, SimpleDilated

import torch
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

_densenets = {
        'densenet121' :torchvision.models.densenet121,
        'densenet169' : torchvision.models.densenet169,
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


class DenseNetP(BasePretrained):
    def __init__(self, model_name, num_classes, n_layers2freeze = None, pretrained=True, **argwks):
        
        num_feat_maps = 1024
        model_base = _densenets[model_name](pretrained=pretrained)
        features_module = model_base.features
        super().__init__(features_module, num_classes, n_layers2freeze, num_feat_maps, **argwks)
        
        
        
class PretrainedSimpleDilated(BasePretrained):
    def __init__(self, 
                 pretrained_path,
                 num_classes, 
                 n_layers2freeze = None,
                 **argkws
                 ):
        
        num_feat_maps = 512
        
        if pretrained_path is not None:
            state = torch.load(pretrained_path, map_location = 'cpu')
            
            old_num_classes = state['state_dict']['fc_clf.2.bias'].shape[0]
            
            
            old_model = SimpleDilated(old_num_classes)
            old_model.load_state_dict(state['state_dict'])
            old_model.eval()
        else:
            old_model = SimpleDilated(2)
            
        features_module = old_model.cnn_clf
        super().__init__(features_module, num_classes, n_layers2freeze, num_feat_maps, **argkws)

        
    def forward(self, x_in):
        feats = self.features(x_in)
        x_out = self.fc_clf(feats)
        return x_out

if __name__ == '_main__':
    pass