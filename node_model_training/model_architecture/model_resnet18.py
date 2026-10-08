"""ResNet18 with extracted_layer cut + standard ResNet residual tail (no LiMAR/bottle)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models as tv_models

from . import backbone_multi as ml


class resnet18_tail(nn.Module):
    """Front cut matches resnet18_multi; tail is remaining ResNet18 residual layers."""

    def __init__(self, block=ml.BasicBlock, layers=None):
        if layers is None:
            layers = [2, 2, 2, 2]
        self.inplanes = 64
        super(resnet18_tail, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _extracted_layer(self):
        return getattr(ml, "extracted_layer", None)

    def forward(self, x, test_flag):
        extracted_layer = self._extracted_layer()

        if test_flag:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            if extracted_layer == "6_point":
                x = self.layer3[0](x)
            elif extracted_layer == "7_point":
                x = self.layer3(x)
            elif extracted_layer == "8_point":
                x = self.layer3(x)
                x = self.layer4[0](x)
            # 5_point: cut after layer2

        # Remaining ResNet18 tail after the cut point
        if extracted_layer == "5_point":
            x = self.layer3(x)
            x = self.layer4(x)
        elif extracted_layer == "6_point":
            x = self.layer3[1](x)
            x = self.layer4(x)
        elif extracted_layer == "7_point":
            x = self.layer4(x)
        elif extracted_layer == "8_point":
            x = self.layer4[1](x)
        else:
            raise ValueError(
                f"Unsupported extracted_layer={extracted_layer!r}; "
                "expected one of 5_point / 6_point / 7_point / 8_point"
            )

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def load_resnet18_tail():
    """Build resnet18_tail and load matching ImageNet pretrained weights."""
    pretrained = tv_models.resnet18(pretrained=True)
    model = resnet18_tail(block=ml.BasicBlock, layers=[2, 2, 2, 2])

    pretrained_dict = pretrained.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


class Transfer_Net_ResNet18(nn.Module):
    """Same forward/predict API as Transfer_Net, without LiMAR or bottle_layer."""

    def __init__(self, num_class):
        super(Transfer_Net_ResNet18, self).__init__()
        self.base_network = load_resnet18_tail()
        self.classifier_layer = nn.Linear(512, num_class)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, source, s_label, test_flag):
        source = self.base_network(source, test_flag)
        source_clf = self.classifier_layer(source)
        return source, source_clf

    def predict(self, x, test_flag):
        features = self.base_network(x, test_flag)
        return self.classifier_layer(features)
