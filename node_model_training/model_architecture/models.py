import torch.nn as nn

from . import backbone_multi
from . import call_resnet18_multi as cl


class Transfer_Net(nn.Module):

    def __init__(self, num_class, base_net='resnet18_multi_new', transfer_loss='cmmd', use_bottleneck=True, bottleneck_width=128, width=512):
        super(Transfer_Net, self).__init__()

        self.base_network = backbone_multi.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss

        if base_net == 'resnet18_multi_new':
            self.base_network = cl.load_resnet18_multi()

        bottle_list = [nn.Linear(256, width), nn.Linear(width, bottleneck_width)]
        classifier_list = [nn.Dropout(0.25), nn.Linear(bottleneck_width, num_class)]

        self.bottle_layer = nn.Sequential(*bottle_list)
        self.classifier_layer = nn.Sequential(*classifier_list)

        self.bottle_layer[0].weight.data.normal_(0, 0.01)
        self.bottle_layer[0].bias.data.fill_(0.0)
        self.bottle_layer[1].weight.data.normal_(0, 0.01)
        self.bottle_layer[1].bias.data.fill_(0.0)
        self.classifier_layer[1].weight.data.normal_(0, 0.01)
        self.classifier_layer[1].bias.data.fill_(0.0)

    def forward(self, source, s_label, test_flag):
        source = self.base_network(source, test_flag)
        source = self.bottle_layer(source)
        source_clf = self.classifier_layer(source)
        return source, source_clf

    def predict(self, x, test_flag):
        features = self.base_network(x, test_flag)
        features = self.bottle_layer(features)
        clf = self.classifier_layer(features)
        return clf
