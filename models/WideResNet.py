import torch
import torchvision
import torch.nn.init


class WideResNet:
    def __init__(self, lr, weight_decay) -> None:
        self.net = self.__get_net()
        self.optimizer = self.__get_optimizer(lr, weight_decay)

    def __get_net(self):
        wideresnet = torchvision.models.wide_resnet50_2(pretrained=True)
        num_ftrs = wideresnet.fc.in_features
        wideresnet.fc = torch.nn.Linear(num_ftrs, 10)
        torch.nn.init.xavier_uniform_(wideresnet.fc.weight)
        return wideresnet

    def __get_optimizer(self, lr, weight_decay):
        params_1x = [param for name, param in self.net.named_parameters() if 'fc' not in str(name)]
        optimizer = torch.optim.Adam(
            [{'params':params_1x},
             {'params': self.net.fc.parameters(), 'lr': lr*10}],
            lr=lr,
            weight_decay=weight_decay)
        return optimizer
