import torch
import torchvision
import torch.nn.init


class ResNet34:
    def __init__(self, lr, weight_decay) -> None:
        self.net = self.__get_net()
        self.optimizer = self.__get_optimizer(lr, weight_decay)

    def __get_net(self):
        resnet = torchvision.models.resnet34(pretrained=True)
        resnet.fc = torch.nn.Linear(resnet.fc.in_features, 10)
        torch.nn.init.xavier_uniform_(resnet.fc.weight)
        return resnet

    def __get_optimizer(self, lr, weight_decay):
        params_1x = [param for name, param in self.net.named_parameters() if 'fc' not in str(name)]
        optimizer = torch.optim.Adam(
            [{'params':params_1x},
             {'params': self.net.fc.parameters(), 'lr': lr*10}],
            lr=lr,
            weight_decay=weight_decay)
        return optimizer
