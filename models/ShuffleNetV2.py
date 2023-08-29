import torch
import torchvision
import torch.nn.init


class ShuffleNetV2:
    def __init__(self, lr, weight_decay) -> None:
        self.net = self.__get_net()
        self.optimizer = self.__get_optimizer(lr, weight_decay)

    def __get_net(self):
        net = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        num_ftrs = net.fc.in_features
        net.fc = torch.nn.Linear(num_ftrs, 10)
        torch.nn.init.xavier_uniform_(net.fc.weight)
        return net

    def __get_optimizer(self, lr, weight_decay):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer
