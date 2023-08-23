import torch
import torchvision
import torch.nn.init

class MobileNetV2Class:
    def __init__(self, lr, weight_decay) -> None:
        self.net = self.__get_net()
        self.optimizer = self.__get_optimizer(lr, weight_decay)

    def __get_net(self):
        net = torchvision.models.mobilenet_v2(pretrained=True)
        num_ftrs = net.classifier[1].in_features
        net.classifier[1] = torch.nn.Linear(num_ftrs, 10)
        torch.nn.init.xavier_uniform_(net.classifier[1].weight)
        return net

    def __get_optimizer(self, lr, weight_decay):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer
