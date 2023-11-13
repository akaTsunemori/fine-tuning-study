import torch
import torchvision
import torch.nn.init


class VGG:
    def __init__(self, lr, weight_decay) -> None:
        self.net = self.__get_net()
        self.optimizer = self.__get_optimizer(lr, weight_decay)

    def __get_net(self):
        vgg = torchvision.models.vgg16(pretrained=True)
        num_ftrs = vgg.classifier[6].in_features
        vgg.classifier[6] = torch.nn.Linear(num_ftrs, 10)
        torch.nn.init.xavier_uniform_(vgg.classifier[6].weight)
        return vgg

    def __get_optimizer(self, lr, weight_decay):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer
