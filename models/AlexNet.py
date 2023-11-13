import torch
import torchvision
import torch.nn.init


class AlexNet:
    def __init__(self, lr, weight_decay) -> None:
        self.net = self.__get_net()
        self.optimizer = self.__get_optimizer(lr, weight_decay)

    def __get_net(self):
        alexnet = torchvision.models.alexnet(pretrained=True)
        num_classes = 10
        alexnet.classifier = torch.nn.Sequential(
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, num_classes)
        )
        torch.nn.init.xavier_uniform_(alexnet.classifier[6].weight)
        return alexnet

    def __get_optimizer(self, lr, weight_decay):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer
