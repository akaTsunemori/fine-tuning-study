import torch
import torchvision
import torch.nn.init


class SqueezeNet:
    def __init__(self, lr, weight_decay) -> None:
        self.net = self.__get_net()
        self.optimizer = self.__get_optimizer(lr, weight_decay)

    def __get_net(self):
        squeezenet = torchvision.models.squeezenet1_0(pretrained=True)
        squeezenet.classifier[1] = torch.nn.Conv2d(512, 10, kernel_size=(1,1), stride=(1,1))
        torch.nn.init.xavier_uniform_(squeezenet.classifier[1].weight)
        return squeezenet

    def __get_optimizer(self, lr, weight_decay):
        params_1x = [param for name, param in self.net.named_parameters() if 'classifier' not in str(name)]
        optimizer = torch.optim.Adam(
            [{'params': params_1x},
             {'params': self.net.classifier.parameters(), 'lr': lr * 10}],
            lr=lr,
            weight_decay=weight_decay)
        return optimizer
