import torch
import torchvision
import torch.nn.init


class DenseNet:
    def __init__(self, lr, weight_decay) -> None:
        self.net = self.__get_net()
        self.optimizer = self.__get_optimizer(lr, weight_decay)

    def __get_net(self):
        net = torchvision.models.densenet121(pretrained=True)
        num_ftrs = net.classifier.in_features
        net.classifier = torch.nn.Linear(num_ftrs, 10)
        torch.nn.init.xavier_uniform_(net.classifier.weight)
        return net

    def __get_optimizer(self, lr, weight_decay):
        params_1x = [param for name, param in self.net.named_parameters() if 'classifier' not in str(name)]
        optimizer = torch.optim.Adam(
            [{'params': params_1x},
             {'params': self.net.classifier.parameters(), 'lr': lr * 10}],
            lr=lr,
            weight_decay=weight_decay)
        return optimizer
