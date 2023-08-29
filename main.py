import sys
from sklearn.metrics import classification_report


from models.AlexNet import AlexNet
from models.DenseNet import DenseNet
from models.GoogLeNet import GoogLeNet
from models.InceptionV3 import InceptionV3
from models.MNASNet import MNASNet
from models.MobileNetV2 import MobileNetV2
from models.ResNet34 import ResNet34
from models.ResNeXt import ResNeXt
from models.ShuffleNetV2 import ShuffleNetV2
from models.SqueezeNet import SqueezeNet
from models.VGG import VGG
from models.WideResNet import WideResNet
from fine_tuning import get_predictions
from fine_tuning import get_valid


def train_model(model_name: str, epochs: int = 20, lr: float = 1e-5, weight_decay: float = 5e-4) -> None:
    if model_name == 'AlexNet':
        model = AlexNet(lr, weight_decay)
    elif model_name == 'DenseNet':
        model = DenseNet(lr, weight_decay)
    elif model_name == 'GoogLeNet':
        model = GoogLeNet(lr, weight_decay)
    elif model_name == 'InceptionV3':
        model = InceptionV3(lr, weight_decay)
    elif model_name == 'MNASNet':
        model = MNASNet(lr, weight_decay)
    elif model_name == 'MobileNetV2':
        model = MobileNetV2(lr, weight_decay)
    elif model_name == 'ResNet34':
        model = ResNet34(lr, weight_decay)
    elif model_name == 'ResNeXt':
        model = ResNeXt(lr, weight_decay)
    elif model_name == 'ShuffleNetV2':
        model = ShuffleNetV2(lr, weight_decay)
    elif model_name == 'SqueezeNet':
        model = SqueezeNet(lr, weight_decay)
    elif model_name == 'VGG':
        model = VGG(lr, weight_decay)
    elif model_name == 'WideResNet':
        model = WideResNet(lr, weight_decay)
    else:
        raise Exception('Model not found!')
    net = model.net
    optimizer = model.optimizer
    valid = get_valid()
    preds = get_predictions(net, optimizer, epochs)
    print('Calculating classification report')
    report = classification_report(valid, preds)
    print(report)


def main() -> None:
    # args = sys.argv[1:]
    # if '--help' in args:
    #     pass
    # if '--epochs' in args:
    #     pass
    # if '--model' in args:
    #     pass
    # print(args)
    train_model(model_name='AlexNet', epochs=1)


if __name__ == '__main__':
    main()
