import argparse
from sklearn.metrics import classification_report

# Project modules
from models.AlexNet import AlexNet
from models.DenseNet import DenseNet
from models.GoogLeNet import GoogLeNet
from models.MNASNet import MNASNet
from models.MobileNetV2 import MobileNetV2
from models.ResNet34 import ResNet34
from models.ResNeXt import ResNeXt
from models.ShuffleNetV2 import ShuffleNetV2
from models.SqueezeNet import SqueezeNet
from models.VGG import VGG
from models.WideResNet import WideResNet
from utils.fine_tuning import FineTuning
from utils.dataset_setup import dataset_setup



def train_model(model_name: str, epochs: int, lr: float, weight_decay: float) -> None:
    models = {
        'AlexNet': AlexNet,
        'DenseNet': DenseNet,
        'GoogLeNet': GoogLeNet,
        'MNASNet': MNASNet,
        'MobileNetV2': MobileNetV2,
        'ResNet34': ResNet34,
        'ResNeXt': ResNeXt,
        'ShuffleNetV2': ShuffleNetV2,
        'SqueezeNet': SqueezeNet,
        'VGG': VGG,
        'WideResNet': WideResNet
    }
    model = models[model_name](lr, weight_decay)
    net = model.net
    optimizer = model.optimizer
    fine_tuning = FineTuning(net, optimizer, epochs)
    target = fine_tuning.get_target_labels()
    preds = fine_tuning.get_predictions()
    print('Computing classification report')
    report = classification_report(target, preds, output_dict=True)
    print(report)
    return report


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ResNet34')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    return parser.parse_args()


def main(args) -> None:
    dataset_setup()
    train_model(
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay)


if __name__ == '__main__':
    main(get_args())
