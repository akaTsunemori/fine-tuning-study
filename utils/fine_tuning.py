from pathlib import Path
import pandas as pd
import torchvision
import torch
import time
import warnings


# Ignore warning messages for a clean output
warnings.filterwarnings('ignore')


class FineTuning:
    def __init__(self, net, optimizer, epochs) -> None:
        root = Path('./data')
        train_dataset = torchvision.datasets.ImageFolder(
            root/'train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),]))
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
        means = []
        stdevs = []
        for X, _ in train_dataloader:
            # Dimensions 0,2,3 are respectively the batch, height and width dimensions
            means.append(X.mean(dim=(0, 2, 3)))
            stdevs.append(X.std(dim=(0, 2, 3)))
        mean = torch.stack(means, dim=0).mean(dim=0)
        stdev = torch.stack(stdevs, dim=0).mean(dim=0)
        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.AutoAugment(
                policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, stdev)])
        train_dataset = torchvision.datasets.ImageFolder(root/'train', transform=train_transforms)
        valid_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, stdev)])
        valid_dataset, test_dataset = [torchvision.datasets.ImageFolder
            (folder, transform=valid_transforms) for folder in [root/'valid', root/'test']]
        num_gpus = torch.cuda.device_count()
        self.train_dataset = train_dataset
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=2*num_gpus, pin_memory=True)
        self.valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=256, shuffle=False, num_workers=2*num_gpus, pin_memory=True)
        self.test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=256, shuffle=False, num_workers=2*num_gpus, pin_memory=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.epochs = epochs
        self.net = net


    def __train(self, scheduler=None, checkpoint_epochs=10):
        net = self.net
        criterion = self.criterion
        optimizer = self.optimizer
        train_dataloader = self.train_dataloader
        valid_dataloader = self.valid_dataloader
        epochs = self.epochs
        device = self.device
        start = time.time()
        net = net.to(device)
        print(f'Training for {epochs} epochs on {device}')
        for epoch in range(1, epochs+1):
            print(f'Epoch {epoch}/{epochs}')
            net.train()
            train_loss = torch.tensor(0., device=device)
            train_accuracy = torch.tensor(0., device=device)
            for X, y in train_dataloader:
                X = X.to(device)
                y = y.to(device)
                preds = net(X)
                loss = criterion(preds, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    train_loss += loss * train_dataloader.batch_size
                    train_accuracy += (torch.argmax(preds, dim=1) == y).sum()
            if valid_dataloader:
                net.eval()
                valid_loss = torch.tensor(0., device=device)
                valid_accuracy = torch.tensor(0., device=device)
                with torch.no_grad():
                    for X, y in valid_dataloader:
                        X = X.to(device)
                        y = y.to(device)
                        preds = net(X)
                        loss = criterion(preds, y)
                        valid_loss += loss * valid_dataloader.batch_size
                        valid_accuracy += (torch.argmax(preds, dim=1) == y).sum()
            if scheduler:
                scheduler.step()
            print(f'Training loss: {train_loss/len(train_dataloader.dataset):.2f}')
            print(f'Training accuracy: {100*train_accuracy/len(train_dataloader.dataset):.2f}')
            if valid_dataloader is not None:
                print(f'Valid loss: {valid_loss/len(valid_dataloader.dataset):.2f}')
                print(f'Valid accuracy: {100*valid_accuracy/len(valid_dataloader.dataset):.2f}')
            if epoch % checkpoint_epochs == 0:
                torch.save({
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, './checkpoint.pth.tar')
            print('')
        end = time.time()
        print(f'Total training time: {end-start:.1f} seconds')
        return net


    def get_predictions(self):
        device = self.device
        net = self.__train()
        test_dataloader = self.test_dataloader
        preds = []
        net.eval()
        with torch.no_grad():
            for X, _ in test_dataloader:
                X = X.to(device)
                preds.extend(
                    net(X).argmax(dim=1).type(torch.int32).cpu().numpy())
        return preds


    def get_target_labels(self):
        data = []
        test_dataloader = self.test_dataloader
        train_dataset = self.train_dataset
        for batch in test_dataloader:
            images, labels = batch
            data.extend(zip(images, labels))
        data_dict = {"id": [], "label": []}
        for idx, (_, label) in enumerate(data, start=1):
            data_dict["id"].append(idx)
            data_dict["label"].append(label)
        df = pd.DataFrame(data_dict)
        df['label'] = df['label'].apply(lambda x: train_dataset.classes[x])
        return df['label']
