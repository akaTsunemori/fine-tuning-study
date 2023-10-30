from pathlib import Path
import pandas as pd
import torchvision
import torch
import time
import warnings


# Ignore warning messages for a clean output
warnings.filterwarnings('ignore')


def train(net, train_dataloader, valid_dataloader, criterion, optimizer, scheduler=None, epochs=10, device='cpu', checkpoint_epochs=10):
    start = time.time()
    print(f'Training for {epochs} epochs on {device}')

    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")

        net.train()  # put network in train mode for Dropout and Batch Normalization
        # loss and accuracy tensors are on the GPU to avoid data transfers
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

        if valid_dataloader is not None:
            net.eval()  # put network in train mode for Dropout and Batch Normalization
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

        if scheduler is not None:
            scheduler.step()

        print(f'Training loss: {train_loss/len(train_dataloader.dataset):.2f}')
        print(
            f'Training accuracy: {100*train_accuracy/len(train_dataloader.dataset):.2f}')

        if valid_dataloader is not None:
            print(
                f'Valid loss: {valid_loss/len(valid_dataloader.dataset):.2f}')
            print(
                f'Valid accuracy: {100*valid_accuracy/len(valid_dataloader.dataset):.2f}')

        if epoch % checkpoint_epochs == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, './checkpoint.pth.tar')

        print()

    end = time.time()
    print(f'Total training time: {end-start:.1f} seconds')
    return net


def get_predictions(net, optimizer, epochs):
    net = net.to(device)
    net = train(net, train_dataloader, valid_dataloader,
                criterion, optimizer, None, epochs, device)
    # The original code was training with the validation dataset.
    # net = train(net, train_valid_dataloader, None,
    #             criterion, optimizer, None, epochs, device)
    preds = []
    net.eval()
    with torch.no_grad():
        for X, _ in test_dataloader:
            X = X.to(device)
            preds.extend(net(X).argmax(dim=1).type(torch.int32).cpu().numpy())
    ids = list(range(1, len(test_dataset)+1))
    ids.sort(key=lambda x: str(x))
    df = pd.DataFrame({'id': ids, 'label': preds})
    df['label'] = df['label'].apply(lambda x: train_dataset.classes[x])
    # df.to_csv('submission.csv', index=False)
    return df['label']


def get_valid():
    data = []
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


root = Path('./data')
input_path = Path('./cifar-10')
print('Loading datasets')
train_dataset = torchvision.datasets.ImageFolder(
    root/'train',
    transform=torchvision.transforms.Compose([
        # Resize step is required as we will use a ResNet model, which accepts at leats 224x224 images
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    ])
)
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
    torchvision.transforms.Normalize(mean, stdev)
])
train_dataset, train_valid_dataset = [torchvision.datasets.ImageFolder
    (folder, transform=train_transforms) for folder in [root/'train', root/'train_valid']]
valid_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean, stdev)
])
valid_dataset, test_dataset = [torchvision.datasets.ImageFolder
    (folder, transform=valid_transforms) for folder in [root/'valid', root/'test']]
num_gpus = torch.cuda.device_count()
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=2*num_gpus, pin_memory=True)
train_valid_dataloader = torch.utils.data.DataLoader(
    train_valid_dataset, batch_size=128, shuffle=True, num_workers=2*num_gpus, pin_memory=True)
valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=256, shuffle=False, num_workers=2*num_gpus, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=256, shuffle=False, num_workers=2*num_gpus, pin_memory=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = torch.nn.CrossEntropyLoss()
