import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# make argparser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--dropout", type=float, default=0.5)
args = parser.parse_args()


class Network(nn.Module):
    def __init__(self, dropout):
        super(Network, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(14 * 14 * 16, 500),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(500, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.convolutional(x)
        # reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x


def trainer(
    model,
    train_loader,
    test_loader,
    trainset,
    testset,
    optimizer,
    device,
    num_epochs=10,
):
    for epoch in range(num_epochs):
        # For each epoch
        train_correct = 0
        model.train()
        for minibatch_no, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # Zero the gradients computed for each weight
            optimizer.zero_grad()
            # Forward pass your image through the network
            output = model(data)
            # Compute the loss
            loss = F.nll_loss(torch.log(output), target)
            # Backward pass through the network
            loss.backward()
            # Update the weights
            optimizer.step()

            # Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target == predicted).sum().cpu().item()
        # Comput the test accuracy
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data = data.to(device)
            with torch.no_grad():
                output = model(data)
            predicted = output.argmax(1).cpu()
            test_correct += (target == predicted).sum().item()
        train_acc = train_correct / len(trainset)
        test_acc = test_correct / len(testset)
        print(
            "Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(
                test=100 * test_acc, train=100 * train_acc
            )
        )


def main():
    # parse args
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    dropout = args.dropout
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trans = transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(1)])
    trainset = datasets.MNIST("./data", train=True, download=True, transform=trans)
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1
    )
    testset = datasets.MNIST(
        "./data", train=False, download=True, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=1
    )
    model = Network(dropout)
    model.to(device)
    # Initialize the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Get the first minibatch
    data = next(iter(train_loader))[0].cuda()
    # Try running the model on a minibatch
    print(
        "Shape of the output from the convolutional part",
        model.convolutional(data).shape,
    )
    model(data)
    # if this runs the model dimensions fit

    trainer(
        model, train_loader, test_loader, trainset, testset, optimizer, device, epochs
    )
