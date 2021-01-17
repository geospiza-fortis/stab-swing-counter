# https://medium.com/secure-and-private-ai-writing-challenge/loading-image-using-pytorch-c2e2dcce6ef2
# https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://www.pluralsight.com/guides/image-classification-with-pytorch

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, utils
import click
from pathlib import Path
import numpy as np


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@click.command()
@click.argument("input", type=click.Path(file_okay=False, exists=True))
@click.argument("output", type=click.Path(file_okay=False))
def train(input, output):
    input = Path(input)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = datasets.ImageFolder(input, transform=transform)
    n = len(dataset)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [int(n * 0.8), n - int(n * 0.8)]
    )
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=True)

    images, labels = next(iter(trainloader))
    classes = ("other", "stab", "swing")

    print(" ".join(classes[labels[j]] for j in range(4)))
    imshow(utils.make_grid(images))

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print("Finished Training")
    state_path = f"{output}/classifier_state.pth"
    torch.save(net.state_dict(), state_path)

    dataiter = iter(valloader)
    images, labels = dataiter.next()

    print("GroundTruth: ", " ".join(classes[labels[j]] for j in range(4)))
    _, predicted = torch.max(outputs, 1)
    print("Predicted: ", " ".join(classes[predicted[j]] for j in range(4)))
    imshow(utils.make_grid(images))


if __name__ == "__main__":
    train()
