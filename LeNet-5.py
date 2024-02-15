# Setup and Library

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Hyper-Parameters

batch_size = 100
learning_rate = 0.01
num_epochs = 2

# Data Preparation

train_data = torchvision.datasets.MNIST(root="\data", train=True, transform=transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST(root="\data", train=False, transform=transforms.ToTensor())

train_load = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_load = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)


# Model Building

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 28, 5, 1)
        self.avg1 = nn.AvgPool2d(2, 2, )
        self.conv2 = nn.Conv2d(28, 10, 5, 2)
        self.avg2 = nn.AvgPool2d(2, 2)
        self.conv3 = nn.Conv2d(10, 128, 2, 1)
        self.fc1 = nn.Linear(128 * 1 * 1, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.avg1(F.relu(self.conv1(x)))
        x = self.avg2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = LeNet5()

# Training the model

n_steps = len(train_load)
cretin = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(model.parameters(),learning_rate)

for epoch in range(num_epochs):

    for i, (image, label) in enumerate(train_load):

        #image = image.reshape(-1,28*28)

        # Forward Pass
        output = model(image)
        loss = cretin(output,label)

        # Backward Pass

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epochs [{epoch+1}/{num_epochs}],Step[{i+1}/{n_steps}],Loss{loss.item()}]')

with torch.no_grad():
    n_correct = 0
    n_sample = 0

    for image, label in test_load:


        output  = model(image)

        # Value, Index

        _, predicted = torch.max(output,-1)
        n_sample += label.shape[0]
        n_correct += (predicted==label).sum().item()
accuracy = 100 * n_correct/ n_sample
print(f"Accuracy ={accuracy}")


