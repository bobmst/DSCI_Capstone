import torch.nn as nn
import numpy as np

import torch
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class BaseNet(nn.Module):
    def __init__(self, num_classes=1):
        super(BaseNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (30, 12), (1, 1)),
            nn.ReLU()
            )

        self.maxpool1 = nn.MaxPool2d(kernel_size=(10, 1))

        self.dropout = nn.Dropout(.6)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 1), (1, 1)),
            nn.ReLU()
            )

        self.maxpool2 = nn.MaxPool2d(kernel_size=(4, 1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 16, (4, 1), (1, 1)),
            nn.ReLU()
            )

        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 1))

        self.fc = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
            )

    def forward(self, x0):
        x0 = x0.unsqueeze(1)
        x0 = self.conv1(x0)
        # x0 = self.relu(x0)
        x0 = self.maxpool1(x0)
        x0 = self.dropout(x0)

        x0 = self.conv2(x0)
        # x0 = self.relu(x0)
        x0 = self.maxpool2(x0)
        x0 = self.dropout(x0)

        x0 = self.conv3(x0)
        # x0 = self.relu(x0)
        x0 = self.maxpool3(x0)
        x0 = self.dropout(x0)

        x0 = x0.view(x0.size(0), -1)
        # x0 = torch.flatten(x0)
        # out = torch.cat([out, fr], dim=1)
        x0 = self.fc(x0)
        # x0 = self.relu(x0)
        x0 = torch.nn.Sigmoid()(x0)

        return x0

    def fit(self, train_loader, test_loader, criterion, optimizer):

        # Epoch loop
        for i in range(30):
            print(f'\n===== EPOCH {i} =====')

            training_loss = 0
            test_loss = 0
            error_rate = 0
            n_samples = 0

            self.train()
            for j, (image, label) in tqdm(enumerate(train_loader)):
                # Forward pass (consider the recommmended functions in homework writeup)
                # image = torch.transpose(image,1,2)
                # image = torch.reshape(image,(32,5000,12,1))
                label = label.reshape(-1, 1)
                output = self.forward(image)

                # Backward pass and optimize (consider the recommmended functions in homework writeup)
                # Make sure to zero out the gradients using optimizer.zero_grad() in each loop
                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track the loss and error rate
                training_loss += loss.item()
                pred = (output > 0.5).to(torch.float32).reshape(-1, 1)
                error_rate += (pred != label).sum().item()
                n_samples += len(label)

            error_rate = 0
            n_samples = 0

            self.eval()
            for j, (image, label) in tqdm(enumerate(test_loader)):
                # image = torch.transpose(image,1,2)
                label = label.reshape(-1, 1)
                output = self.forward(image)
                loss = criterion(output, label)

                # Track the loss and error rate
                test_loss += loss.item()
                pred = (output > 0.5).to(torch.float32).reshape(-1, 1)
                error_rate += (pred != label).sum().item()
                n_samples += len(label)

                # Print/return training loss and error rate in each epoch
            print('training loss for each epoch is:', training_loss)
            print('test loss for each epoch is:', test_loss)
            print('error rate for each epoch is:', error_rate / n_samples)

        pred_list = []
        label_list = []
        for j, (image, label) in tqdm(enumerate(test_loader)):
            image = torch.transpose(image, 1, 2)
            label = label.reshape(-1, 1)
            output = self.forward(image)
            loss = criterion(output, label)

            # Track the loss and error rate
            test_loss += loss.item()
            pred = (output > 0.5).to(torch.float32).reshape(-1, 1)
            error_rate += (pred != label).sum().item()
            n_samples += len(label)

            pred_list.append(pred)
            label_list.append(label)

        # Print the confusion matrix
        cm = confusion_matrix(label_list, pred_list)
        labels = ['Non-CD', 'CD']

        # Plot the confusion matrix
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=labels, yticklabels=labels,
               title='Confusion matrix',
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over the data and create a text annotation for each cell
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2. else "black")
        fig.tight_layout()
        plt.show()


