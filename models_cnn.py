import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

from tqdm import tqdm

# from tqdm.auto import tqdm


class VoiceDataset(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y), "The lengths of X and y must match"
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert the numpy arrays to PyTorch tensors
        sample = torch.tensor(self.X[idx], dtype=torch.float)
        label = torch.tensor(
            self.y[idx], dtype=torch.long
        )  # per torch: synonymous to int64
        return sample, label


class AudioCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AudioCNN, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    def fit(
        self,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        n_epochs=100,
        patience=15,
        save_path="model.pt",
    ):
        best_val_loss = float("inf")
        early_stop_counter = 0

        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

        for epoch in (pbar := tqdm(range(n_epochs), desc="Epoch")):
            print("epoch: ", epoch)
            training_loss, training_correct, n_samples = 0, 0, 0

            self.train()
            for batch_idx, (data, label) in enumerate(train_loader):

                data = data.to(device)
                label = label.to(device)
                output = self.forward(data)

                loss = criterion(output, label)
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                training_correct += pred.eq(label.view_as(pred)).sum().item()
                n_samples += len(label)
                ## DONE with train_loader loop

            # batch stats
            avg_train_acc = training_correct / n_samples
            train_losses.append(training_loss)
            train_accuracies.append(avg_train_acc)

            self.eval()
            val_loss, val_correct, n_samples = 0, 0, 0
            with torch.no_grad():
                for data, label in val_loader:
                    data = data.to(device)
                    label = label.to(device)
                    output = self.forward(data)
                    loss = criterion(output, label)
                    val_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    val_correct += pred.eq(label.view_as(pred)).sum().item()
                    n_samples += len(label)

                    ## DONE with val_loader loop

            # stats per epoch
            avg_val_acc = val_correct / n_samples
            val_losses.append(val_loss)
            val_accuracies.append(avg_val_acc)

            pbar.set_postfix(
                {
                    # "epoch": epoch,
                    "Train loss": training_loss,
                    "Train acc": avg_train_acc,
                    "Val loss": val_loss,
                    "Val acc": avg_val_acc,
                }
            )

            ## Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("Early stopping triggered.")
                    break

        torch.save(
            {
                "model": self,
                "epoch": epoch,
                "batch_size": train_loader.batch_size,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": training_loss,
                "val_loss": val_loss,
                "train_acc": avg_train_acc,
                "val_acc": avg_val_acc,
                "all_train_loss": train_losses,
                "all_val_loss": val_losses,
                "all_train_acc": train_accuracies,
                "all_val_acc": val_accuracies,
                "train_loader": train_loader,
                "val_loader": val_loader,
            },
            save_path,
        )

        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_accuracies = train_accuracies
        self.val_accuracies = val_accuracies

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_epochs = n_epochs

    def fit_binary(
        self,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        n_epochs=100,
        patience=15,
        save_path="model.pt",
    ):
        best_val_loss = float("inf")
        early_stop_counter = 0

        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        for epoch in (pbar := tqdm(range(n_epochs), desc="Epoch")):
            training_loss, training_correct, n_samples = 0, 0, 0

            self.train()
            for batch_idx, (data, label) in enumerate(train_loader):
                data = data.to(device)
                label = label.to(
                    device
                ).float()  # Cast label to float for BCEWithLogitsLoss
                output = self.forward(data)

                # Squeeze the output to remove the extra dimension
                output = output.squeeze(1)

                loss = criterion(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_loss += loss.item()
                pred = (
                    torch.sigmoid(output) > 0.5
                ).float()  # Apply threshold to get binary predictions
                training_correct += pred.eq(label.view_as(pred)).sum().item()
                n_samples += len(label)

            # Calculate average training loss and accuracy
            avg_train_acc = training_correct / n_samples
            train_losses.append(training_loss)
            train_accuracies.append(avg_train_acc)

            self.eval()
            val_loss, val_correct, n_samples = 0, 0, 0
            with torch.no_grad():
                for data, label in val_loader:
                    data = data.to(device)
                    label = label.to(
                        device
                    ).float()  # Cast label to float for BCEWithLogitsLoss
                    output = self.forward(data)

                    # Squeeze the output to remove the extra dimension
                    output = output.squeeze(1)

                    loss = criterion(output, label)
                    val_loss += loss.item()
                    pred = (
                        torch.sigmoid(output) > 0.5
                    ).float()  # Apply threshold to get binary predictions
                    val_correct += pred.eq(label.view_as(pred)).sum().item()
                    n_samples += len(label)

            # Calculate average validation loss and accuracy
            avg_val_acc = val_correct / n_samples
            val_losses.append(val_loss)
            val_accuracies.append(avg_val_acc)

            pbar.set_postfix(
                {
                    "Train loss": training_loss,
                    "Train acc": avg_train_acc,
                    "Val loss": val_loss,
                    "Val acc": avg_val_acc,
                }
            )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("Early stopping triggered.")
                    break

        torch.save(
            {
                "model": self,
                "epoch": epoch,
                "batch_size": train_loader.batch_size,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": training_loss,
                "val_loss": val_loss,
                "train_acc": avg_train_acc,
                "val_acc": avg_val_acc,
                "all_train_loss": train_losses,
                "all_val_loss": val_losses,
                "all_train_acc": train_accuracies,
                "all_val_acc": val_accuracies,
                "train_loader": train_loader,
                "val_loader": val_loader,
            },
            save_path,
        )

        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_accuracies = train_accuracies
        self.val_accuracies = val_accuracies

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_epochs = n_epochs

    def predict(self, data_loader, device):
        pred_list = []
        prob_list = []
        label_list = []

        # for j, (audio, label) in enumerate(tqdm(data_loader)):
        for j, (audio, label) in enumerate(data_loader):
            audio = audio.to(device)
            label = label.to(device)
            output = self.forward(audio)
            # loss = criterion(output, label)

            # Get predicted probabilities
            out_prob = F.softmax(output, dim=1)

            # Get prediction with np.argmax
            pred = output.argmax(dim=1, keepdim=True)

            pred = pred.detach().cpu().numpy().reshape(-1)
            truth = label.detach().cpu().numpy().reshape(-1)

            pred_list.append(pred)
            prob_list.append(out_prob)
            label_list.append(truth)

        return pred_list, label_list

    # def predict_with_prob(self, data_loader, device):
    #     pred_list = []
    #     prob_list = []
    #     label_list = []

    #     # for j, (audio, label) in enumerate(tqdm(data_loader)):
    #     for j, (audio, label) in enumerate(data_loader):
    #         audio = audio.to(device)
    #         label = label.to(device)
    #         output = self.forward(audio)
    #         # loss = criterion(output, label)

    #         # Get predicted probabilities
    #         out_prob = F.softmax(output, dim=1)

    #         # Get prediction with np.argmax
    #         pred = output.argmax(dim=1, keepdim=True)

    #         pred = pred.detach().cpu().numpy().reshape(-1)
    #         truth = label.detach().cpu().numpy().reshape(-1)
    #         prob = out_prob.detach().cpu().numpy().reshape(-1)

    #         pred_list.append(pred)
    #         prob_list.append(prob)
    #         label_list.append(truth)

    #     return pred_list, prob_list, label_list

    def predict_binary(self, data_loader, device, threshold=0.5):
        pred_list = []
        out_prob_list = []
        label_list = []

        for j, (audio, label) in tqdm(enumerate(data_loader)):
            audio = audio.to(device)
            label = label.to(device)
            output = self.forward(audio)

            probs = torch.sigmoid(output)
            pred = (probs >= threshold).long()  # accept by threshold of prob
            probs = probs.detach().cpu().numpy().reshape(-1)
            pred = pred.detach().cpu().numpy().reshape(-1)
            truth = label.detach().cpu().numpy().reshape(-1)

            pred_list.append(pred)
            out_prob_list.append(probs)
            label_list.append(truth)

        return pred_list, out_prob_list, label_list


class BaseCNN(AudioCNN):
    def __init__(self, num_classes=10):
        super().__init__()
        if num_classes is None:
            num_classes = 200
        # Input shape: N x 1 x 17 x 216
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))
        # Output shape: N x 16 x 17 x 108
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        # Output shape: N x 16 x 8 x 27

        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=(3, 5), stride=(1, 1), padding=(1, 1)
        )
        # Output shape: N x 32 x 8 x 27
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4))
        # Output shape: N x 32 x 4 x 6

        self.fc1 = nn.Linear(32 * 4 * 6, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, num_classes)  # Output layer

        self.dropout = nn.Dropout(0.5)  # Dropout layer to reduce overfitting

    # conv->relu->conv->relu->flatten
    # dropout->fc->dropout->fc
    def forward(self, x):
        # Add a channel dimension (N x 1 x 17 x 216)
        x = x.unsqueeze(1)
        # Convolutional and pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 32 * 4 * 6)
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Output layer
        return x


class CNNLSTMNet(AudioCNN):
    def __init__(
        self,
        num_filters_conv1,
        num_filters_conv2,
        num_filters_conv3,
        num_filters_conv4,
        hidden_size_lstm,
        dropout_rate,
        num_classes=10,
    ):
        super(CNNLSTMNet, self).__init__()

        self.hidden_size_lstm = hidden_size_lstm

        self.conv1 = nn.Conv2d(
            1, num_filters_conv1, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv2d(
            num_filters_conv1,
            num_filters_conv2,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv2d(
            num_filters_conv2,
            num_filters_conv3,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
        )
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout3 = nn.Dropout(dropout_rate)

        self.conv4 = nn.Conv2d(
            num_filters_conv3,
            num_filters_conv4,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
        )
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout4 = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(
            input_size=num_filters_conv4 * (3000 // 16),
            hidden_size=hidden_size_lstm,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(hidden_size_lstm * 6, hidden_size_lstm * 3)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_lstm * 3, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.dropout4(x)
        x = x.view(x.size(0), 1, -1)  # Reshape for LSTM
        _, (x, _) = self.lstm(x)
        x = x.transpose(0, 1)
        x = x.reshape(-1, self.hidden_size_lstm * 6)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
