

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
MAX_LEN = 500  # ustawić według max długości sekwencji (można np. Protein.max_len)

def one_hot_encode(seq, max_len=MAX_LEN):
    encoding = np.zeros((max_len, len(AMINO_ACIDS)), dtype=np.float32)
    for i, aa in enumerate(seq[:max_len]):
        if aa in AA_TO_INDEX:
            encoding[i, AA_TO_INDEX[aa]] = 1.0
    return encoding  # [max_len, 20]

class ProteinDataset(Dataset):
    def __init__(self, proteins):
        self.X = []
        self.y = []
        for protein in proteins:
            self.X.append(one_hot_encode(protein.sequence))
            self.y.append(protein.label)
        self.X = torch.tensor(np.array(self.X)).permute(0, 2, 1)  # [N, C, L]
        self.y = torch.tensor(np.array(self.y)).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ProteinCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(20, 32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * (MAX_LEN // 4), 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train_model(train_set, test_set, epochs=10, batch_size=32, lr=1e-3):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        # ewaluacja
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        acc = accuracy_score(all_labels, all_preds)
        test_accuracies.append(acc)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_losses[-1]:.4f} | Test Acc: {acc:.4f}")

    return model, train_losses, test_accuracies

def plot_training(train_losses, test_accuracies):
    fig, ax1 = plt.subplots()
    ax1.plot(train_losses, label='Loss', color='blue')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='blue')
    ax2 = ax1.twinx()
    ax2.plot(test_accuracies, label='Accuracy', color='green')
    ax2.set_ylabel("Accuracy", color='green')
    plt.title("Training Loss & Test Accuracy")
    plt.show()

if __name__ == "__main__":
    print("Start CNN script!")
    from data_provider import get_positive, get_negative
    from protein import Protein

    positives = get_positive()
    negatives = get_negative()

    all_proteins = np.concatenate([positives, negatives])
    np.random.shuffle(all_proteins)

    split_idx = int(0.8 * len(all_proteins))
    train_proteins = all_proteins[:split_idx]
    test_proteins = all_proteins[split_idx:]

    train_set = ProteinDataset(train_proteins)
    test_set = ProteinDataset(test_proteins)

    model, losses, accs = train_model(train_set, test_set, epochs=10)

    plot_training(losses, accs)


