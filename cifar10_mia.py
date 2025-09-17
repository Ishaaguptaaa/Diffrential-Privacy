import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model (same as CIFAR10 version) ---
class CIFARDPModel(nn.Module):
    def __init__(self):
        super(CIFARDPModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 250)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(250, 10)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(B, -1)
        x = torch.sigmoid(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

# --- Attack Model ---
class AttackModel(nn.Module):
    def __init__(self, num_classes=10):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(num_classes, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)).squeeze()

# --- Load Trained Model (on CIFAR10) ---
model = CIFARDPModel().to(device)
model.load_state_dict(torch.load("outputs_cifar10/cifar10_model/pytorch_model.bin", map_location=device))
model.eval()

# --- CIFAR-10 Dataset ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
full_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_subset, shadow_subset = random_split(full_train, [25000, 25000])
test_subset = full_test

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
shadow_loader = DataLoader(shadow_subset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)

# --- Get Model Predictions (logits) ---
def get_logits(model, data_loader):
    logits_list = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            outputs = model(x)
            probs = F.softmax(outputs, dim=1)
            logits_list.append(probs.detach().cpu())
    return torch.cat(logits_list, dim=0)

member_logits = get_logits(model, train_loader)
non_member_logits = get_logits(model, test_loader)

member_labels = torch.ones(len(member_logits))
non_member_labels = torch.zeros(len(non_member_logits))

X = torch.cat([member_logits, non_member_logits])
y = torch.cat([member_labels, non_member_labels])

# --- Prepare Attack Dataset ---
attack_dataset = TensorDataset(X, y)
attack_train_size = int(0.8 * len(attack_dataset))
attack_test_size = len(attack_dataset) - attack_train_size
attack_train_set, attack_test_set = random_split(attack_dataset, [attack_train_size, attack_test_size])

attack_train_loader = DataLoader(attack_train_set, batch_size=128, shuffle=True)
attack_test_loader = DataLoader(attack_test_set, batch_size=128, shuffle=False)

# --- Attack Model Training ---
attack_model = AttackModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(attack_model.parameters(), lr=1e-3)

# --- Logging ---
log_path = "outputs_cifar10/mia_attack_accuracy.txt"
os.makedirs("outputs_cifar10", exist_ok=True)
with open(log_path, "w") as f:
    f.write("Epoch\tAttackAccuracy\n")

# --- Train Attack Model ---
for epoch in range(1, 201):
    attack_model.train()
    for xb, yb in attack_train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = attack_model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- Evaluation ---
    attack_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in attack_test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = attack_model(xb)
            preds = (preds > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch}/200\tAttack Accuracy: {acc:.2f}%")
    with open(log_path, "a") as f:
        f.write(f"{epoch}\t{acc:.2f}\n")
