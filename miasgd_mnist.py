import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, TensorDataset
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CNN model (must match DP-SGD training) ===
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 28x28 -> 28x28
        self.pool = nn.MaxPool2d(2, 2)               # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 14x14 -> 14x14
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [batch, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # -> [batch, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# === Attack model ===
class AttackModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(num_classes, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x)).squeeze()

# === Load trained DP-SGD model ===
model = CNN().to(device)
state_dict = torch.load("outputs_mnist_dpsgd/pytorch_model.bin", map_location=device)
# Fix _module. prefix
state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

# === Data loaders ===
transform = transforms.ToTensor()

full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
full_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Split: 30k members, 30k shadow training
train_subset, shadow_subset = random_split(full_train, [30000, 30000])
test_subset = full_test

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
shadow_loader = DataLoader(shadow_subset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=128, shuffle=False)

# === Helper: get softmax outputs ===
def get_logits(model, loader):
    outputs_list = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            probs = F.softmax(model(x), dim=1)
            outputs_list.append(probs.cpu())
    return torch.cat(outputs_list, dim=0)

# Get member / non-member predictions
member_logits = get_logits(model, train_loader)
non_member_logits = get_logits(model, test_loader)

member_labels = torch.ones(len(member_logits))
non_member_labels = torch.zeros(len(non_member_logits))

X = torch.cat([member_logits, non_member_logits])
y = torch.cat([member_labels, non_member_labels])

attack_dataset = TensorDataset(X, y)
attack_train_size = int(0.8 * len(attack_dataset))
attack_test_size = len(attack_dataset) - attack_train_size
attack_train_set, attack_test_set = random_split(attack_dataset, [attack_train_size, attack_test_size])

attack_train_loader = DataLoader(attack_train_set, batch_size=128, shuffle=True)
attack_test_loader = DataLoader(attack_test_set, batch_size=128, shuffle=False)

# === Train attack model ===
attack_model = AttackModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(attack_model.parameters(), lr=1e-3)

os.makedirs("outputs", exist_ok=True)
log_path = "outputs/mia_attack_accuracy.txt"
with open(log_path, "w") as f:
    f.write("Epoch\tAttackAccuracy\n")

for epoch in range(1, 201):
    # Train
    attack_model.train()
    for xb, yb in attack_train_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = attack_model(xb)
        loss = criterion(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate
    attack_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in attack_test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = (attack_model(xb) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    acc = correct / total * 100
    print(f"Epoch {epoch}/200\tAttack Accuracy: {acc:.2f}%")
    with open(log_path, "a") as f:
        f.write(f"{epoch}\t{acc:.2f}\n")
