import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from tqdm import tqdm

# ✅ Save logs
log_file = "dpsgd_mnist_results.txt"
if os.path.exists(log_file):
    os.remove(log_file)

def log(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

# ✅ CNN architecture (same as DP-Forward)
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
        x = x.view(-1, 64 * 7 * 7)            # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ Data
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# ✅ Model, Optimizer, Loss
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ✅ Privacy Engine
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    target_epsilon=8.0,
    target_delta=1e-5,
    epochs=200,
    max_grad_norm=1.0,
)

# ✅ Train and Test Loops
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy

# ✅ Folder to save model
save_dir = "outputs_mnist_dpsgd"
os.makedirs(save_dir, exist_ok=True)

# ✅ File to store only accuracy per epoch
acc_file = os.path.join(save_dir, "accuracy_per_epoch.txt")
if os.path.exists(acc_file):
    os.remove(acc_file)

# ✅ Training Loop with Timer
log("Training with DP-SGD...\n")
start_time = time.time()

for epoch in range(1, 201):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    
    log(f"Epoch {epoch}/200")
    log(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    log(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc*100:.2f}%\n")

    # ✅ Save only accuracy in a separate file
    with open(acc_file, "a") as f:
        f.write(f"Epoch {epoch}: Train Acc = {train_acc*100:.2f}%, Test Acc = {test_acc*100:.2f}%\n")

# ✅ End timer
total_time = time.time() - start_time
log(f"Total Training Time: {total_time/60:.2f} minutes")

# ✅ Save model in Hugging Face style
torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
with open(os.path.join(save_dir, "config.json"), "w") as f:
    json.dump({"model_type": "CNN", "input_channels": 1, "num_classes": 10}, f)

log(f"Model saved in {save_dir}")
