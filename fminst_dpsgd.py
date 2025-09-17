import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from tqdm import tqdm
import random
import time  # ⬅ Added for timing

# ✅ Reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ✅ Output paths
output_dir = "outputs_fmnist_dpsgd"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, "training_log.txt")
acc_file = os.path.join(output_dir, "accuracy_per_epoch.txt")
model_path = os.path.join(output_dir, "pytorch_model.bin")

# ✅ Remove old logs
for f in [log_file, acc_file]:
    if os.path.exists(f):
        os.remove(f)

def log(message):
    print(message)
    with open(log_file, "a") as f:
        f.write(message + "\n")

def log_acc(epoch, train_acc, test_acc):
    with open(acc_file, "a") as f:
        f.write(f"{epoch},{train_acc:.4f},{test_acc:.4f}\n")

# ✅ CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  
        self.pool = nn.MaxPool2d(2, 2)               
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) 
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = x.view(-1, 64 * 7 * 7)            
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Data (Fashion-MNIST)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.FashionMNIST(root="./data", train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# ✅ Model, Optimizer, Loss
model = CNN().to(device)
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

# ✅ Training & Evaluation
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# ✅ Training Loop with Timing
log("Training on Fashion-MNIST with DP-SGD...\n")
start_time = time.time()

for epoch in range(1, 201):
    epoch_start = time.time()
    
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    
    epoch_end = time.time()
    epoch_time = epoch_end - epoch_start
    
    log(f"Epoch {epoch}/200")
    log(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    log(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc*100:.2f}%")
    log(f"  Epoch Time: {epoch_time:.2f} seconds\n")
    log_acc(epoch, train_acc, test_acc)

end_time = time.time()
total_time = end_time - start_time
log(f"Total Training Time: {total_time:.2f} seconds")

# ✅ Save final model
torch.save(model.state_dict(), model_path)
log(f"Model saved to {model_path}")
