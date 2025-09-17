import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dp_noise import get_noise_multiplier, _max_norm_clip  # Assuming these are implemented elsewhere

# ----------------- CIFAR10 Model -----------------
class CIFARDPModel(nn.Module):
    def __init__(self, noise_factor=0.0, norm_c=1.0):
        super().__init__()
        self.noise_factor = noise_factor
        self.norm_c = norm_c
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.5)
        # Dynamically calculate the flat_dim using a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 32, 32)
            dummy = F.relu(self.conv1(dummy))
            dummy = F.relu(self.conv2(dummy))
            dummy = self.pool(dummy)
            dummy = self.dropout1(dummy)
            flat_dim = dummy.view(1, -1).shape[1]
        self.flat_dim = flat_dim
        self.fc1 = nn.Linear(self.flat_dim, 250)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(250, 10)

    def forward(self, x):
        B = x.size(0)
        if self.noise_factor > 0:
            x = x + self.noise_factor * torch.randn_like(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(B, -1)
        x = _max_norm_clip(x, self.norm_c)
        x = torch.sigmoid(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)

# ----------------- Training Functions -----------------
def train_one_epoch(model, loader, optimizer, criterion, device, log_interval=100):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(loader, start=1):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % log_interval == 0:
            avg = running_loss / log_interval
            print(f"  Train Batch {batch_idx}  Loss: {avg:.4f}")
            running_loss = 0.0

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
    avg_loss = total_loss / len(loader)
    acc = correct / len(loader.dataset)
    return avg_loss, acc

# ----------------- Main Training -----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64
    EPOCHS = 200
    LR = 1e-3
    EPSILON = 8.0
    DELTA = 1e-5
    NORM_C = 1.0

    # Data transforms for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # DP noise
    noise_factor = get_noise_multiplier(
        eps=EPSILON, delta=DELTA,
        batch_size=BATCH_SIZE,
        dataset_size=len(train_ds),
        epoch=EPOCHS,
        local_dp=True, noise_type="aGM"
    )
    print(f"🔐 Noise factor: {noise_factor:.4f}")

    # Model setup
    model = CIFARDPModel(noise_factor=noise_factor, norm_c=NORM_C).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Log file
    log_path = "outputs_cifar10/cifar10_model/epoch_eval_results.txt"
    os.makedirs("outputs_cifar10/cifar10_model", exist_ok=True)
    with open(log_path, "w") as f:
        f.write("Epoch\tTrainAcc(%)\tTestAcc(%)\n")

    # Training loop with timing
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}:")
        train_one_epoch(model, train_loader, optimizer, criterion, device, log_interval=200)
        train_loss, train_acc = evaluate(model, train_loader, criterion, device)
        test_loss, test_acc   = evaluate(model, test_loader, criterion, device)

        train_acc *= 100
        test_acc  *= 100
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

        with open(log_path, "a") as f:
            f.write(f"{epoch}\t{train_acc:.2f}\t{test_acc:.2f}\n")

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print(f"🕒 Total training time: {elapsed_minutes:.2f} minutes")

    # Save model for MIA
    torch.save(model.state_dict(), "outputs_cifar10/cifar10_model/pytorch_model.bin")
    with open(log_path, "a") as f:
        f.write(f"\nTotal Training Time: {elapsed_minutes:.2f} minutes\n")

    print("✅ Training complete! Model saved to outputs_cifar10/cifar10_model/")

if __name__ == "__main__":
    main()
