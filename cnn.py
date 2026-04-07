"""
FashionMNIST CNN Assignment - Complete Implementation
=====================================================
Covers:
  Step 1: Data Preprocessing
  Step 2: Building the CNN Model (Baseline + 2 improved variants)
  Step 3: Training & Evaluation (loss curves, accuracy, confusion matrix)
  Step 4: Experimentation & Improvements
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# STEP 1: DATA PREPROCESSING
def get_dataloaders(augment=False, batch_size=64):
    """
    Load FashionMNIST, normalize to [0,1], optionally apply augmentation.
    Splits training set into 80% train / 20% validation.
    """
    # Base transform: convert to tensor (scales to [0,1]) then normalize
    # Mean=0.2860, Std=0.3530 are the FashionMNIST dataset statistics
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2860], std=[0.3530])
    ])

    # Augmented transform adds random horizontal flip and small rotation
    aug_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2860], std=[0.3530])
    ])

    train_transform = aug_transform if augment else base_transform

    # Load datasets
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FashionMNIST")
    os.makedirs(data_dir, exist_ok=True)

    full_train = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=base_transform
)

    # 80/20 train/validation split
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_dataset, val_dataset = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


# STEP 2: CNN ARCHITECTURES

# Baseline CNN
class BaselineCNN(nn.Module):
    """
    Simple 2-block CNN.
    Conv1: 1→32, 3×3, ReLU, MaxPool 2×2
    Conv2: 32→64, 3×3, ReLU, MaxPool 2×2
    FC: 64×5×5 → 128 → 10
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28×28 → 28×28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # → 14×14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14×14 → 14×14
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # → 7×7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Improvement 1: Dropout + BatchNorm
class ImprovedCNN(nn.Module):
    """
    Adds Batch Normalization after each conv (stabilizes training)
    and Dropout(0.5) before the final layer (reduces overfitting).
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Improvement 2: Deeper network + Data Augmentation
class DeepCNN(nn.Module):
    """
    3-block deeper CNN with BatchNorm and Dropout.
    Designed to be used with augmented data (augment=True).
    Conv3: 64→128, 3×3 with BatchNorm
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # → 14×14

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # → 7×7

            # Block 3 (no pooling to keep spatial info)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# STEP 3: TRAINING & EVALUATION
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n

def train_model(model, train_loader, val_loader, epochs=15, lr=1e-3):
    """
    Train model with Adam optimizer and CrossEntropyLoss.
    Returns history dict with train/val loss and accuracy per epoch.
    """
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()       
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss:   {vl_loss:.4f} Acc: {vl_acc:.4f}")

    return history

def plot_loss_curves(histories, labels, filename="loss_curves.png"):
    """Plot train/val loss curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#3266ad", "#e07b39", "#5a9e5a"]

    for i, (hist, label) in enumerate(zip(histories, labels)):
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], color=colors[i], label=f"{label} (train)", linewidth=2)
        axes[0].plot(epochs, hist["val_loss"],   color=colors[i], label=f"{label} (val)",   linewidth=2, linestyle="--")
        axes[1].plot(epochs, hist["train_acc"],  color=colors[i], label=f"{label} (train)", linewidth=2)
        axes[1].plot(epochs, hist["val_acc"],    color=colors[i], label=f"{label} (val)",   linewidth=2, linestyle="--")

    for ax, title, ylabel in zip(axes,
                                  ["Loss Curves", "Accuracy Curves"],
                                  ["Loss", "Accuracy"]):
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def get_all_preds_labels(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            preds = model(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(model, test_loader, model_name, filename=None):
    y_true, y_pred = get_all_preds_labels(model, test_loader)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {filename}")
    plt.close()

    print(f"\nClassification Report — {model_name}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


# STEP 4: MAIN — RUN ALL EXPERIMENTS
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    EPOCHS = 15

    # Experiment 1: Baseline CNN (no augmentation)
    print("\n" + "="*60)
    print("BASELINE CNN")
    print("="*60)
    train_loader, val_loader, test_loader = get_dataloaders(augment=False)
    baseline = BaselineCNN()
    print(f"Parameters: {count_params(baseline):,}")
    hist_baseline = train_model(baseline, train_loader, val_loader, epochs=EPOCHS)

    # Experiment 2: Improved CNN (BatchNorm + Dropout)
    print("\n" + "="*60)
    print("IMPROVED CNN (BatchNorm + Dropout)")
    print("="*60)
    improved = ImprovedCNN()
    print(f"Parameters: {count_params(improved):,}")
    hist_improved = train_model(improved, train_loader, val_loader, epochs=EPOCHS)

    # Experiment 3: Deep CNN (3 blocks + Augmentation)
    print("\n" + "="*60)
    print("DEEP CNN (3 blocks + Data Augmentation)")
    print("="*60)
    train_loader_aug, val_loader_aug, _ = get_dataloaders(augment=True)
    deep = DeepCNN()
    print(f"Parameters: {count_params(deep):,}")
    hist_deep = train_model(deep, train_loader_aug, val_loader_aug, epochs=EPOCHS)

    # Loss curves for all three models 
    plot_loss_curves(
        [hist_baseline, hist_improved, hist_deep],
        ["Baseline", "Improved (BN+Dropout)", "Deep+Augment"],
        filename="loss_curves.png"
    )

    # Test accuracy summary
    criterion = nn.CrossEntropyLoss()
    print("\n" + "="*60)
    print("FINAL TEST ACCURACY COMPARISON")
    print("="*60)
    for model, name, loader in [
        (baseline, "Baseline",            test_loader),
        (improved, "Improved",            test_loader),
        (deep,     "Deep+Augment",        test_loader),
    ]:
        _, acc = evaluate(model, loader, criterion)
        print(f"  {name:<25} Test Accuracy: {acc*100:.2f}%")

    # Confusion matrices
    plot_confusion_matrix(baseline, test_loader, "Baseline CNN",        "cm_baseline.png")
    plot_confusion_matrix(improved, test_loader, "Improved CNN",        "cm_improved.png")
    plot_confusion_matrix(deep,     test_loader, "Deep CNN + Augment",  "cm_deep.png")


if __name__ == "__main__":
    main()