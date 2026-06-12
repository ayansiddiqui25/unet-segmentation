import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from dataset import get_dataloaders
from model import UNet

# --- CONFIGURATION ---
# These are hyperparameters — values you set before training that control how it runs
DATA_ROOT = "./data/cityscapes"
IMG_SIZE = (256, 256)
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 34
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def calculate_iou(preds, labels, num_classes):
    """
    Calculates mean IoU across all classes.
    
    For each class:
    - Intersection: pixels correctly predicted as that class
    - Union: all pixels predicted OR labeled as that class
    - IoU = Intersection / Union
    
    We average across all classes to get mean IoU (mIoU)
    A score of 1.0 = perfect, 0.0 = no overlap at all
    """
    iou_list = []
    preds = preds.view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        pred_cls = (preds == cls)
        label_cls = (labels == cls)

        intersection = (pred_cls & label_cls).sum().float()
        union = (pred_cls | label_cls).sum().float()

        if union == 0:
            # Class not present in this batch — skip it
            continue

        iou_list.append((intersection / union).item())

    return np.mean(iou_list) if iou_list else 0.0


def train():
    # Use CPU since we don't have an NVIDIA GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading dataset...")
    train_loader, val_loader = get_dataloaders(DATA_ROOT, IMG_SIZE, BATCH_SIZE)

    # Initialize model, loss function, and optimizer
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)

    # CrossEntropyLoss: measures how wrong the model's class predictions are
    # ignore_index=255 skips pixels Cityscapes marks as unlabeled
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Adam optimizer: adjusts model weights based on gradients
    # lr (learning rate) controls how big each adjustment step is
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Track metrics across epochs for plotting
    train_losses = []
    val_losses = []
    val_ious = []

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        # --- TRAINING PHASE ---
        model.train()  # puts model in training mode (enables dropout, batchnorm updates)
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()         # clear gradients from last step
            outputs = model(images)       # forward pass — model makes predictions
            loss = criterion(outputs, masks)  # compare predictions to real masks
            loss.backward()               # backprop — calculate gradients
            optimizer.step()              # update model weights

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # --- VALIDATION PHASE ---
        # torch.no_grad() disables gradient calculation — saves memory during eval
        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Get predicted class per pixel — argmax picks highest scoring class
                preds = outputs.argmax(dim=1)
                val_iou += calculate_iou(preds.cpu(), masks.cpu(), NUM_CLASSES)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val mIoU={val_iou:.4f}")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"  ✓ Saved best model (val loss improved to {val_loss:.4f})")

    # --- PLOT METRICS ---
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_ious, label="Val mIoU", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("Validation mIoU over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig("./outputs/training_metrics.png")
    plt.show()
    print("Training complete. Metrics saved to outputs/training_metrics.png")


if __name__ == "__main__":
    train()