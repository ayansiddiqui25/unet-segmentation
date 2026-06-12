import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

# Cityscapes has 34 classes but we'll map them to 19 trainable classes
# This mapping comes from the official Cityscapes label definitions
IGNORE_INDEX = 255

class CityscapesDataset(Dataset):
    """
    This class handles loading images and masks from disk.
    PyTorch expects a Dataset class with __len__ and __getitem__ methods.
    - __len__ tells PyTorch how many samples you have
    - __getitem__ tells PyTorch how to load one sample by index
    """
    def __init__(self, root, split="train", img_size=(256, 256)):
        self.root = root
        self.split = split
        self.img_size = img_size

        # Define image normalization — these mean/std values are
        # standard across ImageNet-trained models. Normalizing helps
        # the model train faster and more stably.
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),  # converts 0-255 → 0-1
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Masks only get resized — NOT normalized
        # Masks contain class labels (integers like 0,1,2...)
        # not pixel colors, so normalizing them would corrupt the labels
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST)
        ])

        # Collect all image/mask file paths
        self.image_paths = []
        self.mask_paths = []

        img_dir = os.path.join(root, "leftImg8bit", split)
        mask_dir = os.path.join(root, "gtFine", split)

        # Cityscapes organizes files by city — loop through each city folder
        for city in sorted(os.listdir(img_dir)):
            city_img_dir = os.path.join(img_dir, city)
            city_mask_dir = os.path.join(mask_dir, city)

            for fname in sorted(os.listdir(city_img_dir)):
                if fname.endswith("_leftImg8bit.png"):
                    img_path = os.path.join(city_img_dir, fname)
                    # The mask filename follows a specific Cityscapes naming convention
                    mask_fname = fname.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                    mask_path = os.path.join(city_mask_dir, mask_fname)

                    if os.path.exists(mask_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)

        print(f"[{split}] Found {len(self.image_paths)} image-mask pairs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        # Apply transforms
        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        # Convert mask to tensor of long integers (class label indices)
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask


def get_dataloaders(root, img_size=(256, 256), batch_size=4):
    """
    Creates DataLoaders for train, val, and test splits.
    A DataLoader wraps your Dataset and handles:
    - Batching (grouping samples together)
    - Shuffling (randomizing order each epoch)
    - Loading data in parallel (num_workers)
    """
    train_dataset = CityscapesDataset(root, split="train", img_size=img_size)
    val_dataset = CityscapesDataset(root, split="val", img_size=img_size)

    # Cityscapes test set has no public masks so we reuse val for testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader