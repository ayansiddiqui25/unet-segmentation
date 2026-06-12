import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os

from model import UNet

# Cityscapes color palette — each class gets a unique color for visualization
CITYSCAPES_COLORS = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0],
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
    [0, 0, 0], [0, 0, 0]
], dtype=np.uint8)


def predict(image_path, checkpoint_path="./checkpoints/best_model.pth", img_size=(256, 256)):
    """
    Loads a trained model and runs inference on a single image.
    Saves a side-by-side comparison of the original image and predicted mask.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and weights
    model = UNet(in_channels=3, num_classes=34).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Preprocess image — same transforms as training
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

    # Map class indices to colors for visualization
    color_mask = CITYSCAPES_COLORS[pred_mask]

    # Plot original image vs predicted mask side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image.resize(img_size[::-1]))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(color_mask)
    axes[1].set_title("Predicted Segmentation Mask")
    axes[1].axis("off")

    plt.tight_layout()
    output_path = "./outputs/prediction.png"
    plt.savefig(output_path)
    plt.show()
    print(f"Prediction saved to {output_path}")


if __name__ == "__main__":
    # Replace with any image path you want to test
    predict("./data/cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png")