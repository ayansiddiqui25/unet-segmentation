import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Two consecutive convolution layers, each followed by BatchNorm and ReLU.
    This is the basic building block of U-Net — used in both encoder and decoder.
    
    Convolution: scans the image with a filter to detect features (edges, textures, shapes)
    BatchNorm: normalizes activations so training is stable
    ReLU: activation function — introduces non-linearity so the model can learn complex patterns
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Full U-Net architecture for semantic segmentation.
    
    in_channels: number of input channels (3 for RGB images)
    num_classes: number of output classes (34 for Cityscapes)
    features: number of feature channels at each encoder level
    """
    def __init__(self, in_channels=3, num_classes=34, features=[64, 128, 256, 512]):
        super().__init__()

        # --- ENCODER ---
        # Each encoder block applies DoubleConv then MaxPool to shrink spatial size
        # MaxPool2d halves the height and width at each step
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # --- BOTTLENECK ---
        # Deepest point of the network — most compressed representation
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # --- DECODER ---
        # Each decoder step:
        # 1. ConvTranspose2d (upsampling) — doubles spatial size
        # 2. Concatenate with corresponding encoder skip connection
        # 3. DoubleConv to process the combined features
        self.decoder_upsample = nn.ModuleList()
        self.decoder_conv = nn.ModuleList()

        for feature in reversed(features):
            self.decoder_upsample.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder_conv.append(
                DoubleConv(feature * 2, feature)  # *2 because of skip connection concat
            )

        # --- OUTPUT ---
        # Final 1x1 convolution maps feature channels to class scores
        # Output shape: (batch, num_classes, height, width)
        # Each pixel gets a score for each class — highest score = predicted class
        self.output_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Store encoder outputs for skip connections
        skip_connections = []

        # Encoder pass — shrink down
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)  # save for skip connection
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections to match decoder order
        skip_connections = skip_connections[::-1]

        # Decoder pass — expand back up
        for i in range(len(self.decoder_upsample)):
            x = self.decoder_upsample[i](x)
            skip = skip_connections[i]
            x = torch.cat([skip, x], dim=1)  # concatenate along channel dimension
            x = self.decoder_conv[i](x)

        return self.output_conv(x)


if __name__ == "__main__":
    # Quick test — feed a random image through and check output shape
    # Output should be (1, 34, 256, 256) — one class score per pixel
    model = UNet(in_channels=3, num_classes=34)
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")