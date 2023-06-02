import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import glob
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

# Define the U-Net model architecture
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # Decoder
        self.decoder4 = self.conv_block(512, 256)
        self.decoder3 = self.conv_block(256 + 256, 128)
        self.decoder2 = self.conv_block(128 + 128, 64)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)
        print("x1 shape:", x1.shape)
        x2 = self.encoder2(F.max_pool2d(x1, 2))
        print("x2 shape:", x2.shape)
        x3 = self.encoder3(F.max_pool2d(x2, 2))
        print("x3 shape:", x3.shape)
        x4 = self.encoder4(F.max_pool2d(x3, 2))
        print("x4 shape:", x4.shape)

        # Decoder
        x = self.decoder4(x4)
        print("x shape after decoder4:", x.shape)
        x = self.upsample(x)
        print("x shape after upsampling:", x.shape)
        x = self.decoder3(torch.cat([x, x3], dim=1))
        print("x shape after decoder3:", x.shape)
        x = self.upsample(x)
        print("x shape after upsampling:", x.shape)
        x = self.decoder2(torch.cat([x, x2], dim=1))
        print("x shape after decoder2:", x.shape)

        # Final output
        x = self.final_conv(x)
        print("Final output shape:", x.shape)
        return x


# Custom dataset class
class MyDataset(Dataset):
    def __init__(self, source_dir, transform=None):
        self.source_dir = source_dir
        self.transform = transform

    def __len__(self):
        return len(self.source_dir)

    def __getitem__(self, idx):
        source_path = self.source_dir[idx]
        source_image = Image.open(source_path)

        if self.transform is not None:
            source_image = self.transform(source_image)

        return source_image


# Set the paths for training source and masks
training_source = glob.glob("training/source/*.tif")
training_masks = glob.glob("training/masks/*.tif")
target_boundaries = glob.glob("training/target_boundaries/*.tif")

# Define the transformation(s) to be applied to the images
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add more transforms as needed
])

# Create an instance of the dataset
dataset = MyDataset(training_source, transform=transform)

# Create a data loader for inference
batch_size = 1
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Instantiate the U-Net model
model = UNet(in_channels=1, out_channels=1)  # Modify the in_channels and out_channels based on your data

# Load the pre-trained model weights if available
# model.load_state_dict(torch.load("path_to_pretrained_model.pth"))

# Set the model to evaluation mode
model.eval()

# Perform inference on a sample image
sample_idx = 0
sample_image = next(iter(data_loader))
print("Sample image shape:", sample_image.shape)
output = model(sample_image)
print("Output shape:", output.shape)

# Convert the output tensor to an image
output_image = transforms.ToPILImage()(output.squeeze(0).cpu())

# Visualize the sample image and the output
plt.subplot(1, 2, 1)
plt.imshow(sample_image.squeeze(0).permute(1, 2, 0))
plt.title('Sample Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output_image)
plt.title('Output')
plt.axis('off')

plt.show()
