"""
Module 3: Two-Tier Cloud & Shadow Masking Module
Implements spectral thresholding (Tier 1) and a PyTorch Attention U-Net (Tier 2)
to segment thick clouds, thin haze, and terrain shadows.
"""

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional

# Setup Logger
logger = logging.getLogger("DhruvaPipeline.Masking")


def generate_spectral_guess(red_band: np.ndarray, threshold: float = 0.25) -> np.ndarray:
    """
    Tier 1 Masking: Performs quick spectral thresholding on the LISS-IV Red band (Band 3).
    Clouds have high reflectance in the Red spectrum.
    
    Args:
        red_band: Normalized float array of Red band reflectance [0, 1].
        threshold: Reflectance value above which pixels are classed as clouds.
        
    Returns:
        Binary mask (1 for cloud, 0 for clear).
    """
    logger.info(f"Generating spectral guess cloud mask with threshold = {threshold}")
    coarse_mask = (red_band > threshold).astype(np.float32)
    return coarse_mask


class AttentionGate(nn.Module):
    """
    Attention Gate for skip-connections in U-Net.
    Funnels spatial structural keys from encoder to align with decoder gating signals.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1)
        )
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        g: Decoder gating signal (lower spatial resolution, higher depth)
        x: Encoder feature map (higher spatial resolution, skip connection)
        """
        g_in = self.W_g(g)
        # Match dimensions by upsampling g if needed
        if g_in.shape[2:] != x.shape[2:]:
            g_in = F.interpolate(g_in, size=x.shape[2:], mode='bilinear', align_corners=True)
            
        x_in = self.W_x(x)
        
        # Add intermediate activations
        joint_activation = self.relu(g_in + x_in)
        
        # Squeeze to single channel coefficient map
        attention_coef = self.sigmoid(self.psi(joint_activation))
        
        return x * attention_coef


class ConvBlock(nn.Module):
    """Double convolution helper for U-Net architecture."""
    def __init__(self, in_ch: int, out_ch: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Lightweight Attention U-Net segmentation network for fine-grained
    cloud and shadow mask generation.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super(AttentionUNet, self).__init__()
        
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        
        # Encoder (Downsampling)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = ConvBlock(in_channels, filters[0])
        self.Conv2 = ConvBlock(filters[0], filters[1])
        self.Conv3 = ConvBlock(filters[1], filters[2])
        self.Conv4 = ConvBlock(filters[2], filters[3])
        
        # Decoder (Upsampling)
        self.Up4 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.Att4 = AttentionGate(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = ConvBlock(filters[3], filters[2])
        
        self.Up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.Att3 = AttentionGate(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = ConvBlock(filters[2], filters[1])
        
        self.Up2 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.Att2 = AttentionGate(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2)
        self.Up_conv2 = ConvBlock(filters[1], filters[0])
        
        self.Conv_1x1 = nn.Conv2d(filters[0], out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder Pathway
        e1 = self.Conv1(x)
        
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)
        
        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)
        
        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)
        
        # Decoder Pathway with Attention Skip Connections
        d4 = self.Up4(e4)
        x4 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        x3 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        x2 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        out = self.Conv_1x1(d2)
        # Apply Sigmoid to get probability mask [0, 1]
        return torch.sigmoid(out)


class CloudDataset(Dataset):
    """
    Dataset wrapper for LISS-IV patches for cloud segmentation training.
    Uses spectral guess masks as weak-supervision labels.
    """
    def __init__(self, patches: List[np.ndarray], labels: List[np.ndarray]):
        self.patches = patches
        self.labels = labels

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert numpy (C, H, W) to float torch tensors
        patch = torch.from_numpy(self.patches[idx]).float()
        label = torch.from_numpy(self.labels[idx]).float().unsqueeze(0)  # Shape (1, H, W)
        return patch, label


def train_attention_unet(train_patches: List[np.ndarray], train_labels: List[np.ndarray], 
                         epochs: int = 5, batch_size: int = 4, lr: float = 1e-3) -> AttentionUNet:
    """
    Trains the Attention U-Net using BCE loss against spectral guess labels (refinement training).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing Attention U-Net training on {device}...")
    
    dataset = CloudDataset(train_patches, train_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = AttentionUNet(in_channels=3, out_channels=1).to(device)
    
    checkpoint_path = "checkpoints/unet_mask.pth"
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info(f"Successfully loaded pre-trained Attention U-Net checkpoint from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to load Attention U-Net checkpoint: {e}. Training from scratch.")
    else:
        logger.info("No pre-trained Attention U-Net checkpoint found. Initializing random weights.")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x.size(0)
            
        avg_loss = epoch_loss / len(dataset)
        logger.info(f"Epoch {epoch+1}/{epochs} - Binary Cross Entropy Loss: {avg_loss:.5f}")
        
    logger.info("Attention U-Net localized training complete.")
    
    # Save checkpoint
    try:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Saved fine-tuned Attention U-Net checkpoint to {checkpoint_path}")
    except Exception as e:
        logger.warning(f"Failed to save Attention U-Net checkpoint: {e}")
        
    return model


def generate_refined_mask(model: AttentionUNet, optical_patch: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Generates a binary cloud mask for a single LISS-IV optical patch.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Add batch dimension and convert to tensor
    x = torch.from_numpy(optical_patch).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        prob_mask = model(x).squeeze(0).squeeze(0).cpu().numpy()
        
    binary_mask = (prob_mask > threshold).astype(np.uint8)
    return binary_mask


if __name__ == "__main__":
    # Smoke test of model structure
    logger.info("Instantiating Attention U-Net structure for verification...")
    model = AttentionUNet(in_channels=3, out_channels=1)
    dummy_input = torch.randn(1, 3, 256, 256)
    dummy_output = model(dummy_input)
    logger.info(f"Verification successful. Output tensor shape: {dummy_output.shape}")
