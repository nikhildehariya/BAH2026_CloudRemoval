"""
Helper Utilities and Remote Sensing Loss Functions
Implements Spectral Angle Mapper (SAM) Loss, NDVI Consistency Loss,
and Multi-Scale Structural Similarity (MS-SSIM) Loss in PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SAMLoss(nn.Module):
    """
    Spectral Angle Mapper (SAM) Loss.
    Treats multi-spectral pixels as high-dimensional vectors and minimizes the spectral angle between them.
    Guarantees radiometric and band-ratio integrity.
    """
    def __init__(self, eps: float = 1e-8):
        super(SAMLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Tensor of shape (B, C, H, W)
            target: Tensor of shape (B, C, H, W)
        """
        # Calculate dot product along the spectral channel dimension
        dot_product = torch.sum(pred * target, dim=1)
        
        # Calculate L2 norms along the spectral channel dimension
        norm_pred = torch.sqrt(torch.sum(pred ** 2, dim=1))
        norm_target = torch.sqrt(torch.sum(target ** 2, dim=1))
        
        # Calculate spectral angle (theta)
        # Cosine similarity: cos_theta = dot_product / (norm_pred * norm_target)
        cos_theta = dot_product / (norm_pred * norm_target + self.eps)
        
        # Clamp to avoid NaN gradients at bounds [-1, 1]
        cos_theta = torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)
        
        # Spectral angle in radians: arccos(cos_theta)
        sam_angle = torch.acos(cos_theta)
        
        return torch.mean(sam_angle)


class NDVIConsistencyLoss(nn.Module):
    """
    NDVI Consistency Loss.
    Computes NDVI = (NIR - Red) / (NIR + Red) on predicted and target patches.
    Minimizes the Mean Absolute Error (MAE) between the two NDVI maps.
    Assumes standard channel indices: Green = 0, Red = 1, NIR = 2.
    """
    def __init__(self, red_idx: int = 1, nir_idx: int = 2, eps: float = 1e-8):
        super(NDVIConsistencyLoss, self).__init__()
        self.red_idx = red_idx
        self.nir_idx = nir_idx
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Extract Red and NIR bands
        pred_red = pred[:, self.red_idx, :, :]
        pred_nir = pred[:, self.nir_idx, :, :]
        
        target_red = target[:, self.red_idx, :, :]
        target_nir = target[:, self.nir_idx, :, :]
        
        # Compute NDVI
        pred_ndvi = (pred_nir - pred_red) / (pred_nir + pred_red + self.eps)
        target_ndvi = (target_nir - target_red) / (target_nir + target_red + self.eps)
        
        # Enforce boundary checks [-1, 1]
        pred_ndvi = torch.clamp(pred_ndvi, -1.0, 1.0)
        target_ndvi = torch.clamp(target_ndvi, -1.0, 1.0)
        
        # Compute L1 loss on the NDVI maps
        return F.l1_loss(pred_ndvi, target_ndvi)


# SSIM Helper functions for MS-SSIM
def gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    x = torch.arange(window_size).float() - window_size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    return gauss / gauss.sum()


def create_window(window_size: int, channel: int) -> torch.Tensor:
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, 
         window_size: int, channel: int, size_average: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    if size_average:
        return ssim_map.mean(), cs_map.mean()
    else:
        return ssim_map.mean(dim=(1,2,3)), cs_map.mean(dim=(1,2,3))


class MSSSIMLoss(nn.Module):
    """
    Multi-Scale Structural Similarity Index (MS-SSIM) Loss.
    Evaluates similarities over downsampled scales to capture structural features.
    """
    def __init__(self, window_size: int = 11, size_average: bool = True, channel: int = 3):
        super(MSSSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.register_buffer("window", create_window(window_size, channel))

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # Check window channel count alignment
        if img1.shape[1] != self.channel:
            self.channel = img1.shape[1]
            self.window = create_window(self.window_size, self.channel).to(img1.device)
            
        levels = 4
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.3700], device=img1.device)
        mcs = []
        
        img1_scale = img1
        img2_scale = img2
        
        for i in range(levels):
            ssim_val, cs_val = ssim(img1_scale, img2_scale, self.window, self.window_size, self.channel, self.size_average)
            mcs.append(cs_val)
            
            if i < levels - 1:
                # Downsample to next scale
                img1_scale = F.avg_pool2d(img1_scale, 2)
                img2_scale = F.avg_pool2d(img2_scale, 2)
                
        # Calculate combined MS-SSIM
        mcs_tensor = torch.stack(mcs)
        # Handle cases where elements are slightly negative
        mcs_tensor = torch.clamp(mcs_tensor, min=1e-8)
        
        ms_ssim_val = torch.prod(mcs_tensor[:-1] ** weights[:-1]) * (ssim_val ** weights[-1])
        return 1.0 - ms_ssim_val


class JointLoss(nn.Module):
    """
    Combined joint loss optimization framework for Analysis-Ready Data.
    L_total = lambda_1*L1 + lambda_2*L_MSSSIM + lambda_3*L_SAM + lambda_4*L_NDVI
    """
    def __init__(self, l1_weight: float = 1.0, msssim_weight: float = 0.5, 
                 sam_weight: float = 0.2, ndvi_weight: float = 0.3, channels: int = 3):
        super(JointLoss, self).__init__()
        self.l1_weight = l1_weight
        self.msssim_weight = msssim_weight
        self.sam_weight = sam_weight
        self.ndvi_weight = ndvi_weight
        
        self.msssim = MSSSIMLoss(channel=channels)
        self.sam = SAMLoss()
        self.ndvi = NDVIConsistencyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        l1 = F.l1_loss(pred, target)
        loss_msssim = self.msssim(pred, target)
        loss_sam = self.sam(pred, target)
        loss_ndvi = self.ndvi(pred, target)
        
        total_loss = (self.l1_weight * l1 + 
                      self.msssim_weight * loss_msssim + 
                      self.sam_weight * loss_sam + 
                      self.ndvi_weight * loss_ndvi)
                      
        metrics = {
            "L1": l1.item(),
            "MS-SSIM": loss_msssim.item(),
            "SAM": loss_sam.item(),
            "NDVI": loss_ndvi.item(),
            "Total": total_loss.item()
        }
        return total_loss, metrics
