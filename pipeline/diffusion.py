"""
Module 4: Cross-Attention Generative Loop (The Model Core)
Implements the KL-Autoencoder space-compression, Restormer backbone with
Multi-Dilation Gated Attention, Decoupled Masked Cross-Attention,
Sinusoidal Positional Time-Embedding, and the Latent Diffusion Loop.
"""

import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any

# Setup Logger
logger = logging.getLogger("DhruvaPipeline.Diffusion")


class KLAutoencoder(nn.Module):
    """
    Domain-Specific KL-Autoencoder that compresses the 3-band LISS-IV optical images
    by a factor of 8 into a low-dimensional latent space (z) to filter out atmospheric noise.
    Uses sequential Conv2D layers with stride=2 and GELU activations.
    """
    def __init__(self, in_channels: int = 3, latent_channels: int = 4):
        super(KLAutoencoder, self).__init__()
        
        # Encoder: H x W x 3 -> H/8 x W/8 x (latent_channels * 2)
        # Sequential Conv2D layers with stride=2 and GeLU activations
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Downsample to H/2
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # Downsample to H/4
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),# Downsample to H/8
            nn.GELU(),
            nn.Conv2d(128, latent_channels * 2, kernel_size=3, stride=1, padding=1) # Output Mean & Logvar
        )
        
        # Decoder: H/8 x W/8 x latent_channels -> H x W x 3
        # Sequential Transposed Conv2D layers with stride=2 and GeLU activations
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), # Upsample to H/4
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Upsample to H/2
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # Upsample to H
            nn.GELU(),
            nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() # Keeps reflectance values bounded within [0, 1]
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        moments = self.encoder(x)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decode(z)
        return x_recon, mean, logvar


class DecoupledCrossAttention(nn.Module):
    """
    Decoupled Spatial Masked Cross-Attention Block.
    Attention(Q, K, V) = Softmax(Q * K^T / sqrt(d_k)) * V
    
    If the cloud mask M is active (M=1), attention dynamically suppresses optical queries
    self-attention logits (-1e9 logit masks) and heavily prioritizes spatial structural keys/values
    streaming from Sentinel-1 SAR paths. If M=0, it suppresses SAR keys.
    """
    def __init__(self, channels: int, cond_channels: int = 2, num_heads: int = 4, bias: bool = True):
        super(DecoupledCrossAttention, self).__init__()
        self.channels = channels
        self.cond_channels = cond_channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Self-Attention projections (for optical features)
        self.q_opt = nn.Linear(channels, channels, bias=bias)
        self.k_opt = nn.Linear(channels, channels, bias=bias)
        self.v_opt = nn.Linear(channels, channels, bias=bias)
        
        # Cross-Attention projections (for SAR structural conditions)
        self.k_sar = nn.Linear(cond_channels, channels, bias=bias)
        self.v_sar = nn.Linear(cond_channels, channels, bias=bias)
        
        self.out_proj = nn.Linear(channels, channels, bias=bias)

    def forward(self, z_opt: torch.Tensor, z_cond: torch.Tensor, mask_latent: torch.Tensor) -> torch.Tensor:
        """
        z_opt: Optical latent feature tensor, shape (B, C, H, W)
        z_cond: SAR + Historical Prior condition feature tensor, shape (B, cond_C, H, W)
        mask_latent: Downsampled binary cloud mask, shape (B, 1, H, W) where 1=Cloud, 0=Clear
        """
        B, C, H, W = z_opt.shape
        N = H * W
        
        # Reshape to token sequences (B, N, channels)
        z_opt_flat = z_opt.permute(0, 2, 3, 1).reshape(B, N, C)
        z_opt_keys_flat = z_opt.permute(0, 2, 3, 1).reshape(B, N, C)
        z_opt_vals_flat = z_opt.permute(0, 2, 3, 1).reshape(B, N, C)
        
        z_cond_keys_flat = z_cond.permute(0, 2, 3, 1).reshape(B, N, self.cond_channels)
        z_cond_vals_flat = z_cond.permute(0, 2, 3, 1).reshape(B, N, self.cond_channels)
        
        # Map to attention heads
        # Q shape: (B, h, N, d)
        q = self.q_opt(z_opt_flat).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K, V Self-Attention
        k_self = self.k_opt(z_opt_keys_flat).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v_self = self.v_opt(z_opt_vals_flat).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K, V Cross-Attention
        k_cross = self.k_sar(z_cond_keys_flat).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v_cross = self.v_sar(z_cond_vals_flat).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Concatenate keys & values along token dimension
        # K shape: (B, h, 2N, d)
        # V shape: (B, h, 2N, d)
        k = torch.cat([k_self, k_cross], dim=2)
        v = torch.cat([v_self, v_cross], dim=2)
        
        # Dynamic Spatial Masking:
        # Suppress optical self-attention keys (j < N) under cloud (mask=1)
        # Suppress SAR cross-attention keys (j >= N) under clear sky (mask=0)
        m = mask_latent.reshape(B, 1, 1, N) # (B, 1, 1, N)
        
        attn_mask = torch.zeros(B, 1, 1, 2 * N, device=z_opt.device, dtype=q.dtype)
        attn_mask[:, :, :, :N] = m * -1e9          # suppress self-attention where cloudy
        attn_mask[:, :, :, N:] = (1.0 - m) * -1e9  # suppress SAR features where clear-sky
        
        # Call PyTorch's native memory-efficient scaled dot product attention
        # Drops the memory footprint from O(N^2) to O(N) by invoking FlashAttention/Memory-Efficient kernels
        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        ) # Shape: (B, num_heads, N, head_dim)
        
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return out


class GatedDconvFeedForward(nn.Module):
    """
    Gated Dilation Convolutional Feed-Forward Network (GDFN).
    Uses parallel multi-dilation (dilation=1 & dilation=2) depthwise convolutions
    to resolve multi-scale contextual fields without blowing up GPU memory.
    """
    def __init__(self, channels: int, expansion_factor: float = 2.66):
        super(GatedDconvFeedForward, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        
        # Projection and split layers
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1)
        
        # Multi-Dilation parallel depthwise convolutions
        self.dwconv_dil1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=hidden_channels, dilation=1)
        self.dwconv_dil2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=2, groups=hidden_channels, dilation=2)
        
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project & split channels
        x1, x2 = self.project_in(x).chunk(2, dim=1)
        
        # Multi-scale gating
        gated = self.dwconv_dil1(x1) * F.gelu(self.dwconv_dil2(x2))
        
        out = self.project_out(gated)
        return out


class RestormerBlock(nn.Module):
    """
    Restormer Transformer block integrating Decoupled Spatial Masked Cross-Attention
    and Multi-Dilation Gated Feed-Forward network.
    """
    def __init__(self, channels: int, cond_channels: int = 2, num_heads: int = 4):
        super(RestormerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = DecoupledCrossAttention(channels=channels, cond_channels=cond_channels, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GatedDconvFeedForward(channels=channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        # Attention Stage
        x_norm = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        x_norm = self.norm1(x_norm).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.attn(x_norm, cond, mask)
        
        # FFN Stage
        x_norm = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        x_norm = self.norm2(x_norm).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.ffn(x_norm)
        
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Computes sinusoidal positional time encodings to embed diffusion timesteps.
    """
    def __init__(self, embedding_dim: int):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: shape (B, 1)
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) / half_dim
        emb = torch.exp(exponent)
        emb = timesteps @ emb.unsqueeze(0) # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class LatentDiffusionUNet(nn.Module):
    """
    Sinusoidal time-embedded Denoising U-Net containing encoder-decoder Down/Up Conv2D blocks
    and Multi-Dilation Restormer attention modules, resolving skip connections.
    """
    def __init__(self, latent_channels: int = 4, cond_channels: int = 2, model_channels: int = 64):
        super(LatentDiffusionUNet, self).__init__()
        
        # Sinusoidal time MLP projection
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(model_channels),
            nn.Linear(model_channels, model_channels * 4),
            nn.GELU(),
            nn.Linear(model_channels * 4, model_channels * 4)
        )
        
        # Downsampling path
        self.conv_in = nn.Conv2d(latent_channels, model_channels, kernel_size=3, padding=1)
        self.down = nn.Conv2d(model_channels, model_channels * 2, kernel_size=4, stride=2, padding=1) # Downsample to H/2
        self.enc_block = RestormerBlock(channels=model_channels * 2, cond_channels=cond_channels)
        
        # Middle bottleneck block
        self.mid_block = RestormerBlock(channels=model_channels * 2, cond_channels=cond_channels)
        
        # Upsampling path
        self.up = nn.ConvTranspose2d(model_channels * 2, model_channels, kernel_size=4, stride=2, padding=1) # Upsample to H
        self.dec_block = RestormerBlock(channels=model_channels, cond_channels=cond_channels)
        
        # Exit projection
        self.conv_out = nn.Conv2d(model_channels, latent_channels, kernel_size=3, padding=1)
        
        # Time-projection linear layers
        self.time_proj_enc = nn.Linear(model_channels * 4, model_channels * 2)
        self.time_proj_mid = nn.Linear(model_channels * 4, model_channels * 2)
        self.time_proj_dec = nn.Linear(model_channels * 4, model_channels)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        z_t: noisy optical latent, shape (B, 4, H, W)
        t: timestep tensor, shape (B, 1)
        cond: SAR condition latent, shape (B, 4, H, W)
        mask: binary cloud mask, shape (B, 1, H, W)
        """
        # Embed timesteps
        t_emb = self.time_embed(t) # (B, model_channels * 4)
        
        # Encoder downsampling
        h1 = self.conv_in(z_t) # (B, model_channels, H, W)
        
        h2 = self.down(h1) # (B, model_channels * 2, H/2, W/2)
        t_enc = self.time_proj_enc(t_emb).unsqueeze(-1).unsqueeze(-1)
        h2 = h2 + t_enc
        
        # Resize condition inputs to match downsampled resolution
        cond_down = F.interpolate(cond, size=h2.shape[2:], mode='bilinear', align_corners=False)
        mask_down = F.interpolate(mask, size=h2.shape[2:], mode='nearest')
        
        h2 = self.enc_block(h2, cond_down, mask_down)
        
        # Mid bottleneck block
        t_mid = self.time_proj_mid(t_emb).unsqueeze(-1).unsqueeze(-1)
        h2 = self.mid_block(h2 + t_mid, cond_down, mask_down)
        
        # Decoder upsampling
        h3 = self.up(h2) # (B, model_channels, H, W)
        h3 = h3 + h1     # Skip connection
        
        t_dec = self.time_proj_dec(t_emb).unsqueeze(-1).unsqueeze(-1)
        h3 = h3 + t_dec
        
        h3 = self.dec_block(h3, cond, mask)
        
        out_noise = self.conv_out(h3)
        return out_noise


class LatentDiffusionLoop:
    """
    Orchestrates the forward and reverse diffusion sampling loops within LISS-IV latent spaces.
    Uses DDPM/DDIM inspired scheduler equations.
    """
    def __init__(self, unet: LatentDiffusionUNet, num_timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.unet = unet
        self.num_timesteps = num_timesteps
        
        # Linear scheduler parameters
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, z_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adds random Gaussian noise to z_0 at timestep t.
        """
        noise = torch.randn_like(z_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(z_0.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(z_0.device)
        z_t = sqrt_alpha * z_0 + sqrt_one_minus_alpha * noise
        return z_t, noise

    def sample_reverse(self, cond: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Reverse denoising diffusion process (inference).
        Reconstructs latent states recursively using the U-Net.
        """
        device = cond.device
        B, C, H, W = cond.shape
        # Initialize noise with the correct 4 optical latent channels
        z_t = torch.randn(B, 4, H, W, device=device)
        
        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.tensor([[t_idx]], dtype=torch.float32, device=device).expand(B, 1)
            
            with torch.no_grad():
                predicted_noise = self.unet(z_t, t, cond, mask)
                
            beta = self.betas[t_idx].to(device)
            alpha = self.alphas[t_idx].to(device)
            alpha_cumprod = self.alphas_cumprod[t_idx].to(device)
            sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t_idx].to(device)
            
            # Predict mean
            mean = (1.0 / torch.sqrt(alpha)) * (z_t - (beta / sqrt_one_minus_alpha_cumprod) * predicted_noise)
            
            if t_idx > 0:
                noise = torch.randn_like(z_t)
                sigma = torch.sqrt(beta * (1.0 - self.alphas_cumprod_prev[t_idx].to(device)) / (1.0 - alpha_cumprod))
                z_t = mean + sigma * noise
            else:
                z_t = mean
                
        return z_t


if __name__ == "__main__":
    logger.info("Initializing Latent Diffusion module verification...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Verify models
    ae = KLAutoencoder().to(device)
    unet = LatentDiffusionUNet().to(device)
    
    # Test shapes
    opt = torch.rand(1, 3, 256, 256).to(device)
    mask = torch.ones(1, 1, 256, 256).to(device)
    
    # Compress
    recon, mean, logvar = ae(opt)
    z = ae.reparameterize(mean, logvar)
    logger.info(f"Autoencoder validation: input={opt.shape}, latent={z.shape}, recon={recon.shape}")
    
    # UNet
    t = torch.tensor([[5.0]]).to(device)
    z_cond = torch.rand(1, 2, z.shape[2], z.shape[3]).to(device)
    mask_lat = F.interpolate(mask, size=z.shape[2:], mode='nearest')
    
    noise_pred = unet(z, t, z_cond, mask_lat)
    logger.info(f"UNet validation: output shape={noise_pred.shape}")
