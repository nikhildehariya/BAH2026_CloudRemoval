"""
Module 5: Seamless Post-Processing & Export
Implements 2D Gaussian blending for patch stitching and
georeferenced Cloud-Optimized GeoTIFF (COG) export using Rasterio.
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.windows import Window
from typing import Tuple, List, Dict, Any

# Setup Logger
logger = logging.getLogger("DhruvaPipeline.Postprocessing")


def generate_gaussian_weight_map(patch_size: int = 256, overlap: int = 64) -> np.ndarray:
    """
    Generates a 2D flat-topped Gaussian weight map.
    The map has a value of 1.0 in the center region (patch_size - 2*overlap)
    and decays exponentially (Gaussian roll-off) towards the boundaries.
    """
    weight = np.ones((patch_size, patch_size), dtype=np.float32)
    sigma = overlap / 2.5 # standard deviation for decay
    
    for i in range(patch_size):
        for j in range(patch_size):
            # Calculate distance to nearest boundary
            dist_x = min(i, patch_size - 1 - i)
            dist_y = min(j, patch_size - 1 - j)
            dist_edge = min(dist_x, dist_y)
            
            if dist_edge < overlap:
                # Gaussian decay formula
                weight[i, j] = np.exp(-((overlap - dist_edge) ** 2) / (2.0 * sigma ** 2))
                
    return weight


class SeamlessStitcher:
    """
    Accumulator engine to stitch patches back into a full grid.
    Uses a 2D Gaussian weight map to blend overlapping patch boundaries,
    eliminating seam artifacts and blocky boundaries.
    """
    def __init__(self, height: int, width: int, channels: int = 3, 
                 patch_size: int = 256, overlap: int = 64):
        self.height = height
        self.width = width
        self.channels = channels
        self.patch_size = patch_size
        self.overlap = overlap
        
        # Accumulators
        self.image_accumulator = np.zeros((channels, height, width), dtype=np.float32)
        self.weight_accumulator = np.zeros((height, width), dtype=np.float32)
        
        # Generate the shared blending weight map
        self.weight_map = generate_gaussian_weight_map(patch_size, overlap)

    def add_patch(self, window: Window, patch_data: np.ndarray):
        """
        Adds a reconstructed patch to the accumulator grids.
        """
        import gc
        import torch
        
        x_off, y_off = window.col_off, window.row_off
        h_w, w_w = window.height, window.width
        
        # Crop patch data and weight map if the window falls off-boundary (edge patches)
        p_data_cropped = patch_data[:, :h_w, :w_w]
        w_map_cropped = self.weight_map[:h_w, :w_w]
        
        # Accumulate weighted predictions
        # Multiply each channel by the 2D Gaussian weight map
        weighted_patch = p_data_cropped * w_map_cropped[np.newaxis, :, :]
        
        # Add to global arrays
        self.image_accumulator[:, y_off:y_off+h_w, x_off:x_off+w_w] += weighted_patch
        self.weight_accumulator[y_off:y_off+h_w, x_off:x_off+w_w] += w_map_cropped
        
        # Release cached GPU and RAM tensors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_final_reconstruction(self) -> np.ndarray:
        """
        Normalizes the accumulated image grid by the weight grid.
        Handles zero-weight divisions on boundaries.
        """
        logger.info("Normalizing and stitching all reconstructed patches...")
        
        # Add small epsilon to prevent divide-by-zero
        normalized_img = self.image_accumulator / (self.weight_accumulator[np.newaxis, :, :] + 1e-12)
        
        # Clip to valid reflectance scale
        return np.clip(normalized_img, 0.0, 1.0)


def export_cloud_optimized_geotiff(data: np.ndarray, src_metadata: Dict[str, Any], output_path: str):
    """
    Saves the reconstructed optical array as a Cloud-Optimized GeoTIFF (COG).
    Copies CRS, transform and georeferencing metadata from the source LISS-IV scene.
    
    Args:
        data: Reconstructed numpy array, shape (channels, height, width).
        src_metadata: Original metadata dict fetched from rasterio.src.meta.
        output_path: Destination path for the COG file.
    """
    logger.info(f"Exporting georeferenced output to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Update original metadata structure with COG compliant profiles
    cog_meta = src_metadata.copy()
    cog_meta.update({
        'driver': 'GTiff',
        'height': data.shape[1],
        'width': data.shape[2],
        'count': data.shape[0],
        'dtype': 'float32',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'compress': 'lzw',
        'interleave': 'band'
    })
    
    with rasterio.open(output_path, 'w', **cog_meta) as dst:
        # Write bands
        for band_idx in range(data.shape[0]):
            dst.write(data[band_idx].astype(np.float32), band_idx + 1)
            dst.set_band_description(band_idx + 1, f"Band {band_idx + 1}")
            
        # Build overviews (pyramids) for COG standard
        overviews = [2, 4, 8, 16]
        dst.build_overviews(overviews, rasterio.enums.Resampling.average)
        dst.update_tags(ns='rio_overview', resampling='average')
        
    logger.info("Cloud-Optimized GeoTIFF (COG) written successfully.")


if __name__ == "__main__":
    # Test Gaussian weights map
    weight_map = generate_gaussian_weight_map(256, 64)
    logger.info(f"Gaussian weight map generated. Center value: {weight_map[128, 128]:.4f}, Edge value: {weight_map[0, 0]:.4f}")
