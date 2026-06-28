"""
Main Orchestration Script
Generates simulated geospatial datasets, runs the 5 modules end-to-end,
and prints quality metrics (L1, MS-SSIM, SAM, NDVI consistency).
"""

import os
import logging
import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
import torch.nn.functional as F

from pipeline.ingestion import run_ingestion_pipeline
from pipeline.preprocessing import AtmosphericCorrection6S, SubPixelCoRegistration, MemorySafePatchEngine
from pipeline.masking import generate_spectral_guess, train_attention_unet, generate_refined_mask
from pipeline.diffusion import KLAutoencoder, LatentDiffusionUNet, LatentDiffusionLoop
from pipeline.postprocessing import SeamlessStitcher, export_cloud_optimized_geotiff
from pipeline.utils import JointLoss

# Setup Logger
logger = logging.getLogger("DhruvaPipeline.Main")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def generate_simulated_geotiffs(output_dir: str):
    """
    Generates simulated georeferenced LISS-IV (cloudy) and Sentinel-1 (distorted) GeoTIFFs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Ground footprint parameters (Assam, India region)
    crs = 'EPSG:32646' # WGS 84 / UTM zone 46N
    transform_opt = from_origin(250000, 2900000, 5.8, 5.8)  # 5.8m resolution
    transform_sar = from_origin(250010, 2900010, 10.0, 10.0) # 10m resolution, slightly offset
    
    h_opt, w_opt = 512, 512
    h_sar, w_sar = 300, 300
    
    logger.info("Synthesizing mock geospatial datasets for testing...")
    
    # Create clean terrain features (rivers, roads, vegetation)
    # We use spatial gradients to simulate features
    grid_y, grid_x = np.mgrid[0:h_opt, 0:w_opt]
    river_feature = np.sin(grid_x / 50.0) * np.cos(grid_y / 100.0)
    vegetation_feature = np.sin((grid_x + grid_y) / 80.0)
    urban_feature = (np.sin(grid_x / 5.0) > 0.8).astype(np.float32) * (np.sin(grid_y / 5.0) > 0.8).astype(np.float32)
    
    # Base reflectance
    green = np.clip(0.1 + 0.1 * river_feature + 0.05 * urban_feature, 0, 1)
    red = np.clip(0.08 + 0.02 * river_feature + 0.04 * urban_feature, 0, 1)
    nir = np.clip(0.35 + 0.25 * vegetation_feature - 0.15 * river_feature, 0, 1)
    
    # Introduce clouds (blob of high reflectance in the center)
    cloud_dist = np.sqrt((grid_x - 256)**2 + (grid_y - 256)**2)
    cloud_intensity = np.clip(1.0 - (cloud_dist / 180.0), 0.0, 1.0)
    cloud_mask = (cloud_intensity > 0.35).astype(np.float32)
    
    # Mix clouds into optical bands
    green_cloudy = green * (1 - cloud_mask) + cloud_intensity * 0.85
    red_cloudy = red * (1 - cloud_mask) + cloud_intensity * 0.80
    nir_cloudy = nir * (1 - cloud_mask) + cloud_intensity * 0.65 # clouds reflect less in NIR than visible
    
    # Scale to 8-bit DN (Digital Numbers)
    opt_dn = np.stack([green_cloudy, red_cloudy, nir_cloudy]) * 255.0
    opt_dn = np.clip(opt_dn, 0, 255).astype(np.uint8)
    
    # Save cloudy LISS-IV scene
    liss4_path = os.path.join(output_dir, "R2_L4_MX_20260615_087_054.tif")
    with rasterio.open(
        liss4_path, 'w',
        driver='GTiff', height=h_opt, width=w_opt, count=3,
        dtype='uint8', crs=crs, transform=transform_opt
    ) as dst:
        dst.write(opt_dn[0], 1)
        dst.write(opt_dn[1], 2)
        dst.write(opt_dn[2], 3)
        # Update metadata tags for physical correction module extraction
        dst.update_tags(
            SOLAR_ZENITH="32.5",
            SOLAR_AZIMUTH="122.4",
            ACQUISITION_DATE="2026-06-15"
        )
        
    # Create Sentinel-1 SAR GRD VV/VH (Radar signals completely penetrate clouds)
    # We downsample the primary clean structures and add radar speckle noise
    # Rescale to 10m grid and apply coordinate shift
    sar_grid_y, sar_grid_x = np.mgrid[0:h_sar, 0:w_sar]
    
    # Map coordinates from SAR grid to Optical grid for ground truth feature correspondence
    mapped_x = (sar_grid_x * 10.0 + 10) / 5.8
    mapped_y = (sar_grid_y * 10.0 + 10) / 5.8
    
    # Sample structural features in SAR coordinates
    river_sar = np.sin(mapped_x / 50.0) * np.cos(mapped_y / 100.0)
    urban_sar = (np.sin(mapped_x / 5.0) > 0.8).astype(np.float32) * (np.sin(mapped_y / 5.0) > 0.8).astype(np.float32)
    
    # Radar reflectivity: Water absorbs radar (low returns), Urban surfaces bounce radar (high double-bounce returns)
    vv = 0.5 - 0.3 * river_sar + 0.4 * urban_sar
    vh = 0.2 - 0.1 * river_sar + 0.15 * urban_sar
    
    # Add multiplicative speckle noise characteristic of radar
    speckle_vv = np.random.gamma(4, 0.25, size=(h_sar, w_sar))
    speckle_vh = np.random.gamma(4, 0.25, size=(h_sar, w_sar))
    vv = np.clip(vv * speckle_vv, 0, 1.0)
    vh = np.clip(vh * speckle_vh, 0, 1.0)
    
    # Save Sentinel-1 scene
    s1_path = os.path.join(output_dir, "S1A_IW_GRDH_1SDV_20260615T120000_ASC.tif")
    with rasterio.open(
        s1_path, 'w',
        driver='GTiff', height=h_sar, width=w_sar, count=2,
        dtype='float32', crs=crs, transform=transform_sar
    ) as dst:
        dst.write(vv.astype(np.float32), 1)
        dst.write(vh.astype(np.float32), 2)
        
    # Also save a ground-truth cloud-free optical file to calculate metrics at the end
    gt_path = os.path.join(output_dir, "R2_L4_MX_20260615_087_054_GT.tif")
    with rasterio.open(
        gt_path, 'w',
        driver='GTiff', height=h_opt, width=w_opt, count=3,
        dtype='float32', crs=crs, transform=transform_opt
    ) as dst:
        dst.write(green.astype(np.float32), 1)
        dst.write(red.astype(np.float32), 2)
        dst.write(nir.astype(np.float32), 3)

    logger.info("Simulated GeoTIFF files generated.")
    return liss4_path, s1_path, gt_path


def run_pipeline():
    # Setup directories
    raw_dir = "./data/raw"
    processed_dir = "./data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Ingestion / Data generation
    liss4_raw, s1_raw, gt_raw = generate_simulated_geotiffs(raw_dir)
    
    # Bbox/date params
    bbox = (91.5, 26.0, 92.0, 26.5)
    date_range = ("2026-06-01", "2026-06-15")
    selected_liss4, selected_s1 = run_ingestion_pipeline(bbox, date_range, raw_dir)
    
    # 2. Preprocessing
    # Atmospheric Correction (py6S fallback)
    logger.info("Running Module 2 Atmospheric Correction...")
    atmos = AtmosphericCorrection6S.from_tiff_metadata(liss4_raw)
    
    with rasterio.open(liss4_raw) as src:
        liss4_meta = src.meta.copy()
        raw_green = src.read(1)
        raw_red = src.read(2)
        raw_nir = src.read(3)
        
    cal_green = atmos.calibrate_band(raw_green, "Green")
    cal_red = atmos.calibrate_band(raw_red, "Red")
    cal_nir = atmos.calibrate_band(raw_nir, "NIR")
    optical_cal = np.stack([cal_green, cal_red, cal_nir]) # shape: (3, 512, 512)
    
    # Sub-pixel TPS co-registration
    logger.info("Running Module 2 Sub-Pixel TPS Co-registration...")
    with rasterio.open(s1_raw) as src:
        s1_vv = src.read(1)
        s1_vh = src.read(2)
        
    # Upscale SAR to optical grid (5.8m)
    import cv2
    s1_vv_up = cv2.resize(s1_vv, (liss4_meta['width'], liss4_meta['height']), interpolation=cv2.INTER_CUBIC)
    s1_vh_up = cv2.resize(s1_vh, (liss4_meta['width'], liss4_meta['height']), interpolation=cv2.INTER_CUBIC)
    
    # Feature matching (using Optical NIR vs SAR VV)
    coreg = SubPixelCoRegistration()
    opt_pts, sar_pts = coreg.extract_features(cal_nir, s1_vv_up)
    
    # Solve TPS & warp SAR features
    coef_x = coreg.solve_tps(sar_pts, opt_pts)
    warped_vv = coreg.warp_tps(s1_vv_up, sar_pts, coef_x, (liss4_meta['height'], liss4_meta['width']))
    warped_vh = coreg.warp_tps(s1_vh_up, sar_pts, coef_x, (liss4_meta['height'], liss4_meta['width']))
    sar_warped = np.stack([warped_vv, warped_vh]) # shape: (2, 512, 512)
    
    # Temporarily save preprocessed results to run patch extraction
    temp_optical_path = os.path.join(processed_dir, "temp_optical.tif")
    with rasterio.open(temp_optical_path, 'w', **liss4_meta) as dst:
        dst.write((optical_cal * 255.0).astype(np.uint8))
        
    temp_sar_path = os.path.join(processed_dir, "temp_sar.tif")
    sar_meta = liss4_meta.copy()
    sar_meta.update({'count': 2, 'dtype': 'float32'})
    with rasterio.open(temp_sar_path, 'w', **sar_meta) as dst:
        dst.write(sar_warped.astype(np.float32))
        
    # Patch extraction (sliding window 256x256, 64 overlap)
    logger.info("Running Module 2 Memory-Safe Patch Extraction...")
    patch_engine = MemorySafePatchEngine(patch_size=256, overlap=64)
    opt_patches = patch_engine.extract_patches(temp_optical_path)
    sar_patches = patch_engine.extract_patches(temp_sar_path)
    
    # 3. Masking Module
    logger.info("Running Module 3 Two-Tier Cloud & Shadow Masking...")
    # Tier 1: Red-band spectral guess
    coarse_masks = []
    for window, patch in opt_patches:
        # Patch shape: (3, 256, 256). Red band is channel 1 (0-indexed). Scale to [0,1]
        red_band = patch[1] / 255.0
        mask_guess = generate_spectral_guess(red_band, threshold=0.25)
        coarse_masks.append(mask_guess)
        
    # Prepare datasets for U-Net refinement training
    opt_patches_norm = [(p[1] / 255.0).astype(np.float32) for p in opt_patches]
    
    # Train lightweight U-Net for refinement (1 epoch for demo speed)
    unet_mask_model = train_attention_unet(opt_patches_norm, coarse_masks, epochs=1, batch_size=2)
    
    # Generate final refined masks
    refined_masks = []
    for patch in opt_patches_norm:
        ref_mask = generate_refined_mask(unet_mask_model, patch, threshold=0.5)
        refined_masks.append(ref_mask)
        
    # 4. Core Generative Loop (Diffusion simulation)
    logger.info("Running Module 4 Cross-Attention Latent Diffusion Loop...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model definitions
    ae = KLAutoencoder().to(device)
    unet_diff = LatentDiffusionUNet().to(device)
    diffusion_loop = LatentDiffusionLoop(unet=unet_diff, num_timesteps=10) # 10 steps for demo execution speed
    
    # Simulate reconstruction for each patch
    reconstructed_patches = []
    for idx, (window, opt_patch) in enumerate(opt_patches):
        # Normalize
        opt_tensor = torch.from_numpy(opt_patches_norm[idx]).float().unsqueeze(0).to(device)
        sar_tensor = torch.from_numpy(sar_patches[idx][1]).float().unsqueeze(0).to(device)
        mask_tensor = torch.from_numpy(refined_masks[idx]).float().unsqueeze(0).unsqueeze(0).to(device)
        
        # In a real pipeline, the models are trained. Here we simulate the process
        # by blending the target ground truth (reconstructed) and the cloudy input based on the mask,
        # adding minor reconstruction noise to demonstrate the generative diffusion effects.
        with torch.no_grad():
            # Compress to latents
            z_opt = ae.reparameterize(*ae.encode(opt_tensor))
            
            # Map SAR (2 channels) to prior latent (4 channels) using padding/convolutions
            # For simulation, we create a condition latent matching the spatial dimensions
            z_cond = torch.zeros_like(z_opt)
            z_cond[:, :2] = F.interpolate(sar_tensor, size=z_opt.shape[2:], mode='bilinear')
            
            mask_lat = F.interpolate(mask_tensor, size=z_opt.shape[2:], mode='nearest')
            
            # Denoise via Latent Diffusion
            z_recon = diffusion_loop.sample_reverse(z_cond, mask_lat)
            
            # Reconstruction equation: blend input + conditional SAR structures
            # We inject some high-frequency detail from SAR into the reconstructed zones
            z_final = (1.0 - mask_lat) * z_opt + mask_lat * (z_recon * 0.15 + z_cond * 0.85)
            
            # Decode reconstructed latents back to image space
            patch_recon = ae.decode(z_final).squeeze(0).cpu().numpy()
            
        reconstructed_patches.append((window, patch_recon))
        
    # 5. Seamless Post-processing & COG Export
    logger.info("Running Module 5 Seamless Post-processing & Gaussian Stitching...")
    stitcher = SeamlessStitcher(
        height=liss4_meta['height'], 
        width=liss4_meta['width'], 
        channels=3, 
        patch_size=256, 
        overlap=64
    )
    
    for window, patch_data in reconstructed_patches:
        stitcher.add_patch(window, patch_data)
        
    final_reconstructed_optical = stitcher.get_final_reconstruction()
    
    # Save the final georeferenced COG product
    output_cog_path = os.path.join(processed_dir, "TeamDhruva_LISS4_CloudFree.tif")
    export_cloud_optimized_geotiff(final_reconstructed_optical, liss4_meta, output_cog_path)
    
    # 6. Scientific Metric Assessment
    # Load ground-truth cloud-free image to verify quality
    with rasterio.open(gt_raw) as src:
        gt_optical = src.read() # shape: (3, 512, 512)
        
    # Compute metrics using JointLoss
    loss_eval = JointLoss(channels=3)
    pred_t = torch.from_numpy(final_reconstructed_optical).float().unsqueeze(0)
    target_t = torch.from_numpy(gt_optical).float().unsqueeze(0)
    
    total_loss, metrics = loss_eval(pred_t, target_t)
    
    logger.info("=========================================")
    logger.info("TEAM DHRUVA HACKATHON PERFORMANCE METRICS")
    logger.info("=========================================")
    logger.info(f" Mean Absolute Error (L1): {metrics['L1']:.5f}")
    logger.info(f" Multi-Scale SSIM:         {1.0 - metrics['MS-SSIM']:.5f}")
    logger.info(f" Spectral Angle Mapper:    {metrics['SAM']:.5f} rad")
    logger.info(f" NDVI Consistency Score:   {(1.0 - metrics['NDVI']) * 100:.2f}%")
    logger.info(f" Joint Multi-Objective Loss: {metrics['Total']:.5f}")
    logger.info("=========================================")
    
    # Clean up temp assets
    os.remove(temp_optical_path)
    os.remove(temp_sar_path)
    
    logger.info("Pipeline run completed successfully.")


if __name__ == "__main__":
    run_pipeline()
