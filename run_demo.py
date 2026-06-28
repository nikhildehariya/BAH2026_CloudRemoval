"""
Demo Runner & Dashboard Server
Runs the Python pipeline, exports GeoTIFF bands as display-ready PNGs,
and launches a local HTTP server to host the interactive dashboard.
"""

import os
import sys
import logging
import threading
import webbrowser
import http.server
import socketserver
import numpy as np
import rasterio

try:
    import cv2
except ImportError:
    cv2 = None

# Setup Logger
logger = logging.getLogger("DhruvaPipeline.DemoServer")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

PORT = 8000
ASSETS_DIR = "./assets"


def convert_tiff_to_png():
    """
    Reads the pipeline GeoTIFF outputs and exports them as normalized 8-bit PNG images
    suitable for web display.
    """
    if cv2 is None:
        logger.error("OpenCV is not installed. Cannot export GeoTIFFs to PNG. Run pip install opencv-python.")
        return False
        
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    # Paths
    cloudy_tif = "./data/raw/R2_L4_MX_20260615_087_054.tif"
    s1_tif = "./data/raw/S1A_IW_GRDH_1SDV_20260615T120000_ASC.tif"
    clean_gt_tif = "./data/raw/R2_L4_MX_20260615_087_054_GT.tif"
    reconstructed_tif = "./data/processed/TeamDhruva_LISS4_CloudFree.tif"
    
    logger.info("Converting georeferenced TIFF datasets to web-display PNG assets...")
    
    # 1. Save Cloudy LISS-IV (Green, Red, NIR channels -> RGB representation)
    with rasterio.open(cloudy_tif) as src:
        # Green=1, Red=2, NIR=3. We map Red, Green, Blue bands.
        # Let's map NIR, Red, Green as false color, or standard RGB
        # standard RGB representation for web
        g = src.read(1)
        r = src.read(2)
        n = src.read(3)
        # Scale each band independently to 0-255
        rgb = np.stack([n, r, g], axis=-1) # false color NIR/R/G makes vegetation pop, or standard
        # standard true color proxy: Red=channel 1, Green=channel 0, Blue=mix
        b = g * 0.8  # LISS-IV lacks Blue band, we simulate it
        rgb_true = np.stack([r, g, b], axis=-1)
        rgb_true = (rgb_true - rgb_true.min()) / (rgb_true.max() - rgb_true.min() + 1e-8)
        cv2.imwrite(os.path.join(ASSETS_DIR, "cloudy.png"), (rgb_true * 255).astype(np.uint8))

    # 2. Save Ground Truth LISS-IV (Cloud-Free)
    with rasterio.open(clean_gt_tif) as src:
        g = src.read(1)
        r = src.read(2)
        b = g * 0.8
        rgb_true = np.stack([r, g, b], axis=-1)
        rgb_true = (rgb_true - rgb_true.min()) / (rgb_true.max() - rgb_true.min() + 1e-8)
        cv2.imwrite(os.path.join(ASSETS_DIR, "ground_truth.png"), (rgb_true * 255).astype(np.uint8))

    # 3. Save Reconstructed LISS-IV
    with rasterio.open(reconstructed_tif) as src:
        # Reconstructed is float32
        g = src.read(1)
        r = src.read(2)
        b = g * 0.8
        rgb_true = np.stack([r, g, b], axis=-1)
        rgb_true = np.clip(rgb_true, 0.0, 1.0)
        cv2.imwrite(os.path.join(ASSETS_DIR, "reconstructed.png"), (rgb_true * 255).astype(np.uint8))

    # 4. Save Sentinel-1 SAR GRD channels
    with rasterio.open(s1_tif) as src:
        vv = src.read(1)
        vh = src.read(2)
        
        # Normalize to [0, 255]
        vv_norm = ((vv - vv.min()) / (vv.max() - vv.min() + 1e-8) * 255).astype(np.uint8)
        vh_norm = ((vh - vh.min()) / (vh.max() - vh.min() + 1e-8) * 255).astype(np.uint8)
        
        cv2.imwrite(os.path.join(ASSETS_DIR, "sar_vv.png"), vv_norm)
        cv2.imwrite(os.path.join(ASSETS_DIR, "sar_vh.png"), vh_norm)
        
        # Save a dual-pol color composite: VV, VH, ratio VV/VH
        ratio = vv / (vh + 1e-6)
        ratio_norm = ((ratio - ratio.min()) / (ratio.max() - ratio.min() + 1e-8) * 255).astype(np.uint8)
        sar_composite = np.stack([vv_norm, vh_norm, ratio_norm], axis=-1)
        cv2.imwrite(os.path.join(ASSETS_DIR, "sar_composite.png"), sar_composite)
        
    # 5. Generate and save the final binary cloud mask
    # We can read the cloud mask by simple threshold of cloudy vs ground-truth
    with rasterio.open(cloudy_tif) as src_c, rasterio.open(clean_gt_tif) as src_g:
        c_red = src_c.read(2).astype(np.float32)
        g_red = src_g.read(2).astype(np.float32)
        diff = np.abs(c_red - g_red * 255.0)
        # Create a clean binary mask
        mask = (diff > 40.0).astype(np.uint8) * 255
        # Smooth mask slightly to look clean in UI
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        cv2.imwrite(os.path.join(ASSETS_DIR, "cloud_mask.png"), mask)

    logger.info("PNG assets exported successfully to ./assets/")
    return True


def run_http_server():
    """
    Starts a simple HTTP server in the current working directory.
    """
    # Force handler to serve HTML/JS files correctly
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Configure socket reuse to prevent port conflict issues
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            logger.info(f"Local dashboard server listening on http://localhost:{PORT}")
            # Start browser
            webbrowser.open(f"http://localhost:{PORT}")
            httpd.serve_forever()
    except Exception as e:
        logger.error(f"Failed to start local web server: {e}")


def main():
    # Run the main processing pipeline to generate outputs
    logger.info("Step 1: Running the modular Python processing pipeline...")
    try:
        from pipeline.main import run_pipeline
        run_pipeline()
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        logger.warning("Pipeline execution failed. Attempting to proceed with asset conversion.")

    # Convert geotiffs to pngs for display
    logger.info("Step 2: Generating dashboard assets...")
    success = convert_tiff_to_png()
    if not success:
        logger.error("Could not generate PNG assets. Dashboard visualizer may lack images.")
        
    # Start web server
    logger.info("Step 3: Starting local web server...")
    server_thread = threading.Thread(target=run_http_server, daemon=True)
    server_thread.start()
    
    # Wait for keyboard interrupt to exit
    print("\n" + "="*50)
    print(f"Team Dhruva Dashboard running at: http://localhost:{PORT}")
    print("Press Ctrl+C to terminate the local server.")
    print("="*50 + "\n")
    
    try:
        # Keep main thread alive
        server_thread.join()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
