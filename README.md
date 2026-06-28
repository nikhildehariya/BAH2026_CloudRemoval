# BAH 2026: Cloud Removal & Reconstruction for LISS-IV Satellite Imagery
### Team Dhruva | Project Blueprint

This repository implements the **Team Dhruva** project submission for **Problem Statement 2** of the **Bharatiya Antariksh Hackathon (BAH) 2026**. 

Our core innovation is a **Decoupled Cross-Attention Multimodal Latent Diffusion Model (cLDM)** that uses Sentinel-1 Synthetic Aperture Radar (SAR) GRD intensities as a physical structural anchor to reconstruct cloud-covered optical spectral bands in RESOURCESAT-2 LISS-IV imagery without data hallucination.

---

## Repository Structure

* `pipeline/`: Modular Python core files.
  * [ingestion.py](pipeline/ingestion.py): Natively reads TIFF data using Rasterio and prints scene parameters (shapes, CRS, resolution).
  * [preprocessing.py](pipeline/preprocessing.py): Implements Py6S atmospheric correction with astronomical TOA-to-BOA fallback equations, tie-point feature matching, and Thin-Plate Spline (TPS) grid warping.
  * [masking.py](pipeline/masking.py): Two-tier cloud and shadow masking using spectral Red thresholding and PyTorch Attention U-Net.
  * [diffusion.py](pipeline/diffusion.py): Low-dimensional KL-Autoencoder, Restormer blocks with Gated Dconv Feed-Forward Network, and the decoupled spatial masked cross-attention layers.
  * [postprocessing.py](pipeline/postprocessing.py): Stitching patches back via flat-topped 2D Gaussian blending to eliminate border grid seams, and exporting Cloud-Optimized GeoTIFF (COG).
  * [utils.py](pipeline/utils.py): Differentiable validation loss metrics: Spectral Angle Mapper (SAM) loss, NDVI Consistency loss, and MS-SSIM.
  * [main.py](pipeline/main.py): Local orchestrator script running the training/inference simulation end-to-end.
* `cloud_training_pipeline.ipynb`: Production-grade Jupyter notebook designed for Google Colab or Kaggle cloud clusters, using windowed disk-reads via `rasterio.windows.Window` to stream batches on-the-fly.
* `run_demo.py`: Main launcher script that compiles the pipeline, exports PNG frames, and starts a local web server on port 8000.
* `index.html`, `styles.css`, `app.js`: Stunning, interactive browser visualizer control center.
* `requirements.txt`: Python package dependencies.

---

## Installation & Local Execution

1. Clone this repository:
   ```bash
   git clone https://github.com/nikhildehariya/BAH2026_CloudRemoval.git
   cd BAH2026_CloudRemoval
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the processing pipeline and start the dashboard server:
   ```bash
   python run_demo.py
   ```
   Open **[http://localhost:8000](http://localhost:8000)** in your browser to inspect the visualizer panels.
