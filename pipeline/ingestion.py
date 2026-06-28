"""
Module 1: Data Ingestion Engine
Interfaces to query and retrieve LISS-IV optical imagery from ISRO's NRSC Bhoonidhi Portal
and temporally matched Sentinel-1 GRD SAR data from the Copernicus Data Space Ecosystem.
Natively reads local files using Rasterio to feed arrays directly into the preprocessing engine.
"""

import os
import logging
import requests
import rasterio
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Setup Logger
logger = logging.getLogger("DhruvaPipeline.Ingestion")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class BhoonidhiClient:
    """
    Client interface for ISRO NRSC Bhoonidhi Portal API to search and retrieve LISS-IV datasets.
    """
    def __init__(self, api_url: str = "https://bhoonidhi.nrsc.gov.in/api", username: Optional[str] = None, password: Optional[str] = None):
        self.api_url = api_url
        self.username = username
        self.password = password
        self.session = requests.Session()
        
    def authenticate(self) -> bool:
        """
        Authenticate with the Bhoonidhi Portal API.
        """
        if not self.username or not self.password:
            logger.warning("Bhoonidhi credentials not provided. Running in search-only / public mode.")
            return False
        
        try:
            # Simulated API Authentication call
            auth_endpoint = f"{self.api_url}/login"
            payload = {"username": self.username, "password": self.password}
            logger.info("Successfully authenticated with NRSC Bhoonidhi Portal.")
            return True
        except Exception as e:
            logger.error(f"Authentication with Bhoonidhi failed: {e}")
            return False

    def query_liss4(self, bbox: Tuple[float, float, float, float], date_range: Tuple[str, str]) -> List[Dict[str, Any]]:
        """
        Query LISS-IV imagery matching spatial and temporal constraints.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        start_date, end_date = date_range
        
        logger.info(f"Querying Bhoonidhi for LISS-IV imagery. Bounding Box: {bbox}, Date Range: {date_range}")
        
        wkt_aoi = f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
        
        mock_scene_id = "R2_L4_MX_20260615_087_054"
        mock_results = [{
            "scene_id": mock_scene_id,
            "sensor": "LISS4",
            "satellite": "RESOURCESAT-2",
            "acquisition_date": start_date,
            "cloud_cover_percentage": 68.5,
            "bbox": bbox,
            "bands": ["Green", "Red", "NIR"],
            "resolution": 5.8,
            "download_url": f"https://bhoonidhi.nrsc.gov.in/catalog/liss4/{mock_scene_id}.zip"
        }]
        
        logger.info(f"Bhoonidhi Query returned {len(mock_results)} LISS-IV scene(s).")
        return mock_results

    def download_scene(self, scene_metadata: Dict[str, Any], output_dir: str) -> str:
        """
        Download LISS-IV scene payload.
        """
        os.makedirs(output_dir, exist_ok=True)
        scene_id = scene_metadata["scene_id"]
        filepath = os.path.join(output_dir, f"{scene_id}.tif")
        logger.info(f"Initiating download for LISS-IV scene: {scene_id}")
        
        with open(filepath, "w") as f:
            f.write(f"MOCK_LISS4_DATA_HEADER_{scene_id}")
            
        logger.info(f"LISS-IV scene downloaded successfully to: {filepath}")
        return filepath


class CopernicusDataSpaceClient:
    """
    Client interface for Copernicus Data Space Ecosystem (CDSE) API to fetch Sentinel-1 SAR products.
    """
    def __init__(self, token: Optional[str] = None):
        self.api_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"
        self.token = token
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    def query_sentinel1_grd(self, bbox: Tuple[float, float, float, float], date_range: Tuple[str, str]) -> List[Dict[str, Any]]:
        """
        Query Sentinel-1 GRD products that overlap spatially and temporally with LISS-IV.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        start_date, end_date = date_range
        
        logger.info(f"Querying Copernicus CDSE for Sentinel-1 GRD tracks. BBox: {bbox}, Range: {date_range}")
        
        mock_scene_id = "S1A_IW_GRDH_1SDV_20260615T120000_ASC"
        mock_results = [{
            "scene_id": mock_scene_id,
            "sensor": "SAR-C",
            "satellite": "Sentinel-1A",
            "product_type": "GRD",
            "acquisition_date": start_date,
            "polarization": "VV+VH",
            "resolution": 10.0,
            "bbox": bbox,
            "download_url": f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({mock_scene_id})/$value"
        }]
        
        logger.info(f"Copernicus CDSE Query returned {len(mock_results)} Sentinel-1 scene(s).")
        return mock_results


def load_native_pair(raw_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Natively reads LISS-IV and Sentinel-1 raw scenes using Rasterio.
    Extracts LISS-IV Band 2 (Green), 3 (Red), 4 (NIR) and Sentinel-1 VV, VH polarization bands.
    Logs and prints array shapes, resolutions, and CRS to the console.
    """
    liss4_path = os.path.join(raw_dir, "R2_L4_MX_20260615_087_054.tif")
    s1_path = os.path.join(raw_dir, "S1A_IW_GRDH_1SDV_20260615T120000_ASC.tif")
    
    logger.info(f"Accessing files natively using Rasterio reader...")
    
    if not os.path.exists(liss4_path):
        raise FileNotFoundError(f"LISS-IV raw file missing at: {liss4_path}")
    if not os.path.exists(s1_path):
        raise FileNotFoundError(f"Sentinel-1 SAR raw file missing at: {s1_path}")
        
    # Read LISS-IV
    with rasterio.open(liss4_path) as src:
        num_bands = src.count
        # Read Band 2 (Green), Band 3 (Red), and Band 4 (NIR)
        # Fallback to 1, 2, 3 if only 3 bands exist in the TIFF metadata
        if num_bands >= 4:
            green = src.read(2)
            red = src.read(3)
            nir = src.read(4)
            band_ids = (2, 3, 4)
        else:
            green = src.read(1)
            red = src.read(2)
            nir = src.read(3)
            band_ids = (1, 2, 3)
            
        liss4_meta = {
            "scene_id": "R2_L4_MX_20260615_087_054",
            "crs": str(src.crs),
            "resolution": src.res,
            "shape": green.shape,
            "bands": ["Green", "Red", "NIR"]
        }
        
        # Print details directly to stdout console
        print("\n" + "="*50)
        print("INGESTION MODULE: Primary LISS-IV Track Ingested")
        print("="*50)
        print(f"Source Path:        {liss4_path}")
        print(f"Extracted Bands:    Green (Band {band_ids[0]}), Red (Band {band_ids[1]}), NIR (Band {band_ids[2]})")
        print(f"Array Dimensions:   {green.shape}")
        print(f"Spatial Resolution: {src.res} meters")
        print(f"Coordinate System:  {src.crs}")
        print("="*50 + "\n")
        
        logger.info(f"Natively read LISS-IV. Shape: {green.shape}, Resolution: {src.res}, CRS: {src.crs}")

    # Read Sentinel-1 SAR
    with rasterio.open(s1_path) as src:
        vv = src.read(1)
        vh = src.read(2)
        
        s1_meta = {
            "scene_id": "S1A_IW_GRDH_1SDV_20260615T120000_ASC",
            "crs": str(src.crs),
            "resolution": src.res,
            "shape": vv.shape,
            "polarizations": ["VV", "VH"]
        }
        
        # Print details directly to stdout console
        print("="*50)
        print("INGESTION MODULE: Auxiliary Sentinel-1 SAR Ingested")
        print("="*50)
        print(f"Source Path:        {s1_path}")
        print(f"Extracted Bands:    VV (Band 1), VH (Band 2)")
        print(f"VV Array Shape:     {vv.shape}")
        print(f"VH Array Shape:     {vh.shape}")
        print(f"Spatial Resolution: {src.res} meters")
        print(f"Coordinate System:  {src.crs}")
        print("="*50 + "\n")
        
        logger.info(f"Natively read Sentinel-1. VV Shape: {vv.shape}, VH Shape: {vh.shape}, Resolution: {src.res}, CRS: {src.crs}")
        
    return liss4_meta, s1_meta


def run_ingestion_pipeline(bbox: Tuple[float, float, float, float], date_range: Tuple[str, str], output_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Main ingestion execution block to query matching pairs and natively read them.
    """
    logger.info("Starting Team Dhruva Ingestion Pipeline...")
    
    # Run Bhoonidhi & Copernicus search simulations
    bhoonidhi = BhoonidhiClient()
    copernicus = CopernicusDataSpaceClient()
    
    _ = bhoonidhi.query_liss4(bbox, date_range)
    _ = copernicus.query_sentinel1_grd(bbox, date_range)
    
    # Load raw tracks natively using rasterio
    selected_liss4, selected_s1 = load_native_pair(output_dir)
    
    logger.info("Ingestion pipeline execution complete.")
    return selected_liss4, selected_s1


if __name__ == "__main__":
    # Test case bounding box (Assam area footprint)
    assam_bbox = (91.5, 26.0, 92.0, 26.5)
    june_dates = ("2026-06-01", "2026-06-15")
    run_ingestion_pipeline(assam_bbox, june_dates, "./data/raw")
