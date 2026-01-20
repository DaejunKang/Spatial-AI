# Waymo Reconstruction for Omniverse NeuRec

This repository provides a complete pipeline to process Waymo Open Dataset (WOD) segments and reconstruct 3D scenes using 3D Gaussian Splatting (3DGS), tailored for integration with NVIDIA Omniverse NeuRec.

## Pipeline Overview

1.  **Download:** Download Waymo `.tfrecord` segments.
2.  **Extract & Mask:** Extract images, poses, and calibrations. Generate dynamic object masks using 3D Labels or SegFormer.
3.  **Inpaint:** Restore background in dynamic regions using Temporal Warping and Generative Inpainting (Stable Diffusion).
4.  **Reconstruct:** Train 3D Gaussian Splatting model.
5.  **Export:** Export result to USD (Universal Scene Description) for Omniverse.

## 1. Environment Setup

### Prerequisites
*   Python 3.7+ (Recommended 3.9/3.10)
*   CUDA-capable GPU
*   Waymo Open Dataset Account (for download)

### Installation

```bash
# Clone repository
git clone https://github.com/DaejunKang/Spatial-AI.git
cd Spatial-AI
git checkout Waymo_reconstruction

# Install basic dependencies
pip install torch torchvision numpy opencv-python tqdm scipy transformers diffusers accelerate

# Install Waymo Open Dataset (Choose compatible version)
# pip install waymo-open-dataset-tf-2-11-0==1.6.0 

# Optional: Install 3DGS dependencies (for reconstruction)
# pip install diff-gaussian-rasterization simple-knn
# pip install pxr (USD library)
```

## 2. Data Preparation

### 2.1 Download Waymo Data
Downloads `.tfrecord` files from Google Cloud Storage. Requires `gsutil` authentication.

```bash
# Download one segment for testing
python Spatial_AI_Project/Photo-real_project/download_waymo.py /path/to/waymo_raw --split training --limit 1
```

### 2.2 Extract Data

#### Option A: Aggressive Extractor (Recommended - No waymo_open_dataset required)
This extractor directly parses TFRecords without requiring the `waymo_open_dataset` package. It automatically detects images using magic headers and works with both SequenceExample and Example formats.

```bash
# Install minimal dependencies
pip install -r Spatial_AI_Project/Photo-real_project/requirements_extract.txt

# Extract data
python Spatial_AI_Project/Photo-real_project/extract_waymo_data.py /path/to/waymo_raw /path/to/waymo_extracted
```

**Features:**
- ✅ No `waymo_open_dataset` dependency required
- ✅ Automatic image detection via JPEG/PNG magic headers
- ✅ Supports both video sequences (SequenceExample) and snapshots (Example)
- ✅ Generates empty masks (for COLMAP compatibility)
- ✅ Outputs extraction statistics in JSON

**Output Structure:**
```
/path/to/waymo_extracted/segment_xxxx/
├── segment_xxxx/
│   ├── images/
│   │   ├── FRONT/
│   │   ├── FRONT_LEFT/
│   │   ├── FRONT_RIGHT/
│   │   ├── SIDE_LEFT/
│   │   └── SIDE_RIGHT/
│   ├── masks/
│   │   └── (same structure as images, empty black masks)
│   └── extraction_stats.json
```

#### Option B: Original Extractor (Requires waymo_open_dataset)
If you need poses and calibrations, use the original extractor (requires `waymo_open_dataset` package):

```bash
# Install waymo_open_dataset (Python 3.7-3.10)
pip install waymo-open-dataset-tf-2-11-0==1.6.0

# Use original extractor (if available)
# This version extracts poses and calibrations in addition to images
```

## 3. Advanced Preprocessing (Optional but Recommended)

Enhance masks using SegFormer and fill holes using Generative Inpainting.

```bash
cd Spatial_AI_Project/Photo-real_project/preprocessing

# Run full pipeline: SegFormer + Stable Diffusion Inpainting
python run_preprocessing.py /path/to/waymo_extracted/segment_xxxx --use_segformer --inpainting
```

*   **--use_segformer:** Uses HuggingFace SegFormer for pixel-perfect dynamic object segmentation.
*   **--inpainting:** Uses Stable Diffusion to fill masked regions (creates `images_inpainted/`).

## 4. 3D Reconstruction & Export

Train a 3D Gaussian Splatting model and export to USD.

```bash
python Spatial_AI_Project/Photo-real_project/reconstruction.py \
    /path/to/waymo_extracted/segment_xxxx \
    /path/to/output_recon \
    --use_inpainted
```

*   **Input:** Extracted segment directory.
*   **Output:** Directory to save `reconstruction.usd`.
*   **--use_inpainted:** Use inpainted images for training (cleaner background).

## 5. Omniverse Integration

The output `reconstruction.usd` contains the 3D Gaussian Point Cloud.
1.  Open **NVIDIA Omniverse USD Composer**.
2.  File -> Open -> Select `reconstruction.usd`.
3.  The point cloud will be visualized. (Note: Standard USD Points are rendered as spheres/disks. For full splat rendering, a custom Omniverse extension might be required).

## 6. COLMAP Conversion (Alternative)

If you need COLMAP format for other pipelines (e.g., standard 3DGS training):

```bash
python Spatial_AI_Project/Photo-real_project/waymo2colmap.py \
    /path/to/waymo_extracted/segment_xxxx \
    /path/to/colmap_output
```
