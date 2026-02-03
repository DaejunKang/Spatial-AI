import os
import argparse
from glob import glob
from tqdm import tqdm
from .segmentation import SemanticSegmentor
import cv2

# Note: Inpainting functionality has been moved to ../Inpainting/step1_temporal_accumulation.py
# For temporal accumulation-based inpainting, run that script separately after preprocessing

def main():
    parser = argparse.ArgumentParser(description="Run Preprocessing Pipeline (Segmentation Only)")
    parser.add_argument('input_dir', type=str, help="Path to Waymo extracted segment directory (containing images/)")
    parser.add_argument('--use_segformer', action='store_true', help="Use SegFormer for mask generation (instead of existing masks)")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run models (cuda/cpu)")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    images_root = os.path.join(input_dir, 'images')
    masks_root = os.path.join(input_dir, 'masks')
    
    # 1. Initialize Models
    segmentor = None
    if args.use_segformer:
        segmentor = SemanticSegmentor(device=args.device)

    # 2. Iterate Cameras
    cameras = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
    
    for cam in cameras:
        cam_img_dir = os.path.join(images_root, cam)
        cam_mask_dir = os.path.join(masks_root, cam)
        
        if not os.path.exists(cam_img_dir):
            continue
            
        if args.use_segformer and not os.path.exists(cam_mask_dir):
            os.makedirs(cam_mask_dir)
            
        print(f"Processing Camera: {cam}")
        image_files = sorted(glob(os.path.join(cam_img_dir, '*.png')))
        
        for img_path in tqdm(image_files):
            basename = os.path.basename(img_path)
            mask_path = os.path.join(cam_mask_dir, basename)
            
            # --- Segmentation ---
            if args.use_segformer:
                # Generate new mask using SegFormer
                mask = segmentor.process_image(img_path)
                cv2.imwrite(mask_path, mask)
            elif not os.path.exists(mask_path):
                # If not using SegFormer and mask doesn't exist (e.g. from 3D projection)
                print(f"Warning: Mask not found for {basename} and --use_segformer not set. Skipping.")
                continue
                
    print("Preprocessing Complete.")
    print("\nFor inpainting, run the temporal accumulation script:")
    print(f"  python ../Inpainting/step1_temporal_accumulation.py {input_dir}")

if __name__ == "__main__":
    main()
