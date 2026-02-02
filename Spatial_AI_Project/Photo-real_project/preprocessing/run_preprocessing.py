import os
import argparse
from glob import glob
from tqdm import tqdm
from .segmentation import SemanticSegmentor
from .inpainting import Inpainter
import cv2

def main():
    parser = argparse.ArgumentParser(description="Run Preprocessing Pipeline (Segmentation + Inpainting)")
    parser.add_argument('input_dir', type=str, help="Path to Waymo extracted segment directory (containing images/)")
    parser.add_argument('--use_segformer', action='store_true', help="Use SegFormer for mask generation (instead of existing masks)")
    parser.add_argument('--inpainting', action='store_true', help="Perform Generative Inpainting")
    parser.add_argument('--device', type=str, default='cuda', help="Device to run models (cuda/cpu)")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    images_root = os.path.join(input_dir, 'images')
    masks_root = os.path.join(input_dir, 'masks')
    output_images_root = os.path.join(input_dir, 'images_inpainted')
    
    # 1. Initialize Models
    segmentor = None
    if args.use_segformer:
        segmentor = SemanticSegmentor(device=args.device)
        
    inpainter = None
    if args.inpainting:
        inpainter = Inpainter(use_generative=True, device=args.device)
        if not os.path.exists(output_images_root):
            os.makedirs(output_images_root)

    # 2. Iterate Cameras
    cameras = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
    
    for cam in cameras:
        cam_img_dir = os.path.join(images_root, cam)
        cam_mask_dir = os.path.join(masks_root, cam)
        cam_out_dir = os.path.join(output_images_root, cam)
        
        if not os.path.exists(cam_img_dir):
            continue
            
        if args.use_segformer and not os.path.exists(cam_mask_dir):
            os.makedirs(cam_mask_dir)
            
        if args.inpainting and not os.path.exists(cam_out_dir):
            os.makedirs(cam_out_dir)
            
        print(f"Processing Camera: {cam}")
        image_files = sorted(glob(os.path.join(cam_img_dir, '*.png')))
        
        for img_path in tqdm(image_files):
            basename = os.path.basename(img_path)
            mask_path = os.path.join(cam_mask_dir, basename)
            
            # --- Step 1: Segmentation ---
            if args.use_segformer:
                # Generate new mask using SegFormer
                mask = segmentor.process_image(img_path)
                cv2.imwrite(mask_path, mask)
            elif not os.path.exists(mask_path):
                # If not using SegFormer and mask doesn't exist (e.g. from 3D projection)
                print(f"Warning: Mask not found for {basename} and --use_segformer not set. Skipping.")
                continue
                
            # --- Step 2: Inpainting ---
            if args.inpainting:
                out_path = os.path.join(cam_out_dir, basename)
                inpainter.process_frame(img_path, mask_path, out_path)
                
    print("Preprocessing Complete.")

if __name__ == "__main__":
    main()
