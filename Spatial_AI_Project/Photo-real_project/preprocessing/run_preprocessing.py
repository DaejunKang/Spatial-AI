"""
Preprocessing Pipeline Runner

전체 Preprocessing 단계를 순차 실행:
1. LiDAR Point Cloud Projection (선택적)
2. Dynamic Object Masking
3. Semantic Segmentation (선택적)

Usage:
    python run_preprocessing.py /path/to/data --lidar --dynamic_mask --semantic
"""

import os
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run Preprocessing Pipeline for Inpainting"
    )
    parser.add_argument(
        'data_root',
        type=str,
        help='Path to NRE format data directory'
    )
    
    # Stage Selection
    parser.add_argument(
        '--lidar',
        action='store_true',
        help='Run LiDAR point cloud projection to generate depth maps'
    )
    parser.add_argument(
        '--dynamic_mask',
        action='store_true',
        help='Run dynamic object masking (3D bounding box projection)'
    )
    parser.add_argument(
        '--semantic',
        action='store_true',
        help='Use semantic segmentation in dynamic masking'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all preprocessing stages'
    )
    
    # Parameters
    parser.add_argument(
        '--interpolation',
        type=str,
        default='nearest',
        choices=['none', 'nearest', 'linear', 'cubic'],
        help='Depth map interpolation method (default: nearest)'
    )
    parser.add_argument(
        '--dilation',
        type=int,
        default=5,
        help='Mask dilation kernel size (default: 5)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for semantic segmentation (cuda/cpu)'
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_root):
        print(f"Error: Data directory not found: {args.data_root}")
        sys.exit(1)
    
    print("="*70)
    print(">>> Preprocessing Pipeline Started")
    print("="*70)
    print(f"Data root: {args.data_root}")
    print(f"Stages to run:")
    
    # Determine which stages to run
    run_lidar = args.all or args.lidar
    run_dynamic_mask = args.all or args.dynamic_mask
    use_semantic = args.semantic
    
    print(f"  - LiDAR Projection: {'Yes' if run_lidar else 'No'}")
    print(f"  - Dynamic Masking: {'Yes' if run_dynamic_mask else 'No'}")
    print(f"  - Semantic Segmentation: {'Yes' if use_semantic else 'No'}")
    print("="*70)
    
    # Stage 1: LiDAR Point Cloud Projection
    if run_lidar:
        print("\n[Stage 1/2] LiDAR Point Cloud Projection")
        print("-"*70)
        
        try:
            from .lidar_projection import LiDARProjector
            
            projector = LiDARProjector(
                data_root=args.data_root,
                interpolation_method=args.interpolation
            )
            projector.run()
            
            print("✓ LiDAR projection completed successfully")
        
        except Exception as e:
            print(f"✗ LiDAR projection failed: {e}")
            print("  Continuing with remaining stages...")
    
    # Stage 2: Dynamic Object Masking
    if run_dynamic_mask:
        print("\n[Stage 2/2] Dynamic Object Masking")
        print("-"*70)
        
        try:
            from .dynamic_masking import DynamicObjectMasker
            
            masker = DynamicObjectMasker(
                data_root=args.data_root,
                use_semantic_seg=use_semantic,
                dilation_kernel=args.dilation
            )
            masker.run()
            
            print("✓ Dynamic masking completed successfully")
        
        except Exception as e:
            print(f"✗ Dynamic masking failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print(">>> Preprocessing Pipeline Complete!")
    print("="*70)
    print("\nGenerated outputs:")
    
    if run_lidar:
        print(f"  - Depth maps: {args.data_root}/depth_maps/")
        print(f"  - Point masks: {args.data_root}/point_masks/")
    
    if run_dynamic_mask:
        print(f"  - Dynamic object masks: {args.data_root}/masks/")
    
    print("\nNext steps:")
    print("  1. Run Inpainting Step 1 (Temporal Accumulation):")
    print(f"     python Inpainting/step1_temporal_accumulation.py --data_root {args.data_root}")
    print("  2. Run Inpainting Step 2 (Geometric Guide):")
    print(f"     python Inpainting/step2_geometric_guide.py --data_root {args.data_root}")
    print("  3. Run Inpainting Step 3 (Final Inpainting):")
    print(f"     python Inpainting/step3_final_inpainting.py --data_root {args.data_root}")
    print("="*70)

if __name__ == "__main__":
    main()
