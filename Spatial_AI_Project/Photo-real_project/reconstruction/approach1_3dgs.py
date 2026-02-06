"""
Approach 1: 3D Gaussian Splatting (3DGS) for Static Scene Reconstruction

Reference:
    "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

Input:
    - Images: Inpainting된 배경 이미지
    - Camera Extrinsics: World-to-Camera 변환 행렬
    - Camera Intrinsics: 카메라 내부 파라미터
    - (선택) Initial Point Cloud: 초기화용 PLY

Output:
    - Trained 3D Gaussians (.ply)
    - Rendered Novel Views
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

# Data loader
from data_loader import ReconstructionDataset, DataLoaderWrapper, load_initial_point_cloud


class GaussianSplattingTrainer:
    """
    3D Gaussian Splatting 학습 클래스
    
    정적 장면(Static Scene)을 가정하여 학습
    """
    
    def __init__(
        self,
        data_root: str,
        meta_file: str,
        output_dir: str,
        initial_ply: str = None,
        device: str = 'cuda'
    ):
        """
        Args:
            data_root: 데이터 루트 디렉토리
            meta_file: JSON 메타데이터 파일
            output_dir: 출력 디렉토리
            initial_ply: 초기 포인트 클라우드 (선택)
            device: 'cuda' or 'cpu'
        """
        self.data_root = Path(data_root)
        self.meta_file = Path(meta_file)
        self.output_dir = Path(output_dir)
        self.device = device
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터셋 로드
        print("="*70)
        print(">>> Loading Dataset for 3DGS")
        print("="*70)
        
        self.dataset = ReconstructionDataset(
            meta_file=str(self.meta_file),
            data_root=str(self.data_root),
            split='train',
            load_3dgut_params=False,  # 3DGS는 속도 정보 불필요
            image_scale=1.0
        )
        
        # 초기 포인트 클라우드
        self.initial_ply = initial_ply
        if initial_ply and Path(initial_ply).exists():
            print(f"\nLoading initial point cloud: {initial_ply}")
            self.init_points, self.init_colors = load_initial_point_cloud(initial_ply)
            print(f"  Points: {len(self.init_points)}")
        else:
            print("\nNo initial point cloud provided. Will use random initialization.")
            self.init_points = None
            self.init_colors = None
        
        # Gaussian parameters (placeholder - 실제 3DGS 구현 필요)
        self.gaussians = None
        
        print("="*70)
    
    def initialize_gaussians(self):
        """
        Gaussian 파라미터 초기화
        
        각 Gaussian은 다음 파라미터를 가짐:
        - Position (xyz): 3D 위치
        - Color (RGB): 색상
        - Opacity (α): 불투명도
        - Scale (s): 크기
        - Rotation (q): 회전 (quaternion)
        """
        print("\n[1/4] Initializing Gaussians...")
        
        if self.init_points is not None:
            # 초기 포인트 클라우드 사용
            num_points = len(self.init_points)
            
            self.gaussians = {
                'xyz': torch.tensor(self.init_points, dtype=torch.float32, device=self.device),
                'rgb': torch.tensor(self.init_colors, dtype=torch.float32, device=self.device),
                'opacity': torch.ones(num_points, 1, dtype=torch.float32, device=self.device),
                'scale': torch.ones(num_points, 3, dtype=torch.float32, device=self.device) * 0.01,
                'rotation': torch.tensor([[1, 0, 0, 0]] * num_points, dtype=torch.float32, device=self.device)
            }
        else:
            # 랜덤 초기화 (간단한 예시)
            num_points = 10000
            
            self.gaussians = {
                'xyz': torch.randn(num_points, 3, dtype=torch.float32, device=self.device),
                'rgb': torch.rand(num_points, 3, dtype=torch.float32, device=self.device),
                'opacity': torch.ones(num_points, 1, dtype=torch.float32, device=self.device) * 0.5,
                'scale': torch.ones(num_points, 3, dtype=torch.float32, device=self.device) * 0.01,
                'rotation': torch.tensor([[1, 0, 0, 0]] * num_points, dtype=torch.float32, device=self.device)
            }
        
        # Requires grad
        for key in self.gaussians:
            self.gaussians[key].requires_grad = True
        
        print(f"  Initialized {len(self.gaussians['xyz'])} Gaussians")
    
    def render(self, extrinsic, intrinsic, width, height):
        """
        Gaussian Splatting 렌더링
        
        Args:
            extrinsic: [4, 4] Camera pose
            intrinsic: [3, 3] Camera intrinsic
            width: 이미지 너비
            height: 이미지 높이
        
        Returns:
            rendered_image: [3, H, W] RGB 이미지
        """
        # TODO: 실제 Gaussian Splatting 렌더링 구현
        # 여기서는 placeholder만 제공
        # 실제 구현은 diff-gaussian-rasterization 라이브러리 사용
        
        # Placeholder: 검은 이미지 반환
        rendered = torch.zeros(3, height, width, device=self.device)
        
        return rendered
    
    def train(self, num_iterations: int = 30000, log_interval: int = 100):
        """
        3DGS 학습
        
        Args:
            num_iterations: 학습 반복 횟수
            log_interval: 로그 출력 간격
        """
        print("\n[2/4] Starting Training...")
        
        # Optimizer 설정
        optimizer = torch.optim.Adam([
            {'params': [self.gaussians['xyz']], 'lr': 0.00016},
            {'params': [self.gaussians['rgb']], 'lr': 0.0025},
            {'params': [self.gaussians['opacity']], 'lr': 0.05},
            {'params': [self.gaussians['scale']], 'lr': 0.005},
            {'params': [self.gaussians['rotation']], 'lr': 0.001}
        ])
        
        # 학습 루프
        pbar = tqdm(range(num_iterations), desc="Training")
        
        for iteration in pbar:
            # 랜덤 샘플 선택
            idx = np.random.randint(0, len(self.dataset))
            sample = self.dataset[idx]
            
            # GPU로 이동
            gt_image = sample['image'].to(self.device)
            extrinsic = sample['extrinsic'].to(self.device)
            intrinsic = sample['intrinsic'].to(self.device)
            width = sample['width']
            height = sample['height']
            
            # 렌더링
            rendered_image = self.render(extrinsic, intrinsic, width, height)
            
            # Loss 계산 (L1 + SSIM)
            l1_loss = torch.abs(rendered_image - gt_image).mean()
            
            # TODO: SSIM loss 추가
            # from pytorch_msssim import ssim
            # ssim_loss = 1 - ssim(rendered_image, gt_image)
            
            loss = l1_loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 로그
            if iteration % log_interval == 0:
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        print(f"\n  Training completed: {num_iterations} iterations")
    
    def save_gaussians(self, filename: str = 'gaussians.ply'):
        """
        학습된 Gaussian 저장 (PLY 포맷)
        """
        print("\n[3/4] Saving Gaussians...")
        
        output_path = self.output_dir / filename
        
        # TODO: PLY 파일 저장 구현
        # 실제 구현에서는 Gaussian 파라미터를 PLY 포맷으로 저장
        
        # Placeholder
        print(f"  Saved to: {output_path}")
        
        with open(output_path, 'w') as f:
            f.write("PLY format (placeholder)\n")
    
    def render_novel_views(self, num_views: int = 10):
        """
        Novel View Rendering (검증용)
        """
        print("\n[4/4] Rendering Novel Views...")
        
        novel_dir = self.output_dir / 'novel_views'
        novel_dir.mkdir(exist_ok=True)
        
        # Validation 샘플 사용
        for i in range(min(num_views, len(self.dataset))):
            sample = self.dataset[i]
            
            extrinsic = sample['extrinsic'].to(self.device)
            intrinsic = sample['intrinsic'].to(self.device)
            width = sample['width']
            height = sample['height']
            
            # 렌더링
            with torch.no_grad():
                rendered = self.render(extrinsic, intrinsic, width, height)
            
            # 저장 (placeholder)
            # TODO: 실제 이미지 저장
        
        print(f"  Rendered {num_views} novel views to: {novel_dir}")
    
    def run(self, num_iterations: int = 30000):
        """전체 파이프라인 실행"""
        print("\n" + "="*70)
        print(">>> 3D Gaussian Splatting Training Pipeline")
        print("="*70)
        
        # 1. 초기화
        self.initialize_gaussians()
        
        # 2. 학습
        self.train(num_iterations=num_iterations)
        
        # 3. 저장
        self.save_gaussians()
        
        # 4. Novel View Rendering
        self.render_novel_views()
        
        print("\n" + "="*70)
        print(">>> Training Complete!")
        print(f"  Output: {self.output_dir}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="3D Gaussian Splatting Training (Approach 1)"
    )
    parser.add_argument(
        'data_root',
        type=str,
        help='Path to NRE format data root'
    )
    parser.add_argument(
        '--meta_file',
        type=str,
        default='train_meta/train_pairs.json',
        help='JSON metadata file (relative to data_root)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/3dgs',
        help='Output directory (relative to data_root)'
    )
    parser.add_argument(
        '--initial_ply',
        type=str,
        default=None,
        help='Initial point cloud PLY file'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=30000,
        help='Number of training iterations (default: 30000)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    # 경로 설정
    data_root = Path(args.data_root)
    meta_file = data_root / args.meta_file
    output_dir = data_root / args.output_dir
    
    # 메타데이터 생성 필요 여부 확인
    if not meta_file.exists():
        print(f"Error: Metadata file not found: {meta_file}")
        print("\nPlease run prepare_metadata.py first:")
        print(f"  python reconstruction/prepare_metadata.py {args.data_root} --mode 3dgs")
        return
    
    # 초기 PLY 경로 (절대 경로 또는 상대 경로)
    initial_ply = None
    if args.initial_ply:
        initial_ply = Path(args.initial_ply)
        if not initial_ply.is_absolute():
            initial_ply = data_root / initial_ply
    
    # Trainer 생성 및 실행
    trainer = GaussianSplattingTrainer(
        data_root=str(data_root),
        meta_file=str(meta_file),
        output_dir=str(output_dir),
        initial_ply=str(initial_ply) if initial_ply else None,
        device=args.device
    )
    
    trainer.run(num_iterations=args.iterations)


if __name__ == '__main__':
    main()
