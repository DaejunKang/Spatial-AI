"""
Approach 2: 3D Gaussian with Uncertainty and Time (3DGUT)
          Rolling Shutter Compensated Reconstruction

Reference:
    "3DGS with Rolling Shutter Compensation"
    (Custom implementation for autonomous driving)

Input (3DGS + α):
    - Images: Inpainting된 배경 이미지
    - Camera Extrinsics/Intrinsics
    - Ego Velocity: 선속도 및 각속도 [vx, vy, vz, wx, wy, wz]
    - Rolling Shutter: duration, trigger_time

Output:
    - Trained 3D Gaussians with temporal parameters (.ply)
    - Rendered Novel Views (Rolling Shutter 보정됨)
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


class RollingShutterCompensator:
    """
    Rolling Shutter 보정 유틸리티
    
    이미지의 각 행(row)마다 다른 시간에 캡처되므로,
    카메라 포즈를 시간에 따라 보정해야 함
    """
    
    @staticmethod
    def compute_pose_at_time(
        T_base: torch.Tensor,
        velocity: torch.Tensor,
        delta_t: float
    ) -> torch.Tensor:
        """
        시간 오프셋을 고려한 카메라 포즈 계산
        
        Args:
            T_base: [4, 4] 기준 카메라 포즈
            velocity: [6] 선속도 및 각속도 [vx, vy, vz, wx, wy, wz]
            delta_t: 시간 오프셋 (초)
        
        Returns:
            T_adjusted: [4, 4] 보정된 카메라 포즈
        """
        # 선속도 및 각속도 분리
        v = velocity[:3]  # [vx, vy, vz]
        w = velocity[3:]  # [wx, wy, wz]
        
        # Translation offset
        translation_offset = v * delta_t
        
        # Rotation offset (Rodrigues' formula)
        angle = torch.norm(w) * delta_t
        if angle > 1e-6:
            axis = w / torch.norm(w)
            K = torch.tensor([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ], device=w.device)
            
            R_offset = torch.eye(3, device=w.device) + \
                       torch.sin(angle) * K + \
                       (1 - torch.cos(angle)) * (K @ K)
        else:
            R_offset = torch.eye(3, device=w.device)
        
        # 4x4 변환 행렬 구성
        T_offset = torch.eye(4, device=T_base.device)
        T_offset[:3, :3] = R_offset
        T_offset[:3, 3] = translation_offset
        
        # 합성
        T_adjusted = T_offset @ T_base
        
        return T_adjusted
    
    @staticmethod
    def get_pixel_time_offset(
        pixel_y: int,
        image_height: int,
        rs_duration: float,
        rs_trigger_time: float
    ) -> float:
        """
        픽셀의 캡처 시간 오프셋 계산
        
        Args:
            pixel_y: 픽셀의 행(row) 좌표
            image_height: 이미지 전체 높이
            rs_duration: Rolling Shutter 지속 시간 (전체 이미지 readout time)
            rs_trigger_time: 기준 시각 대비 촬영 시작 오프셋
        
        Returns:
            time_offset: 시간 오프셋 (초)
        """
        # 각 행의 상대적 시간
        row_ratio = pixel_y / image_height
        time_offset = rs_trigger_time + row_ratio * rs_duration
        
        return time_offset


class GaussianSplatting3DGUT:
    """
    3DGUT: Rolling Shutter를 고려한 3D Gaussian Splatting
    
    각 픽셀마다 다른 카메라 포즈를 사용하여 렌더링
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
            meta_file: JSON 메타데이터 파일 (3DGUT 형식)
            output_dir: 출력 디렉토리
            initial_ply: 초기 포인트 클라우드 (선택)
            device: 'cuda' or 'cpu'
        """
        self.data_root = Path(data_root)
        self.meta_file = Path(meta_file)
        self.output_dir = Path(output_dir)
        self.device = device
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터셋 로드 (3DGUT 파라미터 포함)
        print("="*70)
        print(">>> Loading Dataset for 3DGUT")
        print("="*70)
        
        self.dataset = ReconstructionDataset(
            meta_file=str(self.meta_file),
            data_root=str(self.data_root),
            split='train',
            load_3dgut_params=True,  # ✅ 3DGUT 파라미터 로드
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
        
        # Gaussian parameters
        self.gaussians = None
        
        # Rolling Shutter Compensator
        self.rs_compensator = RollingShutterCompensator()
        
        print("="*70)
    
    def initialize_gaussians(self):
        """
        Gaussian 파라미터 초기화
        
        3DGS와 동일하지만, 추가로 temporal uncertainty 파라미터 포함 가능
        """
        print("\n[1/4] Initializing Gaussians (3DGUT)...")
        
        if self.init_points is not None:
            num_points = len(self.init_points)
            
            self.gaussians = {
                'xyz': torch.tensor(self.init_points, dtype=torch.float32, device=self.device),
                'rgb': torch.tensor(self.init_colors, dtype=torch.float32, device=self.device),
                'opacity': torch.ones(num_points, 1, dtype=torch.float32, device=self.device),
                'scale': torch.ones(num_points, 3, dtype=torch.float32, device=self.device) * 0.01,
                'rotation': torch.tensor([[1, 0, 0, 0]] * num_points, dtype=torch.float32, device=self.device),
                
                # 3DGUT 추가 파라미터 (선택적)
                'temporal_uncertainty': torch.ones(num_points, 1, dtype=torch.float32, device=self.device) * 0.01
            }
        else:
            num_points = 10000
            
            self.gaussians = {
                'xyz': torch.randn(num_points, 3, dtype=torch.float32, device=self.device),
                'rgb': torch.rand(num_points, 3, dtype=torch.float32, device=self.device),
                'opacity': torch.ones(num_points, 1, dtype=torch.float32, device=self.device) * 0.5,
                'scale': torch.ones(num_points, 3, dtype=torch.float32, device=self.device) * 0.01,
                'rotation': torch.tensor([[1, 0, 0, 0]] * num_points, dtype=torch.float32, device=self.device),
                'temporal_uncertainty': torch.ones(num_points, 1, dtype=torch.float32, device=self.device) * 0.01
            }
        
        # Requires grad
        for key in self.gaussians:
            self.gaussians[key].requires_grad = True
        
        print(f"  Initialized {len(self.gaussians['xyz'])} Gaussians with temporal parameters")
    
    def render_with_rolling_shutter(
        self,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
        velocity: torch.Tensor,
        rs_duration: float,
        rs_trigger_time: float,
        width: int,
        height: int
    ) -> torch.Tensor:
        """
        Rolling Shutter를 고려한 렌더링
        
        Args:
            extrinsic: [4, 4] 기준 카메라 포즈
            intrinsic: [3, 3] 카메라 내부 파라미터
            velocity: [6] 속도 [vx, vy, vz, wx, wy, wz]
            rs_duration: Rolling Shutter 지속 시간
            rs_trigger_time: 촬영 시작 오프셋
            width: 이미지 너비
            height: 이미지 높이
        
        Returns:
            rendered_image: [3, H, W] RGB 이미지
        """
        # TODO: 실제 Rolling Shutter 고려 렌더링 구현
        # 
        # 방법 1: Row-wise 렌더링
        #   각 행마다 다른 카메라 포즈로 렌더링 후 합성
        #
        # 방법 2: Continuous 근사
        #   Gaussian의 위치를 시간에 따라 보간하여 렌더링
        
        # Placeholder: 기본 렌더링 (Rolling Shutter 무시)
        rendered = torch.zeros(3, height, width, device=self.device)
        
        # Example: Row-wise pose adjustment (simplified)
        for row in range(0, height, 10):  # 샘플링 (모든 행 처리 시 너무 느림)
            # 해당 행의 시간 오프셋 계산
            time_offset = self.rs_compensator.get_pixel_time_offset(
                pixel_y=row,
                image_height=height,
                rs_duration=rs_duration,
                rs_trigger_time=rs_trigger_time
            )
            
            # 보정된 카메라 포즈
            extrinsic_adjusted = self.rs_compensator.compute_pose_at_time(
                T_base=extrinsic,
                velocity=velocity,
                delta_t=time_offset
            )
            
            # 해당 행 렌더링 (placeholder)
            # rendered[:, row:row+10, :] = render_row(extrinsic_adjusted, ...)
        
        return rendered
    
    def train(self, num_iterations: int = 30000, log_interval: int = 100):
        """
        3DGUT 학습
        
        Rolling Shutter 보정을 포함한 학습
        """
        print("\n[2/4] Starting Training (3DGUT with Rolling Shutter)...")
        
        # Optimizer 설정
        optimizer = torch.optim.Adam([
            {'params': [self.gaussians['xyz']], 'lr': 0.00016},
            {'params': [self.gaussians['rgb']], 'lr': 0.0025},
            {'params': [self.gaussians['opacity']], 'lr': 0.05},
            {'params': [self.gaussians['scale']], 'lr': 0.005},
            {'params': [self.gaussians['rotation']], 'lr': 0.001},
            {'params': [self.gaussians['temporal_uncertainty']], 'lr': 0.001}
        ])
        
        # 학습 루프
        pbar = tqdm(range(num_iterations), desc="Training (3DGUT)")
        
        for iteration in pbar:
            # 랜덤 샘플 선택
            idx = np.random.randint(0, len(self.dataset))
            sample = self.dataset[idx]
            
            # GPU로 이동
            gt_image = sample['image'].to(self.device)
            extrinsic = sample['extrinsic'].to(self.device)
            intrinsic = sample['intrinsic'].to(self.device)
            velocity = sample['velocity'].to(self.device)  # [6]
            rs_duration = sample['rolling_shutter_duration']
            rs_trigger = sample['rolling_shutter_trigger_time']
            width = sample['width']
            height = sample['height']
            
            # Rolling Shutter 고려 렌더링
            rendered_image = self.render_with_rolling_shutter(
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                velocity=velocity,
                rs_duration=rs_duration,
                rs_trigger_time=rs_trigger,
                width=width,
                height=height
            )
            
            # Loss 계산
            l1_loss = torch.abs(rendered_image - gt_image).mean()
            
            # TODO: SSIM + Temporal consistency loss
            loss = l1_loss
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 로그
            if iteration % log_interval == 0:
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        print(f"\n  Training completed: {num_iterations} iterations")
    
    def save_gaussians(self, filename: str = 'gaussians_3dgut.ply'):
        """
        학습된 Gaussian 저장 (PLY 포맷)
        Temporal uncertainty 파라미터 포함
        """
        print("\n[3/4] Saving Gaussians (3DGUT)...")
        
        output_path = self.output_dir / filename
        
        # TODO: PLY 파일 저장 구현
        print(f"  Saved to: {output_path}")
        
        with open(output_path, 'w') as f:
            f.write("PLY format with temporal parameters (placeholder)\n")
    
    def render_novel_views(self, num_views: int = 10):
        """
        Novel View Rendering (Rolling Shutter 보정)
        """
        print("\n[4/4] Rendering Novel Views (Rolling Shutter Compensated)...")
        
        novel_dir = self.output_dir / 'novel_views'
        novel_dir.mkdir(exist_ok=True)
        
        for i in range(min(num_views, len(self.dataset))):
            sample = self.dataset[i]
            
            extrinsic = sample['extrinsic'].to(self.device)
            intrinsic = sample['intrinsic'].to(self.device)
            velocity = sample['velocity'].to(self.device)
            rs_duration = sample['rolling_shutter_duration']
            rs_trigger = sample['rolling_shutter_trigger_time']
            width = sample['width']
            height = sample['height']
            
            # 렌더링
            with torch.no_grad():
                rendered = self.render_with_rolling_shutter(
                    extrinsic, intrinsic, velocity,
                    rs_duration, rs_trigger,
                    width, height
                )
            
            # 저장 (placeholder)
        
        print(f"  Rendered {num_views} novel views to: {novel_dir}")
    
    def run(self, num_iterations: int = 30000):
        """전체 파이프라인 실행"""
        print("\n" + "="*70)
        print(">>> 3DGUT Training Pipeline (Rolling Shutter Compensated)")
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
        description="3DGUT Training (Approach 2): Rolling Shutter Compensated"
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
        default='outputs/3dgut',
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
        print(f"  python reconstruction/prepare_metadata.py {args.data_root} --mode 3dgut")
        return
    
    # 초기 PLY 경로
    initial_ply = None
    if args.initial_ply:
        initial_ply = Path(args.initial_ply)
        if not initial_ply.is_absolute():
            initial_ply = data_root / initial_ply
    
    # Trainer 생성 및 실행
    trainer = GaussianSplatting3DGUT(
        data_root=str(data_root),
        meta_file=str(meta_file),
        output_dir=str(output_dir),
        initial_ply=str(initial_ply) if initial_ply else None,
        device=args.device
    )
    
    trainer.run(num_iterations=args.iterations)


if __name__ == '__main__':
    main()
