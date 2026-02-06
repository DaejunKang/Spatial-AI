"""
Common Data Loader for 3D Reconstruction

공통 데이터 로더 - 3DGS와 3DGUT 모두 사용 가능
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from typing import Dict, List, Optional, Tuple


class ReconstructionDataset:
    """
    3D Reconstruction을 위한 데이터셋 클래스
    
    공통 입력:
    - Images: RGB 이미지
    - Camera Extrinsics: World-to-Camera 변환 행렬
    - Camera Intrinsics: 카메라 내부 파라미터
    
    3DGUT 추가 입력 (옵션):
    - Velocity: 선속도 및 각속도
    - Rolling Shutter: duration, trigger_time
    """
    
    def __init__(
        self, 
        meta_file: str,
        data_root: str = None,
        split: str = 'train',
        load_3dgut_params: bool = False,
        image_scale: float = 1.0
    ):
        """
        Args:
            meta_file: JSON 메타데이터 파일 경로
            data_root: 데이터 루트 디렉토리 (meta_file에서 상대 경로 사용 시)
            split: 'train' or 'val' or 'test'
            load_3dgut_params: 3DGUT 파라미터 로드 여부
            image_scale: 이미지 스케일 조정 (1.0 = 원본)
        """
        self.meta_file = Path(meta_file)
        self.data_root = Path(data_root) if data_root else self.meta_file.parent.parent
        self.split = split
        self.load_3dgut_params = load_3dgut_params
        self.image_scale = image_scale
        
        # 메타데이터 로드
        with open(self.meta_file, 'r') as f:
            self.metadata = json.load(f)
        
        # 데이터 검증
        self._validate_metadata()
        
        print(f"[ReconstructionDataset] Loaded {len(self.metadata)} samples from {self.meta_file}")
        print(f"  Data root: {self.data_root}")
        print(f"  Split: {self.split}")
        print(f"  Load 3DGUT params: {self.load_3dgut_params}")
    
    def _validate_metadata(self):
        """메타데이터 검증"""
        required_keys = ['file_path', 'transform_matrix', 'intrinsics']
        
        for i, item in enumerate(self.metadata):
            # 필수 키 확인
            for key in required_keys:
                if key not in item:
                    raise ValueError(f"Item {i}: Missing required key '{key}'")
            
            # 3DGUT 파라미터 확인
            if self.load_3dgut_params:
                if 'velocity' not in item or 'rolling_shutter' not in item:
                    raise ValueError(
                        f"Item {i}: 3DGUT mode requires 'velocity' and 'rolling_shutter' fields"
                    )
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        단일 데이터 샘플 로드
        
        Returns:
            dict: {
                'image': Tensor [3, H, W],
                'extrinsic': Tensor [4, 4],
                'intrinsic': Tensor [3, 3],
                'image_path': str,
                'width': int,
                'height': int,
                
                # 3DGUT 전용 (load_3dgut_params=True 시)
                'velocity': Tensor [6],  # [vx, vy, vz, wx, wy, wz]
                'rolling_shutter_duration': float,
                'rolling_shutter_trigger_time': float
            }
        """
        item = self.metadata[idx]
        
        # 1. 이미지 로드
        img_path = self.data_root / item['file_path']
        image = Image.open(img_path).convert('RGB')
        
        # 스케일 조정
        if self.image_scale != 1.0:
            new_w = int(image.width * self.image_scale)
            new_h = int(image.height * self.image_scale)
            image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # PIL → Tensor [3, H, W]
        image_tensor = torch.from_numpy(
            np.array(image).transpose(2, 0, 1) / 255.0
        ).float()
        
        # 2. Extrinsic (Transform Matrix)
        extrinsic = torch.tensor(
            item['transform_matrix'], 
            dtype=torch.float32
        ).reshape(4, 4)
        
        # 3. Intrinsic
        intrinsic_list = item['intrinsics']
        
        # [fx, fy, cx, cy, ...] → 3x3 matrix
        if len(intrinsic_list) >= 4:
            fx, fy, cx, cy = intrinsic_list[:4]
            intrinsic = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=torch.float32)
        else:
            raise ValueError(f"Invalid intrinsics format: {intrinsic_list}")
        
        # 스케일 조정 반영
        if self.image_scale != 1.0:
            intrinsic[0, 0] *= self.image_scale  # fx
            intrinsic[1, 1] *= self.image_scale  # fy
            intrinsic[0, 2] *= self.image_scale  # cx
            intrinsic[1, 2] *= self.image_scale  # cy
        
        # 4. 결과 구성
        result = {
            'image': image_tensor,
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
            'image_path': str(img_path),
            'width': image.width,
            'height': image.height,
            'idx': idx
        }
        
        # 5. 3DGUT 파라미터 (선택적)
        if self.load_3dgut_params:
            # Velocity [vx, vy, vz, wx, wy, wz]
            v = item['velocity']['v']  # [vx, vy, vz]
            w = item['velocity']['w']  # [wx, wy, wz]
            velocity = torch.tensor(v + w, dtype=torch.float32)  # [6]
            
            # Rolling Shutter
            rs_duration = item['rolling_shutter']['duration']
            rs_trigger = item['rolling_shutter']['trigger_time']
            
            result.update({
                'velocity': velocity,
                'rolling_shutter_duration': rs_duration,
                'rolling_shutter_trigger_time': rs_trigger
            })
        
        return result
    
    def get_all_extrinsics(self) -> torch.Tensor:
        """모든 카메라 Extrinsic 반환 (초기화용)"""
        extrinsics = []
        for item in self.metadata:
            ext = torch.tensor(item['transform_matrix'], dtype=torch.float32).reshape(4, 4)
            extrinsics.append(ext)
        return torch.stack(extrinsics)
    
    def get_all_intrinsics(self) -> torch.Tensor:
        """모든 카메라 Intrinsic 반환"""
        intrinsics = []
        for item in self.metadata:
            fx, fy, cx, cy = item['intrinsics'][:4]
            K = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=torch.float32)
            intrinsics.append(K)
        return torch.stack(intrinsics)


class DataLoaderWrapper:
    """
    PyTorch DataLoader 래퍼
    Batch collate 등 처리
    """
    
    def __init__(
        self, 
        dataset: ReconstructionDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0
    ):
        """
        Args:
            dataset: ReconstructionDataset
            batch_size: 배치 크기
            shuffle: 셔플 여부
            num_workers: 멀티프로세싱 워커 수
        """
        from torch.utils.data import DataLoader
        
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Batch collate function"""
        # Tensor stacking
        collated = {
            'image': torch.stack([item['image'] for item in batch]),
            'extrinsic': torch.stack([item['extrinsic'] for item in batch]),
            'intrinsic': torch.stack([item['intrinsic'] for item in batch]),
        }
        
        # Non-tensor fields (리스트로 유지)
        collated['image_path'] = [item['image_path'] for item in batch]
        collated['width'] = [item['width'] for item in batch]
        collated['height'] = [item['height'] for item in batch]
        collated['idx'] = [item['idx'] for item in batch]
        
        # 3DGUT params
        if 'velocity' in batch[0]:
            collated['velocity'] = torch.stack([item['velocity'] for item in batch])
            collated['rolling_shutter_duration'] = torch.tensor(
                [item['rolling_shutter_duration'] for item in batch]
            )
            collated['rolling_shutter_trigger_time'] = torch.tensor(
                [item['rolling_shutter_trigger_time'] for item in batch]
            )
        
        return collated
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def load_initial_point_cloud(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    초기 포인트 클라우드 로드 (3DGS 초기화용)
    
    Args:
        ply_path: PLY 파일 경로
    
    Returns:
        positions: (N, 3) 포인트 위치
        colors: (N, 3) RGB 색상 [0-1]
    """
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("Please install plyfile: pip install plyfile")
    
    plydata = PlyData.read(ply_path)
    vertices = plydata['vertex']
    
    positions = np.stack([
        vertices['x'],
        vertices['y'],
        vertices['z']
    ], axis=1)
    
    # RGB (0-255 → 0-1)
    if 'red' in vertices:
        colors = np.stack([
            vertices['red'],
            vertices['green'],
            vertices['blue']
        ], axis=1) / 255.0
    else:
        colors = np.ones_like(positions) * 0.5  # 회색
    
    return positions, colors


if __name__ == '__main__':
    # 테스트
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('meta_file', type=str, help='JSON metadata file')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--3dgut', action='store_true', help='Load 3DGUT parameters')
    args = parser.parse_args()
    
    # 데이터셋 로드
    dataset = ReconstructionDataset(
        meta_file=args.meta_file,
        data_root=args.data_root,
        load_3dgut_params=args.three_dgut
    )
    
    # 샘플 확인
    sample = dataset[0]
    print("\nSample 0:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} {value.dtype}")
        else:
            print(f"  {key}: {value}")
