# -*- coding: utf-8 -*-
"""
데이터 증강 도구
이미지, 포인트 클라우드 등에 데이터 증강을 적용합니다.
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict
import cv2
from tqdm import tqdm


class PointCloudAugmentor:
    """포인트 클라우드 증강 클래스"""
    
    @staticmethod
    def random_flip(points: np.ndarray, flip_x: bool = False, 
                   flip_y: bool = False) -> np.ndarray:
        """
        포인트 클라우드를 랜덤하게 뒤집습니다.
        
        Args:
            points: 포인트 클라우드 (N, 3) 또는 (N, 4)
            flip_x: X축 뒤집기
            flip_y: Y축 뒤집기
        
        Returns:
            뒤집힌 포인트 클라우드
        """
        if flip_x:
            points[:, 0] = -points[:, 0]
        if flip_y:
            points[:, 1] = -points[:, 1]
        return points
    
    @staticmethod
    def random_rotation(points: np.ndarray, angle_range: tuple = (-np.pi, np.pi)) -> np.ndarray:
        """
        포인트 클라우드를 랜덤하게 회전합니다.
        
        Args:
            points: 포인트 클라우드 (N, 3) 또는 (N, 4)
            angle_range: 회전 각도 범위 (라디안)
        
        Returns:
            회전된 포인트 클라우드
        """
        angle = np.random.uniform(angle_range[0], angle_range[1])
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
        
        points[:, :3] = points[:, :3] @ rotation_matrix.T
        return points
    
    @staticmethod
    def random_scale(points: np.ndarray, scale_range: tuple = (0.95, 1.05)) -> np.ndarray:
        """
        포인트 클라우드를 랜덤하게 스케일링합니다.
        
        Args:
            points: 포인트 클라우드
            scale_range: 스케일 범위
        
        Returns:
            스케일된 포인트 클라우드
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        points[:, :3] *= scale
        return points
    
    @staticmethod
    def random_translation(points: np.ndarray, 
                          translation_range: tuple = (-0.2, 0.2)) -> np.ndarray:
        """
        포인트 클라우드를 랜덤하게 이동합니다.
        
        Args:
            points: 포인트 클라우드
            translation_range: 이동 범위 (미터)
        
        Returns:
            이동된 포인트 클라우드
        """
        translation = np.random.uniform(translation_range[0], translation_range[1], size=3)
        points[:, :3] += translation
        return points
    
    @staticmethod
    def random_dropout(points: np.ndarray, dropout_ratio: float = 0.1) -> np.ndarray:
        """
        포인트를 랜덤하게 제거합니다.
        
        Args:
            points: 포인트 클라우드
            dropout_ratio: 제거 비율
        
        Returns:
            제거된 포인트 클라우드
        """
        num_points = len(points)
        num_keep = int(num_points * (1 - dropout_ratio))
        indices = np.random.choice(num_points, num_keep, replace=False)
        return points[indices]


class ImageAugmentor:
    """이미지 증강 클래스"""
    
    @staticmethod
    def random_flip(image: np.ndarray, flip_horizontal: bool = False,
                   flip_vertical: bool = False) -> np.ndarray:
        """이미지를 뒤집습니다."""
        if flip_horizontal:
            image = cv2.flip(image, 1)
        if flip_vertical:
            image = cv2.flip(image, 0)
        return image
    
    @staticmethod
    def random_brightness(image: np.ndarray, factor_range: tuple = (0.8, 1.2)) -> np.ndarray:
        """밝기를 조정합니다."""
        factor = np.random.uniform(factor_range[0], factor_range[1])
        image = np.clip(image * factor, 0, 255).astype(np.uint8)
        return image
    
    @staticmethod
    def random_contrast(image: np.ndarray, factor_range: tuple = (0.8, 1.2)) -> np.ndarray:
        """대비를 조정합니다."""
        factor = np.random.uniform(factor_range[0], factor_range[1])
        mean = image.mean()
        image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        return image
    
    @staticmethod
    def random_hue_saturation(image: np.ndarray, 
                               hue_range: tuple = (-10, 10),
                               sat_range: tuple = (0.8, 1.2)) -> np.ndarray:
        """색상과 채도를 조정합니다."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        
        # Hue 조정
        hue_shift = np.random.uniform(hue_range[0], hue_range[1])
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        # Saturation 조정
        sat_factor = np.random.uniform(sat_range[0], sat_range[1])
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
        
        hsv = hsv.astype(np.uint8)
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return image


def augment_kitti_dataset(data_root: str, output_root: str, 
                          num_augmentations: int = 2):
    """
    KITTI 데이터셋에 증강을 적용합니다.
    
    Args:
        data_root: 데이터셋 루트 디렉토리
        output_root: 출력 디렉토리
        num_augmentations: 각 샘플당 증강 횟수
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 디렉토리 생성
    for split in ['training', 'testing']:
        for subdir in ['velodyne', 'image_2', 'label_2']:
            (output_root / split / subdir).mkdir(parents=True, exist_ok=True)
    
    lidar_dir = data_root / 'training' / 'velodyne'
    image_dir = data_root / 'training' / 'image_2'
    label_dir = data_root / 'training' / 'label_2'
    
    lidar_files = sorted(lidar_dir.glob('*.bin'))
    
    pc_aug = PointCloudAugmentor()
    img_aug = ImageAugmentor()
    
    print(f"총 {len(lidar_files)}개 파일 처리 중...")
    
    for lidar_file in tqdm(lidar_files):
        # 원본 파일 복사
        base_name = lidar_file.stem
        
        # 원본 복사
        import shutil
        shutil.copy(lidar_file, output_root / 'training' / 'velodyne' / f'{base_name}.bin')
        if (image_dir / f'{base_name}.png').exists():
            shutil.copy(image_dir / f'{base_name}.png', 
                       output_root / 'training' / 'image_2' / f'{base_name}.png')
        if (label_dir / f'{base_name}.txt').exists():
            shutil.copy(label_dir / f'{base_name}.txt',
                       output_root / 'training' / 'label_2' / f'{base_name}.txt')
        
        # 증강 적용
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        
        for aug_idx in range(num_augmentations):
            aug_points = points.copy()
            
            # 포인트 클라우드 증강
            if np.random.rand() > 0.5:
                aug_points = pc_aug.random_flip(aug_points, 
                                              flip_x=np.random.rand() > 0.5,
                                              flip_y=np.random.rand() > 0.5)
            
            if np.random.rand() > 0.5:
                aug_points = pc_aug.random_rotation(aug_points)
            
            if np.random.rand() > 0.5:
                aug_points = pc_aug.random_scale(aug_points)
            
            # 증강된 파일 저장
            aug_name = f'{base_name}_aug{aug_idx}'
            aug_points.tofile(str(output_root / 'training' / 'velodyne' / f'{aug_name}.bin'))
            
            # 이미지 증강
            if (image_dir / f'{base_name}.png').exists():
                image = cv2.imread(str(image_dir / f'{base_name}.png'))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                if np.random.rand() > 0.5:
                    image = img_aug.random_flip(image, 
                                              flip_horizontal=np.random.rand() > 0.5)
                
                if np.random.rand() > 0.5:
                    image = img_aug.random_brightness(image)
                
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_root / 'training' / 'image_2' / f'{aug_name}.png'), image)
            
            # 라벨 복사 (간단한 경우, 실제로는 변환 필요)
            if (label_dir / f'{base_name}.txt').exists():
                shutil.copy(label_dir / f'{base_name}.txt',
                           output_root / 'training' / 'label_2' / f'{aug_name}.txt')


def main():
    parser = argparse.ArgumentParser(description='데이터 증강 도구')
    parser.add_argument('--data-root', type=str, required=True,
                       help='데이터셋 루트 디렉토리')
    parser.add_argument('--output-root', type=str, required=True,
                       help='출력 디렉토리')
    parser.add_argument('--dataset-type', type=str, default='kitti',
                       choices=['kitti', 'nuscenes'],
                       help='데이터셋 타입')
    parser.add_argument('--num-augmentations', type=int, default=2,
                       help='각 샘플당 증강 횟수')
    
    args = parser.parse_args()
    
    if args.dataset_type == 'kitti':
        augment_kitti_dataset(args.data_root, args.output_root, 
                            args.num_augmentations)
    else:
        print(f"⚠ {args.dataset_type} 데이터셋 증강은 아직 구현되지 않았습니다.")


if __name__ == '__main__':
    main()

