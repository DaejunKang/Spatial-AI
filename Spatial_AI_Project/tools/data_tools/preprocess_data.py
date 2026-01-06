# -*- coding: utf-8 -*-
"""
데이터 전처리 도구
- 데이터 정규화
- 데이터 증강
- 데이터 필터링
- 데이터 통계 계산
"""

import argparse
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import pickle


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 전처리 설정 딕셔너리
        """
        self.config = config or {}
    
    def normalize_point_cloud(self, points: np.ndarray, method: str = 'standard') -> np.ndarray:
        """
        포인트 클라우드를 정규화합니다.
        
        Args:
            points: (N, 3) 또는 (N, 4) 형태의 포인트 클라우드
            method: 정규화 방법 ('standard', 'min_max', 'robust')
            
        Returns:
            정규화된 포인트 클라우드
        """
        if method == 'standard':
            # Z-score 정규화
            mean = np.mean(points[:, :3], axis=0)
            std = np.std(points[:, :3], axis=0)
            std = np.where(std == 0, 1, std)  # 0으로 나누기 방지
            normalized = (points[:, :3] - mean) / std
        elif method == 'min_max':
            # Min-Max 정규화
            min_vals = np.min(points[:, :3], axis=0)
            max_vals = np.max(points[:, :3], axis=0)
            ranges = max_vals - min_vals
            ranges = np.where(ranges == 0, 1, ranges)
            normalized = (points[:, :3] - min_vals) / ranges
        elif method == 'robust':
            # Robust 정규화 (median, IQR 사용)
            median = np.median(points[:, :3], axis=0)
            q75 = np.percentile(points[:, :3], 75, axis=0)
            q25 = np.percentile(points[:, :3], 25, axis=0)
            iqr = q75 - q25
            iqr = np.where(iqr == 0, 1, iqr)
            normalized = (points[:, :3] - median) / iqr
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # intensity가 있으면 그대로 유지
        if points.shape[1] >= 4:
            normalized = np.column_stack([normalized, points[:, 3:]])
        
        return normalized
    
    def filter_point_cloud(self, 
                          points: np.ndarray,
                          x_range: Optional[tuple] = None,
                          y_range: Optional[tuple] = None,
                          z_range: Optional[tuple] = None,
                          min_points: int = 0) -> np.ndarray:
        """
        포인트 클라우드를 필터링합니다.
        
        Args:
            points: (N, 3) 또는 (N, 4) 형태의 포인트 클라우드
            x_range: X축 범위 (min, max)
            y_range: Y축 범위 (min, max)
            z_range: Z축 범위 (min, max)
            min_points: 최소 포인트 수
            
        Returns:
            필터링된 포인트 클라우드
        """
        mask = np.ones(len(points), dtype=bool)
        
        if x_range:
            mask &= (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
        if y_range:
            mask &= (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        if z_range:
            mask &= (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
        
        filtered = points[mask]
        
        if len(filtered) < min_points:
            return np.array([])
        
        return filtered
    
    def compute_statistics(self, points: np.ndarray) -> Dict:
        """
        포인트 클라우드 통계를 계산합니다.
        
        Args:
            points: (N, 3) 또는 (N, 4) 형태의 포인트 클라우드
            
        Returns:
            통계 딕셔너리
        """
        stats = {
            'num_points': len(points),
            'x': {
                'min': float(np.min(points[:, 0])),
                'max': float(np.max(points[:, 0])),
                'mean': float(np.mean(points[:, 0])),
                'std': float(np.std(points[:, 0]))
            },
            'y': {
                'min': float(np.min(points[:, 1])),
                'max': float(np.max(points[:, 1])),
                'mean': float(np.mean(points[:, 1])),
                'std': float(np.std(points[:, 1]))
            },
            'z': {
                'min': float(np.min(points[:, 2])),
                'max': float(np.max(points[:, 2])),
                'mean': float(np.mean(points[:, 2])),
                'std': float(np.std(points[:, 2]))
            }
        }
        
        if points.shape[1] >= 4:
            stats['intensity'] = {
                'min': float(np.min(points[:, 3])),
                'max': float(np.max(points[:, 3])),
                'mean': float(np.mean(points[:, 3])),
                'std': float(np.std(points[:, 3]))
            }
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='데이터 전처리 도구')
    parser.add_argument('--input', type=str, required=True, help='입력 파일 경로')
    parser.add_argument('--output', type=str, required=True, help='출력 파일 경로')
    parser.add_argument('--normalize', type=str, choices=['standard', 'min_max', 'robust'],
                       help='정규화 방법')
    parser.add_argument('--filter-x', type=float, nargs=2, metavar=('MIN', 'MAX'),
                       help='X축 필터링 범위')
    parser.add_argument('--filter-y', type=float, nargs=2, metavar=('MIN', 'MAX'),
                       help='Y축 필터링 범위')
    parser.add_argument('--filter-z', type=float, nargs=2, metavar=('MIN', 'MAX'),
                       help='Z축 필터링 범위')
    parser.add_argument('--stats', action='store_true', help='통계 계산 및 저장')
    
    args = parser.parse_args()
    
    # 데이터 로드
    input_path = Path(args.input)
    if input_path.suffix == '.npy':
        points = np.load(args.input)
    elif input_path.suffix == '.pkl':
        with open(args.input, 'rb') as f:
            points = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    # 전처리 수행
    preprocessor = DataPreprocessor()
    
    # 필터링
    if args.filter_x or args.filter_y or args.filter_z:
        points = preprocessor.filter_point_cloud(
            points,
            x_range=tuple(args.filter_x) if args.filter_x else None,
            y_range=tuple(args.filter_y) if args.filter_y else None,
            z_range=tuple(args.filter_z) if args.filter_z else None
        )
    
    # 정규화
    if args.normalize:
        points = preprocessor.normalize_point_cloud(points, method=args.normalize)
    
    # 통계 계산
    if args.stats:
        stats = preprocessor.compute_statistics(points)
        stats_path = Path(args.output).with_suffix('.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Statistics saved to {stats_path}")
    
    # 저장
    output_path = Path(args.output)
    if output_path.suffix == '.npy':
        np.save(args.output, points)
    elif output_path.suffix == '.pkl':
        with open(args.output, 'wb') as f:
            pickle.dump(points, f)
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")
    
    print(f"Processed {len(points)} points")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()

