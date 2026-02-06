"""
Inpainting Step 2: 기하학적 가이드 생성 (Geometric Guide Generation)

Step 1의 시계열 누적 결과에서 여전히 남아있는 구멍을 감지하고,
RANSAC 기반 평면 피팅으로 깊이 값을 기하학적으로 채워
다음 단계의 인페인팅을 위한 가이드를 생성합니다.

Input:
    - data_root/step1_warped/: Step 1 출력 (시계열 누적 이미지)
    - data_root/depths/: 원본 LiDAR depth maps (선택)
    
Output:
    - data_root/step2_depth_guide/: 기하학적으로 채워진 depth guide maps
    - data_root/step2_hole_masks/: 채워야 할 구멍 영역 마스크
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor
import warnings


class GeometricGuideGenerator:
    """
    기하학적 제약을 활용하여 구멍난 영역의 깊이를 추정하는 클래스
    """
    
    def __init__(self, data_root, use_lidar_depth=True, ground_region_ratio=0.6):
        """
        Args:
            data_root: Preprocessing + Step 1 출력 디렉토리
            use_lidar_depth: LiDAR depth를 사용할지 여부 (없으면 pseudo depth 사용)
            ground_region_ratio: 바닥 평면 추정에 사용할 이미지 하단 비율 (0.6 = 하단 40%)
        """
        self.data_root = Path(data_root)
        self.use_lidar_depth = use_lidar_depth
        self.ground_region_ratio = ground_region_ratio
        
        # 디렉토리 설정
        self.step1_dir = self.data_root / 'step1_warped'
        self.depths_dir = self.data_root / 'depths'
        self.output_depth_dir = self.data_root / 'step2_depth_guide'
        self.output_mask_dir = self.data_root / 'step2_hole_masks'
        
        # 출력 디렉토리 생성
        self.output_depth_dir.mkdir(parents=True, exist_ok=True)
        self.output_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1 출력 확인
        if not self.step1_dir.exists():
            raise FileNotFoundError(
                f"Step 1 output not found: {self.step1_dir}\n"
                f"Please run step1_temporal_accumulation.py first."
            )
        
        # 파일 목록
        self.warped_files = sorted(list(self.step1_dir.glob('*.png')))
        
        if len(self.warped_files) == 0:
            raise ValueError(f"No warped images found in {self.step1_dir}")
        
        print(f"[GeometricGuideGenerator] Initialized")
        print(f"  Data root: {self.data_root}")
        print(f"  Step 1 images: {len(self.warped_files)}")
        print(f"  Use LiDAR depth: {self.use_lidar_depth}")
        print(f"  Ground region ratio: {self.ground_region_ratio}")
    
    def run(self):
        """메인 파이프라인 실행"""
        print("\n" + "="*70)
        print(">>> [Step 2] Geometric Guide Generation Started")
        print("="*70)
        
        success_count = 0
        fail_count = 0
        
        for warped_file in tqdm(self.warped_files, desc="Processing frames"):
            try:
                self._process_frame(warped_file)
                success_count += 1
            except Exception as e:
                print(f"\nWarning: Failed to process {warped_file.name}: {e}")
                fail_count += 1
                continue
        
        print("\n" + "="*70)
        print(f">>> [Step 2] Complete!")
        print(f"  Success: {success_count}/{len(self.warped_files)}")
        print(f"  Failed: {fail_count}/{len(self.warped_files)}")
        print(f"  Depth guides saved to: {self.output_depth_dir}")
        print(f"  Hole masks saved to: {self.output_mask_dir}")
        print("="*70)
    
    def _process_frame(self, warped_file):
        """
        개별 프레임 처리
        
        Args:
            warped_file: Step 1 출력 이미지 경로
        """
        # Step 1 결과 로드
        warped_img = cv2.imread(str(warped_file))
        
        if warped_img is None:
            raise ValueError(f"Failed to load image: {warped_file}")
        
        # 구멍 영역 감지 (검은색 픽셀 = 여전히 구멍)
        hole_mask = self._detect_holes(warped_img)
        
        # 원본 depth 로드
        depth_file = self.depths_dir / warped_file.name
        
        if self.use_lidar_depth and depth_file.exists():
            orig_depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            if orig_depth is not None:
                orig_depth = orig_depth.astype(np.float32)
            else:
                # Depth 로드 실패 시 pseudo depth 생성
                orig_depth = self._generate_pseudo_depth(warped_img.shape[:2])
        else:
            # LiDAR depth가 없으면 pseudo depth 생성
            orig_depth = self._generate_pseudo_depth(warped_img.shape[:2])
        
        # 기하학적 평면 피팅으로 구멍 채우기
        completed_depth = self._fill_ground_plane(orig_depth, hole_mask)
        
        # 저장
        output_depth_path = self.output_depth_dir / warped_file.name
        output_mask_path = self.output_mask_dir / warped_file.name
        
        cv2.imwrite(str(output_depth_path), completed_depth.astype(np.uint16))
        cv2.imwrite(str(output_mask_path), hole_mask)
    
    def _detect_holes(self, warped_img, threshold=10):
        """
        Step 1 결과에서 여전히 구멍인 영역 감지
        
        Args:
            warped_img: Step 1 warped 이미지
            threshold: 검은색으로 간주할 임계값
        
        Returns:
            hole_mask: 구멍 영역 마스크 (255=구멍, 0=채워짐)
        """
        # 채널별 합이 낮은 픽셀을 구멍으로 간주
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        hole_mask = (gray < threshold).astype(np.uint8) * 255
        
        # Morphological closing으로 작은 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 너무 작은 구멍은 제거 (연결된 컴포넌트 분석)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            hole_mask, connectivity=8
        )
        
        # 최소 크기 이하의 컴포넌트 제거
        min_hole_size = 50  # 픽셀
        for i in range(1, num_labels):  # 0은 배경
            if stats[i, cv2.CC_STAT_AREA] < min_hole_size:
                hole_mask[labels == i] = 0
        
        return hole_mask
    
    def _generate_pseudo_depth(self, shape, default_depth=10000):
        """
        LiDAR depth가 없을 때 사용할 pseudo depth 생성
        
        Args:
            shape: (height, width)
            default_depth: 기본 깊이 값 (mm 단위)
        
        Returns:
            pseudo_depth: (H, W) uint16 배열
        """
        h, w = shape
        
        # 간단한 선형 gradient (위쪽은 멀고, 아래쪽은 가까움)
        y_coords = np.linspace(default_depth * 2, default_depth * 0.5, h)
        pseudo_depth = np.tile(y_coords[:, np.newaxis], (1, w))
        
        return pseudo_depth.astype(np.uint16)
    
    def _fill_ground_plane(self, depth_map, hole_mask):
        """
        RANSAC을 이용해 깊이맵에서 바닥 평면을 추정하고,
        구멍난 영역의 깊이를 기하학적으로 채움
        
        Args:
            depth_map: 원본 depth map (H, W) float32
            hole_mask: 구멍 영역 마스크 (H, W) uint8
        
        Returns:
            filled_depth: 채워진 depth map (H, W) float32
        """
        h, w = depth_map.shape
        
        # 1. 바닥 평면 추정을 위한 샘플링 영역 설정
        # 이미지 하단 영역 (일반적으로 바닥이 위치)
        y_indices, x_indices = np.indices((h, w))
        ground_region_mask = (y_indices > h * self.ground_region_ratio)
        
        # 유효한 depth 값이 있는 영역 (구멍이 아닌 곳)
        valid_mask = (depth_map > 0) & ground_region_mask & (hole_mask == 0)
        
        valid_count = np.sum(valid_mask)
        
        if valid_count < 100:
            # 데이터가 부족하면 간단한 inpainting으로 대체
            warnings.warn(
                f"Insufficient valid depth points ({valid_count}). "
                f"Using simple inpainting instead."
            )
            filled_depth = self._simple_depth_inpaint(depth_map, hole_mask)
            return filled_depth
        
        # 2. RANSAC 평면 피팅 (Z = a*X + b*Y + c)
        # X, Y: 이미지 좌표, Z: depth 값
        X_train = np.column_stack((
            x_indices[valid_mask],
            y_indices[valid_mask]
        ))
        y_train = depth_map[valid_mask]
        
        try:
            # RANSAC으로 아웃라이어에 강건한 평면 추정
            ransac = RANSACRegressor(
                random_state=42,
                min_samples=100,
                max_trials=100,
                residual_threshold=500.0  # mm 단위
            )
            ransac.fit(X_train, y_train)
            
            # 3. 구멍 영역의 depth 예측
            hole_y, hole_x = np.where(hole_mask > 0)
            
            if len(hole_x) == 0:
                # 구멍이 없으면 원본 반환
                return depth_map
            
            X_pred = np.column_stack((hole_x, hole_y))
            predicted_depth = ransac.predict(X_pred)
            
            # 4. 채우기
            filled_depth = depth_map.copy()
            filled_depth[hole_y, hole_x] = np.clip(predicted_depth, 0, 65535)
            
        except Exception as e:
            warnings.warn(f"RANSAC fitting failed: {e}. Using simple inpainting.")
            filled_depth = self._simple_depth_inpaint(depth_map, hole_mask)
        
        return filled_depth
    
    def _simple_depth_inpaint(self, depth_map, hole_mask):
        """
        RANSAC 실패 시 사용할 간단한 depth inpainting
        
        Note: cv2.inpaint는 8-bit 이미지만 지원하므로,
              depth map을 8-bit로 변환 후 inpaint하고 다시 스케일링합니다.
        
        Args:
            depth_map: 원본 depth map (float32)
            hole_mask: 구멍 마스크 (uint8, 255=구멍)
        
        Returns:
            filled_depth: 채워진 depth map (float32)
        """
        # cv2.inpaint는 uint8 또는 float32 3채널만 지원
        # depth를 0-255 범위의 uint8로 정규화하여 inpainting 수행
        depth_min = np.min(depth_map[depth_map > 0]) if np.any(depth_map > 0) else 0
        depth_max = np.max(depth_map) if np.max(depth_map) > 0 else 1.0
        depth_range = depth_max - depth_min if depth_max > depth_min else 1.0
        
        # 정규화 (0-255)
        depth_normalized = np.clip(
            (depth_map - depth_min) / depth_range * 255, 0, 255
        ).astype(np.uint8)
        
        # cv2.inpaint (8-bit)
        filled_normalized = cv2.inpaint(
            depth_normalized,
            hole_mask,
            inpaintRadius=5,
            flags=cv2.INPAINT_TELEA
        )
        
        # 원래 스케일로 복원
        filled_depth = filled_normalized.astype(np.float32) / 255.0 * depth_range + depth_min
        
        # 원래 유효했던 영역은 원본 값 유지
        valid_mask = (hole_mask == 0) & (depth_map > 0)
        filled_depth[valid_mask] = depth_map[valid_mask]
        
        return filled_depth


def main():
    """CLI 진입점"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Inpainting Step 2: Geometric Guide Generation"
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help="Path to data directory (containing step1_warped/)"
    )
    parser.add_argument(
        '--no_lidar',
        action='store_true',
        help="Don't use LiDAR depth (generate pseudo depth instead)"
    )
    parser.add_argument(
        '--ground_ratio',
        type=float,
        default=0.6,
        help="Ratio of image bottom used for ground plane estimation (default: 0.6)"
    )
    
    args = parser.parse_args()
    
    # 실행
    generator = GeometricGuideGenerator(
        data_root=args.data_root,
        use_lidar_depth=not args.no_lidar,
        ground_region_ratio=args.ground_ratio
    )
    generator.run()


if __name__ == "__main__":
    main()
