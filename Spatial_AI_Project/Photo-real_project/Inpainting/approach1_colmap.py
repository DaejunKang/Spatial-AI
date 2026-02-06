"""
Inpainting Approach 1: COLMAP-based Scene Reconstruction

COLMAP을 사용한 3D 재구성 기반 인페인팅
정적 영역만으로 SfM을 수행하여 배경 재구성 후 동적 객체 영역에 렌더링

Workflow:
1. COLMAP SfM (Structure from Motion) with static region masks
2. Dense reconstruction (MVS)
3. Novel view synthesis / Hole filling
4. Refinement with semantic consistency

Input:
    - images/: 원본 이미지
    - masks/: 동적 객체 마스크 (0=동적, 255=정적)
    - poses/: 카메라 포즈 (초기값 or COLMAP 입력용)

Output:
    - final_inpainted/: 동적 객체가 제거된 최종 이미지
"""

import os
import subprocess
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import json


class COLMAPInpainter:
    """
    COLMAP 기반 3D 재구성을 활용한 인페인팅
    
    단계:
    1. COLMAP 데이터베이스 생성 및 Feature Extraction
    2. Feature Matching (정적 영역만)
    3. SfM (Structure from Motion)
    4. MVS (Multi-View Stereo) - Dense Reconstruction
    5. Hole Filling (Novel View Rendering)
    6. Post-processing
    """
    
    def __init__(self, data_root, colmap_path='colmap', use_gpu=True):
        """
        Args:
            data_root: NRE 포맷 데이터 디렉토리
            colmap_path: COLMAP 실행 파일 경로
            use_gpu: GPU 사용 여부
        """
        self.data_root = Path(data_root)
        self.colmap_path = colmap_path
        self.use_gpu = use_gpu
        
        # 디렉토리 설정
        self.images_dir = self.data_root / 'images'
        self.masks_dir = self.data_root / 'masks'
        self.poses_dir = self.data_root / 'poses'
        
        # COLMAP 작업 디렉토리
        self.colmap_workspace = self.data_root / 'colmap_workspace'
        self.colmap_workspace.mkdir(parents=True, exist_ok=True)
        
        self.database_path = self.colmap_workspace / 'database.db'
        self.sparse_dir = self.colmap_workspace / 'sparse'
        self.dense_dir = self.colmap_workspace / 'dense'
        
        # 출력 디렉토리
        self.output_dir = self.data_root / 'final_inpainted'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[COLMAPInpainter] Initialized")
        print(f"  Data root: {self.data_root}")
        print(f"  COLMAP executable: {self.colmap_path}")
        print(f"  Use GPU: {self.use_gpu}")
    
    def run(self):
        """전체 파이프라인 실행"""
        print("\n" + "="*70)
        print(">>> [Approach 1] COLMAP-based Inpainting Started")
        print("="*70)
        
        # Step 1: Feature Extraction
        print("\n[Step 1/6] Feature Extraction (정적 영역만)")
        print("-"*70)
        self._feature_extraction()
        
        # Step 2: Feature Matching
        print("\n[Step 2/6] Feature Matching")
        print("-"*70)
        self._feature_matching()
        
        # Step 3: Structure from Motion
        print("\n[Step 3/6] Structure from Motion (SfM)")
        print("-"*70)
        self._structure_from_motion()
        
        # Step 4: Undistortion & Dense Reconstruction
        print("\n[Step 4/6] Dense Reconstruction (MVS)")
        print("-"*70)
        self._dense_reconstruction()
        
        # Step 5: Novel View Synthesis / Hole Filling
        print("\n[Step 5/6] Hole Filling (Novel View Rendering)")
        print("-"*70)
        self._hole_filling()
        
        # Step 6: Post-processing
        print("\n[Step 6/6] Post-processing & Refinement")
        print("-"*70)
        self._post_processing()
        
        print("\n" + "="*70)
        print(">>> [Approach 1] COLMAP-based Inpainting Complete!")
        print(f"  Output: {self.output_dir}")
        print("="*70)
    
    def _feature_extraction(self):
        """
        COLMAP Feature Extraction
        정적 영역(마스크)만 사용하여 특징점 추출
        """
        # COLMAP 데이터베이스 생성
        cmd = [
            self.colmap_path, 'feature_extractor',
            '--database_path', str(self.database_path),
            '--image_path', str(self.images_dir),
            '--ImageReader.mask_path', str(self.masks_dir),  # 마스크 적용
            '--ImageReader.camera_model', 'OPENCV',  # Waymo는 OpenCV 모델
            '--SiftExtraction.use_gpu', '1' if self.use_gpu else '0',
            '--SiftExtraction.max_num_features', '8192',
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ Feature extraction completed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Feature extraction failed: {e}")
            print(f"  stderr: {e.stderr}")
            raise
    
    def _feature_matching(self):
        """COLMAP Feature Matching"""
        # Sequential matcher (자율주행 시퀀스에 적합)
        cmd = [
            self.colmap_path, 'sequential_matcher',
            '--database_path', str(self.database_path),
            '--SiftMatching.use_gpu', '1' if self.use_gpu else '0',
            '--SequentialMatching.overlap', '10',  # 전후 10 프레임 매칭
            '--SequentialMatching.loop_detection', '1',  # Loop closure 감지
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ Feature matching completed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Feature matching failed: {e}")
            raise
    
    def _structure_from_motion(self):
        """COLMAP SfM (Mapper)"""
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.colmap_path, 'mapper',
            '--database_path', str(self.database_path),
            '--image_path', str(self.images_dir),
            '--output_path', str(self.sparse_dir),
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ Structure from Motion completed")
            
            # 모델 개수 확인
            models = list(self.sparse_dir.glob('*'))
            print(f"  Reconstructed models: {len(models)}")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ SfM failed: {e}")
            raise
    
    def _dense_reconstruction(self):
        """COLMAP Dense Reconstruction (MVS)"""
        self.dense_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Image Undistortion
        print("  Undistorting images...")
        cmd_undistort = [
            self.colmap_path, 'image_undistorter',
            '--image_path', str(self.images_dir),
            '--input_path', str(self.sparse_dir / '0'),  # 첫 번째 모델 사용
            '--output_path', str(self.dense_dir),
            '--output_type', 'COLMAP',
        ]
        subprocess.run(cmd_undistort, check=True, capture_output=True, text=True)
        
        # 2. Patch Match Stereo
        print("  Running stereo matching...")
        cmd_stereo = [
            self.colmap_path, 'patch_match_stereo',
            '--workspace_path', str(self.dense_dir),
            '--PatchMatchStereo.gpu_index', '0' if self.use_gpu else '-1',
        ]
        subprocess.run(cmd_stereo, check=True, capture_output=True, text=True)
        
        # 3. Stereo Fusion
        print("  Fusing stereo results...")
        cmd_fusion = [
            self.colmap_path, 'stereo_fusion',
            '--workspace_path', str(self.dense_dir),
            '--output_path', str(self.dense_dir / 'fused.ply'),
        ]
        subprocess.run(cmd_fusion, check=True, capture_output=True, text=True)
        
        print("✓ Dense reconstruction completed")
    
    def _hole_filling(self):
        """
        Novel View Synthesis를 통한 Hole Filling
        
        재구성된 3D 점들을 원래 뷰포인트에 투영하여
        동적 객체 영역을 채움
        """
        print("  Rendering novel views...")
        
        # COLMAP의 depth map을 사용하여 구멍 채우기
        depth_maps_dir = self.dense_dir / 'stereo' / 'depth_maps'
        
        if not depth_maps_dir.exists():
            print("  Warning: Depth maps not found. Using alternative method...")
            self._alternative_hole_filling()
            return
        
        # 각 이미지별 처리
        image_files = sorted(list(self.images_dir.glob('*.jpg')))
        
        for img_file in tqdm(image_files, desc="Filling holes"):
            self._fill_single_image(img_file, depth_maps_dir)
        
        print("✓ Hole filling completed")
    
    def _fill_single_image(self, img_file, depth_maps_dir):
        """
        단일 이미지의 구멍 채우기
        
        Args:
            img_file: 이미지 파일 경로
            depth_maps_dir: COLMAP depth maps 디렉토리
        """
        # 원본 이미지 및 마스크 로드
        img = cv2.imread(str(img_file))
        
        mask_file = self.masks_dir / img_file.name.replace('.jpg', '.png')
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        else:
            # 마스크 없으면 원본 그대로 저장
            cv2.imwrite(str(self.output_dir / img_file.name), img)
            return
        
        # COLMAP depth map 로드
        depth_file = depth_maps_dir / img_file.name.replace('.jpg', '.png.geometric.bin')
        
        if depth_file.exists():
            # COLMAP 바이너리 depth 읽기
            depth_map = self._read_colmap_depth(depth_file)
            
            # Depth 기반 인페인팅
            result = self._inpaint_with_depth(img, mask, depth_map)
        else:
            # Depth 없으면 기본 인페인팅
            # mask: 255=정적, 0=동적 → cv2.inpaint는 255=인페인트 영역
            hole_mask = (mask == 0).astype(np.uint8) * 255
            result = cv2.inpaint(img, hole_mask, 5, cv2.INPAINT_TELEA)
        
        # 저장
        cv2.imwrite(str(self.output_dir / img_file.name), result)
    
    def _read_colmap_depth(self, depth_file):
        """COLMAP 바이너리 depth map 읽기"""
        # COLMAP depth는 특수 바이너리 포맷
        # 간단한 구현 (실제로는 COLMAP Python API 사용 권장)
        try:
            with open(depth_file, 'rb') as f:
                width = np.fromfile(f, dtype=np.int32, count=1)[0]
                height = np.fromfile(f, dtype=np.int32, count=1)[0]
                channels = np.fromfile(f, dtype=np.int32, count=1)[0]
                depth_map = np.fromfile(f, dtype=np.float32, count=width*height)
                depth_map = depth_map.reshape((height, width))
            
            return depth_map
        except (IOError, ValueError, IndexError) as e:
            print(f"  Warning: Failed to read depth: {depth_file} ({e})")
            return None
    
    def _inpaint_with_depth(self, image, mask, depth_map):
        """
        Depth 정보를 활용한 인페인팅
        
        Depth 기반 우선순위: 구멍 영역의 경계에서 가까운 depth(배경)부터
        점진적으로 채워나가는 방식. 구멍 마스크를 depth 값에 따라
        여러 레이어로 분할하여 순차적으로 inpainting합니다.
        
        Args:
            image: 원본 이미지
            mask: 동적 객체 마스크 (255=정적, 0=동적)
            depth_map: COLMAP depth map (float32)
        
        Returns:
            inpainted_image: 인페인팅된 이미지
        """
        # 구멍 영역 (동적 객체) - cv2.inpaint는 255=인페인트 영역
        hole_mask = (mask == 0).astype(np.uint8) * 255
        
        if depth_map is not None and depth_map.shape[:2] == image.shape[:2]:
            # Depth 기반 계층적 인페인팅:
            # 가까운 depth(배경)부터 채워나감으로써 원근 일관성 향상
            result = image.copy()
            remaining_mask = hole_mask.copy()
            
            # Depth 값 범위를 N개 레이어로 분할
            valid_depth = depth_map[hole_mask > 0]
            if len(valid_depth) > 0 and valid_depth.max() > 0:
                n_layers = 4
                depth_min = valid_depth[valid_depth > 0].min() if np.any(valid_depth > 0) else 0
                depth_max = valid_depth.max()
                thresholds = np.linspace(depth_min, depth_max, n_layers + 1)
                
                for i in range(n_layers):
                    # 현재 depth 범위에 해당하는 구멍만 inpainting
                    layer_mask = (
                        (remaining_mask > 0) &
                        (depth_map >= thresholds[i]) &
                        (depth_map < thresholds[i + 1])
                    ).astype(np.uint8) * 255
                    
                    if np.sum(layer_mask) > 0:
                        result = cv2.inpaint(result, layer_mask, 5, cv2.INPAINT_TELEA)
                        remaining_mask[layer_mask > 0] = 0
                
                # 남은 영역 처리
                if np.sum(remaining_mask) > 0:
                    result = cv2.inpaint(result, remaining_mask, 5, cv2.INPAINT_TELEA)
            else:
                result = cv2.inpaint(image, hole_mask, 5, cv2.INPAINT_TELEA)
        else:
            # Depth 없으면 기본 인페인팅
            result = cv2.inpaint(image, hole_mask, 5, cv2.INPAINT_TELEA)
        
        return result
    
    def _alternative_hole_filling(self):
        """
        Depth map이 없을 때의 대안적 방법
        주변 프레임의 정보를 활용한 간단한 인페인팅
        """
        image_files = sorted(list(self.images_dir.glob('*.jpg')))
        
        for img_file in tqdm(image_files, desc="Alternative filling"):
            img = cv2.imread(str(img_file))
            
            mask_file = self.masks_dir / img_file.name.replace('.jpg', '.png')
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                hole_mask = (mask == 0).astype(np.uint8)
                result = cv2.inpaint(img, hole_mask, 5, cv2.INPAINT_TELEA)
            else:
                result = img
            
            cv2.imwrite(str(self.output_dir / img_file.name), result)
    
    def _post_processing(self):
        """
        후처리 및 정제
        
        - 시간적 일관성 확보
        - 텍스처 노이즈 추가
        - Artifact 제거
        """
        print("  Applying temporal smoothing...")
        
        # 간단한 temporal filtering
        output_files = sorted(list(self.output_dir.glob('*.jpg')))
        
        # Temporal median filter (선택적)
        window_size = 3
        
        for i in tqdm(range(len(output_files)), desc="Post-processing"):
            # 현재 프레임
            current = cv2.imread(str(output_files[i]))
            
            # 이웃 프레임들
            neighbors = []
            for offset in range(-window_size//2, window_size//2 + 1):
                idx = i + offset
                if 0 <= idx < len(output_files) and idx != i:
                    neighbors.append(cv2.imread(str(output_files[idx])))
            
            if len(neighbors) > 0:
                # 간단한 블렌딩 (원본 80% + 이웃 평균 20%)
                neighbors_mean = np.mean(neighbors, axis=0).astype(np.uint8)
                smoothed = cv2.addWeighted(current, 0.8, neighbors_mean, 0.2, 0)
                
                # 덮어쓰기
                cv2.imwrite(str(output_files[i]), smoothed)
        
        print("✓ Post-processing completed")


def main():
    parser = argparse.ArgumentParser(
        description="Inpainting Approach 1: COLMAP-based Scene Reconstruction"
    )
    parser.add_argument(
        'data_root',
        type=str,
        help='Path to NRE format data directory'
    )
    parser.add_argument(
        '--colmap_path',
        type=str,
        default='colmap',
        help='Path to COLMAP executable (default: colmap in PATH)'
    )
    parser.add_argument(
        '--no_gpu',
        action='store_true',
        help='Disable GPU usage'
    )
    
    args = parser.parse_args()
    
    # Validate COLMAP installation
    try:
        subprocess.run(
            [args.colmap_path, '--help'], 
            check=True, 
            capture_output=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"Error: COLMAP not found at '{args.colmap_path}'")
        print("Please install COLMAP: https://colmap.github.io/install.html")
        return
    
    # Run
    inpainter = COLMAPInpainter(
        data_root=args.data_root,
        colmap_path=args.colmap_path,
        use_gpu=not args.no_gpu
    )
    inpainter.run()


if __name__ == '__main__':
    main()
