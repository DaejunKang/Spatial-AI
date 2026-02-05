"""
Dynamic Object Masking for Inpainting

3D Bounding Box 투영과 Semantic Segmentation을 결합하여
동적 객체 마스크 생성 (Inpainting Stage의 Input으로 활용)

Input:
    - images/*.jpg: 원본 이미지
    - objects/*.json: 동적 객체 3D Bounding Box
    - poses/*.json: 카메라 포즈 및 메타데이터

Output:
    - masks/{cam_name}/*.png: 동적 객체 마스크 (0=동적, 255=정적)
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import math


class DynamicObjectMasker:
    """
    동적 객체 마스크 생성 클래스
    
    주요 기능:
    1. 3D Bounding Box 투영 (정확한 기하학적 마스킹)
    2. Semantic Segmentation (선택적, 보완용)
    3. 시공간 동기화 (타임스탬프 기반)
    """
    
    def __init__(self, data_root, use_semantic_seg=False, dilation_kernel=5):
        """
        Args:
            data_root: NRE 포맷 데이터 루트 디렉토리
            use_semantic_seg: Semantic segmentation 사용 여부
            dilation_kernel: 마스크 팽창 커널 크기 (안전 마진)
        """
        self.data_root = Path(data_root)
        self.use_semantic_seg = use_semantic_seg
        self.dilation_kernel = dilation_kernel
        
        # 디렉토리 설정
        self.images_dir = self.data_root / 'images'
        self.objects_dir = self.data_root / 'objects'
        self.poses_dir = self.data_root / 'poses'
        
        # 출력 디렉토리
        self.masks_dir = self.data_root / 'masks'
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Semantic Segmentation 모델 (선택적)
        self.semantic_model = None
        if self.use_semantic_seg:
            self._initialize_semantic_model()
        
        # 파일 목록
        self.object_files = sorted(list(self.objects_dir.glob('*.json')))
        
        if len(self.object_files) == 0:
            raise FileNotFoundError(f"No object files found in {self.objects_dir}")
        
        print(f"[DynamicObjectMasker] Initialized")
        print(f"  Data root: {self.data_root}")
        print(f"  Object files: {len(self.object_files)}")
        print(f"  Use semantic segmentation: {self.use_semantic_seg}")
        print(f"  Dilation kernel: {self.dilation_kernel}x{self.dilation_kernel}")
    
    def _initialize_semantic_model(self):
        """Semantic Segmentation 모델 초기화"""
        try:
            from .segmentation import SemanticSegmentor
            print("  Loading Semantic Segmentation model...")
            self.semantic_model = SemanticSegmentor()
            print("  Semantic model loaded successfully")
        except Exception as e:
            print(f"  Warning: Failed to load semantic model: {e}")
            print("  Will proceed with 3D bounding box only")
            self.use_semantic_seg = False
    
    def run(self):
        """전체 마스킹 파이프라인 실행"""
        print("\n" + "="*70)
        print(">>> Dynamic Object Masking Started")
        print("="*70)
        
        success_count = 0
        fail_count = 0
        
        for obj_file in tqdm(self.object_files, desc="Processing frames"):
            try:
                self._process_frame(obj_file)
                success_count += 1
            except Exception as e:
                print(f"\nWarning: Failed to process {obj_file.name}: {e}")
                fail_count += 1
                continue
        
        print("\n" + "="*70)
        print(f">>> Dynamic Object Masking Complete!")
        print(f"  Success: {success_count}/{len(self.object_files)}")
        print(f"  Failed: {fail_count}/{len(self.object_files)}")
        print(f"  Masks saved to: {self.masks_dir}")
        print("="*70)
    
    def _process_frame(self, obj_file):
        """
        개별 프레임 처리
        
        Args:
            obj_file: 동적 객체 JSON 파일 경로
        """
        frame_name = obj_file.stem
        
        # 1. Load Dynamic Objects (3D Bounding Boxes)
        with open(obj_file, 'r') as f:
            objects = json.load(f)
        
        # 2. Load Pose & Camera Metadata
        pose_file = self.poses_dir / f"{frame_name}.json"
        if not pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {pose_file}")
        
        with open(pose_file, 'r') as f:
            frame_data = json.load(f)
        
        # 3. Generate Mask for Each Camera
        for cam_name, cam_data in frame_data['cameras'].items():
            self._generate_camera_mask(
                objects, 
                cam_data, 
                cam_name, 
                frame_name
            )
    
    def _generate_camera_mask(self, objects, cam_data, cam_name, frame_name):
        """
        특정 카메라에 대한 동적 객체 마스크 생성
        
        Args:
            objects: 동적 객체 리스트 (3D bounding boxes)
            cam_data: 카메라 메타데이터
            cam_name: 카메라 이름
            frame_name: 프레임 이름
        """
        # 1. 카메라 파라미터 추출
        width = cam_data['width']
        height = cam_data['height']
        intrinsics = np.array(cam_data['intrinsics'])
        T_cam_to_world = np.array(cam_data['pose']).reshape(4, 4)
        
        # 2. World -> Camera 변환 행렬
        T_world_to_cam = np.linalg.inv(T_cam_to_world)
        
        # 3. 마스크 초기화 (255 = 정적/유효, 0 = 동적/무효)
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # 4. 각 동적 객체를 마스크에 투영
        for obj in objects:
            box_center = np.array(obj['box']['center'])  # [x, y, z] in World
            box_size = obj['box']['size']  # [length, width, height]
            box_heading = obj['box']['heading']  # Yaw angle (rad)
            
            # 3D Box → 2D Projection
            projected_points = self._project_3d_box_to_2d(
                box_center,
                box_size,
                box_heading,
                T_world_to_cam,
                intrinsics,
                width,
                height
            )
            
            # 투영된 2D 점들로 마스크 채우기
            if projected_points is not None and len(projected_points) > 0:
                hull = cv2.convexHull(projected_points)
                cv2.fillConvexPoly(mask, hull, 0)  # 검은색(0) = 동적 객체
        
        # 5. Semantic Segmentation 결합 (선택적)
        if self.use_semantic_seg:
            img_path = self.images_dir / cam_data['img_path']
            if img_path.exists():
                semantic_mask = self.semantic_model.process_image(str(img_path))
                # 두 마스크의 교집합 (더 보수적)
                mask = cv2.bitwise_and(mask, semantic_mask)
        
        # 6. 마스크 팽창 (안전 마진 - Inpainting 품질 향상)
        if self.dilation_kernel > 0:
            kernel = np.ones((self.dilation_kernel, self.dilation_kernel), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)  # 동적 영역(0) 확장
        
        # 7. 저장
        self._save_mask(mask, cam_name, frame_name)
    
    def _project_3d_box_to_2d(self, center, size, heading, T_world_to_cam, intrinsics, width, height):
        """
        3D Bounding Box를 2D 이미지로 투영
        
        Args:
            center: [x, y, z] 박스 중심 (World 좌표계)
            size: [length, width, height] 박스 크기
            heading: Yaw angle (rad)
            T_world_to_cam: World to Camera 변환 행렬 (4x4)
            intrinsics: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
            width: 이미지 너비
            height: 이미지 높이
        
        Returns:
            projected_points: Nx2 투영된 2D 포인트, 또는 None
        """
        l, w, h = size
        
        # 1. 3D Box 8개 코너 생성 (Box 중심 기준)
        # Rotation around Z-axis (Yaw)
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        R_z = np.array([
            [cos_h, -sin_h, 0],
            [sin_h, cos_h, 0],
            [0, 0, 1]
        ])
        
        # 8 corners (relative to center)
        x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
        y_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
        z_corners = np.array([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2])
        
        corners_3d = np.vstack([x_corners, y_corners, z_corners])  # 3x8
        
        # Rotate
        corners_3d = R_z @ corners_3d
        
        # Translate to world position
        corners_3d = corners_3d + center.reshape(3, 1)
        
        # 2. World -> Camera 변환
        corners_world_homo = np.vstack([corners_3d, np.ones((1, 8))])  # 4x8
        corners_cam_homo = T_world_to_cam @ corners_world_homo  # 4x8
        corners_cam = corners_cam_homo[:3, :]  # 3x8
        
        # 3. 카메라 앞에 있는 코너만 선택 (Z > 0)
        valid_mask = corners_cam[2, :] > 0.1
        if not np.any(valid_mask):
            return None  # 모든 코너가 카메라 뒤쪽
        
        corners_cam_valid = corners_cam[:, valid_mask].T  # Nx3
        
        # 4. 3D -> 2D 투영 (OpenCV with distortion)
        fx, fy, cx, cy = intrinsics[:4]
        k1, k2, p1, p2, k3 = intrinsics[4:9]
        
        camera_matrix = np.array([[fx, 0, cx], 
                                   [0, fy, cy], 
                                   [0, 0, 1]])
        dist_coeffs = np.array([k1, k2, p1, p2, k3])
        
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        
        try:
            projected_2d, _ = cv2.projectPoints(
                corners_cam_valid,
                rvec,
                tvec,
                camera_matrix,
                dist_coeffs
            )
            
            projected_2d = projected_2d.squeeze().astype(np.int32)
            
            # 이미지 내부 포인트만 (최소 하나라도 있어야 함)
            valid_x = (projected_2d[:, 0] >= 0) & (projected_2d[:, 0] < width)
            valid_y = (projected_2d[:, 1] >= 0) & (projected_2d[:, 1] < height)
            valid = valid_x & valid_y
            
            if np.any(valid):
                return projected_2d[valid]
            else:
                # 일부만 이미지 밖이면 경계로 클리핑
                projected_2d[:, 0] = np.clip(projected_2d[:, 0], 0, width - 1)
                projected_2d[:, 1] = np.clip(projected_2d[:, 1], 0, height - 1)
                return projected_2d
        
        except Exception as e:
            print(f"Projection error: {e}")
            return None
    
    def _save_mask(self, mask, cam_name, frame_name):
        """마스크 저장"""
        # 카메라별 디렉토리 생성
        mask_cam_dir = self.masks_dir / cam_name
        mask_cam_dir.mkdir(parents=True, exist_ok=True)
        
        # 저장
        mask_path = mask_cam_dir / f"{frame_name}.png"
        cv2.imwrite(str(mask_path), mask)


def main():
    parser = argparse.ArgumentParser(
        description="Generate dynamic object masks for inpainting"
    )
    parser.add_argument(
        'data_root',
        type=str,
        help='Path to NRE format data directory'
    )
    parser.add_argument(
        '--use_semantic',
        action='store_true',
        help='Use semantic segmentation in addition to 3D boxes'
    )
    parser.add_argument(
        '--dilation',
        type=int,
        default=5,
        help='Dilation kernel size for safety margin (default: 5)'
    )
    
    args = parser.parse_args()
    
    # 실행
    masker = DynamicObjectMasker(
        data_root=args.data_root,
        use_semantic_seg=args.use_semantic,
        dilation_kernel=args.dilation
    )
    masker.run()


if __name__ == '__main__':
    main()
