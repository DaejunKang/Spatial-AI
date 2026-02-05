"""
LiDAR Point Cloud to Multi-View Image Projection

시공간 동기화된 LiDAR 포인트를 다중 카메라 이미지에 투영하여
깊이 맵 및 검증용 데이터 생성

Input:
    - point_clouds/*.bin: LiDAR 포인트 클라우드 (Nx3 float32)
    - poses/*.json: 프레임별 카메라 포즈 및 메타데이터
    - images/*.jpg: 원본 이미지

Output:
    - depth_maps/{cam_name}/*.png: 투영된 깊이 맵 (uint16)
    - point_masks/{cam_name}/*.png: LiDAR 포인트 마스크 (uint8)
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse


class LiDARProjector:
    """
    LiDAR 포인트 클라우드를 다중 뷰 이미지에 투영하는 클래스
    
    주요 기능:
    1. 시공간 동기화: 타임스탬프 기반 LiDAR-Image 매칭
    2. 3D-2D 투영: 카메라 intrinsic/extrinsic 활용
    3. 깊이 맵 생성: 투영된 포인트로부터 dense depth map 생성
    """
    
    def __init__(self, data_root, interpolation_method='nearest'):
        """
        Args:
            data_root: NRE 포맷 데이터 루트 디렉토리
            interpolation_method: 깊이 맵 보간 방법 ('nearest', 'linear', 'cubic')
        """
        self.data_root = Path(data_root)
        self.interpolation_method = interpolation_method
        
        # 디렉토리 설정
        self.point_cloud_dir = self.data_root / 'point_clouds'
        self.poses_dir = self.data_root / 'poses'
        self.images_dir = self.data_root / 'images'
        
        # 출력 디렉토리
        self.depth_maps_dir = self.data_root / 'depth_maps'
        self.point_masks_dir = self.data_root / 'point_masks'
        
        self.depth_maps_dir.mkdir(parents=True, exist_ok=True)
        self.point_masks_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 목록 수집
        self.point_cloud_files = sorted(list(self.point_cloud_dir.glob('*.bin')))
        
        if len(self.point_cloud_files) == 0:
            raise FileNotFoundError(f"No point cloud files found in {self.point_cloud_dir}")
        
        print(f"[LiDARProjector] Initialized")
        print(f"  Data root: {self.data_root}")
        print(f"  Point cloud files: {len(self.point_cloud_files)}")
        print(f"  Interpolation: {self.interpolation_method}")
    
    def run(self):
        """전체 투영 파이프라인 실행"""
        print("\n" + "="*70)
        print(">>> LiDAR Point Cloud to Multi-View Image Projection Started")
        print("="*70)
        
        success_count = 0
        fail_count = 0
        
        for pc_file in tqdm(self.point_cloud_files, desc="Processing frames"):
            try:
                self._process_frame(pc_file)
                success_count += 1
            except Exception as e:
                print(f"\nWarning: Failed to process {pc_file.name}: {e}")
                fail_count += 1
                continue
        
        print("\n" + "="*70)
        print(f">>> LiDAR Projection Complete!")
        print(f"  Success: {success_count}/{len(self.point_cloud_files)}")
        print(f"  Failed: {fail_count}/{len(self.point_cloud_files)}")
        print(f"  Depth maps saved to: {self.depth_maps_dir}")
        print(f"  Point masks saved to: {self.point_masks_dir}")
        print("="*70)
    
    def _process_frame(self, pc_file):
        """
        개별 프레임 처리
        
        Args:
            pc_file: 포인트 클라우드 파일 경로
        """
        frame_name = pc_file.stem
        
        # 1. Load Point Cloud (Nx3 float32, Local World 좌표계)
        points_world = np.fromfile(str(pc_file), dtype=np.float32).reshape(-1, 3)
        
        if len(points_world) == 0:
            print(f"Warning: Empty point cloud for {frame_name}")
            return
        
        # 2. Load Pose & Camera Metadata
        pose_file = self.poses_dir / f"{frame_name}.json"
        if not pose_file.exists():
            raise FileNotFoundError(f"Pose file not found: {pose_file}")
        
        with open(pose_file, 'r') as f:
            frame_data = json.load(f)
        
        # 3. Project to Each Camera
        for cam_name, cam_data in frame_data['cameras'].items():
            self._project_to_camera(
                points_world, 
                cam_data, 
                cam_name, 
                frame_name
            )
    
    def _project_to_camera(self, points_world, cam_data, cam_name, frame_name):
        """
        포인트 클라우드를 특정 카메라에 투영
        
        Args:
            points_world: Nx3 포인트 (Local World 좌표계)
            cam_data: 카메라 메타데이터 (intrinsic, pose, etc.)
            cam_name: 카메라 이름
            frame_name: 프레임 이름
        """
        # 1. 카메라 파라미터 추출
        width = cam_data['width']
        height = cam_data['height']
        intrinsics = np.array(cam_data['intrinsics'])  # [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        T_cam_to_world = np.array(cam_data['pose']).reshape(4, 4)
        
        # 2. World -> Camera 변환
        T_world_to_cam = np.linalg.inv(T_cam_to_world)
        
        # Homogeneous coordinates
        points_world_homo = np.hstack([points_world, np.ones((len(points_world), 1))])
        points_cam_homo = (T_world_to_cam @ points_world_homo.T).T  # Nx4
        points_cam = points_cam_homo[:, :3]  # Nx3
        
        # 3. 카메라 앞에 있는 포인트만 선택 (Z > 0)
        valid_mask = points_cam[:, 2] > 0.1  # 0.1m 이상
        points_cam_valid = points_cam[valid_mask]
        
        if len(points_cam_valid) == 0:
            # 투영 가능한 포인트 없음
            self._save_empty_outputs(cam_name, frame_name, width, height)
            return
        
        # 4. 3D -> 2D 투영 (with distortion)
        projected_points = self._project_with_distortion(
            points_cam_valid, 
            intrinsics, 
            width, 
            height
        )
        
        if projected_points is None or len(projected_points) == 0:
            self._save_empty_outputs(cam_name, frame_name, width, height)
            return
        
        # 5. 깊이 맵 생성
        depth_map = self._create_depth_map(
            projected_points, 
            points_cam_valid[:, 2],  # Z values (depth)
            width, 
            height
        )
        
        # 6. 포인트 마스크 생성 (LiDAR 포인트가 투영된 픽셀)
        point_mask = self._create_point_mask(
            projected_points, 
            width, 
            height
        )
        
        # 7. 저장
        self._save_outputs(depth_map, point_mask, cam_name, frame_name)
    
    def _project_with_distortion(self, points_cam, intrinsics, width, height):
        """
        카메라 왜곡을 포함한 3D-2D 투영
        
        Args:
            points_cam: Nx3 카메라 좌표계 포인트
            intrinsics: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
            width: 이미지 너비
            height: 이미지 높이
        
        Returns:
            projected_points: Mx2 투영된 2D 포인트 (이미지 내부만)
            valid_indices: 유효한 포인트 인덱스
        """
        fx, fy, cx, cy = intrinsics[:4]
        k1, k2, p1, p2, k3 = intrinsics[4:9]
        
        # OpenCV projectPoints 사용
        camera_matrix = np.array([[fx, 0, cx], 
                                   [0, fy, cy], 
                                   [0, 0, 1]])
        dist_coeffs = np.array([k1, k2, p1, p2, k3])
        
        # Identity rotation/translation (이미 카메라 좌표계)
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        
        try:
            projected_2d, _ = cv2.projectPoints(
                points_cam, 
                rvec, 
                tvec, 
                camera_matrix, 
                dist_coeffs
            )
            projected_2d = projected_2d.squeeze()  # Nx2
            
            # 이미지 경계 내부 포인트만 선택
            valid_x = (projected_2d[:, 0] >= 0) & (projected_2d[:, 0] < width)
            valid_y = (projected_2d[:, 1] >= 0) & (projected_2d[:, 1] < height)
            valid_mask = valid_x & valid_y
            
            return projected_2d[valid_mask].astype(np.int32)
        
        except Exception as e:
            print(f"Projection failed: {e}")
            return None
    
    def _create_depth_map(self, projected_points, depths, width, height):
        """
        희소 포인트로부터 깊이 맵 생성
        
        Args:
            projected_points: Nx2 2D 포인트 좌표
            depths: N 깊이 값
            width: 이미지 너비
            height: 이미지 높이
        
        Returns:
            depth_map: (H, W) uint16 깊이 맵 (mm 단위)
        """
        # 초기화 (0 = 깊이 없음)
        depth_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.uint16)
        
        # 각 포인트를 깊이 맵에 누적
        for (x, y), depth in zip(projected_points, depths):
            if 0 <= x < width and 0 <= y < height:
                depth_map[y, x] += depth
                count_map[y, x] += 1
        
        # 평균 계산 (여러 포인트가 같은 픽셀에 투영된 경우)
        valid_mask = count_map > 0
        depth_map[valid_mask] /= count_map[valid_mask]
        
        # 보간 (선택적)
        if self.interpolation_method != 'none':
            depth_map = self._interpolate_depth(depth_map, valid_mask)
        
        # mm 단위로 변환 (uint16 저장용)
        depth_map_mm = (depth_map * 1000).astype(np.uint16)
        
        return depth_map_mm
    
    def _interpolate_depth(self, depth_map, valid_mask):
        """
        희소 깊이 맵을 보간하여 밀집 깊이 맵 생성
        
        Args:
            depth_map: 희소 깊이 맵
            valid_mask: 유효한 픽셀 마스크
        
        Returns:
            dense_depth_map: 보간된 깊이 맵
        """
        if self.interpolation_method == 'nearest':
            # Nearest neighbor 보간 (가장 빠름)
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(valid_mask.astype(np.uint8), kernel, iterations=10)
            
            # Inpainting으로 간단한 보간
            depth_map_inpaint = cv2.inpaint(
                (depth_map * valid_mask).astype(np.float32),
                (1 - valid_mask).astype(np.uint8),
                inpaintRadius=5,
                flags=cv2.INPAINT_NS
            )
            
            return depth_map_inpaint
        
        elif self.interpolation_method == 'linear':
            # TODO: Linear interpolation 구현
            return depth_map
        
        else:
            return depth_map
    
    def _create_point_mask(self, projected_points, width, height):
        """
        LiDAR 포인트가 투영된 픽셀 마스크 생성
        
        Args:
            projected_points: Nx2 2D 포인트
            width: 이미지 너비
            height: 이미지 높이
        
        Returns:
            point_mask: (H, W) uint8 마스크 (255 = LiDAR 포인트 존재)
        """
        point_mask = np.zeros((height, width), dtype=np.uint8)
        
        for x, y in projected_points:
            if 0 <= x < width and 0 <= y < height:
                point_mask[y, x] = 255
        
        # 약간 팽창하여 시각화 개선
        kernel = np.ones((3, 3), np.uint8)
        point_mask = cv2.dilate(point_mask, kernel, iterations=1)
        
        return point_mask
    
    def _save_outputs(self, depth_map, point_mask, cam_name, frame_name):
        """출력 파일 저장"""
        # 카메라별 디렉토리 생성
        depth_cam_dir = self.depth_maps_dir / cam_name
        mask_cam_dir = self.point_masks_dir / cam_name
        
        depth_cam_dir.mkdir(parents=True, exist_ok=True)
        mask_cam_dir.mkdir(parents=True, exist_ok=True)
        
        # 저장
        depth_path = depth_cam_dir / f"{frame_name}.png"
        mask_path = mask_cam_dir / f"{frame_name}.png"
        
        cv2.imwrite(str(depth_path), depth_map)
        cv2.imwrite(str(mask_path), point_mask)
    
    def _save_empty_outputs(self, cam_name, frame_name, width, height):
        """빈 출력 저장 (투영 실패 시)"""
        empty_depth = np.zeros((height, width), dtype=np.uint16)
        empty_mask = np.zeros((height, width), dtype=np.uint8)
        
        self._save_outputs(empty_depth, empty_mask, cam_name, frame_name)


def main():
    parser = argparse.ArgumentParser(
        description="Project LiDAR point clouds to multi-view images"
    )
    parser.add_argument(
        'data_root',
        type=str,
        help='Path to NRE format data directory'
    )
    parser.add_argument(
        '--interpolation',
        type=str,
        default='nearest',
        choices=['none', 'nearest', 'linear', 'cubic'],
        help='Depth map interpolation method (default: nearest)'
    )
    
    args = parser.parse_args()
    
    # 실행
    projector = LiDARProjector(
        data_root=args.data_root,
        interpolation_method=args.interpolation
    )
    projector.run()


if __name__ == '__main__':
    main()
