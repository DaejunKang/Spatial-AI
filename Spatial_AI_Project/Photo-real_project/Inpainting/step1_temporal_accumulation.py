"""
Inpainting Step 1: 시계열 누적 (Temporal Accumulation)

preprocessing 파이프라인의 output (raw images, masking images)을 입력으로 받아서
시계열 정보를 활용한 정적 배경 복원을 수행합니다.

Input:
    - data_root/images/: Raw images from preprocessing
    - data_root/masks/: Masking images (0=동적 객체, 255=정적 영역)
    - data_root/poses/: Camera pose JSON files
    - data_root/depths/: (Optional) LiDAR depth maps

Output:
    - data_root/step1_warped/: 시계열 누적으로 구멍이 메워진 이미지
"""

import os
import json
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path


class TemporalStaticAccumulator:
    """
    시계열 정보를 활용하여 정적 배경 포인트 클라우드를 누적하고
    각 프레임에 다시 투영하여 동적 객체 영역을 채우는 클래스
    """
    
    def __init__(self, data_root, voxel_size=0.05, sample_interval=5):
        """
        Args:
            data_root: preprocessing 출력 디렉토리 (images/, masks/, poses/ 포함)
            voxel_size: Voxel downsampling 크기 (미터 단위)
            sample_interval: Forward pass에서 샘플링 간격 (속도 향상)
        """
        self.data_root = Path(data_root)
        self.voxel_size = voxel_size
        self.sample_interval = sample_interval
        self.global_pcd = o3d.geometry.PointCloud()
        
        # 디렉토리 확인
        self.images_dir = self.data_root / 'images'
        self.masks_dir = self.data_root / 'masks'
        self.poses_dir = self.data_root / 'poses'
        self.depths_dir = self.data_root / 'depths'
        self.output_dir = self.data_root / 'step1_warped'
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pose 파일 목록 로드
        if not self.poses_dir.exists():
            raise FileNotFoundError(f"Poses directory not found: {self.poses_dir}")
        
        self.pose_files = sorted(list(self.poses_dir.glob('*.json')))
        
        if len(self.pose_files) == 0:
            raise ValueError(f"No pose files found in {self.poses_dir}")
        
        print(f"[TemporalStaticAccumulator] Loaded {len(self.pose_files)} frames")
        print(f"  Data root: {self.data_root}")
        print(f"  Voxel size: {self.voxel_size}m")
        print(f"  Sample interval: {self.sample_interval}")
        
    def run(self):
        """메인 파이프라인 실행"""
        print("\n" + "="*70)
        print(">>> [Step 1] Temporal Static Accumulation Started")
        print("="*70)
        
        # Forward Pass: 정적 포인트 클라우드 전역 누적
        print("\n[1/2] Forward Pass: Accumulating static point cloud...")
        self._forward_accumulation()
        
        # Backward Pass: 구멍 메우기 (Reprojection)
        print("\n[2/2] Backward Pass: Reprojecting to fill holes...")
        self._backward_reprojection()
        
        print("\n" + "="*70)
        print(f">>> [Step 1] Complete! Results saved to: {self.output_dir}")
        print("="*70)
        
    def _forward_accumulation(self):
        """
        Forward Pass: 모든 프레임의 정적 영역을 전역 좌표계로 변환하여 누적
        """
        for i, pose_file in enumerate(tqdm(self.pose_files[::self.sample_interval], 
                                           desc="Accumulating frames")):
            try:
                self._process_frame_to_world(pose_file)
            except Exception as e:
                print(f"\nWarning: Failed to process {pose_file.name}: {e}")
                continue
        
        # Voxel Downsampling으로 중복 제거 및 정제
        print(f"\n  Original points: {len(self.global_pcd.points):,}")
        
        if len(self.global_pcd.points) == 0:
            print("  WARNING: No points accumulated! Check your data paths.")
            return
            
        self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=self.voxel_size)
        print(f"  Downsampled points: {len(self.global_pcd.points):,}")
        
        # Statistical outlier removal (노이즈 제거)
        self.global_pcd, _ = self.global_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        print(f"  After outlier removal: {len(self.global_pcd.points):,}")
        
        # [추가] 3DGS 학습 초기화용 PLY 파일 저장
        save_path = self.output_dir / 'accumulated_static.ply'
        o3d.io.write_point_cloud(str(save_path), self.global_pcd)
        print(f"  Saved global point cloud to: {save_path}")
        
    def _backward_reprojection(self):
        """
        Backward Pass: 전역 포인트 클라우드를 각 프레임에 다시 투영하여
        동적 객체가 있던 영역을 정적 배경으로 채움
        """
        for pose_file in tqdm(self.pose_files, desc="Reprojecting frames"):
            try:
                self._render_static_background(pose_file)
            except Exception as e:
                print(f"\nWarning: Failed to render {pose_file.name}: {e}")
                continue
    
    def _process_frame_to_world(self, pose_file):
        """
        개별 프레임을 처리하여 정적 영역의 3D 포인트를 전역 좌표계로 변환
        
        Args:
            pose_file: Pose JSON 파일 경로
        """
        # Pose 메타데이터 로드
        with open(pose_file, 'r') as f:
            meta = json.load(f)
        
        frame_id = pose_file.stem
        
        # 각 카메라에 대해 처리
        for cam_name, cam_info in meta['cameras'].items():
            # 파일 경로 구성
            mask_path = self.masks_dir / f"{frame_id}_{cam_name}.png"
            img_path = self.images_dir / f"{frame_id}_{cam_name}.jpg"
            
            # JPG가 없으면 PNG 시도
            if not img_path.exists():
                img_path = self.images_dir / f"{frame_id}_{cam_name}.png"
            
            # Depth가 있으면 사용, 없으면 스킵 (또는 단안 depth 추정 사용)
            depth_path = self.depths_dir / f"{frame_id}_{cam_name}.png"
            
            if not img_path.exists() or not mask_path.exists():
                continue
            
            # 이미지 로드
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            color = cv2.imread(str(img_path))
            
            if mask is None or color is None:
                continue
            
            # Depth 로드 (있는 경우)
            if depth_path.exists():
                depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            else:
                # Depth가 없으면 pseudo depth 사용 (모든 픽셀을 일정 거리로)
                # 실제 사용 시에는 monocular depth estimation 사용 권장
                depth = np.full(mask.shape, 10000, dtype=np.uint16)  # 10m
            
            if depth is None:
                continue
            
            # 정적 영역 필터링 (mask: 0=동적, 255=정적)
            static_mask = (mask > 200) & (depth > 0)
            
            # 2D -> 3D Backprojection (Camera 좌표계)
            points_3d, colors = self._backproject(
                depth, color, cam_info['intrinsics'], static_mask
            )
            
            if len(points_3d) == 0:
                continue
            
            # Camera -> World Transform
            pose = np.array(cam_info['pose']).reshape(4, 4)
            points_world = (pose[:3, :3] @ points_3d.T).T + pose[:3, 3]
            
            # Global Point Cloud에 추가
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            self.global_pcd += pcd
    
    def _render_static_background(self, pose_file):
        """
        전역 포인트 클라우드를 현재 시점에 다시 투영하여
        구멍이 메워진 이미지 생성
        
        Args:
            pose_file: Pose JSON 파일 경로
        """
        if len(self.global_pcd.points) == 0:
            return
        
        # Pose 메타데이터 로드
        with open(pose_file, 'r') as f:
            meta = json.load(f)
        
        frame_id = pose_file.stem
        
        pts = np.asarray(self.global_pcd.points)
        clrs = np.asarray(self.global_pcd.colors)
        
        # 각 카메라에 대해 렌더링
        for cam_name, cam_info in meta['cameras'].items():
            # 원본 이미지 로드 (블렌딩용)
            img_path = self.images_dir / f"{frame_id}_{cam_name}.jpg"
            if not img_path.exists():
                img_path = self.images_dir / f"{frame_id}_{cam_name}.png"
            
            mask_path = self.masks_dir / f"{frame_id}_{cam_name}.png"
            
            if not img_path.exists() or not mask_path.exists():
                continue
            
            original_img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            if original_img is None or mask is None:
                continue
            
            # 카메라 파라미터
            intrinsics = cam_info['intrinsics']
            K = np.array([
                [intrinsics[0], 0, intrinsics[2]],
                [0, intrinsics[1], intrinsics[3]],
                [0, 0, 1]
            ])
            width = cam_info['width']
            height = cam_info['height']
            pose = np.array(cam_info['pose']).reshape(4, 4)
            
            # World -> Camera 변환
            inv_pose = np.linalg.inv(pose)
            pts_cam = (inv_pose[:3, :3] @ pts.T).T + inv_pose[:3, 3]
            
            # Frustum Culling (카메라 앞쪽만)
            valid_idx = pts_cam[:, 2] > 0.5
            pts_valid = pts_cam[valid_idx]
            clrs_valid = clrs[valid_idx]
            
            if len(pts_valid) == 0:
                continue
            
            # 이미지 평면으로 투영
            pts_2d_homogeneous = (K @ pts_valid.T).T
            pts_2d = pts_2d_homogeneous[:, :2] / pts_2d_homogeneous[:, 2:3]
            
            # Z-buffer를 위한 depth 값
            z_values = pts_valid[:, 2]
            
            # 캔버스 생성 (빈 공간은 검은색)
            warped_img = np.zeros((height, width, 3), dtype=np.uint8)
            z_buffer = np.full((height, width), np.inf, dtype=np.float32)
            
            # 픽셀 좌표 계산
            u = np.round(pts_2d[:, 0]).astype(int)
            v = np.round(pts_2d[:, 1]).astype(int)
            
            # 이미지 범위 내 픽셀만 선택
            valid_uv = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            
            u_valid = u[valid_uv]
            v_valid = v[valid_uv]
            z_valid = z_values[valid_uv]
            clrs_valid_filtered = clrs_valid[valid_uv]
            
            # Z-buffering: 가까운 점만 그리기
            for i in range(len(u_valid)):
                uu, vv, zz = u_valid[i], v_valid[i], z_valid[i]
                if zz < z_buffer[vv, uu]:
                    z_buffer[vv, uu] = zz
                    warped_img[vv, uu] = (clrs_valid_filtered[i] * 255).astype(np.uint8)
            
            # Hole filling: 작은 구멍을 inpainting으로 채움
            warped_gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
            hole_mask = (warped_gray == 0).astype(np.uint8) * 255
            
            # 작은 구멍만 inpainting (큰 영역은 다음 스텝에서 처리)
            kernel = np.ones((3, 3), np.uint8)
            hole_mask_dilated = cv2.dilate(hole_mask, kernel, iterations=1)
            
            warped_img = cv2.inpaint(
                warped_img, hole_mask_dilated, 3, cv2.INPAINT_TELEA
            )
            
            # 원본 이미지와 블렌딩: 정적 영역은 원본 유지, 동적 영역은 warped 사용
            # mask: 255=정적(원본 유지), 0=동적(warped 사용)
            static_mask_norm = (mask / 255.0)[:, :, np.newaxis]
            
            result_img = (
                original_img * static_mask_norm +
                warped_img * (1 - static_mask_norm)
            ).astype(np.uint8)
            
            # 저장: Step 1 결과물
            output_path = self.output_dir / f"{frame_id}_{cam_name}.png"
            cv2.imwrite(str(output_path), result_img)
    
    def _backproject(self, depth, color, intrinsics, mask):
        """
        2D 이미지를 3D 포인트 클라우드로 변환 (Backprojection)
        
        Args:
            depth: Depth map (uint16, mm 단위)
            color: RGB 이미지
            intrinsics: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
            mask: Boolean mask (True인 픽셀만 처리)
        
        Returns:
            points: (N, 3) 3D 좌표
            colors: (N, 3) RGB 색상
        """
        fx, fy, cx, cy = intrinsics[:4]
        
        # Mask에서 유효한 픽셀 좌표 추출
        rows, cols = np.where(mask)
        
        if len(rows) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        
        # Depth 값 (mm -> m)
        z = depth[rows, cols] / 1000.0
        
        # 픽셀 좌표
        u, v = cols, rows
        
        # 3D 좌표 계산
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points = np.stack([x, y, z], axis=1)
        
        # 색상 (BGR -> RGB)
        colors = color[rows, cols][:, ::-1]
        
        return points, colors


def main():
    """CLI 진입점"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Inpainting Step 1: Temporal Accumulation"
    )
    parser.add_argument(
        'data_root',
        type=str,
        help="Path to preprocessing output directory (containing images/, masks/, poses/)"
    )
    parser.add_argument(
        '--voxel_size',
        type=float,
        default=0.05,
        help="Voxel downsampling size in meters (default: 0.05)"
    )
    parser.add_argument(
        '--sample_interval',
        type=int,
        default=5,
        help="Frame sampling interval for forward pass (default: 5)"
    )
    
    args = parser.parse_args()
    
    # 실행
    accumulator = TemporalStaticAccumulator(
        data_root=args.data_root,
        voxel_size=args.voxel_size,
        sample_interval=args.sample_interval
    )
    accumulator.run()


if __name__ == "__main__":
    main()
