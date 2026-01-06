"""
자율주행 AI 개발을 위한 센서 데이터 시각화 유틸리티
- LiDAR 포인트 클라우드 시각화
- 카메라 이미지 시각화
- 멀티모달 데이터 시각화
- 센서 데이터 동기화 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Dict, Union
import torch
from PIL import Image
import cv2


class LiDARVisualizer:
    """LiDAR 포인트 클라우드 시각화 클래스"""
    
    def __init__(self, point_size: float = 1.0):
        """
        Args:
            point_size: 포인트 크기
        """
        self.point_size = point_size
    
    def visualize_point_cloud(self,
                             points: np.ndarray,
                             colors: Optional[np.ndarray] = None,
                             save_path: Optional[str] = None,
                             title: Optional[str] = None,
                             view_angle: Tuple[float, float] = (30, 45)) -> None:
        """
        3D 포인트 클라우드를 시각화합니다.
        
        Args:
            points: (N, 3) 또는 (N, 4) 형태의 포인트 클라우드 [x, y, z, (intensity)]
            colors: (N, 3) 형태의 RGB 색상 (None이면 거리 또는 intensity 기반)
            save_path: 저장 경로
            title: 그래프 제목
            view_angle: (elevation, azimuth) 시야각
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        if points.shape[1] >= 4:
            # intensity를 색상으로 사용
            if colors is None:
                intensity = points[:, 3]
                colors = plt.cm.viridis((intensity - intensity.min()) / 
                                       (intensity.max() - intensity.min() + 1e-8))[:, :3]
        
        if colors is None:
            # 거리 기반 색상
            distances = np.linalg.norm(points[:, :3], axis=1)
            colors = plt.cm.jet((distances - distances.min()) / 
                               (distances.max() - distances.min() + 1e-8))[:, :3]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=colors, s=self.point_size, alpha=0.6)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title or 'LiDAR Point Cloud')
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()
    
    def visualize_bev_point_cloud(self,
                                 points: np.ndarray,
                                 save_path: Optional[str] = None,
                                 title: Optional[str] = None,
                                 x_range: Tuple[float, float] = (-50, 50),
                                 y_range: Tuple[float, float] = (-50, 50)) -> None:
        """
        포인트 클라우드를 BEV (Bird's Eye View)로 시각화합니다.
        
        Args:
            points: (N, 3) 또는 (N, 4) 형태의 포인트 클라우드
            save_path: 저장 경로
            title: 그래프 제목
            x_range: X축 범위
            y_range: Y축 범위
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        # 범위 필터링
        mask = ((points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
                (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]))
        filtered_points = points[mask]
        
        # 높이 또는 intensity를 색상으로 사용
        if points.shape[1] >= 4:
            colors = filtered_points[:, 3]  # intensity
        else:
            colors = filtered_points[:, 2]  # z (height)
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        scatter = ax.scatter(filtered_points[:, 0], filtered_points[:, 1],
                           c=colors, s=self.point_size, alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Height/Intensity')
        
        # Ego vehicle 위치
        ax.plot(0, 0, 'ro', markersize=15, label='Ego Vehicle')
        ax.arrow(0, 0, 0, 3, head_width=1, head_length=0.5, fc='red', ec='red', label='Forward')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_title(title or 'BEV Point Cloud')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()
    
    def visualize_point_cloud_with_boxes(self,
                                        points: np.ndarray,
                                        boxes: List[Dict],
                                        save_path: Optional[str] = None,
                                        title: Optional[str] = None) -> None:
        """
        포인트 클라우드와 3D 박스를 함께 시각화합니다.
        
        Args:
            points: (N, 3) 형태의 포인트 클라우드
            boxes: 박스 리스트 [{'center': [x, y, z], 'size': [w, l, h], 'rotation': yaw, 'class': str}, ...]
            save_path: 저장 경로
            title: 그래프 제목
        """
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 포인트 클라우드
        distances = np.linalg.norm(points[:, :3], axis=1)
        colors = plt.cm.jet((distances - distances.min()) / 
                           (distances.max() - distances.min() + 1e-8))[:, :3]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=colors, s=self.point_size, alpha=0.3)
        
        # 박스 그리기
        box_colors = plt.cm.tab10(np.linspace(0, 1, len(boxes)))
        for i, box in enumerate(boxes):
            center = np.array(box['center'])
            size = np.array(box['size'])  # [w, l, h]
            yaw = box.get('rotation', 0)
            
            # 박스 모서리 계산
            w, l, h = size
            corners = np.array([
                [-l/2, -w/2, -h/2],
                [l/2, -w/2, -h/2],
                [l/2, w/2, -h/2],
                [-l/2, w/2, -h/2],
                [-l/2, -w/2, h/2],
                [l/2, -w/2, h/2],
                [l/2, w/2, h/2],
                [-l/2, w/2, h/2]
            ])
            
            # 회전 적용
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1]
            ])
            corners = corners @ rotation_matrix.T
            corners += center
            
            # 박스 그리기
            color = box_colors[i]
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # 하단
                [4, 5], [5, 6], [6, 7], [7, 4],  # 상단
                [0, 4], [1, 5], [2, 6], [3, 7]   # 수직
            ]
            
            for edge in edges:
                ax.plot3D(*zip(corners[edge[0]], corners[edge[1]]), 
                         color=color, linewidth=2)
            
            # 클래스 라벨
            if 'class' in box:
                ax.text(center[0], center[1], center[2] + h/2 + 1,
                       box['class'], fontsize=10, color=color)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title or 'Point Cloud with 3D Boxes')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()


class CameraVisualizer:
    """카메라 이미지 시각화 클래스"""
    
    def visualize_multi_camera(self,
                               images: Dict[str, Union[np.ndarray, Image.Image]],
                               save_path: Optional[str] = None,
                               title: Optional[str] = None) -> None:
        """
        여러 카메라 이미지를 함께 시각화합니다.
        
        Args:
            images: {'CAM_FRONT': image, 'CAM_FRONT_LEFT': image, ...} 형태의 딕셔너리
            save_path: 저장 경로
            title: 그래프 제목
        """
        num_cams = len(images)
        cols = 3
        rows = (num_cams + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
        axes = axes.flatten() if num_cams > 1 else [axes]
        
        for idx, (cam_name, image) in enumerate(images.items()):
            ax = axes[idx]
            
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            ax.imshow(image)
            ax.set_title(cam_name)
            ax.axis('off')
        
        # 빈 subplot 숨기기
        for idx in range(num_cams, len(axes)):
            axes[idx].axis('off')
        
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()
    
    def visualize_image_with_boxes(self,
                                   image: Union[np.ndarray, Image.Image],
                                   boxes_2d: List[Dict],
                                   save_path: Optional[str] = None,
                                   title: Optional[str] = None) -> None:
        """
        이미지에 2D 박스를 그립니다.
        
        Args:
            image: 원본 이미지
            boxes_2d: 박스 리스트 [{'bbox': [x1, y1, x2, y2], 'score': float, 'class': str}, ...]
            save_path: 저장 경로
            title: 그래프 제목
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, box in enumerate(boxes_2d):
            bbox = box['bbox']
            x1, y1, x2, y2 = bbox
            
            color = colors[i % len(colors)]
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            
            # 라벨
            label = f"{box.get('class', 'obj')}: {box.get('score', 0):.2f}"
            ax.text(x1, y1 - 5, label, color=color, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.set_title(title or 'Image with 2D Boxes')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()


class MultiModalVisualizer:
    """멀티모달 데이터 시각화 클래스"""
    
    def visualize_lidar_camera_fusion(self,
                                     lidar_points: np.ndarray,
                                     camera_images: Dict[str, np.ndarray],
                                     lidar_to_camera: Optional[Dict[str, np.ndarray]] = None,
                                     save_path: Optional[str] = None) -> None:
        """
        LiDAR와 카메라 데이터를 융합하여 시각화합니다.
        
        Args:
            lidar_points: (N, 3) 또는 (N, 4) 형태의 LiDAR 포인트
            camera_images: {'CAM_FRONT': image, ...} 형태의 카메라 이미지
            lidar_to_camera: {'CAM_FRONT': transform_matrix, ...} 변환 행렬
            save_path: 저장 경로
        """
        num_cams = len(camera_images)
        fig, axes = plt.subplots(2, num_cams, figsize=(6 * num_cams, 12))
        
        if num_cams == 1:
            axes = axes.reshape(2, 1)
        
        for idx, (cam_name, image) in enumerate(camera_images.items()):
            # 원본 이미지
            axes[0, idx].imshow(image)
            axes[0, idx].set_title(f'{cam_name} - Original')
            axes[0, idx].axis('off')
            
            # LiDAR 포인트를 카메라에 투영
            if lidar_to_camera and cam_name in lidar_to_camera:
                # 간단한 투영 예시 (실제로는 intrinsic/extrinsic 행렬 사용)
                projected = self._project_lidar_to_camera(
                    lidar_points, lidar_to_camera[cam_name], image.shape[:2]
                )
                
                axes[1, idx].imshow(image)
                if len(projected) > 0:
                    axes[1, idx].scatter(projected[:, 0], projected[:, 1],
                                       c=projected[:, 2], s=1, alpha=0.6, cmap='jet')
                axes[1, idx].set_title(f'{cam_name} - LiDAR Projection')
            else:
                axes[1, idx].imshow(image)
                axes[1, idx].set_title(f'{cam_name} - No Projection')
            
            axes[1, idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()
    
    def _project_lidar_to_camera(self,
                                 points: np.ndarray,
                                 transform: np.ndarray,
                                 image_shape: Tuple[int, int]) -> np.ndarray:
        """
        LiDAR 포인트를 카메라 이미지에 투영합니다.
        (간단한 예시 구현)
        """
        # 실제 구현에서는 intrinsic/extrinsic 행렬을 사용
        # 여기서는 간단한 예시만 제공
        if len(points) == 0:
            return np.array([])
        
        # 변환 적용 (실제로는 더 복잡한 변환 필요)
        points_3d = points[:, :3]
        projected = points_3d @ transform[:3, :3].T + transform[:3, 3]
        
        # 이미지 범위 내 포인트만 필터링
        h, w = image_shape
        mask = ((projected[:, 0] >= 0) & (projected[:, 0] < w) &
                (projected[:, 1] >= 0) & (projected[:, 1] < h) &
                (projected[:, 2] > 0))
        
        return projected[mask]


# 사용 예시 함수
def example_usage():
    """사용 예시"""
    # LiDAR 시각화
    lidar_vis = LiDARVisualizer()
    
    # 가상의 포인트 클라우드 생성
    points = np.random.randn(10000, 4)  # (N, 4) [x, y, z, intensity]
    points[:, :3] *= 10  # 스케일 조정
    
    lidar_vis.visualize_point_cloud(
        points,
        save_path='lidar_3d.png',
        title='3D LiDAR Point Cloud'
    )
    
    lidar_vis.visualize_bev_point_cloud(
        points,
        save_path='lidar_bev.png',
        title='BEV LiDAR Point Cloud'
    )
    
    # 카메라 시각화
    camera_vis = CameraVisualizer()
    
    # 가상의 이미지 생성
    images = {
        'CAM_FRONT': np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8),
        'CAM_FRONT_LEFT': np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8),
        'CAM_FRONT_RIGHT': np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8),
    }
    
    camera_vis.visualize_multi_camera(
        images,
        save_path='multi_camera.png',
        title='Multi-Camera View'
    )


if __name__ == '__main__':
    example_usage()

