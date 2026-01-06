"""
자율주행 AI 개발을 위한 Feature 시각화 유틸리티
- 중간 레이어 feature 시각화
- BEV feature 시각화
- Feature map 시각화
- Attention map 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


class FeatureVisualizer:
    """중간 레이어 feature 시각화를 위한 클래스"""
    
    def __init__(self, colormap='jet', alpha=0.6):
        """
        Args:
            colormap: 시각화에 사용할 colormap ('jet', 'viridis', 'hot', etc.)
            alpha: 오버레이 투명도
        """
        self.colormap = colormap
        self.alpha = alpha
    
    def visualize_feature_map(self, 
                             feature: Union[np.ndarray, torch.Tensor],
                             save_path: Optional[str] = None,
                             title: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Feature map을 시각화합니다.
        
        Args:
            feature: (C, H, W) 또는 (H, W) 형태의 feature map
            save_path: 저장 경로 (None이면 표시만)
            title: 그래프 제목
            figsize: figure 크기
        """
        if isinstance(feature, torch.Tensor):
            feature = feature.detach().cpu().numpy()
        
        # (C, H, W) -> (H, W)로 변환 (평균 또는 첫 번째 채널)
        if len(feature.shape) == 3:
            feature = np.mean(feature, axis=0)
        
        plt.figure(figsize=figsize)
        plt.imshow(feature, cmap=self.colormap)
        plt.colorbar(label='Feature Value')
        if title:
            plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()
    
    def visualize_feature_channels(self,
                                   feature: Union[np.ndarray, torch.Tensor],
                                   num_channels: int = 16,
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (16, 12)) -> None:
        """
        Feature map의 여러 채널을 그리드로 시각화합니다.
        
        Args:
            feature: (C, H, W) 형태의 feature map
            num_channels: 시각화할 채널 수
            save_path: 저장 경로
            figsize: figure 크기
        """
        if isinstance(feature, torch.Tensor):
            feature = feature.detach().cpu().numpy()
        
        C, H, W = feature.shape
        num_channels = min(num_channels, C)
        
        # 그리드 크기 계산
        cols = 4
        rows = (num_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if num_channels > 1 else [axes]
        
        for i in range(num_channels):
            ax = axes[i]
            im = ax.imshow(feature[i], cmap=self.colormap)
            ax.set_title(f'Channel {i}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        # 빈 subplot 숨기기
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()
    
    def overlay_feature_on_image(self,
                                 image: Union[np.ndarray, Image.Image],
                                 feature: Union[np.ndarray, torch.Tensor],
                                 save_path: Optional[str] = None,
                                 title: Optional[str] = None) -> np.ndarray:
        """
        Feature map을 원본 이미지에 오버레이합니다.
        
        Args:
            image: 원본 이미지 (H, W, 3) 또는 PIL Image
            feature: (H, W) 또는 (C, H, W) 형태의 feature map
            save_path: 저장 경로
            title: 그래프 제목
            
        Returns:
            오버레이된 이미지
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if isinstance(feature, torch.Tensor):
            feature = feature.detach().cpu().numpy()
        
        # Feature 정규화 및 리사이즈
        if len(feature.shape) == 3:
            feature = np.mean(feature, axis=0)
        
        # 이미지 크기에 맞게 리사이즈
        if feature.shape != image.shape[:2]:
            feature = cv2.resize(feature, (image.shape[1], image.shape[0]))
        
        # Feature 정규화 [0, 1]
        feature = (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)
        
        # Colormap 적용
        feature_colored = cm.get_cmap(self.colormap)(feature)[:, :, :3]
        feature_colored = (feature_colored * 255).astype(np.uint8)
        
        # 오버레이
        overlay = cv2.addWeighted(image, 1 - self.alpha, feature_colored, self.alpha, 0)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(overlay)
        if title:
            plt.title(title)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()
        
        return overlay


class BEVFeatureVisualizer:
    """BEV (Bird's Eye View) Feature 시각화 클래스"""
    
    def __init__(self, 
                 x_range: Tuple[float, float] = (-50, 50),
                 y_range: Tuple[float, float] = (-50, 50),
                 resolution: float = 0.5):
        """
        Args:
            x_range: X축 범위 (미터)
            y_range: Y축 범위 (미터)
            resolution: 해상도 (미터/픽셀)
        """
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        
        # BEV 그리드 크기 계산
        self.grid_h = int((y_range[1] - y_range[0]) / resolution)
        self.grid_w = int((x_range[1] - x_range[0]) / resolution)
    
    def visualize_bev_feature(self,
                              bev_feature: Union[np.ndarray, torch.Tensor],
                              save_path: Optional[str] = None,
                              title: Optional[str] = None,
                              show_ego: bool = True) -> None:
        """
        BEV feature를 시각화합니다.
        
        Args:
            bev_feature: (H, W) 또는 (C, H, W) 형태의 BEV feature
            save_path: 저장 경로
            title: 그래프 제목
            show_ego: Ego vehicle 위치 표시 여부
        """
        if isinstance(bev_feature, torch.Tensor):
            bev_feature = bev_feature.detach().cpu().numpy()
        
        if len(bev_feature.shape) == 3:
            bev_feature = np.mean(bev_feature, axis=0)
        
        # BEV feature를 그리드 크기에 맞게 리사이즈
        if bev_feature.shape != (self.grid_h, self.grid_w):
            bev_feature = cv2.resize(bev_feature, (self.grid_w, self.grid_h))
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # BEV feature 시각화
        im = ax.imshow(bev_feature, 
                      extent=[self.x_range[0], self.x_range[1], 
                             self.y_range[0], self.y_range[1]],
                      cmap='jet', origin='lower')
        plt.colorbar(im, ax=ax, label='Feature Value')
        
        # Ego vehicle 위치 표시
        if show_ego:
            ax.plot(0, 0, 'ro', markersize=15, label='Ego Vehicle')
            ax.arrow(0, 0, 0, 3, head_width=1, head_length=0.5, 
                    fc='red', ec='red', label='Forward')
            ax.legend()
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(title or 'BEV Feature Map')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()
    
    def visualize_bev_with_detections(self,
                                     bev_feature: Union[np.ndarray, torch.Tensor],
                                     detections: List[Dict],
                                     save_path: Optional[str] = None,
                                     title: Optional[str] = None) -> None:
        """
        BEV feature와 detection 결과를 함께 시각화합니다.
        
        Args:
            bev_feature: (H, W) 또는 (C, H, W) 형태의 BEV feature
            detections: Detection 결과 리스트 [{'bbox': [x, y, w, h, yaw], 'score': float, 'class': str}, ...]
            save_path: 저장 경로
            title: 그래프 제목
        """
        if isinstance(bev_feature, torch.Tensor):
            bev_feature = bev_feature.detach().cpu().numpy()
        
        if len(bev_feature.shape) == 3:
            bev_feature = np.mean(bev_feature, axis=0)
        
        if bev_feature.shape != (self.grid_h, self.grid_w):
            bev_feature = cv2.resize(bev_feature, (self.grid_w, self.grid_h))
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # BEV feature 시각화
        im = ax.imshow(bev_feature,
                      extent=[self.x_range[0], self.x_range[1],
                             self.y_range[0], self.y_range[1]],
                      cmap='jet', origin='lower', alpha=0.5)
        plt.colorbar(im, ax=ax, label='Feature Value')
        
        # Detection 박스 그리기
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, det in enumerate(detections):
            bbox = det['bbox']  # [x, y, w, h, yaw]
            x, y, w, h, yaw = bbox
            
            # 박스 모서리 계산
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            
            corners = np.array([
                [-w/2, -h/2],
                [w/2, -h/2],
                [w/2, h/2],
                [-w/2, h/2]
            ])
            
            # 회전 적용
            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw]
            ])
            corners = corners @ rotation_matrix.T
            corners[:, 0] += x
            corners[:, 1] += y
            
            # 박스 그리기
            color = colors[i % len(colors)]
            ax.plot(corners[[0, 1, 2, 3, 0], 0],
                   corners[[0, 1, 2, 3, 0], 1],
                   color=color, linewidth=2, label=f"{det.get('class', 'obj')} ({det.get('score', 0):.2f})")
        
        # Ego vehicle
        ax.plot(0, 0, 'ro', markersize=15, label='Ego Vehicle')
        ax.arrow(0, 0, 0, 3, head_width=1, head_length=0.5, fc='red', ec='red')
        
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title(title or 'BEV Feature with Detections')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()


class AttentionVisualizer:
    """Attention map 시각화 클래스"""
    
    def visualize_attention_map(self,
                               attention: Union[np.ndarray, torch.Tensor],
                               image: Optional[Union[np.ndarray, Image.Image]] = None,
                               save_path: Optional[str] = None,
                               title: Optional[str] = None) -> None:
        """
        Attention map을 시각화합니다.
        
        Args:
            attention: (H, W) 또는 (N, N) 형태의 attention map
            image: 원본 이미지 (오버레이용)
            save_path: 저장 경로
            title: 그래프 제목
        """
        if isinstance(attention, torch.Tensor):
            attention = attention.detach().cpu().numpy()
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Attention 정규화
        if len(attention.shape) == 2:
            attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
        
        if image is not None:
            # 이미지에 오버레이
            if attention.shape != image.shape[:2]:
                attention = cv2.resize(attention, (image.shape[1], image.shape[0]))
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(attention, cmap='jet')
            axes[1].set_title('Attention Map')
            axes[1].axis('off')
            
            # 오버레이
            attention_colored = cm.get_cmap('jet')(attention)[:, :, :3]
            attention_colored = (attention_colored * 255).astype(np.uint8)
            overlay = cv2.addWeighted(image, 0.5, attention_colored, 0.5, 0)
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(attention, cmap='jet')
            plt.colorbar(im, ax=ax)
            if title:
                ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()
    
    def visualize_multi_head_attention(self,
                                      attention: Union[np.ndarray, torch.Tensor],
                                      num_heads: Optional[int] = None,
                                      save_path: Optional[str] = None) -> None:
        """
        Multi-head attention을 시각화합니다.
        
        Args:
            attention: (num_heads, H, W) 또는 (num_heads, N, N) 형태의 attention
            num_heads: Head 수 (None이면 자동 감지)
            save_path: 저장 경로
        """
        if isinstance(attention, torch.Tensor):
            attention = attention.detach().cpu().numpy()
        
        if len(attention.shape) == 3:
            num_heads = attention.shape[0]
        else:
            raise ValueError("Attention shape should be (num_heads, H, W) or (num_heads, N, N)")
        
        cols = 4
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten() if num_heads > 1 else [axes]
        
        for i in range(num_heads):
            ax = axes[i]
            attn = attention[i]
            attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
            im = ax.imshow(attn, cmap='jet')
            ax.set_title(f'Head {i}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        # 빈 subplot 숨기기
        for i in range(num_heads, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()


# 사용 예시 함수
def example_usage():
    """사용 예시"""
    # Feature map 시각화
    feature_vis = FeatureVisualizer()
    
    # 가상의 feature map 생성
    feature_map = np.random.randn(256, 64, 64)  # (C, H, W)
    
    # 단일 feature map 시각화
    feature_vis.visualize_feature_map(
        np.mean(feature_map, axis=0),
        save_path='feature_map.png',
        title='Feature Map Visualization'
    )
    
    # 여러 채널 시각화
    feature_vis.visualize_feature_channels(
        feature_map,
        num_channels=16,
        save_path='feature_channels.png'
    )
    
    # BEV feature 시각화
    bev_vis = BEVFeatureVisualizer(x_range=(-50, 50), y_range=(-50, 50))
    bev_feature = np.random.randn(200, 200)  # (H, W)
    bev_vis.visualize_bev_feature(
        bev_feature,
        save_path='bev_feature.png',
        title='BEV Feature Map'
    )
    
    # Attention 시각화
    attn_vis = AttentionVisualizer()
    attention = np.random.rand(64, 64)
    attn_vis.visualize_attention_map(
        attention,
        save_path='attention_map.png'
    )


if __name__ == '__main__':
    example_usage()

