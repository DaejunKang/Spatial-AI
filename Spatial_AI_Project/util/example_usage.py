# -*- coding: utf-8 -*-
"""
자율주행 AI 유틸리티 사용 예시 스크립트
"""

import numpy as np
import sys
import os

# util 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import (
    FeatureVisualizer,
    BEVFeatureVisualizer,
    AttentionVisualizer,
    LiDARVisualizer,
    CameraVisualizer,
    PerformanceAnalyzer,
    DetectionAnalyzer
)


def example_feature_visualization():
    """Feature 시각화 예시"""
    print("=" * 50)
    print("Feature 시각화 예시")
    print("=" * 50)
    
    # Feature map 시각화
    vis = FeatureVisualizer(colormap='jet', alpha=0.6)
    
    # 가상의 feature map 생성 (예: CNN 중간 레이어 출력)
    feature_map = np.random.randn(256, 64, 64)  # (C, H, W)
    
    # 단일 feature map 시각화 (채널 평균)
    print("1. Feature map 시각화 (채널 평균)")
    vis.visualize_feature_map(
        np.mean(feature_map, axis=0),
        save_path='example_feature_map.png',
        title='CNN Feature Map (Channel Average)'
    )
    
    # 여러 채널 시각화
    print("2. 여러 채널 시각화")
    vis.visualize_feature_channels(
        feature_map,
        num_channels=16,
        save_path='example_feature_channels.png'
    )
    
    # 이미지에 feature 오버레이
    print("3. 이미지에 feature 오버레이")
    image = np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8)
    vis.overlay_feature_on_image(
        image,
        np.mean(feature_map, axis=0),
        save_path='example_feature_overlay.png',
        title='Feature Overlay on Image'
    )


def example_bev_visualization():
    """BEV Feature 시각화 예시"""
    print("=" * 50)
    print("BEV Feature 시각화 예시")
    print("=" * 50)
    
    bev_vis = BEVFeatureVisualizer(x_range=(-50, 50), y_range=(-50, 50), resolution=0.5)
    
    # 가상의 BEV feature 생성
    bev_feature = np.random.randn(200, 200)  # (H, W)
    bev_feature = (bev_feature - bev_feature.min()) / (bev_feature.max() - bev_feature.min())
    
    print("1. BEV feature 시각화")
    bev_vis.visualize_bev_feature(
        bev_feature,
        save_path='example_bev_feature.png',
        title='BEV Feature Map',
        show_ego=True
    )
    
    # Detection 결과와 함께 시각화
    print("2. BEV feature + Detection 결과")
    detections = [
        {
            'bbox': [10, 5, 4, 2, 0.1],  # [x, y, w, h, yaw]
            'score': 0.95,
            'class': 'car'
        },
        {
            'bbox': [-15, 8, 1.5, 0.5, 1.5],
            'score': 0.87,
            'class': 'pedestrian'
        },
        {
            'bbox': [20, -10, 6, 2.5, -0.3],
            'score': 0.92,
            'class': 'truck'
        }
    ]
    
    bev_vis.visualize_bev_with_detections(
        bev_feature,
        detections,
        save_path='example_bev_with_detections.png',
        title='BEV Feature with Detections'
    )


def example_attention_visualization():
    """Attention 시각화 예시"""
    print("=" * 50)
    print("Attention 시각화 예시")
    print("=" * 50)
    
    attn_vis = AttentionVisualizer()
    
    # 가상의 attention map 생성
    attention = np.random.rand(64, 64)
    attention = attention / attention.sum()  # 정규화
    
    print("1. Attention map 시각화")
    attn_vis.visualize_attention_map(
        attention,
        save_path='example_attention_map.png'
    )
    
    # 이미지와 함께 시각화
    print("2. 이미지에 attention 오버레이")
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    attn_vis.visualize_attention_map(
        attention,
        image=image,
        save_path='example_attention_overlay.png',
        title='Attention Map Overlay'
    )
    
    # Multi-head attention 시각화
    print("3. Multi-head attention 시각화")
    multi_head_attention = np.random.rand(8, 64, 64)  # (num_heads, H, W)
    for i in range(8):
        multi_head_attention[i] = multi_head_attention[i] / multi_head_attention[i].sum()
    
    attn_vis.visualize_multi_head_attention(
        multi_head_attention,
        num_heads=8,
        save_path='example_multi_head_attention.png'
    )


def example_lidar_visualization():
    """LiDAR 시각화 예시"""
    print("=" * 50)
    print("LiDAR 시각화 예시")
    print("=" * 50)
    
    lidar_vis = LiDARVisualizer(point_size=1.0)
    
    # 가상의 LiDAR 포인트 클라우드 생성
    num_points = 10000
    points = np.random.randn(num_points, 4)
    points[:, 0] = points[:, 0] * 20  # x: -20 ~ 20m
    points[:, 1] = points[:, 1] * 20  # y: -20 ~ 20m
    points[:, 2] = np.abs(points[:, 2] * 2)  # z: 0 ~ 2m (지면 위)
    points[:, 3] = np.random.rand(num_points)  # intensity
    
    print("1. 3D 포인트 클라우드 시각화")
    lidar_vis.visualize_point_cloud(
        points,
        save_path='example_lidar_3d.png',
        title='3D LiDAR Point Cloud',
        view_angle=(30, 45)
    )
    
    print("2. BEV 포인트 클라우드 시각화")
    lidar_vis.visualize_bev_point_cloud(
        points,
        save_path='example_lidar_bev.png',
        title='BEV LiDAR Point Cloud',
        x_range=(-30, 30),
        y_range=(-30, 30)
    )
    
    # 3D 박스와 함께 시각화
    print("3. 포인트 클라우드 + 3D 박스")
    boxes = [
        {
            'center': [10, 5, 1],
            'size': [4, 2, 1.5],  # [w, l, h]
            'rotation': 0.1,
            'class': 'car'
        },
        {
            'center': [-8, 12, 0.8],
            'size': [1.5, 0.5, 1.7],
            'rotation': 1.5,
            'class': 'pedestrian'
        }
    ]
    
    lidar_vis.visualize_point_cloud_with_boxes(
        points,
        boxes,
        save_path='example_lidar_with_boxes.png',
        title='Point Cloud with 3D Boxes'
    )


def example_camera_visualization():
    """카메라 시각화 예시"""
    print("=" * 50)
    print("카메라 시각화 예시")
    print("=" * 50)
    
    camera_vis = CameraVisualizer()
    
    # 가상의 카메라 이미지 생성
    images = {
        'CAM_FRONT': np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8),
        'CAM_FRONT_LEFT': np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8),
        'CAM_FRONT_RIGHT': np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8),
        'CAM_BACK': np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8),
        'CAM_BACK_LEFT': np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8),
        'CAM_BACK_RIGHT': np.random.randint(0, 255, (800, 1200, 3), dtype=np.uint8),
    }
    
    print("1. 멀티 카메라 시각화")
    camera_vis.visualize_multi_camera(
        images,
        save_path='example_multi_camera.png',
        title='Multi-Camera View'
    )
    
    # 2D 박스와 함께 시각화
    print("2. 이미지 + 2D Detection 박스")
    boxes_2d = [
        {'bbox': [200, 150, 400, 350], 'score': 0.95, 'class': 'car'},
        {'bbox': [600, 200, 750, 450], 'score': 0.87, 'class': 'pedestrian'},
        {'bbox': [800, 100, 1100, 400], 'score': 0.92, 'class': 'truck'},
    ]
    
    camera_vis.visualize_image_with_boxes(
        images['CAM_FRONT'],
        boxes_2d,
        save_path='example_camera_with_boxes.png',
        title='Camera Image with 2D Detections'
    )


def example_model_analysis():
    """모델 분석 예시"""
    print("=" * 50)
    print("모델 분석 예시")
    print("=" * 50)
    
    analyzer = PerformanceAnalyzer()
    
    # 학습 곡선 시각화
    print("1. 학습 곡선 시각화")
    logs = {
        'loss': [2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.35],
        'accuracy': [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95],
        'val_loss': [2.6, 2.1, 1.6, 1.3, 1.1, 0.9, 0.7, 0.6, 0.5, 0.45],
        'val_accuracy': [0.48, 0.58, 0.68, 0.73, 0.78, 0.83, 0.88, 0.90, 0.92, 0.93]
    }
    
    analyzer.plot_training_curves(
        logs,
        save_path='example_training_curves.png'
    )
    
    # Detection 분석
    print("2. Detection 결과 분석")
    det_analyzer = DetectionAnalyzer()
    
    predictions = [
        {'bbox': [10, 10, 4, 2, 0], 'score': 0.9, 'class': 'car'},
        {'bbox': [20, 20, 3, 1.5, 0], 'score': 0.8, 'class': 'pedestrian'},
        {'bbox': [15, 15, 5, 2.5, 0.2], 'score': 0.75, 'class': 'truck'},
    ]
    
    ground_truths = [
        {'bbox': [10.5, 10.2, 4.1, 2.1, 0], 'class': 'car'},
        {'bbox': [20.3, 20.1, 3.2, 1.6, 0], 'class': 'pedestrian'},
        {'bbox': [15.2, 15.1, 5.1, 2.6, 0.2], 'class': 'truck'},
    ]
    
    results = det_analyzer.analyze_detection_results(
        predictions,
        ground_truths,
        class_names=['car', 'pedestrian', 'truck'],
        iou_threshold=0.5,
        save_path='example_detection_metrics.png'
    )
    
    print("\nDetection 결과:")
    for class_name, metrics in results.items():
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall: {metrics['recall']:.3f}")
        print(f"    F1 Score: {metrics['f1']:.3f}")
    
    # 오류 분석
    print("\n3. 오류 분석")
    det_analyzer.plot_error_analysis(
        predictions,
        ground_truths,
        save_path='example_error_analysis.png'
    )


def main():
    """모든 예시 실행"""
    print("\n" + "=" * 50)
    print("자율주행 AI 유틸리티 사용 예시")
    print("=" * 50 + "\n")
    
    try:
        example_feature_visualization()
        print("\n")
        
        example_bev_visualization()
        print("\n")
        
        example_attention_visualization()
        print("\n")
        
        example_lidar_visualization()
        print("\n")
        
        example_camera_visualization()
        print("\n")
        
        example_model_analysis()
        print("\n")
        
        print("=" * 50)
        print("모든 예시가 완료되었습니다!")
        print("생성된 이미지 파일들을 확인하세요.")
        print("=" * 50)
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

