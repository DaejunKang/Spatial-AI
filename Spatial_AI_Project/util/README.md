# 자율주행 AI 개발 유틸리티

이 디렉토리는 자율주행 AI 개발에 활용할 수 있는 다양한 유틸리티 코드를 포함합니다.

## 모듈 구성

### 1. `feature_visualization.py` - Feature 시각화
중간 레이어 feature, BEV feature, Attention map 등을 시각화하는 유틸리티입니다.

**주요 클래스:**
- `FeatureVisualizer`: 일반적인 feature map 시각화
- `BEVFeatureVisualizer`: BEV (Bird's Eye View) feature 시각화
- `AttentionVisualizer`: Attention map 시각화

**사용 예시:**
```python
from util import FeatureVisualizer, BEVFeatureVisualizer

# Feature map 시각화
vis = FeatureVisualizer()
feature_map = model.get_intermediate_feature()  # (C, H, W)
vis.visualize_feature_map(feature_map, save_path='feature.png')

# BEV feature 시각화
bev_vis = BEVFeatureVisualizer(x_range=(-50, 50), y_range=(-50, 50))
bev_feature = model.get_bev_feature()  # (H, W)
bev_vis.visualize_bev_feature(bev_feature, save_path='bev.png')
```

### 2. `sensor_visualization.py` - 센서 데이터 시각화
LiDAR, 카메라 등 센서 데이터를 시각화하는 유틸리티입니다.

**주요 클래스:**
- `LiDARVisualizer`: LiDAR 포인트 클라우드 시각화
- `CameraVisualizer`: 카메라 이미지 시각화
- `MultiModalVisualizer`: 멀티모달 데이터 융합 시각화

**사용 예시:**
```python
from util import LiDARVisualizer, CameraVisualizer

# LiDAR 시각화
lidar_vis = LiDARVisualizer()
points = load_lidar_data()  # (N, 4) [x, y, z, intensity]
lidar_vis.visualize_point_cloud(points, save_path='lidar_3d.png')
lidar_vis.visualize_bev_point_cloud(points, save_path='lidar_bev.png')

# 카메라 시각화
camera_vis = CameraVisualizer()
images = {
    'CAM_FRONT': load_camera_image('front.jpg'),
    'CAM_FRONT_LEFT': load_camera_image('front_left.jpg'),
}
camera_vis.visualize_multi_camera(images, save_path='multi_camera.png')
```

### 3. `model_analysis.py` - 모델 분석
모델 성능 분석, 학습 곡선 시각화, Detection 결과 분석 등을 제공합니다.

**주요 클래스:**
- `PerformanceAnalyzer`: 모델 성능 분석
- `DetectionAnalyzer`: Detection 결과 분석

**사용 예시:**
```python
from util import PerformanceAnalyzer, DetectionAnalyzer

# 학습 곡선 시각화
analyzer = PerformanceAnalyzer()
logs = {
    'loss': [2.5, 2.0, 1.5, 1.2, 1.0],
    'accuracy': [0.5, 0.6, 0.7, 0.75, 0.8],
}
analyzer.plot_training_curves(logs, save_path='curves.png')

# Detection 분석
det_analyzer = DetectionAnalyzer()
results = det_analyzer.analyze_detection_results(
    predictions, ground_truths, 
    class_names=['car', 'pedestrian', 'truck'],
    save_path='detection_metrics.png'
)
```

## 주요 기능

### Feature 시각화
- **중간 레이어 feature 시각화**: 모델의 중간 레이어에서 추출한 feature map을 시각화
- **BEV feature 시각화**: Bird's Eye View feature를 지도 형태로 시각화
- **Feature 채널 시각화**: 여러 채널을 그리드로 시각화
- **이미지 오버레이**: Feature map을 원본 이미지에 오버레이
- **Attention map 시각화**: Attention 메커니즘의 가중치를 시각화

### 센서 데이터 시각화
- **3D 포인트 클라우드**: LiDAR 데이터를 3D로 시각화
- **BEV 포인트 클라우드**: LiDAR 데이터를 위에서 본 형태로 시각화
- **3D 박스 시각화**: Detection 결과를 3D 박스로 표시
- **멀티 카메라 시각화**: 여러 카메라 이미지를 함께 표시
- **센서 융합 시각화**: LiDAR와 카메라 데이터를 함께 시각화

### 모델 분석
- **학습 곡선**: Loss, Accuracy 등의 학습 과정 시각화
- **Confusion Matrix**: 분류 성능 분석
- **Precision-Recall 곡선**: Detection 성능 분석
- **오류 분석**: 예측 오류의 분포 분석

## 의존성

```bash
pip install numpy matplotlib torch pillow opencv-python scikit-learn pandas
```

## 사용 팁

1. **Feature 시각화**: 모델의 중간 레이어에서 feature를 추출한 후 시각화하여 모델이 무엇을 학습하는지 확인
2. **BEV 시각화**: 자율주행에서 중요한 BEV representation을 시각화하여 공간적 이해도 향상
3. **센서 융합**: 여러 센서 데이터를 함께 시각화하여 데이터 품질 확인
4. **성능 분석**: 학습 과정과 결과를 분석하여 모델 개선 방향 도출

## 확장 가능성

이 유틸리티들은 기본적인 기능을 제공하며, 프로젝트의 특성에 맞게 확장할 수 있습니다:
- 특정 모델 아키텍처에 맞는 feature 추출 함수 추가
- 프로젝트 특화 시각화 스타일 적용
- 추가 분석 메트릭 구현

