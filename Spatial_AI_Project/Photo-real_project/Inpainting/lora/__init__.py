"""
Style LoRA Training Pipeline for Autonomous Driving Scenes

Waymo/KITTI 데이터셋의 이미지를 사용하여 Stable Diffusion v1.5의
스타일(도로 질감, 색감)을 학습시키는 모듈 패키지입니다.

모듈 구성:
    - train_style_lora: LoRA 학습 (StyleLoRADataset + StyleLoRATrainer)
    - lora_inference: 추론 & 품질 평가 (LoRAInference + LoRAQualityEvaluator)
    - training_dataset_builder: 학습 데이터셋 빌더 (TrainingDatasetBuilder)
    - lora_ui: Gradio 기반 통합 UI

Usage:
    # 패키지 레벨 import
    from Inpainting.lora import (
        StyleLoRADataset,
        StyleLoRATrainer,
        train_style_lora,
        LoRAInference,
        LoRAQualityEvaluator,
        TrainingDatasetBuilder,
    )

    # Quick-start
    from Inpainting.lora import train_style_lora
    result = train_style_lora(data_root="/path/to/data", output_dir="./lora_output")
"""

__version__ = "1.0.0"
__author__ = "Photo-real_project Team"

# Training
try:
    from .train_style_lora import (
        StyleLoRADataset,
        StyleLoRATrainer,
        train_style_lora,
    )
except ImportError:
    StyleLoRADataset = None
    StyleLoRATrainer = None
    train_style_lora = None

# Inference & Evaluation
try:
    from .lora_inference import LoRAInference, LoRAQualityEvaluator
except ImportError:
    LoRAInference = None
    LoRAQualityEvaluator = None

# Dataset Builder
try:
    from .training_dataset_builder import TrainingDatasetBuilder
except ImportError:
    TrainingDatasetBuilder = None

__all__ = [
    "StyleLoRADataset",
    "StyleLoRATrainer",
    "train_style_lora",
    "LoRAInference",
    "LoRAQualityEvaluator",
    "TrainingDatasetBuilder",
]
