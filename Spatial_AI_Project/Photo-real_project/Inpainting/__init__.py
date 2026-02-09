"""
Inpainting module for Photo-real_project

Multi-stage inpainting pipeline:
1. Temporal Accumulation: 시계열 정보를 활용한 정적 배경 누적
2. Geometric Guide: RANSAC 기반 기하학적 depth 추정
3. Final Inpainting: Multi-view consistent 생성형 인페인팅

Style LoRA Training Pipeline (lora/ 하위 패키지):
- lora.train_style_lora: Waymo/KITTI 데이터셋으로 SD v1.5 스타일 LoRA 학습
- lora.lora_inference: 학습된 LoRA로 이미지 생성 & 품질 평가
- lora.training_dataset_builder: 학습 데이터셋 빌더
- lora.lora_ui: Gradio 기반 통합 사용자 인터페이스

Usage:
    from Photo-real_project.Inpainting import (
        TemporalStaticAccumulator,
        GeometricGuideGenerator,
        FinalInpainter,
        run_generative_inpainting,
        StyleLoRADataset,
        StyleLoRATrainer,
        train_style_lora,
        LoRAInference,
        LoRAQualityEvaluator,
        TrainingDatasetBuilder,
    )
"""

__version__ = "2.1.0"
__author__ = "Photo-real_project Team"

# ---------------------------------------------------------------------------
# Inpainting Pipeline (현재 디렉토리)
# ---------------------------------------------------------------------------
try:
    from .step1_temporal_accumulation import TemporalStaticAccumulator
except ImportError:
    TemporalStaticAccumulator = None

try:
    from .step2_geometric_guide import GeometricGuideGenerator
except ImportError:
    GeometricGuideGenerator = None

try:
    from .step3_final_inpainting import GenerativeInpainter, run_step3
    # Backward-compatible aliases
    FinalInpainter = GenerativeInpainter
    run_generative_inpainting = run_step3
except ImportError:
    GenerativeInpainter = None
    FinalInpainter = None
    run_generative_inpainting = None
    run_step3 = None

# ---------------------------------------------------------------------------
# Style LoRA Training Pipeline (lora/ 하위 패키지)
# ---------------------------------------------------------------------------
try:
    from .lora.training_dataset_builder import TrainingDatasetBuilder
except ImportError:
    TrainingDatasetBuilder = None

try:
    from .lora.train_style_lora import (
        StyleLoRADataset,
        StyleLoRATrainer,
        train_style_lora,
    )
except ImportError:
    StyleLoRADataset = None
    StyleLoRATrainer = None
    train_style_lora = None

try:
    from .lora.lora_inference import LoRAInference, LoRAQualityEvaluator
except ImportError:
    LoRAInference = None
    LoRAQualityEvaluator = None

__all__ = [
    # Inpainting Pipeline
    "TemporalStaticAccumulator",
    "GeometricGuideGenerator",
    "GenerativeInpainter",
    "FinalInpainter",  # backward-compatible alias
    "run_step3",
    "run_generative_inpainting",  # backward-compatible alias
    # Style LoRA Training Pipeline (lora/)
    "TrainingDatasetBuilder",
    "StyleLoRADataset",
    "StyleLoRATrainer",
    "train_style_lora",
    "LoRAInference",
    "LoRAQualityEvaluator",
]
