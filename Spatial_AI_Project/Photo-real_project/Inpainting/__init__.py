"""
Inpainting module for Photo-real_project

Multi-stage inpainting pipeline:
1. Temporal Accumulation: 시계열 정보를 활용한 정적 배경 누적
2. Geometric Guide: RANSAC 기반 기하학적 depth 추정
3. Final Inpainting: Multi-view consistent 생성형 인페인팅

Usage:
    from Photo-real_project.Inpainting import (
        TemporalStaticAccumulator,
        GeometricGuideGenerator,
        FinalInpainter,
        run_generative_inpainting
    )
"""

__version__ = "1.0.0"
__author__ = "Photo-real_project Team"

# Import main classes
try:
    from .step1_temporal_accumulation import TemporalStaticAccumulator
except ImportError:
    TemporalStaticAccumulator = None

try:
    from .step2_geometric_guide import GeometricGuideGenerator
except ImportError:
    GeometricGuideGenerator = None

try:
    from .step3_final_inpainting import FinalInpainter, run_generative_inpainting
except ImportError:
    FinalInpainter = None
    run_generative_inpainting = None

__all__ = [
    "TemporalStaticAccumulator",
    "GeometricGuideGenerator",
    "FinalInpainter",
    "run_generative_inpainting",
]
