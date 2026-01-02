"""
mmdet3d 호환성 모듈 - mmdet3d 의존성 없이 핵심 기능 구현
"""
import torch
import torch.nn as nn

# ==================== mmdet3d.core.bbox.coders ====================
def build_bbox_coder(cfg, default_args=None):
    """Bbox coder 빌드 함수 (mmdet3d.core.bbox.coders.build_bbox_coder 호환)
    
    Note: 실제 구현은 복잡하므로, mmdet3d가 있으면 사용하고 없으면 NotImplementedError
    """
    try:
        from mmdet3d.core.bbox.coders import build_bbox_coder as _build
        return _build(cfg, default_args)
    except ImportError:
        raise NotImplementedError("build_bbox_coder requires mmdet3d.core.bbox.coders")


# ==================== mmdet3d.core ====================
def bbox3d2result(bboxes, scores, labels):
    """Bbox 3D 결과 변환 함수 (mmdet3d.core.bbox3d2result 호환)
    
    Note: 실제 구현은 복잡하므로, mmdet3d가 있으면 사용하고 없으면 NotImplementedError
    """
    try:
        from mmdet3d.core import bbox3d2result as _func
        return _func(bboxes, scores, labels)
    except ImportError:
        raise NotImplementedError("bbox3d2result requires mmdet3d.core")


# ==================== mmdet3d.models.builder ====================
def build_head(cfg, default_args=None):
    """Head 빌드 함수 (mmdet3d.models.builder.build_head 호환)
    
    Note: 실제 구현은 복잡하므로, mmdet3d가 있으면 사용하고 없으면 NotImplementedError
    """
    try:
        from mmdet3d.models.builder import build_head as _build
        return _build(cfg, default_args)
    except ImportError:
        raise NotImplementedError("build_head requires mmdet3d.models.builder")


# ==================== mmdet3d.models.detectors ====================
# MVXTwoStageDetector는 복잡한 base class이므로, mmdet3d가 있으면 사용하고 없으면 stub 제공
try:
    from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
except ImportError:
    class MVXTwoStageDetector(nn.Module):
        """MVXTwoStageDetector stub (mmdet3d.models.detectors.mvx_two_stage.MVXTwoStageDetector 호환)
        
        Note: 실제 구현은 mmdet3d 필요
        """
        def __init__(self, *args, **kwargs):
            super(MVXTwoStageDetector, self).__init__()
            raise NotImplementedError("MVXTwoStageDetector requires mmdet3d.models.detectors.mvx_two_stage")


# ==================== mmdet3d.models.dense_heads ====================
# FreeAnchor3DHead는 복잡한 head class이므로, mmdet3d가 있으면 사용하고 없으면 stub 제공
try:
    from mmdet3d.models.dense_heads.free_anchor3d_head import FreeAnchor3DHead
except ImportError:
    class FreeAnchor3DHead(nn.Module):
        """FreeAnchor3DHead stub (mmdet3d.models.dense_heads.free_anchor3d_head.FreeAnchor3DHead 호환)
        
        Note: 실제 구현은 mmdet3d 필요
        """
        def __init__(self, *args, **kwargs):
            super(FreeAnchor3DHead, self).__init__()
            raise NotImplementedError("FreeAnchor3DHead requires mmdet3d.models.dense_heads.free_anchor3d_head")

