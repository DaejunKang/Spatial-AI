"""
Base bbox coder classes - mmdet의 BaseBBoxCoder를 대체
"""
import torch


class BaseBBoxCoder:
    """Base bbox coder that encodes/decodes the bbox coordinates."""
    
    def encode(self, bboxes, gt_bboxes):
        """Encode bboxes.
        
        Args:
            bboxes (Tensor): Source bboxes, e.g., object proposals.
            gt_bboxes (Tensor): Target of the transformation, e.g., ground-truth boxes.
        
        Returns:
            Tensor: Encoded bboxes.
        """
        raise NotImplementedError
    
    def decode(self, bboxes, pred_bboxes):
        """Decode bboxes.
        
        Args:
            bboxes (Tensor): Source bboxes, e.g., object proposals.
            pred_bboxes (Tensor): Encoded bboxes.
        
        Returns:
            Tensor: Decoded bboxes.
        """
        raise NotImplementedError

