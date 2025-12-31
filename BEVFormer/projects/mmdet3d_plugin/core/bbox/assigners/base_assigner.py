"""
Base assigner classes - mmdet의 BaseAssigner와 AssignResult를 대체
"""
import torch


class AssignResult:
    """Stores assignments between predicted and truth boxes.
    
    Attributes:
        num_gts (int): the number of truth boxes considered when computing this
            assignment
        gt_inds (LongTensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.
        max_overlaps (FloatTensor): the iou between the predicted box and its
            assigned truth box.
        labels (None | LongTensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.
    """
    
    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        
        # Interface for possible user-defined properties
        self._extra_properties = {}
    
    @property
    def num_preds(self):
        """int: the number of predictions in this assignment"""
        return len(self.gt_inds)
    
    def set_extra_property(self, key, value):
        """Set user-defined new property."""
        self._extra_properties[key] = value
    
    def get_extra_property(self, key):
        """Get user-defined property."""
        return self._extra_properties.get(key, None)
    
    @property
    def info(self):
        """dict: a dictionary of info about the object"""
        basic_info = {
            'num_gts': self.num_gts,
            'num_preds': self.num_preds,
            'gt_inds': self.gt_inds,
            'max_overlaps': self.max_overlaps,
            'labels': self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info


class BaseAssigner:
    """Base assigner that assigns boxes to ground truth boxes."""
    
    def assign(self, bbox_pred, cls_pred, gt_bboxes, gt_labels, gt_bboxes_ignore=None):
        """Assign boxes to ground truth boxes.
        
        Args:
            bbox_pred (Tensor): Predicted boxes.
            cls_pred (Tensor): Predicted classification logits.
            gt_bboxes (Tensor): Ground truth boxes.
            gt_labels (Tensor): Ground truth labels.
            gt_bboxes_ignore (Tensor, optional): Ground truth boxes to ignore.
        
        Returns:
            AssignResult: The assignment result.
        """
        raise NotImplementedError


