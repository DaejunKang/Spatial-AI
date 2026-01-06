"""
자율주행 AI 개발을 위한 모델 분석 유틸리티
- 모델 성능 분석
- 학습 곡선 시각화
- 예측 결과 분석
- 오류 분석
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import torch
from collections import defaultdict
import pandas as pd


class PerformanceAnalyzer:
    """모델 성능 분석 클래스"""
    
    def plot_training_curves(self,
                            logs: Dict[str, List[float]],
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        학습 곡선을 시각화합니다.
        
        Args:
            logs: {'loss': [values], 'accuracy': [values], ...} 형태의 로그
            save_path: 저장 경로
            figsize: figure 크기
        """
        num_metrics = len(logs)
        cols = 2
        rows = (num_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if num_metrics > 1 else [axes]
        
        for idx, (metric_name, values) in enumerate(logs.items()):
            ax = axes[idx]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, marker='o', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(f'{metric_name} Curve')
            ax.grid(True, alpha=0.3)
        
        # 빈 subplot 숨기기
        for idx in range(num_metrics, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()
    
    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             class_names: List[str],
                             save_path: Optional[str] = None,
                             normalize: bool = True) -> None:
        """
        Confusion matrix를 시각화합니다.
        
        Args:
            y_true: 실제 라벨
            y_pred: 예측 라벨
            class_names: 클래스 이름 리스트
            save_path: 저장 경로
            normalize: 정규화 여부
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # 클래스 이름 설정
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title='Confusion Matrix',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # 값 표시
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'{cm[i, j]:.2f}',
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()
    
    def plot_precision_recall_curve(self,
                                    y_true: np.ndarray,
                                    y_scores: np.ndarray,
                                    class_names: Optional[List[str]] = None,
                                    save_path: Optional[str] = None) -> None:
        """
        Precision-Recall 곡선을 시각화합니다.
        
        Args:
            y_true: 실제 라벨 (one-hot 또는 클래스 인덱스)
            y_scores: 예측 점수
            class_names: 클래스 이름 리스트
            save_path: 저장 경로
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        if len(y_true.shape) == 1:
            # 이진 분류
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)
            
            plt.figure(figsize=(10, 8))
            plt.plot(recall, precision, linewidth=2, label=f'AP = {ap:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            # 다중 클래스
            num_classes = y_true.shape[1] if len(y_true.shape) > 1 else len(np.unique(y_true))
            if class_names is None:
                class_names = [f'Class {i}' for i in range(num_classes)]
            
            plt.figure(figsize=(12, 8))
            
            for i in range(num_classes):
                if len(y_true.shape) > 1:
                    y_true_class = y_true[:, i]
                    y_scores_class = y_scores[:, i]
                else:
                    y_true_class = (y_true == i).astype(int)
                    y_scores_class = y_scores[:, i] if y_scores.ndim > 1 else y_scores
                
                precision, recall, _ = precision_recall_curve(y_true_class, y_scores_class)
                ap = average_precision_score(y_true_class, y_scores_class)
                
                plt.plot(recall, precision, linewidth=2, 
                        label=f'{class_names[i]} (AP = {ap:.3f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()


class DetectionAnalyzer:
    """Detection 결과 분석 클래스"""
    
    def analyze_detection_results(self,
                                  predictions: List[Dict],
                                  ground_truths: List[Dict],
                                  class_names: List[str],
                                  iou_threshold: float = 0.5,
                                  save_path: Optional[str] = None) -> Dict:
        """
        Detection 결과를 분석합니다.
        
        Args:
            predictions: 예측 결과 [{'bbox': [x, y, w, h, yaw], 'score': float, 'class': str}, ...]
            ground_truths: Ground truth [{'bbox': [x, y, w, h, yaw], 'class': str}, ...]
            class_names: 클래스 이름 리스트
            iou_threshold: IoU 임계값
            save_path: 저장 경로
            
        Returns:
            분석 결과 딕셔너리
        """
        # 클래스별 통계
        class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        # 각 클래스별로 분석
        for class_name in class_names:
            pred_class = [p for p in predictions if p.get('class') == class_name]
            gt_class = [g for g in ground_truths if g.get('class') == class_name]
            
            # IoU 계산 및 매칭
            matched_gt = set()
            for pred in sorted(pred_class, key=lambda x: x.get('score', 0), reverse=True):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_class):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = self._calculate_iou_3d(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    class_stats[class_name]['tp'] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    class_stats[class_name]['fp'] += 1
            
            # False Negative 계산
            class_stats[class_name]['fn'] = len(gt_class) - len(matched_gt)
        
        # Precision, Recall 계산
        results = {}
        for class_name in class_names:
            stats = class_stats[class_name]
            tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            results[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        
        # 시각화
        if save_path:
            self._plot_detection_metrics(results, class_names, save_path)
        
        return results
    
    def _calculate_iou_3d(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        3D IoU를 계산합니다. (간단한 구현)
        실제로는 더 정확한 3D IoU 계산이 필요합니다.
        """
        # 간단한 2D IoU 기반 계산 (실제로는 3D IoU 필요)
        x1, y1, w1, h1, _ = bbox1
        x2, y2, w2, h2, _ = bbox2
        
        # 2D IoU 계산
        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-8)
    
    def _plot_detection_metrics(self,
                                results: Dict,
                                class_names: List[str],
                                save_path: str) -> None:
        """Detection 메트릭을 시각화합니다."""
        precisions = [results[cls]['precision'] for cls in class_names]
        recalls = [results[cls]['recall'] for cls in class_names]
        f1_scores = [results[cls]['f1'] for cls in class_names]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Detection Metrics by Class')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    
    def plot_error_analysis(self,
                           predictions: List[Dict],
                           ground_truths: List[Dict],
                           save_path: Optional[str] = None) -> None:
        """
        오류 분석을 시각화합니다.
        
        Args:
            predictions: 예측 결과
            ground_truths: Ground truth
            save_path: 저장 경로
        """
        # 거리별 오류 분석
        distance_errors = []
        size_errors = []
        
        # 간단한 매칭 (실제로는 더 정교한 매칭 필요)
        for pred, gt in zip(predictions[:min(len(predictions), len(ground_truths))], 
                           ground_truths[:min(len(predictions), len(ground_truths))]):
            pred_center = np.array(pred['bbox'][:2])
            gt_center = np.array(gt['bbox'][:2])
            distance_error = np.linalg.norm(pred_center - gt_center)
            distance_errors.append(distance_error)
            
            pred_size = np.array(pred['bbox'][2:4])
            gt_size = np.array(gt['bbox'][2:4])
            size_error = np.linalg.norm(pred_size - gt_size)
            size_errors.append(size_error)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].hist(distance_errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Distance Error (m)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Center Distance Error Distribution')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(size_errors, bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Size Error (m)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Size Error Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        else:
            plt.show()
        plt.close()


# 사용 예시 함수
def example_usage():
    """사용 예시"""
    # 성능 분석
    analyzer = PerformanceAnalyzer()
    
    # 학습 곡선
    logs = {
        'loss': [2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6],
        'accuracy': [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        'val_loss': [2.6, 2.1, 1.6, 1.3, 1.1, 0.9, 0.7],
        'val_accuracy': [0.48, 0.58, 0.68, 0.73, 0.78, 0.83, 0.88]
    }
    
    analyzer.plot_training_curves(
        logs,
        save_path='training_curves.png'
    )
    
    # Detection 분석
    det_analyzer = DetectionAnalyzer()
    
    predictions = [
        {'bbox': [10, 10, 4, 2, 0], 'score': 0.9, 'class': 'car'},
        {'bbox': [20, 20, 3, 1.5, 0], 'score': 0.8, 'class': 'pedestrian'},
    ]
    
    ground_truths = [
        {'bbox': [10.5, 10.2, 4.1, 2.1, 0], 'class': 'car'},
        {'bbox': [20.3, 20.1, 3.2, 1.6, 0], 'class': 'pedestrian'},
    ]
    
    results = det_analyzer.analyze_detection_results(
        predictions,
        ground_truths,
        class_names=['car', 'pedestrian'],
        save_path='detection_metrics.png'
    )


if __name__ == '__main__':
    example_usage()

