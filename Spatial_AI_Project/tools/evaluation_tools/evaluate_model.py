# -*- coding: utf-8 -*-
"""
모델 평가 도구
- 메트릭 계산
- 결과 저장
- 성능 비교
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import time


class ModelEvaluator:
    """모델 평가 클래스"""
    
    def __init__(self):
        pass
    
    def calculate_detection_metrics(self,
                                   predictions: List[Dict],
                                   ground_truths: List[Dict],
                                   iou_threshold: float = 0.5) -> Dict:
        """
        Detection 메트릭을 계산합니다.
        
        Args:
            predictions: 예측 결과 리스트
            ground_truths: Ground truth 리스트
            iou_threshold: IoU 임계값
            
        Returns:
            메트릭 딕셔너리
        """
        # 간단한 구현 (실제로는 더 정교한 매칭 필요)
        tp = 0
        fp = 0
        fn = 0
        
        matched_gt = set()
        
        for pred in sorted(predictions, key=lambda x: x.get('score', 0), reverse=True):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gt:
                    continue
                
                iou = self._calculate_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn = len(ground_truths) - len(matched_gt)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """IoU 계산 (간단한 2D 구현)"""
        x1, y1, w1, h1 = bbox1[:4]
        x2, y2, w2, h2 = bbox2[:4]
        
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
    
    def calculate_inference_time(self,
                                model,
                                input_data,
                                num_iterations: int = 100,
                                warmup: int = 10) -> Dict:
        """
        추론 시간을 측정합니다.
        
        Args:
            model: 모델
            input_data: 입력 데이터
            num_iterations: 반복 횟수
            warmup: 워밍업 횟수
            
        Returns:
            시간 측정 결과
        """
        # 워밍업
        for _ in range(warmup):
            _ = model(input_data)
        
        # 측정
        times = []
        for _ in range(num_iterations):
            start = time.time()
            _ = model(input_data)
            end = time.time()
            times.append(end - start)
        
        times = np.array(times)
        
        return {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'median': float(np.median(times)),
            'p95': float(np.percentile(times, 95)),
            'p99': float(np.percentile(times, 99)),
            'fps': float(1.0 / np.mean(times))
        }
    
    def compare_results(self,
                       results1: Dict,
                       results2: Dict) -> Dict:
        """
        두 평가 결과를 비교합니다.
        
        Args:
            results1: 첫 번째 결과
            results2: 두 번째 결과
            
        Returns:
            비교 결과
        """
        comparison = {}
        
        for key in set(results1.keys()) | set(results2.keys()):
            if key in results1 and key in results2:
                val1 = results1[key]
                val2 = results2[key]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff = val2 - val1
                    pct_change = (diff / (abs(val1) + 1e-8)) * 100
                    comparison[key] = {
                        'value1': val1,
                        'value2': val2,
                        'diff': diff,
                        'pct_change': pct_change
                    }
        
        return comparison


def main():
    parser = argparse.ArgumentParser(description='모델 평가 도구')
    parser.add_argument('--predictions', type=str, required=True,
                       help='예측 결과 파일 (JSON)')
    parser.add_argument('--ground-truths', type=str, required=True,
                       help='Ground truth 파일 (JSON)')
    parser.add_argument('--output', type=str, required=True,
                       help='평가 결과 저장 경로')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU 임계값')
    
    args = parser.parse_args()
    
    # 데이터 로드
    with open(args.predictions, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    with open(args.ground_truths, 'r', encoding='utf-8') as f:
        ground_truths = json.load(f)
    
    # 평가 수행
    evaluator = ModelEvaluator()
    metrics = evaluator.calculate_detection_metrics(
        predictions, ground_truths, args.iou_threshold
    )
    
    # 결과 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print("Evaluation Results:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

