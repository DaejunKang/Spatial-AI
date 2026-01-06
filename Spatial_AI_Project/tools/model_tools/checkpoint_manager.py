# -*- coding: utf-8 -*-
"""
체크포인트 관리 도구
- 체크포인트 비교
- 체크포인트 병합
- 체크포인트 검증
- 모델 정보 추출
"""

import argparse
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from collections import OrderedDict


class CheckpointManager:
    """체크포인트 관리 클래스"""
    
    def __init__(self):
        pass
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        체크포인트를 로드합니다.
        
        Args:
            checkpoint_path: 체크포인트 파일 경로
            
        Returns:
            체크포인트 딕셔너리
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint
    
    def get_model_info(self, checkpoint: Dict) -> Dict:
        """
        체크포인트에서 모델 정보를 추출합니다.
        
        Args:
            checkpoint: 체크포인트 딕셔너리
            
        Returns:
            모델 정보 딕셔너리
        """
        info = {
            'has_state_dict': 'state_dict' in checkpoint,
            'has_optimizer': 'optimizer' in checkpoint,
            'has_scheduler': 'lr_scheduler' in checkpoint,
            'has_meta': 'meta' in checkpoint,
        }
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            info['num_parameters'] = sum(p.numel() for p in state_dict.values())
            info['num_layers'] = len(state_dict)
            info['layer_names'] = list(state_dict.keys())[:10]  # 처음 10개만
        
        if 'meta' in checkpoint:
            info['meta'] = checkpoint['meta']
        
        if 'epoch' in checkpoint:
            info['epoch'] = checkpoint['epoch']
        
        if 'iter' in checkpoint:
            info['iter'] = checkpoint['iter']
        
        return info
    
    def compare_checkpoints(self, 
                           checkpoint1_path: str,
                           checkpoint2_path: str) -> Dict:
        """
        두 체크포인트를 비교합니다.
        
        Args:
            checkpoint1_path: 첫 번째 체크포인트 경로
            checkpoint2_path: 두 번째 체크포인트 경로
            
        Returns:
            비교 결과 딕셔너리
        """
        ckpt1 = self.load_checkpoint(checkpoint1_path)
        ckpt2 = self.load_checkpoint(checkpoint2_path)
        
        comparison = {
            'same_keys': True,
            'different_keys': [],
            'same_values': True,
            'different_layers': [],
            'max_diff': 0.0,
            'mean_diff': 0.0
        }
        
        if 'state_dict' not in ckpt1 or 'state_dict' not in ckpt2:
            comparison['same_keys'] = False
            return comparison
        
        state_dict1 = ckpt1['state_dict']
        state_dict2 = ckpt2['state_dict']
        
        keys1 = set(state_dict1.keys())
        keys2 = set(state_dict2.keys())
        
        if keys1 != keys2:
            comparison['same_keys'] = False
            comparison['different_keys'] = {
                'only_in_1': list(keys1 - keys2),
                'only_in_2': list(keys2 - keys1)
            }
        
        # 공통 키에 대해 값 비교
        common_keys = keys1 & keys2
        diffs = []
        
        for key in common_keys:
            if not torch.equal(state_dict1[key], state_dict2[key]):
                comparison['same_values'] = False
                diff = torch.abs(state_dict1[key] - state_dict2[key]).max().item()
                diffs.append(diff)
                comparison['different_layers'].append({
                    'layer': key,
                    'max_diff': diff
                })
        
        if diffs:
            comparison['max_diff'] = max(diffs)
            comparison['mean_diff'] = sum(diffs) / len(diffs)
        
        return comparison
    
    def merge_checkpoints(self,
                         checkpoint_paths: List[str],
                         output_path: str,
                         merge_strategy: str = 'average') -> None:
        """
        여러 체크포인트를 병합합니다.
        
        Args:
            checkpoint_paths: 체크포인트 파일 경로 리스트
            output_path: 출력 경로
            merge_strategy: 병합 전략 ('average', 'first', 'last')
        """
        checkpoints = [self.load_checkpoint(path) for path in checkpoint_paths]
        
        if merge_strategy == 'first':
            merged = checkpoints[0]
        elif merge_strategy == 'last':
            merged = checkpoints[-1]
        elif merge_strategy == 'average':
            # state_dict 평균
            state_dicts = [ckpt['state_dict'] for ckpt in checkpoints if 'state_dict' in ckpt]
            
            if not state_dicts:
                raise ValueError("No state_dict found in checkpoints")
            
            merged_state_dict = OrderedDict()
            for key in state_dicts[0].keys():
                if all(key in sd for sd in state_dicts):
                    tensors = [sd[key] for sd in state_dicts]
                    merged_state_dict[key] = torch.stack(tensors).mean(dim=0)
                else:
                    merged_state_dict[key] = state_dicts[0][key]
            
            merged = checkpoints[0].copy()
            merged['state_dict'] = merged_state_dict
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        torch.save(merged, output_path)
        print(f"Merged checkpoint saved to {output_path}")
    
    def extract_state_dict(self,
                          checkpoint_path: str,
                          output_path: str,
                          prefix: Optional[str] = None) -> None:
        """
        체크포인트에서 state_dict만 추출합니다.
        
        Args:
            checkpoint_path: 체크포인트 파일 경로
            output_path: 출력 경로
            prefix: 제거할 키 prefix (예: 'module.')
        """
        checkpoint = self.load_checkpoint(checkpoint_path)
        
        if 'state_dict' not in checkpoint:
            raise ValueError("No state_dict found in checkpoint")
        
        state_dict = checkpoint['state_dict']
        
        if prefix:
            # prefix 제거
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        torch.save(state_dict, output_path)
        print(f"State dict saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='체크포인트 관리 도구')
    subparsers = parser.add_subparsers(dest='command', help='명령어')
    
    # 정보 추출
    info_parser = subparsers.add_parser('info', help='체크포인트 정보 추출')
    info_parser.add_argument('checkpoint', type=str, help='체크포인트 파일 경로')
    info_parser.add_argument('--output', type=str, help='정보 저장 경로 (JSON)')
    
    # 비교
    compare_parser = subparsers.add_parser('compare', help='체크포인트 비교')
    compare_parser.add_argument('checkpoint1', type=str, help='첫 번째 체크포인트')
    compare_parser.add_argument('checkpoint2', type=str, help='두 번째 체크포인트')
    compare_parser.add_argument('--output', type=str, help='비교 결과 저장 경로 (JSON)')
    
    # 병합
    merge_parser = subparsers.add_parser('merge', help='체크포인트 병합')
    merge_parser.add_argument('checkpoints', type=str, nargs='+', help='체크포인트 파일 경로들')
    merge_parser.add_argument('--output', type=str, required=True, help='출력 경로')
    merge_parser.add_argument('--strategy', type=str, choices=['average', 'first', 'last'],
                            default='average', help='병합 전략')
    
    # 추출
    extract_parser = subparsers.add_parser('extract', help='state_dict 추출')
    extract_parser.add_argument('checkpoint', type=str, help='체크포인트 파일 경로')
    extract_parser.add_argument('--output', type=str, required=True, help='출력 경로')
    extract_parser.add_argument('--prefix', type=str, help='제거할 키 prefix')
    
    args = parser.parse_args()
    
    manager = CheckpointManager()
    
    if args.command == 'info':
        checkpoint = manager.load_checkpoint(args.checkpoint)
        info = manager.get_model_info(checkpoint)
        
        print(json.dumps(info, indent=2, ensure_ascii=False))
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, ensure_ascii=False)
    
    elif args.command == 'compare':
        comparison = manager.compare_checkpoints(args.checkpoint1, args.checkpoint2)
        
        print(json.dumps(comparison, indent=2, ensure_ascii=False))
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    elif args.command == 'merge':
        manager.merge_checkpoints(args.checkpoints, args.output, args.strategy)
    
    elif args.command == 'extract':
        manager.extract_state_dict(args.checkpoint, args.output, args.prefix)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

