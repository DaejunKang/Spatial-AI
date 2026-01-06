# -*- coding: utf-8 -*-
"""
실험 관리 도구
- 실험 설정 관리
- 실험 결과 추적
- 실험 비교
- 하이퍼파라미터 검색
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import shutil


class ExperimentManager:
    """실험 관리 클래스"""
    
    def __init__(self, experiment_dir: str):
        """
        Args:
            experiment_dir: 실험 디렉토리 경로
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    def create_experiment(self,
                         name: str,
                         config: Dict,
                         description: Optional[str] = None) -> str:
        """
        새로운 실험을 생성합니다.
        
        Args:
            name: 실험 이름
            config: 실험 설정
            description: 실험 설명
            
        Returns:
            실험 디렉토리 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{name}_{timestamp}"
        exp_dir = self.experiment_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 설정 저장
        config_path = exp_dir / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # 메타데이터 저장
        metadata = {
            'name': name,
            'description': description,
            'created_at': timestamp,
            'config': config
        }
        
        metadata_path = exp_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 서브디렉토리 생성
        (exp_dir / 'checkpoints').mkdir(exist_ok=True)
        (exp_dir / 'logs').mkdir(exist_ok=True)
        (exp_dir / 'results').mkdir(exist_ok=True)
        (exp_dir / 'visualizations').mkdir(exist_ok=True)
        
        print(f"Experiment created: {exp_dir}")
        return str(exp_dir)
    
    def save_results(self,
                    experiment_name: str,
                    results: Dict,
                    epoch: Optional[int] = None) -> None:
        """
        실험 결과를 저장합니다.
        
        Args:
            experiment_name: 실험 이름
            results: 결과 딕셔너리
            epoch: 에포크 번호
        """
        exp_dir = self._find_experiment(experiment_name)
        if exp_dir is None:
            raise ValueError(f"Experiment not found: {experiment_name}")
        
        if epoch is not None:
            results_path = exp_dir / 'results' / f'epoch_{epoch}.json'
        else:
            results_path = exp_dir / 'results' / 'final_results.json'
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def load_experiment(self, experiment_name: str) -> Dict:
        """
        실험 정보를 로드합니다.
        
        Args:
            experiment_name: 실험 이름
            
        Returns:
            실험 정보 딕셔너리
        """
        exp_dir = self._find_experiment(experiment_name)
        if exp_dir is None:
            raise ValueError(f"Experiment not found: {experiment_name}")
        
        metadata_path = exp_dir / 'metadata.json'
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        config_path = exp_dir / 'config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        metadata['config'] = config
        metadata['experiment_dir'] = str(exp_dir)
        
        return metadata
    
    def list_experiments(self) -> List[Dict]:
        """
        모든 실험을 나열합니다.
        
        Returns:
            실험 리스트
        """
        experiments = []
        
        for exp_dir in self.experiment_dir.iterdir():
            if exp_dir.is_dir():
                metadata_path = exp_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    metadata['experiment_dir'] = str(exp_dir)
                    experiments.append(metadata)
        
        return sorted(experiments, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def compare_experiments(self, experiment_names: List[str]) -> Dict:
        """
        여러 실험을 비교합니다.
        
        Args:
            experiment_names: 실험 이름 리스트
            
        Returns:
            비교 결과
        """
        experiments = []
        for name in experiment_names:
            exp_info = self.load_experiment(name)
            experiments.append(exp_info)
        
        comparison = {
            'experiments': experiments,
            'config_differences': self._compare_configs([exp['config'] for exp in experiments])
        }
        
        return comparison
    
    def _find_experiment(self, experiment_name: str) -> Optional[Path]:
        """실험 디렉토리를 찾습니다."""
        for exp_dir in self.experiment_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith(experiment_name):
                return exp_dir
        return None
    
    def _compare_configs(self, configs: List[Dict]) -> Dict:
        """설정들을 비교합니다."""
        if not configs:
            return {}
        
        all_keys = set()
        for config in configs:
            all_keys.update(config.keys())
        
        differences = {}
        for key in all_keys:
            values = [config.get(key) for config in configs]
            if len(set(str(v) for v in values)) > 1:
                differences[key] = values
        
        return differences


def main():
    parser = argparse.ArgumentParser(description='실험 관리 도구')
    subparsers = parser.add_subparsers(dest='command', help='명령어')
    
    # 실험 생성
    create_parser = subparsers.add_parser('create', help='새 실험 생성')
    create_parser.add_argument('--name', type=str, required=True, help='실험 이름')
    create_parser.add_argument('--config', type=str, required=True, help='설정 파일 경로')
    create_parser.add_argument('--description', type=str, help='실험 설명')
    create_parser.add_argument('--experiment-dir', type=str, default='./experiments',
                              help='실험 디렉토리')
    
    # 실험 목록
    list_parser = subparsers.add_parser('list', help='실험 목록')
    list_parser.add_argument('--experiment-dir', type=str, default='./experiments',
                           help='실험 디렉토리')
    
    # 실험 로드
    load_parser = subparsers.add_parser('load', help='실험 로드')
    load_parser.add_argument('--name', type=str, required=True, help='실험 이름')
    load_parser.add_argument('--experiment-dir', type=str, default='./experiments',
                            help='실험 디렉토리')
    
    # 실험 비교
    compare_parser = subparsers.add_parser('compare', help='실험 비교')
    compare_parser.add_argument('--experiments', type=str, nargs='+', required=True,
                              help='실험 이름들')
    compare_parser.add_argument('--experiment-dir', type=str, default='./experiments',
                               help='실험 디렉토리')
    compare_parser.add_argument('--output', type=str, help='비교 결과 저장 경로')
    
    args = parser.parse_args()
    
    manager = ExperimentManager(args.experiment_dir)
    
    if args.command == 'create':
        with open(args.config, 'r', encoding='utf-8') as f:
            if args.config.endswith('.yaml') or args.config.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        manager.create_experiment(args.name, config, args.description)
    
    elif args.command == 'list':
        experiments = manager.list_experiments()
        for exp in experiments:
            print(f"{exp['name']}: {exp.get('created_at', 'unknown')}")
    
    elif args.command == 'load':
        exp_info = manager.load_experiment(args.name)
        print(json.dumps(exp_info, indent=2, ensure_ascii=False))
    
    elif args.command == 'compare':
        comparison = manager.compare_experiments(args.experiments)
        print(json.dumps(comparison, indent=2, ensure_ascii=False))
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

