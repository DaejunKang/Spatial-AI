"""
mmcv 호환성 모듈 - mmcv 의존성 없이 핵심 기능 구현
CUDA toolkit 의존성 없이 동작하도록 구현
"""
import os
import os.path as osp
import json
import pickle
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
from PIL import Image
import io

# yaml은 선택적 의존성
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ==================== Config 클래스 ====================
class Config:
    """설정 파일 로딩 및 관리 클래스 (mmcv.Config 대체)"""
    
    def __init__(self, cfg_dict=None, filename=None, text=None):
        if cfg_dict is None:
            cfg_dict = {}
        self._cfg_dict = cfg_dict
        self.filename = filename
        self.text = text
    
    @staticmethod
    def fromfile(filename):
        """파일에서 설정 로드"""
        if not osp.isfile(filename):
            raise FileNotFoundError(f'Config file {filename} not found')
        
        ext = osp.splitext(filename)[1]
        if ext not in ['.py', '.json', '.yaml', '.yml']:
            raise ValueError(f'Unsupported config file format: {ext}')
        
        if ext == '.py':
            cfg_dict = Config._load_py_file(filename)
        elif ext in ['.yaml', '.yml']:
            if not HAS_YAML:
                raise ImportError("yaml package is required for YAML config files. Install it with: pip install pyyaml")
            with open(filename, 'r', encoding='utf-8') as f:
                cfg_dict = yaml.safe_load(f)
        elif ext == '.json':
            with open(filename, 'r', encoding='utf-8') as f:
                cfg_dict = json.load(f)
        
        return Config(cfg_dict=cfg_dict, filename=filename)
    
    @staticmethod
    def _load_py_file(filename):
        """Python 파일에서 설정 로드"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg_dict = {}
        for key in dir(module):
            if not key.startswith('_'):
                cfg_dict[key] = getattr(module, key)
        return cfg_dict
    
    def __getattr__(self, name):
        if name in self._cfg_dict:
            value = self._cfg_dict[name]
            if isinstance(value, dict):
                return Config(cfg_dict=value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        if name.startswith('_') or name in ['filename', 'text']:
            super().__setattr__(name, value)
        else:
            self._cfg_dict[name] = value
    
    def get(self, key, default=None):
        """설정 값 가져오기"""
        return self._cfg_dict.get(key, default)
    
    def merge_from_dict(self, options):
        """딕셔너리에서 설정 병합"""
        if options is None:
            return
        for key, value in options.items():
            if key in self._cfg_dict:
                if isinstance(self._cfg_dict[key], dict) and isinstance(value, dict):
                    self._cfg_dict[key].update(value)
                else:
                    self._cfg_dict[key] = value
            else:
                self._cfg_dict[key] = value
    
    def dump(self, file=None):
        """설정을 파일로 저장"""
        if file is None:
            file = self.filename
        if file is None:
            raise ValueError("No filename specified")
        
        ext = osp.splitext(file)[1]
        if ext == '.json':
            with open(file, 'w', encoding='utf-8') as f:
                json.dump(self._cfg_dict, f, indent=2, ensure_ascii=False)
        elif ext in ['.yaml', '.yml']:
            if not HAS_YAML:
                raise ImportError("yaml package is required for YAML config files. Install it with: pip install pyyaml")
            with open(file, 'w', encoding='utf-8') as f:
                yaml.dump(self._cfg_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            # Python 파일로 저장
            with open(file, 'w', encoding='utf-8') as f:
                f.write(self.pretty_text)
    
    @property
    def pretty_text(self):
        """포맷된 텍스트 표현"""
        return json.dumps(self._cfg_dict, indent=2, ensure_ascii=False)


# ==================== DictAction ====================
class DictAction(argparse.Action):
    """argparse에서 딕셔너리 형태의 인자를 파싱하는 액션"""
    
    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            # 값 파싱 시도
            try:
                val = eval(val)  # 리스트, 튜플 등 파싱
            except:
                pass  # 문자열로 유지
            options[key] = val
        setattr(namespace, self.dest, options)


# ==================== 분산 학습 관련 ====================
def get_dist_info():
    """분산 학습 정보 가져오기"""
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def init_dist(launcher, backend='nccl', **kwargs):
    """분산 학습 초기화"""
    if launcher == 'pytorch':
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = kwargs.get('master_addr', '127.0.0.1')
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = str(kwargs.get('master_port', 29500))
        
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        torch.cuda.set_device(local_rank)
    elif launcher == 'slurm':
        # SLURM 환경에서의 초기화
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NPROCS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        torch.cuda.set_device(local_rank)
    # MPI는 별도 구현 필요


# ==================== 파일/디렉토리 유틸리티 ====================
def mkdir_or_exist(dir_name, mode=0o777):
    """디렉토리가 없으면 생성"""
    if dir_name == '' or dir_name is None:
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    """파일 존재 확인"""
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def list_from_file(filename, prefix='', offset=0, max_num=0, encoding='utf-8'):
    """파일에서 리스트 읽기"""
    count = 0
    item_list = []
    with open(filename, 'r', encoding=encoding) as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if max_num > 0 and count >= max_num:
                break
            item = line.strip()
            if item:
                item_list.append(prefix + item)
                count += 1
    return item_list


def is_filepath(path):
    """파일 경로인지 확인"""
    return isinstance(path, str) and (osp.isfile(path) or osp.isdir(path))


# ==================== 파일 I/O ====================
def load(file, file_format=None, **kwargs):
    """파일 로드 (pickle, json, yaml 등)"""
    if file_format is None:
        file_format = osp.splitext(file)[1]
    
    if file_format in ['.pkl', '.pickle']:
        with open(file, 'rb') as f:
            return pickle.load(f, **kwargs)
    elif file_format in ['.json']:
        with open(file, 'r', encoding='utf-8') as f:
            return json.load(f, **kwargs)
    elif file_format in ['.yaml', '.yml']:
        if not HAS_YAML:
            raise ImportError("yaml package is required for YAML files. Install it with: pip install pyyaml")
        with open(file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f, **kwargs)
    else:
        raise ValueError(f'Unsupported file format: {file_format}')


def dump(obj, file=None, file_format=None, **kwargs):
    """파일 저장 (pickle, json, yaml 등)"""
    if file_format is None:
        file_format = osp.splitext(file)[1] if file else '.pkl'
    
    if file_format in ['.pkl', '.pickle']:
        with open(file, 'wb') as f:
            pickle.dump(obj, f, **kwargs)
    elif file_format in ['.json']:
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, **kwargs)
    elif file_format in ['.yaml', '.yml']:
        if not HAS_YAML:
            raise ImportError("yaml package is required for YAML files. Install it with: pip install pyyaml")
        with open(file, 'w', encoding='utf-8') as f:
            yaml.dump(obj, f, default_flow_style=False, allow_unicode=True, **kwargs)
    else:
        raise ValueError(f'Unsupported file format: {file_format}')


# ==================== 이미지 처리 ====================
def imread(img_path, flag='color', channel_order='bgr', backend='cv2'):
    """이미지 읽기"""
    if backend == 'cv2':
        if flag == 'color':
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        elif flag == 'grayscale':
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if channel_order == 'rgb' and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    elif backend == 'pillow':
        img = Image.open(img_path)
        if flag == 'color':
            img = img.convert('RGB')
        elif flag == 'grayscale':
            img = img.convert('L')
        img = np.array(img)
        if channel_order == 'bgr' and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    else:
        raise ValueError(f'Unsupported backend: {backend}')


def imwrite(img, file_path, params=None):
    """이미지 저장"""
    if isinstance(img, np.ndarray):
        cv2.imwrite(file_path, img, params)
    else:
        raise TypeError(f'Unsupported image type: {type(img)}')


def imfrombytes(content, flag='color', channel_order='bgr', backend='cv2'):
    """바이트에서 이미지 로드"""
    if backend == 'cv2':
        nparr = np.frombuffer(content, np.uint8)
        if flag == 'color':
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif flag == 'grayscale':
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if channel_order == 'rgb' and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    else:
        raise ValueError(f'Unsupported backend: {backend}')


# ==================== 진행 상황 추적 ====================
def track_iter_progress(tasks, bar_width=50):
    """반복 작업 진행 상황 추적"""
    if isinstance(tasks, (list, tuple)):
        total = len(tasks)
    else:
        total = tasks
    
    for i, item in enumerate(tasks):
        yield item
        if (i + 1) % max(1, total // bar_width) == 0 or i == total - 1:
            percent = (i + 1) / total * 100
            bar = '=' * int(percent / 100 * bar_width)
            print(f'\r[{bar:<{bar_width}}] {percent:.1f}%', end='', flush=True)
    print()  # 줄바꿈


def track_parallel_progress(func, tasks, nproc=1, **kwargs):
    """병렬 작업 진행 상황 추적"""
    if nproc == 1:
        results = []
        for task in track_iter_progress(tasks):
            results.append(func(task, **kwargs))
        return results
    else:
        # 병렬 처리 (multiprocessing 사용)
        from multiprocessing import Pool
        with Pool(nproc) as pool:
            results = list(track_iter_progress(
                pool.starmap(func, [(task,) + tuple(kwargs.values()) for task in tasks]),
                bar_width=50
            ))
        return results


# ==================== 체크포인트 관련 ====================
def load_checkpoint(model, filename, map_location='cpu', strict=False):
    """체크포인트 로드"""
    if not osp.isfile(filename):
        raise FileNotFoundError(f'Checkpoint file {filename} not found')
    
    checkpoint = torch.load(filename, map_location=map_location)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """체크포인트 저장"""
    checkpoint = {
        'state_dict': model.state_dict(),
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if meta is not None:
        checkpoint['meta'] = meta
    
    mkdir_or_exist(osp.dirname(filename))
    torch.save(checkpoint, filename)


def load_state_dict(module, state_dict, strict=False):
    """상태 딕셔너리 로드"""
    return module.load_state_dict(state_dict, strict=strict)


# ==================== 데이터 병렬화 ====================
class MMDataParallel(nn.DataParallel):
    """데이터 병렬화 (mmcv.MMDataParallel 대체)"""
    
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__(module, device_ids, output_device, dim)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class MMDistributedDataParallel(nn.parallel.DistributedDataParallel):
    """분산 데이터 병렬화 (mmcv.MMDistributedDataParallel 대체)"""
    
    def __init__(self, module, device_ids=None, output_device=None, dim=0, 
                 broadcast_buffers=True, find_unused_parameters=False, **kwargs):
        super().__init__(
            module,
            device_ids=device_ids,
            output_device=output_device,
            dim=dim,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            **kwargs
        )
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# ==================== 모델 유틸리티 ====================
def fuse_conv_bn(module):
    """Conv-BN 융합"""
    def _fuse_conv_bn(conv, bn):
        """단일 Conv-BN 융합"""
        conv.weight.data = (conv.weight.data * bn.weight.data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) /
                           (bn.running_var.data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + bn.eps).sqrt())
        if conv.bias is not None:
            conv.bias.data = (conv.bias.data - bn.running_mean.data) * bn.weight.data / \
                            (bn.running_var.data + bn.eps).sqrt() + bn.bias.data
        else:
            conv.bias = nn.Parameter(-bn.running_mean.data * bn.weight.data /
                                   (bn.running_var.data + bn.eps).sqrt() + bn.bias.data)
        return conv
    
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            for bn_name, bn in list(child.named_children()):
                if isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    setattr(child, bn_name, nn.Identity())
                    _fuse_conv_bn(child, bn)
        else:
            fuse_conv_bn(child)
    return module


def wrap_fp16_model(model):
    """FP16 모델 래핑 (실제로는 모델을 그대로 반환)"""
    # FP16 지원은 torch.cuda.amp를 사용하도록 권장
    return model


# ==================== 기타 유틸리티 ====================
def import_modules_from_strings(imports, allow_failed_imports=False):
    """문자열 리스트에서 모듈 임포트"""
    if isinstance(imports, str):
        imports = [imports]
    
    for imp in imports:
        try:
            import importlib
            importlib.import_module(imp)
        except ImportError as e:
            if not allow_failed_imports:
                raise ImportError(f'Failed to import {imp}: {e}')


# PyTorch 버전 정보
TORCH_VERSION = torch.__version__


def digit_version(version_str):
    """버전 문자열을 숫자 리스트로 변환"""
    version_digits = []
    for x in version_str.split('.'):
        if x.isdigit():
            version_digits.append(int(x))
        else:
            # 알파벳이 포함된 경우 (예: '1.8.1+cu111')
            break
    return version_digits


# ==================== Runner 관련 (force_fp32, auto_fp16, BaseModule) ====================
def force_fp32(apply_to=None, out_fp16=False):
    """FP32 강제 데코레이터 (mmcv.runner.force_fp32 호환)
    
    Args:
        apply_to: 적용할 인자 목록 (사용하지 않음)
        out_fp16: 출력을 FP16으로 변환할지 여부 (사용하지 않음)
    
    Returns:
        decorator: 데코레이터 함수
    """
    def decorator(func):
        # 실제로는 함수를 그대로 반환 (PyTorch의 autocast 사용 권장)
        return func
    return decorator


def auto_fp16(apply_to=None, out_fp16=False):
    """FP16 자동 데코레이터 (mmcv.runner.auto_fp16 호환)
    
    Args:
        apply_to: 적용할 인자 목록 (사용하지 않음)
        out_fp16: 출력을 FP16으로 변환할지 여부 (사용하지 않음)
    
    Returns:
        decorator: 데코레이터 함수
    """
    def decorator(func):
        # 실제로는 함수를 그대로 반환 (PyTorch의 autocast 사용 권장)
        return func
    return decorator


class BaseModule(nn.Module):
    """기본 모듈 클래스 (mmcv.runner.BaseModule 호환)
    
    init_cfg를 지원하는 기본 모듈 클래스
    """
    def __init__(self, init_cfg=None):
        super(BaseModule, self).__init__()
        self._is_init = False
        self.init_cfg = init_cfg
        
        # init_cfg가 있으면 초기화
        if init_cfg is not None:
            self.init_weights()
    
    def init_weights(self):
        """가중치 초기화"""
        if self._is_init:
            return
        
        if self.init_cfg is not None:
            # init_cfg에 따른 초기화 로직 (간단한 버전)
            # 실제 사용 시 필요에 따라 확장 가능
            pass
        
        # 하위 모듈 초기화
        for module in self.children():
            if hasattr(module, 'init_weights'):
                module.init_weights()
        
        self._is_init = True


# ==================== Hook 시스템 ====================
from .registry import Registry

HOOKS = Registry('hook')


class Hook:
    """Hook 기본 클래스 (mmcv.runner.hooks.hook.Hook 호환)"""
    
    def before_run(self, runner):
        """Runner 시작 전 호출"""
        pass
    
    def after_run(self, runner):
        """Runner 종료 후 호출"""
        pass
    
    def before_epoch(self, runner):
        """Epoch 시작 전 호출"""
        pass
    
    def after_epoch(self, runner):
        """Epoch 종료 후 호출"""
        pass
    
    def before_iter(self, runner):
        """Iteration 시작 전 호출"""
        pass
    
    def after_iter(self, runner):
        """Iteration 종료 후 호출"""
        pass
    
    def before_train_epoch(self, runner):
        """Train epoch 시작 전 호출"""
        pass
    
    def after_train_epoch(self, runner):
        """Train epoch 종료 후 호출"""
        pass
    
    def before_train_iter(self, runner):
        """Train iteration 시작 전 호출"""
        pass
    
    def after_train_iter(self, runner):
        """Train iteration 종료 후 호출"""
        pass
    
    def before_val_epoch(self, runner):
        """Val epoch 시작 전 호출"""
        pass
    
    def after_val_epoch(self, runner):
        """Val epoch 종료 후 호출"""
        pass
    
    def before_val_iter(self, runner):
        """Val iteration 시작 전 호출"""
        pass
    
    def after_val_iter(self, runner):
        """Val iteration 종료 후 호출"""
        pass


# ==================== Optimizer Registry ====================
OPTIMIZERS = Registry('optimizer')
