# -*- coding: utf-8 -*-
"""
모델 내보내기 및 변환 도구
PyTorch 모델을 ONNX, TensorRT, TorchScript 등으로 변환합니다.
"""

import argparse
import torch
import torch.onnx
import onnx
from pathlib import Path
from typing import Dict, Optional
import numpy as np


def export_to_onnx(model: torch.nn.Module,
                   dummy_input: Dict[str, torch.Tensor],
                   output_path: str,
                   opset_version: int = 11,
                   dynamic_axes: Optional[Dict] = None) -> str:
    """
    PyTorch 모델을 ONNX로 내보냅니다.
    
    Args:
        model: PyTorch 모델
        dummy_input: 더미 입력 (딕셔너리 형태)
        output_path: 출력 경로
        opset_version: ONNX opset 버전
        dynamic_axes: 동적 축 설정
    
    Returns:
        내보낸 파일 경로
    """
    model.eval()
    
    # 입력을 튜플로 변환
    input_names = list(dummy_input.keys())
    input_values = tuple(dummy_input.values())
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            input_values,
            output_path,
            input_names=input_names,
            output_names=['output'],
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
    
    # ONNX 모델 검증
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"✓ ONNX 모델이 저장되었습니다: {output_path}")
    return output_path


def export_to_torchscript(model: torch.nn.Module,
                          dummy_input: Dict[str, torch.Tensor],
                          output_path: str,
                          method: str = 'trace') -> str:
    """
    PyTorch 모델을 TorchScript로 내보냅니다.
    
    Args:
        model: PyTorch 모델
        dummy_input: 더미 입력
        output_path: 출력 경로
        method: 내보내기 방법 ('trace' 또는 'script')
    
    Returns:
        내보낸 파일 경로
    """
    model.eval()
    
    if method == 'trace':
        # Tracing 방식
        input_values = tuple(dummy_input.values())
        with torch.no_grad():
            traced_model = torch.jit.trace(model, input_values)
        traced_model.save(output_path)
    else:
        # Scripting 방식
        scripted_model = torch.jit.script(model)
        scripted_model.save(output_path)
    
    print(f"✓ TorchScript 모델이 저장되었습니다: {output_path}")
    return output_path


def optimize_onnx_model(onnx_path: str, output_path: str) -> str:
    """
    ONNX 모델을 최적화합니다.
    
    Args:
        onnx_path: 입력 ONNX 모델 경로
        output_path: 최적화된 모델 출력 경로
    
    Returns:
        최적화된 모델 경로
    """
    try:
        import onnxoptimizer
        
        model = onnx.load(onnx_path)
        optimized_model = onnxoptimizer.optimize(model)
        onnx.save(optimized_model, output_path)
        
        print(f"✓ 최적화된 ONNX 모델이 저장되었습니다: {output_path}")
        return output_path
    except ImportError:
        print("⚠ onnxoptimizer가 설치되지 않았습니다. 최적화를 건너뜁니다.")
        return onnx_path


def quantize_model(model: torch.nn.Module,
                  dummy_input: Dict[str, torch.Tensor],
                  output_path: str,
                  quantization_type: str = 'dynamic') -> str:
    """
    모델을 양자화합니다.
    
    Args:
        model: PyTorch 모델
        dummy_input: 더미 입력
        output_path: 출력 경로
        quantization_type: 양자화 타입 ('dynamic', 'static', 'qat')
    
    Returns:
        양자화된 모델 경로
    """
    model.eval()
    
    if quantization_type == 'dynamic':
        # 동적 양자화
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        torch.save(quantized_model.state_dict(), output_path)
    elif quantization_type == 'static':
        # 정적 양자화 (더 복잡한 설정 필요)
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        # Calibration 필요
        # torch.quantization.convert(model, inplace=True)
        print("⚠ 정적 양자화는 calibration 데이터가 필요합니다.")
        return None
    else:
        print(f"⚠ 지원하지 않는 양자화 타입: {quantization_type}")
        return None
    
    print(f"✓ 양자화된 모델이 저장되었습니다: {output_path}")
    return output_path


def compare_model_sizes(checkpoint_path: str, onnx_path: str, 
                       torchscript_path: str) -> Dict:
    """
    모델 크기를 비교합니다.
    
    Args:
        checkpoint_path: PyTorch 체크포인트 경로
        onnx_path: ONNX 모델 경로
        torchscript_path: TorchScript 모델 경로
    
    Returns:
        크기 비교 결과
    """
    sizes = {}
    
    if checkpoint_path and Path(checkpoint_path).exists():
        sizes['pytorch'] = Path(checkpoint_path).stat().st_size / (1024 * 1024)  # MB
    
    if onnx_path and Path(onnx_path).exists():
        sizes['onnx'] = Path(onnx_path).stat().st_size / (1024 * 1024)  # MB
    
    if torchscript_path and Path(torchscript_path).exists():
        sizes['torchscript'] = Path(torchscript_path).stat().st_size / (1024 * 1024)  # MB
    
    print("\n모델 크기 비교:")
    for format_name, size_mb in sizes.items():
        print(f"  {format_name}: {size_mb:.2f} MB")
    
    return sizes


def main():
    parser = argparse.ArgumentParser(description='모델 내보내기 및 변환')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='PyTorch 체크포인트 경로')
    parser.add_argument('--config', type=str, required=True,
                       help='모델 설정 파일 경로')
    parser.add_argument('--output-dir', type=str, default='./exported_models',
                       help='출력 디렉토리')
    parser.add_argument('--format', type=str, nargs='+',
                       choices=['onnx', 'torchscript', 'quantized'],
                       default=['onnx'],
                       help='내보낼 포맷')
    parser.add_argument('--opset-version', type=int, default=11,
                       help='ONNX opset 버전')
    parser.add_argument('--optimize', action='store_true',
                       help='ONNX 모델 최적화')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 모델 로드 (실제 구현에서는 config 기반으로 모델 생성)
    print("⚠ 모델 로드 기능은 실제 모델 구조에 맞게 구현해야 합니다.")
    print("  예시 코드만 제공합니다.")
    
    # 더미 입력 생성 (실제로는 모델에 맞게 조정)
    dummy_input = {
        'img': torch.randn(1, 3, 800, 1200),
        'img_metas': None
    }
    
    # 모델 내보내기
    if 'onnx' in args.format:
        onnx_path = output_dir / 'model.onnx'
        print("\n[ONNX 내보내기]")
        # export_to_onnx(model, dummy_input, str(onnx_path), args.opset_version)
        if args.optimize:
            optimized_path = output_dir / 'model_optimized.onnx'
            # optimize_onnx_model(str(onnx_path), str(optimized_path))
    
    if 'torchscript' in args.format:
        torchscript_path = output_dir / 'model.pt'
        print("\n[TorchScript 내보내기]")
        # export_to_torchscript(model, dummy_input, str(torchscript_path))
    
    if 'quantized' in args.format:
        quantized_path = output_dir / 'model_quantized.pt'
        print("\n[모델 양자화]")
        # quantize_model(model, dummy_input, str(quantized_path))


if __name__ == '__main__':
    main()

