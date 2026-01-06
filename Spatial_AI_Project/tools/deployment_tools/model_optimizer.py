# -*- coding: utf-8 -*-
"""
모델 최적화 도구
- 모델 양자화
- 모델 압축
- ONNX 변환
- TensorRT 변환
"""

import argparse
import torch
from pathlib import Path
from typing import Dict, Optional
import onnx
import onnxruntime as ort


class ModelOptimizer:
    """모델 최적화 클래스"""
    
    def __init__(self):
        pass
    
    def quantize_model(self,
                      model: torch.nn.Module,
                      calibration_data,
                      method: str = 'dynamic') -> torch.nn.Module:
        """
        모델을 양자화합니다.
        
        Args:
            model: PyTorch 모델
            calibration_data: 캘리브레이션 데이터
            method: 양자화 방법 ('dynamic', 'static', 'qat')
            
        Returns:
            양자화된 모델
        """
        if method == 'dynamic':
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        elif method == 'static':
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            # 캘리브레이션 (간단한 예시)
            with torch.no_grad():
                for data in calibration_data:
                    _ = model(data)
            torch.quantization.convert(model, inplace=True)
            quantized_model = model
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        return quantized_model
    
    def export_to_onnx(self,
                      model: torch.nn.Module,
                      output_path: str,
                      input_shape: tuple,
                      input_names: Optional[list] = None,
                      output_names: Optional[list] = None,
                      opset_version: int = 11) -> None:
        """
        모델을 ONNX 형식으로 내보냅니다.
        
        Args:
            model: PyTorch 모델
            output_path: 출력 경로
            input_shape: 입력 텐서 형태
            input_names: 입력 이름 리스트
            output_names: 출력 이름 리스트
            opset_version: ONNX opset 버전
        """
        model.eval()
        
        dummy_input = torch.randn(input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=input_names or ['input'],
            output_names=output_names or ['output'],
            opset_version=opset_version,
            export_params=True,
            do_constant_folding=True,
            verbose=False
        )
        
        print(f"Model exported to ONNX: {output_path}")
        
        # ONNX 모델 검증
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed")
    
    def optimize_onnx(self,
                     onnx_path: str,
                     output_path: str) -> None:
        """
        ONNX 모델을 최적화합니다.
        
        Args:
            onnx_path: 입력 ONNX 파일 경로
            output_path: 출력 경로
        """
        from onnxsim import simplify
        
        model = onnx.load(onnx_path)
        model_simp, check = simplify(model)
        
        if check:
            onnx.save(model_simp, output_path)
            print(f"Optimized ONNX model saved to {output_path}")
        else:
            print("Model could not be simplified")
            onnx.save(model, output_path)
    
    def test_onnx_model(self,
                       onnx_path: str,
                       input_data: torch.Tensor) -> Dict:
        """
        ONNX 모델을 테스트합니다.
        
        Args:
            onnx_path: ONNX 파일 경로
            input_data: 입력 데이터
            
        Returns:
            테스트 결과
        """
        session = ort.InferenceSession(onnx_path)
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        input_numpy = input_data.numpy()
        outputs = session.run([output_name], {input_name: input_numpy})
        
        return {
            'output_shape': outputs[0].shape,
            'output_dtype': str(outputs[0].dtype)
        }
    
    def prune_model(self,
                   model: torch.nn.Module,
                   amount: float = 0.2) -> torch.nn.Module:
        """
        모델을 프루닝합니다.
        
        Args:
            model: PyTorch 모델
            amount: 프루닝 비율
            
        Returns:
            프루닝된 모델
        """
        import torch.nn.utils.prune as prune
        
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=amount)
                prune.remove(module, 'weight')
        
        return model


def main():
    parser = argparse.ArgumentParser(description='모델 최적화 도구')
    subparsers = parser.add_subparsers(dest='command', help='명령어')
    
    # ONNX 변환
    onnx_parser = subparsers.add_parser('onnx', help='ONNX 변환')
    onnx_parser.add_argument('--model', type=str, required=True, help='모델 파일 경로')
    onnx_parser.add_argument('--output', type=str, required=True, help='출력 경로')
    onnx_parser.add_argument('--input-shape', type=int, nargs='+', required=True,
                           help='입력 형태 (예: 1 3 224 224)')
    onnx_parser.add_argument('--opset', type=int, default=11, help='ONNX opset 버전')
    
    # ONNX 최적화
    optimize_parser = subparsers.add_parser('optimize', help='ONNX 최적화')
    optimize_parser.add_argument('--onnx', type=str, required=True, help='ONNX 파일 경로')
    optimize_parser.add_argument('--output', type=str, required=True, help='출력 경로')
    
    # ONNX 테스트
    test_parser = subparsers.add_parser('test', help='ONNX 모델 테스트')
    test_parser.add_argument('--onnx', type=str, required=True, help='ONNX 파일 경로')
    
    args = parser.parse_args()
    
    optimizer = ModelOptimizer()
    
    if args.command == 'onnx':
        # 모델 로드
        checkpoint = torch.load(args.model, map_location='cpu')
        if 'state_dict' in checkpoint:
            # 체크포인트에서 모델 로드하는 경우
            # 실제로는 모델 클래스를 import해서 사용해야 함
            print("Warning: Model class required for ONNX export")
            print("Please implement model loading in the script")
        else:
            model = checkpoint
        
        input_shape = tuple(args.input_shape)
        optimizer.export_to_onnx(model, args.output, input_shape, opset_version=args.opset)
    
    elif args.command == 'optimize':
        optimizer.optimize_onnx(args.onnx, args.output)
    
    elif args.command == 'test':
        # 더미 입력 생성
        onnx_model = onnx.load(args.onnx)
        input_shape = [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        dummy_input = torch.randn(*input_shape)
        
        result = optimizer.test_onnx_model(args.onnx, dummy_input)
        print(f"Test result: {result}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

