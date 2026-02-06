"""
Inpainting Approach 2: Sequential Multi-Stage Pipeline

시계열 누적 → 기하학적 가이드 → AI 생성 기반 순차적 인페인팅
3단계로 구성된 점진적 배경 복원 파이프라인

Workflow:
Step 1: Temporal Accumulation (시계열 누적)
        - 여러 프레임의 정적 배경을 3D로 누적
        - 다시 투영하여 동적 객체 영역 채우기

Step 2: Geometric Guide Generation (기하학적 가이드 생성)
        - LiDAR depth 또는 평면 추정 활용
        - Step 1에서 못 채운 구멍의 기하학적 힌트 생성

Step 3: Final Inpainting (최종 인페인팅)
        - Stable Diffusion + ControlNet
        - 고품질 AI 기반 생성

Input:
    - images/: 원본 이미지
    - masks/: 동적 객체 마스크
    - poses/: 카메라 포즈
    - depth_maps/: (선택) LiDAR depth

Output:
    - final_inpainted/: 동적 객체가 제거된 최종 이미지
"""

import os
import sys
import argparse
from pathlib import Path


class SequentialInpainter:
    """
    순차적 3단계 인페인팅 파이프라인
    
    기존 step1, step2, step3를 순차적으로 실행하는 Wrapper
    """
    
    def __init__(self, data_root, 
                 voxel_size=0.05, 
                 sample_interval=5,
                 use_lidar=True,
                 ground_ratio=0.6,
                 lora_path=None):
        """
        Args:
            data_root: NRE 포맷 데이터 디렉토리
            voxel_size: Voxel downsampling 크기 (Step 1)
            sample_interval: Forward pass 샘플링 간격 (Step 1)
            use_lidar: LiDAR depth 사용 여부 (Step 2)
            ground_ratio: 바닥 평면 추정 비율 (Step 2)
            lora_path: LoRA 가중치 경로 (Step 3)
        """
        self.data_root = Path(data_root)
        self.voxel_size = voxel_size
        self.sample_interval = sample_interval
        self.use_lidar = use_lidar
        self.ground_ratio = ground_ratio
        self.lora_path = lora_path
        
        # 스크립트 경로
        script_dir = Path(__file__).parent
        self.step1_script = script_dir / 'step1_temporal_accumulation.py'
        self.step2_script = script_dir / 'step2_geometric_guide.py'
        self.step3_script = script_dir / 'step3_final_inpainting.py'
        
        # 중간 출력 디렉토리
        self.step1_dir = self.data_root / 'step1_warped'
        self.step2_dir = self.data_root / 'step2_depth_guide'
        self.step3_dir = self.data_root / 'step3_final_inpainted'
        
        # 최종 출력 디렉토리 (Approach 1과 동일)
        self.output_dir = self.data_root / 'final_inpainted'
        
        print(f"[SequentialInpainter] Initialized")
        print(f"  Data root: {self.data_root}")
        print(f"  Voxel size: {self.voxel_size}")
        print(f"  Sample interval: {self.sample_interval}")
        print(f"  Use LiDAR: {self.use_lidar}")
        print(f"  Ground ratio: {self.ground_ratio}")
    
    def run(self):
        """전체 파이프라인 실행"""
        print("\n" + "="*70)
        print(">>> [Approach 2] Sequential Multi-Stage Inpainting Started")
        print("="*70)
        
        # Step 1: Temporal Accumulation
        print("\n[Step 1/3] Temporal Accumulation (시계열 누적)")
        print("-"*70)
        self._run_step1()
        
        # Step 2: Geometric Guide Generation
        print("\n[Step 2/3] Geometric Guide Generation (기하학적 가이드 생성)")
        print("-"*70)
        self._run_step2()
        
        # Step 3: Final Inpainting
        print("\n[Step 3/3] Final Inpainting (AI 기반 최종 생성)")
        print("-"*70)
        self._run_step3()
        
        # Copy to final output directory
        print("\n[Finalization] Copying to final output directory...")
        print("-"*70)
        self._finalize_output()
        
        print("\n" + "="*70)
        print(">>> [Approach 2] Sequential Inpainting Complete!")
        print(f"  Output: {self.output_dir}")
        print("="*70)
    
    def _run_step1(self):
        """
        Step 1: Temporal Accumulation 실행
        
        시계열 정보를 활용한 배경 누적 및 투영
        직접 import하여 호출 (subprocess 대비 에러 전파 개선)
        """
        try:
            from step1_temporal_accumulation import TemporalStaticAccumulator
        except ImportError:
            # 패키지 경로 문제 시 상대 import 시도
            from .step1_temporal_accumulation import TemporalStaticAccumulator
        
        try:
            accumulator = TemporalStaticAccumulator(
                data_root=str(self.data_root),
                voxel_size=self.voxel_size,
                sample_interval=self.sample_interval
            )
            accumulator.run()
            
            print("Step 1 completed successfully")
            
            # 출력 확인
            if self.step1_dir.exists():
                output_count = len(list(self.step1_dir.glob('*.png')))
                print(f"  Generated {output_count} warped images")
            
        except Exception as e:
            print(f"Step 1 failed: {e}")
            raise
    
    def _run_step2(self):
        """
        Step 2: Geometric Guide Generation 실행
        
        LiDAR depth 또는 평면 추정으로 기하학적 가이드 생성
        직접 import하여 호출 (subprocess 대비 에러 전파 개선)
        """
        try:
            from step2_geometric_guide import GeometricGuideGenerator
        except ImportError:
            from .step2_geometric_guide import GeometricGuideGenerator
        
        try:
            generator = GeometricGuideGenerator(
                data_root=str(self.data_root),
                use_lidar_depth=self.use_lidar,
                ground_region_ratio=self.ground_ratio
            )
            generator.run()
            
            print("Step 2 completed successfully")
            
            # 출력 확인
            if self.step2_dir.exists():
                output_count = len(list(self.step2_dir.glob('*.png')))
                print(f"  Generated {output_count} depth guides")
            
        except Exception as e:
            print(f"Step 2 failed: {e}")
            raise
    
    def _run_step3(self):
        """
        Step 3: Final Inpainting 실행
        
        Stable Diffusion + ControlNet 기반 최종 생성
        직접 import하여 호출 (subprocess 대비 에러 전파 개선)
        """
        try:
            from step3_final_inpainting import run_step3
        except ImportError:
            from .step3_final_inpainting import run_step3
        
        try:
            run_step3(
                data_root=str(self.data_root),
                lora_path=str(self.lora_path) if self.lora_path else None
            )
            
            print("Step 3 completed successfully")
            
            # 출력 확인
            if self.step3_dir.exists():
                output_count = len(list(self.step3_dir.glob('*.jpg')))
                output_count += len(list(self.step3_dir.glob('*.png')))
                print(f"  Generated {output_count} final images")
            
        except Exception as e:
            print(f"Step 3 failed: {e}")
            raise
    
    def _finalize_output(self):
        """
        최종 출력 디렉토리로 복사
        
        Approach 1과 동일한 경로에 결과 저장
        (final_inpainted/)
        """
        import shutil
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 3 결과를 final_inpainted/로 복사
        if self.step3_dir.exists():
            output_files = list(self.step3_dir.glob('*.jpg'))
            
            for src_file in output_files:
                dst_file = self.output_dir / src_file.name
                shutil.copy2(src_file, dst_file)
            
            print(f"✓ Copied {len(output_files)} images to {self.output_dir}")
        else:
            print("✗ Step 3 output not found. Cannot finalize.")
            raise FileNotFoundError(f"Step 3 output directory not found: {self.step3_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Inpainting Approach 2: Sequential Multi-Stage Pipeline"
    )
    parser.add_argument(
        'data_root',
        type=str,
        help='Path to NRE format data directory'
    )
    
    # Step 1 Parameters
    parser.add_argument(
        '--voxel_size',
        type=float,
        default=0.05,
        help='Voxel downsampling size in meters (default: 0.05)'
    )
    parser.add_argument(
        '--sample_interval',
        type=int,
        default=5,
        help='Forward pass sampling interval (default: 5)'
    )
    
    # Step 2 Parameters
    parser.add_argument(
        '--no_lidar',
        action='store_true',
        help='Do not use LiDAR depth, generate pseudo depth instead'
    )
    parser.add_argument(
        '--ground_ratio',
        type=float,
        default=0.6,
        help='Ground plane estimation ratio (default: 0.6)'
    )
    
    # Step 3 Parameters
    parser.add_argument(
        '--lora_path',
        type=str,
        default=None,
        help='Path to trained LoRA weights (.safetensors)'
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_root):
        print(f"Error: Data directory not found: {args.data_root}")
        return
    
    # Run
    inpainter = SequentialInpainter(
        data_root=args.data_root,
        voxel_size=args.voxel_size,
        sample_interval=args.sample_interval,
        use_lidar=not args.no_lidar,
        ground_ratio=args.ground_ratio,
        lora_path=args.lora_path
    )
    inpainter.run()


if __name__ == '__main__':
    main()
