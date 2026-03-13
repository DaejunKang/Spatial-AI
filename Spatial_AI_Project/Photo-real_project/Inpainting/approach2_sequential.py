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
    - final_output/rgb/: 동적 객체가 제거된 최종 이미지
    - final_output/depth/: Composited dense depth maps (uint16, mm)
    - final_output/confidence/: Confidence maps (uint8, 0~255)
    - final_output/method_log/: 프레임별 추정 방법 기록 (JSON)
    - final_output/point_cloud/: 정적 포인트 클라우드 (PLY)
    - final_inpainted/: 하위 호환 (rgb/ 복사)
"""

import os
import sys
import argparse
from pathlib import Path

# Inpainting 디렉토리를 sys.path에 추가하여 step 모듈 import 가능하게 함
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


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
        
        # 중간 출력 디렉토리
        self.step1_dir = self.data_root / 'step1_warped'
        self.step2_dir = self.data_root / 'step2_depth_guide'
        self.step3_dir = self.data_root / 'step3_final_inpainted'
        self.step3_depth_dir = self.data_root / 'step3_depth'
        self.step3_conf_dir = self.data_root / 'step3_confidence'
        self.step3_method_dir = self.data_root / 'step3_method_log'

        # 최종 출력 디렉토리 (구조화)
        self.final_output_dir = self.data_root / 'final_output'
        self.final_rgb_dir = self.final_output_dir / 'rgb'
        self.final_depth_dir = self.final_output_dir / 'depth'
        self.final_conf_dir = self.final_output_dir / 'confidence'
        self.final_method_dir = self.final_output_dir / 'method_log'
        self.final_pc_dir = self.final_output_dir / 'point_cloud'

        # 하위 호환 디렉토리
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
        from step1_temporal_accumulation import TemporalStaticAccumulator
        
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
        from step2_geometric_guide import GeometricGuideGenerator
        
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
        from step3_final_inpainting import run_step3
        
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
        최종 출력 디렉토리(final_output/)로 조립.

        final_output/
          ├── rgb/          ← step3_final_inpainted/
          ├── depth/        ← step3_depth/
          ├── confidence/   ← step3_confidence/
          ├── method_log/   ← step3_method_log/
          └── point_cloud/  ← step1_warped/accumulated_static.ply

        하위 호환: final_inpainted/ = rgb 복사
        """
        import shutil

        # final_output/ 서브 디렉토리 생성
        for d in [self.final_rgb_dir, self.final_depth_dir,
                  self.final_conf_dir, self.final_method_dir, self.final_pc_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.step3_dir.exists():
            raise FileNotFoundError(f"Step 3 output directory not found: {self.step3_dir}")

        # RGB 복사
        rgb_files = list(self.step3_dir.glob('*.jpg')) + list(self.step3_dir.glob('*.png'))
        for src in rgb_files:
            shutil.copy2(src, self.final_rgb_dir / src.name)
            shutil.copy2(src, self.output_dir / src.name)  # 하위 호환
        print(f"  RGB: {len(rgb_files)} files → {self.final_rgb_dir}")

        # Depth 복사
        depth_count = 0
        if self.step3_depth_dir.exists():
            for src in self.step3_depth_dir.glob('*.png'):
                shutil.copy2(src, self.final_depth_dir / src.name)
                depth_count += 1
        print(f"  Depth: {depth_count} files → {self.final_depth_dir}")

        # Confidence 복사
        conf_count = 0
        if self.step3_conf_dir.exists():
            for src in self.step3_conf_dir.glob('*.png'):
                shutil.copy2(src, self.final_conf_dir / src.name)
                conf_count += 1
        print(f"  Confidence: {conf_count} files → {self.final_conf_dir}")

        # Method Log 복사
        log_count = 0
        if self.step3_method_dir.exists():
            for src in self.step3_method_dir.glob('*.json'):
                shutil.copy2(src, self.final_method_dir / src.name)
                log_count += 1
        print(f"  Method Log: {log_count} files → {self.final_method_dir}")

        # Point Cloud 복사
        ply_src = self.step1_dir / 'accumulated_static.ply'
        if ply_src.exists():
            shutil.copy2(ply_src, self.final_pc_dir / 'accumulated_static.ply')
            print(f"  Point Cloud: {ply_src.name} → {self.final_pc_dir}")
        else:
            print(f"  Point Cloud: not found ({ply_src})")

        print(f"\n  Final output: {self.final_output_dir}")
        print(f"  Legacy compat: {self.output_dir} ({len(rgb_files)} files)")


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
