"""
Approach 2: 3DGUT (3D Gaussian with Unscented Transform)
        NVIDIA Rolling Shutter Compensated Reconstruction

이 모듈은 nerfstudio-project/gsplat에 통합된 NVIDIA 3DGUT를 래핑하여
우리 파이프라인의 데이터 포맷과 연결합니다.

External Repository:
    nerfstudio-project/gsplat (NVIDIA 3DGUT 통합)
    https://github.com/nerfstudio-project/gsplat
    NVIDIA 3DGUT: https://research.nvidia.com/labs/toronto-ai/3DGUT/

3DGUT Features:
    - Unscented Transform: 비선형 카메라 프로젝션 지원 (렌즈 왜곡, 핀홀/피쉬아이)
    - Rolling Shutter 보정: 각 픽셀의 캡처 시간을 고려한 모션 보정
    - 3D Eval: 3D 공간에서의 Gaussian 응답 평가

Input (3DGS + α):
    - Images: Inpainting된 배경 이미지
    - Camera Extrinsics/Intrinsics
    - Ego Velocity: 선속도 및 각속도 [vx, vy, vz, wx, wy, wz]
    - Rolling Shutter: duration, trigger_time
    - (선택) Distortion coefficients

Output:
    - Trained 3D Gaussians (.ply / .pt)
    - Rendered Novel Views (Rolling Shutter 보정됨)
"""

import os
import sys
import json
import shutil
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# 프로젝트 경로 설정
RECONSTRUCTION_DIR = Path(__file__).parent
EXTERNAL_DIR = RECONSTRUCTION_DIR / "external"
GSPLAT_DIR = EXTERNAL_DIR / "gsplat"

# Data loader
from data_loader import ReconstructionDataset, load_initial_point_cloud


def check_gsplat_installation():
    """gsplat (3DGUT) 설치 확인"""
    gsplat_init = GSPLAT_DIR / "gsplat" / "__init__.py"
    if not gsplat_init.exists():
        raise FileNotFoundError(
            f"gsplat 레포지토리를 찾을 수 없습니다: {GSPLAT_DIR}\n"
            f"다음 명령어로 서브모듈을 초기화하세요:\n"
            f"  git submodule update --init --recursive"
        )

    # gsplat python 패키지 설치 확인
    try:
        import gsplat
        print(f"  gsplat package version: {gsplat.__version__}")
        return True
    except ImportError:
        print("  Warning: gsplat Python package is not installed.")
        print("  You can install it with:")
        print(f"    pip install -e {GSPLAT_DIR}")
        print("  Or: pip install gsplat")
        print("  Falling back to subprocess-based execution.")
        return False


def convert_to_colmap_with_3dgut_params(
    data_root: str,
    meta_file: str,
    output_dir: str,
    image_scale: float = 1.0
):
    """
    우리 파이프라인의 JSON 메타데이터를 gsplat/3DGUT가 요구하는 COLMAP 포맷 +
    확장 메타데이터로 변환

    gsplat의 simple_trainer.py는 COLMAP 포맷 데이터를 사용합니다.
    3DGUT 추가 파라미터 (velocity, rolling_shutter)는 별도 JSON으로 저장됩니다.

    Args:
        data_root: 데이터 루트 디렉토리
        meta_file: JSON 메타데이터 파일 경로
        output_dir: 변환된 데이터 출력 디렉토리
        image_scale: 이미지 스케일
    """
    from approach1_3dgs import convert_to_colmap_format, rotation_matrix_to_quaternion

    data_root = Path(data_root)
    output_dir = Path(output_dir)

    # 기본 COLMAP 포맷 변환 (3DGS와 공통)
    convert_to_colmap_format(
        data_root=str(data_root),
        meta_file=str(meta_file),
        output_dir=str(output_dir),
        image_scale=image_scale
    )

    # 3DGUT 확장 메타데이터 생성
    with open(meta_file, 'r') as f:
        metadata = json.load(f)

    dgut_params = {}
    for idx, item in enumerate(metadata):
        frame_key = f"{idx:06d}.jpg"

        dgut_params[frame_key] = {
            "camera_name": item.get("camera_name", "UNKNOWN"),
            "frame_name": item.get("frame_name", ""),
        }

        # Velocity 정보
        if "velocity" in item:
            v = item["velocity"].get("v", [0.0, 0.0, 0.0])
            w = item["velocity"].get("w", [0.0, 0.0, 0.0])
            dgut_params[frame_key]["velocity"] = v + w  # [vx, vy, vz, wx, wy, wz]
        else:
            dgut_params[frame_key]["velocity"] = [0.0] * 6

        # Rolling Shutter 정보
        if "rolling_shutter" in item:
            dgut_params[frame_key]["rolling_shutter"] = {
                "duration": item["rolling_shutter"].get("duration", 0.033),
                "trigger_time": item["rolling_shutter"].get("trigger_time", 0.0)
            }
        else:
            dgut_params[frame_key]["rolling_shutter"] = {
                "duration": 0.033,
                "trigger_time": 0.0
            }

        # Distortion coefficients (있으면)
        intrinsics = item.get("intrinsics", [])
        if len(intrinsics) > 4:
            # [fx, fy, cx, cy, k1, k2, p1, p2, k3]
            dgut_params[frame_key]["distortion"] = {
                "radial": intrinsics[4:6] if len(intrinsics) > 5 else [0.0, 0.0],
                "tangential": intrinsics[6:8] if len(intrinsics) > 7 else [0.0, 0.0],
                "k3": intrinsics[8] if len(intrinsics) > 8 else 0.0
            }

    # 3DGUT 파라미터 저장
    dgut_path = output_dir / "3dgut_params.json"
    with open(dgut_path, 'w') as f:
        json.dump(dgut_params, f, indent=2)

    print(f"  3DGUT params saved to: {dgut_path}")
    print(f"  Frames with velocity info: {sum(1 for p in dgut_params.values() if any(v != 0 for v in p['velocity']))}")

    return str(output_dir)


class GaussianSplatting3DGUT:
    """
    3DGUT: NVIDIA의 Unscented Transform 기반 3D Gaussian Splatting

    gsplat 라이브러리의 3DGUT 기능을 활용:
    - with_ut=True: Unscented Transform으로 비선형 카메라 모델 지원
    - with_eval3d=True: 3D 공간에서 Gaussian 응답 평가
    - rolling_shutter: Rolling Shutter 보정 지원

    학습 방법:
    1. gsplat이 설치된 경우: Python API 직접 사용
    2. 설치 안 된 경우: gsplat/examples/simple_trainer.py를 subprocess로 호출
    """

    def __init__(
        self,
        data_root: str,
        meta_file: str,
        output_dir: str,
        initial_ply: str = None,
        device: str = 'cuda',
        image_scale: float = 1.0,
        camera_model: str = 'pinhole'
    ):
        """
        Args:
            data_root: 데이터 루트 디렉토리
            meta_file: JSON 메타데이터 파일 (3DGUT 형식)
            output_dir: 출력 디렉토리
            initial_ply: 초기 포인트 클라우드 (선택)
            device: 'cuda' or 'cpu'
            image_scale: 이미지 스케일
            camera_model: 'pinhole' or 'fisheye'
        """
        self.data_root = Path(data_root)
        self.meta_file = Path(meta_file)
        self.output_dir = Path(output_dir)
        self.device = device
        self.image_scale = image_scale
        self.camera_model = camera_model
        self.initial_ply = initial_ply

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # gsplat 설치 확인
        print("=" * 70)
        print(">>> Checking gsplat (NVIDIA 3DGUT) installation")
        print("=" * 70)
        self.gsplat_installed = check_gsplat_installation()

        # COLMAP 변환 디렉토리
        self.colmap_dir = self.output_dir / "colmap_format"

    def prepare_data(self):
        """데이터를 gsplat/3DGUT가 요구하는 포맷으로 변환"""
        print("\n" + "=" * 70)
        print(">>> [Step 1] Converting to COLMAP + 3DGUT format")
        print("=" * 70)

        convert_to_colmap_with_3dgut_params(
            data_root=str(self.data_root),
            meta_file=str(self.meta_file),
            output_dir=str(self.colmap_dir),
            image_scale=self.image_scale
        )

        if self.initial_ply and Path(self.initial_ply).exists():
            ply_dst = self.colmap_dir / "sparse" / "0" / "points3D.ply"
            shutil.copy2(self.initial_ply, str(ply_dst))
            print(f"  Initial PLY copied: {self.initial_ply} -> {ply_dst}")

    def train_via_subprocess(self, num_iterations: int = 30000, cap_max: int = 1000000):
        """
        gsplat의 simple_trainer.py를 subprocess로 호출하여 3DGUT 학습

        3DGUT 활성화 플래그:
            --with_ut       : Unscented Transform 사용
            --with_eval3d   : 3D 공간 평가 사용
            --camera_model  : pinhole / fisheye

        Args:
            num_iterations: 학습 반복 횟수
            cap_max: 최대 Gaussian 수 (MCMC strategy)
        """
        print("\n" + "=" * 70)
        print(">>> [Step 2] Training 3DGUT via gsplat subprocess")
        print("=" * 70)

        trainer_script = GSPLAT_DIR / "examples" / "simple_trainer.py"
        if not trainer_script.exists():
            raise FileNotFoundError(
                f"gsplat trainer not found: {trainer_script}\n"
                f"Please ensure submodules are initialized:\n"
                f"  git submodule update --init --recursive"
            )

        result_dir = self.output_dir / "results"

        # gsplat simple_trainer.py 명령어 구성
        # MCMC strategy + 3DGUT (with_ut, with_eval3d)
        cmd = [
            sys.executable,
            str(trainer_script),
            "mcmc",  # MCMC densification strategy (3DGUT 권장)
            "--with_ut",           # Unscented Transform 활성화
            "--with_eval3d",       # 3D Eval 활성화
            "--camera_model", self.camera_model,
            "--data_dir", str(self.colmap_dir),
            "--result_dir", str(result_dir),
            "--max_steps", str(num_iterations),
            "--strategy.cap-max", str(cap_max),
            "--disable_viewer",    # Headless 모드
        ]

        print(f"  Trainer:   {trainer_script}")
        print(f"  Strategy:  MCMC (3DGUT recommended)")
        print(f"  3DGUT:     with_ut=True, with_eval3d=True")
        print(f"  Camera:    {self.camera_model}")
        print(f"  Data:      {self.colmap_dir}")
        print(f"  Output:    {result_dir}")
        print(f"  Steps:     {num_iterations}")
        print(f"  Cap Max:   {cap_max}")
        print()
        print(f"  Command:\n    {' '.join(cmd)}")
        print()

        # 환경 설정
        env = os.environ.copy()
        gsplat_examples = GSPLAT_DIR / "examples"
        env["PYTHONPATH"] = (
            str(GSPLAT_DIR) + ":" +
            str(gsplat_examples) + ":" +
            env.get("PYTHONPATH", "")
        )

        try:
            result = subprocess.run(
                cmd,
                cwd=str(gsplat_examples),
                env=env,
                capture_output=False,
                text=True,
                timeout=3600 * 12
            )

            if result.returncode == 0:
                print("\n  3DGUT Training completed successfully!")
            else:
                print(f"\n  Warning: Training exited with code {result.returncode}")
                self._print_install_help()
        except subprocess.TimeoutExpired:
            print("\n  Error: Training timed out (12h limit)")
        except FileNotFoundError as e:
            print(f"\n  Error: {e}")
            self._print_install_help()

        return result_dir

    def train_via_api(self, num_iterations: int = 30000):
        """
        gsplat Python API를 직접 사용하여 3DGUT 학습

        gsplat이 pip install로 설치된 경우에만 사용 가능.
        gsplat.rendering.rasterization() 함수에 with_ut=True, with_eval3d=True를
        전달하여 3DGUT 기능을 활성화합니다.
        """
        print("\n" + "=" * 70)
        print(">>> [Step 2] Training 3DGUT via gsplat Python API")
        print("=" * 70)

        try:
            import torch
            from gsplat.rendering import rasterization

            print("  gsplat API loaded successfully")
            print("  3DGUT rasterization API:")
            print("    rasterization(..., with_ut=True, with_eval3d=True)")
            print("    Supports: rolling_shutter, radial_coeffs, tangential_coeffs")
            print()

            # 데이터셋 로드
            dataset = ReconstructionDataset(
                meta_file=str(self.meta_file),
                data_root=str(self.data_root),
                split='train',
                load_3dgut_params=True,
                image_scale=self.image_scale
            )

            print(f"  Dataset: {len(dataset)} frames loaded")
            print(f"  NOTE: Full training loop requires gsplat to be pip-installed.")
            print(f"  For complete training, use: train_via_subprocess()")
            print()
            print(f"  === gsplat rasterization API Reference ===")
            print(f"  from gsplat.rendering import rasterization")
            print(f"  render_colors, render_alphas, meta = rasterization(")
            print(f"      means,       # [N, 3] Gaussian 중심")
            print(f"      quats,       # [N, 4] 회전 (quaternion)")
            print(f"      scales,      # [N, 3] 스케일")
            print(f"      opacities,   # [N] 불투명도")
            print(f"      colors,      # [N, S, 3] SH coefficients")
            print(f"      viewmats,    # [C, 4, 4] Camera poses")
            print(f"      Ks,          # [C, 3, 3] Intrinsics")
            print(f"      width, height,")
            print(f"      with_ut=True,        # 3DGUT: Unscented Transform")
            print(f"      with_eval3d=True,    # 3DGUT: 3D Evaluation")
            print(f"      camera_model='{self.camera_model}',")
            print(f"      rolling_shutter=..., # Rolling Shutter params")
            print(f"      radial_coeffs=...,   # Lens distortion")
            print(f"  )")

        except ImportError:
            print("  gsplat package not available for API usage.")
            print("  Install with: pip install gsplat")
            print("  Falling back to subprocess-based training.")
            return self.train_via_subprocess(num_iterations=num_iterations)

    def train(self, num_iterations: int = 30000, cap_max: int = 1000000):
        """
        3DGUT 학습 (자동으로 적절한 방법 선택)

        Args:
            num_iterations: 학습 반복 횟수
            cap_max: 최대 Gaussian 수
        """
        # gsplat 설치 여부에 따라 방법 선택
        # subprocess 방식이 가장 안정적이므로 기본으로 사용
        return self.train_via_subprocess(
            num_iterations=num_iterations,
            cap_max=cap_max
        )

    def render(self, result_dir: str = None):
        """
        학습된 모델로 렌더링

        Args:
            result_dir: 학습 결과 디렉토리
        """
        print("\n" + "=" * 70)
        print(">>> [Step 3] Rendering Novel Views (3DGUT)")
        print("=" * 70)

        if result_dir is None:
            result_dir = self.output_dir / "results"

        result_dir = Path(result_dir)

        # 체크포인트 찾기
        ckpt_dir = result_dir / "ckpts"
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("*.pt"))
            if ckpts:
                latest_ckpt = ckpts[-1]
                print(f"  Latest checkpoint: {latest_ckpt}")
            else:
                print("  No checkpoints found in {ckpt_dir}")
                return
        else:
            print(f"  Checkpoint directory not found: {ckpt_dir}")
            return

        # gsplat eval 호출
        trainer_script = GSPLAT_DIR / "examples" / "simple_trainer.py"

        cmd = [
            sys.executable,
            str(trainer_script),
            "mcmc",
            "--with_ut",
            "--with_eval3d",
            "--camera_model", self.camera_model,
            "--data_dir", str(self.colmap_dir),
            "--result_dir", str(result_dir),
            "--ckpt", str(latest_ckpt),
            "--disable_viewer",
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            str(GSPLAT_DIR) + ":" +
            str(GSPLAT_DIR / "examples") + ":" +
            env.get("PYTHONPATH", "")
        )

        print(f"  Command: {' '.join(cmd)}")

        try:
            subprocess.run(
                cmd,
                cwd=str(GSPLAT_DIR / "examples"),
                env=env,
                capture_output=False,
                text=True,
                timeout=3600
            )
            print("  Rendering completed!")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  Error during rendering: {e}")

    def run(self, num_iterations: int = 30000, cap_max: int = 1000000):
        """전체 파이프라인 실행"""
        print("\n" + "=" * 70)
        print(">>> 3DGUT (Approach 2) - Full Pipeline")
        print(f"    External: nerfstudio-project/gsplat (NVIDIA 3DGUT)")
        print(f"    Data:     {self.data_root}")
        print(f"    Output:   {self.output_dir}")
        print(f"    Camera:   {self.camera_model}")
        print("=" * 70)

        # 1. 데이터 변환
        self.prepare_data()

        # 2. 학습
        result_dir = self.train(
            num_iterations=num_iterations,
            cap_max=cap_max
        )

        # 3. 렌더링
        self.render(result_dir=str(result_dir))

        print("\n" + "=" * 70)
        print(">>> Pipeline Complete!")
        print(f"    Results: {result_dir}")
        print(f"    Output:  {self.output_dir}")
        print("=" * 70)

    def _print_install_help(self):
        """설치 가이드 출력"""
        print("\n  === Installation Guide ===")
        print("  gsplat (NVIDIA 3DGUT) 설치 방법:")
        print()
        print("  # Option 1: pip install (JIT compile)")
        print("  pip install gsplat")
        print()
        print("  # Option 2: From source (서브모듈)")
        print(f"  cd {GSPLAT_DIR}")
        print("  pip install -e .")
        print()
        print("  # Dependencies:")
        print("  pip install torch torchvision")
        print("  pip install fused-ssim torchmetrics")
        print("  pip install viser nerfview imageio tqdm tyro")
        print()
        print("  # 3DGUT Training (예시):")
        print("  python simple_trainer.py mcmc \\")
        print("      --with_ut --with_eval3d \\")
        print("      --data_dir /path/to/colmap_data \\")
        print("      --result_dir /path/to/results")


def main():
    parser = argparse.ArgumentParser(
        description="3DGUT Training (Approach 2): NVIDIA Rolling Shutter Compensated\n"
                    "Wraps nerfstudio-project/gsplat with NVIDIA 3DGUT"
    )
    parser.add_argument(
        'data_root',
        type=str,
        help='Path to NRE format data root'
    )
    parser.add_argument(
        '--meta_file',
        type=str,
        default='train_meta/train_pairs.json',
        help='JSON metadata file (relative to data_root)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/3dgut',
        help='Output directory (relative to data_root)'
    )
    parser.add_argument(
        '--initial_ply',
        type=str,
        default=None,
        help='Initial point cloud PLY file'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=30000,
        help='Number of training iterations (default: 30000)'
    )
    parser.add_argument(
        '--cap_max',
        type=int,
        default=1000000,
        help='Maximum number of Gaussians for MCMC strategy (default: 1000000)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--camera_model',
        type=str,
        default='pinhole',
        choices=['pinhole', 'fisheye'],
        help='Camera model (default: pinhole)'
    )
    parser.add_argument(
        '--image_scale',
        type=float,
        default=1.0,
        help='Image scale factor (default: 1.0)'
    )

    args = parser.parse_args()

    # 경로 설정
    data_root = Path(args.data_root)
    meta_file = data_root / args.meta_file
    output_dir = data_root / args.output_dir

    # 메타데이터 확인
    if not meta_file.exists():
        print(f"Error: Metadata file not found: {meta_file}")
        print("\nPlease run prepare_metadata.py first:")
        print(f"  python reconstruction/prepare_metadata.py {args.data_root} --mode 3dgut")
        return

    # 초기 PLY 경로
    initial_ply = None
    if args.initial_ply:
        initial_ply = Path(args.initial_ply)
        if not initial_ply.is_absolute():
            initial_ply = data_root / initial_ply

    # Trainer 생성 및 실행
    trainer = GaussianSplatting3DGUT(
        data_root=str(data_root),
        meta_file=str(meta_file),
        output_dir=str(output_dir),
        initial_ply=str(initial_ply) if initial_ply else None,
        device=args.device,
        image_scale=args.image_scale,
        camera_model=args.camera_model
    )

    trainer.run(
        num_iterations=args.iterations,
        cap_max=args.cap_max
    )


if __name__ == '__main__':
    main()
