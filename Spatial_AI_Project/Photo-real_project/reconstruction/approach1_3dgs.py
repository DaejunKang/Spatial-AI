"""
Approach 1: 3D Gaussian Splatting (3DGS) for Static Scene Reconstruction

이 모듈은 graphdeco-inria/gaussian-splatting 공식 레퍼런스 구현을 래핑하여
우리 파이프라인의 데이터 포맷과 연결합니다.

External Repository:
    graphdeco-inria/gaussian-splatting
    "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    https://github.com/graphdeco-inria/gaussian-splatting

Input:
    - Images: Inpainting된 배경 이미지
    - Camera Extrinsics: World-to-Camera 변환 행렬
    - Camera Intrinsics: 카메라 내부 파라미터
    - (선택) Initial Point Cloud: 초기화용 PLY

Output:
    - Trained 3D Gaussians (.ply)
    - Rendered Novel Views
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
GAUSSIAN_SPLATTING_DIR = EXTERNAL_DIR / "gaussian-splatting"

# Data loader
from data_loader import ReconstructionDataset, load_initial_point_cloud


def check_3dgs_installation():
    """3DGS 외부 모듈 설치 확인"""
    train_script = GAUSSIAN_SPLATTING_DIR / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(
            f"3DGS 레포지토리를 찾을 수 없습니다: {GAUSSIAN_SPLATTING_DIR}\n"
            f"다음 명령어로 서브모듈을 초기화하세요:\n"
            f"  git submodule update --init --recursive"
        )
    return True


def convert_to_colmap_format(
    data_root: str,
    meta_file: str,
    output_dir: str,
    image_scale: float = 1.0
):
    """
    우리 파이프라인의 JSON 메타데이터를 3DGS가 요구하는 COLMAP 포맷으로 변환

    3DGS는 다음 구조를 요구:
        output_dir/
        ├── images/           # 이미지 파일
        ├── sparse/
        │   └── 0/
        │       ├── cameras.bin (or .txt)
        │       ├── images.bin (or .txt)
        │       └── points3D.bin (or .txt)

    Args:
        data_root: 데이터 루트 디렉토리
        meta_file: JSON 메타데이터 파일 경로
        output_dir: COLMAP 포맷 출력 디렉토리
        image_scale: 이미지 스케일
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    # 메타데이터 로드
    with open(meta_file, 'r') as f:
        metadata = json.load(f)

    print(f"[convert_to_colmap_format] Converting {len(metadata)} frames")

    # 출력 디렉토리 생성
    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # --- cameras.txt ---
    # COLMAP camera format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # PINHOLE model: fx, fy, cx, cy
    cameras_txt = []
    images_txt = []

    # 카메라 ID 매핑 (unique intrinsics -> camera_id)
    camera_map = {}
    camera_id_counter = 1

    for idx, item in enumerate(metadata):
        # 이미지 복사/링크
        src_path = data_root / item['file_path']
        if not src_path.exists():
            print(f"  Warning: Image not found: {src_path}, skipping")
            continue

        dst_name = f"{idx:06d}.jpg"
        dst_path = images_dir / dst_name

        # 심볼릭 링크 생성 (디스크 절약)
        if not dst_path.exists():
            try:
                dst_path.symlink_to(src_path.resolve())
            except OSError:
                shutil.copy2(str(src_path), str(dst_path))

        # Intrinsics
        intrinsics = item['intrinsics']
        fx, fy, cx, cy = intrinsics[:4]
        width = item.get('width', 1920)
        height = item.get('height', 1280)

        # 스케일 적용
        if image_scale != 1.0:
            fx *= image_scale
            fy *= image_scale
            cx *= image_scale
            cy *= image_scale
            width = int(width * image_scale)
            height = int(height * image_scale)

        # 카메라 고유 키 (같은 intrinsics면 같은 camera_id)
        cam_key = f"{fx:.4f}_{fy:.4f}_{cx:.4f}_{cy:.4f}_{width}_{height}"
        if cam_key not in camera_map:
            camera_map[cam_key] = camera_id_counter
            # PINHOLE: fx, fy, cx, cy
            cameras_txt.append(
                f"{camera_id_counter} PINHOLE {width} {height} {fx} {fy} {cx} {cy}"
            )
            camera_id_counter += 1

        cam_id = camera_map[cam_key]

        # Extrinsic (4x4 flat -> rotation quaternion + translation)
        transform = np.array(item['transform_matrix']).reshape(4, 4)
        R = transform[:3, :3]
        t = transform[:3, 3]

        # Rotation matrix -> quaternion (COLMAP convention: qw, qx, qy, qz)
        quat = rotation_matrix_to_quaternion(R)

        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        image_id = idx + 1
        images_txt.append(
            f"{image_id} {quat[0]:.10f} {quat[1]:.10f} {quat[2]:.10f} {quat[3]:.10f} "
            f"{t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {cam_id} {dst_name}"
        )
        # COLMAP images.txt의 두 번째 줄 (2D points, 비워둠)
        images_txt.append("")

    # cameras.txt 저장
    with open(sparse_dir / "cameras.txt", 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras_txt)}\n")
        for line in cameras_txt:
            f.write(line + "\n")

    # images.txt 저장
    with open(sparse_dir / "images.txt", 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(cameras_txt)}\n")
        for line in images_txt:
            f.write(line + "\n")

    # points3D.txt (빈 파일 - 3DGS는 random init 가능)
    with open(sparse_dir / "points3D.txt", 'w') as f:
        f.write("# 3D point list (empty - using random or PLY initialization)\n")

    print(f"  COLMAP format saved to: {output_dir}")
    print(f"  Cameras: {len(cameras_txt)}")
    print(f"  Images: {len(images_txt) // 2}")

    return str(output_dir)


def rotation_matrix_to_quaternion(R):
    """
    Rotation matrix (3x3) -> Quaternion (qw, qx, qy, qz)
    COLMAP convention
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return np.array([qw, qx, qy, qz])


class GaussianSplattingTrainer:
    """
    3D Gaussian Splatting 학습 래퍼 클래스

    graphdeco-inria/gaussian-splatting 공식 구현을 래핑하여
    우리 파이프라인의 데이터 포맷에 맞게 변환 후 학습을 실행합니다.
    """

    def __init__(
        self,
        data_root: str,
        meta_file: str,
        output_dir: str,
        initial_ply: str = None,
        device: str = 'cuda',
        image_scale: float = 1.0
    ):
        """
        Args:
            data_root: 데이터 루트 디렉토리
            meta_file: JSON 메타데이터 파일
            output_dir: 출력 디렉토리
            initial_ply: 초기 포인트 클라우드 (선택)
            device: 'cuda' or 'cpu'
            image_scale: 이미지 스케일
        """
        self.data_root = Path(data_root)
        self.meta_file = Path(meta_file)
        self.output_dir = Path(output_dir)
        self.device = device
        self.image_scale = image_scale
        self.initial_ply = initial_ply

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 3DGS 설치 확인
        check_3dgs_installation()

        # COLMAP 변환 디렉토리
        self.colmap_dir = self.output_dir / "colmap_format"

    def prepare_data(self):
        """데이터를 3DGS가 요구하는 COLMAP 포맷으로 변환"""
        print("\n" + "=" * 70)
        print(">>> [Step 1] Converting to COLMAP format for 3DGS")
        print("=" * 70)

        convert_to_colmap_format(
            data_root=str(self.data_root),
            meta_file=str(self.meta_file),
            output_dir=str(self.colmap_dir),
            image_scale=self.image_scale
        )

        # 초기 PLY가 있으면 points3D.ply로 복사
        if self.initial_ply and Path(self.initial_ply).exists():
            ply_dst = self.colmap_dir / "sparse" / "0" / "points3D.ply"
            shutil.copy2(self.initial_ply, str(ply_dst))
            print(f"  Initial PLY copied: {self.initial_ply} -> {ply_dst}")

    def train(self, num_iterations: int = 30000, sh_degree: int = 3):
        """
        3DGS 학습 실행

        graphdeco-inria/gaussian-splatting의 train.py를 subprocess로 호출합니다.

        Args:
            num_iterations: 학습 반복 횟수
            sh_degree: Spherical Harmonics 차수 (0~3)
        """
        print("\n" + "=" * 70)
        print(">>> [Step 2] Training 3D Gaussian Splatting")
        print("=" * 70)

        train_script = GAUSSIAN_SPLATTING_DIR / "train.py"
        model_dir = self.output_dir / "model"

        cmd = [
            sys.executable,
            str(train_script),
            "-s", str(self.colmap_dir),
            "-m", str(model_dir),
            "--iterations", str(num_iterations),
            "--sh_degree", str(sh_degree),
        ]

        print(f"  Command: {' '.join(cmd)}")
        print(f"  Source:  {self.colmap_dir}")
        print(f"  Output:  {model_dir}")
        print(f"  Iterations: {num_iterations}")
        print()

        # 환경 설정 (3DGS 모듈 경로 추가)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(GAUSSIAN_SPLATTING_DIR) + ":" + env.get("PYTHONPATH", "")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(GAUSSIAN_SPLATTING_DIR),
                env=env,
                capture_output=False,
                text=True,
                timeout=3600 * 12  # 12시간 타임아웃
            )

            if result.returncode == 0:
                print("\n  Training completed successfully!")
            else:
                print(f"\n  Warning: Training exited with code {result.returncode}")
                print("  Possible issues:")
                print("  - CUDA/GPU not available")
                print("  - Missing dependencies (diff-gaussian-rasterization, simple-knn)")
                print("  - Install: pip install plyfile tqdm")
                print(f"  - Check: {GAUSSIAN_SPLATTING_DIR}/environment.yml")
        except subprocess.TimeoutExpired:
            print("\n  Error: Training timed out (12h limit)")
        except FileNotFoundError:
            print(f"\n  Error: Train script not found: {train_script}")

        return model_dir

    def render(self, model_dir: str = None):
        """
        학습된 모델로 렌더링 실행

        Args:
            model_dir: 학습된 모델 디렉토리
        """
        print("\n" + "=" * 70)
        print(">>> [Step 3] Rendering Novel Views")
        print("=" * 70)

        if model_dir is None:
            model_dir = self.output_dir / "model"

        render_script = GAUSSIAN_SPLATTING_DIR / "render.py"

        cmd = [
            sys.executable,
            str(render_script),
            "-s", str(self.colmap_dir),
            "-m", str(model_dir),
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(GAUSSIAN_SPLATTING_DIR) + ":" + env.get("PYTHONPATH", "")

        print(f"  Command: {' '.join(cmd)}")

        try:
            subprocess.run(
                cmd,
                cwd=str(GAUSSIAN_SPLATTING_DIR),
                env=env,
                capture_output=False,
                text=True,
                timeout=3600
            )
            print("  Rendering completed!")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  Error during rendering: {e}")

    def run(self, num_iterations: int = 30000, sh_degree: int = 3):
        """전체 파이프라인 실행"""
        print("\n" + "=" * 70)
        print(">>> 3D Gaussian Splatting (Approach 1) - Full Pipeline")
        print(f"    External: graphdeco-inria/gaussian-splatting")
        print(f"    Data:     {self.data_root}")
        print(f"    Output:   {self.output_dir}")
        print("=" * 70)

        # 1. 데이터 변환
        self.prepare_data()

        # 2. 학습
        model_dir = self.train(
            num_iterations=num_iterations,
            sh_degree=sh_degree
        )

        # 3. 렌더링
        self.render(model_dir=str(model_dir))

        print("\n" + "=" * 70)
        print(">>> Pipeline Complete!")
        print(f"    Model:  {model_dir}")
        print(f"    Output: {self.output_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="3D Gaussian Splatting Training (Approach 1)\n"
                    "Wraps graphdeco-inria/gaussian-splatting"
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
        default='outputs/3dgs',
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
        '--sh_degree',
        type=int,
        default=3,
        help='Spherical Harmonics degree (default: 3)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
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
        print(f"  python reconstruction/prepare_metadata.py {args.data_root} --mode 3dgs")
        return

    # 초기 PLY 경로
    initial_ply = None
    if args.initial_ply:
        initial_ply = Path(args.initial_ply)
        if not initial_ply.is_absolute():
            initial_ply = data_root / initial_ply

    # Trainer 생성 및 실행
    trainer = GaussianSplattingTrainer(
        data_root=str(data_root),
        meta_file=str(meta_file),
        output_dir=str(output_dir),
        initial_ply=str(initial_ply) if initial_ply else None,
        device=args.device,
        image_scale=args.image_scale
    )

    trainer.run(
        num_iterations=args.iterations,
        sh_degree=args.sh_degree
    )


if __name__ == '__main__':
    main()
