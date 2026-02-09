"""
3D Scene Reconstruction - Legacy Entry Point

이 파일은 하위 호환성을 위한 래퍼입니다.
실제 구현은 reconstruction/ 패키지에 있습니다:

    - reconstruction/approach1_3dgs.py   (3DGS Static Scene)
    - reconstruction/approach2_3dgut.py  (3DGUT Alpasim-Optimized)
    - reconstruction/data_loader.py      (공통 데이터 로더)
    - reconstruction/losses.py           (공통 Loss 함수)
    - reconstruction/prepare_metadata.py (메타데이터 생성)

Usage:
    # 3DGS (Static Scene)
    python reconstruction/approach1_3dgs.py /path/to/nre_format

    # 3DGUT (Alpasim, Rectified Input)
    python reconstruction/approach2_3dgut.py /path/to/nre_format

    # 3DGUT (Raw Input, Rolling Shutter)
    python reconstruction/approach2_3dgut.py /path/to/nre_format --raw_input
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="3D Scene Reconstruction (Router)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
이 스크립트는 적절한 reconstruction 스크립트로 라우팅합니다.
직접 실행하려면:

    python reconstruction/approach1_3dgs.py /path/to/data   # 3DGS
    python reconstruction/approach2_3dgut.py /path/to/data  # 3DGUT
        """,
    )
    parser.add_argument("data_root", type=str, help="NRE format data root")
    parser.add_argument(
        "--approach",
        type=str,
        default="3dgut",
        choices=["3dgs", "3dgut"],
        help="Reconstruction approach (default: 3dgut)",
    )
    parser.add_argument(
        "--raw_input",
        action="store_true",
        help="(3DGUT only) Input has Rolling Shutter distortion",
    )
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--initial_ply", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # reconstruction/ 패키지 경로 추가
    recon_dir = Path(__file__).resolve().parent / "reconstruction"
    sys.path.insert(0, str(recon_dir))

    if args.approach == "3dgs":
        from approach1_3dgs import GaussianSplattingTrainer

        data_root = Path(args.data_root)
        meta_file = data_root / "train_meta" / "train_pairs.json"
        output_dir = data_root / "outputs" / "3dgs"

        trainer = GaussianSplattingTrainer(
            data_root=str(data_root),
            meta_file=str(meta_file),
            output_dir=str(output_dir),
            initial_ply=args.initial_ply,
            device=args.device,
        )
        trainer.run(num_iterations=args.iterations)

    elif args.approach == "3dgut":
        from approach2_3dgut import GaussianSplatting3DGUT

        data_root = Path(args.data_root)
        meta_file = data_root / "train_meta" / "train_pairs.json"
        output_dir = data_root / "outputs" / "3dgut_rectified"

        trainer = GaussianSplatting3DGUT(
            data_root=str(data_root),
            meta_file=str(meta_file),
            output_dir=str(output_dir),
            initial_ply=args.initial_ply,
            device=args.device,
        )
        trainer.run(
            num_iterations=args.iterations,
            is_input_corrected=not args.raw_input,
        )


if __name__ == "__main__":
    main()
