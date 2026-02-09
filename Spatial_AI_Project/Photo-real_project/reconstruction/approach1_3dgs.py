"""
Approach 1: 3D Gaussian Splatting (3DGS) for Static Scene Reconstruction

Reference:
    "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
    https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

Description:
    정적 장면(Static Scene)을 가정한 표준 3D Gaussian Splatting 학습 스크립트입니다.
    Rolling Shutter 보정 없이 단일 카메라 포즈로 렌더링합니다.

    approach2_3dgut.py의 Rectified 모드와 유사하지만,
    Velocity / Temporal Uncertainty 파라미터가 전혀 없는 순수한 3DGS입니다.

Input:
    - Images: Inpainting된 배경 이미지
    - Camera Extrinsics: World-to-Camera 변환 행렬
    - Camera Intrinsics: 카메라 내부 파라미터
    - (선택) Initial Point Cloud: 초기화용 PLY

Output:
    - Trained 3D Gaussians (.ply)
    - Rendered Novel Views

Prerequisites:
    pip install diff-gaussian-rasterization simple-knn plyfile

Usage:
    python reconstruction/approach1_3dgs.py /path/to/nre_format
    python reconstruction/approach1_3dgs.py /path/to/nre_format \\
        --initial_ply step1_warped/accumulated_static.ply \\
        --iterations 30000
"""

import os
import sys
import math
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime

# PLY I/O
try:
    from plyfile import PlyData, PlyElement
except ImportError:
    PlyData = None
    PlyElement = None

# 3DGS Core Libraries
_HAS_RASTERIZER = False
try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    _HAS_RASTERIZER = True
except ImportError:
    GaussianRasterizationSettings = None
    GaussianRasterizer = None

_HAS_SIMPLE_KNN = False
try:
    from simple_knn._C import distCUDA2
    _HAS_SIMPLE_KNN = True
except ImportError:
    distCUDA2 = None

# Local Modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import ReconstructionDataset, load_initial_point_cloud

# ── Shared Loss Functions (approach2_3dgut.py와 동일) ────────────────────
from approach2_3dgut import l1_loss, ssim, get_projection_matrix


class GaussianSplattingTrainer:
    """
    3D Gaussian Splatting 학습 클래스 (Static Scene)

    정적 장면을 가정하며, RS/Velocity 관련 파라미터 없이 순수 3DGS를 학습합니다.
    """

    def __init__(
        self,
        data_root: str,
        meta_file: str,
        output_dir: str,
        initial_ply: str = None,
        device: str = "cuda",
    ):
        self.data_root = Path(data_root)
        self.meta_file = Path(meta_file)
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "novel_views").mkdir(exist_ok=True)

        if not _HAS_RASTERIZER:
            print(
                "[WARNING] diff-gaussian-rasterization 미설치 → Fallback 렌더링 사용"
            )

        # Dataset (3DGS → velocity 불필요)
        print("=" * 70)
        print(">>> Loading Dataset for 3DGS (Static Scene)")
        print("=" * 70)

        self.dataset = ReconstructionDataset(
            meta_file=str(self.meta_file),
            data_root=str(self.data_root),
            split="train",
            load_3dgut_params=False,
            image_scale=1.0,
        )

        # Initial Point Cloud
        if initial_ply and Path(initial_ply).exists():
            print(f"\n>>> Loading initial point cloud: {initial_ply}")
            self.init_points, self.init_colors = load_initial_point_cloud(initial_ply)
            print(f"  Points: {len(self.init_points)}")
        else:
            print(">>> No initial PLY. Using random initialization.")
            self.init_points = None
            self.init_colors = None

        self.gaussians = None
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        print("=" * 70)

    # ------------------------------------------------------------------
    def initialize_gaussians(self):
        """Gaussian 파라미터 초기화"""
        print("\n[1/4] Initializing Gaussians (3DGS)...")

        if self.init_points is not None:
            num_points = len(self.init_points)
            xyz = torch.tensor(self.init_points, dtype=torch.float32, device=self.device)
            rgb = torch.tensor(self.init_colors, dtype=torch.float32, device=self.device).clamp(1e-6, 1 - 1e-6)
            f_dc = torch.log(rgb / (1 - rgb))

            if _HAS_SIMPLE_KNN and num_points > 1:
                dist2 = torch.clamp_min(distCUDA2(xyz), 1e-7)
                scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
            else:
                scales = torch.log(torch.ones(num_points, 3, device=self.device) * 0.01)

            self.gaussians = {
                "xyz": xyz,
                "f_dc": f_dc,
                "opacity": torch.logit(torch.ones(num_points, 1, device=self.device) * 0.1),
                "scale": scales,
                "rotation": torch.zeros(num_points, 4, device=self.device),
            }
            self.gaussians["rotation"][:, 0] = 1.0
        else:
            num_points = 50000
            self.gaussians = {
                "xyz": (torch.rand(num_points, 3, device=self.device) - 0.5) * 10,
                "f_dc": torch.rand(num_points, 3, device=self.device),
                "opacity": torch.logit(torch.ones(num_points, 1, device=self.device) * 0.1),
                "scale": torch.log(torch.ones(num_points, 3, device=self.device) * 0.1),
                "rotation": torch.zeros(num_points, 4, device=self.device),
            }
            self.gaussians["rotation"][:, 0] = 1.0

        for key in self.gaussians:
            self.gaussians[key].requires_grad = True

        print(f"  Initialized {len(self.gaussians['xyz'])} Gaussians")

    # ------------------------------------------------------------------
    def render(self, extrinsic, intrinsic, width, height):
        """Gaussian Splatting 렌더링"""
        if not _HAS_RASTERIZER:
            return torch.zeros(3, height, width, device=self.device)

        fx, fy = intrinsic[0, 0].item(), intrinsic[1, 1].item()
        fovx = 2 * math.atan(width / (2 * fx))
        fovy = 2 * math.atan(height / (2 * fy))

        znear, zfar = 0.01, 100.0
        proj_matrix = get_projection_matrix(znear, zfar, fovx, fovy, self.device).transpose(0, 1)
        view_matrix = extrinsic.transpose(0, 1)
        full_proj = view_matrix @ proj_matrix
        campos = torch.inverse(view_matrix)[:3, 3]

        means3D = self.gaussians["xyz"]
        scales = torch.exp(self.gaussians["scale"])
        rotations = F.normalize(self.gaussians["rotation"], dim=-1)
        opacity = torch.sigmoid(self.gaussians["opacity"])
        shs = self.gaussians["f_dc"].unsqueeze(1)
        means2D = torch.zeros_like(means3D, requires_grad=True, device=self.device)

        settings = GaussianRasterizationSettings(
            image_height=int(height),
            image_width=int(width),
            tanfovx=math.tan(0.5 * fovx),
            tanfovy=math.tan(0.5 * fovy),
            bg=self.bg_color,
            scale_modifier=1.0,
            viewmatrix=view_matrix,
            projmatrix=full_proj,
            sh_degree=0,
            campos=campos,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=settings)
        image, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        return image

    # ------------------------------------------------------------------
    def train(
        self,
        num_iterations: int = 30000,
        log_interval: int = 100,
        save_interval: int = 5000,
        lambda_dssim: float = 0.2,
    ):
        """3DGS 학습"""
        print(f"\n[2/4] Starting Training (3DGS Static)")
        print(f"  Iterations: {num_iterations}, Lambda DSSIM: {lambda_dssim}")

        optimizer = torch.optim.Adam(
            [
                {"params": [self.gaussians["xyz"]], "lr": 0.00016},
                {"params": [self.gaussians["f_dc"]], "lr": 0.0025},
                {"params": [self.gaussians["opacity"]], "lr": 0.05},
                {"params": [self.gaussians["scale"]], "lr": 0.005},
                {"params": [self.gaussians["rotation"]], "lr": 0.001},
            ],
            lr=0.0,
            eps=1e-15,
        )

        pbar = tqdm(range(num_iterations), desc="Training (3DGS)")
        loss_history = []

        for iteration in pbar:
            idx = np.random.randint(0, len(self.dataset))
            sample = self.dataset[idx]

            gt_image = sample["image"].to(self.device)
            extrinsic = sample["extrinsic"].to(self.device)
            intrinsic = sample["intrinsic"].to(self.device)
            width, height = sample["width"], sample["height"]

            rendered_image = self.render(extrinsic, intrinsic, width, height)

            Ll1 = l1_loss(rendered_image, gt_image)
            loss_ssim = 1.0 - ssim(rendered_image, gt_image)
            total_loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * loss_ssim

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_val = total_loss.item()
            loss_history.append(loss_val)

            if iteration % log_interval == 0:
                pbar.set_postfix({"Loss": f"{loss_val:.4f}", "Pts": len(self.gaussians["xyz"])})

            if save_interval > 0 and (iteration + 1) % save_interval == 0:
                self.save_gaussians(filename=f"gaussians_step{iteration + 1:06d}.ply")

        stats = {
            "total_iterations": num_iterations,
            "final_loss": loss_history[-1] if loss_history else 0,
            "num_gaussians": len(self.gaussians["xyz"]),
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.output_dir / "training_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n  Training completed: {num_iterations} iterations (loss={stats['final_loss']:.6f})")

    # ------------------------------------------------------------------
    def save_gaussians(self, filename: str = "gaussians.ply"):
        """학습된 Gaussian을 PLY로 저장"""
        print(f"\n[3/4] Saving Gaussians → {filename}")
        output_path = self.output_dir / filename

        xyz = self.gaussians["xyz"].detach().cpu().numpy()
        f_dc = self.gaussians["f_dc"].detach().cpu().numpy()
        opacities = self.gaussians["opacity"].detach().cpu().numpy()
        scales = self.gaussians["scale"].detach().cpu().numpy()
        rotations = self.gaussians["rotation"].detach().cpu().numpy()

        if PlyElement is not None:
            dtype_full = [
                ("x", "f4"), ("y", "f4"), ("z", "f4"),
                ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
                ("opacity", "f4"),
                ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
                ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
            ]
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            elements["x"] = xyz[:, 0]
            elements["y"] = xyz[:, 1]
            elements["z"] = xyz[:, 2]
            elements["nx"] = 0
            elements["ny"] = 0
            elements["nz"] = 0
            elements["f_dc_0"] = f_dc[:, 0]
            elements["f_dc_1"] = f_dc[:, 1]
            elements["f_dc_2"] = f_dc[:, 2]
            elements["opacity"] = opacities[:, 0]
            elements["scale_0"] = scales[:, 0]
            elements["scale_1"] = scales[:, 1]
            elements["scale_2"] = scales[:, 2]
            elements["rot_0"] = rotations[:, 0]
            elements["rot_1"] = rotations[:, 1]
            elements["rot_2"] = rotations[:, 2]
            elements["rot_3"] = rotations[:, 3]

            el = PlyElement.describe(elements, "vertex")
            PlyData([el]).write(str(output_path))
        else:
            output_path = output_path.with_suffix(".npz")
            np.savez_compressed(str(output_path), xyz=xyz, f_dc=f_dc, opacity=opacities, scale=scales, rotation=rotations)

        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {output_path} ({file_size:.1f} MB, {xyz.shape[0]} Gaussians)")

    # ------------------------------------------------------------------
    def render_novel_views(self, num_views: int = 10):
        """Novel View 렌더링"""
        print("\n[4/4] Rendering Novel Views...")
        novel_dir = self.output_dir / "novel_views"
        novel_dir.mkdir(exist_ok=True)

        rendered_count = 0
        with torch.no_grad():
            for i in range(min(num_views, len(self.dataset))):
                sample = self.dataset[i]
                extrinsic = sample["extrinsic"].to(self.device)
                intrinsic = sample["intrinsic"].to(self.device)
                width, height = sample["width"], sample["height"]

                img = self.render(extrinsic, intrinsic, width, height)

                try:
                    from torchvision.utils import save_image
                    save_image(img.clamp(0, 1), novel_dir / f"view_{i:03d}.png")
                    rendered_count += 1
                except ImportError:
                    import cv2
                    img_np = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(str(novel_dir / f"view_{i:03d}.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                    rendered_count += 1
                except Exception as e:
                    print(f"  [Warning] View {i} save failed: {e}")

        print(f"  Rendered {rendered_count} novel views → {novel_dir}")

    # ------------------------------------------------------------------
    def run(self, num_iterations: int = 30000, lambda_dssim: float = 0.2, save_interval: int = 5000):
        """전체 파이프라인"""
        print("\n" + "=" * 70)
        print(">>> 3D Gaussian Splatting Training Pipeline (Static Scene)")
        print("=" * 70)

        self.initialize_gaussians()
        self.train(num_iterations=num_iterations, lambda_dssim=lambda_dssim, save_interval=save_interval)
        self.save_gaussians()
        self.render_novel_views()

        print("\n" + "=" * 70)
        print(">>> Training Complete!")
        print(f"  Output: {self.output_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="3DGS Training (Approach 1): Static Scene")
    parser.add_argument("data_root", type=str, help="Path to NRE format data root")
    parser.add_argument("--meta_file", type=str, default="train_meta/train_pairs.json")
    parser.add_argument("--output_dir", type=str, default="outputs/3dgs")
    parser.add_argument("--initial_ply", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    data_root = Path(args.data_root)
    meta_file = data_root / args.meta_file
    output_dir = data_root / args.output_dir

    if not meta_file.exists():
        print(f"Error: Metadata file not found: {meta_file}")
        print(f"  Run: python reconstruction/prepare_metadata.py {args.data_root} --mode 3dgs")
        return

    initial_ply = None
    if args.initial_ply:
        initial_ply = Path(args.initial_ply)
        if not initial_ply.is_absolute():
            initial_ply = data_root / initial_ply
        initial_ply = str(initial_ply)

    trainer = GaussianSplattingTrainer(
        data_root=str(data_root),
        meta_file=str(meta_file),
        output_dir=str(output_dir),
        initial_ply=initial_ply,
        device=args.device,
    )
    trainer.run(num_iterations=args.iterations, lambda_dssim=args.lambda_dssim, save_interval=args.save_interval)


if __name__ == "__main__":
    main()
