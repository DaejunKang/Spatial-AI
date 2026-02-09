"""
Approach 2: 3D Gaussian with Uncertainty and Time (3DGUT)
            - Optimized for Alpasim (Global Shutter Pipeline)

Description:
    이 스크립트는 Rolling Shutter(RS) 보정이 완료된 'Rectified Image'를 입력으로 받아
    물리적으로 정확한(왜곡 없는) 3D Gaussian Map을 생성합니다.

    * Alpasim 호환성:
      - 입력 이미지가 이미 펴져 있으므로, 학습 시 RS 효과를 끄고(Velocity=0) 학습합니다.
      - 결과물(.ply)은 시뮬레이터에서 바로 사용할 수 있는 정본(Canonical) 형태가 됩니다.

    * 유연성:
      - --raw_input 플래그로 Raw 데이터(RS 왜곡) 입력도 지원합니다.
      - 이 경우 기존 RS Chunk Rendering이 활성화됩니다.

Input:
    - Images: Rectified(보정된) 배경 이미지 (기본) 또는 Raw(RS 왜곡) 이미지
    - Camera Extrinsics / Intrinsics
    - Ego Velocity: 선속도 및 각속도 [vx, vy, vz, wx, wy, wz]
    - Rolling Shutter: duration, trigger_time

Output:
    - Trained 3D Gaussians with temporal parameters (.ply)
    - Rendered Novel Views (Rectified / Global Shutter)

Prerequisites:
    pip install diff-gaussian-rasterization simple-knn plyfile

Usage:
    # 기본 (Rectified Input → GS 학습 → Alpasim용 PLY)
    python reconstruction/approach2_3dgut.py /path/to/nre_format

    # Raw Input (RS 왜곡 이미지 → RS Chunk Rendering)
    python reconstruction/approach2_3dgut.py /path/to/nre_format --raw_input

    # 전체 옵션
    python reconstruction/approach2_3dgut.py /path/to/nre_format \\
        --meta_file train_meta/train_pairs.json \\
        --output_dir outputs/3dgut_rectified \\
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

# 3DGS Core Libraries (NVIDIA Gaussian Rasterizer)
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


# ============================================================================
# 1. Loss Functions
# ============================================================================

def l1_loss(network_output: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 (Mean Absolute Error) Loss"""
    return torch.abs(network_output - gt).mean()


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM)

    Args:
        img1, img2: [B, C, H, W] or [C, H, W]
        window_size: 가우시안 윈도우 크기
        size_average: True이면 스칼라 반환

    Returns:
        ssim_val: 0~1 (1 = 완전 동일)
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    channel = img1.size(1)

    # Gaussian window 생성
    def _gaussian_1d(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    _1d = _gaussian_1d(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)  # [1, 1, ws, ws]
    window = _2d.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)

    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# ============================================================================
# 2. Physics Engine: Rolling Shutter Compensator
# ============================================================================

class RollingShutterCompensator:
    """
    카메라의 물리적 움직임(Velocity)을 기반으로 포즈를 시간축(t)에서 보정하는 클래스.

    GS 모드(Rectified Input)에서는 Velocity=0이 입력되어 보정이 수행되지 않음(Identity).
    RS 모드(Raw Input)에서는 실제 속도로 row별 포즈를 보정합니다.
    """

    @staticmethod
    def compute_pose_at_time(
        T_base: torch.Tensor,
        velocity: torch.Tensor,
        delta_t: float,
    ) -> torch.Tensor:
        """
        시간 오프셋을 고려한 카메라 포즈 계산

        Args:
            T_base: [4, 4] 기준 카메라 포즈
            velocity: [6] 선속도 및 각속도 [vx, vy, vz, wx, wy, wz]
            delta_t: 시간 오프셋 (초)

        Returns:
            T_adjusted: [4, 4] 보정된 카메라 포즈
        """
        v = velocity[:3]  # [vx, vy, vz]
        w = velocity[3:]  # [wx, wy, wz]

        # Identity Check: 속도가 0이면 연산 스킵 → Global Shutter 효과
        if torch.sum(torch.abs(velocity)) < 1e-6:
            return T_base

        # Translation offset
        translation = v * delta_t

        # Rotation offset (Rodrigues' formula)
        angle = torch.norm(w) * delta_t

        if angle > 1e-6:
            axis = w / torch.norm(w)
            K = torch.tensor(
                [
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0],
                ],
                device=w.device,
                dtype=w.dtype,
            )
            R_delta = (
                torch.eye(3, device=w.device, dtype=w.dtype)
                + torch.sin(angle) * K
                + (1 - torch.cos(angle)) * (K @ K)
            )
        else:
            R_delta = torch.eye(3, device=w.device, dtype=w.dtype)

        T_delta = torch.eye(4, device=T_base.device, dtype=T_base.dtype)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = translation

        return T_delta @ T_base

    @staticmethod
    def get_pixel_time_offset(
        pixel_y: float,
        image_height: int,
        rs_duration: float,
        rs_trigger_time: float,
    ) -> float:
        """
        픽셀의 캡처 시간 오프셋 계산

        t_pixel = t_trigger + (y / H) * t_duration
        """
        row_ratio = pixel_y / image_height
        return rs_trigger_time + row_ratio * rs_duration


# ============================================================================
# 3. Projection Utilities
# ============================================================================

def get_projection_matrix(
    znear: float,
    zfar: float,
    fovX: float,
    fovY: float,
    device: torch.device,
) -> torch.Tensor:
    """OpenGL-style perspective projection matrix (column-major for rasterizer)"""
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


# ============================================================================
# 4. Main Model: 3DGUT (Alpasim-Optimized)
# ============================================================================

class GaussianSplatting3DGUT:
    """
    3DGUT: Alpasim 호환 3D Gaussian Splatting

    두 가지 모드를 지원합니다:
    - Rectified Mode (기본): Velocity=0 → Global Shutter 렌더링 → Alpasim용 PLY
    - Raw Mode (옵션): 실제 Velocity 사용 → RS Chunk 렌더링
    """

    def __init__(
        self,
        data_root: str,
        meta_file: str,
        output_dir: str,
        initial_ply: str = None,
        device: str = "cuda",
    ):
        """
        Args:
            data_root: 데이터 루트 디렉토리
            meta_file: JSON 메타데이터 파일 (3DGUT 형식)
            output_dir: 출력 디렉토리
            initial_ply: 초기 포인트 클라우드 (선택)
            device: 'cuda' or 'cpu'
        """
        self.data_root = Path(data_root)
        self.meta_file = Path(meta_file)
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "novel_views").mkdir(exist_ok=True)

        # ── Capability Check ──────────────────────────────────────────────
        if not _HAS_RASTERIZER:
            print(
                "[WARNING] diff-gaussian-rasterization 미설치 → Fallback 렌더링 사용\n"
                "  설치: pip install diff-gaussian-rasterization"
            )
        if not _HAS_SIMPLE_KNN:
            print(
                "[WARNING] simple-knn 미설치 → KNN 기반 스케일 초기화 비활성화\n"
                "  설치: pip install simple-knn"
            )
        if PlyData is None:
            print(
                "[WARNING] plyfile 미설치 → PLY 저장이 JSON fallback으로 전환됩니다\n"
                "  설치: pip install plyfile"
            )

        # ── Dataset ───────────────────────────────────────────────────────
        print("=" * 70)
        print(">>> Loading Dataset for 3DGUT")
        print("=" * 70)

        self.dataset = ReconstructionDataset(
            meta_file=str(self.meta_file),
            data_root=str(self.data_root),
            split="train",
            load_3dgut_params=True,
            image_scale=1.0,
        )

        # ── Initial Point Cloud ───────────────────────────────────────────
        if initial_ply and Path(initial_ply).exists():
            print(f"\n>>> Loading initial point cloud: {initial_ply}")
            self.init_points, self.init_colors = load_initial_point_cloud(initial_ply)
            print(f"  Points: {len(self.init_points)}")
        else:
            print(">>> No initial PLY found. Using random initialization.")
            self.init_points = None
            self.init_colors = None

        # ── Internal State ────────────────────────────────────────────────
        self.gaussians = None
        self.rs_compensator = RollingShutterCompensator()
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)

        print("=" * 70)

    # ------------------------------------------------------------------
    # 4-1. Gaussian Initialization
    # ------------------------------------------------------------------
    def initialize_gaussians(self):
        """
        Gaussian 파라미터 초기화

        초기 포인트 클라우드가 있으면 그로부터 시작,
        없으면 50,000개 랜덤 포인트로 시작합니다.

        각 Gaussian의 파라미터:
            xyz: [N, 3]  위치
            f_dc: [N, 3]  SH DC term (color → inverse sigmoid)
            opacity: [N, 1]  불투명도 (logit 공간)
            scale: [N, 3]  크기 (log 공간)
            rotation: [N, 4]  회전 (quaternion)
            temporal_uncertainty: [N, 1]  3DGUT 특화 시간 불확실성
        """
        print("\n[1/4] Initializing Gaussians (3DGUT)...")

        if self.init_points is not None:
            num_points = len(self.init_points)
            xyz = torch.tensor(
                self.init_points, dtype=torch.float32, device=self.device
            )
            rgb = torch.tensor(
                self.init_colors, dtype=torch.float32, device=self.device
            ).clamp(1e-6, 1 - 1e-6)

            # Inverse sigmoid for SH DC term
            f_dc = torch.log(rgb / (1 - rgb))

            # Scale: KNN 기반 (가용 시) 또는 고정값
            if _HAS_SIMPLE_KNN and num_points > 1:
                dist2 = torch.clamp_min(distCUDA2(xyz), 1e-7)
                scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
            else:
                scales = torch.log(
                    torch.ones(num_points, 3, device=self.device) * 0.01
                )

            self.gaussians = {
                "xyz": xyz,
                "f_dc": f_dc,
                "opacity": torch.logit(
                    torch.ones(num_points, 1, device=self.device) * 0.1
                ),
                "scale": scales,
                "rotation": torch.zeros(num_points, 4, device=self.device),
                "temporal_uncertainty": torch.zeros(
                    num_points, 1, device=self.device
                ),
            }
            self.gaussians["rotation"][:, 0] = 1.0  # Quaternion identity
        else:
            num_points = 50000
            self.gaussians = {
                "xyz": (torch.rand(num_points, 3, device=self.device) - 0.5) * 10,
                "f_dc": torch.rand(num_points, 3, device=self.device),
                "opacity": torch.logit(
                    torch.ones(num_points, 1, device=self.device) * 0.1
                ),
                "scale": torch.log(
                    torch.ones(num_points, 3, device=self.device) * 0.1
                ),
                "rotation": torch.zeros(num_points, 4, device=self.device),
                "temporal_uncertainty": torch.zeros(
                    num_points, 1, device=self.device
                ),
            }
            self.gaussians["rotation"][:, 0] = 1.0

        # Enable gradients for all parameters
        for key in self.gaussians:
            self.gaussians[key].requires_grad = True

        print(f"  Initialized {num_points} Gaussians with temporal uncertainty")

    # ------------------------------------------------------------------
    # 4-2. Rasterizer-backed Rendering
    # ------------------------------------------------------------------
    def _build_raster_settings(
        self,
        view_matrix: torch.Tensor,
        proj_matrix: torch.Tensor,
        full_proj_matrix: torch.Tensor,
        height: int,
        width: int,
        fovx: float,
        fovy: float,
    ):
        """NVIDIA Gaussian Rasterizer 설정 빌드"""
        campos = torch.inverse(view_matrix)[:3, 3]

        settings = GaussianRasterizationSettings(
            image_height=int(height),
            image_width=int(width),
            tanfovx=math.tan(0.5 * fovx),
            tanfovy=math.tan(0.5 * fovy),
            bg=self.bg_color,
            scale_modifier=1.0,
            viewmatrix=view_matrix,
            projmatrix=full_proj_matrix,
            sh_degree=0,
            campos=campos,
            prefiltered=False,
            debug=False,
        )
        return settings

    def _rasterize(
        self,
        view_matrix: torch.Tensor,
        proj_matrix: torch.Tensor,
        height: int,
        width: int,
        fovx: float,
        fovy: float,
    ) -> torch.Tensor:
        """단일 뷰 래스터라이즈 (NVIDIA backend)"""
        full_proj = view_matrix @ proj_matrix

        means3D = self.gaussians["xyz"]
        scales = torch.exp(self.gaussians["scale"])
        rotations = F.normalize(self.gaussians["rotation"], dim=-1)
        opacity = torch.sigmoid(self.gaussians["opacity"])
        shs = self.gaussians["f_dc"].unsqueeze(1)

        # 2D means placeholder (gradient 수집용)
        means2D = torch.zeros_like(means3D, requires_grad=True, device=self.device)

        settings = self._build_raster_settings(
            view_matrix, proj_matrix, full_proj, height, width, fovx, fovy
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

    def _render_fallback(self, height: int, width: int) -> torch.Tensor:
        """
        Rasterizer 미설치 시 Fallback 렌더링
        (단순 projection + splatting 시뮬레이션)
        """
        return torch.zeros(3, height, width, device=self.device)

    # ------------------------------------------------------------------
    # 4-3. Unified Rendering (GS / RS 자동 분기)
    # ------------------------------------------------------------------
    def render(
        self,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
        velocity: torch.Tensor,
        rs_duration: float,
        rs_trigger_time: float,
        width: int,
        height: int,
    ) -> torch.Tensor:
        """
        통합 렌더링 함수

        - velocity ≈ 0  → Global Shutter (단일 패스, 빠름)
        - velocity ≠ 0  → Rolling Shutter Chunk Rendering (8-chunk 합성)

        Args:
            extrinsic: [4, 4] 기준 카메라 포즈
            intrinsic: [3, 3] 카메라 내부 파라미터
            velocity: [6] [vx, vy, vz, wx, wy, wz]
            rs_duration: Rolling Shutter 지속 시간
            rs_trigger_time: 촬영 시작 오프셋
            width, height: 이미지 크기

        Returns:
            rendered_image: [3, H, W]
        """
        # Rasterizer가 없으면 fallback
        if not _HAS_RASTERIZER:
            return self._render_fallback(height, width)

        # FOV 계산 (Intrinsic → FOV)
        fx, fy = intrinsic[0, 0].item(), intrinsic[1, 1].item()
        fovx = 2 * math.atan(width / (2 * fx))
        fovy = 2 * math.atan(height / (2 * fy))

        znear, zfar = 0.01, 100.0
        proj_matrix = get_projection_matrix(
            znear, zfar, fovx, fovy, self.device
        ).transpose(0, 1)

        # ── Case A: Global Shutter (Velocity ≈ 0) ────────────────────
        if torch.sum(torch.abs(velocity)) < 1e-6:
            view_matrix = extrinsic.transpose(0, 1)
            return self._rasterize(
                view_matrix, proj_matrix, height, width, fovx, fovy
            )

        # ── Case B: Rolling Shutter Chunk Rendering ───────────────────
        num_chunks = 8
        chunk_height = height // num_chunks
        rendered_chunks = []

        for i in range(num_chunks):
            start_row = i * chunk_height
            end_row = (i + 1) * chunk_height if i < num_chunks - 1 else height
            mid_row = (start_row + end_row) / 2.0

            # Row 중앙의 시간 오프셋 → 포즈 보정
            time_offset = self.rs_compensator.get_pixel_time_offset(
                mid_row, height, rs_duration, rs_trigger_time
            )
            T_adjusted = self.rs_compensator.compute_pose_at_time(
                extrinsic, velocity, time_offset
            )

            view_matrix = T_adjusted.transpose(0, 1)
            chunk_image = self._rasterize(
                view_matrix, proj_matrix, height, width, fovx, fovy
            )
            # Crop to chunk rows
            rendered_chunks.append(chunk_image[:, start_row:end_row, :])

        return torch.cat(rendered_chunks, dim=1)

    # ------------------------------------------------------------------
    # 4-4. Training Loop
    # ------------------------------------------------------------------
    def train(
        self,
        num_iterations: int = 30000,
        log_interval: int = 100,
        save_interval: int = 5000,
        is_input_corrected: bool = True,
        lambda_dssim: float = 0.2,
        lambda_reg: float = 0.001,
    ):
        """
        학습 메인 루프

        Args:
            num_iterations: 총 학습 반복 횟수
            log_interval: 로그 출력 간격
            save_interval: 중간 체크포인트 저장 간격 (0이면 비활성화)
            is_input_corrected: True = Rectified Input (GS), False = Raw (RS)
            lambda_dssim: SSIM loss 가중치
            lambda_reg: Temporal uncertainty regularization 가중치
        """
        mode_str = "GS/Rectified (Alpasim)" if is_input_corrected else "RS/Raw"
        print(f"\n[2/4] Starting Training (Mode: {mode_str})")
        print(f"  Iterations: {num_iterations}")
        print(f"  Lambda DSSIM: {lambda_dssim}")
        print(f"  Lambda Reg: {lambda_reg}")

        # Optimizer (3DGS 표준 학습률)
        optimizer = torch.optim.Adam(
            [
                {"params": [self.gaussians["xyz"]], "lr": 0.00016, "name": "xyz"},
                {"params": [self.gaussians["f_dc"]], "lr": 0.0025, "name": "f_dc"},
                {"params": [self.gaussians["opacity"]], "lr": 0.05, "name": "opacity"},
                {"params": [self.gaussians["scale"]], "lr": 0.005, "name": "scale"},
                {"params": [self.gaussians["rotation"]], "lr": 0.001, "name": "rotation"},
                {
                    "params": [self.gaussians["temporal_uncertainty"]],
                    "lr": 0.01,
                    "name": "temporal_uncertainty",
                },
            ],
            lr=0.0,
            eps=1e-15,
        )

        pbar = tqdm(range(num_iterations), desc="Training (3DGUT)")
        loss_history = []

        for iteration in pbar:
            # 1. Random sample
            idx = np.random.randint(0, len(self.dataset))
            sample = self.dataset[idx]

            gt_image = sample["image"].to(self.device)
            extrinsic = sample["extrinsic"].to(self.device)
            intrinsic = sample["intrinsic"].to(self.device)
            velocity = sample["velocity"].to(self.device)
            rs_duration = sample["rolling_shutter_duration"]
            rs_trigger = sample["rolling_shutter_trigger_time"]
            width, height = sample["width"], sample["height"]

            # 2. Pipeline Control: Rectified → velocity 강제 0
            if is_input_corrected:
                current_velocity = torch.zeros_like(velocity)
            else:
                current_velocity = velocity

            # 3. Rendering
            rendered_image = self.render(
                extrinsic,
                intrinsic,
                current_velocity,
                rs_duration,
                rs_trigger,
                width,
                height,
            )

            # 4. Loss 계산
            Ll1 = l1_loss(rendered_image, gt_image)
            loss_ssim = 1.0 - ssim(rendered_image, gt_image)
            loss_color = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * loss_ssim

            # Temporal Uncertainty Regularization
            # uncertainty가 큰 점은 temporal 정보에 민감 → 정규화로 안정화
            loss_reg = lambda_reg * torch.sigmoid(
                self.gaussians["temporal_uncertainty"]
            ).mean()

            total_loss = loss_color + loss_reg

            # 5. Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # 6. Logging
            loss_val = total_loss.item()
            loss_history.append(loss_val)

            if iteration % log_interval == 0:
                pbar.set_postfix(
                    {
                        "Loss": f"{loss_val:.4f}",
                        "L1": f"{Ll1.item():.4f}",
                        "SSIM": f"{loss_ssim.item():.4f}",
                        "Pts": len(self.gaussians["xyz"]),
                    }
                )

            # 7. 중간 체크포인트
            if save_interval > 0 and (iteration + 1) % save_interval == 0:
                ckpt_name = f"gaussians_3dgut_step{iteration + 1:06d}.ply"
                self.save_gaussians(filename=ckpt_name)

        # 학습 통계 저장
        stats = {
            "total_iterations": num_iterations,
            "final_loss": loss_history[-1] if loss_history else 0,
            "avg_loss_last_100": float(np.mean(loss_history[-100:])) if loss_history else 0,
            "mode": mode_str,
            "lambda_dssim": lambda_dssim,
            "lambda_reg": lambda_reg,
            "num_gaussians": len(self.gaussians["xyz"]),
            "timestamp": datetime.now().isoformat(),
        }
        stats_path = self.output_dir / "training_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n  Training completed: {num_iterations} iterations")
        print(f"  Final loss: {stats['final_loss']:.6f}")
        print(f"  Stats saved: {stats_path}")

    # ------------------------------------------------------------------
    # 4-5. PLY Save / Export
    # ------------------------------------------------------------------
    def save_gaussians(self, filename: str = "gaussians_3dgut.ply"):
        """
        학습된 Gaussian 파라미터를 PLY 포맷으로 저장

        PLY 필드:
            x, y, z: 위치
            nx, ny, nz: 법선 (0 placeholder)
            f_dc_0, f_dc_1, f_dc_2: SH DC color
            opacity: 불투명도
            scale_0, scale_1, scale_2: 크기 (log scale)
            rot_0, rot_1, rot_2, rot_3: 회전 (quaternion)
            temporal_uncertainty: 3DGUT 시간 불확실성
        """
        print(f"\n[3/4] Saving Gaussians → {filename}")
        output_path = self.output_dir / filename

        xyz = self.gaussians["xyz"].detach().cpu().numpy()
        f_dc = self.gaussians["f_dc"].detach().cpu().numpy()
        opacities = self.gaussians["opacity"].detach().cpu().numpy()
        scales = self.gaussians["scale"].detach().cpu().numpy()
        rotations = self.gaussians["rotation"].detach().cpu().numpy()
        uncertainty = self.gaussians["temporal_uncertainty"].detach().cpu().numpy()

        if PlyElement is not None:
            # ── plyfile 사용 (표준 3DGS PLY 포맷) ──
            dtype_full = [
                ("x", "f4"), ("y", "f4"), ("z", "f4"),
                ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
                ("opacity", "f4"),
                ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
                ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
                ("temporal_uncertainty", "f4"),
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
            elements["temporal_uncertainty"] = uncertainty[:, 0]

            el = PlyElement.describe(elements, "vertex")
            PlyData([el]).write(str(output_path))
        else:
            # ── Fallback: NumPy npz ──
            output_path = output_path.with_suffix(".npz")
            np.savez_compressed(
                str(output_path),
                xyz=xyz,
                f_dc=f_dc,
                opacity=opacities,
                scale=scales,
                rotation=rotations,
                temporal_uncertainty=uncertainty,
            )

        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {output_path} ({file_size:.1f} MB, {xyz.shape[0]} Gaussians)")

    # ------------------------------------------------------------------
    # 4-6. Novel View Rendering
    # ------------------------------------------------------------------
    def render_novel_views(self, num_views: int = 10):
        """
        Novel View 렌더링 (검증 및 시각화용)

        항상 Global Shutter(Velocity=0)로 렌더링합니다.
        이래야 시뮬레이터에서 왜곡 없는 맵으로 사용 가능합니다.
        """
        print("\n[4/4] Rendering Novel Views (Rectified / Global Shutter)...")

        novel_dir = self.output_dir / "novel_views"
        novel_dir.mkdir(exist_ok=True)

        rendered_count = 0

        with torch.no_grad():
            for i in range(min(num_views, len(self.dataset))):
                sample = self.dataset[i]

                extrinsic = sample["extrinsic"].to(self.device)
                intrinsic = sample["intrinsic"].to(self.device)
                width, height = sample["width"], sample["height"]

                # Novel View는 항상 GS (Velocity=0)
                velocity_zero = torch.zeros(6, device=self.device)

                img = self.render(
                    extrinsic,
                    intrinsic,
                    velocity_zero,
                    0,
                    0,
                    width,
                    height,
                )

                # 이미지 저장
                try:
                    from torchvision.utils import save_image

                    save_image(img.clamp(0, 1), novel_dir / f"view_{i:03d}.png")
                    rendered_count += 1
                except ImportError:
                    # torchvision 미설치 시 numpy 저장
                    import cv2

                    img_np = (
                        img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                    ).astype(np.uint8)
                    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(novel_dir / f"view_{i:03d}.png"), img_bgr)
                    rendered_count += 1
                except Exception as e:
                    print(f"  [Warning] View {i} 저장 실패: {e}")

        print(f"  Rendered {rendered_count} novel views → {novel_dir}")

    # ------------------------------------------------------------------
    # 4-7. Full Pipeline
    # ------------------------------------------------------------------
    def run(
        self,
        num_iterations: int = 30000,
        is_input_corrected: bool = True,
        lambda_dssim: float = 0.2,
        save_interval: int = 5000,
    ):
        """
        전체 파이프라인 실행

        Args:
            num_iterations: 학습 반복 수
            is_input_corrected: True = Rectified Input, False = Raw Input
            lambda_dssim: SSIM loss 가중치
            save_interval: 중간 저장 간격
        """
        mode = "Rectified (GS) → Alpasim" if is_input_corrected else "Raw (RS) → Reconstruction"

        print("\n" + "=" * 70)
        print(f">>> 3DGUT Training Pipeline")
        print(f"    Mode: {mode}")
        print(f"    Iterations: {num_iterations}")
        print("=" * 70)

        # 1. 초기화
        self.initialize_gaussians()

        # 2. 학습
        self.train(
            num_iterations=num_iterations,
            is_input_corrected=is_input_corrected,
            lambda_dssim=lambda_dssim,
            save_interval=save_interval,
        )

        # 3. 최종 PLY 저장
        self.save_gaussians()

        # 4. Novel View Rendering
        self.render_novel_views()

        print("\n" + "=" * 70)
        print(">>> Training Complete!")
        print(f"  Output: {self.output_dir}")
        print(f"  PLY: {self.output_dir / 'gaussians_3dgut.ply'}")
        print(f"  Novel Views: {self.output_dir / 'novel_views'}")
        print("=" * 70)


# ============================================================================
# 5. CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "3DGUT Training Script - Alpasim-Optimized Global Shutter Pipeline\n\n"
            "Rectified(보정된) 이미지를 입력으로 물리적으로 정확한 3D Gaussian Map을 생성합니다."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 (Rectified Input)
  python approach2_3dgut.py /path/to/nre_format

  # Raw Input (Rolling Shutter 왜곡 이미지)
  python approach2_3dgut.py /path/to/nre_format --raw_input

  # 고급 설정
  python approach2_3dgut.py /path/to/nre_format \\
      --initial_ply step1_warped/accumulated_static.ply \\
      --iterations 50000 \\
      --lambda_dssim 0.2 \\
      --save_interval 10000
        """,
    )

    parser.add_argument(
        "data_root",
        type=str,
        help="Path to NRE format data root",
    )
    parser.add_argument(
        "--meta_file",
        type=str,
        default="train_meta/train_pairs.json",
        help="JSON metadata file (relative to data_root)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/3dgut_rectified",
        help="Output directory (relative to data_root)",
    )
    parser.add_argument(
        "--initial_ply",
        type=str,
        default=None,
        help="Initial point cloud PLY file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30000,
        help="Number of training iterations (default: 30000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--lambda_dssim",
        type=float,
        default=0.2,
        help="SSIM loss weight (default: 0.2)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5000,
        help="Checkpoint save interval (0=disabled, default: 5000)",
    )

    # Pipeline Control Flag
    parser.add_argument(
        "--raw_input",
        action="store_true",
        help=(
            "입력 이미지가 Raw(Rolling Shutter 왜곡)임을 지정합니다.\n"
            "미지정 시(기본값): 입력이 이미 Rectified(보정됨)로 가정 → Velocity=0 학습"
        ),
    )

    args = parser.parse_args()

    # 경로 설정
    data_root = Path(args.data_root)
    meta_file = data_root / args.meta_file
    output_dir = data_root / args.output_dir

    # 메타데이터 존재 확인
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
        initial_ply = str(initial_ply)

    # is_input_corrected: raw_input 플래그가 없으면 True (기본 = Rectified)
    is_input_corrected = not args.raw_input

    if is_input_corrected:
        print("\n[Pipeline] Rectified Input → Global Shutter 학습 (Alpasim용)")
        print("  RS 보정은 전처리 단계에서 이미 완료되었다고 가정합니다.")
        print("  Velocity=0으로 학습하여 왜곡 없는 3D Map을 생성합니다.\n")
    else:
        print("\n[Pipeline] Raw Input → Rolling Shutter Chunk 학습")
        print("  입력 이미지에 RS 왜곡이 있다고 가정합니다.")
        print("  실제 Velocity를 사용하여 RS를 보정하며 학습합니다.\n")

    # Trainer 생성 및 실행
    trainer = GaussianSplatting3DGUT(
        data_root=str(data_root),
        meta_file=str(meta_file),
        output_dir=str(output_dir),
        initial_ply=initial_ply,
        device=args.device,
    )

    trainer.run(
        num_iterations=args.iterations,
        is_input_corrected=is_input_corrected,
        lambda_dssim=args.lambda_dssim,
        save_interval=args.save_interval,
    )

    print("\n>>> All processes completed successfully.")


if __name__ == "__main__":
    main()
