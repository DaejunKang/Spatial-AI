"""
3D Reconstruction 공통 Loss 함수 및 유틸리티

approach1_3dgs.py 와 approach2_3dgut.py 가 공통으로 사용합니다.
"""

import math
import torch
import torch.nn.functional as F


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

    def _gaussian_1d(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    _1d = _gaussian_1d(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).unsqueeze(0).unsqueeze(0)
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


def get_projection_matrix(
    znear: float,
    zfar: float,
    fovX: float,
    fovY: float,
    device: torch.device,
) -> torch.Tensor:
    """OpenGL-style perspective projection matrix"""
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
