"""
Inpainting Step 2: Depth Guide Generation (LiDAR-Guided Monocular Depth)

Step 1의 시계열 누적 결과에서 여전히 남아있는 구멍을 감지하고,
Dense한 Depth Guide Map을 생성하여 Step 3 (ControlNet Inpainting)의
기하학적 가이드로 제공합니다.

=== 핵심 아이디어 ===
기존 문제점:
    - RANSAC Plane Fitting은 "바닥=평면" 가정에 의존
    - 벽, 연석, 경사로, 곡면 등에서 depth가 깨짐
    - 이미지 텍스처(구조)를 전혀 보지 않고 좌표만으로 추정

솔루션: LiDAR-Guided Monocular Depth Estimation
    1. Depth Anything (Monocular Depth) → 이미지 전체의 "상대적" 형상 추론
    2. Sparse LiDAR 포인트 → "절대적" 스케일의 앵커 역할
    3. RANSAC Linear Regression → Mono depth를 Metric depth로 변환
       공식: Metric_Depth = Scale × Mono_Depth + Shift
    4. Fusion → LiDAR가 있는 곳은 LiDAR 신뢰, 없는 곳은 보정된 AI depth

Fallback 전략:
    - LiDAR + Depth Anything → 최고 품질 (기본)
    - Depth Anything만 (LiDAR 없음) → 상대적 형상은 정확, 절대 스케일 없음
    - RANSAC Plane Fitting → Depth Anything 미설치 시 기존 방식
    - Simple Inpainting → 모든 것이 실패했을 때 최후 수단

Input:
    - data_root/step1_warped/: Step 1 출력 (시계열 누적 이미지)
    - data_root/depths/: 원본 LiDAR depth maps (Sparse, 선택)

Output:
    - data_root/step2_depth_guide/: Dense depth guide maps (uint16, mm)
    - data_root/step2_hole_masks/: 채워야 할 구멍 영역 마스크 (uint8)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import RANSACRegressor
import warnings


# ---------------------------------------------------------------------------
# Depth Anything 모델 로드 (Optional Dependency)
# ---------------------------------------------------------------------------
_DEPTH_ESTIMATOR = None
_DEPTH_MODEL_LOADED = False


def _load_depth_anything(device="cuda:0"):
    """
    Depth Anything 모델을 lazy-load 합니다.
    transformers 라이브러리가 없거나 모델 다운로드에 실패하면 None을 반환합니다.
    
    Returns:
        depth_estimator: HuggingFace depth-estimation pipeline 또는 None
    """
    global _DEPTH_ESTIMATOR, _DEPTH_MODEL_LOADED
    
    if _DEPTH_MODEL_LOADED:
        return _DEPTH_ESTIMATOR
    
    _DEPTH_MODEL_LOADED = True
    
    try:
        from transformers import pipeline as hf_pipeline
        import torch
        
        # GPU 사용 가능 여부 확인
        if "cuda" in device and not torch.cuda.is_available():
            device = "cpu"
            print("  [Depth Anything] CUDA not available, falling back to CPU")
        
        print(f"  [Depth Anything] Loading model on {device}...")
        _DEPTH_ESTIMATOR = hf_pipeline(
            task="depth-estimation",
            model="LiheYoung/depth-anything-small-hf",
            device=device
        )
        print("  [Depth Anything] Model loaded successfully")
        
    except ImportError:
        warnings.warn(
            "transformers 패키지가 설치되지 않았습니다.\n"
            "Depth Anything을 사용하려면: pip install transformers torch\n"
            "기존 RANSAC Plane Fitting 방식으로 fallback합니다."
        )
        _DEPTH_ESTIMATOR = None
        
    except Exception as e:
        warnings.warn(
            f"Depth Anything 모델 로드 실패: {e}\n"
            f"기존 RANSAC Plane Fitting 방식으로 fallback합니다."
        )
        _DEPTH_ESTIMATOR = None
    
    return _DEPTH_ESTIMATOR


class GeometricGuideGenerator:
    """
    LiDAR-Guided Monocular Depth 기반 Depth Guide 생성 클래스
    
    처리 우선순위:
    1. [최선] Depth Anything + Sparse LiDAR → 전체 Dense Metric Depth
    2. [차선] Depth Anything만 (LiDAR 없음) → 상대적 Dense Depth
    3. [Fallback] RANSAC Plane Fitting → 바닥 평면 가정
    4. [최후] Simple cv2.inpaint → 주변 값 보간
    """
    
    def __init__(
        self,
        data_root,
        use_lidar_depth=True,
        ground_region_ratio=0.6,
        device="cuda:0",
        mono_alignment_min_points=50,
        ransac_residual_threshold=500.0
    ):
        """
        Args:
            data_root: Preprocessing + Step 1 출력 디렉토리
            use_lidar_depth: LiDAR depth를 사용할지 여부
            ground_region_ratio: 바닥 평면 추정에 사용할 이미지 하단 비율
                                 (RANSAC fallback에서만 사용)
            device: Depth Anything 추론 디바이스 ("cuda:0" or "cpu")
            mono_alignment_min_points: Mono→Metric 정합에 필요한 최소 LiDAR 포인트 수
            ransac_residual_threshold: RANSAC 허용 잔차 (mm 단위)
        """
        self.data_root = Path(data_root)
        self.use_lidar_depth = use_lidar_depth
        self.ground_region_ratio = ground_region_ratio
        self.device = device
        self.mono_alignment_min_points = mono_alignment_min_points
        self.ransac_residual_threshold = ransac_residual_threshold
        
        # 디렉토리 설정
        self.step1_dir = self.data_root / 'step1_warped'
        self.depths_dir = self.data_root / 'depths'
        self.output_depth_dir = self.data_root / 'step2_depth_guide'
        self.output_mask_dir = self.data_root / 'step2_hole_masks'
        
        # 출력 디렉토리 생성
        self.output_depth_dir.mkdir(parents=True, exist_ok=True)
        self.output_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1 출력 확인
        if not self.step1_dir.exists():
            raise FileNotFoundError(
                f"Step 1 output not found: {self.step1_dir}\n"
                f"Please run step1_temporal_accumulation.py first."
            )
        
        # 파일 목록
        self.warped_files = sorted(list(self.step1_dir.glob('*.png')))
        if len(self.warped_files) == 0:
            # jpg도 시도
            self.warped_files = sorted(list(self.step1_dir.glob('*.jpg')))
        
        if len(self.warped_files) == 0:
            raise ValueError(f"No warped images found in {self.step1_dir}")
        
        # Depth Anything 모델 로드 (Lazy - 첫 사용 시 실제 로드)
        self.depth_estimator = None  # run() 시작 시 로드
        
        print(f"[GeometricGuideGenerator] Initialized")
        print(f"  Data root: {self.data_root}")
        print(f"  Step 1 images: {len(self.warped_files)}")
        print(f"  Use LiDAR depth: {self.use_lidar_depth}")
        print(f"  Device: {self.device}")
        print(f"  Min alignment points: {self.mono_alignment_min_points}")
    
    # =========================================================================
    #  Main Pipeline
    # =========================================================================
    
    def run(self):
        """메인 파이프라인 실행"""
        print("\n" + "=" * 70)
        print(">>> [Step 2] Depth Guide Generation Started")
        print("=" * 70)
        
        # Depth Anything 모델 로드 시도
        print("\n[0/2] Loading Depth Estimation model...")
        self.depth_estimator = _load_depth_anything(device=self.device)
        
        if self.depth_estimator is not None:
            print("  Mode: LiDAR-Guided Monocular Depth (Depth Anything)")
        else:
            print("  Mode: RANSAC Plane Fitting (Fallback)")
        
        # 프레임 처리
        print(f"\n[1/2] Processing {len(self.warped_files)} frames...")
        success_count = 0
        fail_count = 0
        method_stats = {"mono_lidar": 0, "mono_only": 0, "ransac": 0, "inpaint": 0}
        
        for warped_file in tqdm(self.warped_files, desc="Processing frames"):
            try:
                method = self._process_frame(warped_file)
                success_count += 1
                method_stats[method] = method_stats.get(method, 0) + 1
            except Exception as e:
                print(f"\n  Warning: Failed to process {warped_file.name}: {e}")
                fail_count += 1
                continue
        
        # 결과 요약
        print("\n" + "=" * 70)
        print(f">>> [Step 2] Complete!")
        print(f"  Success: {success_count}/{len(self.warped_files)}")
        print(f"  Failed:  {fail_count}/{len(self.warped_files)}")
        print(f"  Methods used:")
        print(f"    - Mono + LiDAR alignment: {method_stats.get('mono_lidar', 0)}")
        print(f"    - Mono only (no LiDAR):   {method_stats.get('mono_only', 0)}")
        print(f"    - RANSAC Plane Fitting:    {method_stats.get('ransac', 0)}")
        print(f"    - Simple Inpainting:       {method_stats.get('inpaint', 0)}")
        print(f"  Output: {self.output_depth_dir}")
        print("=" * 70)
    
    # =========================================================================
    #  Frame Processing (Routing Logic)
    # =========================================================================
    
    def _process_frame(self, warped_file):
        """
        개별 프레임 처리 - 사용 가능한 데이터에 따라 최적 방법 선택
        
        Args:
            warped_file: Step 1 출력 이미지 경로
        
        Returns:
            method: 사용된 방법 이름 (통계용)
        """
        # Step 1 결과 로드 (RGB)
        warped_img = cv2.imread(str(warped_file))
        if warped_img is None:
            raise ValueError(f"Failed to load image: {warped_file}")
        
        # 구멍 영역 감지 (검은색 픽셀 = 여전히 구멍)
        hole_mask = self._detect_holes(warped_img)
        
        # Sparse LiDAR Depth 로드 시도
        depth_file = self.depths_dir / warped_file.name
        sparse_lidar = None
        
        if self.use_lidar_depth and depth_file.exists():
            raw = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
            if raw is not None:
                sparse_lidar = raw.astype(np.float32)
        
        # ---- Routing: 최적 방법 선택 ----
        method = "inpaint"  # 기본값 (최후 수단)
        
        if self.depth_estimator is not None:
            # Depth Anything 사용 가능
            if sparse_lidar is not None:
                # [최선] Mono + LiDAR Alignment
                completed_depth = self._fill_depth_with_monocular_guide(
                    warped_img, sparse_lidar, hole_mask
                )
                method = "mono_lidar"
            else:
                # [차선] Mono Only (스케일 보정 없이 상대적 depth)
                completed_depth = self._fill_depth_mono_only(
                    warped_img, hole_mask
                )
                method = "mono_only"
        else:
            # Depth Anything 미설치 → Legacy fallback
            if sparse_lidar is not None:
                # [Fallback] RANSAC Plane Fitting
                completed_depth = self._fill_ground_plane(sparse_lidar, hole_mask)
                method = "ransac"
            else:
                # [최후] Pseudo depth + Simple inpaint
                pseudo = self._generate_pseudo_depth(warped_img.shape[:2])
                completed_depth = self._simple_depth_inpaint(pseudo, hole_mask)
                method = "inpaint"
        
        # 저장
        output_depth_path = self.output_depth_dir / warped_file.name
        output_mask_path = self.output_mask_dir / warped_file.name
        
        cv2.imwrite(str(output_depth_path), completed_depth.astype(np.uint16))
        cv2.imwrite(str(output_mask_path), hole_mask)
        
        return method
    
    # =========================================================================
    #  [핵심] LiDAR-Guided Monocular Depth Estimation
    # =========================================================================
    
    def _fill_depth_with_monocular_guide(self, rgb_image, sparse_depth, hole_mask):
        """
        LiDAR-Guided Monocular Depth: 핵심 솔루션
        
        RGB 이미지에서 Monocular Depth(상대적 형상)를 추론하고,
        Sparse LiDAR 데이터(절대적 스케일)에 맞춰 Scale/Shift를 보정합니다.
        
        왜 이게 RANSAC Plane보다 나은가?
        -----------------------------------------
        1. 이미지 텍스처를 "봄" → 벽, 연석, 경사로의 3D 형상을 추론
        2. 평면 가정 없음 → 곡면, 계단, 경사에서도 동작
        3. LiDAR로 절대 스케일 보정 → 물리적으로 정확한 metric depth
        4. Dense 출력 → 모든 픽셀에 depth 값 존재
        
        Pipeline:
        ---------
        RGB Image ─────► Depth Anything ─────► Relative Depth (Dense)
                                                     │
        Sparse LiDAR ──► Alignment Mask ──┐          │
                                          ▼          ▼
                                   RANSAC Regression
                                   (Scale, Shift 추정)
                                          │
                                          ▼
                                   Metric Depth (Dense)
                                          │
        Sparse LiDAR ──► Fusion ◄─────────┘
                              │
                              ▼
                        Final Depth Guide
        
        Args:
            rgb_image: (H, W, 3) BGR 이미지 (Step 1 warped 결과)
            sparse_depth: (H, W) float32, LiDAR depth (mm, 0=무효)
            hole_mask: (H, W) uint8, 구멍 마스크 (255=구멍, 0=유효)
        
        Returns:
            final_depth: (H, W) float32, Dense Metric Depth (mm)
        """
        h, w = rgb_image.shape[:2]
        
        # ------------------------------------------------------------------
        # Stage 1: Monocular Depth Estimation (형상 추론)
        # ------------------------------------------------------------------
        # Depth Anything은 이미지 전체의 "상대적" 깊이를 추론합니다.
        # 출력은 값이 클수록 "멀다"는 의미일 수도, "가깝다"는 의미일 수도 있음.
        # → 뒤에서 Linear Regression이 부호(Scale)를 자동 보정합니다.
        mono_depth = self._estimate_monocular_depth(rgb_image)
        
        if mono_depth is None:
            # 추론 실패 시 RANSAC fallback
            warnings.warn("Monocular depth estimation failed. Falling back to RANSAC.")
            return self._fill_ground_plane(sparse_depth, hole_mask)
        
        # 크기 맞추기 (모델 출력이 원본과 다를 수 있음)
        if mono_depth.shape[:2] != (h, w):
            mono_depth = cv2.resize(
                mono_depth, (w, h), interpolation=cv2.INTER_LINEAR
            )
        
        # ------------------------------------------------------------------
        # Stage 2: Alignment Mask (신뢰할 수 있는 앵커 포인트 선택)
        # ------------------------------------------------------------------
        # "LiDAR 데이터가 존재하면서, 구멍이 아닌 곳"만 사용합니다.
        # 이 지점들은 Mono depth와 실제 Metric depth를 동시에 알고 있으므로
        # 변환 공식 (Scale, Shift)을 추정하는 데 사용됩니다.
        valid_mask = (sparse_depth > 0) & (hole_mask == 0)
        valid_count = np.sum(valid_mask)
        
        if valid_count < self.mono_alignment_min_points:
            # 정합할 LiDAR 포인트가 부족하면 Mono-only 모드
            warnings.warn(
                f"LiDAR points for alignment too sparse ({valid_count} < "
                f"{self.mono_alignment_min_points}). Using mono-only depth."
            )
            return self._finalize_mono_only(mono_depth, sparse_depth, hole_mask)
        
        # ------------------------------------------------------------------
        # Stage 3: Scale & Shift Regression (Mono → Metric 변환)
        # ------------------------------------------------------------------
        # 공식: Metric_Depth = Scale × Mono_Depth + Shift
        #
        # Depth Anything 출력이 "Disparity(가까울수록 큰 값)" 형태이든
        # "Depth(멀수록 큰 값)" 형태이든, 선형 회귀가 알아서
        # Scale의 부호를 맞춰줍니다. (음수 Scale = 반전 관계)
        #
        # RANSAC을 사용하는 이유:
        # - LiDAR 포인트 중 일부가 동적 객체(잔여)에 걸려 있을 수 있음
        # - 이런 아웃라이어를 자동 무시하여 robust한 변환식 추정
        
        X_mono = mono_depth[valid_mask].reshape(-1, 1)   # Monocular 값
        y_metric = sparse_depth[valid_mask]               # LiDAR 값 (mm)
        
        try:
            regressor = RANSACRegressor(
                random_state=42,
                min_samples=min(100, valid_count // 2),
                max_trials=200,
                residual_threshold=self.ransac_residual_threshold
            )
            regressor.fit(X_mono, y_metric)
            
            scale = regressor.estimator_.coef_[0]
            shift = regressor.estimator_.intercept_
            
            # 정합 품질 확인
            inlier_ratio = np.sum(regressor.inlier_mask_) / len(regressor.inlier_mask_)
            
        except Exception as e:
            warnings.warn(f"Scale/Shift regression failed: {e}. Using mono-only.")
            return self._finalize_mono_only(mono_depth, sparse_depth, hole_mask)
        
        # ------------------------------------------------------------------
        # Stage 4: 전체 맵 보정 (Calibration)
        # ------------------------------------------------------------------
        # 추정된 Scale/Shift를 이미지 전체에 적용하여
        # "상대적 depth"를 "절대적 metric depth"로 변환합니다.
        aligned_depth = scale * mono_depth + shift
        
        # 물리적으로 불가능한 값 제거
        aligned_depth = np.clip(aligned_depth, 0, 65535)
        
        # ------------------------------------------------------------------
        # Stage 5: Fusion (LiDAR + Aligned Mono 합성)
        # ------------------------------------------------------------------
        # 전략:
        #   - LiDAR 데이터가 있는 곳 → LiDAR 값 사용 (신뢰도 100%)
        #   - LiDAR가 없는 곳 (구멍 포함) → 보정된 AI depth 사용
        #   - 경계 영역 → Soft blending (optional)
        
        final_depth = sparse_depth.copy().astype(np.float32)
        
        # LiDAR가 비어있거나 구멍인 영역을 보정된 mono depth로 채움
        fill_mask = (sparse_depth == 0) | (hole_mask > 0)
        final_depth[fill_mask] = aligned_depth[fill_mask]
        
        # (Optional) 경계 부드럽게: LiDAR↔Mono 전환 경계의 계단 현상 완화
        final_depth = self._smooth_depth_boundary(
            final_depth, fill_mask, kernel_size=5
        )
        
        return final_depth
    
    # =========================================================================
    #  Monocular Depth Only (LiDAR 없을 때)
    # =========================================================================
    
    def _fill_depth_mono_only(self, rgb_image, hole_mask):
        """
        LiDAR가 없을 때: Depth Anything만으로 Dense Depth 생성
        
        절대 스케일은 없지만, 상대적 형상(벽, 바닥, 연석)은 정확합니다.
        ControlNet depth guide로 사용하기에 충분합니다.
        (ControlNet은 상대적 구조만 필요하며, 절대 스케일은 불필요)
        
        Args:
            rgb_image: (H, W, 3) BGR
            hole_mask: (H, W) uint8
        
        Returns:
            depth_guide: (H, W) float32 (0-65535 정규화)
        """
        h, w = rgb_image.shape[:2]
        
        mono_depth = self._estimate_monocular_depth(rgb_image)
        
        if mono_depth is None:
            return self._generate_pseudo_depth((h, w)).astype(np.float32)
        
        if mono_depth.shape[:2] != (h, w):
            mono_depth = cv2.resize(mono_depth, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 상대적 depth를 uint16 범위(0-65535)로 정규화
        d_min = mono_depth.min()
        d_max = mono_depth.max()
        d_range = d_max - d_min if d_max > d_min else 1.0
        
        normalized = (mono_depth - d_min) / d_range * 65535.0
        
        return normalized.astype(np.float32)
    
    def _finalize_mono_only(self, mono_depth, sparse_depth, hole_mask):
        """
        LiDAR 포인트가 부족할 때: Mono depth를 정규화하되,
        가용한 LiDAR 포인트는 그대로 사용합니다.
        """
        h, w = mono_depth.shape[:2]
        
        # Mono depth를 uint16 범위로 정규화
        d_min = mono_depth.min()
        d_max = mono_depth.max()
        d_range = d_max - d_min if d_max > d_min else 1.0
        normalized = (mono_depth - d_min) / d_range * 65535.0
        
        # LiDAR가 있는 곳은 LiDAR 값 사용
        final = normalized.astype(np.float32)
        if sparse_depth is not None:
            lidar_valid = (sparse_depth > 0) & (hole_mask == 0)
            final[lidar_valid] = sparse_depth[lidar_valid]
        
        return final
    
    # =========================================================================
    #  Depth Anything 추론 유틸리티
    # =========================================================================
    
    def _estimate_monocular_depth(self, bgr_image):
        """
        Depth Anything 모델로 단안 깊이 추정
        
        Args:
            bgr_image: (H, W, 3) BGR numpy array
        
        Returns:
            mono_depth: (H', W') float32 numpy array 또는 None (실패 시)
        """
        if self.depth_estimator is None:
            return None
        
        try:
            # BGR → RGB → PIL
            pil_img = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
            
            # 추론
            result = self.depth_estimator(pil_img)
            
            # 결과 추출 (HuggingFace pipeline 출력 형식)
            if isinstance(result, dict):
                depth_output = result.get("depth", result.get("predicted_depth", None))
            else:
                depth_output = result
            
            # PIL Image → numpy
            if hasattr(depth_output, 'numpy'):
                # torch Tensor
                mono_depth = depth_output.squeeze().cpu().numpy()
            elif isinstance(depth_output, Image.Image):
                # PIL Image
                mono_depth = np.array(depth_output).astype(np.float32)
            elif isinstance(depth_output, np.ndarray):
                mono_depth = depth_output.astype(np.float32)
            else:
                warnings.warn(f"Unexpected depth output type: {type(depth_output)}")
                return None
            
            return mono_depth.astype(np.float32)
            
        except Exception as e:
            warnings.warn(f"Monocular depth estimation error: {e}")
            return None
    
    # =========================================================================
    #  경계 부드럽게 (Depth Boundary Smoothing)
    # =========================================================================
    
    @staticmethod
    def _smooth_depth_boundary(depth_map, fill_mask, kernel_size=5):
        """
        LiDAR↔Mono depth 전환 경계를 부드럽게 처리
        
        경계(fill_mask의 에지)에서만 가우시안 블러를 적용하여
        계단 현상(depth discontinuity artifact)을 완화합니다.
        
        Args:
            depth_map: (H, W) float32
            fill_mask: (H, W) bool, Mono depth로 채운 영역
            kernel_size: 블러 커널 크기
        
        Returns:
            smoothed: (H, W) float32
        """
        # 경계 에지 추출 (fill_mask의 테두리)
        fill_uint8 = fill_mask.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(fill_uint8, kernel, iterations=2)
        eroded = cv2.erode(fill_uint8, kernel, iterations=2)
        boundary = ((dilated > 0) & (eroded == 0))  # 경계 밴드
        
        if not np.any(boundary):
            return depth_map
        
        # 경계 영역에만 블러 적용
        blurred = cv2.GaussianBlur(
            depth_map, (kernel_size, kernel_size), sigmaX=0
        )
        
        result = depth_map.copy()
        result[boundary] = blurred[boundary]
        
        return result
    
    # =========================================================================
    #  Hole Detection
    # =========================================================================
    
    def _detect_holes(self, warped_img, threshold=10):
        """
        Step 1 결과에서 여전히 구멍인 영역 감지
        
        Args:
            warped_img: Step 1 warped 이미지
            threshold: 검은색으로 간주할 임계값
        
        Returns:
            hole_mask: 구멍 영역 마스크 (255=구멍, 0=채워짐)
        """
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        hole_mask = (gray < threshold).astype(np.uint8) * 255
        
        # Morphological closing으로 작은 노이즈 제거
        kernel = np.ones((3, 3), np.uint8)
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 너무 작은 구멍은 제거 (connected component)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            hole_mask, connectivity=8
        )
        
        min_hole_size = 50  # 픽셀
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_hole_size:
                hole_mask[labels == i] = 0
        
        return hole_mask
    
    # =========================================================================
    #  Legacy Fallback: RANSAC Plane Fitting
    # =========================================================================
    
    def _fill_ground_plane(self, depth_map, hole_mask):
        """
        [Fallback] RANSAC 기반 바닥 평면 추정
        
        Depth Anything이 사용 불가능할 때의 기존 방식입니다.
        
        한계:
        - "바닥 = 평면" 가정에 의존 → 벽, 연석, 경사에서 실패
        - 이미지 텍스처를 보지 않음 → 구조 정보 활용 불가
        
        Args:
            depth_map: (H, W) float32
            hole_mask: (H, W) uint8 (255=구멍)
        
        Returns:
            filled_depth: (H, W) float32
        """
        h, w = depth_map.shape
        
        y_indices, x_indices = np.indices((h, w))
        ground_region_mask = (y_indices > h * self.ground_region_ratio)
        
        valid_mask = (depth_map > 0) & ground_region_mask & (hole_mask == 0)
        valid_count = np.sum(valid_mask)
        
        if valid_count < 100:
            warnings.warn(
                f"Insufficient valid depth points ({valid_count}). "
                f"Using simple inpainting."
            )
            return self._simple_depth_inpaint(depth_map, hole_mask)
        
        X_train = np.column_stack((
            x_indices[valid_mask],
            y_indices[valid_mask]
        ))
        y_train = depth_map[valid_mask]
        
        try:
            ransac = RANSACRegressor(
                random_state=42,
                min_samples=min(100, valid_count // 2),
                max_trials=100,
                residual_threshold=self.ransac_residual_threshold
            )
            ransac.fit(X_train, y_train)
            
            hole_y, hole_x = np.where(hole_mask > 0)
            
            if len(hole_x) == 0:
                return depth_map
            
            X_pred = np.column_stack((hole_x, hole_y))
            predicted_depth = ransac.predict(X_pred)
            
            filled_depth = depth_map.copy()
            filled_depth[hole_y, hole_x] = np.clip(predicted_depth, 0, 65535)
            
        except Exception as e:
            warnings.warn(f"RANSAC fitting failed: {e}. Using simple inpainting.")
            filled_depth = self._simple_depth_inpaint(depth_map, hole_mask)
        
        return filled_depth
    
    # =========================================================================
    #  최후 수단: Simple Depth Inpainting
    # =========================================================================
    
    def _simple_depth_inpaint(self, depth_map, hole_mask):
        """
        모든 방법이 실패했을 때의 최후 수단: 주변 값 보간
        
        cv2.inpaint는 8-bit만 지원하므로 정규화 후 처리합니다.
        
        Args:
            depth_map: (H, W) float32
            hole_mask: (H, W) uint8 (255=구멍)
        
        Returns:
            filled_depth: (H, W) float32
        """
        depth_min = np.min(depth_map[depth_map > 0]) if np.any(depth_map > 0) else 0
        depth_max = np.max(depth_map) if np.max(depth_map) > 0 else 1.0
        depth_range = depth_max - depth_min if depth_max > depth_min else 1.0
        
        depth_normalized = np.clip(
            (depth_map - depth_min) / depth_range * 255, 0, 255
        ).astype(np.uint8)
        
        filled_normalized = cv2.inpaint(
            depth_normalized, hole_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA
        )
        
        filled_depth = filled_normalized.astype(np.float32) / 255.0 * depth_range + depth_min
        
        valid_mask = (hole_mask == 0) & (depth_map > 0)
        filled_depth[valid_mask] = depth_map[valid_mask]
        
        return filled_depth
    
    # =========================================================================
    #  Pseudo Depth (LiDAR가 전혀 없을 때)
    # =========================================================================
    
    def _generate_pseudo_depth(self, shape, default_depth=10000):
        """
        LiDAR depth가 전혀 없을 때 사용할 pseudo depth
        
        Args:
            shape: (height, width)
            default_depth: 기본 깊이 값 (mm 단위)
        
        Returns:
            pseudo_depth: (H, W) float32
        """
        h, w = shape
        y_coords = np.linspace(default_depth * 2, default_depth * 0.5, h)
        pseudo_depth = np.tile(y_coords[:, np.newaxis], (1, w))
        return pseudo_depth.astype(np.float32)


# =============================================================================
#  CLI Entry Point
# =============================================================================

def main():
    """CLI 진입점"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Inpainting Step 2: Depth Guide Generation\n"
                    "(LiDAR-Guided Monocular Depth with Depth Anything)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 기본 실행 (Depth Anything + LiDAR, GPU)
  python step2_geometric_guide.py --data_root /path/to/data

  # CPU 모드
  python step2_geometric_guide.py --data_root /path/to/data --device cpu

  # LiDAR 없이 Mono-only
  python step2_geometric_guide.py --data_root /path/to/data --no_lidar

  # Legacy RANSAC 모드 (Depth Anything 비활성화)
  python step2_geometric_guide.py --data_root /path/to/data --no_mono
"""
    )
    parser.add_argument(
        '--data_root', type=str, required=True,
        help="Path to data directory (containing step1_warped/)"
    )
    parser.add_argument(
        '--no_lidar', action='store_true',
        help="Don't use LiDAR depth"
    )
    parser.add_argument(
        '--no_mono', action='store_true',
        help="Disable Depth Anything (use legacy RANSAC only)"
    )
    parser.add_argument(
        '--device', type=str, default='cuda:0',
        choices=['cuda:0', 'cuda:1', 'cpu'],
        help="Device for Depth Anything inference (default: cuda:0)"
    )
    parser.add_argument(
        '--ground_ratio', type=float, default=0.6,
        help="Ground plane ratio for RANSAC fallback (default: 0.6)"
    )
    parser.add_argument(
        '--min_alignment_points', type=int, default=50,
        help="Minimum LiDAR points for mono-metric alignment (default: 50)"
    )
    
    args = parser.parse_args()
    
    # --no_mono 옵션: Depth Anything 강제 비활성화
    if args.no_mono:
        global _DEPTH_MODEL_LOADED, _DEPTH_ESTIMATOR
        _DEPTH_MODEL_LOADED = True
        _DEPTH_ESTIMATOR = None
        print("[INFO] Depth Anything disabled by --no_mono flag. Using RANSAC fallback.")
    
    generator = GeometricGuideGenerator(
        data_root=args.data_root,
        use_lidar_depth=not args.no_lidar,
        ground_region_ratio=args.ground_ratio,
        device=args.device,
        mono_alignment_min_points=args.min_alignment_points
    )
    generator.run()


if __name__ == "__main__":
    main()
