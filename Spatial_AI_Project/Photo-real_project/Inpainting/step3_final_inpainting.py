"""
Inpainting Step 3: Generative AI Final Inpainting

Stable Diffusion + ControlNet + LoRA 기반의 High-Fidelity Inpainter
Step 1(시계열 누적)과 Step 2(기하학적 가이드)를 결합하여
생성형 AI 기반의 최종 인페인팅을 수행합니다.

Input:
    - data_root/step1_warped/: Step 1 시계열 누적 결과
    - data_root/step2_depth_guide/: Step 2 depth guide
    - data_root/images/: 원본 이미지
    - data_root/masks/: 원본 동적 객체 마스크

Output:
    - data_root/step3_final_inpainted/: 최종 인페인팅 결과
    - data_root/step3_depth/: Composited dense depth maps (uint16, mm)
    - data_root/step3_confidence/: Confidence maps (uint8, 0~255)
"""

import os
import json
import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler

class GenerativeInpainter:
    """
    Stable Diffusion + ControlNet + LoRA 기반의 High-Fidelity Inpainter
    """
    def __init__(self, lora_path=None, device="cuda"):
        print(f">>> Initializing Generative Inpainter on {device}...")
        
        # 1. Load ControlNet (Depth Model)
        # Depth 정보를 가이드로 받아 구조를 유지하는 ControlNet v1.1 로드
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth", 
            torch_dtype=torch.float16
        )

        # 2. Load Base Model (Stable Diffusion v1.5)
        # ControlNet v1.1과 호환성이 가장 좋은 SD 1.5 베이스 사용
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None  # 자율주행 데이터 처리 속도를 위해 비활성화
        ).to(device)

        # 3. Scheduler Optimization (속도 향상)
        # 기본 스케줄러보다 2배 이상 빠른 UniPC 사용
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload() # VRAM 절약 (필요시 활성화)

        # 4. Load LoRA (Optional)
        # 우리가 'training/'에서 학습시킨 Waymo 전용 LoRA 가중치가 있다면 로드
        self.use_lora = False
        if lora_path and os.path.exists(lora_path):
            print(f">>> Loading LoRA weights from {lora_path}...")
            self.pipe.load_lora_weights(lora_path)
            self.trigger_word = "WaymoStyle road" # 학습 시 설정한 트리거 단어
            self.use_lora = True
        else:
            print(">>> No LoRA weights found. Using default style.")
            self.trigger_word = "high quality realistic asphalt road"

        self.device = device

    def process(self, image, mask, depth_map, sd_resolution=512):
        """
        Input:
            image: (H, W, 3) numpy array [BGR] - 구멍난 이미지
            mask: (H, W) numpy array [0 or 255] - 인페인팅 영역
            depth_map: (H, W) numpy array [uint16 or float] - 기하학적 가이드
            sd_resolution: Stable Diffusion 입력 해상도 (기본 512, 768도 가능)
        Output:
            inpainted_image: (H, W, 3) numpy array [BGR]
        """
        # 1. Preprocessing (Numpy -> PIL)
        h_orig, w_orig = image.shape[:2]
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)
        
        # Depth Map Normalization for ControlNet
        # ControlNet은 0~255 사이의 3채널 RGB 포맷 Depth 이미지를 기대함
        if depth_map.dtype == np.uint16:
            depth_norm = (depth_map.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
        elif depth_map.dtype == np.float32 or depth_map.dtype == np.float64:
            d_min = depth_map[depth_map > 0].min() if np.any(depth_map > 0) else 0
            d_max = depth_map.max() if depth_map.max() > 0 else 1.0
            d_range = d_max - d_min if d_max > d_min else 1.0
            depth_norm = np.clip((depth_map - d_min) / d_range * 255, 0, 255).astype(np.uint8)
        else:
            depth_norm = depth_map.astype(np.uint8)
            
        depth_pil = Image.fromarray(np.stack([depth_norm]*3, axis=-1))

        # 2. 리사이즈: Stable Diffusion은 고정 해상도 입력 필요
        # SD 1.5는 512x512, 768x768 등 64의 배수 해상도 지원
        # 원본이 크면 다운스케일 후 인페인팅하고 원본 크기로 복원
        sd_w = (w_orig // 64) * 64 if w_orig <= sd_resolution * 2 else sd_resolution
        sd_h = (h_orig // 64) * 64 if h_orig <= sd_resolution * 2 else sd_resolution
        
        # 최소 64x64, 최대 1024x1024 제한
        sd_w = max(64, min(sd_w, 1024))
        sd_h = max(64, min(sd_h, 1024))
        
        needs_resize = (sd_w != w_orig) or (sd_h != h_orig)
        
        if needs_resize:
            img_pil = img_pil.resize((sd_w, sd_h), Image.LANCZOS)
            mask_pil = mask_pil.resize((sd_w, sd_h), Image.NEAREST)
            depth_pil = depth_pil.resize((sd_w, sd_h), Image.LANCZOS)

        # 3. Prompt Engineering
        # 자율주행 도로 환경에 특화된 프롬프트
        positive_prompt = (
            f"{self.trigger_word}, sharp focus, photorealistic, 8k uhd, "
            f"detailed pavement texture, driving scene, clear lane markings"
        )
        negative_prompt = (
            "blur, low quality, artifacts, watermark, text, "
            "cars, pedestrians, objects, obstacles, distortions"
        )

        # 4. Inference
        with torch.inference_mode():
            result = self.pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=img_pil,
                mask_image=mask_pil,
                control_image=depth_pil,
                width=sd_w,
                height=sd_h,
                num_inference_steps=20,  # 속도와 품질의 타협점 (UniPC 기준)
                guidance_scale=7.5,
                controlnet_conditioning_scale=0.8,  # Depth 가이드 강도 (0.0~1.0)
                strength=1.0  # 1.0 = 마스크 영역을 완전히 새로 그림
            ).images[0]

        # 5. Post-processing: 원본 해상도로 복원
        if needs_resize:
            result = result.resize((w_orig, h_orig), Image.LANCZOS)
        
        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
        # 원본 보존 (마스크 바깥 영역은 원본 그대로 유지)
        # Diffusion 모델이 미세하게 원본 영역을 바꿀 수 있으므로 강제 합성
        final_image = image.copy()
        final_image[mask > 0] = result_np[mask > 0]
        
        return final_image

def compose_depth(lidar_depth, step1_zbuffer, step2_guide, step1_meta, step2_meta):
    """
    여러 출처의 depth를 우선순위에 따라 합성.
    Pseudo depth(10m) 기반 Z-buffer는 자동 제외.

    Args:
        lidar_depth: (H, W) uint16, LiDAR sparse depth (mm). None이면 스킵.
        step1_zbuffer: (H, W) uint16, Step 1 Z-buffer depth (mm). None이면 스킵.
        step2_guide: (H, W) uint16 or float32, Step 2 dense depth guide.
        step1_meta: dict, Step 1 메타 (depth_source 포함)
        step2_meta: dict, Step 2 메타 (method 포함)

    Returns:
        depth_final: (H, W) uint16, composited depth (mm)
    """
    h, w = step2_guide.shape[:2]
    depth_final = np.zeros((h, w), dtype=np.float32)

    # 1순위: LiDAR sparse depth
    if lidar_depth is not None:
        lidar_valid = lidar_depth > 0
        depth_final[lidar_valid] = lidar_depth[lidar_valid].astype(np.float32)

    # 2순위: Step 1 Z-buffer (LiDAR 기반만, pseudo depth 제외)
    if step1_zbuffer is not None and step1_meta.get('depth_source') == 'lidar':
        zbuf_valid = (step1_zbuffer > 0) & (depth_final == 0)
        depth_final[zbuf_valid] = step1_zbuffer[zbuf_valid].astype(np.float32)

    # 3순위: Step 2 depth guide (나머지)
    remaining = depth_final == 0
    if np.any(remaining):
        depth_final[remaining] = step2_guide[remaining].astype(np.float32)

    return depth_final.clip(0, 65535).astype(np.uint16)


def create_confidence_map(orig_mask, step1_filled_mask, step1_meta, step2_meta, ai_inpainted_mask=None):
    """
    복원 출처별 신뢰도 맵 생성 (8단계 세분화).

    Args:
        orig_mask: (H, W) uint8, 원본 마스크 (0=동적, 255=정적)
        step1_filled_mask: (H, W) bool, Step 1이 채운 영역
        step1_meta: dict, depth_source 포함
        step2_meta: dict, method 포함
        ai_inpainted_mask: (H, W) bool, Step 3 AI가 생성한 영역. None이면 스킵.

    Returns:
        confidence: (H, W) uint8
    """
    h, w = orig_mask.shape[:2]
    confidence = np.full((h, w), 255, dtype=np.uint8)  # 기본: 원본 배경

    dynamic_mask = (orig_mask < 200)  # 동적 객체 영역

    # Step 1 복원 영역
    if step1_filled_mask is not None:
        step1_region = dynamic_mask & step1_filled_mask
        if step1_meta.get('depth_source') == 'lidar':
            confidence[step1_region] = 224  # LiDAR 기반 Z-buffer
        else:
            confidence[step1_region] = 192  # Pseudo depth 기반 (depth 무의미)

    # Step 2/3 복원 영역 — 사용된 방법에 따라 차등
    step1_filled = step1_filled_mask if step1_filled_mask is not None else np.zeros((h, w), dtype=bool)
    step23_region = dynamic_mask & (~step1_filled)

    method = step2_meta.get('method', 'inpaint')
    if method == 'mono_lidar':
        confidence[step23_region] = 160
    elif method == 'mono_only':
        confidence[step23_region] = 128
    elif method == 'ransac':
        confidence[step23_region] = 96
    else:  # 'inpaint' or fallback
        confidence[step23_region] = 0

    # AI 생성 영역은 더 낮은 confidence로 override
    if ai_inpainted_mask is not None:
        ai_region = dynamic_mask & ai_inpainted_mask
        # AI 생성 영역은 Step 2 방법보다 낮은 64로 설정
        confidence[ai_region] = np.minimum(confidence[ai_region], 64)

    return confidence


def run_step3(data_root, lora_path=None):
    print(">>> [Step 3] Final Inpainting with ControlNet & LoRA...")

    # 경로 설정
    step1_dir = os.path.join(data_root, 'step1_warped')       # Step 1 결과 (RGB)
    step1_depth_dir = os.path.join(data_root, 'step1_depth')  # Step 1 Z-buffer depth
    step1_meta_dir = os.path.join(data_root, 'step1_meta')    # Step 1 메타
    step2_dir = os.path.join(data_root, 'step2_depth_guide')  # Step 2 결과
    step2_meta_dir = os.path.join(data_root, 'step2_meta')    # Step 2 메타
    lidar_dir = os.path.join(data_root, 'depth_maps')         # 원본 LiDAR
    orig_dir = os.path.join(data_root, 'images')              # 원본 이미지
    mask_dir = os.path.join(data_root, 'masks')               # 동적 객체 마스크

    out_dir = os.path.join(data_root, 'step3_final_inpainted')
    out_depth_dir = os.path.join(data_root, 'step3_depth')
    out_conf_dir = os.path.join(data_root, 'step3_confidence')
    out_method_dir = os.path.join(data_root, 'step3_method_log')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_depth_dir, exist_ok=True)
    os.makedirs(out_conf_dir, exist_ok=True)
    os.makedirs(out_method_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(step1_dir) if f.endswith('.jpg') or f.endswith('.png')])

    # 모델 초기화 (루프 밖에서 한 번만 수행)
    inpainter = GenerativeInpainter(lora_path=lora_path)

    for f in tqdm(files):
        # 파일명에서 확장자 분리 후 일관된 경로 생성
        stem = os.path.splitext(f)[0]

        # 1. Load Inputs
        warped_img = cv2.imread(os.path.join(step1_dir, f))

        # Step 2 depth guide (png 우선, jpg fallback)
        depth_path = os.path.join(step2_dir, f"{stem}.png")
        if not os.path.exists(depth_path):
            depth_path = os.path.join(step2_dir, f"{stem}.jpg")
        depth_guide = cv2.imread(depth_path, -1)

        # 원본 이미지 (jpg 우선, png fallback)
        orig_path = os.path.join(orig_dir, f"{stem}.jpg")
        if not os.path.exists(orig_path):
            orig_path = os.path.join(orig_dir, f"{stem}.png")
        orig_img = cv2.imread(orig_path)

        # 마스크 (플랫 구조: masks/{stem}.png)
        mask_path = os.path.join(mask_dir, f"{stem}.png")
        orig_mask = cv2.imread(mask_path, 0)

        if warped_img is None or orig_mask is None or depth_guide is None:
            print(f"  Skipping {f}: missing inputs "
                  f"(warped={warped_img is not None}, mask={orig_mask is not None}, "
                  f"depth={depth_guide is not None})")
            continue

        # Step 1 Z-buffer depth 로드 (존재하면)
        step1_zbuffer = None
        s1_depth_path = os.path.join(step1_depth_dir, f"{stem}.png")
        if os.path.exists(s1_depth_path):
            step1_zbuffer = cv2.imread(s1_depth_path, cv2.IMREAD_UNCHANGED)

        # LiDAR sparse depth 로드 (존재하면)
        lidar_depth = None
        lidar_path = os.path.join(lidar_dir, f"{stem}.png")
        if os.path.exists(lidar_path):
            lidar_depth = cv2.imread(lidar_path, cv2.IMREAD_UNCHANGED)

        # Step 1/2 메타 로드
        step1_meta = {'depth_source': 'pseudo', 'filled_ratio': 0.0}
        s1_meta_path = os.path.join(step1_meta_dir, f"{stem}.json")
        if os.path.exists(s1_meta_path):
            with open(s1_meta_path, 'r') as mf:
                step1_meta = json.load(mf)

        step2_meta = {'method': 'inpaint'}
        s2_meta_path = os.path.join(step2_meta_dir, f"{stem}.json")
        if os.path.exists(s2_meta_path):
            with open(s2_meta_path, 'r') as mf:
                step2_meta = json.load(mf)

        # 2. Logic: Temporal Fusion으로도 못 채운 '진짜 구멍' 찾기
        missing_mask = (np.sum(warped_img, axis=2) == 0).astype(np.uint8) * 255

        # 최종 마스크: (원래 동적 객체 자리) AND (Temporal Fusion 실패 자리)
        target_mask = cv2.bitwise_and(orig_mask, missing_mask)

        # 입력 이미지 준비: 원본 + Warped(복원된 배경) 합성
        base_image = orig_img.copy()
        valid_warp = (missing_mask == 0)
        mask_bool = (orig_mask > 0)
        base_image[mask_bool & valid_warp] = warped_img[mask_bool & valid_warp]

        # Step 1이 채운 영역 마스크
        step1_filled_mask = mask_bool & valid_warp

        # 3. AI Inpainting 실행 (남은 구멍이 있을 때만)
        ai_inpainted = False
        if np.sum(target_mask) > 100:
            result = inpainter.process(base_image, target_mask, depth_guide)
            ai_inpainted = True
        else:
            result = base_image

        # AI가 실제로 생성한 영역 마스크
        ai_inpainted_mask = (target_mask > 0) if ai_inpainted else None

        # 4. Composited Depth Map 생성
        composited_depth = compose_depth(
            lidar_depth, step1_zbuffer, depth_guide,
            step1_meta, step2_meta
        )

        # 5. Confidence Map 생성
        confidence = create_confidence_map(
            orig_mask, step1_filled_mask,
            step1_meta, step2_meta,
            ai_inpainted_mask=ai_inpainted_mask
        )

        # 6. Method Log 생성
        dynamic_area = max(int(np.sum(mask_bool)), 1)
        method_log = {
            'frame': stem,
            'step1': {
                'depth_source': step1_meta.get('depth_source', 'unknown'),
                'filled_ratio': step1_meta.get('filled_ratio', 0.0),
            },
            'step2': {
                'method': step2_meta.get('method', 'unknown'),
                'lidar_anchor_count': step2_meta.get('lidar_anchor_count', 0),
                'ransac_inlier_ratio': step2_meta.get('ransac_inlier_ratio', 0.0),
                'scale': step2_meta.get('scale', 0.0),
                'shift': step2_meta.get('shift', 0.0),
            },
            'step3': {
                'ai_inpainted': ai_inpainted,
                'ai_filled_ratio': float(np.sum(target_mask > 0)) / dynamic_area if ai_inpainted else 0.0,
            }
        }

        # 7. 저장
        cv2.imwrite(os.path.join(out_dir, f), result)
        cv2.imwrite(os.path.join(out_depth_dir, f"{stem}.png"), composited_depth)
        cv2.imwrite(os.path.join(out_conf_dir, f"{stem}.png"), confidence)
        with open(os.path.join(out_method_dir, f"{stem}.json"), 'w') as mf:
            json.dump(method_log, mf, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/waymo/nre_format')
    parser.add_argument('--lora_path', type=str, default=None, help='Path to trained LoRA .safetensors')
    args = parser.parse_args()
    
    run_step3(args.data_root, args.lora_path)
