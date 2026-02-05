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
"""

import os
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

    def process(self, image, mask, depth_map):
        """
        Input:
            image: (H, W, 3) numpy array [BGR] - 구멍난 이미지
            mask: (H, W) numpy array [0 or 255] - 인페인팅 영역
            depth_map: (H, W) numpy array [uint16 or float] - 기하학적 가이드
        Output:
            inpainted_image: (H, W, 3) numpy array [BGR]
        """
        # 1. Preprocessing (Numpy -> PIL)
        h, w = image.shape[:2]
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)
        
        # Depth Map Normalization for ControlNet
        # ControlNet은 0~255 사이의 3채널 RGB 포맷 Depth 이미지를 기대함
        if depth_map.dtype == np.uint16:
            depth_norm = (depth_map / 65535.0 * 255).astype(np.uint8)
        else:
            depth_norm = depth_map.astype(np.uint8)
            
        depth_pil = Image.fromarray(np.stack([depth_norm]*3, axis=-1))

        # 2. Prompt Engineering
        # 자율주행 도로 환경에 특화된 프롬프트
        positive_prompt = (
            f"{self.trigger_word}, sharp focus, photorealistic, 8k uhd, "
            f"detailed pavement texture, driving scene, clear lane markings"
        )
        negative_prompt = (
            "blur, low quality, artifacts, watermark, text, "
            "cars, pedestrians, objects, obstacles, distortions"
        )

        # 3. Inference
        with torch.inference_mode():
            result = self.pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=img_pil,
                mask_image=mask_pil,
                control_image=depth_pil,
                num_inference_steps=20, # 속도와 품질의 타협점 (UniPC 기준)
                guidance_scale=7.5,
                controlnet_conditioning_scale=0.8, # Depth 가이드를 얼마나 강하게 따를지 (0.0 ~ 1.0)
                strength=1.0 # 1.0 = 마스크 영역을 완전히 새로 그림
            ).images[0]

        # 4. Post-processing (PIL -> Numpy BGR)
        result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
        # 원본 보존 (마스크 바깥 영역은 원본 그대로 유지)
        # Diffusion 모델이 미세하게 원본 영역을 바꿀 수 있으므로 강제 합성
        final_image = image.copy()
        final_image[mask > 0] = result_np[mask > 0]
        
        return final_image

def run_step3(data_root, lora_path=None):
    print(">>> [Step 3] Final Inpainting with ControlNet & LoRA...")
    
    # 경로 설정
    step1_dir = os.path.join(data_root, 'step1_warped') # Step 1 결과
    step2_dir = os.path.join(data_root, 'step2_depth_guide') # Step 2 결과
    orig_dir = os.path.join(data_root, 'images') # 원본 (색상 참조용)
    mask_dir = os.path.join(data_root, 'masks') # 동적 객체 마스크
    
    out_dir = os.path.join(data_root, 'step3_final_inpainted')
    os.makedirs(out_dir, exist_ok=True)
    
    files = sorted([f for f in os.listdir(step1_dir) if f.endswith('.jpg') or f.endswith('.png')])
    
    # 모델 초기화 (루프 밖에서 한 번만 수행)
    inpainter = GenerativeInpainter(lora_path=lora_path)
    
    for f in tqdm(files):
        # 1. Load Inputs
        warped_img = cv2.imread(os.path.join(step1_dir, f))
        depth_guide = cv2.imread(os.path.join(step2_dir, f.replace('.jpg', '.png')), -1)
        orig_img = cv2.imread(os.path.join(orig_dir, f.replace('.png', '.jpg')))
        orig_mask = cv2.imread(os.path.join(mask_dir, f.replace('.jpg', '.png')), 0)
        
        if warped_img is None or orig_mask is None or depth_guide is None:
            # 하나라도 없으면 건너뜀 (혹은 에러 로그)
            continue

        # 2. Logic: Temporal Fusion으로도 못 채운 '진짜 구멍' 찾기
        # warped_img가 검은색(0)인 영역이 Step 1 실패 영역
        missing_mask = (np.sum(warped_img, axis=2) == 0).astype(np.uint8) * 255
        
        # 최종 마스크: (원래 동적 객체 자리) AND (Temporal Fusion 실패 자리)
        # 즉, 배경을 어디선가 가져왔으면 굳이 AI로 그릴 필요 없음 -> 구멍만 AI가 그림
        target_mask = cv2.bitwise_and(orig_mask, missing_mask)
        
        # 입력 이미지 준비: 원본 + Warped(복원된 배경) 합성
        base_image = orig_img.copy()
        valid_warp = (missing_mask == 0) # Warped 데이터가 존재하는 곳
        # 주의: Warped 이미지가 원본보다 우선순위가 낮을 수 있음 (블러 발생 시).
        # 하지만 여기선 "동적 객체가 있던 자리"에 한해서는 Warped를 씀
        mask_bool = (orig_mask > 0)
        base_image[mask_bool & valid_warp] = warped_img[mask_bool & valid_warp]

        # 3. AI Inpainting 실행 (남은 구멍이 있을 때만)
        if np.sum(target_mask) > 100: # 픽셀 100개 이상 구멍일 때만 수행
            result = inpainter.process(base_image, target_mask, depth_guide)
        else:
            result = base_image

        # 4. 저장
        cv2.imwrite(os.path.join(out_dir, f), result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data/waymo/nre_format')
    parser.add_argument('--lora_path', type=str, default=None, help='Path to trained LoRA .safetensors')
    args = parser.parse_args()
    
    run_step3(args.data_root, args.lora_path)
