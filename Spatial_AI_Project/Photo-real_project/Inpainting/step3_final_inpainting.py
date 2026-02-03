"""
Inpainting Step 3: Multi-view Consistent Final Inpainting

Step 1(시계열 누적)과 Step 2(기하학적 가이드)를 결합하여
생성형 AI 기반의 최종 인페인팅을 수행합니다.

Input:
    - data_root/step1_warped/: Step 1 시계열 누적 결과
    - data_root/step2_depth_guide/: Step 2 depth guide
    - data_root/step2_hole_masks/: Step 2 구멍 마스크
    - data_root/images/: 원본 이미지
    - data_root/masks/: 원본 동적 객체 마스크

Output:
    - data_root/step3_final_inpainted/: 최종 인페인팅 결과
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings


class FinalInpainter:
    """
    Multi-view consistent 최종 인페인팅을 수행하는 클래스
    
    Step 1의 실제 시계열 데이터와 Step 2의 기하학적 가이드를 결합하여
    남은 구멍을 생성형 AI로 채웁니다.
    """
    
    def __init__(self, data_root, use_generative_ai=False, noise_level=5):
        """
        Args:
            data_root: 데이터 루트 디렉토리
            use_generative_ai: 생성형 AI 사용 여부 (False면 OpenCV inpainting)
            noise_level: 텍스처 노이즈 레벨 (0-255)
        """
        self.data_root = Path(data_root)
        self.use_generative_ai = use_generative_ai
        self.noise_level = noise_level
        
        # 디렉토리 설정
        self.step1_dir = self.data_root / 'step1_warped'
        self.step2_depth_dir = self.data_root / 'step2_depth_guide'
        self.step2_mask_dir = self.data_root / 'step2_hole_masks'
        self.images_dir = self.data_root / 'images'
        self.masks_dir = self.data_root / 'masks'
        self.output_dir = self.data_root / 'step3_final_inpainted'
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1, 2 결과 확인
        if not self.step1_dir.exists():
            raise FileNotFoundError(
                f"Step 1 output not found: {self.step1_dir}\n"
                f"Please run step1_temporal_accumulation.py first."
            )
        
        if not self.step2_depth_dir.exists():
            warnings.warn(
                f"Step 2 depth guide not found: {self.step2_depth_dir}\n"
                f"Will proceed without depth guidance."
            )
        
        # 파일 목록
        self.warped_files = sorted(list(self.step1_dir.glob('*.png')))
        
        if len(self.warped_files) == 0:
            raise ValueError(f"No warped images found in {self.step1_dir}")
        
        # 생성형 AI 초기화 (옵션)
        if self.use_generative_ai:
            self._initialize_generative_model()
        
        print(f"[FinalInpainter] Initialized")
        print(f"  Data root: {self.data_root}")
        print(f"  Input images: {len(self.warped_files)}")
        print(f"  Use generative AI: {self.use_generative_ai}")
        print(f"  Noise level: {self.noise_level}")
    
    def _initialize_generative_model(self):
        """
        생성형 AI 모델 초기화 (Stable Diffusion Inpainting)
        
        실제 사용 시:
        from diffusers import StableDiffusionInpaintPipeline
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(...)
        """
        print("  [INFO] Generative AI mode enabled")
        print("  [TODO] Implement Stable Diffusion pipeline loading")
        
        # Placeholder: 실제 구현 시 아래 코드 활성화
        # try:
        #     from diffusers import StableDiffusionInpaintPipeline
        #     import torch
        #     
        #     model_id = "stabilityai/stable-diffusion-2-inpainting"
        #     self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #         model_id,
        #         torch_dtype=torch.float16
        #     )
        #     self.pipe = self.pipe.to("cuda")
        #     print("  [INFO] Stable Diffusion model loaded successfully")
        # except Exception as e:
        #     warnings.warn(f"Failed to load generative model: {e}")
        #     self.use_generative_ai = False
    
    def run(self):
        """메인 파이프라인 실행"""
        print("\n" + "="*70)
        print(">>> [Step 3] Multi-view Consistent Final Inpainting Started")
        print("="*70)
        
        success_count = 0
        fail_count = 0
        
        for warped_file in tqdm(self.warped_files, desc="Inpainting frames"):
            try:
                self._process_frame(warped_file)
                success_count += 1
            except Exception as e:
                print(f"\nWarning: Failed to process {warped_file.name}: {e}")
                fail_count += 1
                continue
        
        print("\n" + "="*70)
        print(f">>> [Step 3] Complete!")
        print(f"  Success: {success_count}/{len(self.warped_files)}")
        print(f"  Failed: {fail_count}/{len(self.warped_files)}")
        print(f"  Results saved to: {self.output_dir}")
        print("="*70)
    
    def _process_frame(self, warped_file):
        """
        개별 프레임 처리
        
        Args:
            warped_file: Step 1 warped 이미지 경로
        """
        # 1. Load all inputs
        warped_img = cv2.imread(str(warped_file))
        
        if warped_img is None:
            raise ValueError(f"Failed to load warped image: {warped_file}")
        
        # 원본 이미지 (여러 확장자 시도)
        orig_img = None
        for ext in ['.jpg', '.png', '.jpeg']:
            orig_path = self.images_dir / warped_file.name.replace('.png', ext)
            if orig_path.exists():
                orig_img = cv2.imread(str(orig_path))
                if orig_img is not None:
                    break
        
        if orig_img is None:
            # 원본이 없으면 warped를 원본으로 사용
            orig_img = warped_img.copy()
        
        # Step 2 depth guide
        depth_guide = None
        depth_path = self.step2_depth_dir / warped_file.name
        if depth_path.exists():
            depth_guide = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        
        # Step 2 hole mask
        hole_mask = None
        mask_path = self.step2_mask_dir / warped_file.name
        if mask_path.exists():
            hole_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # 2. 마스크 생성 및 병합
        missing_mask = self._compute_missing_mask(warped_img, hole_mask)
        
        # 3. Base 이미지 생성 (원본 + Warped 융합)
        base_image = self._create_base_image(orig_img, warped_img, missing_mask)
        
        # 4. 최종 인페인팅
        final_result = self._run_inpainting(base_image, missing_mask, depth_guide)
        
        # 5. 저장
        output_path = self.output_dir / warped_file.name
        cv2.imwrite(str(output_path), final_result)
    
    def _compute_missing_mask(self, warped_img, hole_mask=None):
        """
        채워야 할 구멍 마스크 계산
        
        Args:
            warped_img: Step 1 warped 이미지
            hole_mask: Step 2 hole mask (선택)
        
        Returns:
            missing_mask: 255=구멍, 0=유효
        """
        # Step 1에서 채워지지 않은 영역 (검은색 픽셀)
        gray = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
        warped_missing = (gray < 10).astype(np.uint8) * 255
        
        if hole_mask is not None:
            # Step 2의 구멍 마스크와 병합
            missing_mask = cv2.bitwise_or(warped_missing, hole_mask)
        else:
            missing_mask = warped_missing
        
        # Morphological cleaning
        kernel = np.ones((3, 3), np.uint8)
        missing_mask = cv2.morphologyEx(missing_mask, cv2.MORPH_CLOSE, kernel)
        
        return missing_mask
    
    def _create_base_image(self, orig_img, warped_img, missing_mask):
        """
        원본 이미지와 warped 이미지를 융합하여 base 이미지 생성
        
        Fusion Logic:
        - Warped 이미지의 유효한 부분(시계열 누적 배경)이 우선순위가 높음
        - 구멍난 부분은 원본 이미지로 채움
        
        Args:
            orig_img: 원본 이미지
            warped_img: Step 1 warped 이미지
            missing_mask: 구멍 마스크
        
        Returns:
            base_image: 융합된 base 이미지
        """
        # 크기 맞추기
        if orig_img.shape != warped_img.shape:
            orig_img = cv2.resize(orig_img, (warped_img.shape[1], warped_img.shape[0]))
        
        base_image = orig_img.copy()
        
        # Warped 이미지의 유효한 영역 (검은색이 아닌 곳)
        valid_warp_mask = (missing_mask == 0)
        
        # 유효한 warped 데이터로 덮어쓰기
        base_image[valid_warp_mask] = warped_img[valid_warp_mask]
        
        return base_image
    
    def _run_inpainting(self, image, mask, depth_guide=None):
        """
        실제 인페인팅 수행
        
        Args:
            image: Base 이미지
            mask: 채워야 할 마스크 (255=구멍)
            depth_guide: Depth guide (선택)
        
        Returns:
            result: 인페인팅된 이미지
        """
        if self.use_generative_ai:
            # Generative AI 인페인팅
            result = self._generative_inpainting(image, mask, depth_guide)
        else:
            # OpenCV 기반 인페인팅 (Fallback)
            result = self._opencv_inpainting(image, mask)
        
        # 텍스처 노이즈 추가 (Sim-to-Real 갭 완화)
        if self.noise_level > 0:
            result = self._add_texture_noise(result, mask, self.noise_level)
        
        return result
    
    def _generative_inpainting(self, image, mask, depth_guide):
        """
        생성형 AI 기반 인페인팅 (Stable Diffusion)
        
        Args:
            image: 입력 이미지
            mask: 마스크
            depth_guide: Depth conditioning
        
        Returns:
            result: 생성된 이미지
        """
        # TODO: Stable Diffusion Inpainting 파이프라인 호출
        # 
        # from PIL import Image
        # 
        # image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # mask_pil = Image.fromarray(mask)
        # 
        # prompt = "realistic road surface, asphalt texture, outdoor scene"
        # negative_prompt = "car, vehicle, person, blur, artifacts"
        # 
        # result_pil = self.pipe(
        #     prompt=prompt,
        #     negative_prompt=negative_prompt,
        #     image=image_pil,
        #     mask_image=mask_pil,
        #     # control_image=depth_guide_pil,  # ControlNet depth conditioning
        #     num_inference_steps=50,
        #     guidance_scale=7.5
        # ).images[0]
        # 
        # result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        # return result
        
        warnings.warn("Generative AI not implemented. Falling back to OpenCV inpainting.")
        return self._opencv_inpainting(image, mask)
    
    def _opencv_inpainting(self, image, mask):
        """
        OpenCV 기반 인페인팅 (Telea 알고리즘)
        
        Args:
            image: 입력 이미지
            mask: 마스크
        
        Returns:
            result: 인페인팅된 이미지
        """
        # Telea 알고리즘 사용 (더 매끄러운 결과)
        result = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return result
    
    def _add_texture_noise(self, image, mask, noise_level):
        """
        마스크 영역에 텍스처 노이즈 추가
        
        너무 매끈한 인페인팅 결과를 자연스럽게 만들기 위함
        
        Args:
            image: 인페인팅된 이미지
            mask: 인페인팅된 영역 마스크
            noise_level: 노이즈 강도 (0-255)
        
        Returns:
            noisy_image: 노이즈가 추가된 이미지
        """
        # Gaussian noise 생성
        noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
        
        # 마스크 영역에만 노이즈 추가
        mask_3ch = (mask[:, :, np.newaxis] > 0).astype(np.float32)
        
        # 부드러운 블렌딩을 위해 마스크를 약간 blur
        mask_3ch = cv2.GaussianBlur(mask_3ch, (5, 5), 0)
        
        # 노이즈 추가
        noisy_image = image.astype(np.int16) + (noise * mask_3ch).astype(np.int16)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image


def run_generative_inpainting(image, mask, depth_guide=None):
    """
    생성형 인페인팅 함수 (독립 실행용)
    
    이 함수는 외부에서 직접 호출 가능하도록 만들어진 래퍼입니다.
    실제 Stable Diffusion 파이프라인을 여기에 구현하세요.
    
    Args:
        image: 구멍난 이미지 (numpy array, BGR)
        mask: 채워야 할 영역 (numpy array, 255=구멍)
        depth_guide: 기하학적 깊이 힌트 (numpy array, uint16)
    
    Returns:
        result: 인페인팅된 이미지 (numpy array, BGR)
    
    Example:
        >>> image = cv2.imread('input.jpg')
        >>> mask = cv2.imread('mask.png', 0)
        >>> depth = cv2.imread('depth.png', -1)
        >>> result = run_generative_inpainting(image, mask, depth)
        >>> cv2.imwrite('output.jpg', result)
    """
    # Placeholder: OpenCV inpainting
    result = cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)
    
    # 노이즈 추가 (선택)
    noise = np.random.normal(0, 5, result.shape).astype(np.int16)
    mask_3ch = (mask[:, :, np.newaxis] > 0).astype(np.float32)
    result = result.astype(np.int16) + (noise * mask_3ch).astype(np.int16)
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


def main():
    """CLI 진입점"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Inpainting Step 3: Multi-view Consistent Final Inpainting"
    )
    parser.add_argument(
        'data_root',
        type=str,
        help="Path to data directory (containing step1_warped/ and step2_depth_guide/)"
    )
    parser.add_argument(
        '--use_ai',
        action='store_true',
        help="Use generative AI (Stable Diffusion) for inpainting"
    )
    parser.add_argument(
        '--noise_level',
        type=int,
        default=5,
        help="Texture noise level (0-255, default: 5)"
    )
    
    args = parser.parse_args()
    
    # 실행
    inpainter = FinalInpainter(
        data_root=args.data_root,
        use_generative_ai=args.use_ai,
        noise_level=args.noise_level
    )
    inpainter.run()


if __name__ == "__main__":
    main()
