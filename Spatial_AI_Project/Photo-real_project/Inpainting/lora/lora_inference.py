"""
LoRA Inference & Quality Evaluation Module

학습된 Style LoRA를 사용하여 이미지를 생성하고
품질을 평가하는 모듈입니다.

Usage:
    # CLI
    python lora_inference.py \
        --lora_path ./lora_output/pytorch_lora_weights.safetensors \
        --prompt "WaymoStyle road, photorealistic asphalt" \
        --output_dir ./generated_samples

    # Python API
    from lora_inference import LoRAInference
    infer = LoRAInference(lora_path="./lora_output")
    images = infer.generate("WaymoStyle road, sharp focus", num_images=4)
    infer.save_images(images, "./output")
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch

logger = logging.getLogger(__name__)


class LoRAInference:
    """
    학습된 Style LoRA를 사용한 이미지 생성기

    Text-to-Image 및 Inpainting 모드를 지원합니다.
    """

    def __init__(
        self,
        lora_path: Optional[str] = None,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        use_controlnet: bool = False,
    ):
        """
        Args:
            lora_path: LoRA 가중치 경로 (.safetensors 파일 또는 디렉토리)
            base_model: 베이스 Stable Diffusion 모델
            device: 디바이스 (None=자동)
            use_controlnet: ControlNet (Depth) 사용 여부
        """
        self.lora_path = lora_path
        self.base_model = base_model
        self.use_controlnet = use_controlnet

        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.pipe = None
        self.trigger_word = self._load_trigger_word()

        logger.info(f"LoRAInference 초기화 (device={self.device})")
        if self.trigger_word:
            logger.info(f"  트리거 단어: {self.trigger_word}")

    def _load_trigger_word(self) -> str:
        """학습 설정에서 트리거 단어 로드"""
        if self.lora_path is None:
            return "high quality realistic road"

        lora_dir = Path(self.lora_path)
        if lora_dir.is_file():
            lora_dir = lora_dir.parent

        config_path = lora_dir / "training_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                return config.get("trigger_word", "WaymoStyle road")
            except Exception:
                pass

        return "WaymoStyle road"

    def load_pipeline(self):
        """추론 파이프라인 로드"""
        from diffusers import (
            StableDiffusionPipeline,
            StableDiffusionControlNetInpaintPipeline,
            ControlNetModel,
            UniPCMultistepScheduler,
        )

        logger.info("파이프라인 로드 중...")

        if self.use_controlnet:
            # ControlNet + Inpainting 파이프라인
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth",
                torch_dtype=torch.float16,
            )
            self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                self.base_model,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                safety_checker=None,
            ).to(self.device)
        else:
            # Text-to-Image 파이프라인
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                safety_checker=None,
            ).to(self.device)

        # 스케줄러 최적화
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )

        # LoRA 로드
        if self.lora_path:
            lora_path = Path(self.lora_path)
            if lora_path.is_file():
                # .safetensors 파일 직접 지정
                self.pipe.load_lora_weights(
                    str(lora_path.parent),
                    weight_name=lora_path.name,
                )
            elif lora_path.is_dir():
                self.pipe.load_lora_weights(str(lora_path))
            else:
                logger.warning(f"LoRA 가중치를 찾을 수 없습니다: {lora_path}")

            logger.info(f"LoRA 가중치 로드 완료: {self.lora_path}")
        else:
            logger.info("LoRA 없이 기본 SD v1.5 사용")

        # 메모리 최적화
        if self.device.type == "cuda":
            self.pipe.enable_model_cpu_offload()

        logger.info("파이프라인 로드 완료")

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Text-to-Image 생성

        Args:
            prompt: 생성 프롬프트
            negative_prompt: 네거티브 프롬프트
            num_images: 생성할 이미지 수
            width: 출력 너비
            height: 출력 높이
            num_inference_steps: 디노이징 스텝 수
            guidance_scale: CFG 스케일
            seed: 랜덤 시드

        Returns:
            images: 생성된 PIL Image 리스트
        """
        if self.pipe is None:
            self.load_pipeline()

        if negative_prompt is None:
            negative_prompt = (
                "blur, low quality, artifacts, watermark, text, "
                "cars, pedestrians, objects, distortions, anime"
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        return result.images

    def inpaint(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Union[Image.Image, np.ndarray],
        depth_map: Optional[Union[Image.Image, np.ndarray]] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_scale: float = 0.8,
        strength: float = 1.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        인페인팅 수행

        Args:
            image: 입력 이미지 (PIL or BGR numpy)
            mask: 인페인팅 마스크 (0=유지, 255=인페인팅)
            depth_map: 깊이 맵 (ControlNet 사용 시)
            prompt: 생성 프롬프트
            negative_prompt: 네거티브 프롬프트
            num_inference_steps: 스텝 수
            guidance_scale: CFG 스케일
            controlnet_scale: ControlNet 강도
            strength: 인페인팅 강도
            seed: 랜덤 시드

        Returns:
            result: 인페인팅된 이미지
        """
        if self.pipe is None:
            self.load_pipeline()

        # numpy -> PIL 변환
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)

        if isinstance(mask, np.ndarray):
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = Image.fromarray(mask)

        # 프롬프트 설정
        if prompt is None:
            prompt = (
                f"{self.trigger_word}, sharp focus, photorealistic, 8k uhd, "
                "detailed pavement texture, driving scene, clear lane markings"
            )
        if negative_prompt is None:
            negative_prompt = (
                "blur, low quality, artifacts, watermark, text, "
                "cars, pedestrians, objects, obstacles, distortions"
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # SD 호환 크기로 조정
        w, h = image.size
        sd_w = max(64, min((w // 64) * 64, 1024))
        sd_h = max(64, min((h // 64) * 64, 1024))

        img_resized = image.resize((sd_w, sd_h), Image.LANCZOS)
        mask_resized = mask.resize((sd_w, sd_h), Image.NEAREST)

        kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": img_resized,
            "mask_image": mask_resized,
            "width": sd_w,
            "height": sd_h,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "generator": generator,
        }

        # ControlNet 사용 시 depth 추가
        if self.use_controlnet and depth_map is not None:
            if isinstance(depth_map, np.ndarray):
                if depth_map.dtype == np.uint16:
                    depth_norm = (
                        depth_map.astype(np.float32) / 65535.0 * 255
                    ).astype(np.uint8)
                elif depth_map.dtype in (np.float32, np.float64):
                    d_min = (
                        depth_map[depth_map > 0].min()
                        if np.any(depth_map > 0)
                        else 0
                    )
                    d_max = depth_map.max() if depth_map.max() > 0 else 1.0
                    d_range = d_max - d_min if d_max > d_min else 1.0
                    depth_norm = np.clip(
                        (depth_map - d_min) / d_range * 255, 0, 255
                    ).astype(np.uint8)
                else:
                    depth_norm = depth_map.astype(np.uint8)

                depth_pil = Image.fromarray(
                    np.stack([depth_norm] * 3, axis=-1)
                )
            else:
                depth_pil = depth_map

            depth_pil = depth_pil.resize((sd_w, sd_h), Image.LANCZOS)
            kwargs["control_image"] = depth_pil
            kwargs["controlnet_conditioning_scale"] = controlnet_scale

        with torch.inference_mode():
            result = self.pipe(**kwargs).images[0]

        # 원본 크기로 복원
        if (sd_w, sd_h) != (w, h):
            result = result.resize((w, h), Image.LANCZOS)

        return result

    def batch_generate(
        self,
        prompts: List[str],
        output_dir: str,
        **kwargs,
    ) -> List[str]:
        """
        배치 이미지 생성

        Args:
            prompts: 프롬프트 리스트
            output_dir: 출력 디렉토리
            **kwargs: generate()에 전달할 추가 인자

        Returns:
            saved_paths: 저장된 파일 경로 리스트
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for i, prompt in enumerate(tqdm(prompts, desc="배치 생성")):
            images = self.generate(prompt=prompt, **kwargs)
            for j, img in enumerate(images):
                fname = f"generated_{i:04d}_{j:02d}.png"
                fpath = output_path / fname
                img.save(fpath, quality=95)
                saved_paths.append(str(fpath))

        logger.info(f"배치 생성 완료: {len(saved_paths)}장 → {output_dir}")
        return saved_paths

    @staticmethod
    def save_images(
        images: List[Image.Image], output_dir: str, prefix: str = "sample"
    ) -> List[str]:
        """이미지 리스트를 파일로 저장"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, img in enumerate(images):
            p = out / f"{prefix}_{i:04d}.png"
            img.save(p, quality=95)
            paths.append(str(p))
        return paths

    def compare_with_without_lora(
        self,
        prompt: str,
        seed: int = 42,
        num_inference_steps: int = 25,
        output_dir: Optional[str] = None,
    ) -> Tuple[List[Image.Image], List[Image.Image]]:
        """
        LoRA 적용 전/후 비교 이미지 생성

        Args:
            prompt: 생성 프롬프트
            seed: 고정 시드
            num_inference_steps: 스텝 수
            output_dir: 비교 이미지 저장 디렉토리

        Returns:
            (without_lora, with_lora): 각 4장씩
        """
        from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

        logger.info("LoRA 전/후 비교 생성 중...")

        # LoRA 없이 생성
        pipe_base = StableDiffusionPipeline.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(self.device)
        pipe_base.scheduler = UniPCMultistepScheduler.from_config(
            pipe_base.scheduler.config
        )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        without_lora = pipe_base(
            prompt=prompt,
            num_images_per_prompt=4,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images

        del pipe_base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # LoRA 적용 생성
        with_lora = self.generate(
            prompt=prompt,
            num_images=4,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        # 저장
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(without_lora):
                img.save(out / f"without_lora_{i:02d}.png")
            for i, img in enumerate(with_lora):
                img.save(out / f"with_lora_{i:02d}.png")

            # 비교 그리드 생성
            w, h = without_lora[0].size
            grid = Image.new("RGB", (w * 4, h * 2), "black")
            for i in range(4):
                grid.paste(without_lora[i], (i * w, 0))
                grid.paste(with_lora[i], (i * w, h))
            grid.save(out / "comparison_grid.png")
            logger.info(f"비교 이미지 저장: {out}")

        return without_lora, with_lora


# ---------------------------------------------------------------------------
# Quality Evaluation
# ---------------------------------------------------------------------------

class LoRAQualityEvaluator:
    """
    생성 이미지 품질 평가기

    지원 메트릭:
        - PSNR (Peak Signal-to-Noise Ratio)
        - SSIM (Structural Similarity Index)
        - LPIPS (Learned Perceptual Image Patch Similarity)
        - FID (Frechet Inception Distance) - 배치 평가용
    """

    @staticmethod
    def compute_psnr(
        img1: np.ndarray, img2: np.ndarray
    ) -> float:
        """PSNR 계산 (높을수록 좋음)"""
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float("inf")
        return 10 * np.log10(255.0**2 / mse)

    @staticmethod
    def compute_ssim(
        img1: np.ndarray, img2: np.ndarray
    ) -> float:
        """SSIM 계산 (높을수록 좋음, 최대 1.0)"""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return float(ssim_map.mean())

    @staticmethod
    def compute_sharpness(img: np.ndarray) -> float:
        """이미지 선명도 측정 (Laplacian variance, 높을수록 선명)"""
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def compute_color_histogram_similarity(
        img1: np.ndarray, img2: np.ndarray
    ) -> float:
        """색상 히스토그램 유사도 (0~1, 높을수록 유사)"""
        similarity = 0.0
        for c in range(3):
            hist1 = cv2.calcHist([img1], [c], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [c], None, [256], [0, 256])
            cv2.normalize(hist1, hist1)
            cv2.normalize(hist2, hist2)
            similarity += cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity / 3.0

    def evaluate_pair(
        self,
        generated: Union[str, np.ndarray, Image.Image],
        reference: Union[str, np.ndarray, Image.Image],
    ) -> Dict[str, float]:
        """
        생성 이미지와 참조 이미지 비교

        Args:
            generated: 생성된 이미지
            reference: 참조 이미지

        Returns:
            metrics: 평가 메트릭 딕셔너리
        """
        # 이미지 로드
        gen = self._load_as_numpy(generated)
        ref = self._load_as_numpy(reference)

        # 크기 맞추기
        if gen.shape != ref.shape:
            gen = cv2.resize(gen, (ref.shape[1], ref.shape[0]))

        metrics = {
            "psnr": self.compute_psnr(gen, ref),
            "ssim": self.compute_ssim(gen, ref),
            "sharpness_generated": self.compute_sharpness(gen),
            "sharpness_reference": self.compute_sharpness(ref),
            "color_similarity": self.compute_color_histogram_similarity(gen, ref),
        }

        return metrics

    def evaluate_generated_only(
        self, generated: Union[str, np.ndarray, Image.Image]
    ) -> Dict[str, float]:
        """
        생성 이미지 자체 품질 평가 (참조 불필요)

        Args:
            generated: 생성된 이미지

        Returns:
            metrics: 평가 메트릭
        """
        gen = self._load_as_numpy(generated)

        metrics = {
            "sharpness": self.compute_sharpness(gen),
            "brightness_mean": float(gen.mean()),
            "brightness_std": float(gen.std()),
            "edge_density": self._compute_edge_density(gen),
        }

        return metrics

    def evaluate_batch(
        self,
        generated_dir: str,
        reference_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        배치 품질 평가

        Args:
            generated_dir: 생성 이미지 디렉토리
            reference_dir: 참조 이미지 디렉토리 (옵션)

        Returns:
            avg_metrics: 평균 메트릭
        """
        gen_path = Path(generated_dir)
        gen_files = sorted(
            [
                p
                for p in gen_path.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )

        if not gen_files:
            return {}

        all_metrics: List[Dict[str, float]] = []

        if reference_dir:
            ref_path = Path(reference_dir)
            ref_files = sorted(
                [
                    p
                    for p in ref_path.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
            )
            pairs = list(zip(gen_files, ref_files))
            for gen_f, ref_f in tqdm(pairs, desc="배치 평가"):
                m = self.evaluate_pair(str(gen_f), str(ref_f))
                all_metrics.append(m)
        else:
            for gen_f in tqdm(gen_files, desc="배치 평가"):
                m = self.evaluate_generated_only(str(gen_f))
                all_metrics.append(m)

        # 평균 계산
        if not all_metrics:
            return {}

        avg = {}
        for key in all_metrics[0]:
            values = [m[key] for m in all_metrics if not np.isinf(m[key])]
            if values:
                avg[key] = float(np.mean(values))

        avg["num_evaluated"] = len(all_metrics)
        return avg

    @staticmethod
    def _load_as_numpy(
        img: Union[str, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """이미지를 numpy BGR로 로드"""
        if isinstance(img, str):
            arr = cv2.imread(img)
            if arr is None:
                raise FileNotFoundError(f"이미지 로드 실패: {img}")
            return arr
        elif isinstance(img, Image.Image):
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray):
            return img
        else:
            raise TypeError(f"지원하지 않는 이미지 타입: {type(img)}")

    @staticmethod
    def _compute_edge_density(img: np.ndarray) -> float:
        """에지 밀도 계산"""
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        edges = cv2.Canny(gray, 100, 200)
        return float(np.sum(edges > 0) / edges.size)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LoRA Inference & Quality Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="명령")

    # generate 서브커맨드
    gen_parser = subparsers.add_parser("generate", help="이미지 생성")
    gen_parser.add_argument("--lora_path", type=str, default=None)
    gen_parser.add_argument(
        "--prompt",
        type=str,
        default="WaymoStyle road, photorealistic asphalt, sharp focus, 8k",
    )
    gen_parser.add_argument("--negative_prompt", type=str, default=None)
    gen_parser.add_argument("--output_dir", type=str, default="./generated")
    gen_parser.add_argument("--num_images", type=int, default=4)
    gen_parser.add_argument("--width", type=int, default=512)
    gen_parser.add_argument("--height", type=int, default=512)
    gen_parser.add_argument("--steps", type=int, default=25)
    gen_parser.add_argument("--guidance_scale", type=float, default=7.5)
    gen_parser.add_argument("--seed", type=int, default=42)

    # compare 서브커맨드
    cmp_parser = subparsers.add_parser("compare", help="LoRA 전/후 비교")
    cmp_parser.add_argument("--lora_path", type=str, required=True)
    cmp_parser.add_argument(
        "--prompt",
        type=str,
        default="WaymoStyle road, photorealistic asphalt, sharp focus",
    )
    cmp_parser.add_argument("--output_dir", type=str, default="./comparison")
    cmp_parser.add_argument("--seed", type=int, default=42)

    # evaluate 서브커맨드
    eval_parser = subparsers.add_parser("evaluate", help="품질 평가")
    eval_parser.add_argument(
        "--generated_dir", type=str, required=True, help="생성 이미지 디렉토리"
    )
    eval_parser.add_argument(
        "--reference_dir", type=str, default=None, help="참조 이미지 디렉토리"
    )
    eval_parser.add_argument(
        "--output_json", type=str, default=None, help="결과 저장 경로"
    )

    args = parser.parse_args()

    if args.command == "generate":
        infer = LoRAInference(lora_path=args.lora_path)
        images = infer.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
            width=args.width,
            height=args.height,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        paths = LoRAInference.save_images(images, args.output_dir)
        print(f"생성 완료: {len(paths)}장 → {args.output_dir}")

    elif args.command == "compare":
        infer = LoRAInference(lora_path=args.lora_path)
        infer.compare_with_without_lora(
            prompt=args.prompt,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print(f"비교 완료 → {args.output_dir}")

    elif args.command == "evaluate":
        evaluator = LoRAQualityEvaluator()
        metrics = evaluator.evaluate_batch(
            generated_dir=args.generated_dir,
            reference_dir=args.reference_dir,
        )

        print("\n평가 결과:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"결과 저장: {args.output_json}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
