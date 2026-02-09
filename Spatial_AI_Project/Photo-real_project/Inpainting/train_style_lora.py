"""
Style LoRA Training Pipeline for Autonomous Driving Scenes

Waymo/KITTI 데이터셋의 이미지를 사용하여 Stable Diffusion v1.5의
스타일(도로 질감, 색감)을 학습시키는 스크립트입니다.
학습된 결과물(.safetensors)은 Step 3 Inpainting에 사용됩니다.

Usage:
    # 기본 학습
    python train_style_lora.py \
        --data_root /path/to/waymo_nre_format \
        --output_dir ./lora_output \
        --trigger_word "WaymoStyle road"

    # 고급 설정
    python train_style_lora.py \
        --data_root /path/to/waymo_nre_format \
        --output_dir ./lora_output \
        --trigger_word "WaymoStyle road" \
        --resolution 512 \
        --train_batch_size 2 \
        --max_train_steps 1500 \
        --learning_rate 1e-4 \
        --lora_rank 16 \
        --save_every_n_steps 250

Input:
    - data_root/images/ 또는 data_root/step3_final_inpainted/: 학습용 이미지
    - data_root/masks/ (선택): 동적 객체 마스크 (깨끗한 프레임 필터링용)

Output:
    - output_dir/pytorch_lora_weights.safetensors: 학습된 LoRA 가중치
    - output_dir/training_config.json: 학습 설정 기록
    - output_dir/checkpoints/: 중간 체크포인트
    - output_dir/samples/: 학습 중 생성된 샘플 이미지
"""

import os
import sys
import json
import math
import time
import shutil
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Logging 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

class StyleLoRADataset(Dataset):
    """
    자율주행 도로 스타일 학습용 데이터셋

    Waymo/KITTI 이미지에서 동적 객체가 적은 깨끗한 프레임만 선별하여
    텍스트-이미지 쌍으로 구성합니다.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(
        self,
        data_root: str,
        trigger_word: str = "WaymoStyle road",
        resolution: int = 512,
        dynamic_threshold: float = 0.05,
        use_inpainted: bool = True,
        center_crop: bool = True,
        random_flip: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            data_root: 데이터 루트 디렉토리 (Waymo NRE 포맷)
            trigger_word: LoRA 학습용 트리거 단어
            resolution: 학습 해상도 (SD v1.5 기본 512)
            dynamic_threshold: 동적 객체 비율 임계값 (0-1)
            use_inpainted: Step 3 인페인팅 결과 우선 사용
            center_crop: 중앙 크롭 사용
            random_flip: 좌우 반전 증강
            max_samples: 최대 샘플 수 제한
        """
        super().__init__()
        self.trigger_word = trigger_word
        self.resolution = resolution
        self.dynamic_threshold = dynamic_threshold
        self.center_crop = center_crop
        self.random_flip = random_flip

        root = Path(data_root)

        # 이미지 소스 결정 (인페인팅 결과 우선)
        if use_inpainted and (root / "step3_final_inpainted").exists():
            self.images_dir = root / "step3_final_inpainted"
            logger.info("이미지 소스: step3_final_inpainted (인페인팅 결과)")
        elif (root / "images").exists():
            self.images_dir = root / "images"
            logger.info("이미지 소스: images (원본)")
        else:
            raise FileNotFoundError(
                f"이미지 디렉토리를 찾을 수 없습니다: {root}/images 또는 {root}/step3_final_inpainted"
            )

        self.masks_dir = root / "masks"

        # 이미지 목록 수집 + 필터링
        self.image_paths = self._collect_clean_images(max_samples)
        logger.info(f"총 {len(self.image_paths)}장의 학습 이미지 로드 완료")

        # 프롬프트 변형 템플릿 (다양한 도로 상황 표현)
        self.prompt_templates = [
            f"{trigger_word}, photorealistic asphalt road, driving scene, sharp focus",
            f"{trigger_word}, high quality road texture, urban street, clear markings",
            f"{trigger_word}, realistic pavement, highway, detailed surface",
            f"{trigger_word}, road scene, lane markings, photorealistic, 8k quality",
            f"{trigger_word}, asphalt texture, driving view, sunny day",
            f"{trigger_word}, urban road surface, detailed, realistic lighting",
            f"{trigger_word}, highway pavement, clear road, photographic quality",
            f"{trigger_word}, street scene, concrete road, sharp details",
        ]

    def _collect_clean_images(self, max_samples: Optional[int]) -> List[Path]:
        """동적 객체가 적은 깨끗한 프레임만 선별"""
        all_images = sorted(
            [
                p
                for p in self.images_dir.iterdir()
                if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
        )

        if not all_images:
            raise ValueError(f"이미지 파일을 찾을 수 없습니다: {self.images_dir}")

        clean_images: List[Path] = []
        masks_exist = self.masks_dir.exists()

        for img_path in all_images:
            if max_samples and len(clean_images) >= max_samples:
                break

            if masks_exist:
                # 마스크 기반 필터링
                mask_path = self.masks_dir / (img_path.stem + ".png")
                if not mask_path.exists():
                    mask_path = self.masks_dir / (img_path.stem + ".jpg")

                if mask_path.exists():
                    try:
                        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            dynamic_ratio = np.sum(mask < 128) / mask.size
                            if dynamic_ratio > self.dynamic_threshold:
                                continue  # 동적 객체가 너무 많은 프레임 건너뜀
                    except Exception:
                        pass

            clean_images.append(img_path)

        return clean_images

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]

        # 이미지 로드
        image = Image.open(img_path).convert("RGB")

        # 리사이즈 & 크롭
        image = self._preprocess_image(image)

        # 좌우 반전 증강
        if self.random_flip and torch.rand(1).item() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # PIL -> Tensor ([-1, 1] 정규화)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # (3, H, W)
        image_tensor = image_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]

        # 프롬프트 랜덤 선택
        prompt = self.prompt_templates[idx % len(self.prompt_templates)]

        return {
            "pixel_values": image_tensor,
            "input_ids": prompt,  # 텍스트는 나중에 토크나이저로 인코딩
        }

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """이미지 리사이즈 및 크롭"""
        w, h = image.size

        if self.center_crop:
            # 중앙 정사각형 크롭 후 리사이즈
            crop_size = min(w, h)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            image = image.crop((left, top, left + crop_size, top + crop_size))
            image = image.resize(
                (self.resolution, self.resolution), Image.LANCZOS
            )
        else:
            # 비율 유지 리사이즈
            ratio = self.resolution / min(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            image = image.resize((new_w, new_h), Image.LANCZOS)
            # 64의 배수로 크롭
            new_w = (new_w // 64) * 64
            new_h = (new_h // 64) * 64
            left = (image.width - new_w) // 2
            top = (image.height - new_h) // 2
            image = image.crop((left, top, left + new_w, top + new_h))

        return image


# ---------------------------------------------------------------------------
# 2. Trainer
# ---------------------------------------------------------------------------

class StyleLoRATrainer:
    """
    Stable Diffusion v1.5용 Style LoRA 학습기

    diffusers 라이브러리의 PEFT(LoRA) 통합을 활용하여
    U-Net의 attention 레이어에 Low-Rank Adaptation을 적용합니다.

    학습 전략:
        1. VAE Encoder로 이미지를 latent space로 인코딩
        2. 노이즈 추가 (Forward diffusion)
        3. Text Encoder로 프롬프트 임베딩 생성
        4. U-Net이 노이즈를 예측 (LoRA가 적용된 attention layers)
        5. MSE loss로 역전파 → LoRA 가중치만 업데이트
    """

    def __init__(
        self,
        pretrained_model: str = "runwayml/stable-diffusion-v1-5",
        output_dir: str = "./lora_output",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        learning_rate: float = 1e-4,
        lr_scheduler: str = "cosine",
        lr_warmup_ratio: float = 0.05,
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        max_train_steps: int = 1000,
        mixed_precision: str = "fp16",
        seed: int = 42,
        save_every_n_steps: int = 250,
        validation_prompt: Optional[str] = None,
        validation_every_n_steps: int = 250,
        max_grad_norm: float = 1.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_weight_decay: float = 1e-2,
        adam_epsilon: float = 1e-8,
        noise_offset: float = 0.0,
        snr_gamma: Optional[float] = None,
        device: Optional[str] = None,
    ):
        self.pretrained_model = pretrained_model
        self.output_dir = Path(output_dir)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.lr_scheduler_name = lr_scheduler
        self.lr_warmup_ratio = lr_warmup_ratio
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_train_steps = max_train_steps
        self.mixed_precision = mixed_precision
        self.seed = seed
        self.save_every_n_steps = save_every_n_steps
        self.validation_prompt = validation_prompt
        self.validation_every_n_steps = validation_every_n_steps
        self.max_grad_norm = max_grad_norm
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_weight_decay = adam_weight_decay
        self.adam_epsilon = adam_epsilon
        self.noise_offset = noise_offset
        self.snr_gamma = snr_gamma

        # 디바이스 결정
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            logger.warning("GPU를 사용할 수 없습니다. CPU 모드로 전환합니다 (매우 느림).")

        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)

        # 모델 컴포넌트 (나중에 초기화)
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.noise_scheduler = None
        self.optimizer = None
        self.lr_scheduler = None

        logger.info(f"StyleLoRATrainer 초기화 완료")
        logger.info(f"  디바이스: {self.device}")
        logger.info(f"  LoRA Rank: {lora_rank}, Alpha: {lora_alpha}")
        logger.info(f"  학습률: {learning_rate}")
        logger.info(f"  최대 학습 스텝: {max_train_steps}")

    def _load_models(self):
        """Stable Diffusion v1.5 모델 컴포넌트 로드"""
        from diffusers import (
            AutoencoderKL,
            DDPMScheduler,
            UNet2DConditionModel,
        )
        from transformers import CLIPTextModel, CLIPTokenizer

        logger.info(f"모델 로드 중: {self.pretrained_model}")

        # Tokenizer & Text Encoder
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.pretrained_model, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model, subfolder="text_encoder"
        ).to(self.device)
        self.text_encoder.requires_grad_(False)  # 고정

        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model, subfolder="vae"
        ).to(self.device)
        self.vae.requires_grad_(False)  # 고정

        # U-Net
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model, subfolder="unet"
        ).to(self.device)

        # Noise Scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model, subfolder="scheduler"
        )

        # Mixed precision
        weight_dtype = torch.float32
        if self.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.weight_dtype = weight_dtype
        self.vae.to(dtype=weight_dtype)
        self.text_encoder.to(dtype=weight_dtype)

        logger.info("모델 로드 완료")

    def _setup_lora(self):
        """LoRA 어댑터를 U-Net에 적용"""
        from peft import LoraConfig, get_peft_model

        # LoRA 적용 대상: U-Net의 Cross-Attention과 Self-Attention 레이어
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=[
                "to_k",         # Cross-Attention Key
                "to_q",         # Cross-Attention Query
                "to_v",         # Cross-Attention Value
                "to_out.0",     # Cross-Attention Output
            ],
        )

        self.unet = get_peft_model(self.unet, lora_config)

        # 학습 가능 파라미터 확인
        trainable_params = sum(
            p.numel() for p in self.unet.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.unet.parameters())
        logger.info(
            f"LoRA 적용 완료: 학습 파라미터 {trainable_params:,} / "
            f"전체 {total_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )

        return trainable_params

    def _setup_optimizer(self, num_training_steps: int):
        """옵티마이저 및 스케줄러 설정"""
        from torch.optim import AdamW

        # LoRA 파라미터만 옵티마이저에 등록
        lora_params = [p for p in self.unet.parameters() if p.requires_grad]

        self.optimizer = AdamW(
            lora_params,
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.adam_weight_decay,
            eps=self.adam_epsilon,
        )

        # 학습률 스케줄러
        warmup_steps = int(num_training_steps * self.lr_warmup_ratio)

        if self.lr_scheduler_name == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=num_training_steps, eta_min=1e-6
            )
        elif self.lr_scheduler_name == "linear":
            from torch.optim.lr_scheduler import LinearLR
            self.lr_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=num_training_steps,
            )
        elif self.lr_scheduler_name == "constant":
            from torch.optim.lr_scheduler import LambdaLR
            self.lr_scheduler = LambdaLR(self.optimizer, lambda _: 1.0)
        else:
            # cosine with warmup (기본)
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(
                    max(1, num_training_steps - warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

        logger.info(
            f"옵티마이저: AdamW (lr={self.learning_rate}), "
            f"스케줄러: {self.lr_scheduler_name} (warmup={warmup_steps})"
        )

    def _encode_prompt(self, prompts: List[str]) -> torch.Tensor:
        """텍스트 프롬프트를 CLIP 임베딩으로 변환"""
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)

        with torch.no_grad():
            encoder_hidden_states = self.text_encoder(text_input_ids)[0]

        return encoder_hidden_states.to(dtype=self.weight_dtype)

    def _compute_snr_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Min-SNR weighting (논문: Efficient Diffusion Training via Min-SNR Weighting)"""
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=timesteps.device, dtype=torch.float32
        )
        sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod[timesteps]) ** 0.5
        snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
        # Min-SNR-gamma clipping
        msnr_weights = torch.stack(
            [snr, self.snr_gamma * torch.ones_like(snr)], dim=1
        ).min(dim=1)[0] / snr
        return msnr_weights

    @torch.no_grad()
    def _generate_validation_samples(
        self,
        prompt: str,
        step: int,
        num_images: int = 4,
        num_inference_steps: int = 25,
    ):
        """학습 중 검증용 샘플 이미지 생성"""
        from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

        logger.info(f"검증 샘플 생성 중 (step {step})...")

        # 임시로 Inference 파이프라인 구성
        pipe = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model,
            unet=self.unet,
            torch_dtype=self.weight_dtype,
            safety_checker=None,
        ).to(self.device)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        images = pipe(
            prompt=prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device=self.device).manual_seed(self.seed),
        ).images

        # 그리드로 저장
        grid_w = int(math.ceil(math.sqrt(num_images)))
        grid_h = int(math.ceil(num_images / grid_w))
        w, h = images[0].size
        grid = Image.new("RGB", (w * grid_w, h * grid_h), "black")

        for i, img in enumerate(images):
            grid.paste(img, ((i % grid_w) * w, (i // grid_w) * h))

        save_path = self.output_dir / "samples" / f"step_{step:06d}.png"
        grid.save(save_path, quality=95)
        logger.info(f"  검증 샘플 저장: {save_path}")

        # 메모리 해제
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _save_checkpoint(self, step: int, loss: float):
        """체크포인트 저장"""
        from peft import PeftModel

        ckpt_dir = self.output_dir / "checkpoints" / f"step_{step:06d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # LoRA 가중치 저장
        if isinstance(self.unet, PeftModel):
            self.unet.save_pretrained(str(ckpt_dir))
        else:
            # fallback: state_dict 저장
            lora_state = {
                k: v
                for k, v in self.unet.state_dict().items()
                if "lora" in k.lower()
            }
            torch.save(lora_state, ckpt_dir / "lora_weights.pt")

        # 메타데이터 저장
        meta = {"step": step, "loss": loss, "learning_rate": self.learning_rate}
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"  체크포인트 저장: {ckpt_dir} (loss={loss:.6f})")

    def _save_final(self, training_config: dict):
        """최종 LoRA 가중치를 .safetensors로 저장"""
        from peft import PeftModel

        logger.info("최종 LoRA 가중치 저장 중...")

        # PEFT save
        if isinstance(self.unet, PeftModel):
            self.unet.save_pretrained(str(self.output_dir))

        # safetensors 파일 찾기 및 이름 정리
        safetensors_files = list(self.output_dir.glob("*.safetensors"))
        adapter_dir = self.output_dir / "default"
        if adapter_dir.exists():
            safetensors_files.extend(adapter_dir.glob("*.safetensors"))

        if safetensors_files:
            # 최종 파일을 표준 이름으로 복사
            src = safetensors_files[0]
            dst = self.output_dir / "pytorch_lora_weights.safetensors"
            if src != dst:
                shutil.copy2(src, dst)
            logger.info(f"  최종 LoRA 가중치: {dst}")
            logger.info(f"  파일 크기: {dst.stat().st_size / 1024 / 1024:.2f} MB")

        # 학습 설정 저장
        config_path = self.output_dir / "training_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)
        logger.info(f"  학습 설정: {config_path}")

    def train(self, dataset: StyleLoRADataset) -> Dict:
        """
        메인 학습 루프

        Args:
            dataset: StyleLoRADataset 인스턴스

        Returns:
            training_log: 학습 기록 딕셔너리
        """
        start_time = time.time()

        # 1. 모델 로드
        self._load_models()

        # 2. LoRA 적용
        trainable_params = self._setup_lora()

        # 3. DataLoader 구성
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
            drop_last=True,
        )

        # 4. 옵티마이저 & 스케줄러
        self._setup_optimizer(self.max_train_steps)

        # 5. Mixed precision scaler
        scaler = None
        if self.mixed_precision == "fp16" and self.device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")

        # 6. 학습 설정 기록
        training_config = {
            "pretrained_model": self.pretrained_model,
            "trigger_word": dataset.trigger_word,
            "resolution": dataset.resolution,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "learning_rate": self.learning_rate,
            "lr_scheduler": self.lr_scheduler_name,
            "train_batch_size": self.train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_train_steps": self.max_train_steps,
            "mixed_precision": self.mixed_precision,
            "seed": self.seed,
            "num_train_images": len(dataset),
            "trainable_params": trainable_params,
            "noise_offset": self.noise_offset,
            "snr_gamma": self.snr_gamma,
            "device": str(self.device),
        }

        logger.info("=" * 70)
        logger.info("LoRA 학습 시작")
        logger.info("=" * 70)
        for k, v in training_config.items():
            logger.info(f"  {k}: {v}")
        logger.info("=" * 70)

        # 재현성
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

        # 7. 학습 루프
        self.unet.train()
        global_step = 0
        running_loss = 0.0
        loss_history: List[float] = []
        best_loss = float("inf")

        progress_bar = tqdm(
            total=self.max_train_steps, desc="LoRA 학습", unit="step"
        )

        while global_step < self.max_train_steps:
            for batch in dataloader:
                if global_step >= self.max_train_steps:
                    break

                pixel_values = batch["pixel_values"].to(
                    self.device, dtype=self.weight_dtype
                )
                prompts = batch["input_ids"]  # List[str]

                # === Forward Pass ===
                with torch.no_grad():
                    # (a) VAE Encode: Image -> Latent
                    latents = self.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor

                # (b) 노이즈 샘플링
                noise = torch.randn_like(latents)
                if self.noise_offset > 0:
                    noise += self.noise_offset * torch.randn(
                        latents.shape[0],
                        latents.shape[1],
                        1,
                        1,
                        device=latents.device,
                        dtype=latents.dtype,
                    )

                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                    dtype=torch.long,
                )

                # (c) Forward diffusion (노이즈 추가)
                noisy_latents = self.noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # (d) 프롬프트 인코딩
                encoder_hidden_states = self._encode_prompt(prompts)

                # (e) U-Net 예측
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        model_pred = self.unet(
                            noisy_latents.to(dtype=self.weight_dtype),
                            timesteps,
                            encoder_hidden_states,
                        ).sample
                else:
                    model_pred = self.unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample

                # (f) 타겟 결정 (epsilon 또는 v-prediction)
                if self.noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif self.noise_scheduler.config.prediction_type == "v_prediction":
                    target = self.noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type: "
                        f"{self.noise_scheduler.config.prediction_type}"
                    )

                # (g) Loss 계산
                if self.snr_gamma is not None:
                    snr_weights = self._compute_snr_weights(timesteps)
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * snr_weights
                    )
                    loss = loss.mean()
                else:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )

                # Gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                # === Backward Pass ===
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation step
                if (global_step + 1) % self.gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.unet.parameters() if p.requires_grad],
                        self.max_grad_norm,
                    )
                    if scaler is not None:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # 기록
                step_loss = loss.item() * self.gradient_accumulation_steps
                running_loss += step_loss
                loss_history.append(step_loss)
                global_step += 1

                # Progress bar 업데이트
                avg_loss = running_loss / global_step
                current_lr = self.optimizer.param_groups[0]["lr"]
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=f"{step_loss:.4f}",
                    avg_loss=f"{avg_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                )

                # 체크포인트 저장
                if (
                    self.save_every_n_steps > 0
                    and global_step % self.save_every_n_steps == 0
                ):
                    self._save_checkpoint(global_step, avg_loss)

                    if avg_loss < best_loss:
                        best_loss = avg_loss

                # 검증 샘플 생성
                if (
                    self.validation_prompt
                    and self.validation_every_n_steps > 0
                    and global_step % self.validation_every_n_steps == 0
                ):
                    self.unet.eval()
                    self._generate_validation_samples(
                        self.validation_prompt, global_step
                    )
                    self.unet.train()

        progress_bar.close()

        # 8. 학습 완료 — 최종 저장
        elapsed = time.time() - start_time
        training_config["training_time_seconds"] = elapsed
        training_config["final_loss"] = (
            running_loss / max(global_step, 1)
        )
        training_config["best_loss"] = best_loss

        self._save_final(training_config)

        # 학습 기록 저장
        log_path = self.output_dir / "training_log.json"
        log_data = {
            "config": training_config,
            "loss_history": loss_history[-100:],  # 마지막 100개만
            "final_avg_loss": running_loss / max(global_step, 1),
            "total_steps": global_step,
            "training_time_minutes": elapsed / 60,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        logger.info("=" * 70)
        logger.info("LoRA 학습 완료!")
        logger.info(f"  총 스텝: {global_step}")
        logger.info(f"  최종 평균 Loss: {running_loss / max(global_step, 1):.6f}")
        logger.info(f"  최소 Loss: {best_loss:.6f}")
        logger.info(f"  학습 시간: {elapsed / 60:.1f}분")
        logger.info(f"  출력 디렉토리: {self.output_dir}")
        logger.info("=" * 70)

        return log_data


# ---------------------------------------------------------------------------
# 3. Quick-start 함수
# ---------------------------------------------------------------------------

def train_style_lora(
    data_root: str,
    output_dir: str = "./lora_output",
    trigger_word: str = "WaymoStyle road",
    resolution: int = 512,
    train_batch_size: int = 1,
    max_train_steps: int = 1000,
    learning_rate: float = 1e-4,
    lora_rank: int = 16,
    save_every_n_steps: int = 250,
    max_samples: Optional[int] = None,
    pretrained_model: str = "runwayml/stable-diffusion-v1-5",
    validation_prompt: Optional[str] = None,
) -> Dict:
    """
    Quick-start 함수: 한 줄로 LoRA 학습을 시작합니다.

    Args:
        data_root: 데이터 루트 (Waymo NRE 포맷)
        output_dir: 출력 디렉토리
        trigger_word: 트리거 단어
        resolution: 학습 해상도
        train_batch_size: 배치 크기
        max_train_steps: 최대 학습 스텝
        learning_rate: 학습률
        lora_rank: LoRA rank
        save_every_n_steps: 체크포인트 저장 간격
        max_samples: 최대 학습 이미지 수
        pretrained_model: 베이스 모델
        validation_prompt: 검증 프롬프트

    Returns:
        training_log: 학습 기록
    """
    # 검증 프롬프트 자동 설정
    if validation_prompt is None:
        validation_prompt = f"{trigger_word}, photorealistic asphalt road, sharp focus, 8k"

    # 데이터셋 생성
    dataset = StyleLoRADataset(
        data_root=data_root,
        trigger_word=trigger_word,
        resolution=resolution,
        max_samples=max_samples,
    )

    # 트레이너 생성
    trainer = StyleLoRATrainer(
        pretrained_model=pretrained_model,
        output_dir=output_dir,
        lora_rank=lora_rank,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        max_train_steps=max_train_steps,
        save_every_n_steps=save_every_n_steps,
        validation_prompt=validation_prompt,
    )

    # 학습 실행
    return trainer.train(dataset)


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Style LoRA Training Pipeline for Autonomous Driving Scenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 학습
  python train_style_lora.py --data_root /path/to/waymo_nre_format

  # 고급 설정
  python train_style_lora.py \\
      --data_root /path/to/waymo_nre_format \\
      --output_dir ./my_lora \\
      --trigger_word "WaymoStyle road" \\
      --resolution 512 \\
      --train_batch_size 2 \\
      --max_train_steps 1500 \\
      --lora_rank 16

  # KITTI 데이터셋 사용
  python train_style_lora.py \\
      --data_root /path/to/kitti_data \\
      --trigger_word "KITTIStyle road" \\
      --resolution 512
        """,
    )

    # 필수 인자
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="데이터 루트 디렉토리 (images/ 또는 step3_final_inpainted/ 포함)",
    )

    # 출력 설정
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_output",
        help="학습 결과 출력 디렉토리 (기본: ./lora_output)",
    )

    # 학습 하이퍼파라미터
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="베이스 Stable Diffusion 모델 (기본: runwayml/stable-diffusion-v1-5)",
    )
    parser.add_argument(
        "--trigger_word",
        type=str,
        default="WaymoStyle road",
        help="LoRA 트리거 단어 (기본: 'WaymoStyle road')",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="학습 이미지 해상도 (기본: 512)",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="학습 배치 크기 (기본: 1)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation 스텝 (기본: 4)",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="최대 학습 스텝 (기본: 1000)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="학습률 (기본: 1e-4)",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant", "cosine_with_warmup"],
        help="학습률 스케줄러 (기본: cosine)",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (기본: 16, 높을수록 표현력 증가)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (기본: 32)",
    )

    # 저장 및 검증
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=250,
        help="체크포인트 저장 간격 (기본: 250)",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="검증용 프롬프트 (미설정 시 자동 생성)",
    )
    parser.add_argument(
        "--validation_every_n_steps",
        type=int,
        default=250,
        help="검증 샘플 생성 간격 (기본: 250)",
    )

    # 데이터 설정
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="최대 학습 이미지 수 (기본: 제한 없음)",
    )
    parser.add_argument(
        "--dynamic_threshold",
        type=float,
        default=0.05,
        help="동적 객체 비율 임계값 (기본: 0.05)",
    )
    parser.add_argument(
        "--use_original",
        action="store_true",
        help="인페인팅 결과 대신 원본 이미지 사용",
    )
    parser.add_argument(
        "--no_flip",
        action="store_true",
        help="좌우 반전 증강 비활성화",
    )

    # 고급 설정
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "no"],
        help="Mixed precision (기본: fp16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본: 42)",
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0.0,
        help="Noise offset (기본: 0.0, 권장: 0.1)",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="Min-SNR gamma (기본: None, 권장: 5.0)",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (기본: 1.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="디바이스 (기본: 자동 감지)",
    )

    args = parser.parse_args()

    # 검증 프롬프트 자동 생성
    if args.validation_prompt is None:
        args.validation_prompt = (
            f"{args.trigger_word}, photorealistic asphalt road, sharp focus, 8k"
        )

    # 데이터셋 생성
    dataset = StyleLoRADataset(
        data_root=args.data_root,
        trigger_word=args.trigger_word,
        resolution=args.resolution,
        dynamic_threshold=args.dynamic_threshold,
        use_inpainted=not args.use_original,
        random_flip=not args.no_flip,
        max_samples=args.max_samples,
    )

    if len(dataset) == 0:
        logger.error("학습 이미지를 찾을 수 없습니다. data_root 경로를 확인하세요.")
        sys.exit(1)

    # 트레이너 생성 및 학습 실행
    trainer = StyleLoRATrainer(
        pretrained_model=args.pretrained_model,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_train_steps=args.max_train_steps,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        save_every_n_steps=args.save_every_n_steps,
        validation_prompt=args.validation_prompt,
        validation_every_n_steps=args.validation_every_n_steps,
        max_grad_norm=args.max_grad_norm,
        noise_offset=args.noise_offset,
        snr_gamma=args.snr_gamma,
        device=args.device,
    )

    result = trainer.train(dataset)

    print(f"\n학습 완료! 결과: {args.output_dir}")
    print(f"  LoRA 가중치: {args.output_dir}/pytorch_lora_weights.safetensors")
    print(f"  Step 3에서 사용: python step3_final_inpainting.py "
          f"--lora_path {args.output_dir}/pytorch_lora_weights.safetensors")


if __name__ == "__main__":
    main()
