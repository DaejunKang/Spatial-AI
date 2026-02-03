"""
Training Dataset Builder for Generative AI Models

Inpainting 파이프라인의 결과물을 활용하여 생성형 AI 모델 학습을 위한
데이터셋을 구축합니다.

Supported Datasets:
1. LoRA Dataset: Style adaptation을 위한 깨끗한 배경 이미지
2. ControlNet Dataset: Condition-guided generation을 위한 이미지 쌍

Output Format:
- HuggingFace 표준 format (metadata.jsonl)
- train_text_to_image_lora.py 호환
- train_controlnet.py 호환
"""

import os
import json
import cv2
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Optional
import warnings


class TrainingDatasetBuilder:
    """
    생성형 AI 모델 학습을 위한 데이터셋 빌더
    
    Waymo 데이터에서 동적 객체가 적은 깨끗한 프레임을 선별하여
    LoRA 및 ControlNet 학습용 데이터셋을 생성합니다.
    """
    
    def __init__(
        self,
        data_root: str,
        output_dir: str,
        dynamic_threshold: float = 0.05,
        use_step3_results: bool = True
    ):
        """
        Args:
            data_root: 전처리/인페인팅 결과가 있는 루트 디렉토리
            output_dir: 학습 데이터셋 출력 디렉토리
            dynamic_threshold: 동적 객체 비율 임계값 (0-1)
            use_step3_results: Step 3 인페인팅 결과 사용 여부
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.dynamic_threshold = dynamic_threshold
        self.use_step3_results = use_step3_results
        
        # 입력 디렉토리 설정
        if use_step3_results and (self.data_root / 'step3_final_inpainted').exists():
            self.images_dir = self.data_root / 'step3_final_inpainted'
            print(f"  Using Step 3 inpainted images")
        else:
            self.images_dir = self.data_root / 'images'
            print(f"  Using original images")
        
        self.masks_dir = self.data_root / 'masks'
        self.depths_dir = self.data_root / 'depths'
        self.step2_depth_dir = self.data_root / 'step2_depth_guide'
        
        # 출력 디렉토리 설정
        self.lora_dir = self.output_dir / 'lora_dataset'
        self.controlnet_dir = self.output_dir / 'controlnet_dataset'
        
        # 입력 디렉토리 확인
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {self.images_dir}\n"
                f"Please run preprocessing/inpainting pipeline first."
            )
        
        print(f"[TrainingDatasetBuilder] Initialized")
        print(f"  Data root: {self.data_root}")
        print(f"  Images source: {self.images_dir}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Dynamic threshold: {self.dynamic_threshold * 100:.1f}%")
    
    def is_clean_frame(self, mask_path: Path, threshold: Optional[float] = None) -> bool:
        """
        동적 객체가 적은 깨끗한 프레임인지 판별
        
        Args:
            mask_path: 마스크 파일 경로
            threshold: 동적 객체 비율 임계값 (None이면 self.dynamic_threshold 사용)
        
        Returns:
            clean: 깨끗한 프레임 여부
        """
        if threshold is None:
            threshold = self.dynamic_threshold
        
        if not mask_path.exists():
            # 마스크가 없으면 깨끗하다고 가정
            return True
        
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return False
            
            # 동적 객체 영역 비율 계산 (mask: 0=동적, 255=정적)
            dynamic_pixels = np.sum(mask < 128)  # 어두운 픽셀 = 동적
            total_pixels = mask.size
            dynamic_ratio = dynamic_pixels / total_pixels
            
            return dynamic_ratio < threshold
            
        except Exception as e:
            warnings.warn(f"Failed to read mask {mask_path}: {e}")
            return False
    
    def build_lora_dataset(
        self,
        trigger_word: str = "WaymoStyle road",
        max_samples: Optional[int] = None
    ) -> int:
        """
        LoRA 학습용 데이터셋 생성
        
        Waymo 특유의 스타일(도로 텍스처, 조명 등)을 학습하기 위한
        깨끗한 배경 이미지를 선별하여 저장합니다.
        
        Args:
            trigger_word: LoRA 트리거 워드
            max_samples: 최대 샘플 수 (None이면 제한 없음)
        
        Returns:
            saved_count: 저장된 샘플 수
        """
        print("\n" + "="*70)
        print(">>> Building LoRA Training Dataset")
        print("="*70)
        
        # 출력 디렉토리 생성
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일 목록
        image_files = sorted(list(self.images_dir.glob('*.jpg')))
        image_files.extend(sorted(list(self.images_dir.glob('*.png'))))
        
        print(f"  Found {len(image_files)} images")
        print(f"  Trigger word: \"{trigger_word}\"")
        print(f"  Max samples: {max_samples if max_samples else 'unlimited'}")
        
        metadata = []
        saved_count = 0
        
        for img_path in tqdm(image_files, desc="Processing images"):
            # 최대 샘플 수 체크
            if max_samples and saved_count >= max_samples:
                break
            
            # 마스크 경로 찾기
            mask_path = self._find_corresponding_mask(img_path)
            
            # 깨끗한 프레임만 선별
            if not self.is_clean_frame(mask_path):
                continue
            
            # 이미지 복사
            dst_filename = f"{saved_count:06d}{img_path.suffix}"
            dst_path = self.lora_dir / dst_filename
            
            try:
                shutil.copy(img_path, dst_path)
                
                # 메타데이터 추가
                metadata.append({
                    "file_name": dst_filename,
                    "text": trigger_word,
                    "original_file": img_path.name
                })
                
                saved_count += 1
                
            except Exception as e:
                warnings.warn(f"Failed to copy {img_path}: {e}")
                continue
        
        # metadata.jsonl 저장
        metadata_path = self.lora_dir / 'metadata.jsonl'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for entry in metadata:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"\n  ✓ Saved {saved_count} LoRA training samples")
        print(f"  ✓ Dataset location: {self.lora_dir}")
        print(f"  ✓ Metadata: {metadata_path}")
        
        return saved_count
    
    def build_controlnet_dataset(
        self,
        condition_type: str = 'canny',
        prompt: str = "high quality road scene, asphalt, realistic texture",
        max_samples: Optional[int] = None,
        canny_low: int = 100,
        canny_high: int = 200
    ) -> int:
        """
        ControlNet 학습용 데이터셋 생성
        
        원본 이미지(Target)와 Condition 이미지(Canny/Depth) 쌍을 생성합니다.
        
        Args:
            condition_type: 'canny' 또는 'depth'
            prompt: 생성 프롬프트
            max_samples: 최대 샘플 수
            canny_low: Canny edge detection lower threshold
            canny_high: Canny edge detection upper threshold
        
        Returns:
            saved_count: 저장된 샘플 수
        """
        print("\n" + "="*70)
        print(f">>> Building ControlNet Training Dataset ({condition_type.upper()})")
        print("="*70)
        
        # 출력 디렉토리 생성
        train_dir = self.controlnet_dir / 'train'
        cond_dir = self.controlnet_dir / 'conditioning_images'
        train_dir.mkdir(parents=True, exist_ok=True)
        cond_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일 목록
        image_files = sorted(list(self.images_dir.glob('*.jpg')))
        image_files.extend(sorted(list(self.images_dir.glob('*.png'))))
        
        print(f"  Found {len(image_files)} images")
        print(f"  Condition type: {condition_type}")
        print(f"  Prompt: \"{prompt}\"")
        print(f"  Max samples: {max_samples if max_samples else 'unlimited'}")
        
        metadata = []
        saved_count = 0
        
        for img_path in tqdm(image_files, desc="Processing images"):
            # 최대 샘플 수 체크
            if max_samples and saved_count >= max_samples:
                break
            
            # 마스크 경로 찾기
            mask_path = self._find_corresponding_mask(img_path)
            
            # 깨끗한 프레임만 선별
            if not self.is_clean_frame(mask_path):
                continue
            
            try:
                # 1. Target Image 복사
                target_filename = f"{saved_count:06d}{img_path.suffix}"
                target_path = train_dir / target_filename
                shutil.copy(img_path, target_path)
                
                # 2. Condition Image 생성
                cond_filename = f"{saved_count:06d}_cond.png"
                cond_path = cond_dir / cond_filename
                
                if condition_type == 'canny':
                    # Canny Edge 생성
                    success = self._generate_canny_edge(
                        img_path, cond_path, canny_low, canny_high
                    )
                elif condition_type == 'depth':
                    # Depth map 복사 또는 생성
                    success = self._generate_depth_condition(img_path, cond_path)
                else:
                    raise ValueError(f"Unknown condition type: {condition_type}")
                
                if not success:
                    # Condition 생성 실패 시 target도 삭제
                    target_path.unlink()
                    continue
                
                # 3. 메타데이터 추가
                metadata.append({
                    "text": prompt,
                    "image": f"train/{target_filename}",
                    "conditioning_image": f"conditioning_images/{cond_filename}",
                    "original_file": img_path.name
                })
                
                saved_count += 1
                
            except Exception as e:
                warnings.warn(f"Failed to process {img_path}: {e}")
                continue
        
        # metadata.jsonl 저장
        metadata_path = self.controlnet_dir / 'metadata.jsonl'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for entry in metadata:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"\n  ✓ Saved {saved_count} ControlNet training samples")
        print(f"  ✓ Target images: {train_dir}")
        print(f"  ✓ Conditioning images: {cond_dir}")
        print(f"  ✓ Metadata: {metadata_path}")
        
        return saved_count
    
    def _find_corresponding_mask(self, img_path: Path) -> Path:
        """
        이미지에 대응하는 마스크 파일 찾기
        
        Args:
            img_path: 이미지 파일 경로
        
        Returns:
            mask_path: 마스크 파일 경로
        """
        # 확장자를 png로 변경
        mask_name = img_path.stem + '.png'
        mask_path = self.masks_dir / mask_name
        
        # jpg도 시도
        if not mask_path.exists():
            mask_name = img_path.stem + '.jpg'
            mask_path = self.masks_dir / mask_name
        
        return mask_path
    
    def _generate_canny_edge(
        self,
        img_path: Path,
        output_path: Path,
        low_threshold: int = 100,
        high_threshold: int = 200
    ) -> bool:
        """
        Canny edge detection으로 condition 이미지 생성
        
        Args:
            img_path: 입력 이미지 경로
            output_path: 출력 경로
            low_threshold: Canny lower threshold
            high_threshold: Canny upper threshold
        
        Returns:
            success: 성공 여부
        """
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            # Grayscale 변환
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Canny edge detection
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            
            # 저장
            cv2.imwrite(str(output_path), edges)
            
            return True
            
        except Exception as e:
            warnings.warn(f"Canny edge generation failed for {img_path}: {e}")
            return False
    
    def _generate_depth_condition(self, img_path: Path, output_path: Path) -> bool:
        """
        Depth map을 condition 이미지로 복사 또는 생성
        
        Args:
            img_path: 입력 이미지 경로
            output_path: 출력 경로
        
        Returns:
            success: 성공 여부
        """
        # Step 2 depth guide가 있으면 우선 사용
        depth_name = img_path.stem + '.png'
        
        # 1. Step 2 depth guide 확인
        step2_depth = self.step2_depth_dir / depth_name
        if step2_depth.exists():
            try:
                shutil.copy(step2_depth, output_path)
                return True
            except Exception as e:
                warnings.warn(f"Failed to copy Step 2 depth: {e}")
        
        # 2. 원본 depth 확인
        orig_depth = self.depths_dir / depth_name
        if orig_depth.exists():
            try:
                shutil.copy(orig_depth, output_path)
                return True
            except Exception as e:
                warnings.warn(f"Failed to copy original depth: {e}")
        
        # 3. Pseudo depth 생성 (fallback)
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return False
            
            h, w = img.shape[:2]
            
            # 간단한 gradient depth (위쪽 = 멀리, 아래쪽 = 가까이)
            y_coords = np.linspace(255, 0, h, dtype=np.uint8)
            pseudo_depth = np.tile(y_coords[:, np.newaxis], (1, w))
            
            cv2.imwrite(str(output_path), pseudo_depth)
            
            return True
            
        except Exception as e:
            warnings.warn(f"Pseudo depth generation failed for {img_path}: {e}")
            return False
    
    def build_all(
        self,
        lora_trigger: str = "WaymoStyle road",
        controlnet_prompt: str = "high quality road scene, asphalt, realistic texture",
        max_samples_per_dataset: Optional[int] = None
    ) -> Dict[str, int]:
        """
        모든 데이터셋 생성
        
        Args:
            lora_trigger: LoRA 트리거 워드
            controlnet_prompt: ControlNet 프롬프트
            max_samples_per_dataset: 각 데이터셋당 최대 샘플 수
        
        Returns:
            counts: 각 데이터셋별 샘플 수
        """
        counts = {}
        
        # LoRA dataset
        counts['lora'] = self.build_lora_dataset(
            trigger_word=lora_trigger,
            max_samples=max_samples_per_dataset
        )
        
        # ControlNet Canny dataset
        counts['controlnet_canny'] = self.build_controlnet_dataset(
            condition_type='canny',
            prompt=controlnet_prompt,
            max_samples=max_samples_per_dataset
        )
        
        # ControlNet Depth dataset
        counts['controlnet_depth'] = self.build_controlnet_dataset(
            condition_type='depth',
            prompt=controlnet_prompt,
            max_samples=max_samples_per_dataset
        )
        
        return counts


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(
        description="Training Dataset Builder for Generative AI Models"
    )
    
    # 입출력 경로
    parser.add_argument(
        'data_root',
        type=str,
        help="Path to preprocessing/inpainting output directory"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help="Output directory (default: data_root/gen_ai_train)"
    )
    
    # 데이터셋 선택
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['lora', 'controlnet_canny', 'controlnet_depth', 'all'],
        help="Dataset type to build"
    )
    
    # 파라미터
    parser.add_argument(
        '--dynamic_threshold',
        type=float,
        default=0.05,
        help="Dynamic object ratio threshold (0-1, default: 0.05)"
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help="Maximum samples per dataset (default: unlimited)"
    )
    parser.add_argument(
        '--lora_trigger',
        type=str,
        default="WaymoStyle road",
        help="LoRA trigger word"
    )
    parser.add_argument(
        '--controlnet_prompt',
        type=str,
        default="high quality road scene, asphalt, realistic texture",
        help="ControlNet generation prompt"
    )
    parser.add_argument(
        '--use_original',
        action='store_true',
        help="Use original images instead of Step 3 inpainted results"
    )
    parser.add_argument(
        '--canny_low',
        type=int,
        default=100,
        help="Canny edge detection lower threshold"
    )
    parser.add_argument(
        '--canny_high',
        type=int,
        default=200,
        help="Canny edge detection upper threshold"
    )
    
    args = parser.parse_args()
    
    # 출력 디렉토리 설정
    if args.output_dir is None:
        args.output_dir = str(Path(args.data_root) / 'gen_ai_train')
    
    # 빌더 초기화
    builder = TrainingDatasetBuilder(
        data_root=args.data_root,
        output_dir=args.output_dir,
        dynamic_threshold=args.dynamic_threshold,
        use_step3_results=not args.use_original
    )
    
    # 데이터셋 생성
    if args.mode == 'all':
        counts = builder.build_all(
            lora_trigger=args.lora_trigger,
            controlnet_prompt=args.controlnet_prompt,
            max_samples_per_dataset=args.max_samples
        )
        
        print("\n" + "="*70)
        print(">>> All Datasets Built Successfully!")
        print("="*70)
        for dataset_name, count in counts.items():
            print(f"  {dataset_name}: {count} samples")
        
    elif args.mode == 'lora':
        builder.build_lora_dataset(
            trigger_word=args.lora_trigger,
            max_samples=args.max_samples
        )
        
    elif args.mode == 'controlnet_canny':
        builder.build_controlnet_dataset(
            condition_type='canny',
            prompt=args.controlnet_prompt,
            max_samples=args.max_samples,
            canny_low=args.canny_low,
            canny_high=args.canny_high
        )
        
    elif args.mode == 'controlnet_depth':
        builder.build_controlnet_dataset(
            condition_type='depth',
            prompt=args.controlnet_prompt,
            max_samples=args.max_samples
        )


if __name__ == "__main__":
    main()
