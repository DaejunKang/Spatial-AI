"""
Style LoRA Training Pipeline UI for Autonomous Driving Scenes

Gradio 기반 사용자 인터페이스로 다음 기능을 제공합니다:
    1. 데이터셋 준비 (Waymo/KITTI 이미지 필터링 & 프리뷰)
    2. LoRA 학습 (하이퍼파라미터 설정 & 학습 실행)
    3. 추론 & 테스트 (Text-to-Image, Inpainting)
    4. 품질 평가 (LoRA 전/후 비교, 메트릭)

Usage:
    python lora_ui.py [--port 7860] [--share]
"""

import os
import sys
import json
import time
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Lazy imports for heavy dependencies
_gradio = None
_torch = None


def _import_gradio():
    global _gradio
    if _gradio is None:
        import gradio as gr
        _gradio = gr
    return _gradio


def _import_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def scan_dataset(data_root: str) -> Dict:
    """데이터셋 디렉토리 스캔 및 통계 반환"""
    root = Path(data_root)
    stats = {
        "data_root": str(root),
        "exists": root.exists(),
        "images_dir": None,
        "num_images": 0,
        "masks_dir": None,
        "num_masks": 0,
        "has_inpainted": False,
        "has_depth": False,
        "sample_images": [],
    }

    if not root.exists():
        return stats

    # 이미지 소스 탐지
    for img_dir_name in ["step3_final_inpainted", "images"]:
        img_dir = root / img_dir_name
        if img_dir.exists():
            exts = {".jpg", ".jpeg", ".png", ".bmp"}
            images = [
                p for p in img_dir.iterdir() if p.suffix.lower() in exts
            ]
            if images:
                stats["images_dir"] = str(img_dir)
                stats["num_images"] = len(images)
                # 샘플 이미지 (최대 6장)
                stats["sample_images"] = [
                    str(p) for p in sorted(images)[:6]
                ]
                break

    # 마스크
    masks_dir = root / "masks"
    if masks_dir.exists():
        mask_files = list(masks_dir.rglob("*.png")) + list(masks_dir.rglob("*.jpg"))
        stats["masks_dir"] = str(masks_dir)
        stats["num_masks"] = len(mask_files)

    # 기타 디렉토리
    stats["has_inpainted"] = (root / "step3_final_inpainted").exists()
    stats["has_depth"] = (
        (root / "step2_depth_guide").exists()
        or (root / "depth_maps").exists()
    )

    return stats


def get_gpu_info() -> str:
    """GPU 정보 문자열 반환"""
    try:
        torch = _import_torch()
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            mem_used = torch.cuda.memory_allocated(0) / (1024**3)
            return (
                f"GPU: {name}\n"
                f"VRAM: {mem_used:.1f} / {mem_total:.1f} GB"
            )
        else:
            return "GPU: 사용 불가 (CPU 모드)"
    except Exception:
        return "GPU 정보를 가져올 수 없습니다."


# ---------------------------------------------------------------------------
# Tab 1: 데이터셋 준비
# ---------------------------------------------------------------------------

def create_dataset_tab():
    """데이터셋 준비 탭 UI 구성"""
    gr = _import_gradio()

    with gr.Tab("1. 데이터셋 준비"):
        gr.Markdown(
            """
            ## 데이터셋 준비
            Waymo/KITTI 데이터셋 경로를 지정하고 학습용 이미지를 필터링합니다.
            - **images/** 또는 **step3_final_inpainted/** 디렉토리의 이미지를 사용합니다.
            - 동적 객체가 적은 깨끗한 프레임만 선별합니다.
            """
        )

        with gr.Row():
            data_root_input = gr.Textbox(
                label="데이터 루트 경로",
                placeholder="/path/to/waymo_nre_format",
                value="",
                scale=3,
            )
            scan_btn = gr.Button("스캔", variant="primary", scale=1)

        # 스캔 결과
        dataset_info = gr.JSON(label="데이터셋 정보")
        sample_gallery = gr.Gallery(
            label="샘플 이미지 미리보기",
            columns=3,
            height=300,
        )

        with gr.Row():
            with gr.Column():
                trigger_word_input = gr.Textbox(
                    label="트리거 단어",
                    value="WaymoStyle road",
                    info="LoRA 학습 시 사용할 고유 키워드",
                )
                dynamic_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    value=0.05,
                    step=0.01,
                    label="동적 객체 임계값",
                    info="이 비율 이상 동적 객체가 있는 프레임은 제외",
                )
            with gr.Column():
                max_samples_input = gr.Number(
                    label="최대 샘플 수",
                    value=0,
                    info="0이면 제한 없음",
                )
                use_inpainted_check = gr.Checkbox(
                    label="인페인팅 결과 사용",
                    value=True,
                    info="step3_final_inpainted 결과를 우선 사용",
                )

        build_dataset_btn = gr.Button(
            "학습 데이터셋 빌드", variant="secondary"
        )
        build_output = gr.Textbox(
            label="빌드 결과", lines=5, interactive=False
        )

        # 이벤트 핸들러
        def on_scan(data_root):
            if not data_root.strip():
                return {}, []
            stats = scan_dataset(data_root.strip())
            gallery_images = []
            for p in stats.get("sample_images", []):
                try:
                    img = Image.open(p)
                    gallery_images.append(img)
                except Exception:
                    pass
            return stats, gallery_images

        def on_build_dataset(
            data_root, trigger_word, threshold, max_samples, use_inpainted
        ):
            if not data_root.strip():
                return "데이터 루트 경로를 입력하세요."
            try:
                from training_dataset_builder import TrainingDatasetBuilder

                max_s = int(max_samples) if max_samples and int(max_samples) > 0 else None
                output_dir = str(Path(data_root) / "lora_train_data")

                builder = TrainingDatasetBuilder(
                    data_root=data_root.strip(),
                    output_dir=output_dir,
                    dynamic_threshold=threshold,
                    use_step3_results=use_inpainted,
                )
                count = builder.build_lora_dataset(
                    trigger_word=trigger_word,
                    max_samples=max_s,
                )
                return (
                    f"데이터셋 빌드 완료!\n"
                    f"  저장 위치: {output_dir}/lora_dataset/\n"
                    f"  저장된 이미지: {count}장\n"
                    f"  트리거 단어: {trigger_word}\n"
                    f"  메타데이터: {output_dir}/lora_dataset/metadata.jsonl"
                )
            except Exception as e:
                return f"오류 발생: {str(e)}"

        scan_btn.click(
            fn=on_scan,
            inputs=[data_root_input],
            outputs=[dataset_info, sample_gallery],
        )

        build_dataset_btn.click(
            fn=on_build_dataset,
            inputs=[
                data_root_input,
                trigger_word_input,
                dynamic_threshold,
                max_samples_input,
                use_inpainted_check,
            ],
            outputs=[build_output],
        )

    return data_root_input, trigger_word_input


# ---------------------------------------------------------------------------
# Tab 2: LoRA 학습
# ---------------------------------------------------------------------------

# 학습 상태를 전역으로 관리
_training_state = {
    "is_training": False,
    "progress": 0,
    "max_steps": 0,
    "current_loss": 0.0,
    "log": "",
    "result": None,
}


def create_training_tab(data_root_input, trigger_word_input):
    """LoRA 학습 탭 UI 구성"""
    gr = _import_gradio()

    with gr.Tab("2. LoRA 학습"):
        gr.Markdown(
            """
            ## Style LoRA 학습
            Stable Diffusion v1.5에 자율주행 도로 스타일을 학습시킵니다.
            - U-Net의 Attention 레이어에 LoRA (Low-Rank Adaptation) 적용
            - 학습 결과: `.safetensors` 파일 (~10-50MB)
            """
        )

        # GPU 정보
        gpu_info_box = gr.Textbox(
            label="GPU 정보",
            value="스캔 중...",
            interactive=False,
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 기본 설정")
                train_data_root = gr.Textbox(
                    label="데이터 루트 경로",
                    placeholder="/path/to/waymo_nre_format",
                )
                train_output_dir = gr.Textbox(
                    label="출력 디렉토리",
                    value="./lora_output",
                )
                train_trigger = gr.Textbox(
                    label="트리거 단어",
                    value="WaymoStyle road",
                )
                pretrained_model = gr.Textbox(
                    label="베이스 모델",
                    value="runwayml/stable-diffusion-v1-5",
                )

            with gr.Column():
                gr.Markdown("### 학습 하이퍼파라미터")
                resolution = gr.Slider(
                    minimum=256,
                    maximum=768,
                    value=512,
                    step=64,
                    label="학습 해상도",
                )
                max_train_steps = gr.Slider(
                    minimum=100,
                    maximum=5000,
                    value=1000,
                    step=100,
                    label="최대 학습 스텝",
                )
                learning_rate = gr.Number(
                    label="학습률", value=1e-4
                )
                train_batch_size = gr.Slider(
                    minimum=1, maximum=8, value=1, step=1,
                    label="배치 크기",
                )
                grad_accum = gr.Slider(
                    minimum=1, maximum=16, value=4, step=1,
                    label="Gradient Accumulation",
                )

        with gr.Accordion("고급 설정", open=False):
            with gr.Row():
                lora_rank = gr.Slider(
                    minimum=4,
                    maximum=128,
                    value=16,
                    step=4,
                    label="LoRA Rank",
                    info="높을수록 표현력 증가 (메모리 사용 증가)",
                )
                lora_alpha = gr.Slider(
                    minimum=8,
                    maximum=128,
                    value=32,
                    step=8,
                    label="LoRA Alpha",
                )
            with gr.Row():
                lr_scheduler = gr.Dropdown(
                    choices=["cosine", "linear", "constant", "cosine_with_warmup"],
                    value="cosine",
                    label="LR 스케줄러",
                )
                mixed_precision = gr.Dropdown(
                    choices=["fp16", "bf16", "no"],
                    value="fp16",
                    label="Mixed Precision",
                )
            with gr.Row():
                noise_offset = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    value=0.0,
                    step=0.05,
                    label="Noise Offset",
                    info="색 대비 향상 (권장: 0.1)",
                )
                save_every = gr.Slider(
                    minimum=50,
                    maximum=1000,
                    value=250,
                    step=50,
                    label="체크포인트 저장 간격",
                )
            seed_input = gr.Number(label="랜덤 시드", value=42)
            train_max_samples = gr.Number(
                label="최대 학습 이미지 수 (0=무제한)", value=0
            )

        # 학습 시작/중지 버튼
        with gr.Row():
            start_btn = gr.Button(
                "학습 시작", variant="primary", scale=2
            )
            # 상태 표시
            training_status = gr.Textbox(
                label="학습 상태", lines=3, interactive=False, scale=3
            )

        # 학습 로그
        training_log = gr.Textbox(
            label="학습 로그",
            lines=12,
            interactive=False,
            max_lines=50,
        )

        # 결과 표시
        with gr.Row():
            result_info = gr.JSON(label="학습 결과")
            result_samples = gr.Gallery(
                label="학습 중 생성된 샘플",
                columns=4,
                height=250,
            )

        # 이벤트 핸들러
        def on_load_gpu_info():
            return get_gpu_info()

        def on_start_training(
            data_root,
            output_dir,
            trigger,
            base_model,
            res,
            steps,
            lr,
            batch,
            ga,
            rank,
            alpha,
            scheduler,
            mp,
            n_offset,
            save_every_val,
            seed,
            max_img,
        ):
            global _training_state

            if _training_state["is_training"]:
                return "이미 학습이 진행 중입니다.", "", {}, []

            if not data_root.strip():
                return "데이터 루트 경로를 입력하세요.", "", {}, []

            _training_state["is_training"] = True
            _training_state["max_steps"] = int(steps)
            _training_state["log"] = ""

            try:
                from train_style_lora import StyleLoRADataset, StyleLoRATrainer

                max_s = int(max_img) if max_img and int(max_img) > 0 else None

                # 데이터셋 생성
                dataset = StyleLoRADataset(
                    data_root=data_root.strip(),
                    trigger_word=trigger,
                    resolution=int(res),
                    max_samples=max_s,
                )

                if len(dataset) == 0:
                    _training_state["is_training"] = False
                    return "학습 이미지를 찾을 수 없습니다.", "", {}, []

                status_msg = (
                    f"학습 시작: {len(dataset)}장 이미지\n"
                    f"설정: rank={int(rank)}, steps={int(steps)}, "
                    f"lr={lr}, batch={int(batch)}"
                )

                # 트레이너 생성
                trainer = StyleLoRATrainer(
                    pretrained_model=base_model,
                    output_dir=output_dir,
                    lora_rank=int(rank),
                    lora_alpha=int(alpha),
                    learning_rate=float(lr),
                    lr_scheduler=scheduler,
                    train_batch_size=int(batch),
                    gradient_accumulation_steps=int(ga),
                    max_train_steps=int(steps),
                    mixed_precision=mp,
                    seed=int(seed),
                    save_every_n_steps=int(save_every_val),
                    validation_prompt=f"{trigger}, photorealistic asphalt road, 8k",
                    validation_every_n_steps=int(save_every_val),
                    noise_offset=float(n_offset),
                )

                # 학습 실행
                result = trainer.train(dataset)
                _training_state["is_training"] = False
                _training_state["result"] = result

                # 샘플 이미지 로드
                sample_dir = Path(output_dir) / "samples"
                sample_images = []
                if sample_dir.exists():
                    for p in sorted(sample_dir.glob("*.png")):
                        try:
                            sample_images.append(Image.open(p))
                        except Exception:
                            pass

                final_status = (
                    f"학습 완료!\n"
                    f"총 스텝: {result.get('total_steps', 0)}\n"
                    f"최종 Loss: {result.get('final_avg_loss', 0):.6f}\n"
                    f"학습 시간: {result.get('training_time_minutes', 0):.1f}분"
                )

                log_text = json.dumps(
                    result.get("config", {}), indent=2, ensure_ascii=False
                )

                return (
                    final_status,
                    log_text,
                    result.get("config", {}),
                    sample_images,
                )

            except Exception as e:
                _training_state["is_training"] = False
                import traceback
                tb = traceback.format_exc()
                return f"학습 오류: {str(e)}", tb, {}, []

        # 페이지 로드 시 GPU 정보 표시
        gpu_info_box.value = get_gpu_info()

        start_btn.click(
            fn=on_start_training,
            inputs=[
                train_data_root,
                train_output_dir,
                train_trigger,
                pretrained_model,
                resolution,
                max_train_steps,
                learning_rate,
                train_batch_size,
                grad_accum,
                lora_rank,
                lora_alpha,
                lr_scheduler,
                mixed_precision,
                noise_offset,
                save_every,
                seed_input,
                train_max_samples,
            ],
            outputs=[
                training_status,
                training_log,
                result_info,
                result_samples,
            ],
        )

    return train_output_dir


# ---------------------------------------------------------------------------
# Tab 3: 추론 & 테스트
# ---------------------------------------------------------------------------

def create_inference_tab(default_output_dir):
    """추론 & 테스트 탭 UI 구성"""
    gr = _import_gradio()

    with gr.Tab("3. 추론 & 테스트"):
        gr.Markdown(
            """
            ## LoRA 추론 & 테스트
            학습된 LoRA 가중치로 이미지를 생성하거나 인페인팅을 수행합니다.
            """
        )

        with gr.Tabs():
            # --- Text-to-Image 서브탭 ---
            with gr.Tab("Text-to-Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t2i_lora_path = gr.Textbox(
                            label="LoRA 가중치 경로",
                            placeholder="./lora_output/pytorch_lora_weights.safetensors",
                        )
                        t2i_prompt = gr.Textbox(
                            label="프롬프트",
                            value="WaymoStyle road, photorealistic asphalt, sharp focus, 8k uhd, driving scene",
                            lines=3,
                        )
                        t2i_negative = gr.Textbox(
                            label="네거티브 프롬프트",
                            value="blur, low quality, artifacts, watermark, text, cars, pedestrians",
                            lines=2,
                        )
                        with gr.Row():
                            t2i_width = gr.Slider(
                                256, 1024, 512, step=64, label="너비"
                            )
                            t2i_height = gr.Slider(
                                256, 1024, 512, step=64, label="높이"
                            )
                        with gr.Row():
                            t2i_steps = gr.Slider(
                                10, 50, 25, step=5, label="스텝 수"
                            )
                            t2i_cfg = gr.Slider(
                                1.0, 20.0, 7.5, step=0.5,
                                label="Guidance Scale",
                            )
                        with gr.Row():
                            t2i_num = gr.Slider(
                                1, 8, 4, step=1, label="생성 수"
                            )
                            t2i_seed = gr.Number(label="시드", value=42)

                        t2i_btn = gr.Button(
                            "이미지 생성", variant="primary"
                        )

                    with gr.Column(scale=2):
                        t2i_output = gr.Gallery(
                            label="생성된 이미지",
                            columns=2,
                            height=500,
                        )
                        t2i_status = gr.Textbox(
                            label="상태", interactive=False
                        )

                def on_t2i_generate(
                    lora_path, prompt, negative, width, height,
                    steps, cfg, num, seed,
                ):
                    try:
                        from lora_inference import LoRAInference

                        lp = lora_path.strip() if lora_path.strip() else None
                        infer = LoRAInference(lora_path=lp)

                        t0 = time.time()
                        images = infer.generate(
                            prompt=prompt,
                            negative_prompt=negative if negative.strip() else None,
                            num_images=int(num),
                            width=int(width),
                            height=int(height),
                            num_inference_steps=int(steps),
                            guidance_scale=float(cfg),
                            seed=int(seed) if seed else None,
                        )
                        elapsed = time.time() - t0

                        status = (
                            f"생성 완료: {len(images)}장, "
                            f"소요 시간: {elapsed:.1f}초"
                        )
                        return images, status
                    except Exception as e:
                        return [], f"오류: {str(e)}"

                t2i_btn.click(
                    fn=on_t2i_generate,
                    inputs=[
                        t2i_lora_path,
                        t2i_prompt,
                        t2i_negative,
                        t2i_width,
                        t2i_height,
                        t2i_steps,
                        t2i_cfg,
                        t2i_num,
                        t2i_seed,
                    ],
                    outputs=[t2i_output, t2i_status],
                )

            # --- Inpainting 서브탭 ---
            with gr.Tab("Inpainting"):
                gr.Markdown(
                    """
                    ### LoRA + ControlNet 인페인팅
                    학습된 LoRA를 사용하여 동적 객체 영역을 자율주행 도로 스타일로 채웁니다.
                    """
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        inp_lora_path = gr.Textbox(
                            label="LoRA 가중치 경로",
                            placeholder="./lora_output/pytorch_lora_weights.safetensors",
                        )
                        inp_image = gr.Image(
                            label="입력 이미지", type="pil"
                        )
                        inp_mask = gr.Image(
                            label="마스크 (흰색=인페인팅)", type="pil"
                        )
                        inp_prompt = gr.Textbox(
                            label="프롬프트",
                            value="WaymoStyle road, photorealistic, sharp focus",
                        )
                        with gr.Row():
                            inp_steps = gr.Slider(
                                10, 50, 20, step=5, label="스텝 수"
                            )
                            inp_cfg = gr.Slider(
                                1.0, 20.0, 7.5, step=0.5,
                                label="CFG Scale",
                            )
                        inp_strength = gr.Slider(
                            0.0, 1.0, 1.0, step=0.1,
                            label="인페인팅 강도",
                        )
                        inp_btn = gr.Button(
                            "인페인팅 실행", variant="primary"
                        )

                    with gr.Column(scale=2):
                        inp_output = gr.Image(
                            label="인페인팅 결과", type="pil"
                        )
                        inp_status = gr.Textbox(
                            label="상태", interactive=False
                        )

                def on_inpaint(
                    lora_path, image, mask, prompt, steps, cfg, strength
                ):
                    if image is None or mask is None:
                        return None, "이미지와 마스크를 업로드하세요."
                    try:
                        from lora_inference import LoRAInference

                        lp = lora_path.strip() if lora_path.strip() else None
                        infer = LoRAInference(
                            lora_path=lp, use_controlnet=False
                        )

                        # 마스크를 그레이스케일로 변환
                        mask_gray = mask.convert("L")

                        t0 = time.time()
                        result = infer.inpaint(
                            image=image,
                            mask=np.array(mask_gray),
                            prompt=prompt,
                            num_inference_steps=int(steps),
                            guidance_scale=float(cfg),
                            strength=float(strength),
                        )
                        elapsed = time.time() - t0

                        return result, f"인페인팅 완료 ({elapsed:.1f}초)"
                    except Exception as e:
                        return None, f"오류: {str(e)}"

                inp_btn.click(
                    fn=on_inpaint,
                    inputs=[
                        inp_lora_path,
                        inp_image,
                        inp_mask,
                        inp_prompt,
                        inp_steps,
                        inp_cfg,
                        inp_strength,
                    ],
                    outputs=[inp_output, inp_status],
                )


# ---------------------------------------------------------------------------
# Tab 4: 품질 평가
# ---------------------------------------------------------------------------

def create_evaluation_tab():
    """품질 평가 탭 UI 구성"""
    gr = _import_gradio()

    with gr.Tab("4. 품질 평가"):
        gr.Markdown(
            """
            ## 품질 평가 & 비교
            생성 이미지의 품질 메트릭을 계산하고 LoRA 전/후를 비교합니다.
            """
        )

        with gr.Tabs():
            # --- 개별 평가 ---
            with gr.Tab("이미지 비교"):
                with gr.Row():
                    eval_gen = gr.Image(
                        label="생성된 이미지", type="pil"
                    )
                    eval_ref = gr.Image(
                        label="참조 이미지 (선택)", type="pil"
                    )

                eval_btn = gr.Button("평가 실행", variant="primary")
                eval_result = gr.JSON(label="평가 메트릭")

                def on_evaluate(gen_img, ref_img):
                    if gen_img is None:
                        return {"error": "생성 이미지를 업로드하세요."}
                    try:
                        from lora_inference import LoRAQualityEvaluator

                        evaluator = LoRAQualityEvaluator()

                        if ref_img is not None:
                            metrics = evaluator.evaluate_pair(gen_img, ref_img)
                        else:
                            metrics = evaluator.evaluate_generated_only(gen_img)

                        # 숫자 포맷
                        formatted = {}
                        for k, v in metrics.items():
                            if isinstance(v, float):
                                formatted[k] = round(v, 4)
                            else:
                                formatted[k] = v
                        return formatted
                    except Exception as e:
                        return {"error": str(e)}

                eval_btn.click(
                    fn=on_evaluate,
                    inputs=[eval_gen, eval_ref],
                    outputs=[eval_result],
                )

            # --- LoRA 전/후 비교 ---
            with gr.Tab("LoRA 비교"):
                gr.Markdown(
                    """
                    ### LoRA 전/후 스타일 비교
                    동일한 프롬프트와 시드로 LoRA 적용 전/후 이미지를 비교합니다.
                    """
                )
                cmp_lora_path = gr.Textbox(
                    label="LoRA 가중치 경로",
                    placeholder="./lora_output/pytorch_lora_weights.safetensors",
                )
                cmp_prompt = gr.Textbox(
                    label="비교 프롬프트",
                    value="WaymoStyle road, photorealistic asphalt, sharp focus, 8k quality",
                )
                cmp_seed = gr.Number(label="시드", value=42)
                cmp_btn = gr.Button("비교 실행", variant="primary")

                with gr.Row():
                    cmp_without = gr.Gallery(
                        label="LoRA 미적용 (기본 SD v1.5)",
                        columns=4,
                        height=250,
                    )
                with gr.Row():
                    cmp_with = gr.Gallery(
                        label="LoRA 적용",
                        columns=4,
                        height=250,
                    )
                cmp_status = gr.Textbox(
                    label="상태", interactive=False
                )

                def on_compare(lora_path, prompt, seed):
                    if not lora_path.strip():
                        return [], [], "LoRA 경로를 입력하세요."
                    try:
                        from lora_inference import LoRAInference

                        infer = LoRAInference(lora_path=lora_path.strip())

                        t0 = time.time()
                        without_lora, with_lora = (
                            infer.compare_with_without_lora(
                                prompt=prompt,
                                seed=int(seed),
                            )
                        )
                        elapsed = time.time() - t0

                        return (
                            without_lora,
                            with_lora,
                            f"비교 완료 ({elapsed:.1f}초)",
                        )
                    except Exception as e:
                        return [], [], f"오류: {str(e)}"

                cmp_btn.click(
                    fn=on_compare,
                    inputs=[cmp_lora_path, cmp_prompt, cmp_seed],
                    outputs=[cmp_without, cmp_with, cmp_status],
                )

            # --- 배치 평가 ---
            with gr.Tab("배치 평가"):
                batch_gen_dir = gr.Textbox(
                    label="생성 이미지 디렉토리",
                    placeholder="./generated/",
                )
                batch_ref_dir = gr.Textbox(
                    label="참조 이미지 디렉토리 (선택)",
                    placeholder="./reference/",
                )
                batch_eval_btn = gr.Button(
                    "배치 평가", variant="primary"
                )
                batch_result = gr.JSON(label="배치 평가 결과")

                def on_batch_eval(gen_dir, ref_dir):
                    if not gen_dir.strip():
                        return {"error": "생성 이미지 디렉토리를 입력하세요."}
                    try:
                        from lora_inference import LoRAQualityEvaluator

                        evaluator = LoRAQualityEvaluator()
                        ref = ref_dir.strip() if ref_dir.strip() else None
                        metrics = evaluator.evaluate_batch(
                            generated_dir=gen_dir.strip(),
                            reference_dir=ref,
                        )
                        formatted = {}
                        for k, v in metrics.items():
                            if isinstance(v, float):
                                formatted[k] = round(v, 4)
                            else:
                                formatted[k] = v
                        return formatted
                    except Exception as e:
                        return {"error": str(e)}

                batch_eval_btn.click(
                    fn=on_batch_eval,
                    inputs=[batch_gen_dir, batch_ref_dir],
                    outputs=[batch_result],
                )


# ---------------------------------------------------------------------------
# Tab 5: Step 3 연동
# ---------------------------------------------------------------------------

def create_pipeline_tab():
    """Step 3 Inpainting 파이프라인 연동 탭"""
    gr = _import_gradio()

    with gr.Tab("5. Step 3 연동"):
        gr.Markdown(
            """
            ## Step 3 Final Inpainting 연동
            학습된 LoRA를 Step 3 파이프라인에 적용하여 전체 데이터셋을 인페인팅합니다.

            **워크플로우:**
            ```
            데이터셋 준비 → LoRA 학습 → Step 3 인페인팅 적용
            ```
            """
        )

        with gr.Row():
            with gr.Column():
                pipe_data_root = gr.Textbox(
                    label="데이터 루트 경로",
                    placeholder="/path/to/waymo_nre_format",
                )
                pipe_lora_path = gr.Textbox(
                    label="LoRA 가중치 경로",
                    placeholder="./lora_output/pytorch_lora_weights.safetensors",
                )

            with gr.Column():
                pipe_info = gr.Markdown(
                    """
                    ### 사전 요구사항
                    - Step 1 결과: `step1_warped/`
                    - Step 2 결과: `step2_depth_guide/`
                    - 마스크: `masks/`
                    - 원본 이미지: `images/`
                    """
                )

        pipe_btn = gr.Button(
            "Step 3 인페인팅 실행 (LoRA 적용)",
            variant="primary",
        )
        pipe_status = gr.Textbox(
            label="실행 상태", lines=8, interactive=False
        )

        def on_run_pipeline(data_root, lora_path):
            if not data_root.strip():
                return "데이터 루트 경로를 입력하세요."
            try:
                from step3_final_inpainting import run_step3

                lp = lora_path.strip() if lora_path.strip() else None

                t0 = time.time()
                run_step3(data_root.strip(), lora_path=lp)
                elapsed = time.time() - t0

                return (
                    f"Step 3 인페인팅 완료!\n"
                    f"소요 시간: {elapsed / 60:.1f}분\n"
                    f"결과 위치: {data_root}/step3_final_inpainted/\n"
                    f"LoRA 적용: {'예' if lp else '아니오 (기본 SD)'}"
                )
            except Exception as e:
                import traceback
                return f"오류: {str(e)}\n\n{traceback.format_exc()}"

        pipe_btn.click(
            fn=on_run_pipeline,
            inputs=[pipe_data_root, pipe_lora_path],
            outputs=[pipe_status],
        )


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def create_app() -> "gradio.Blocks":
    """Gradio 앱 생성"""
    gr = _import_gradio()

    with gr.Blocks(
        title="Style LoRA Training Pipeline - Autonomous Driving",
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 1400px; margin: auto; }
            .tab-nav button { font-size: 16px; font-weight: bold; }
        """,
    ) as app:
        gr.Markdown(
            """
            # Style LoRA Training Pipeline for Autonomous Driving Scenes

            Waymo/KITTI 데이터셋의 이미지를 사용하여 **Stable Diffusion v1.5**의
            스타일(도로 질감, 색감)을 학습시키고, 학습된 결과물(`.safetensors`)을
            **Step 3 Inpainting**에 사용할 수 있는 통합 파이프라인입니다.

            ---

            | 단계 | 설명 | 소요 시간 |
            |------|------|-----------|
            | **1. 데이터셋 준비** | Waymo/KITTI 이미지 필터링 & 프리뷰 | ~5분 |
            | **2. LoRA 학습** | SD v1.5 U-Net의 Attention 레이어 Fine-tuning | ~2-4시간 |
            | **3. 추론 & 테스트** | Text-to-Image, Inpainting | ~10초/장 |
            | **4. 품질 평가** | PSNR, SSIM, 선명도, LoRA 전/후 비교 | ~1분 |
            | **5. Step 3 연동** | 학습된 LoRA로 전체 데이터셋 인페인팅 | ~10-30분 |
            """
        )

        # 탭 구성
        data_root_input, trigger_word_input = create_dataset_tab()
        output_dir = create_training_tab(data_root_input, trigger_word_input)
        create_inference_tab(output_dir)
        create_evaluation_tab()
        create_pipeline_tab()

        gr.Markdown(
            """
            ---
            **Photo-real Project** | Style LoRA Training Pipeline v1.0
            """
        )

    return app


def main():
    """UI 서버 실행"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Style LoRA Training Pipeline UI"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="서버 포트 (기본: 7860)"
    )
    parser.add_argument(
        "--share", action="store_true", help="Gradio 공유 링크 생성"
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="서버 바인딩 주소 (기본: 0.0.0.0)",
    )
    args = parser.parse_args()

    app = create_app()
    app.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
