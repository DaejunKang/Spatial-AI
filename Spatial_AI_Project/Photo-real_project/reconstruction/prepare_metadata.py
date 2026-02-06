"""
JSON Metadata 생성 스크립트

Inpainting 결과 + Parsing stage 메타데이터를 결합하여
3DGS/3DGUT 학습용 JSON 생성

Input:
    - final_inpainted/: Inpainting 완료된 이미지
    - poses/: 카메라 포즈 및 메타데이터 (Parsing stage 출력)

Output:
    - train_meta/train_pairs.json: 학습용 메타데이터
    - (선택) val_meta/val_pairs.json: 검증용 메타데이터
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def prepare_3dgs_metadata(
    data_root: str,
    output_file: str,
    train_ratio: float = 0.9,
    camera_filter: list = None
):
    """
    3DGS용 메타데이터 생성 (Static Scene)
    
    Args:
        data_root: NRE 포맷 데이터 루트
        output_file: 출력 JSON 파일 경로
        train_ratio: 학습/검증 분할 비율
        camera_filter: 사용할 카메라 리스트 (None = 전체)
    """
    data_root = Path(data_root)
    
    # 디렉토리 확인
    final_inpainted_dir = data_root / 'final_inpainted'
    poses_dir = data_root / 'poses'
    
    if not final_inpainted_dir.exists():
        raise FileNotFoundError(f"final_inpainted not found: {final_inpainted_dir}")
    
    if not poses_dir.exists():
        raise FileNotFoundError(f"poses not found: {poses_dir}")
    
    # 이미지 파일 수집
    image_files = sorted(list(final_inpainted_dir.glob('*.jpg')))
    if len(image_files) == 0:
        image_files = sorted(list(final_inpainted_dir.glob('*.png')))
    
    print(f"Found {len(image_files)} inpainted images")
    
    # 메타데이터 생성
    metadata = []
    
    for img_file in tqdm(image_files, desc="Processing frames"):
        # 대응하는 pose 파일 찾기
        frame_name = img_file.stem  # 확장자 제외
        
        # 카메라 이름 추출 (예: seq0_000000_FRONT.jpg → FRONT)
        parts = frame_name.split('_')
        if len(parts) >= 3:
            camera_name = parts[-1]  # 마지막 부분이 카메라 이름
            base_name = '_'.join(parts[:-1])  # 카메라 제외한 부분
        else:
            camera_name = 'UNKNOWN'
            base_name = frame_name
        
        # 카메라 필터링
        if camera_filter and camera_name not in camera_filter:
            continue
        
        # Pose JSON 로드
        pose_file = poses_dir / f"{base_name}.json"
        
        if not pose_file.exists():
            print(f"Warning: Pose file not found for {img_file.name}, skipping")
            continue
        
        with open(pose_file, 'r') as f:
            pose_data = json.load(f)
        
        # 카메라 데이터 추출
        if camera_name not in pose_data.get('cameras', {}):
            print(f"Warning: Camera {camera_name} not in pose data, skipping")
            continue
        
        cam_data = pose_data['cameras'][camera_name]
        
        # 메타데이터 아이템 생성
        item = {
            "file_path": str(img_file.relative_to(data_root)),
            "transform_matrix": cam_data['pose'],  # [4x4] flattened
            "intrinsics": cam_data['intrinsics'],  # [fx, fy, cx, cy, k1, ...]
            "width": cam_data['width'],
            "height": cam_data['height'],
            "camera_name": camera_name,
            "frame_name": base_name
        }
        
        metadata.append(item)
    
    print(f"Generated {len(metadata)} metadata items")
    
    # Train/Val Split
    num_train = int(len(metadata) * train_ratio)
    train_metadata = metadata[:num_train]
    val_metadata = metadata[num_train:]
    
    print(f"Train: {len(train_metadata)}, Val: {len(val_metadata)}")
    
    # 저장
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    print(f"Saved to: {output_path}")
    
    # Validation 저장 (있으면)
    if len(val_metadata) > 0:
        val_path = output_path.parent.parent / 'val_meta' / output_path.name
        val_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(val_path, 'w') as f:
            json.dump(val_metadata, f, indent=2)
        
        print(f"Validation saved to: {val_path}")
    
    return train_metadata, val_metadata


def prepare_3dgut_metadata(
    data_root: str,
    output_file: str,
    train_ratio: float = 0.9,
    camera_filter: list = None
):
    """
    3DGUT용 메타데이터 생성 (Rolling Shutter Compensated)
    
    3DGS 메타데이터 + velocity + rolling_shutter 정보 추가
    """
    data_root = Path(data_root)
    
    # 디렉토리 확인
    final_inpainted_dir = data_root / 'final_inpainted'
    poses_dir = data_root / 'poses'
    
    if not final_inpainted_dir.exists():
        raise FileNotFoundError(f"final_inpainted not found: {final_inpainted_dir}")
    
    if not poses_dir.exists():
        raise FileNotFoundError(f"poses not found: {poses_dir}")
    
    # 이미지 파일 수집
    image_files = sorted(list(final_inpainted_dir.glob('*.jpg')))
    if len(image_files) == 0:
        image_files = sorted(list(final_inpainted_dir.glob('*.png')))
    
    print(f"Found {len(image_files)} inpainted images")
    
    # 메타데이터 생성
    metadata = []
    
    for img_file in tqdm(image_files, desc="Processing frames (3DGUT)"):
        # 대응하는 pose 파일 찾기
        frame_name = img_file.stem
        
        # 카메라 이름 추출
        parts = frame_name.split('_')
        if len(parts) >= 3:
            camera_name = parts[-1]
            base_name = '_'.join(parts[:-1])
        else:
            camera_name = 'UNKNOWN'
            base_name = frame_name
        
        # 카메라 필터링
        if camera_filter and camera_name not in camera_filter:
            continue
        
        # Pose JSON 로드
        pose_file = poses_dir / f"{base_name}.json"
        
        if not pose_file.exists():
            print(f"Warning: Pose file not found for {img_file.name}, skipping")
            continue
        
        with open(pose_file, 'r') as f:
            pose_data = json.load(f)
        
        # 카메라 데이터 추출
        if camera_name not in pose_data.get('cameras', {}):
            print(f"Warning: Camera {camera_name} not in pose data, skipping")
            continue
        
        cam_data = pose_data['cameras'][camera_name]
        
        # Ego Velocity 추출 (프레임 레벨)
        ego_velocity = pose_data.get('ego_velocity', {})
        if not ego_velocity:
            print(f"Warning: No ego_velocity in {base_name}, using zero velocity")
            ego_velocity = {
                "linear": [0.0, 0.0, 0.0],
                "angular": [0.0, 0.0, 0.0]
            }
        
        # Rolling Shutter 추출 (카메라 레벨)
        rolling_shutter = cam_data.get('rolling_shutter', {})
        if not rolling_shutter:
            print(f"Warning: No rolling_shutter in {camera_name}, using default")
            rolling_shutter = {
                "duration": 0.033,  # 30fps 가정
                "trigger_time": 0.0
            }
        
        # 메타데이터 아이템 생성 (3DGUT)
        item = {
            # 기본 필드 (3DGS와 동일)
            "file_path": str(img_file.relative_to(data_root)),
            "transform_matrix": cam_data['pose'],
            "intrinsics": cam_data['intrinsics'],
            "width": cam_data['width'],
            "height": cam_data['height'],
            "camera_name": camera_name,
            "frame_name": base_name,
            
            # 3DGUT 전용 필드
            "velocity": {
                "v": ego_velocity['linear'],    # [vx, vy, vz]
                "w": ego_velocity['angular']    # [wx, wy, wz]
            },
            "rolling_shutter": {
                "duration": rolling_shutter['duration'],
                "trigger_time": rolling_shutter.get('trigger_time', 0.0)
            }
        }
        
        metadata.append(item)
    
    print(f"Generated {len(metadata)} metadata items (3DGUT)")
    
    # Train/Val Split
    num_train = int(len(metadata) * train_ratio)
    train_metadata = metadata[:num_train]
    val_metadata = metadata[num_train:]
    
    print(f"Train: {len(train_metadata)}, Val: {len(val_metadata)}")
    
    # 저장
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(train_metadata, f, indent=2)
    
    print(f"Saved to: {output_path}")
    
    # Validation 저장
    if len(val_metadata) > 0:
        val_path = output_path.parent.parent / 'val_meta' / output_path.name
        val_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(val_path, 'w') as f:
            json.dump(val_metadata, f, indent=2)
        
        print(f"Validation saved to: {val_path}")
    
    return train_metadata, val_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON metadata for 3D Reconstruction"
    )
    parser.add_argument(
        'data_root',
        type=str,
        help='Path to NRE format data root'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='train_meta/train_pairs.json',
        help='Output JSON file path (relative to data_root)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['3dgs', '3dgut'],
        default='3dgs',
        help='Metadata mode: 3dgs (static) or 3dgut (rolling shutter)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.9,
        help='Train/Val split ratio (default: 0.9)'
    )
    parser.add_argument(
        '--camera_filter',
        type=str,
        nargs='+',
        default=None,
        help='Filter cameras (e.g., FRONT FRONT_LEFT)'
    )
    
    args = parser.parse_args()
    
    # 출력 경로 (data_root 기준)
    output_path = Path(args.data_root) / args.output
    
    if args.mode == '3dgs':
        print("="*70)
        print(">>> Generating 3DGS Metadata (Static Scene)")
        print("="*70)
        
        prepare_3dgs_metadata(
            data_root=args.data_root,
            output_file=str(output_path),
            train_ratio=args.train_ratio,
            camera_filter=args.camera_filter
        )
    
    elif args.mode == '3dgut':
        print("="*70)
        print(">>> Generating 3DGUT Metadata (Rolling Shutter)")
        print("="*70)
        
        prepare_3dgut_metadata(
            data_root=args.data_root,
            output_file=str(output_path),
            train_ratio=args.train_ratio,
            camera_filter=args.camera_filter
        )
    
    print("\n" + "="*70)
    print(">>> Metadata Generation Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
