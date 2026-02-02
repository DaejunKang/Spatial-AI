"""
NRE(Neural Reconstruction Engine)나 3DGS 학습을 위한 데이터셋 생성 스크립트

Waymo2NRE 스크립트로 생성한 개별 JSON 파일들(poses/*.json)을 하나로 모으고,
학습용(Train)과 검증용(Validation) 데이터셋으로 나누어 생성합니다.
"""

import os
import json
import glob
import random
import numpy as np
from tqdm import tqdm


class NREPairGenerator:
    """
    NRE / 3DGS 학습을 위한 Pair Dataset Generator
    
    기능:
    1. Waymo2NRE로 생성된 개별 Frame JSON을 읽어들임.
    2. 학습(Train) / 검증(Val) 데이터셋 분할 (기본: 10프레임마다 검증용).
    3. 5개의 카메라(Front, Side...)를 개별 학습 데이터 포인트로 분리.
    4. Rolling Shutter 및 Ego Velocity 정보를 포함한 최종 JSON 생성.
    """

    def __init__(self, data_root, output_dir, val_interval=10):
        self.data_root = data_root
        self.output_dir = output_dir
        self.val_interval = val_interval
        
        # 데이터 경로 확인
        self.pose_dir = os.path.join(data_root, 'poses')
        self.image_dir = os.path.join(data_root, 'images')
        
        if not os.path.exists(self.pose_dir):
            raise FileNotFoundError(f"Poses directory not found: {self.pose_dir}")
            
        # 프레임 리스트 로드 및 정렬 (타임스탬프 순서 보장)
        self.frame_files = sorted(glob.glob(os.path.join(self.pose_dir, '*.json')))
        print(f"Found {len(self.frame_files)} frames in {self.pose_dir}")

    def generate(self):
        train_pairs = []
        val_pairs = []

        print("Generating dataset pairs...")
        for idx, json_path in enumerate(tqdm(self.frame_files)):
            # 1. 프레임 데이터 로드
            with open(json_path, 'r') as f:
                frame_data = json.load(f)

            # 2. 이미지 단위로 데이터 분해 (Flattening)
            # 하나의 프레임(Timestamp)에 5개의 카메라 이미지가 존재함
            # 이를 각각 독립적인 학습 데이터로 변환
            frame_items = self._parse_frame(frame_data)

            # 3. Train / Val 분할 (시계열 순서 유지)
            if idx % self.val_interval == 0:
                val_pairs.extend(frame_items)
            else:
                train_pairs.extend(frame_items)

        # 4. JSON 파일 저장
        self._save_json(train_pairs, 'train_pairs.json')
        self._save_json(val_pairs, 'val_pairs.json')

    def _parse_frame(self, frame_data):
        """
        하나의 Frame JSON을 5개의 Image Pair Dict로 변환
        """
        items = []
        timestamp = frame_data['timestamp']
        ego_velocity = frame_data['ego_velocity']  # {linear: [], angular: []}
        
        for cam_name, cam_info in frame_data['cameras'].items():
            # 이미지 경로 검증
            img_rel_path = cam_info['img_path']  # relative path from data_root
            full_img_path = os.path.join(self.data_root, img_rel_path)
            
            # (Optional) 이미지가 실제로 존재하는지 체크
            if not os.path.exists(full_img_path):
                print(f"Warning: Image not found {full_img_path}")
                continue

            # Pair Item 구성
            item = {
                "file_path": img_rel_path,
                "timestamp": timestamp,
                "camera_id": cam_name,
                
                # --- Geometry ---
                # Camera-to-World Pose (4x4 Flattened)
                "transform_matrix": cam_info['pose'], 
                
                # Intrinsics [fx, fy, cx, cy, k1, k2, p1, p2, k3]
                "intrinsics": cam_info['intrinsics'],
                "width": cam_info['width'],
                "height": cam_info['height'],

                # --- Dynamic Compensation Data ---
                # Ego Velocity (Local World Frame)
                "velocity": {
                    "v": ego_velocity['linear'],  # [vx, vy, vz]
                    "w": ego_velocity['angular']  # [wx, wy, wz]
                },
                
                # Rolling Shutter Metadata
                "rolling_shutter": {
                    "duration": cam_info['rolling_shutter']['duration'],
                    "trigger_time": cam_info['rolling_shutter']['trigger_time']
                }
            }
            items.append(item)
            
        return items

    def _save_json(self, data, filename):
        save_path = os.path.join(self.output_dir, filename)
        
        # 메타데이터 래핑
        output_data = {
            "meta": {
                "total_frames": len(data),
                "coordinate_system": "Right-Down-Front (Waymo Native)",
                "world_origin": "Aligned to Frame 0 Vehicle Pose"
            },
            "frames": data
        }
        
        with open(save_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Saved {len(data)} pairs to {save_path}")


if __name__ == "__main__":
    # 설정: Waymo2NREMinimal의 출력 경로 지정
    DATA_ROOT = './data/waymo/nre_format'
    
    # 생성기 실행
    generator = NREPairGenerator(
        data_root=DATA_ROOT, 
        output_dir=DATA_ROOT,
        val_interval=8  # 8프레임마다 하나씩 검증셋으로 사용 (약 12.5%)
    )
    generator.generate()
