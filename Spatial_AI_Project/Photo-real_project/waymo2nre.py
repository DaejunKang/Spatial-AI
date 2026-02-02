"""
Waymo Open Dataset을 NRE(Neural Rendering Engine) 포맷으로 변환
TensorFlow 의존성을 최소화한 경량 변환기
"""

import os
import json
import math
import glob
import numpy as np
import argparse

try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-11-0" '
        'to install the official devkit first.')

from waymo_utils import (
    MinimalTFRecordReader,
    ensure_dir,
    decode_image_opencv,
    get_camera_name_map,
    transform_pose_to_local,
    get_calibration_dict
)
import cv2


class Waymo2NRE(object):
    """
    Waymo 데이터셋을 NRE 포맷으로 변환하는 클래스
    
    NRE 포맷 구조:
    - images/: 각 카메라의 이미지 파일
    - poses/: 각 프레임의 지오메트리 정보 (카메라 포즈, 속도 등)
    - intrinsics/: 카메라 내부 파라미터 (사용하지 않음, poses에 통합)
    - objects/: 각 프레임의 동적 객체 정보
    """
    
    def __init__(self, load_dir, save_dir, prefix, use_tensorflow=False):
        """
        Args:
            load_dir (str): Waymo TFRecord 파일들이 있는 디렉토리
            save_dir (str): NRE 포맷으로 저장할 디렉토리
            prefix (str): 파일명 접두사 (예: 'seq0_')
            use_tensorflow (bool): TensorFlow 사용 여부 (기본값: False)
        """
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        self.use_tensorflow = use_tensorflow
        
        # Directory Structure
        self.dirs = {
            'images': os.path.join(save_dir, 'images'),
            'poses': os.path.join(save_dir, 'poses'),
            'intrinsics': os.path.join(save_dir, 'intrinsics'),
            'objects': os.path.join(save_dir, 'objects')
        }
        
        # Create directories
        for d in self.dirs.values():
            ensure_dir(d)

        self.tfrecord_pathnames = sorted(glob.glob(os.path.join(load_dir, '*.tfrecord')))
        
        if not self.tfrecord_pathnames:
            print(f"Warning: No .tfrecord files found in {load_dir}")

    def convert(self):
        """전체 변환 프로세스 실행"""
        print(f"Start converting {len(self.tfrecord_pathnames)} tfrecords...")
        print(f"Mode: {'TensorFlow' if self.use_tensorflow else 'Minimal (No TensorFlow)'}")
        
        for i, pathname in enumerate(self.tfrecord_pathnames):
            print(f"Processing segment {i}: {os.path.basename(pathname)}")
            self.process_one_segment(i, pathname)
            
        print("Conversion Finished!")

    def process_one_segment(self, file_idx, pathname):
        """
        하나의 TFRecord 세그먼트 처리
        
        Args:
            file_idx (int): 파일 인덱스
            pathname (str): TFRecord 파일 경로
        """
        # Choose reader based on use_tensorflow flag
        if self.use_tensorflow:
            import tensorflow as tf
            dataset = tf.data.TFRecordDataset(pathname, compression_type='')
            frame_iterator = enumerate(dataset)
        else:
            reader = MinimalTFRecordReader(pathname)
            frame_iterator = enumerate(reader)
        
        world_origin_inv = None
        
        # Segment 내 프레임 순회
        for frame_idx, data in frame_iterator:
            frame = dataset_pb2.Frame()
            
            # Parse frame data
            if self.use_tensorflow:
                frame.ParseFromString(bytearray(data.numpy()))
            else:
                frame.ParseFromString(data)
            
            # 첫 프레임 기준 World Origin 설정 (Jittering 방지)
            if frame_idx == 0:
                first_frame_pose = np.array(frame.pose.transform).reshape(4, 4)
                world_origin_inv = np.linalg.inv(first_frame_pose)

            # 1. Coordinate Transformation (Global -> Local World)
            current_pose_global = np.array(frame.pose.transform).reshape(4, 4)
            T_vehicle_to_world = transform_pose_to_local(current_pose_global, world_origin_inv)

            # 2. Process Sensors & Objects
            self.save_images_and_geometry(frame, file_idx, frame_idx, T_vehicle_to_world, world_origin_inv)
            self.save_dynamic_objects(frame, file_idx, frame_idx, T_vehicle_to_world)

    def save_images_and_geometry(self, frame, file_idx, frame_idx, T_vehicle_to_world, world_origin_inv):
        """
        이미지와 지오메트리 정보 저장
        
        Args:
            frame: Waymo Frame 객체
            file_idx (int): 파일 인덱스
            frame_idx (int): 프레임 인덱스
            T_vehicle_to_world: Vehicle to World 변환 행렬 (4x4)
            world_origin_inv: World Origin 역행렬 (4x4)
        """
        frame_name = f"{self.prefix}{file_idx:03d}{frame_idx:03d}"
        
        geometry_info = {
            "frame_idx": frame_idx,
            "timestamp": frame.timestamp_micros / 1e6,
            "cameras": {}
        }

        # --- Velocity Rotation (Global -> Local World) ---
        R_inv = world_origin_inv[:3, :3]
        
        # Extract Velocity from the first image metadata (shared by ego vehicle)
        if len(frame.images) > 0:
            img0 = frame.images[0]
            v_global = np.array([img0.velocity.v_x, img0.velocity.v_y, img0.velocity.v_z])
            w_global = np.array([img0.velocity.w_x, img0.velocity.w_y, img0.velocity.w_z])
            
            v_local = R_inv @ v_global
            w_local = R_inv @ w_global

            geometry_info['ego_velocity'] = {
                "linear": v_local.tolist(),
                "angular": w_local.tolist()
            }

        # --- Camera Processing ---
        cam_name_map = get_camera_name_map()
        
        for img in frame.images:
            cam_name = cam_name_map.get(img.name, 'UNKNOWN')
            if cam_name == 'UNKNOWN':
                continue
            
            # 1. Save Image
            img_path = os.path.join(self.dirs['images'], f"{frame_name}_{cam_name}.jpg")
            
            # Decode and save image
            image_decoded = decode_image_opencv(img.image)
            
            if image_decoded is not None:
                cv2.imwrite(img_path, image_decoded)
            else:
                print(f"Warning: Failed to decode image {frame_name}_{cam_name}")
                continue

            # 2. Calibration & Pose
            calib = next((c for c in frame.context.camera_calibrations if c.name == img.name), None)
            if calib is None:
                print(f"Warning: No calibration found for camera {cam_name}")
                continue
            
            T_cam_to_vehicle = np.array(calib.extrinsic.transform).reshape(4, 4)
            T_cam_to_world = T_vehicle_to_world @ T_cam_to_vehicle
            
            # 3. Rolling Shutter Logic
            readout = 0.0
            if hasattr(img, 'rolling_shutter_params') and hasattr(img.rolling_shutter_params, 'shutter'):
                readout = img.rolling_shutter_params.shutter
            
            if readout == 0.0:
                readout = max(0.0, img.camera_readout_done_time - img.camera_trigger_time)

            geometry_info['cameras'][cam_name] = {
                "img_path": os.path.relpath(img_path, self.save_dir),
                "width": calib.width,
                "height": calib.height,
                "intrinsics": list(calib.intrinsic),
                "pose": T_cam_to_world.flatten().tolist(),
                "rolling_shutter": {
                    "duration": readout,
                    "trigger_time": img.camera_trigger_time
                }
            }

        # Save Geometry JSON
        json_path = os.path.join(self.dirs['poses'], f"{frame_name}.json")
        with open(json_path, 'w') as f:
            json.dump(geometry_info, f, indent=4)

    def save_dynamic_objects(self, frame, file_idx, frame_idx, T_vehicle_to_world):
        """
        동적 객체 정보 저장
        
        Args:
            frame: Waymo Frame 객체
            file_idx (int): 파일 인덱스
            frame_idx (int): 프레임 인덱스
            T_vehicle_to_world: Vehicle to World 변환 행렬 (4x4)
        """
        frame_name = f"{self.prefix}{file_idx:03d}{frame_idx:03d}"
        objects = []

        for label in frame.laser_labels:
            # Filter for dynamic objects: Vehicle(1), Pedestrian(2), Cyclist(4)
            if label.type not in [1, 2, 4]:
                continue

            # Box Center Conversion (Vehicle -> Local World)
            box_center_veh = np.array([
                label.box.center_x,
                label.box.center_y,
                label.box.center_z,
                1.0
            ])
            box_center_world = T_vehicle_to_world @ box_center_veh

            # Heading Conversion
            veh_yaw = math.atan2(T_vehicle_to_world[1, 0], T_vehicle_to_world[0, 0])
            box_heading_world = label.box.heading + veh_yaw

            objects.append({
                "id": label.id,
                "class": dataset_pb2.Label.Type.Name(label.type),
                "box": {
                    "center": box_center_world[:3].tolist(),
                    "size": [label.box.length, label.box.width, label.box.height],
                    "heading": box_heading_world
                },
                "speed": [label.metadata.speed_x, label.metadata.speed_y]
            })

        # Save Objects JSON
        json_path = os.path.join(self.dirs['objects'], f"{frame_name}.json")
        with open(json_path, 'w') as f:
            json.dump(objects, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description='Convert Waymo Open Dataset to NRE format (Minimal Mode)'
    )
    parser.add_argument(
        'load_dir',
        type=str,
        help='Directory containing Waymo .tfrecord files'
    )
    parser.add_argument(
        'save_dir',
        type=str,
        help='Directory to save NRE format data'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='seq0_',
        help='Prefix for output filenames (default: seq0_)'
    )
    parser.add_argument(
        '--use-tensorflow',
        action='store_true',
        help='Use TensorFlow for TFRecord reading (default: False, uses minimal reader)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.load_dir):
        print(f"Error: Input directory not found: {args.load_dir}")
        return
    
    # Create converter and run
    converter = Waymo2NRE(
        load_dir=args.load_dir,
        save_dir=args.save_dir,
        prefix=args.prefix,
        use_tensorflow=args.use_tensorflow
    )
    converter.convert()


if __name__ == '__main__':
    main()
