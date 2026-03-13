"""
Waymo Open Dataset을 NRE(Neural Rendering Engine) 포맷으로 변환
TensorFlow/MMCV 의존성 완전 제거 버전
"""

import os
import json
import math
import struct
import glob
import zlib
import numpy as np
import cv2
import argparse

try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-11-0" '
        'to install the official devkit first.')


# -------------------------------------------------------------------------
# 1. Lightweight TFRecord Reader (No TensorFlow dependency)
# -------------------------------------------------------------------------
class MinimalTFRecordReader:
    """
    TensorFlow 설치 없이 .tfrecord 파일을 읽기 위한 경량 리더
    """
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, 'rb') as f:
            while True:
                # 1. Read Length (uint64)
                length_bytes = f.read(8)
                if not length_bytes:
                    break  # End of file
                
                # 2. Read CRC of Length (uint32) - Skip validation for speed
                f.read(4)
                
                # 3. Read Data
                length = struct.unpack('<Q', length_bytes)[0]
                data = f.read(length)
                
                # 4. Read CRC of Data (uint32) - Skip validation
                f.read(4)
                
                yield data


# -------------------------------------------------------------------------
# 2. Converter Class
# -------------------------------------------------------------------------
class Waymo2NRE(object):
    """
    Waymo 데이터셋을 NRE 포맷으로 변환하는 클래스 (Minimal Mode Only)
    
    NRE 포맷 구조:
    - images/: 각 카메라의 이미지 파일
    - poses/: 각 프레임의 지오메트리 정보 (카메라 포즈, 속도 등)
    - objects/: 각 프레임의 동적 객체 정보
    """
    
    def __init__(self, load_dir, save_dir, prefix):
        """
        Args:
            load_dir (str): Waymo TFRecord 파일들이 있는 디렉토리
            save_dir (str): NRE 포맷으로 저장할 디렉토리
            prefix (str): 파일명 접두사 (예: 'seq0_')
        """
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        
        # Directory Structure
        self.dirs = {
            'images': os.path.join(save_dir, 'images'),
            'poses': os.path.join(save_dir, 'poses'),
            'objects': os.path.join(save_dir, 'objects'),
            'point_clouds': os.path.join(save_dir, 'point_clouds')
        }
        
        # Create directories (no external dependencies)
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        self.tfrecord_pathnames = sorted(glob.glob(os.path.join(load_dir, '*.tfrecord')))
        
        if not self.tfrecord_pathnames:
            print(f"Warning: No .tfrecord files found in {load_dir}")

    def convert(self):
        """전체 변환 프로세스 실행"""
        print(f"Start converting {len(self.tfrecord_pathnames)} tfrecords (Minimal Mode)...")
        
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
        # Custom Reader 사용 (No TensorFlow)
        reader = MinimalTFRecordReader(pathname)
        
        world_origin_inv = None
        
        # Segment 내 프레임 순회
        for frame_idx, data_bytes in enumerate(reader):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(data_bytes)
            
            # 첫 프레임 기준 World Origin 설정 (Jittering 방지)
            if frame_idx == 0:
                first_frame_pose = np.array(frame.pose.transform).reshape(4, 4)
                world_origin_inv = np.linalg.inv(first_frame_pose)

            # 1. Coordinate Transformation (Global -> Local World)
            current_pose_global = np.array(frame.pose.transform).reshape(4, 4)
            T_vehicle_to_world = world_origin_inv @ current_pose_global

            # 2. Process Sensors & Objects
            self.save_images_and_geometry(frame, file_idx, frame_idx, T_vehicle_to_world, world_origin_inv)
            self.save_dynamic_objects(frame, file_idx, frame_idx, T_vehicle_to_world)
            self.save_lidar_point_cloud(frame, file_idx, frame_idx, T_vehicle_to_world)

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
        for img in frame.images:
            cam_name = dataset_pb2.CameraName.Name.Name(img.name)
            
            # 1. Save Image (OpenCV로 디코딩)
            img_path = os.path.join(self.dirs['images'], f"{frame_name}_{cam_name}.jpg")
            
            # byte array -> numpy array -> decode -> write
            np_arr = np.frombuffer(img.image, np.uint8)
            image_decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image_decoded is not None:
                cv2.imwrite(img_path, image_decoded)
            else:
                print(f"Warning: Failed to decode image {frame_name}_{cam_name}")
                continue

            # 2. Calibration & Pose
            calib = next(c for c in frame.context.camera_calibrations if c.name == img.name)
            
            T_cam_to_vehicle = np.array(calib.extrinsic.transform).reshape(4, 4)
            T_cam_to_world = T_vehicle_to_world @ T_cam_to_vehicle
            
            # 3. Rolling Shutter Logic
            # camera_readout_done_time - camera_trigger_time = shutter duration
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

    def save_lidar_point_cloud(self, frame, file_idx, frame_idx, T_vehicle_to_world):
        """
        LiDAR 포인트 클라우드 저장 (Range Image 디코딩 구현)

        Waymo의 range_image_compressed를 zlib으로 해제하고,
        MatrixFloat proto를 파싱하여 (range, inclination, azimuth) →
        3D 포인트 (Vehicle 좌표계 → Local World 좌표계)로 변환합니다.

        Args:
            frame: Waymo Frame 객체
            file_idx (int): 파일 인덱스
            frame_idx (int): 프레임 인덱스
            T_vehicle_to_world: Vehicle to World 변환 행렬 (4x4)
        """
        frame_name = f"{self.prefix}{file_idx:03d}{frame_idx:03d}"

        # LiDAR Calibration 인덱싱 (laser.name → calibration 매핑)
        laser_calibrations = {c.name: c for c in frame.context.laser_calibrations}

        # 모든 LiDAR 센서의 포인트를 합침
        all_points = []

        for laser in frame.lasers:
            if len(laser.ri_return1.range_image_compressed) == 0:
                continue

            calib = laser_calibrations.get(laser.name)
            if calib is None:
                continue

            # --- Range Image 디코딩 ---
            try:
                range_image = self._decompress_range_image(
                    laser.ri_return1.range_image_compressed
                )
            except Exception as e:
                print(f"Warning: Failed to decompress range image "
                      f"(laser={laser.name}): {e}")
                continue

            if range_image is None:
                continue

            # --- Range Image → 3D Points (Sensor Frame) ---
            points_sensor = self._range_image_to_points(range_image, calib)

            if len(points_sensor) == 0:
                continue

            # --- Sensor Frame → Vehicle Frame ---
            T_sensor_to_vehicle = np.array(
                calib.extrinsic.transform
            ).reshape(4, 4)

            points_homo = np.hstack([
                points_sensor,
                np.ones((len(points_sensor), 1))
            ])
            points_vehicle = (T_sensor_to_vehicle @ points_homo.T).T[:, :3]

            # --- Vehicle Frame → Local World Frame ---
            points_homo_v = np.hstack([
                points_vehicle,
                np.ones((len(points_vehicle), 1))
            ])
            points_world = (T_vehicle_to_world @ points_homo_v.T).T[:, :3]

            all_points.append(points_world)

        # 결과 저장
        if len(all_points) > 0:
            points_array = np.concatenate(all_points, axis=0).astype(np.float32)

            bin_path = os.path.join(
                self.dirs['point_clouds'], f"{frame_name}.bin"
            )
            points_array.tofile(bin_path)
        else:
            print(f"Warning: No LiDAR points extracted for frame {frame_name}")

    @staticmethod
    def _decompress_range_image(compressed_data):
        """
        Waymo range_image_compressed를 디코딩

        Args:
            compressed_data: zlib 압축된 MatrixFloat protobuf bytes

        Returns:
            np.ndarray: (H, W, C) float32 배열, 첫 채널이 range (m)
        """
        decompressed = zlib.decompress(compressed_data)
        range_image_proto = dataset_pb2.MatrixFloat()
        range_image_proto.ParseFromString(decompressed)

        dims = list(range_image_proto.shape.dims)
        if len(dims) < 2:
            return None

        h, w = dims[0], dims[1]
        c = dims[2] if len(dims) > 2 else 1

        data = np.array(range_image_proto.data, dtype=np.float32)

        if data.size != h * w * c:
            return None

        return data.reshape(h, w, c)

    @staticmethod
    def _range_image_to_points(range_image, calibration):
        """
        Range image를 3D 포인트 클라우드로 변환 (Sensor 좌표계)

        Waymo range image 구조:
        - 행(row) = beam inclination (수직 각도)
        - 열(col) = azimuth (수평 각도)
        - 값    = [range, intensity, elongation, ...]

        Args:
            range_image: (H, W, C) ndarray, 채널 0 = range (m)
            calibration: LiDAR calibration proto

        Returns:
            points: (N, 3) ndarray, 유효한 3D 포인트 (Sensor 좌표계)
        """
        ranges = range_image[:, :, 0]
        h, w = ranges.shape

        # Beam inclinations (수직 각도)
        if len(calibration.beam_inclinations) > 0:
            inclinations = np.array(calibration.beam_inclinations)
            # Waymo는 beam_inclinations를 내림차순으로 저장할 수 있음
            if len(inclinations) != h:
                inclinations = np.linspace(
                    calibration.beam_inclination_min,
                    calibration.beam_inclination_max,
                    h
                )
        else:
            inclinations = np.linspace(
                calibration.beam_inclination_min,
                calibration.beam_inclination_max,
                h
            )

        # Azimuth angles (수평 각도, π ~ -π)
        azimuths = np.linspace(np.pi, -np.pi, w, endpoint=False)

        # Meshgrid
        inc_grid, az_grid = np.meshgrid(inclinations, azimuths, indexing='ij')

        # 유효한 포인트만 (range > 0)
        valid = ranges > 0
        r = ranges[valid]
        inc = inc_grid[valid]
        az = az_grid[valid]

        # 구면좌표 → 직교좌표 (Sensor Frame)
        cos_inc = np.cos(inc)
        x = r * cos_inc * np.cos(az)
        y = r * cos_inc * np.sin(az)
        z = r * np.sin(inc)

        return np.stack([x, y, z], axis=1)


def main():
    parser = argparse.ArgumentParser(
        description='Convert Waymo Open Dataset to NRE format (No TensorFlow/MMCV)'
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
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.load_dir):
        print(f"Error: Input directory not found: {args.load_dir}")
        return
    
    # Create converter and run
    converter = Waymo2NRE(
        load_dir=args.load_dir,
        save_dir=args.save_dir,
        prefix=args.prefix
    )
    converter.convert()


if __name__ == '__main__':
    main()
