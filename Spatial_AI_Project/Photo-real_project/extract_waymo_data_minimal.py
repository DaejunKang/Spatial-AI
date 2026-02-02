"""
Waymo Open Dataset에서 이미지와 마스크 추출 (TensorFlow/MMCV 의존성 제거)
COLMAP 전처리용 - 동적 객체 마스킹 포함
"""

import os
import struct
import numpy as np
import cv2
import argparse
import json
from glob import glob
from tqdm import tqdm

try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    print("Warning: 'waymo_open_dataset' not found.")
    print("Please install it: pip install waymo-open-dataset-tf-2-11-0")
    dataset_pb2 = None


# -------------------------------------------------------------------------
# Lightweight TFRecord Reader (No TensorFlow)
# -------------------------------------------------------------------------
class MinimalTFRecordReader:
    """TensorFlow 없이 .tfrecord 파일 읽기"""
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, 'rb') as f:
            while True:
                length_bytes = f.read(8)
                if not length_bytes:
                    break
                
                f.read(4)  # Skip CRC
                length = struct.unpack('<Q', length_bytes)[0]
                data = f.read(length)
                f.read(4)  # Skip CRC
                
                yield data


# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
def ensure_dir(path):
    """디렉토리 생성"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_camera_name_map():
    """Waymo 카메라 ID to Name 매핑"""
    return {
        1: 'FRONT',
        2: 'FRONT_LEFT',
        3: 'FRONT_RIGHT',
        4: 'SIDE_LEFT',
        5: 'SIDE_RIGHT'
    }


def project_3d_box_to_2d(box, T_c_v, intrinsic, width, height):
    """
    3D 바운딩 박스를 2D 이미지로 투영
    
    Args:
        box: Waymo Label.Box 객체
        T_c_v: Camera to Vehicle 변환 역행렬 (4x4)
        intrinsic: 카메라 내부 파라미터 [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        width: 이미지 너비
        height: 이미지 높이
    
    Returns:
        np.ndarray: 투영된 2D 포인트들, 또는 None
    """
    l, w, h = box.length, box.width, box.height
    heading = box.heading
    
    # Rotation around Z-axis
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    R_z = np.array([
        [cos_h, -sin_h, 0],
        [sin_h, cos_h, 0],
        [0, 0, 1]
    ])
    
    # 8 Corners
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    
    corners_3d = np.vstack([x_corners, y_corners, z_corners])  # 3x8
    corners_3d = R_z @ corners_3d
    corners_3d = corners_3d + np.array([box.center_x, box.center_y, box.center_z]).reshape(3, 1)
    
    # Homogeneous coordinates
    corners_3d_homo = np.vstack([corners_3d, np.ones((1, 8))])  # 4x8
    
    # Transform to Camera Frame
    corners_c = T_c_v @ corners_3d_homo  # 4x8
    
    # Check if in front of camera
    valid_corners = corners_c[2, :] > 0
    if not np.any(valid_corners):
        return None
    
    # OpenCV projection with distortion
    fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
    k1, k2, p1, p2, k3 = intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]
    
    r_vec, _ = cv2.Rodrigues(np.eye(3))
    t_vec = np.zeros((3, 1))
    dist_coeffs = np.array([k1, k2, p1, p2, k3])
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    points_3d_cam = corners_c[:3, :].T  # Nx3
    
    try:
        projected_points, _ = cv2.projectPoints(
            points_3d_cam, r_vec, t_vec, camera_matrix, dist_coeffs
        )
        return projected_points.squeeze().astype(np.int32)
    except:
        return None


# -------------------------------------------------------------------------
# Main Processing Functions
# -------------------------------------------------------------------------
def save_image_and_mask(frame, frame_idx, output_dir):
    """
    5개 카메라의 이미지와 동적 객체 마스크 저장
    """
    cam_name_map = get_camera_name_map()
    labels = frame.laser_labels
    
    # Calibration 정보 추출
    calibrations = {}
    for camera in frame.context.camera_calibrations:
        cam_name = cam_name_map.get(camera.name, 'UNKNOWN')
        if cam_name != 'UNKNOWN':
            calibrations[cam_name] = {
                'extrinsic': np.array(camera.extrinsic.transform).reshape(4, 4),
                'intrinsic': np.array(camera.intrinsic),
                'width': camera.width,
                'height': camera.height
            }

    for img in frame.images:
        cam_name = cam_name_map.get(img.name, 'UNKNOWN')
        if cam_name == 'UNKNOWN' or cam_name not in calibrations:
            continue
        
        # 디렉토리 생성
        cam_dir = os.path.join(output_dir, 'images', cam_name)
        ensure_dir(cam_dir)
        
        # 이미지 디코딩 (OpenCV)
        np_arr = np.frombuffer(img.image, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            print(f"Warning: Failed to decode image for camera {cam_name}")
            continue
        
        # 이미지 저장
        img_path = os.path.join(cam_dir, f'{frame_idx:06d}.png')
        cv2.imwrite(img_path, img_bgr)
        
        # --- 마스크 생성 ---
        calib_data = calibrations[cam_name]
        T_v_c = calib_data['extrinsic']
        intrinsic = calib_data['intrinsic']
        width, height = calib_data['width'], calib_data['height']
        
        # Vehicle -> Camera 변환
        T_c_v = np.linalg.inv(T_v_c)
        
        # 마스크 초기화 (흰색 = 유효, 검은색 = 동적 객체)
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # 동적 객체 투영
        for label in labels:
            # Vehicle(1), Pedestrian(2), Cyclist(4)
            if label.type not in [1, 2, 4]:
                continue
            
            projected_points = project_3d_box_to_2d(
                label.box, T_c_v, intrinsic, width, height
            )
            
            if projected_points is not None and len(projected_points) > 0:
                hull = cv2.convexHull(projected_points)
                cv2.fillConvexPoly(mask, hull, 0)  # 검은색으로 채우기
        
        # 마스크 저장
        mask_dir = os.path.join(output_dir, 'masks', cam_name)
        ensure_dir(mask_dir)
        mask_path = os.path.join(mask_dir, f'{frame_idx:06d}.png')
        cv2.imwrite(mask_path, mask)


def get_pose(frame):
    """Vehicle 글로벌 포즈 반환"""
    pose = np.array(frame.pose.transform).reshape(4, 4)
    return pose.tolist()


def get_calib(frame):
    """카메라 Calibration 정보 반환"""
    cam_name_map = get_camera_name_map()
    calibs = {}
    
    for camera in frame.context.camera_calibrations:
        cam_name = cam_name_map.get(camera.name, 'UNKNOWN')
        if cam_name == 'UNKNOWN':
            continue
        
        calibs[cam_name] = {
            'extrinsic': np.array(camera.extrinsic.transform).reshape(4, 4).tolist(),
            'intrinsic': np.array(camera.intrinsic).tolist(),
            'width': camera.width,
            'height': camera.height
        }
    
    return calibs


def extract_tfrecord(tfrecord_path, output_dir):
    """TFRecord 파일 처리"""
    if dataset_pb2 is None:
        print("Error: waymo_open_dataset module is missing.")
        return

    reader = MinimalTFRecordReader(tfrecord_path)
    
    print(f"Processing {tfrecord_path}...")
    
    vehicle_poses = {}
    calibration_info = None
    
    # 디렉토리 생성
    ensure_dir(os.path.join(output_dir, 'images'))
    
    for i, data in enumerate(tqdm(reader, desc="Processing frames")):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(data)
        
        # 이미지 및 마스크 저장
        save_image_and_mask(frame, i, output_dir)
        
        # Pose 저장
        vehicle_poses[f'{i:06d}'] = get_pose(frame)
        
        # Calibration 저장 (첫 프레임만)
        if calibration_info is None:
            calibration_info = get_calib(frame)
    
    # Pose JSON 저장
    poses_dir = os.path.join(output_dir, 'poses')
    ensure_dir(poses_dir)
    with open(os.path.join(poses_dir, 'vehicle_poses.json'), 'w') as f:
        json.dump(vehicle_poses, f, indent=4)
    
    # Calibration JSON 저장
    calib_dir = os.path.join(output_dir, 'calibration')
    ensure_dir(calib_dir)
    with open(os.path.join(calib_dir, 'intrinsics_extrinsics.json'), 'w') as f:
        json.dump(calibration_info, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description='Extract images and masks from Waymo TFRecords (No TensorFlow)'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to .tfrecord file or directory'
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory to save extracted data'
    )
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_dir = args.output_dir
    
    if os.path.isdir(input_path):
        tfrecord_files = sorted(glob(os.path.join(input_path, '*.tfrecord')))
    else:
        tfrecord_files = [input_path]
    
    print(f"Found {len(tfrecord_files)} TFRecord files.")
    
    for tf_file in tfrecord_files:
        segment_name = os.path.splitext(os.path.basename(tf_file))[0]
        segment_out_dir = os.path.join(output_dir, segment_name)
        
        extract_tfrecord(tf_file, segment_out_dir)
    
    print("\nExtraction complete!")


if __name__ == '__main__':
    main()
