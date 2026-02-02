"""
Waymo 데이터셋 처리를 위한 공통 유틸리티 모듈
TFRecord 읽기, 좌표 변환, 이미지 처리 등 중복 기능 통합
"""

import os
import struct
import numpy as np
import cv2

try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    print("Warning: 'waymo_open_dataset' not found.")
    dataset_pb2 = None


# -------------------------------------------------------------------------
# 1. Lightweight TFRecord Reader (No TensorFlow dependency)
# -------------------------------------------------------------------------
class MinimalTFRecordReader:
    """
    TensorFlow 설치 없이 .tfrecord 파일을 읽기 위한 경량 리더
    TFRecord 포맷: [length(8bytes)][crc_length(4bytes)][data][crc_data(4bytes)]
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
# 2. Common Utility Functions
# -------------------------------------------------------------------------
def ensure_dir(path):
    """디렉토리가 없으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def decode_image_opencv(image_bytes):
    """
    OpenCV를 사용하여 이미지 디코딩 (JPEG/PNG 지원)
    
    Args:
        image_bytes: 이미지 바이트 데이터
        
    Returns:
        np.ndarray: BGR 포맷 이미지, 실패시 None
    """
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image_decoded = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image_decoded


def get_camera_name_map():
    """Waymo 카메라 ID to Name 매핑"""
    return {
        1: 'FRONT',
        2: 'FRONT_LEFT',
        3: 'FRONT_RIGHT',
        4: 'SIDE_LEFT',
        5: 'SIDE_RIGHT'
    }


def transform_pose_to_local(frame_pose_global, world_origin_inv):
    """
    글로벌 Pose를 로컬 월드 좌표계로 변환
    
    Args:
        frame_pose_global: 4x4 글로벌 포즈 행렬
        world_origin_inv: 첫 프레임 포즈의 역행렬 (월드 원점)
        
    Returns:
        np.ndarray: 로컬 월드 좌표계 포즈 (4x4)
    """
    return world_origin_inv @ frame_pose_global


def project_3d_box_to_2d(box, T_c_v, intrinsic, width, height):
    """
    3D 바운딩 박스를 2D 이미지 평면에 투영
    
    Args:
        box: Waymo Label.Box 객체
        T_c_v: Camera to Vehicle 변환 행렬 (4x4)
        intrinsic: 카메라 내부 파라미터 [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        width: 이미지 너비
        height: 이미지 높이
        
    Returns:
        np.ndarray: 투영된 2D 포인트들 (Nx2), 또는 None (투영 실패시)
    """
    l, w, h = box.length, box.width, box.height
    heading = box.heading
    
    # Rotation around Z-axis (Vehicle Frame)
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    R_z = np.array([
        [cos_h, -sin_h, 0],
        [sin_h, cos_h, 0],
        [0, 0, 1]
    ])
    
    # 8 Corners relative to center
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    
    corners_3d = np.vstack([x_corners, y_corners, z_corners])  # 3x8
    corners_3d = R_z @ corners_3d  # Rotate
    corners_3d = corners_3d + np.array([box.center_x, box.center_y, box.center_z]).reshape(3, 1)
    
    # Add homogeneous coord
    corners_3d_homo = np.vstack([corners_3d, np.ones((1, 8))])  # 4x8
    
    # Transform to Camera Frame
    corners_c = T_c_v @ corners_3d_homo  # 4x8
    
    # Check if points are in front of camera
    valid_corners = corners_c[2, :] > 0
    if not np.any(valid_corners):
        return None
    
    # Use OpenCV for accurate projection with distortion
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
        projected_points = projected_points.squeeze().astype(np.int32)
        return projected_points
    except:
        return None


def get_calibration_dict(frame):
    """
    프레임에서 카메라 Calibration 정보 추출
    
    Args:
        frame: Waymo Frame 객체
        
    Returns:
        dict: {cam_name: {extrinsic, intrinsic, width, height}}
    """
    cam_name_map = get_camera_name_map()
    calibs = {}
    
    for camera in frame.context.camera_calibrations:
        cam_name = cam_name_map.get(camera.name, 'UNKNOWN')
        if cam_name == 'UNKNOWN':
            continue
        
        extrinsic = np.array(camera.extrinsic.transform).reshape(4, 4)
        intrinsic = np.array(camera.intrinsic)
        
        calibs[cam_name] = {
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
            'width': camera.width,
            'height': camera.height
        }
    
    return calibs


def quaternion_to_rotation_matrix(qvec):
    """
    쿼터니언을 회전 행렬로 변환
    
    Args:
        qvec: [w, x, y, z] 쿼터니언
        
    Returns:
        np.ndarray: 3x3 회전 행렬
    """
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])


def rotation_matrix_to_quaternion(R):
    """
    회전 행렬을 쿼터니언으로 변환
    
    Args:
        R: 3x3 회전 행렬
        
    Returns:
        np.ndarray: [w, x, y, z] 쿼터니언
    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0:
        q *= -1
    return q
