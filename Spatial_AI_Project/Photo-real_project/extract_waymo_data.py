import os
import tensorflow as tf
import numpy as np
import cv2
import argparse
import json
from glob import glob
from tqdm import tqdm

# Try to import waymo_open_dataset
try:
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import range_image_utils
    from waymo_open_dataset.utils import transform_utils
    from waymo_open_dataset.utils import camera_segmentation_utils
except ImportError:
    print("Warning: 'waymo_open_dataset' not found. This script requires the Waymo Open Dataset package.")
    print("Please install it in a compatible environment (Python 3.7 - 3.10 recommended).")
    dataset_pb2 = None

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image_and_mask(frame, frame_idx, output_dir):
    """
    Save images from 5 cameras and generate masks for dynamic objects.
    """
    # Camera names mapping based on Waymo Proto
    cam_name_map = {
        1: 'FRONT',
        2: 'FRONT_LEFT',
        3: 'FRONT_RIGHT',
        4: 'SIDE_LEFT',
        5: 'SIDE_RIGHT'
    }

    # Extract 3D Labels (Laser Labels)
    labels = frame.laser_labels
    
    # We need to project these labels to each camera.
    # Waymo provides a helper, but implementing projection manually allows better control without heavy util dependencies if possible.
    # However, projection requires strict math. Let's use basic projection logic.
    
    # Pre-parse calibration for this frame
    calibrations = {c.name: c for c in frame.context.camera_calibrations}

    for img in frame.images:
        cam_name = cam_name_map.get(img.name, 'UNKNOWN')
        if cam_name == 'UNKNOWN':
            continue
            
        cam_dir = os.path.join(output_dir, 'images', cam_name)
        ensure_dir(cam_dir)
        
        # Decode image
        img_array = tf.image.decode_jpeg(img.image).numpy()
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Save Image
        img_path = os.path.join(cam_dir, f'{frame_idx:06d}.png')
        cv2.imwrite(img_path, img_bgr)
        
        # --- Generate Mask ---
        # 1. Get Calibration
        calib = calibrations[img.name]
        extrinsic = np.array(calib.extrinsic.transform).reshape(4, 4) # Camera to Vehicle
        intrinsic = np.array(calib.intrinsic) # [fu, fv, cu, cv, k1, k2, p1, p2, k3]
        width, height = calib.width, calib.height
        
        # 2. Compute Transform: Vehicle -> Camera
        # T_camera_vehicle = inv(T_vehicle_camera)
        T_v_c = extrinsic
        T_c_v = np.linalg.inv(T_v_c)
        
        # 3. Create Mask Image (White = Valid/Static, Black = Invalid/Dynamic)
        # Usually COLMAP masks: 0 (black) is ignored.
        # So we draw dynamic objects as BLACK (0), background as WHITE (255).
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # 4. Project Labels
        for label in labels:
            # Filter for dynamic objects (Vehicle, Pedestrian, Cyclist)
            # Type: UNKNOWN=0, VEHICLE=1, PEDESTRIAN=2, SIGN=3, CYCLIST=4
            if label.type not in [1, 2, 4]: 
                continue
                
            box = label.box
            
            # Box Center (in Vehicle Frame)
            center_v = np.array([box.center_x, box.center_y, box.center_z, 1.0])
            
            # Transform to Camera Frame
            center_c = T_c_v @ center_v
            
            # Check if in front of camera (z > 0)
            if center_c[2] <= 0:
                continue
                
            # Get 8 corners of the 3D Box in Vehicle Frame
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
            
            corners_3d = np.vstack([x_corners, y_corners, z_corners]) # 3x8
            corners_3d = R_z @ corners_3d # Rotate
            corners_3d = corners_3d + np.array([box.center_x, box.center_y, box.center_z]).reshape(3, 1) # Translate
            
            # Add homogeneous coord
            corners_3d_homo = np.vstack([corners_3d, np.ones((1, 8))]) # 4x8
            
            # Transform to Camera Frame
            corners_c = T_c_v @ corners_3d_homo # 4x8
            
            # Project to Image Plane
            # u = fx * x / z + cx
            # v = fy * y / z + cy
            # Distortions are ignored here for simplicity (usually okay for masking)
            # or we can use opencv projectPoints
            
            fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
            k1, k2, p1, p2, k3 = intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]
            
            # Use OpenCV for accurate projection with distortion
            r_vec, _ = cv2.Rodrigues(np.eye(3))
            t_vec = np.zeros((3, 1))
            dist_coeffs = np.array([k1, k2, p1, p2, k3])
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            
            # Only project points in front of camera
            valid_corners = corners_c[2, :] > 0
            if not np.any(valid_corners):
                continue
                
            points_3d_cam = corners_c[:3, :].T # Nx3
            
            projected_points, _ = cv2.projectPoints(points_3d_cam, r_vec, t_vec, camera_matrix, dist_coeffs)
            projected_points = projected_points.squeeze().astype(np.int32)
            
            # Draw Convex Hull of the projected points as mask
            if len(projected_points) > 0:
                hull = cv2.convexHull(projected_points)
                cv2.fillConvexPoly(mask, hull, 0) # Fill with BLACK (0)
                
        # Save Mask
        # COLMAP expects .png.mask extension usually, or just a separate folder.
        # Let's save as .png.mask in the same folder for simplicity, or we can separate.
        # If COLMAP uses --ImageReader.mask_path, it's better to have same filename in a separate folder.
        # For now, let's create a parallel 'masks' folder.
        
        mask_dir = os.path.join(output_dir, 'masks', cam_name)
        ensure_dir(mask_dir)
        
        mask_path = os.path.join(mask_dir, f'{frame_idx:06d}.png') # COLMAP usually reads png masks
        cv2.imwrite(mask_path, mask)

def get_pose(frame):
    """
    Return vehicle global pose (4x4 matrix) as list.
    """
    # transform is a list of 16 floats (row-major)
    pose = np.array(frame.pose.transform).reshape(4, 4)
    return pose.tolist()

def get_calib(frame):
    """
    Return camera calibrations (Intrinsics and Extrinsics) as dictionary.
    """
    cam_name_map = {
        1: 'FRONT',
        2: 'FRONT_LEFT',
        3: 'FRONT_RIGHT',
        4: 'SIDE_LEFT',
        5: 'SIDE_RIGHT'
    }

    calibs = {}
    for camera in frame.context.camera_calibrations:
        cam_name = cam_name_map.get(camera.name, 'UNKNOWN')
        if cam_name == 'UNKNOWN':
            continue
        
        # Extrinsic: Vehicle to Camera
        extrinsic = np.array(camera.extrinsic.transform).reshape(4, 4).tolist()
        
        # Intrinsic: [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}]
        intrinsic = np.array(camera.intrinsic).tolist()
        
        calibs[cam_name] = {
            'extrinsic': extrinsic, # Camera to Vehicle
            'intrinsic': intrinsic,
            'width': camera.width,
            'height': camera.height
        }
    return calibs

def extract_tfrecord(tfrecord_path, output_dir):
    if dataset_pb2 is None:
        print("Error: waymo_open_dataset module is missing. Cannot proceed.")
        return

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    
    print(f"Processing {tfrecord_path}...")
    
    vehicle_poses = {} # Key: frame_idx, Value: 4x4 matrix
    calibration_info = None # Assuming calibration is constant for the segment (usually true)
    
    # Pre-create image dirs
    ensure_dir(os.path.join(output_dir, 'images'))
    
    for i, data in enumerate(tqdm(dataset)):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        
        # Save Images and Masks
        save_image_and_mask(frame, i, output_dir)
        
        # Store Pose
        vehicle_poses[f'{i:06d}'] = get_pose(frame)
        
        # Store Calibration (Only once from the first frame)
        if calibration_info is None:
            calibration_info = get_calib(frame)
            
    # Save Poses to JSON
    poses_dir = os.path.join(output_dir, 'poses')
    ensure_dir(poses_dir)
    with open(os.path.join(poses_dir, 'vehicle_poses.json'), 'w') as f:
        json.dump(vehicle_poses, f, indent=4)
        
    # Save Calibration to JSON
    calib_dir = os.path.join(output_dir, 'calibration')
    ensure_dir(calib_dir)
    with open(os.path.join(calib_dir, 'intrinsics_extrinsics.json'), 'w') as f:
        json.dump(calibration_info, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Extract data from Waymo Open Dataset TFRecords')
    parser.add_argument('input_path', type=str, help='Path to .tfrecord file or directory containing .tfrecord files')
    parser.add_argument('output_dir', type=str, help='Directory to save extracted data')
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_dir = args.output_dir
    
    if os.path.isdir(input_path):
        tfrecord_files = sorted(glob(os.path.join(input_path, '*.tfrecord')))
    else:
        tfrecord_files = [input_path]
        
    print(f"Found {len(tfrecord_files)} TFRecord files.")
    
    for tf_file in tfrecord_files:
        # Create a subdirectory for each segment if processing multiple files
        # segment_name = os.path.splitext(os.path.basename(tf_file))[0]
        # segment_out_dir = os.path.join(output_dir, segment_name)
        
        # For simplicity, if only one file or user wants merged output, use output_dir directly.
        # But usually Waymo segments are distinct. Let's create subfolder per segment.
        segment_name = os.path.splitext(os.path.basename(tf_file))[0]
        segment_out_dir = os.path.join(output_dir, segment_name)
        
        extract_tfrecord(tf_file, segment_out_dir)

if __name__ == '__main__':
    main()
