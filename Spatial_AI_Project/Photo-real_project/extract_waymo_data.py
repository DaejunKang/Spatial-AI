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

# Import common utilities
from waymo_utils import (
    ensure_dir,
    get_camera_name_map,
    project_3d_box_to_2d,
    get_calibration_dict
)

def save_image_and_mask(frame, frame_idx, output_dir):
    """
    Save images from 5 cameras and generate masks for dynamic objects.
    """
    # Camera names mapping
    cam_name_map = get_camera_name_map()

    # Extract 3D Labels (Laser Labels)
    labels = frame.laser_labels
    
    # Pre-parse calibration for this frame
    calibrations = get_calibration_dict(frame)
    calibrations_raw = {c.name: c for c in frame.context.camera_calibrations}

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
        cam_name = cam_name_map.get(img.name, 'UNKNOWN')
        if cam_name not in calibrations:
            continue
            
        calib_data = calibrations[cam_name]
        T_v_c = calib_data['extrinsic']  # Camera to Vehicle
        intrinsic = calib_data['intrinsic']
        width, height = calib_data['width'], calib_data['height']
        
        # 2. Compute Transform: Vehicle -> Camera
        T_c_v = np.linalg.inv(T_v_c)
        
        # 3. Create Mask Image (White = Valid/Static, Black = Invalid/Dynamic)
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # 4. Project Labels using common utility
        for label in labels:
            # Filter for dynamic objects (Vehicle, Pedestrian, Cyclist)
            if label.type not in [1, 2, 4]: 
                continue
            
            # Project 3D box to 2D
            projected_points = project_3d_box_to_2d(
                label.box, T_c_v, intrinsic, width, height
            )
            
            # Draw mask if projection successful
            if projected_points is not None and len(projected_points) > 0:
                hull = cv2.convexHull(projected_points)
                cv2.fillConvexPoly(mask, hull, 0)  # Fill with BLACK (0)
                
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
    calibs = get_calibration_dict(frame)
    
    # Convert numpy arrays to lists for JSON serialization
    for cam_name, calib_data in calibs.items():
        calibs[cam_name] = {
            'extrinsic': calib_data['extrinsic'].tolist(),
            'intrinsic': calib_data['intrinsic'].tolist(),
            'width': calib_data['width'],
            'height': calib_data['height']
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
