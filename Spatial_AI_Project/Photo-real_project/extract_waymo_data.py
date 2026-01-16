import os
import tensorflow as tf
import numpy as np
import cv2
import argparse
from glob import glob
from tqdm import tqdm

# Try to import waymo_open_dataset
try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    print("Warning: 'waymo_open_dataset' not found. This script requires the Waymo Open Dataset package.")
    print("Please install it in a compatible environment (Python 3.7 - 3.10 recommended).")
    # For coding assistance purpose, we define a dummy placeholder if import fails, 
    # but execution will fail later.
    dataset_pb2 = None

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(frame, frame_idx, output_dir):
    """
    Save images from 5 cameras.
    """
    # Camera names mapping based on Waymo Proto
    # 1: FRONT, 2: FRONT_LEFT, 3: FRONT_RIGHT, 4: SIDE_LEFT, 5: SIDE_RIGHT
    cam_name_map = {
        1: 'FRONT',
        2: 'FRONT_LEFT',
        3: 'FRONT_RIGHT',
        4: 'SIDE_LEFT',
        5: 'SIDE_RIGHT'
    }

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
        
        img_path = os.path.join(cam_dir, f'{frame_idx:06d}.png')
        cv2.imwrite(img_path, img_bgr)

def save_pose(frame, frame_idx, output_dir):
    """
    Save vehicle global pose (4x4 matrix).
    """
    pose_dir = os.path.join(output_dir, 'poses')
    ensure_dir(pose_dir)
    
    # transform is a list of 16 floats (row-major)
    pose = np.array(frame.pose.transform).reshape(4, 4)
    
    pose_path = os.path.join(pose_dir, f'{frame_idx:06d}.txt')
    np.savetxt(pose_path, pose, fmt='%.8f')

def save_calib(frame, frame_idx, output_dir):
    """
    Save camera calibrations (Intrinsics and Extrinsics).
    """
    calib_dir = os.path.join(output_dir, 'calib')
    ensure_dir(calib_dir)
    
    cam_name_map = {
        1: 'FRONT',
        2: 'FRONT_LEFT',
        3: 'FRONT_RIGHT',
        4: 'SIDE_LEFT',
        5: 'SIDE_RIGHT'
    }

    calib_file = os.path.join(calib_dir, f'{frame_idx:06d}.txt')
    
    with open(calib_file, 'w') as f:
        for camera in frame.context.camera_calibrations:
            cam_name = cam_name_map.get(camera.name, 'UNKNOWN')
            if cam_name == 'UNKNOWN':
                continue
            
            # Extrinsic: Vehicle to Camera
            # Note: Waymo stores extrinsic as Transform from Camera to Vehicle usually? 
            # Proto says: "transform from camera frame to vehicle frame"
            extrinsic = np.array(camera.extrinsic.transform).reshape(4, 4)
            
            # Intrinsic: [f_u, f_v, c_u, c_v, k{1, 2}, p{1, 2}, k{3}]
            intrinsic = np.array(camera.intrinsic)
            
            f.write(f'Camera: {cam_name}\n')
            f.write('Extrinsic (Camera to Vehicle):\n')
            np.savetxt(f, extrinsic, fmt='%.8f')
            f.write('Intrinsic (1d array: f_u, f_v, c_u, c_v, ...):\n')
            np.savetxt(f, intrinsic[None, :], fmt='%.8f') # Save as row
            f.write(f'Width: {camera.width}\n')
            f.write(f'Height: {camera.height}\n')
            f.write('-' * 20 + '\n')

def extract_tfrecord(tfrecord_path, output_dir):
    if dataset_pb2 is None:
        print("Error: waymo_open_dataset module is missing. Cannot proceed.")
        return

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    
    print(f"Processing {tfrecord_path}...")
    for i, data in enumerate(tqdm(dataset)):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        
        save_image(frame, i, output_dir)
        save_pose(frame, i, output_dir)
        save_calib(frame, i, output_dir)

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
