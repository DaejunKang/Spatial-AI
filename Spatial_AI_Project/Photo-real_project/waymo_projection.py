import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from glob import glob
from tqdm import tqdm

# Try to import waymo_open_dataset
try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    dataset_pb2 = None
    print("Warning: 'waymo_open_dataset' not found. Please install it to run this script.")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_camera_calibrations(frame):
    """
    Extract camera calibrations from the frame context.
    Returns a dictionary mapping camera name (int) to calibration data.
    """
    calibrations = {}
    for calib in frame.context.camera_calibrations:
        calibrations[calib.name] = calib
    return calibrations

def compute_box_3d_corners(box):
    """
    Compute the 8 corners of a 3D bounding box in the object's local frame,
    then rotate and translate to the vehicle frame.
    """
    l, w, h = box.length, box.width, box.height
    heading = box.heading
    cx, cy, cz = box.center_x, box.center_y, box.center_z

    # 8 corners in local frame (centered at 0,0,0)
    # Order: Front-Left-Top, Front-Left-Bottom, ...
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    
    corners = np.vstack([x_corners, y_corners, z_corners]) # 3x8
    
    # Rotation matrix around Z-axis
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    R = np.array([
        [cos_h, -sin_h, 0],
        [sin_h, cos_h, 0],
        [0, 0, 1]
    ])
    
    # Rotate and Translate
    corners_3d = R @ corners
    corners_3d[0, :] += cx
    corners_3d[1, :] += cy
    corners_3d[2, :] += cz
    
    return corners_3d

def project_corners_to_image(corners_3d, extrinsic, intrinsic):
    """
    Project 3D corners from Vehicle Frame to Image Plane.
    
    Args:
        corners_3d: 3xN numpy array of 3D points in Vehicle Frame.
        extrinsic: Waymo CameraExtrinsic (Transform from Camera to Vehicle).
        intrinsic: Waymo CameraIntrinsic.
        
    Returns:
        projected_points: Nx2 numpy array of 2D image coordinates.
    """
    # 1. Transform Vehicle -> Camera
    # Extrinsic provided is Camera -> Vehicle
    T_c_v_mat = np.array(extrinsic.transform).reshape(4, 4)
    T_v_c_mat = np.linalg.inv(T_c_v_mat) # Vehicle -> Camera
    
    # Convert corners to homogeneous coordinates
    corners_homo = np.vstack([corners_3d, np.ones((1, corners_3d.shape[1]))]) # 4xN
    
    # Transform to Camera Frame
    corners_cam = T_v_c_mat @ corners_homo # 4xN
    
    # Check if points are in front of the camera (z > 0)
    # We only care if the box is generally in front. 
    # Individual corners might be behind if the box is clipping the camera plane, 
    # but for simplicity we project all.
    
    # 2. Project to Image Plane using OpenCV
    fx = intrinsic[0]
    fy = intrinsic[1]
    cx = intrinsic[2]
    cy = intrinsic[3]
    k1 = intrinsic[4]
    k2 = intrinsic[5]
    p1 = intrinsic[6]
    p2 = intrinsic[7]
    k3 = intrinsic[8]
    
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coeffs = np.array([k1, k2, p1, p2, k3])
    
    # Points in camera frame (3xN)
    points_3d_cam = corners_cam[:3, :].T # Nx3
    
    # Use cv2.projectPoints
    # r_vec and t_vec are zero because points are already in camera frame
    r_vec = np.zeros((3, 1))
    t_vec = np.zeros((3, 1))
    
    image_points, _ = cv2.projectPoints(points_3d_cam, r_vec, t_vec, camera_matrix, dist_coeffs)
    image_points = image_points.squeeze() # Nx2
    
    return image_points, corners_cam

def process_frame(frame, frame_idx, output_dir, viz=True, mask=True):
    cam_name_map = {
        1: 'FRONT',
        2: 'FRONT_LEFT',
        3: 'FRONT_RIGHT',
        4: 'SIDE_LEFT',
        5: 'SIDE_RIGHT'
    }
    
    calibrations = get_camera_calibrations(frame)
    labels = frame.laser_labels
    
    for img in frame.images:
        cam_name = cam_name_map.get(img.name, f'UNKNOWN_{img.name}')
        calib = calibrations.get(img.name)
        
        if not calib:
            continue
            
        # Decode Image
        img_tensor = tf.image.decode_jpeg(img.image)
        img_np = img_tensor.numpy()
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        height, width = img_bgr.shape[:2]
        
        # Prepare Mask (Black background, White objects)
        mask_img = np.zeros((height, width), dtype=np.uint8)
        
        # Prepare Visualization Image
        viz_img = img_bgr.copy()
        
        for label in labels:
            # Optionally filter by type: 
            # 1: VEHICLE, 2: PEDESTRIAN, 3: SIGN, 4: CYCLIST
            if label.type not in [1, 2, 4]:
                continue
                
            box = label.box
            
            # 1. Compute 3D corners in Vehicle Frame
            corners_3d = compute_box_3d_corners(box)
            
            # 2. Project to Image
            pts_2d, pts_cam = project_corners_to_image(corners_3d, calib.extrinsic, calib.intrinsic)
            
            # Check if box is in front of camera (at least some corners)
            if np.any(pts_cam[2, :] <= 0):
                continue
                
            # Check if box is within image boundaries (roughly)
            # A more robust check is whether the convex hull intersects the image rect.
            # For now, we check if at least one point is within meaningful range.
            if np.all(pts_2d[:, 0] < 0) or np.all(pts_2d[:, 0] > width) or \
               np.all(pts_2d[:, 1] < 0) or np.all(pts_2d[:, 1] > height):
                continue

            pts_2d_int = pts_2d.astype(np.int32)
            
            # 3. Create Mask (Convex Hull of projections)
            hull = cv2.convexHull(pts_2d_int)
            cv2.fillConvexPoly(mask_img, hull, 255)
            
            # 4. Visualization (Draw Box)
            if viz:
                # Draw base lines
                # 0-1, 1-2, 2-3, 3-0 (Bottom face? No, checking order)
                # Helper to draw lines
                def draw_line(i, j):
                    pt1 = tuple(pts_2d_int[i])
                    pt2 = tuple(pts_2d_int[j])
                    cv2.line(viz_img, pt1, pt2, (0, 255, 0), 2)
                
                # Based on compute_box_3d_corners order:
                # x: [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
                # y: [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
                # z: [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
                # Indices:
                # 0: (+, +, +), 1: (+, -, +), 2: (-, -, +), 3: (-, +, +)  (Top)
                # 4: (+, +, -), 5: (+, -, -), 6: (-, -, -), 7: (-, +, -)  (Bottom)
                
                # Top face
                draw_line(0, 1); draw_line(1, 2); draw_line(2, 3); draw_line(3, 0)
                # Bottom face
                draw_line(4, 5); draw_line(5, 6); draw_line(6, 7); draw_line(7, 4)
                # Vertical pillars
                draw_line(0, 4); draw_line(1, 5); draw_line(2, 6); draw_line(3, 7)

        # Save results
        if mask:
            mask_out_dir = os.path.join(output_dir, 'masks', cam_name)
            ensure_dir(mask_out_dir)
            cv2.imwrite(os.path.join(mask_out_dir, f'{frame_idx:06d}.png'), mask_img)
            
        if viz:
            viz_out_dir = os.path.join(output_dir, 'vis', cam_name)
            ensure_dir(viz_out_dir)
            cv2.imwrite(os.path.join(viz_out_dir, f'{frame_idx:06d}.jpg'), viz_img)
            
        # Also save original image if needed (optional)
        img_out_dir = os.path.join(output_dir, 'images', cam_name)
        ensure_dir(img_out_dir)
        cv2.imwrite(os.path.join(img_out_dir, f'{frame_idx:06d}.jpg'), img_bgr)

def main():
    parser = argparse.ArgumentParser(description='Project Waymo 3D boxes to 2D images and generate masks.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to .tfrecord file or directory.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory.')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of frames to process.')
    
    args = parser.parse_args()
    
    if dataset_pb2 is None:
        print("Error: waymo_open_dataset is not installed.")
        return

    if os.path.isdir(args.input_path):
        tfrecord_files = sorted(glob(os.path.join(args.input_path, '*.tfrecord')))
    else:
        tfrecord_files = [args.input_path]
        
    print(f"Found {len(tfrecord_files)} TFRecord files.")
    
    frame_count = 0
    
    for tf_file in tfrecord_files:
        print(f"Processing {tf_file}...")
        dataset = tf.data.TFRecordDataset(tf_file, compression_type='')
        
        for data in tqdm(dataset):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            
            process_frame(frame, frame_count, args.output_dir)
            
            frame_count += 1
            if args.limit and frame_count >= args.limit:
                print(f"Reached limit of {args.limit} frames.")
                return

if __name__ == '__main__':
    main()
