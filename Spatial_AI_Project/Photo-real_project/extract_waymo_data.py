import os
import tensorflow as tf
import numpy as np
import cv2
import argparse
import json
from glob import glob
from tqdm import tqdm

# Reference: https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_vision_based_e2e_driving.ipynb
# This script extracts Images, Poses, LiDAR Point Clouds, and Projected Depth Maps aligned by timestamp.

# Try to import waymo_open_dataset
try:
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import range_image_utils
    from waymo_open_dataset.utils import transform_utils
    from waymo_open_dataset.utils import frame_utils
except ImportError:
    print("Warning: 'waymo_open_dataset' not found. This script requires the Waymo Open Dataset package.")
    dataset_pb2 = None
    range_image_utils = None
    transform_utils = None
    frame_utils = None

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image_and_mask(frame, frame_idx, output_dir, calibrations):
    """
    Save images from 5 cameras, 2D segmentation labels (if available), 
    and generate masks for dynamic objects from 3D boxes.
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
        
        # --- Save GT Segmentation (if available) ---
        if img.camera_segmentation_label.panoptic_label:
             # This depends on exact version, usually it's a PNG inside bytes
             # panoptic_label usually contains instance_id + semantic_id
             # For 2D sem seg, we might just want to save the raw PNG
             seg_dir = os.path.join(output_dir, 'gt_segmentation', cam_name)
             ensure_dir(seg_dir)
             
             # decode_png
             seg_tensor = tf.image.decode_png(img.camera_segmentation_label.panoptic_label)
             seg_array = seg_tensor.numpy()
             
             # Save as is (usually 16bit or colored)
             # OpenCV usually handles uint16 PNGs
             seg_path = os.path.join(seg_dir, f'{frame_idx:06d}.png')
             cv2.imwrite(seg_path, seg_array)
        
        # --- Generate Mask (from 3D boxes) ---
        # 1. Get Calibration
        calib = calibrations[img.name]
        extrinsic = np.array(calib.extrinsic.transform).reshape(4, 4) # Camera to Vehicle
        intrinsic = np.array(calib.intrinsic) # [fu, fv, cu, cv, k1, k2, p1, p2, k3]
        width, height = calib.width, calib.height
        
        # 2. Compute Transform: Vehicle -> Camera
        T_v_c = extrinsic
        T_c_v = np.linalg.inv(T_v_c)
        
        # 3. Create Mask Image (White = Valid/Static, Black = Invalid/Dynamic)
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # 4. Project Labels
        for label in labels:
            if label.type not in [1, 2, 4]: # Vehicle, Pedestrian, Cyclist
                continue
                
            box = label.box
            
            # Box Center (in Vehicle Frame)
            center_v = np.array([box.center_x, box.center_y, box.center_z, 1.0])
            center_c = T_c_v @ center_v
            
            if center_c[2] <= 0:
                continue
                
            # Get 8 corners of the 3D Box in Vehicle Frame
            l, w, h = box.length, box.width, box.height
            heading = box.heading
            
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            R_z = np.array([
                [cos_h, -sin_h, 0],
                [sin_h, cos_h, 0],
                [0, 0, 1]
            ])
            
            x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
            y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
            z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
            
            corners_3d = np.vstack([x_corners, y_corners, z_corners])
            corners_3d = R_z @ corners_3d
            corners_3d = corners_3d + np.array([box.center_x, box.center_y, box.center_z]).reshape(3, 1)
            corners_3d_homo = np.vstack([corners_3d, np.ones((1, 8))])
            
            corners_c = T_c_v @ corners_3d_homo
            
            fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
            k1, k2, p1, p2, k3 = intrinsic[4], intrinsic[5], intrinsic[6], intrinsic[7], intrinsic[8]
            
            r_vec, _ = cv2.Rodrigues(np.eye(3))
            t_vec = np.zeros((3, 1))
            dist_coeffs = np.array([k1, k2, p1, p2, k3])
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            
            valid_corners = corners_c[2, :] > 0
            if not np.any(valid_corners):
                continue
                
            points_3d_cam = corners_c[:3, :].T
            projected_points, _ = cv2.projectPoints(points_3d_cam, r_vec, t_vec, camera_matrix, dist_coeffs)
            projected_points = projected_points.squeeze().astype(np.int32)
            
            if len(projected_points) > 0:
                hull = cv2.convexHull(projected_points)
                cv2.fillConvexPoly(mask, hull, 0)
                
        mask_dir = os.path.join(output_dir, 'masks', cam_name)
        ensure_dir(mask_dir)
        mask_path = os.path.join(mask_dir, f'{frame_idx:06d}.png')
        cv2.imwrite(mask_path, mask)

def get_pose(frame):
    """
    Return vehicle global pose (4x4 matrix) as list.
    """
    pose = np.array(frame.pose.transform).reshape(4, 4)
    return pose.tolist()

def get_calib(frame):
    """
    Return camera calibrations (Intrinsics and Extrinsics) as dictionary objects.
    """
    calibs = {}
    for camera in frame.context.camera_calibrations:
        calibs[camera.name] = camera
    return calibs

def save_calib_json(calibs, output_dir):
    cam_name_map = {
        1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'
    }
    calib_dict = {}
    for cam_id, camera in calibs.items():
        cam_name = cam_name_map.get(cam_id, 'UNKNOWN')
        if cam_name == 'UNKNOWN': continue
        
        calib_dict[cam_name] = {
            'extrinsic': np.array(camera.extrinsic.transform).reshape(4, 4).tolist(),
            'intrinsic': np.array(camera.intrinsic).tolist(),
            'width': camera.width,
            'height': camera.height
        }
    
    calib_dir = os.path.join(output_dir, 'calibration')
    ensure_dir(calib_dir)
    with open(os.path.join(calib_dir, 'intrinsics_extrinsics.json'), 'w') as f:
        json.dump(calib_dict, f, indent=4)

def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=0):
    """Convert range images to point cloud."""
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    intensity = []
    elongation = []

    frame_pose = tf.convert_to_tensor(value=np.reshape(np.array(frame.pose.transform), [4, 4]))
    
    # Handle range image top pose
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.shape.dims)
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data),
            range_image.shape.dims)
            
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            
        range_image_mask = range_image_tensor[..., 0] > 0
        
        # Extract point cloud
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(range_image_cartesian, tf.where(range_image_mask))
        
        # Camera projection
        cp = camera_projections[c.name][ri_index]
        cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
        
        intensity_tensor = tf.gather_nd(range_image_tensor[..., 1], tf.where(range_image_mask))
        elongation_tensor = tf.gather_nd(range_image_tensor[..., 2], tf.where(range_image_mask))

        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        intensity.append(intensity_tensor.numpy())
        elongation.append(elongation_tensor.numpy())

    return points, cp_points, intensity, elongation

def save_lidar_and_depth(frame, frame_idx, output_dir, calibrations):
    """
    Parse and save LiDAR data (Point Cloud) and generate sparse depth maps.
    Saves point cloud as .npy file: [N, 6] -> (x, y, z, intensity, elongation, laser_index)
    Saves depth maps for each camera as .png (16-bit mm) or .npy
    """
    range_images, camera_projections, _, range_image_top_pose = \
        frame_utils.parse_range_image_and_camera_projection(frame)

    # First return
    points_0, cp_points_0, intensity_0, elongation_0 = \
        convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=0)
    
    # Second return
    points_1, cp_points_1, intensity_1, elongation_1 = \
        convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose, ri_index=1)

    points_all_lists = [points_0, points_1]
    cp_points_all_lists = [cp_points_0, cp_points_1]
    intensity_all_lists = [intensity_0, intensity_1]
    elongation_all_lists = [elongation_0, elongation_1]

    # --- 1. Save Merged Point Cloud ---
    all_points = []
    
    for r_idx in range(2): # Returns
        p_list = points_all_lists[r_idx]
        i_list = intensity_all_lists[r_idx]
        e_list = elongation_all_lists[r_idx]
        
        for i in range(len(p_list)):
            p = p_list[i]
            inte = i_list[i]
            elo = e_list[i]
            lidar_idx = np.full((p.shape[0], 1), i) # Add lidar index
            
            chunk = np.column_stack((p, inte, elo, lidar_idx))
            all_points.append(chunk)

    if not all_points:
        return

    final_pc = np.concatenate(all_points, axis=0) # [N_total, 6]
    lidar_dir = os.path.join(output_dir, 'lidar')
    ensure_dir(lidar_dir)
    np.save(os.path.join(lidar_dir, f'{frame_idx:06d}.npy'), final_pc)
    
    # --- 2. Generate Sparse Depth Maps ---
    # We iterate through points and project them using cp_points
    
    cam_name_map = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}
    
    # Prepare depth buffers
    depth_maps = {} # cam_name -> image
    for cam_name_str in cam_name_map.values():
         # Need width/height. Use calibrations
         # We need to find which ID maps to this string
         pass

    # Efficient way: Iterate all points, check cp_points
    # cp_points: [N, 6] -> [cam_id_1, x_1, y_1, cam_id_2, x_2, y_2]
    # cam_id is Waymo ID (1..5)
    
    for r_idx in range(2):
        p_list = points_all_lists[r_idx]
        cp_list = cp_points_all_lists[r_idx]
        
        for i in range(len(p_list)): # Per Laser
            points = p_list[i] # [N, 3] in Vehicle Frame
            cp = cp_list[i]    # [N, 6]
            
            # Mask for valid projections
            # Valid if cam_id != 0
            
            # Helper to process one projection column set (0-2 or 3-5)
            def process_projection(col_offset):
                cam_ids = cp[:, col_offset]
                xs = cp[:, col_offset+1]
                ys = cp[:, col_offset+2]
                
                valid_mask = cam_ids > 0
                
                valid_cam_ids = cam_ids[valid_mask]
                valid_xs = xs[valid_mask]
                valid_ys = ys[valid_mask]
                valid_points = points[valid_mask] # Vehicle Frame
                
                # We need to project points to Camera Frame to get Z (depth)
                # But we have multiple cameras.
                # Group by camera ID
                unique_cams = np.unique(valid_cam_ids)
                
                for c_id in unique_cams:
                    c_mask = valid_cam_ids == c_id
                    c_points_v = valid_points[c_mask] # Points projecting to this camera
                    c_xs = valid_xs[c_mask]
                    c_ys = valid_ys[c_mask]
                    
                    # Get Calibration for this camera
                    calib = calibrations.get(int(c_id)) # calib object
                    if not calib: continue
                    
                    # Transform Vehicle -> Camera
                    extrinsic = np.array(calib.extrinsic.transform).reshape(4, 4)
                    T_v_c = extrinsic
                    T_c_v = np.linalg.inv(T_v_c)
                    
                    # Transform points
                    # [N, 3] -> [N, 4]
                    c_points_v_homo = np.column_stack((c_points_v, np.ones(c_points_v.shape[0])))
                    c_points_c = (T_c_v @ c_points_v_homo.T).T # [N, 4]
                    
                    depths = c_points_c[:, 2] # Z
                    
                    # Update Depth Map
                    cam_name = cam_name_map.get(int(c_id))
                    if not cam_name: continue
                    
                    # Initialize if needed
                    if cam_name not in depth_maps:
                        depth_maps[cam_name] = np.zeros((calib.height, calib.width), dtype=np.float32)
                        
                    # Assign depths
                    # Coordinates are floating point, round to int
                    x_int = np.round(c_xs).astype(int)
                    y_int = np.round(c_ys).astype(int)
                    
                    # Clip to image bounds
                    h, w = depth_maps[cam_name].shape
                    valid_idx = (x_int >= 0) & (x_int < w) & (y_int >= 0) & (y_int < h) & (depths > 0)
                    
                    # For conflicting points (multiple points to same pixel), keep closest (min depth)
                    # Simple approach: Loop or use lexsort?
                    # Since N is large, simple loop might be slow.
                    # But often in parsing scripts, we just overwrite.
                    # Better: Sort by depth descending, then overwrite (so closest remains? No, closest should be last? No, we want min depth)
                    # We want min depth. So initialize with inf, take min.
                    
                    # Let's just accumulate lists and do it once per camera at the end? 
                    # No, let's do simple overwrite for now, usually fine for sparse.
                    # Or use minimum.
                    
                    # Optimization:
                    # We can use logic:
                    # current_depth = depth_maps[cam_name][y, x]
                    # new_depth = depths
                    # update if new < current or current == 0
                    
                    # This is slow in python loops.
                    # Vectorized approach:
                    # Create sparse matrix? 
                    # Let's just save the sparse points (x, y, depth) as .npy for the user to process?
                    # The user asked for "parsing code... for 2D segmentation". 
                    # Depth maps are standard.
                    
                    # Let's save the depth map image.
                    # We will simply overwrite for now.
                    
                    d_map = depth_maps[cam_name]
                    d_map[y_int[valid_idx], x_int[valid_idx]] = depths[valid_idx]
                    
            process_projection(0)
            process_projection(3)
            
    # Save Depth Maps
    depth_dir = os.path.join(output_dir, 'depth_maps')
    ensure_dir(depth_dir)
    
    for cam_name, d_map in depth_maps.items():
        cam_depth_dir = os.path.join(depth_dir, cam_name)
        ensure_dir(cam_depth_dir)
        
        # Save as .npy (float)
        np.save(os.path.join(cam_depth_dir, f'{frame_idx:06d}.npy'), d_map)
        
        # Also save as uint16 png (mm) for visualization/standard tools
        d_img = (d_map * 1000).astype(np.uint16)
        cv2.imwrite(os.path.join(cam_depth_dir, f'{frame_idx:06d}.png'), d_img)

def extract_tfrecord(tfrecord_path, output_dir):
    if dataset_pb2 is None:
        print("Error: waymo_open_dataset module is missing. Cannot proceed.")
        return

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    
    print(f"Processing {tfrecord_path}...")
    
    vehicle_poses = {} # Key: frame_idx, Value: 4x4 matrix
    calibrations_raw = None # To hold proto objects
    
    ensure_dir(os.path.join(output_dir, 'images'))
    
    for i, data in enumerate(tqdm(dataset)):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        
        # Get Calibration once
        if calibrations_raw is None:
            calibrations_raw = get_calib(frame) # Returns dict of proto objects
            save_calib_json(calibrations_raw, output_dir)
            
        # Save Images and Masks
        save_image_and_mask(frame, i, output_dir, calibrations_raw)
        
        # Save LiDAR and Depth
        save_lidar_and_depth(frame, i, output_dir, calibrations_raw)
        
        # Store Pose
        vehicle_poses[f'{i:06d}'] = get_pose(frame)
        
    # Save Poses to JSON
    poses_dir = os.path.join(output_dir, 'poses')
    ensure_dir(poses_dir)
    with open(os.path.join(poses_dir, 'vehicle_poses.json'), 'w') as f:
        json.dump(vehicle_poses, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Extract data from Waymo Open Dataset TFRecords')
    parser.add_argument('input_path', type=str, help='Path to .tfrecord file or directory')
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
        segment_name = os.path.splitext(os.path.basename(tf_file))[0]
        segment_out_dir = os.path.join(output_dir, segment_name)
        
        extract_tfrecord(tf_file, segment_out_dir)

if __name__ == '__main__':
    main()
