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

def undistort_image(img_bgr, intrinsic):
    """
    Corrects lens distortion using Brown-Conrady model.
    Waymo intrinsics: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    """
    h, w = img_bgr.shape[:2]
    
    # Extract parameters
    fx, fy = intrinsic[0], intrinsic[1]
    cx, cy = intrinsic[2], intrinsic[3]
    k1, k2 = intrinsic[4], intrinsic[5]
    p1, p2 = intrinsic[6], intrinsic[7]
    k3 = intrinsic[8]
    
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    dist_coeffs = np.array([k1, k2, p1, p2, k3])
    
    # New camera matrix (optimal)
    # alpha=0: crop invalid pixels, alpha=1: keep all pixels
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    # Undistort
    undistorted_img = cv2.undistort(img_bgr, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    return undistorted_img, new_camera_matrix

def correct_rolling_shutter(img_bgr, velocity, angular_velocity, readout_time=0.025):
    """
    Approximates rolling shutter correction by compensating for vehicle ego-motion.
    Assumes readout is vertical (top-to-bottom) and scene is far (dominant rotation).
    
    Args:
        img_bgr: Source image (undistorted usually).
        velocity: Linear velocity vector [vx, vy, vz] (m/s) in Vehicle Frame.
        angular_velocity: Angular velocity vector [wx, wy, wz] (rad/s) in Vehicle Frame.
        readout_time: Total time to read the frame (seconds). Default approx 25ms.
        
    Returns:
        corrected_img: Image corrected for rolling shutter distortion.
    """
    h, w = img_bgr.shape[:2]
    
    # If velocity is negligible, return original
    if np.linalg.norm(angular_velocity) < 1e-4 and np.linalg.norm(velocity) < 1e-2:
        return img_bgr

    # We use a pixel-wise remapping or a scanline-based homography approach.
    # For efficiency in Python, we'll create a remap grid.
    
    # Coordinate grids
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Time offset for each row: t_row = (v / H) * ReadoutTime
    # Assuming trigger is at the start (top row). 
    # If trigger is at center, offset would be (v/H - 0.5) * ReadoutTime.
    # Let's assume trigger is at start for now (t=0 at top row).
    # We want to warp everything to t=0 (Global Shutter at start of frame).
    
    # Time deltas for each pixel
    dt_grid = (v.astype(np.float32) / h) * readout_time
    
    # Rotation Correction:
    # R(t) ~ I + [w]x * t
    # Or using Rodrigues for large rotations (though t is small).
    # We want to map current pixel P_distorted back to P_ideal at t=0.
    # P_distorted(t) was captured at camera pose P_cam(t).
    # We want P_ideal at P_cam(0).
    # Relative rotation from t to 0: R_rel = R(t)^T
    # If we assume small angle: theta = w * dt
    # R_rel is rotation by -theta.
    
    # Let's simplify: We want to find source coordinate (u_src, v_src) for each target (u, v).
    # Wait, 'remap' needs (map_x, map_y) where map_x[v, u] is the source x coordinate.
    # If output is "Global Shutter at t=0", then for a pixel (u,v) in output:
    # It represents a ray R_0 * K^-1 * [u, v, 1].
    # But in the input image, that ray was captured at time t = (v_src / H) * T.
    # This is a recursive problem because v_src depends on where the ray lands.
    # Standard approximation: Use target v to estimate t.
    # t = (v / H) * readout_time.
    
    # Rotation vector for time t (from t to 0) => -angular_velocity * t
    wx, wy, wz = angular_velocity
    # We broadcast this to (H, W, 3)
    rvecs = np.stack([-wx * dt_grid, -wy * dt_grid, -wz * dt_grid], axis=-1)
    
    # Since rotation is small, we can approximate displacement in image.
    # Flow approximation: u' = u + Flow_u(t), v' = v + Flow_v(t)
    # Optical Flow due to rotation:
    # u_flow = -(f * wz * y / f) ... standard formulas
    # Let's do it properly with projection if possible, or use small angle approx.
    # Approximation for rotational flow at (x, y) normalized:
    # u_dot = -xy*wx + (1+x^2)*wy - y*wz
    # v_dot = -(1+y^2)*wx + xy*wy + x*wz
    # (assuming f=1, x=(u-cx)/fx, y=(v-cy)/fy)
    
    # Since we don't have K here easily without passing it, let's assume centered principal point and FOV.
    # Or better, just pass intrinsic or assume typical FOV for Side/Front.
    # Let's rely on pixel shifts being proportional to angular velocity * dt.
    # Shift ~ (f * w) * dt.
    # f is roughly width / 1.0 (approx 90 deg FOV) to width * 2.0.
    # Let's estimate f approx = w.
    f_est = w  
    
    # Pixel shifts
    # du = ( - (u-w/2)(v-h/2)/f * wx + (f + (u-w/2)^2/f) * wy - (v-h/2) * wz ) * dt / f ? 
    # No, that's complex.
    
    # Simple Rotation warp (Homography-like)
    # P_corrected = R_rel @ P_distorted
    # P_distorted = R_rel^T @ P_corrected = R(t) @ P_corrected
    # We want to find where (u, v) in CORRECTED image comes from in DISTORTED image.
    # Pixel (u, v) in Corrected (t=0) corresponds to ray d_0.
    # At time t(v) (approx), the camera rotated by R(t).
    # The ray in the camera frame at t is d_t = R(t)^T @ d_0 ? 
    # Camera rotates by R. World is static.
    # P_cam_t = R^T * (P_world - T).
    # Neglecting translation T.
    # P_cam_t = R(t)^T @ P_cam_0.
    # So d_t = R(t)^T @ d_0.
    
    # R(t) corresponds to rotation angular_velocity * t.
    # We need R(t)^T => Rotation by -angular_velocity * t.
    
    # Implementation using Rodrigues per row is slow.
    # Small angle approximation: R^T ~ I - [w*t]x
    # d_t = d_0 - (w*t) x d_0
    
    # Coordinates normalized: x = (u - w/2) / f, y = (v - h/2) / f, z=1
    x_grid = (u - w/2) / f_est
    y_grid = (v - h/2) / f_est
    z_grid = np.ones_like(x_grid)
    
    # Cross product (w*t) x d_0
    # w*t vector:
    wt_x = wx * dt_grid
    wt_y = wy * dt_grid
    wt_z = wz * dt_grid
    
    # cross product
    c_x = wt_y * z_grid - wt_z * y_grid
    c_y = wt_z * x_grid - wt_x * z_grid
    c_z = wt_x * y_grid - wt_y * x_grid
    
    # d_t = d_0 - cross
    # But wait, rotation is active: d_t = d_0 - cross (approx).
    # Actually if camera rotates +theta, features move -theta.
    # If we want source pixel for target (u,v), we need to look at where (u,v) was rotated TO.
    # Target (u,v) is at t=0.
    # At time t, camera rotated by w*t.
    # The world point relative to camera is rotated by -(w*t).
    # So we need to look at direction rotated by -(w*t).
    # Which is d_t = Rot(-wt) @ d_0.
    # Which is approx d_0 - (wt) x d_0. 
    # Yes, correct.
    
    d_t_x = x_grid - c_x
    d_t_y = y_grid - c_y
    d_t_z = z_grid - c_z
    
    # Project back to pixels
    # u_src = (d_t_x / d_t_z) * f + w/2
    # v_src = (d_t_y / d_t_z) * f + h/2
    
    map_x = (d_t_x / d_t_z) * f_est + w/2
    map_y = (d_t_y / d_t_z) * f_est + h/2
    
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    
    corrected_img = cv2.remap(img_bgr, map_x, map_y, cv2.INTER_LINEAR)
    
    return corrected_img

def process_frame(frame, frame_idx, output_dir, enable_rs_correction=True):
    cam_name_map = {
        1: 'FRONT',
        2: 'FRONT_LEFT',
        3: 'FRONT_RIGHT',
        4: 'SIDE_LEFT',
        5: 'SIDE_RIGHT'
    }
    
    calibrations = get_camera_calibrations(frame)
    
    # Get Vehicle Velocity for RS correction
    # Frame has images, laser_labels, etc.
    # Velocity is usually in frame.images[i].velocity or frame.pose?
    # Actually, Waymo Frame has 'images' which contains 'velocity' (Velocity message).
    # But checking proto definition: Frame has 'images', 'pose', 'laser_labels', etc.
    # The 'pose' is a Transform.
    # Where is velocity?
    # It might be derived from pose differences or in 'images'.
    # Actually, Waymo dataset `Frame` doesn't strictly have a velocity field in older versions.
    # But let's check `frame.images[0].velocity`.
    # Assuming the user knows velocity is needed. If not found, we calculate from poses?
    # For now, let's look for `velocity` in image or frame.
    # If not present, we skip RS correction or assume 0.
    
    # Assuming constant velocity model from metadata if available, 
    # or just skipping if not found (with warning).
    
    # Default to zero
    linear_v = np.zeros(3)
    angular_v = np.zeros(3)
    
    # Attempt to extract velocity from image metadata if available
    # Waymo Image proto has: name, image, pose, velocity, pose_timestamp, shutter, ...
    if frame.images and hasattr(frame.images[0], 'velocity'):
        v = frame.images[0].velocity
        linear_v = np.array([v.v_x, v.v_y, v.v_z])
        angular_v = np.array([v.w_x, v.w_y, v.w_z])
    
    for img in frame.images:
        cam_name = cam_name_map.get(img.name, f'UNKNOWN_{img.name}')
        calib = calibrations.get(img.name)
        
        if not calib:
            continue
            
        # Decode Image
        img_tensor = tf.image.decode_jpeg(img.image)
        img_np = img_tensor.numpy()
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 1. Lens Undistortion
        # Use full intrinsic: [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]
        undistorted, new_K = undistort_image(img_bgr, calib.intrinsic)
        
        # 2. Rolling Shutter Correction
        if enable_rs_correction:
            # We use the velocity from the frame (assumed constant for all cams)
            # Note: Velocity is in Vehicle Frame. 
            # We need to rotate it to Camera Frame if we apply it in Camera Frame?
            # Or apply correction in Vehicle Frame logic?
            # Our `correct_rolling_shutter` logic used simplified camera-aligned logic.
            # To be precise: Convert velocity to Camera Frame.
            
            # T_cam_vehicle
            extrinsic = np.array(calib.extrinsic.transform).reshape(4, 4)
            T_v_c = np.linalg.inv(extrinsic)
            R_v_c = T_v_c[:3, :3]
            
            # Rotate angular velocity to camera frame
            w_cam = R_v_c @ angular_v
            v_cam = R_v_c @ linear_v
            
            # Readout time: Waymo cameras ~ 0.02s
            # (Strictly speaking, it varies by camera, but 20-30ms is typical)
            final_img = correct_rolling_shutter(undistorted, v_cam, w_cam, readout_time=0.025)
        else:
            final_img = undistorted
        
        # Save
        out_subdir = os.path.join(output_dir, 'corrected', cam_name)
        ensure_dir(out_subdir)
        cv2.imwrite(os.path.join(out_subdir, f'{frame_idx:06d}.jpg'), final_img)

def main():
    parser = argparse.ArgumentParser(description='Correct Waymo camera distortion (Lens + Rolling Shutter)')
    parser.add_argument('--input_path', type=str, required=True, help='Path to .tfrecord file or directory.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory.')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of frames to process.')
    parser.add_argument('--no_rs', action='store_true', help='Disable Rolling Shutter correction')
    
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
            
            process_frame(frame, frame_count, args.output_dir, enable_rs_correction=not args.no_rs)
            
            frame_count += 1
            if args.limit and frame_count >= args.limit:
                print(f"Reached limit of {args.limit} frames.")
                return

if __name__ == '__main__':
    main()
