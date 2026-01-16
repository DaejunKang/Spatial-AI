import os
import json
import numpy as np
import argparse
import sqlite3

def qvec2rotmat(qvec):
    """
    Convert quaternion to rotation matrix.
    qvec = [w, x, y, z]
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
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    """
    Convert rotation matrix to quaternion.
    """
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0:
        q *= -1
    return q

def write_cameras_txt(cameras, output_path):
    """
    Write cameras.txt file for COLMAP.
    cameras: dict of {cam_name: {intrinsic, width, height}}
    """
    with open(output_path, 'w') as f:
        f.write("# Camera list with one line of data per camera.\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        
        # Waymo provides intrinsics as [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]
        # This maps closely to OPENCV model in COLMAP: fx, fy, cx, cy, k1, k2, p1, p2
        # Note: Waymo k3 is usually 0 or small, COLMAP OPENCV model doesn't use k3 directly in 8 params?
        # COLMAP 'OPENCV' model params: fx, fy, cx, cy, k1, k2, p1, p2
        # COLMAP 'FULL_OPENCV' model params: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        
        for cam_id, (cam_name, cam_data) in enumerate(cameras.items(), start=1):
            w = cam_data['width']
            h = cam_data['height']
            # waymo intrinsic: [f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3]
            intr = cam_data['intrinsic']
            
            # Use FULL_OPENCV to accommodate k3 if needed, or just OPENCV if k3 is negligible.
            # Let's use OPENCV and ignore k3 for simplicity/compatibility unless k3 is large.
            # Or better, FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
            # We have up to k3. k4-k6 can be 0.
            
            # Map Waymo to COLMAP FULL_OPENCV
            params = [
                intr[0], intr[1], intr[2], intr[3], # fx, fy, cx, cy
                intr[4], intr[5], intr[6], intr[7], # k1, k2, p1, p2
                intr[8], 0.0, 0.0, 0.0              # k3, k4, k5, k6
            ]
            
            params_str = " ".join(map(str, params))
            f.write(f"{cam_id} FULL_OPENCV {w} {h} {params_str}\n")
            
    return {name: i for i, name in enumerate(cameras.keys(), start=1)}

def write_images_txt(poses, cameras, cam_name_to_id, output_path, calibration):
    """
    Write images.txt file for COLMAP.
    poses: dict {frame_id: 4x4 matrix (vehicle pose)}
    calibration: dict {cam_name: {extrinsic, ...}}
    """
    with open(output_path, 'w') as f:
        f.write("# Image list with two lines of data per image.\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        image_id = 1
        sorted_frame_ids = sorted(poses.keys())
        
        for frame_id in sorted_frame_ids:
            # T_global_vehicle (Vehicle pose in global frame)
            T_g_v = np.array(poses[frame_id])
            
            for cam_name, cam_id in cam_name_to_id.items():
                if cam_name not in calibration:
                    continue
                    
                # T_vehicle_camera (Camera extrinsic: Camera frame in Vehicle frame? Or Vehicle to Camera?)
                # Waymo doc: "extrinsic: Transform from camera frame to vehicle frame."
                # So it is T_vehicle_camera.
                T_v_c = np.array(calibration[cam_name]['extrinsic'])
                
                # We need T_camera_global (Global point -> Camera frame)
                # T_global_camera = T_global_vehicle * T_vehicle_camera
                T_g_c = T_g_v @ T_v_c
                
                # COLMAP expects World to Camera transform (T_cw)
                # T_cw = inv(T_gc)
                T_c_g = np.linalg.inv(T_g_c)
                
                # Extract Rotation (qvec) and Translation (tvec)
                R = T_c_g[:3, :3]
                t = T_c_g[:3, 3]
                q = rotmat2qvec(R)
                
                img_name = f"{cam_name}/{frame_id}.png"
                
                f.write(f"{image_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {cam_id} {img_name}\n")
                f.write("\n") # Empty points line
                
                image_id += 1

def write_points3d_txt(output_path):
    """
    Write empty points3D.txt.
    """
    with open(output_path, 'w') as f:
        f.write("# 3D point list with one line of data per point.\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

def main():
    parser = argparse.ArgumentParser(description="Convert Waymo extracted data to COLMAP text format.")
    parser.add_argument('input_dir', type=str, help="Path to segment directory (containing poses/ and calibration/)")
    parser.add_argument('output_dir', type=str, help="Directory to save COLMAP .txt files")
    
    args = parser.parse_args()
    
    poses_path = os.path.join(args.input_dir, 'poses', 'vehicle_poses.json')
    calib_path = os.path.join(args.input_dir, 'calibration', 'intrinsics_extrinsics.json')
    
    if not os.path.exists(poses_path) or not os.path.exists(calib_path):
        print("Error: Input files not found.")
        return
        
    with open(poses_path, 'r') as f:
        poses = json.load(f)
        
    with open(calib_path, 'r') as f:
        calibration = json.load(f)
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Write cameras.txt
    cam_name_to_id = write_cameras_txt(calibration, os.path.join(args.output_dir, 'cameras.txt'))
    
    # Write images.txt
    write_images_txt(poses, calibration, cam_name_to_id, os.path.join(args.output_dir, 'images.txt'), calibration)
    
    # Write points3D.txt (Empty)
    write_points3d_txt(os.path.join(args.output_dir, 'points3D.txt'))
    
    print(f"COLMAP files written to {args.output_dir}")

if __name__ == "__main__":
    main()
