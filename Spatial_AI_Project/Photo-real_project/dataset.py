import os
import glob
import numpy as np
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class WaymoExtractedDataset(Dataset):
    """
    Dataset class for loading extracted Waymo Open Dataset data.
    Assumes data was extracted using the updated 'extract_waymo_data.py' script (JSON format).

    Directory Structure:
        root_dir/
            segment_name_1/
                images/
                    FRONT/ (000000.png, ...)
                    ...
                poses/
                    vehicle_poses.json
                calibration/
                    intrinsics_extrinsics.json
            segment_name_2/
            ...
    """

    def __init__(self, root_dir, split='train', cameras=None, transform=None):
        """
        Args:
            root_dir (str): Root directory containing extracted segments.
            split (str): 'train' or 'val' (currently loads all found segments).
            cameras (list): List of camera names to load. Default: ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        if cameras is None:
            self.cameras = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
        else:
            self.cameras = cameras

        self.samples = []
        self._load_dataset()

    def _load_dataset(self):
        # Find all segment directories
        segment_dirs = sorted([d for d in glob.glob(os.path.join(self.root_dir, '*')) if os.path.isdir(d)])

        if not segment_dirs:
            print(f"Warning: No segment directories found in {self.root_dir}")
            return

        for seg_dir in segment_dirs:
            segment_name = os.path.basename(seg_dir)

            # Load Poses JSON
            poses_path = os.path.join(seg_dir, 'poses', 'vehicle_poses.json')
            if not os.path.exists(poses_path):
                print(f"Warning: Poses file not found in {seg_dir}. Skipping.")
                continue

            with open(poses_path, 'r') as f:
                vehicle_poses = json.load(f) # Dict: frame_id -> 4x4 list

            # Load Calibration JSON
            calib_path = os.path.join(seg_dir, 'calibration', 'intrinsics_extrinsics.json')
            if not os.path.exists(calib_path):
                print(f"Warning: Calibration file not found in {seg_dir}. Skipping.")
                continue

            with open(calib_path, 'r') as f:
                calibration_info = json.load(f)

            sorted_frame_ids = sorted(vehicle_poses.keys())

            for frame_id in sorted_frame_ids:
                all_cams_exist = True
                image_paths = {}

                for cam in self.cameras:
                    img_p = os.path.join(seg_dir, 'images', cam, f'{frame_id}.png')
                    if not os.path.exists(img_p):
                        all_cams_exist = False
                        break
                    image_paths[cam] = img_p

                if not all_cams_exist:
                    continue

                self.samples.append({
                    'segment': segment_name,
                    'frame_id': frame_id,
                    'image_paths': image_paths,
                    'pose': vehicle_poses[frame_id],
                    'calib': calibration_info
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        # Load Images
        images = {}
        for cam, path in sample_info['image_paths'].items():
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            images[cam] = img

        # Global Pose (Vehicle to Global) - 4x4 Matrix
        pose = torch.tensor(sample_info['pose'], dtype=torch.float32)

        # Calibration
        calibs = {}
        full_calib = sample_info['calib']

        for cam in self.cameras:
            if cam in full_calib:
                cam_data = full_calib[cam]
                calibs[cam] = {
                    'extrinsic': torch.tensor(cam_data['extrinsic'], dtype=torch.float32),
                    'intrinsic': torch.tensor(cam_data['intrinsic'], dtype=torch.float32),
                    'width': cam_data['width'],
                    'height': cam_data['height']
                }

        return {
            'images': images, # Dict of [C, H, W] tensors
            'pose': pose,     # [4, 4] tensor
            'calib': calibs,  # Dict of Dicts (intrinsics, extrinsics)
            'meta': {
                'segment': sample_info['segment'],
                'frame_id': sample_info['frame_id']
            }
        }


class NREDataset(Dataset):
    """
    NRE 포맷 데이터를 로드하는 Dataset 클래스.
    waymo2nre.py 파이프라인의 출력 형식에 맞춤.

    Directory Structure (NRE format):
        root_dir/
            images/
                {prefix}{file_idx:03d}{frame_idx:03d}_{cam_name}.jpg
            poses/
                {prefix}{file_idx:03d}{frame_idx:03d}.json
            objects/
                {prefix}{file_idx:03d}{frame_idx:03d}.json
            point_clouds/
                {prefix}{file_idx:03d}{frame_idx:03d}.bin
            depth_maps/            (optional, from preprocessing)
                {frame_id}_{cam_name}.png
            masks/                 (optional, from preprocessing)
                {frame_id}_{cam_name}.png
            final_inpainted/       (optional, from inpainting)
                {frame_id}_{cam_name}.jpg
    """

    CAMERAS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']

    def __init__(self, root_dir, cameras=None, transform=None,
                 use_inpainted=False, use_depth=False):
        """
        Args:
            root_dir (str): NRE 포맷 데이터 루트 디렉토리
            cameras (list): 로드할 카메라 목록 (None이면 pose에 있는 모든 카메라)
            transform (callable): 이미지 변환 함수
            use_inpainted (bool): True면 final_inpainted/ 이미지 사용
            use_depth (bool): True면 depth_maps/ 도 함께 반환
        """
        self.root_dir = root_dir
        self.cameras = cameras
        self.transform = transform
        self.use_inpainted = use_inpainted
        self.use_depth = use_depth

        # 디렉토리 경로
        self.images_dir = os.path.join(root_dir, 'images')
        self.poses_dir = os.path.join(root_dir, 'poses')
        self.objects_dir = os.path.join(root_dir, 'objects')
        self.depth_maps_dir = os.path.join(root_dir, 'depth_maps')
        self.inpainted_dir = os.path.join(root_dir, 'final_inpainted')

        # Pose 파일 기준으로 프레임 목록 수집
        self.frame_ids = []
        self.frame_data_cache = {}

        pose_files = sorted(glob.glob(os.path.join(self.poses_dir, '*.json')))
        for pf in pose_files:
            frame_id = os.path.splitext(os.path.basename(pf))[0]
            self.frame_ids.append(frame_id)

        if len(self.frame_ids) == 0:
            print(f"Warning: No pose files found in {self.poses_dir}")

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]

        # Pose/Camera metadata 로드
        pose_path = os.path.join(self.poses_dir, f"{frame_id}.json")
        with open(pose_path, 'r') as f:
            frame_meta = json.load(f)

        cam_names = self.cameras if self.cameras else list(frame_meta['cameras'].keys())

        # 이미지 로드
        images = {}
        for cam in cam_names:
            if cam not in frame_meta['cameras']:
                continue

            if self.use_inpainted:
                # Inpainted 이미지 우선
                img_path = os.path.join(self.inpainted_dir, f"{frame_id}_{cam}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.inpainted_dir, f"{frame_id}_{cam}.png")
            else:
                img_path = os.path.join(self.images_dir, f"{frame_id}_{cam}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.images_dir, f"{frame_id}_{cam}.png")

            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            images[cam] = img

        # 카메라 Calibration
        calibs = {}
        for cam in cam_names:
            if cam not in frame_meta['cameras']:
                continue
            cam_data = frame_meta['cameras'][cam]
            intrinsics = cam_data['intrinsics']
            pose_flat = cam_data['pose']

            calibs[cam] = {
                'intrinsics': torch.tensor(intrinsics, dtype=torch.float32),
                'pose': torch.tensor(pose_flat, dtype=torch.float32).reshape(4, 4),
                'width': cam_data['width'],
                'height': cam_data['height'],
            }

        # Depth maps (선택)
        depths = {}
        if self.use_depth:
            for cam in cam_names:
                depth_path = os.path.join(
                    self.depth_maps_dir, f"{frame_id}_{cam}.png"
                )
                if os.path.exists(depth_path):
                    import cv2
                    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                    if depth is not None:
                        depths[cam] = torch.tensor(
                            depth.astype(np.float32) / 1000.0,  # mm → m
                            dtype=torch.float32
                        )

        result = {
            'images': images,
            'calibs': calibs,
            'meta': {
                'frame_id': frame_id,
                'timestamp': frame_meta.get('timestamp', 0.0),
            }
        }

        if self.use_depth:
            result['depths'] = depths

        return result


# Example Usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <data_root> [--nre]")
        print("  --nre: Use NRE format (default: legacy Waymo extracted format)")
        sys.exit(1)

    dataset_root = sys.argv[1]
    use_nre = "--nre" in sys.argv

    if use_nre:
        dataset = NREDataset(dataset_root)
        print(f"[NREDataset] Loaded {len(dataset)} frames")
        if len(dataset) > 0:
            sample = dataset[0]
            print("  Images:", list(sample['images'].keys()))
            print("  Calibs:", list(sample['calibs'].keys()))
            print("  Frame ID:", sample['meta']['frame_id'])
    else:
        if os.path.exists(dataset_root):
            dataset = WaymoExtractedDataset(dataset_root)
            print(f"[WaymoExtractedDataset] Loaded {len(dataset)} samples")
            if len(dataset) > 0:
                sample = dataset[0]
                print("  Images:", list(sample['images'].keys()))
                print("  Pose shape:", sample['pose'].shape)
        else:
            print(f"Root directory {dataset_root} does not exist.")
