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
                
            # Iterate through frames available in poses
            # Assuming images exist for these frames (we can verify)
            
            sorted_frame_ids = sorted(vehicle_poses.keys())
            
            for frame_id in sorted_frame_ids:
                # Verify all requested cameras exist for this frame
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
                # Default to tensor if no transform provided
                img = transforms.ToTensor()(img)
            images[cam] = img
            
        # Global Pose (Vehicle to Global) - 4x4 Matrix
        pose = torch.tensor(sample_info['pose'], dtype=torch.float32)
        
        # Calibration
        # Filter only requested cameras and convert to tensor
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

# Example Usage
if __name__ == "__main__":
    # Dummy test if directory exists
    dataset_root = "/workspace/waymo_extracted_data"
    if os.path.exists(dataset_root):
        dataset = WaymoExtractedDataset(dataset_root)
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print("Sample keys:", sample.keys())
            print("Images keys:", sample['images'].keys())
            print("Pose shape:", sample['pose'].shape)
            print("Calib keys:", sample['calib'].keys())
            print("Front Cam Intrinsic:", sample['calib']['FRONT']['intrinsic'])
    else:
        print(f"Root directory {dataset_root} does not exist. Please run extract_waymo_data.py first.")
