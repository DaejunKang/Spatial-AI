import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import math
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from glob import glob

# --- Placeholder for diff-gaussian-rasterization ---
# In a real environment, you must install:
# pip install diff-gaussian-rasterization
# pip install simple-knn
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
except ImportError:
    print("Warning: 'diff_gaussian_rasterization' not found. Training will strictly fail, but structure is provided.")
    GaussianRasterizationSettings = None
    GaussianRasterizer = None

# =================================================================================
# 1. Data Loader Module
# =================================================================================

class WaymoReconLoader(Dataset):
    def __init__(self, root_dir, cameras=None, use_inpainted=True):
        self.root_dir = root_dir
        self.use_inpainted = use_inpainted
        self.cameras = cameras if cameras else ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
        self.samples = []
        self._load_dataset()

    def _load_dataset(self):
        # Determine image source folder
        img_folder_name = 'images_inpainted' if self.use_inpainted and os.path.exists(os.path.join(self.root_dir, 'images_inpainted')) else 'images'
        print(f"Loading images from: {img_folder_name}")
        
        # Load Poses
        poses_path = os.path.join(self.root_dir, 'poses', 'vehicle_poses.json')
        calib_path = os.path.join(self.root_dir, 'calibration', 'intrinsics_extrinsics.json')
        
        with open(poses_path, 'r') as f:
            poses = json.load(f)
        with open(calib_path, 'r') as f:
            calib = json.load(f)
            
        frame_ids = sorted(poses.keys())
        
        for fid in frame_ids:
            # Vehicle Pose (Global)
            T_g_v = np.array(poses[fid])
            
            for cam in self.cameras:
                if cam not in calib: continue
                
                # Camera Paths
                img_path = os.path.join(self.root_dir, img_folder_name, cam, f"{fid}.png")
                # Mask Path (always in 'masks')
                mask_path = os.path.join(self.root_dir, 'masks', cam, f"{fid}.png")
                
                if not os.path.exists(img_path): continue
                
                # Extrinsic: Vehicle -> Camera
                T_v_c = np.array(calib[cam]['extrinsic'])
                
                # Global -> Camera (T_cw) for 3DGS
                # T_global_camera = T_global_vehicle * T_vehicle_camera
                T_g_c = T_g_v @ T_v_c
                # 3DGS expects World to Camera (View Matrix) -> Inverse of Camera to World
                # But here T_g_c IS Camera-to-World (Camera in Global).
                # So ViewMatrix (World-to-Camera) is inv(T_g_c).
                T_c_g = np.linalg.inv(T_g_c)
                
                # Intrinsics
                intr = calib[cam]['intrinsic'] # [fx, fy, cx, cy, ...]
                width = calib[cam]['width']
                height = calib[cam]['height']
                
                self.samples.append({
                    'image_path': img_path,
                    'mask_path': mask_path if os.path.exists(mask_path) else None,
                    'R': T_c_g[:3, :3],
                    'T': T_c_g[:3, 3],
                    'fx': intr[0], 'fy': intr[1], 'cx': intr[2], 'cy': intr[3],
                    'width': width, 'height': height
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # This dataset is designed to load EVERYTHING into CPU RAM for 3DGS optimization loop
        # But for 'DataLoader' pattern, we return single item.
        # In standard 3DGS, we usually convert these to a "Camera" class list.
        s = self.samples[idx]
        
        img = cv2.imread(s['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 # (3, H, W)
        
        # Mask (if exists)
        if s['mask_path']:
            mask = cv2.imread(s['mask_path'], cv2.IMREAD_GRAYSCALE)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0 # (1, H, W)
        else:
            mask_tensor = torch.ones((1, s['height'], s['width'])).float()
            
        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'R': torch.tensor(s['R'], dtype=torch.float32),
            'T': torch.tensor(s['T'], dtype=torch.float32),
            'intr': torch.tensor([s['fx'], s['fy'], s['cx'], s['cy']], dtype=torch.float32),
            'wh': torch.tensor([s['width'], s['height']], dtype=torch.int32)
        }

# =================================================================================
# 2. Gaussian Model Module
# =================================================================================

class GaussianModel:
    def __init__(self, sh_degree=3):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.setup_functions()

    def setup_functions(self):
        # Activation functions
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = lambda x: x # simplified
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = lambda x: torch.log(x/(1-x))
        self.rotation_activation = torch.nn.functional.normalize

    def create_from_pcd(self, pcd_points, pcd_colors):
        # Initialize from Structure-from-Motion points (COLMAP or Waymo LiDAR)
        fused_point_cloud = torch.tensor(np.asarray(pcd_points)).float().cuda()
        fused_color = torch.tensor(np.asarray(pcd_colors)).float().cuda() / 255.0
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print(f"Number of points at init: {fused_point_cloud.shape[0]}")

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

# Simple KNN placeholder (normally compiled CUDA)
def distCUDA2(points):
    # Dummy implementation for script validity without CUDA compilation
    # Returns squared distance to nearest neighbor
    return torch.ones(points.shape[0], device=points.device) * 0.1

# =================================================================================
# 3. USD Exporter Module
# =================================================================================

class USDExporter:
    def __init__(self, output_path):
        self.output_path = output_path

    def export(self, gaussians: GaussianModel):
        """
        Export Gaussian Splats to USD format compatible with Omniverse.
        Using pxr.UsdGeomPoints if available, or creating ASCII USD manually.
        """
        xyz = gaussians.get_xyz.detach().cpu().numpy()
        colors = gaussians._features_dc.detach().cpu().numpy() # [N, 1, 3] usually
        colors = (colors.squeeze() + 0.5).clip(0, 1) # SH dc is not exactly RGB, but close approx
        
        # Omniverse expects Points
        try:
            from pxr import Usd, UsdGeom, Vt, Gf
            
            stage = Usd.Stage.CreateNew(self.output_path)
            xform = UsdGeom.Xform.Define(stage, '/World')
            points = UsdGeom.Points.Define(stage, '/World/Gaussians')
            
            # Set Points
            points.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(xyz))
            
            # Set Colors (displayColor)
            points.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(colors))
            
            # Set Widths (Scale approximation)
            # 3DGS scales are 3D (anisotropic), USD Points are usually isotropic widths.
            # We take mean scale for visualization
            scales = gaussians.get_scaling.detach().cpu().numpy()
            mean_scales = np.mean(scales, axis=1)
            points.CreateWidthsAttr(Vt.FloatArray.FromNumpy(mean_scales))
            
            # Save
            stage.GetRootLayer().Save()
            print(f"Exported USD to {self.output_path}")
            
        except ImportError:
            print("Warning: 'pxr' library not found. Exporting simplified ASCII USD.")
            self._export_ascii(xyz, colors, self.output_path)
            
    def _export_ascii(self, xyz, colors, path):
        # Minimal ASCII USD generator
        header = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1.0
    upAxis = "Y"
)

def Xform "World"
{
    def Points "Gaussians"
    {
        point3f[] points = ["""
        
        with open(path, 'w') as f:
            f.write(header)
            # Write points
            for i in range(len(xyz)):
                f.write(f"({xyz[i][0]},{xyz[i][1]},{xyz[i][2]})")
                if i < len(xyz)-1: f.write(", ")
            
            f.write("]\n")
            
            # Write colors
            f.write("        color3f[] primvars:displayColor = [")
            for i in range(len(colors)):
                f.write(f"({colors[i][0]},{colors[i][1]},{colors[i][2]})")
                if i < len(colors)-1: f.write(", ")
            f.write("]\n")
            
            f.write("    }\n}\n")
        print(f"Exported ASCII USD to {path}")

# =================================================================================
# 4. Main Training Pipeline
# =================================================================================

def train_pipeline(args):
    # 1. Load Data
    loader = WaymoReconLoader(args.input_dir, use_inpainted=args.use_inpainted)
    dataloader = DataLoader(loader, batch_size=1, shuffle=True, num_workers=4)
    
    # 2. Initialize Gaussians
    gaussians = GaussianModel(sh_degree=3)
    
    # Init with random point cloud or from COLMAP points3D.txt (Recommended)
    # For now, create sparse random points for testing if no pcd provided
    print("Initializing Gaussians (Random Mock)...")
    pts = np.random.rand(1000, 3) * 10.0 - 5.0 # Random box
    cols = np.random.rand(1000, 3) * 255.0
    gaussians.create_from_pcd(pts, cols)
    
    # Optimizer
    optimizer = torch.optim.Adam(gaussians.get_xyz, lr=0.00016)
    
    # 3. Training Loop
    iterations = 500 # Demo iterations
    print(f"Starting Training for {iterations} iterations...")
    
    progress_bar = tqdm(range(iterations))
    for iteration in progress_bar:
        try:
            # Get random batch
            batch = next(iter(dataloader))
            
            # Prepare Viewpoint (Mock Camera)
            # In real 3DGS, you construct a Camera object with Projection Matrix
            # and pass to Rasterizer.
            
            if GaussianRasterizer is None:
                # Mock Loop
                loss = torch.tensor(0.0, requires_grad=True)
            else:
                # Real Render Call (Pseudo-code)
                # render_pkg = render(viewpoint_cam, gaussians, background)
                # image = render_pkg["render"]
                # gt_image = batch['image'].cuda()
                # loss = l1_loss(image, gt_image)
                pass
            
            loss.backward()
            
            # Densification & Optimizer Step
            with torch.no_grad():
                # densification_logic(gaussians, iteration)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
        except Exception as e:
            # Handle data loading end
            pass
            
    # 4. Export
    output_usd = os.path.join(args.output_dir, "reconstruction.usd")
    exporter = USDExporter(output_usd)
    exporter.export(gaussians)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--use_inpainted', action='store_true', default=True)
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    train_pipeline(args)
