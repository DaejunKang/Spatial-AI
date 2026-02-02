import cv2
import numpy as np
import torch
import os
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline

class Inpainter:
    def __init__(self, use_generative=True, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.use_generative = use_generative
        
        if self.use_generative:
            print("Loading Stable Diffusion Inpaint Pipeline...")
            # Use a lightweight version or standard SD 2 inpainting
            model_id = "stabilityai/stable-diffusion-2-inpainting"
            try:
                self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                )
                self.pipe = self.pipe.to(self.device)
                self.pipe.enable_attention_slicing() # Optimize memory
            except Exception as e:
                print(f"Warning: Failed to load Generative Model ({e}). Falling back to simple inpainting.")
                self.use_generative = False

    def temporal_warp_fill(self, current_img, current_mask, neighbor_imgs, neighbor_poses, current_pose, K):
        """
        1st Stage: Fill missing regions using temporal neighbors (Warping).
        
        Args:
            current_img: (H, W, 3)
            current_mask: (H, W) 0=missing, 255=valid
            neighbor_imgs: list of images
            neighbor_poses: list of 4x4 matrices (T_global_camera for neighbors)
            current_pose: 4x4 matrix (T_global_camera for current)
            K: 3x3 Intrinsic matrix
        """
        # Simplified Homography warping assuming planar ground or Depth estimation.
        # Without depth, we can only do rotation-based warping (infinite depth) or Homography.
        # For strict multi-view filling, we need depth maps. 
        # Here, we implement a simple placeholder for Structure-based filling or just skip if no depth.
        # As a robust baseline, we will skip complex warping without depth and rely on Generative Inpainting.
        # If the user wants specific reprojection, we need Depth priors (e.g. from Monodepth).
        
        # For this module, we will implement a basic Navier-Stokes based inpainting for small holes
        # as a "1st stage" fast fill before Generative model.
        
        # Inpaint small holes with OpenCV (Telea/NS)
        # Dilate mask slightly to cover edges
        kernel = np.ones((5,5), np.uint8)
        mask_dilated = cv2.dilate((255 - current_mask).astype(np.uint8), kernel, iterations=2)
        
        # OpenCV inpainting expects mask where non-zero pixels are the area to inpaint.
        # current_mask: 0 is dynamic (to fill). So use (255 - current_mask).
        filled_img = cv2.inpaint(current_img, mask_dilated, 3, cv2.INPAINT_TELEA)
        
        return filled_img, mask_dilated

    def generative_inpaint(self, image, mask):
        """
        2nd Stage: Generative Inpainting using Stable Diffusion.
        Args:
            image: (H, W, 3) numpy array (BGR)
            mask: (H, W) numpy array (0=valid, 255=hole to fill) - Note: Logic inverted for SD
        """
        if not self.use_generative:
            return image

        # Convert to PIL
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask) # Mask should be white (255) for hole
        
        # Resize for SD (must be multiple of 8, usually 512x512 recommended for speed)
        orig_w, orig_h = image_pil.size
        process_w, process_h = 512, 512
        
        image_resized = image_pil.resize((process_w, process_h))
        mask_resized = mask_pil.resize((process_w, process_h))
        
        prompt = "background, empty road, realistic texture, high quality"
        negative_prompt = "car, person, vehicle, pedestrian, artifacts, blur"
        
        with torch.no_grad():
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_resized,
                mask_image=mask_resized,
                height=process_h,
                width=process_w,
                num_inference_steps=20
            ).images[0]
            
        # Resize back
        output = output.resize((orig_w, orig_h))
        output_np = cv2.cvtColor(np.array(output), cv2.COLOR_RGB2BGR)
        
        # Blend original valid pixels with inpainted
        # mask is 255 where hole exists.
        # result = original * (1-mask) + inpainted * mask
        mask_norm = mask[:, :, np.newaxis] / 255.0
        result = image.astype(np.float32) * (1 - mask_norm) + output_np.astype(np.float32) * mask_norm
        
        return result.astype(np.uint8)

    def process_frame(self, image_path, mask_path, output_path):
        """
        Pipeline: Load -> Simple Fill -> Generative Fill -> Save
        """
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            return

        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # 0=dynamic, 255=static
        
        # Invert mask for processing: 255 = Hole to fill
        hole_mask = 255 - mask
        
        # 1. Simple Inpainting (Fast fill for small regions/edges)
        filled_1, _ = self.temporal_warp_fill(img, mask, None, None, None, None)
        
        # 2. Generative Inpainting (Large regions)
        # Only if hole area is significant
        if np.sum(hole_mask) > 100: # threshold
            final_img = self.generative_inpaint(filled_1, hole_mask)
        else:
            final_img = filled_1
            
        cv2.imwrite(output_path, final_img)

if __name__ == "__main__":
    # Test
    # python inpainting.py image.png mask.png output.png
    pass
