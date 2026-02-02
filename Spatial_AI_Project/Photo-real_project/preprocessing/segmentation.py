import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import cv2

class SemanticSegmentor:
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024", device='cuda'):
        """
        Initialize SegFormer model.
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Loading SegFormer model ({model_name}) on {self.device}...")
        
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Cityscapes classes to mask (Dynamic objects)
        # 11: person, 12: rider, 13: car, 14: truck, 15: bus, 16: train, 17: motorcycle, 18: bicycle
        self.dynamic_classes = [11, 12, 13, 14, 15, 16, 17, 18]

    def process_image(self, image_path):
        """
        Generate binary mask for dynamic objects.
        Returns:
            mask (np.array): 255 for static (background), 0 for dynamic (foreground)
        """
        image = Image.open(image_path).convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Upsample logits to original image size
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1], # (H, W)
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0] # (H, W)
        pred_seg = pred_seg.cpu().numpy()
        
        # Create Mask: 0 for dynamic, 255 for static
        mask = np.ones_like(pred_seg, dtype=np.uint8) * 255
        
        for cls_id in self.dynamic_classes:
            mask[pred_seg == cls_id] = 0
            
        return mask

if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        seg = SemanticSegmentor()
        mask = seg.process_image(sys.argv[1])
        cv2.imwrite("test_mask.png", mask)
        print("Mask saved to test_mask.png")
