"""
mmcv/mmdet import를 호환성 레이어로 변경하는 스크립트
"""
import os
import re

# 변경할 파일 목록과 변경 내용
replacements = [
    # (파일 경로, (old_pattern, new_pattern))
    # force_fp32, auto_fp16
    ("bevformer/modules/transformer.py", 
     ("from mmcv.runner import force_fp32, auto_fp16", 
      "from ...utils.mmcv_compat import force_fp32, auto_fp16")),
    ("bevformer/modules/spatial_cross_attention.py",
     ("from mmcv.runner import force_fp32, auto_fp16",
      "from ...utils.mmcv_compat import force_fp32, auto_fp16")),
    ("bevformer/modules/encoder.py",
     ("from mmcv.runner import force_fp32, auto_fp16",
      "from ...utils.mmcv_compat import force_fp32, auto_fp16")),
    ("bevformer/detectors/bevformer.py",
     ("from mmcv.runner import force_fp32, auto_fp16",
      "from ...utils.mmcv_compat import force_fp32, auto_fp16")),
    ("bevformer/detectors/bevformer_fp16.py",
     ("from mmcv.runner import force_fp32, auto_fp16",
      "from ...utils.mmcv_compat import force_fp32, auto_fp16")),
    ("bevformer/dense_heads/bevformer_head.py",
     ("from mmcv.runner import force_fp32, auto_fp16",
      "from ...utils.mmcv_compat import force_fp32, auto_fp16")),
    ("bevformer/dense_heads/bev_head.py",
     ("from mmcv.runner import force_fp32, auto_fp16",
      "from ...utils.mmcv_compat import force_fp32, auto_fp16")),
]

base_path = "Spatial_AI_Project/Ref_AI_project/mmdet3d_plugin"

for rel_path, (old, new) in replacements:
    file_path = os.path.join(base_path, rel_path)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old in content:
            content = content.replace(old, new)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
        else:
            print(f"Skipped (pattern not found): {file_path}")
    else:
        print(f"File not found: {file_path}")

print("Done!")

