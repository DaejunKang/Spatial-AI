import os
import re

# Spatial_AI_Project 디렉토리로 이동
os.chdir('Spatial_AI_Project')

# 모든 .py, .md 파일 찾기
files_to_update = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith(('.py', '.md', '.ipynb')):
            filepath = os.path.join(root, file)
            files_to_update.append(filepath)

# 각 파일에서 projects를 Ref_AI_project로 변경
encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'cp949', 'euc-kr']
for filepath in files_to_update:
    content = None
    encoding_used = None
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            encoding_used = enc
            break
        except:
            continue
    
    if content is None:
        print(f'Error: Could not read {filepath} with any encoding')
        continue
    
    try:
        original_content = content
        
        # projects. -> Ref_AI_project.
        content = re.sub(r'\bprojects\.', 'Ref_AI_project.', content)
        # projects/ -> Ref_AI_project/
        content = re.sub(r'\bprojects/', 'Ref_AI_project/', content)
        # 'projects/mmdet3d_plugin/' -> 'Ref_AI_project/mmdet3d_plugin/'
        content = re.sub(r"'projects/", "'Ref_AI_project/", content)
        content = re.sub(r'"projects/', '"Ref_AI_project/', content)
        
        if content != original_content:
            # 원본 인코딩 유지하거나 UTF-8로 저장
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'Updated: {filepath}')
    except Exception as e:
        print(f'Error processing {filepath}: {e}')

print('Done!')

