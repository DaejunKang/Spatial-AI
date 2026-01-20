"""
Waymo E2E Dataset "Aggressive" Extractor
- Priority: tf.train.SequenceExample > tf.train.Example
- Detection: Checks JPEG Magic Header (FF D8) + Size > 10KB (Ignores key names if needed)
- Output: Images, Empty Masks, JSON Stats
"""

import os
import tensorflow as tf
import numpy as np
import cv2
import argparse
import json
from tqdm import tqdm
from glob import glob

# TF 로그 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def infer_camera_name(key):
    """키 이름에서 카메라 이름을 추론, 실패 시 키 자체를 사용"""
    key_upper = key.upper()
    if 'FRONT_LEFT' in key_upper: return 'FRONT_LEFT'
    if 'FRONT_RIGHT' in key_upper: return 'FRONT_RIGHT'
    if 'SIDE_LEFT' in key_upper: return 'SIDE_LEFT'
    if 'SIDE_RIGHT' in key_upper: return 'SIDE_RIGHT'
    if 'FRONT' in key_upper: return 'FRONT'
    
    # 5개 카메라 순서 추정 (Waymo 표준 인덱스)
    if key == '0' or key == 'image_0': return 'FRONT'
    if key == '1' or key == 'image_1': return 'FRONT_LEFT'
    if key == '2' or key == 'image_2': return 'FRONT_RIGHT'
    if key == '3' or key == 'image_3': return 'SIDE_LEFT'
    if key == '4' or key == 'image_4': return 'SIDE_RIGHT'
    
    # 추론 불가 시 특수문자 제거 후 반환
    return key.replace('/', '_').replace('.', '_')

def is_image_data(byte_data):
    """
    데이터가 이미지인지 확인하는 강력한 검사
    1. 크기가 5KB 이상인가?
    2. JPEG 헤더(FF D8)로 시작하는가?
    """
    if len(byte_data) < 5120: # 5KB 미만은 이미지 아님
        return False
    
    # JPEG Header check (Magic Number)
    if byte_data.startswith(b'\xff\xd8'):
        return True
    
    # PNG Header check
    if byte_data.startswith(b'\x89PNG'):
        return True
        
    return False

def extract_universal(tfrecord_path, output_dir):
    print(f"🚀 Processing: {os.path.basename(tfrecord_path)}")
    ensure_dir(output_dir)
    
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    
    # 통계용 변수
    stats = {}
    total_images_saved = 0
    records_processed = 0

    for i, raw_record in enumerate(tqdm(dataset, desc="Scanning Records")):
        records_processed += 1
        record_bytes = raw_record.numpy()
        
        # =========================================================
        # STRATEGY 1: SequenceExample (Video/Time-series) - Priority
        # =========================================================
        try:
            seq_ex = tf.train.SequenceExample()
            seq_ex.ParseFromString(record_bytes)
            
            # --- 1. Scenario ID 추출 ---
            context = seq_ex.context.feature
            scenario_id = f"segment_{i}"
            
            # ID 키 후보군 검색
            for id_key in ['scenario/id', 'scenario_id', 'context.name', 'segment_id']:
                if id_key in context:
                    val = context[id_key].bytes_list.value
                    if val:
                        scenario_id = val[0].decode('utf-8')
                        break
            
            # --- 2. 이미지 키 자동 감지 (feature_lists 내부) ---
            feature_lists = seq_ex.feature_lists.feature_list
            image_keys = []
            
            # 모든 키를 뒤져서 "이미지스러운" 데이터를 찾음
            for key, feat_list in feature_lists.items():
                if len(feat_list.feature) > 0:
                    first_feat = feat_list.feature[0]
                    if first_feat.bytes_list.value:
                        first_data = first_feat.bytes_list.value[0]
                        # [핵심] 키 이름 무관하게 데이터 내용으로 판별
                        if is_image_data(first_data):
                            image_keys.append(key)

            # 이미지가 발견되면 Sequence 추출 모드 실행
            if image_keys:
                # 통계 초기화
                if scenario_id not in stats:
                    stats[scenario_id] = {'count': 0, 'timestamps': []}
                
                # Timestamp 찾기 (없으면 인덱스 사용)
                timestamps = []
                # Feature list 중 길이가 가장 긴 것을 기준으로 길이 산정
                seq_len = max([len(feature_lists[k].feature) for k in image_keys])
                
                # 타임스탬프 리스트가 따로 있는지 확인
                ts_key = None
                for k in feature_lists.keys():
                    if 'timestamp' in k and len(feature_lists[k].feature) == seq_len:
                        ts_key = k
                        break
                
                # --- 프레임 순회 및 저장 ---
                for t in range(seq_len):
                    # Timestamp 결정
                    curr_ts = t
                    if ts_key:
                        ts_val = feature_lists[ts_key].feature[t].int64_list.value
                        if ts_val: curr_ts = ts_val[0]
                    
                    # 통계 저장
                    stats[scenario_id]['timestamps'].append(curr_ts)
                    stats[scenario_id]['count'] += 1
                    
                    for key in image_keys:
                        f_list = feature_lists[key].feature
                        if t >= len(f_list): continue
                        if not f_list[t].bytes_list.value: continue
                        
                        img_bytes = f_list[t].bytes_list.value[0]
                        cam_name = infer_camera_name(key)
                        
                        # 이미지 디코딩
                        np_arr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        
                        if img is not None:
                            # 경로: output/scenario_id/images/CAM/timestamp.png
                            img_dir = os.path.join(output_dir, scenario_id, 'images', cam_name)
                            mask_dir = os.path.join(output_dir, scenario_id, 'masks', cam_name)
                            ensure_dir(img_dir)
                            ensure_dir(mask_dir)
                            
                            fname = f"{curr_ts}.png"
                            
                            # 이미지 저장
                            cv2.imwrite(os.path.join(img_dir, fname), img)
                            
                            # 마스크 저장 (Black)
                            h, w = img.shape[:2]
                            mask = np.zeros((h, w), dtype=np.uint8)
                            cv2.imwrite(os.path.join(mask_dir, fname), mask)
                            
                            total_images_saved += 1
                
                continue # SequenceExample 처리 성공 시 다음 레코드로

        except Exception as e:
            # SequenceExample 파싱 자체가 실패하면 Example로 넘어감
            pass

        # =========================================================
        # STRATEGY 2: Example (Snapshot/Frame) - Fallback
        # =========================================================
        try:
            ex = tf.train.Example()
            ex.ParseFromString(record_bytes)
            features = ex.features.feature
            
            # ID 추출
            scenario_id = f"segment_{i//200}"
            if 'scenario/id' in features:
                scenario_id = features['scenario/id'].bytes_list.value[0].decode('utf-8')
            
            # Timestamp 추출
            curr_ts = i
            if 'timestamp_micros' in features:
                curr_ts = features['timestamp_micros'].int64_list.value[0]

            # 이미지 키 찾기 (내용 기반)
            image_keys = []
            for key, feat in features.items():
                if feat.bytes_list.value:
                    if is_image_data(feat.bytes_list.value[0]):
                        image_keys.append(key)
            
            if image_keys:
                if scenario_id not in stats:
                    stats[scenario_id] = {'count': 0, 'timestamps': []}
                stats[scenario_id]['count'] += 1
                stats[scenario_id]['timestamps'].append(curr_ts)

                for key in image_keys:
                    img_bytes = features[key].bytes_list.value[0]
                    cam_name = infer_camera_name(key)
                    
                    np_arr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    
                    if img is not None:
                        img_dir = os.path.join(output_dir, scenario_id, 'images', cam_name)
                        mask_dir = os.path.join(output_dir, scenario_id, 'masks', cam_name)
                        ensure_dir(img_dir)
                        ensure_dir(mask_dir)
                        
                        fname = f"{curr_ts}.png"
                        cv2.imwrite(os.path.join(img_dir, fname), img)
                        
                        h, w = img.shape[:2]
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.imwrite(os.path.join(mask_dir, fname), mask)
                        
                        total_images_saved += 1

        except Exception:
            pass

    # --- 최종 결과 리포트 ---
    print("\n" + "="*50)
    print("📊 Extraction Statistics Report")
    print("="*50)
    
    if total_images_saved == 0:
        print("❌ CRITICAL FAILURE: No images were extracted.")
        print("   Possible reasons:")
        print("   1. File contains NO images (LiDAR only or Motion dataset).")
        print("   2. Images are compressed in a format OpenCV cannot read (unlikely).")
        return

    print(f"✅ Total Images Saved: {total_images_saved}")
    print(f"✅ Total Scenarios Found: {len(stats)}")
    print(f"✅ Total Records Processed: {records_processed}")
    
    print("\n[Scenario Detail]")
    for sid, data in list(stats.items())[:5]: # 처음 5개만 출력
        ts_list = sorted(data['timestamps'])
        start_ts = ts_list[0] if ts_list else "N/A"
        end_ts = ts_list[-1] if ts_list else "N/A"
        print(f"  - ID: {sid}")
        print(f"    Frames: {data['count']}")
        print(f"    Timestamp Range: {start_ts} ~ {end_ts}")
    
    if len(stats) > 5:
        print(f"  ... and {len(stats) - 5} more scenarios")
    
    # 통계를 JSON으로 저장
    stats_file = os.path.join(output_dir, 'extraction_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n💾 Statistics saved to: {stats_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Waymo E2E Dataset "Aggressive" Extractor - Extracts images from TFRecords without requiring waymo_open_dataset package'
    )
    parser.add_argument('input_path', type=str, help='Path to .tfrecord file or directory containing .tfrecord files')
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
        # 각 파일마다 별도 디렉토리 생성
        segment_name = os.path.splitext(os.path.basename(tf_file))[0]
        segment_out_dir = os.path.join(output_dir, segment_name)
        
        extract_universal(tf_file, segment_out_dir)

if __name__ == '__main__':
    main()
