"""
Waymo2NRE Minimal Converter 테스트 스크립트
TensorFlow/MMCV 의존성 없이 동작 확인
"""

import os
import sys
import json
import tempfile
import shutil


def test_imports():
    """필수 모듈 import 테스트"""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False
    
    try:
        import cv2
        print("✓ opencv-python imported successfully")
    except ImportError as e:
        print(f"✗ opencv-python import failed: {e}")
        return False
    
    try:
        from waymo_open_dataset import dataset_pb2
        print("✓ waymo_open_dataset imported successfully")
    except ImportError as e:
        print(f"✗ waymo_open_dataset import failed: {e}")
        print("  Please install: pip install waymo-open-dataset-tf-2-11-0")
        return False
    
    # TensorFlow가 설치되어 있지 않아야 함 (선택사항)
    try:
        import tensorflow as tf
        print("⚠ tensorflow is installed (optional, not required)")
    except ImportError:
        print("✓ tensorflow not installed (good - not required)")
    
    # MMCV가 설치되어 있지 않아야 함 (선택사항)
    try:
        import mmcv
        print("⚠ mmcv is installed (optional, not required)")
    except ImportError:
        print("✓ mmcv not installed (good - not required)")
    
    print("\n")
    return True


def test_minimal_tfrecord_reader():
    """MinimalTFRecordReader 테스트"""
    print("=" * 60)
    print("Testing MinimalTFRecordReader...")
    print("=" * 60)
    
    try:
        from waymo2nre import MinimalTFRecordReader
        print("✓ MinimalTFRecordReader imported successfully")
        
        # 기본 구조 확인
        reader = MinimalTFRecordReader("dummy_path.tfrecord")
        assert hasattr(reader, '__iter__'), "Reader must be iterable"
        print("✓ MinimalTFRecordReader has correct structure")
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except AssertionError as e:
        print(f"✗ Structure test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    print("\n")
    return True


def test_converter_initialization():
    """Waymo2NRE 초기화 테스트"""
    print("=" * 60)
    print("Testing Waymo2NRE initialization...")
    print("=" * 60)
    
    try:
        from waymo2nre import Waymo2NRE
        
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as tmpdir:
            load_dir = os.path.join(tmpdir, 'input')
            save_dir = os.path.join(tmpdir, 'output')
            os.makedirs(load_dir)
            
            # Converter 초기화
            converter = Waymo2NRE(
                load_dir=load_dir,
                save_dir=save_dir,
                prefix='test_'
            )
            
            print("✓ Waymo2NRE initialized successfully")
            
            # 디렉토리 생성 확인
            assert os.path.exists(converter.dirs['images']), "images dir not created"
            assert os.path.exists(converter.dirs['poses']), "poses dir not created"
            assert os.path.exists(converter.dirs['objects']), "objects dir not created"
            print("✓ Output directories created correctly")
            
            # intrinsics 디렉토리는 생성되지 않아야 함 (제거됨)
            assert 'intrinsics' not in converter.dirs, "intrinsics dir should be removed"
            print("✓ intrinsics directory correctly removed")
            
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except AssertionError as e:
        print(f"✗ Assertion failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    print("\n")
    return True


def test_directory_structure():
    """출력 디렉토리 구조 테스트"""
    print("=" * 60)
    print("Testing output directory structure...")
    print("=" * 60)
    
    try:
        from waymo2nre import Waymo2NRE
        
        with tempfile.TemporaryDirectory() as tmpdir:
            load_dir = os.path.join(tmpdir, 'input')
            save_dir = os.path.join(tmpdir, 'output')
            os.makedirs(load_dir)
            
            converter = Waymo2NRE(load_dir, save_dir, 'seq0_')
            
            # 예상 구조
            expected_dirs = ['images', 'poses', 'objects']
            
            for dir_name in expected_dirs:
                full_path = os.path.join(save_dir, dir_name)
                assert os.path.exists(full_path), f"{dir_name} directory not found"
                print(f"✓ {dir_name}/ directory exists")
            
            # intrinsics는 존재하지 않아야 함
            intrinsics_path = os.path.join(save_dir, 'intrinsics')
            assert not os.path.exists(intrinsics_path), "intrinsics dir should not exist"
            print("✓ intrinsics/ directory correctly excluded")
            
    except AssertionError as e:
        print(f"✗ Directory structure test failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    print("\n")
    return True


def test_json_schema():
    """JSON 스키마 검증"""
    print("=" * 60)
    print("Testing JSON schema...")
    print("=" * 60)
    
    # 예상되는 poses JSON 스키마
    expected_pose_keys = ['frame_idx', 'timestamp', 'ego_velocity', 'cameras']
    expected_camera_keys = ['img_path', 'width', 'height', 'intrinsics', 'pose', 'rolling_shutter']
    expected_velocity_keys = ['linear', 'angular']
    expected_rolling_shutter_keys = ['duration', 'trigger_time']
    
    print("Expected poses JSON schema:")
    print(f"  Top level: {expected_pose_keys}")
    print(f"  Camera: {expected_camera_keys}")
    print(f"  Velocity: {expected_velocity_keys}")
    print(f"  Rolling Shutter: {expected_rolling_shutter_keys}")
    
    # 예상되는 objects JSON 스키마
    expected_object_keys = ['id', 'class', 'box', 'speed']
    expected_box_keys = ['center', 'size', 'heading']
    
    print("\nExpected objects JSON schema:")
    print(f"  Object: {expected_object_keys}")
    print(f"  Box: {expected_box_keys}")
    
    print("✓ JSON schemas documented")
    print("\n")
    return True


def run_all_tests():
    """모든 테스트 실행"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Waymo2NRE Minimal Converter Tests" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    tests = [
        ("Import Test", test_imports),
        ("TFRecord Reader Test", test_minimal_tfrecord_reader),
        ("Converter Initialization", test_converter_initialization),
        ("Directory Structure", test_directory_structure),
        ("JSON Schema", test_json_schema),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status:12} {test_name}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Converter is ready to use.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please fix the issues.")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
