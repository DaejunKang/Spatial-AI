import os
import subprocess
import argparse
import sys

def check_gsutil():
    """Check if gsutil is installed."""
    try:
        subprocess.check_call(['gsutil', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def download_waymo_segment(output_dir, split='training', segment_limit=None):
    """
    Download Waymo Open Dataset segments using gsutil.
    
    Args:
        output_dir: Directory to save .tfrecord files.
        split: 'training' or 'validation'.
        segment_limit: Number of segments to download (optional).
    """
    
    # Base GCS bucket URL (v1.2 is commonly used, check for latest version if needed)
    # Note: Access requires authentication with Google Cloud account registered for Waymo Dataset
    base_url = f"gs://waymo_open_dataset_v_1_2_0/{split}/"
    
    print(f"Checking access to {base_url}...")
    try:
        # List files
        cmd = ['gsutil', 'ls', base_url]
        result = subprocess.check_output(cmd, text=True)
        files = result.strip().split('\n')
        
        # Filter for .tfrecord files
        tfrecord_files = [f for f in files if f.endswith('.tfrecord')]
        print(f"Found {len(tfrecord_files)} files in {split} set.")
        
        if segment_limit:
            tfrecord_files = tfrecord_files[:segment_limit]
            print(f"Downloading first {segment_limit} files...")
        else:
            print("Downloading ALL files...")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, file_url in enumerate(tfrecord_files):
            print(f"[{i+1}/{len(tfrecord_files)}] Downloading {os.path.basename(file_url)} ...")
            subprocess.check_call(['gsutil', '-m', 'cp', file_url, output_dir])
            
        print("Download complete.")
        
    except subprocess.CalledProcessError as e:
        print("\n[Error] Failed to access/download files.")
        print("Please ensure you have:")
        print("1. Installed Google Cloud SDK (gsutil).")
        print("2. Run 'gcloud auth login' and authenticated with the email registered for Waymo Open Dataset.")
        print(f"Details: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download Waymo Open Dataset segments.")
    parser.add_argument('output_dir', type=str, help="Directory to save downloaded .tfrecord files")
    parser.add_argument('--split', type=str, default='training', choices=['training', 'validation'], help="Dataset split (training/validation)")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of segments to download (e.g. 1 for testing)")
    
    args = parser.parse_args()
    
    if not check_gsutil():
        print("Error: 'gsutil' command not found.")
        print("Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install")
        sys.exit(1)
        
    download_waymo_segment(args.output_dir, args.split, args.limit)

if __name__ == "__main__":
    main()
