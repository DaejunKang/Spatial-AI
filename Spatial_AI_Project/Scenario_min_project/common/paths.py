"""visionary-nvidia unified 데이터셋 경로 (clip 단위 중첩 레이아웃).

레이아웃: unified_dataset/nvidia/<clip>/{sensor,labels,anno}
  sensor/<cam>/<clip>.<cam>.{mp4,timestamps.parquet}
  labels/egomotion/<clip>.egomotion.parquet
  anno/obj3d/v1.8.0/{frames,tracks}.parquet   (3DOD, lidar frame x=전방 y=좌우)
  anno/map/v1.1.0/{frames,scene}.parquet      (lane/OpenLane)
"""

from pathlib import Path

ROOT = "/katech/datasets/visionary-nvidia"
BASE = f"{ROOT}/unified_dataset/nvidia"
CAM = "camera_front_wide_120fov"
OBJ3D_VER = "v1.8.0"
MAP_VER = "v1.1.0"


def clip_dir(cid: str) -> Path:
    return Path(BASE) / cid


def video_path(cid: str, cam: str = CAM) -> Path:
    return clip_dir(cid) / "sensor" / cam / f"{cid}.{cam}.mp4"


def camera_ts(cid: str, cam: str = CAM) -> Path:
    return clip_dir(cid) / "sensor" / cam / f"{cid}.{cam}.timestamps.parquet"


def egomotion_path(cid: str) -> Path:
    return clip_dir(cid) / "labels" / "egomotion" / f"{cid}.egomotion.parquet"


def obj3d_frames(cid: str) -> Path:
    return clip_dir(cid) / "anno" / "obj3d" / OBJ3D_VER / "frames.parquet"


def map_frames(cid: str) -> Path:
    return clip_dir(cid) / "anno" / "map" / MAP_VER / "frames.parquet"


def clip_ids(split: str = "diverse_set-test") -> list[str]:
    split = split.replace(".txt", "")
    f = Path(ROOT) / "curation" / f"{split}.txt"
    return [x.strip() for x in f.read_text().splitlines() if x.strip()]


# obj3d class::subclass → 어휘 object_type
OBJ3D_CLASS_MAP = {
    "vehicle::car": "vehicle", "vehicle::truck": "large_vehicle",
    "vehicle::bus": "large_vehicle", "vehicle::trailer": "large_vehicle",
    "vehicle::construction_vehicle": "large_vehicle",
    "pedestrian::pedestrian": "pedestrian",
    "twowheeler::bicycle": "bicycle_micromobility",
    "twowheeler::twowheeler": "bicycle_micromobility",
    "twowheeler::motorcycle": "motorcycle",
    "object::traffic_cone": "other", "object::bollard": "other",
}
