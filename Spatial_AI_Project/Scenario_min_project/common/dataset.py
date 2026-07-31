"""PhysicalAI-AV-curated 데이터셋 접근 및 프레임 샘플링 헬퍼."""

import base64
from pathlib import Path

from config import (
    CAMERA,
    CLIP_INDEX,
    DATASET_ROOT,
    FRAME_MAX_SIDE,
    JPEG_QUALITY,
    NUM_FRAMES,
    SPLIT_FILE,
)


def load_clip_ids(split_file: str = SPLIT_FILE) -> list[str]:
    """split 파일에서 클립 ID 목록을 읽는다."""
    path = Path(DATASET_ROOT) / "curation" / split_file
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def all_clip_ids() -> list[str]:
    """metadata 인덱스에서 전체 클립 ID(1,966개)를 읽는다. (전체 확장용)"""
    import pyarrow.parquet as pq

    p = Path(DATASET_ROOT) / "metadata" / "clip_index_curated.parquet"
    ids = pq.read_table(p, columns=["clip_id"]).to_pydict()["clip_id"]
    return list(dict.fromkeys(ids))  # 원순서 유지 dedup


def video_path(clip_id: str, camera: str = CAMERA) -> Path:
    """클립 ID 에 대응하는 mp4 경로."""
    return Path(DATASET_ROOT) / "camera" / camera / f"{clip_id}.{camera}.mp4"


def get_clip(index: int = CLIP_INDEX, split_file: str = SPLIT_FILE):
    """split 내 index 번째 클립의 (clip_id, mp4 경로) 를 반환한다."""
    clip_ids = load_clip_ids(split_file)
    clip_id = clip_ids[index]
    return clip_id, video_path(clip_id)


# --- 프레임 샘플링 ----------------------------------------------------------
def _read_uniform(path: Path | str, num_frames: int, fps_hint: float | None = None):
    """전체 클립에서 num_frames 개를 시간축 균등 샘플링해 (frame_index, bgr) 목록 반환."""
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = fps_hint or cap.get(cv2.CAP_PROP_FPS) or 0.0
    if total <= 0:
        cap.release()
        raise RuntimeError(f"프레임 수를 알 수 없음: {path}")

    n = min(num_frames, total)
    indices = np.linspace(0, total - 1, n).round().astype(int)

    out = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            out.append((int(idx), frame))
    cap.release()
    if not out:
        raise RuntimeError(f"프레임 추출 실패: {path}")
    return out, fps


def sample_frames(
    path: Path | str,
    num_frames: int = NUM_FRAMES,
    max_side: int = FRAME_MAX_SIDE,
    jpeg_quality: int = JPEG_QUALITY,
) -> list[bytes]:
    """전 구간 균등 샘플링한 프레임을 개별 JPEG 바이트 목록으로 반환."""
    import cv2

    items, _ = _read_uniform(path, num_frames)
    frames: list[bytes] = []
    for _, frame in items:
        frame = _resize_max_side(frame, max_side)
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        if ok:
            frames.append(buf.tobytes())
    if not frames:
        raise RuntimeError(f"프레임 인코딩 실패: {path}")
    return frames


def build_montage(
    items: list[tuple[str, "object"]],
    cols: int = 0,
    cell_max_side: int = 640,
    jpeg_quality: int = JPEG_QUALITY,
) -> bytes:
    """(라벨, BGR프레임) 목록을 시간순 그리드 몽타주 JPEG 로 합친다.

    각 셀 좌상단에 라벨을 그린다. 서버의 '프롬프트당 이미지 최대 5장' 제약을
    지키면서 전 구간(또는 국소 구간) 커버리지를 하나의 이미지로 담기 위함.
    """
    import cv2
    import math

    if not items:
        raise RuntimeError("몽타주 프레임 없음")
    n = len(items)
    if cols <= 0:
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    h0, w0 = items[0][1].shape[:2]
    scale = cell_max_side / max(h0, w0)
    cw, ch = int(round(w0 * scale)), int(round(h0 * scale))

    canvas = _blank(rows * ch, cols * cw)
    for i, (label, frame) in enumerate(items):
        r, c = divmod(i, cols)
        cell = _letterbox(frame, cw, ch)
        _label(cell, label)
        canvas[r * ch : (r + 1) * ch, c * cw : (c + 1) * cw] = cell

    ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("몽타주 인코딩 실패")
    return buf.tobytes()


def sample_montage(
    path: Path | str,
    num_frames: int = NUM_FRAMES,
    cols: int = 0,
    cell_max_side: int = 640,
    jpeg_quality: int = JPEG_QUALITY,
) -> bytes:
    """전 구간 균등 샘플링한 num_frames 프레임을 시간순 그리드 몽타주 JPEG 로."""
    items, fps = _read_uniform(path, num_frames)
    labeled = [
        (f"#{i} f{fidx}" + (f" {fidx / fps:.1f}s" if fps else ""), frame)
        for i, (fidx, frame) in enumerate(items)
    ]
    return build_montage(labeled, cols, cell_max_side, jpeg_quality)


# --- 10fps work-frame (coarse-to-fine 용) -----------------------------------
def read_work_frames(path: Path | str, work_fps: int = 10):
    """30fps 소스를 work_fps 로 리샘플해 (work_idx, bgr) 목록을 반환.

    work_idx 는 0부터 시작하는 리샘플 프레임 번호. 원본 프레임 = work_idx * stride.
    반환: (items, src_fps, stride). t초 = work_idx / work_fps.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / work_fps)))

    items = []
    src_idx = 0
    w = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if src_idx % stride == 0:
            items.append((w, frame))
            w += 1
        src_idx += 1
    cap.release()
    if not items:
        raise RuntimeError(f"work 프레임 추출 실패: {path}")
    return items, src_fps, stride


def _blank(h: int, w: int):
    import numpy as np

    return np.zeros((h, w, 3), dtype="uint8")


def _letterbox(frame, cw: int, ch: int):
    """비율 유지로 (cw, ch) 안에 맞추고 검은 여백으로 채운다."""
    import cv2

    h, w = frame.shape[:2]
    scale = min(cw / w, ch / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    cell = _blank(ch, cw)
    y0, x0 = (ch - nh) // 2, (cw - nw) // 2
    cell[y0 : y0 + nh, x0 : x0 + nw] = resized
    return cell


def _label(cell, text: str) -> None:
    """셀 좌상단에 배경 있는 텍스트 라벨을 그린다."""
    import cv2

    font, sc, th = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, tht), _ = cv2.getTextSize(text, font, sc, th)
    cv2.rectangle(cell, (0, 0), (tw + 8, tht + 8), (0, 0, 0), -1)
    cv2.putText(cell, text, (4, tht + 4), font, sc, (0, 255, 0), th, cv2.LINE_AA)


def _resize_max_side(frame, max_side: int):
    """긴 변이 max_side 를 넘으면 비율 유지로 축소."""
    import cv2

    h, w = frame.shape[:2]
    longest = max(h, w)
    if max_side <= 0 or longest <= max_side:
        return frame
    scale = max_side / longest
    new = (int(round(w * scale)), int(round(h * scale)))
    return cv2.resize(frame, new, interpolation=cv2.INTER_AREA)


def video_meta(path: Path | str) -> dict:
    """영상 기본 메타(프레임수·fps·해상도·길이)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "frames": total,
        "fps": fps,
        "width": w,
        "height": h,
        "duration_s": (total / fps) if fps else 0.0,
    }


# --- 윈도우 서브클립 -------------------------------------------------------
def write_subclip(
    path: Path | str,
    t0: float,
    t1: float,
    out_path: Path | str,
    max_side: int = 0,
    out_fps: float | None = None,
) -> dict:
    """[t0, t1]초 구간을 mp4 서브클립으로 저장.

    반환: {frame0, frame1(원본 인덱스), src_fps, n_written, path}.
    out_fps 미지정 시 원본 fps 유지(=실시간 길이 보존).
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = max(0, int(round(t0 * src_fps)))
    f1 = min(total - 1, int(round(t1 * src_fps)))
    out_fps = out_fps or src_fps

    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    writer = None
    n = 0
    for fidx in range(f0, f1 + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frame = _resize_max_side(frame, max_side)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h)
            )
        writer.write(frame)
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    if n == 0:
        raise RuntimeError(f"서브클립 프레임 없음: {path} [{t0},{t1}]")
    return {"frame0": f0, "frame1": f1, "src_fps": src_fps, "n_written": n,
            "path": str(out_path)}


# --- 인코딩 -----------------------------------------------------------------
def to_data_uri(path: Path) -> str:
    """mp4 파일을 base64 data URI 로 인코딩한다(비디오 직접 전송용)."""
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:video/mp4;base64,{b64}"


def jpeg_to_data_uri(jpg: bytes) -> str:
    """JPEG 바이트를 image data URI 로 인코딩한다."""
    b64 = base64.b64encode(jpg).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
