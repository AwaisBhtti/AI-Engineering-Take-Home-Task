"""
run.py — Video Indexing at Speed
AI Engineering Take-Home

Reads Parquet files from a folder, extracts MP4 video bytes, decodes frames,
runs YOLO11l object detection, and writes per-second results to a JSONL file.

Usage:
    python run.py --input <folder of .parquet files> --output results.jsonl
"""

# Configure offline sandbox & writable config directory BEFORE importing ultralytics.
import os
os.environ["YOLO_TELEMETRY"] = "0"
os.environ["ULTRALYTICS_OFFLINE"] = "1"
os.environ["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"

import argparse
import io
import json
import math
import pathlib
import sys

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants — fixed by the brief, do not change
# ---------------------------------------------------------------------------

# Anchor model path relative to script directory
MODEL_PATH   = str(pathlib.Path(__file__).resolve().parent / "yolo11l.pt")
IMGSZ        = 640
CONF         = 0.4
IOU          = 0.7   # Ultralytics default class-aware NMS threshold

# Exact class names as specified in the brief (section 1 and section 4)
TARGET_CLASSES = {"person", "car", "dog", "laptop", "cell phone"}

# COCO class indices for the five target classes (YOLO11l is COCO-pretrained)
COCO_IDX_TO_NAME = {
    0:  "person",
    2:  "car",
    16: "dog",
    63: "laptop",
    67: "cell phone",
}
TARGET_INDICES_LIST = [0, 2, 16, 63, 67]

# OpenCV rotation codes for common rotation tag values (degrees clockwise)
ROTATE_MAP = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

CPU_COUNT = os.cpu_count() or 1


# ---------------------------------------------------------------------------
# Video decoding helpers
# ---------------------------------------------------------------------------

def _get_rotation_code(stream, frame) -> int | None:
    """
    Determine rotation code from frame property, stream metadata, or side data.

    1. PyAV 15+: frame.rotation returns counter-clockwise degrees.
    2. Legacy metadata tag: 'rotate' holds clockwise degrees.
    3. PyAV 12-13: stream.side_data dict holds DISPLAYMATRIX.
    """
    # 1. PyAV 15+: frame.rotation is counter-clockwise degrees
    rot = getattr(frame, "rotation", None)
    if rot:
        deg = (-int(round(rot))) % 360
        return ROTATE_MAP.get(deg)

    # 2. Legacy stream metadata tag (clockwise degrees)
    if hasattr(stream, "metadata") and stream.metadata:
        rotate_tag = (
            stream.metadata.get("rotate")
            or stream.metadata.get("ROTATION")
            or stream.metadata.get("rotation")
        )
        if rotate_tag:
            try:
                return ROTATE_MAP.get(int(rotate_tag) % 360)
            except (ValueError, TypeError):
                pass

    # 3. PyAV 12-13 dict side_data
    side_data = getattr(stream, "side_data", None)
    if isinstance(side_data, dict):
        dm = side_data.get("DISPLAYMATRIX")
        if dm is not None:
            try:
                deg = (-int(round(float(dm)))) % 360
                return ROTATE_MAP.get(deg)
            except (ValueError, TypeError):
                pass

    return None


def decode_frames(video_bytes: bytes):
    """
    Decode all frames from a video stored as raw bytes in a single pass.

    Yields (second_index: int, frame_bgr: np.ndarray) for every frame.
    - Uses exact Fraction arithmetic on integer PTS differences to prevent float drift.
    - Clamps non-monotonic PTS jumps.
    - Demuxes with per-packet AVError recovery so single corrupt packets don't abort playable video.
    - Flushes remaining frames from the decoder at EOF.
    - Rotation is determined on the first decoded frame and applied to all frames.
    """
    container = av.open(io.BytesIO(video_bytes))
    try:
        video_streams = container.streams.video
        if not video_streams:
            raise ValueError("No video stream found")

        stream = video_streams[0]
        # Enable multi-threaded FFmpeg CPU frame decoding across available cores
        try:
            stream.thread_type = "AUTO"
        except Exception:
            pass

        rotate_code = None
        rotation_determined = False
        first_pts = None
        first_time = None
        last_second = 0

        def _process_frame(frame):
            nonlocal rotate_code, rotation_determined, first_pts, first_time, last_second

            if not rotation_determined:
                rotate_code = _get_rotation_code(stream, frame)
                rotation_determined = True

            if frame.pts is not None:
                if first_pts is None:
                    first_pts = frame.pts
                # Exact Fraction arithmetic: prevents floating-point drift across boundaries
                rel = (frame.pts - first_pts) * stream.time_base
                second = math.floor(rel) if rel > 0 else 0
            elif getattr(frame, "time", None) is not None:
                t = float(frame.time)
                if first_time is None:
                    first_time = t
                rel = t - first_time
                second = math.floor(rel) if rel > 0 else 0
            else:
                return None

            # Non-monotonic PTS guard: prevent backward jumps
            if second < last_second:
                second = last_second
            last_second = second

            img = frame.to_ndarray(format="bgr24")
            if rotate_code is not None:
                img = cv2.rotate(img, rotate_code)

            return second, img

        # 1. Demux packets with per-packet AVError recovery
        for packet in container.demux(stream):
            if packet.size == 0:
                continue
            try:
                for frame in packet.decode():
                    res = _process_frame(frame)
                    if res is not None:
                        yield res
            except av.AVError:
                continue

        # 2. Flush remaining buffered frames from codec context
        try:
            for frame in stream.decode(None):
                res = _process_frame(frame)
                if res is not None:
                    yield res
        except Exception:
            pass

    finally:
        container.close()


# ---------------------------------------------------------------------------
# Per-second aggregation
# ---------------------------------------------------------------------------

def _floor_median(values: list) -> int:
    """
    Compute median of integers, rounding DOWN when halfway between two numbers.
    """
    n = len(values)
    if n == 0:
        return 0
    s = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    else:
        return (s[mid - 1] + s[mid]) // 2  # integer floor division


def aggregate_second(frame_detections: list) -> tuple:
    """
    Aggregate per-frame detection results into per-second output values.

    Rules (brief section 5):
        An object is PRESENT if detected in >= 50% of that second's frames.
        People count = floor(median(person count per frame)).
    """
    n = len(frame_detections)
    if n == 0:
        return [], 0

    class_frame_counts = {cls: 0 for cls in TARGET_CLASSES}
    people_per_frame   = []

    for frame_dict in frame_detections:
        people_in_frame = frame_dict.get("person", 0)
        people_per_frame.append(people_in_frame)

        for cls in TARGET_CLASSES:
            if cls in frame_dict:
                class_frame_counts[cls] += 1

    threshold = n / 2.0
    objects   = sorted(
        cls for cls, cnt in class_frame_counts.items()
        if cnt >= threshold
    )

    people = _floor_median(people_per_frame)

    return objects, people


# ---------------------------------------------------------------------------
# YOLO inference
# ---------------------------------------------------------------------------

def run_inference(model: YOLO, frames: list) -> list:
    """
    Run YOLO11l on a batch of BGR frames with direct target class filtering.
    """
    if not frames:
        return []

    # Passing classes directly skips NMS work on the other 75 COCO classes
    results = model(
        frames,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        classes=TARGET_INDICES_LIST,
        verbose=False,
        stream=False,
    )

    per_frame = []
    for result in results:
        frame_counts = {}
        if result.boxes is not None and len(result.boxes):
            cls_indices = result.boxes.cls.int().tolist()
            for idx in cls_indices:
                name = COCO_IDX_TO_NAME.get(idx)
                if name:
                    frame_counts[name] = frame_counts.get(name, 0) + 1
        per_frame.append(frame_counts)

    return per_frame


# ---------------------------------------------------------------------------
# Single-video processing (streaming output, low memory footprint)
# ---------------------------------------------------------------------------

def process_video(
    model:       YOLO,
    video_bytes: bytes,
    video_id:    str,
    out_file,
    batch_size:  int = 32,
) -> None:
    """
    Process one video end-to-end with progressive line-by-line streaming.

    Writes completed seconds to out_file as soon as the video timestamp
    advances, keeping memory flat and preserving all completed seconds if
    an error occurs later in the stream.
    """
    if not video_bytes:
        out_file.write(json.dumps({"video": video_id, "error": "empty or missing video bytes"}) + "\n")
        out_file.flush()
        return

    second_buckets: dict = {}
    next_second_to_write = 0
    last_seen_second     = -1

    def _write_second(sec: int) -> None:
        frame_dets = second_buckets.pop(sec, [])
        objects, people = aggregate_second(frame_dets)
        line = {
            "video":   video_id,
            "second":  sec,
            "objects": objects,
            "people":  people,
        }
        out_file.write(json.dumps(line) + "\n")

    frame_buffer:  list = []   # [(bgr_array), ...]
    second_buffer: list = []   # parallel second indices

    def _flush_buffer() -> None:
        if frame_buffer:
            detections = run_inference(model, frame_buffer)
            for sec, det in zip(second_buffer, detections):
                second_buckets.setdefault(sec, []).append(det)
            frame_buffer.clear()
            second_buffer.clear()

    def _emit_ready_seconds(current_sec: int) -> None:
        """Stream out completed seconds strictly earlier than current_sec."""
        nonlocal next_second_to_write
        while next_second_to_write < current_sec:
            _write_second(next_second_to_write)
            next_second_to_write += 1

    try:
        for second, frame in decode_frames(video_bytes):
            last_seen_second = max(last_seen_second, second)
            frame_buffer.append(frame)
            second_buffer.append(second)

            # Flush inference batch when full
            if len(frame_buffer) >= batch_size:
                _flush_buffer()
                _emit_ready_seconds(second)

        # Flush remaining frames at end of video
        _flush_buffer()

        # Emit all remaining seconds up to and including last_seen_second
        if last_seen_second >= 0:
            while next_second_to_write <= last_seen_second:
                _write_second(next_second_to_write)
                next_second_to_write += 1
        else:
            raise ValueError("no frames decoded")

        out_file.flush()

    except Exception as exc:
        # If video fails partway through:
        # 1. Flush any remaining decoded frames
        try:
            _flush_buffer()
        except Exception:
            pass

        # 2. Emit all completed seconds strictly before last_seen_second
        # (Preserves completed seconds, fills gaps with empty seconds, and avoids partial crash second)
        try:
            if last_seen_second > 0:
                while next_second_to_write < last_seen_second:
                    _write_second(next_second_to_write)
                    next_second_to_write += 1
        except Exception:
            pass

        # 3. Append exactly one error line (section 4 requirement)
        error_msg = str(exc).replace("\n", " ")[:200]
        out_file.write(json.dumps({"video": video_id, "error": error_msg}) + "\n")
        out_file.flush()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Index objects in video stored in Parquet files using YOLO11l."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing .parquet files (each row = one video in 'mp4' column)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the JSONL results file (e.g. results.jsonl)",
    )
    args = parser.parse_args()

    input_dir   = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    # Validate input directory
    if not input_dir.is_dir():
        print(f"ERROR: --input must be a directory, got: {input_dir}", file=sys.stderr)
        sys.exit(1)

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: No .parquet files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists before opening
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(parquet_files)} parquet file(s) in {input_dir}")
    print(f"Output: {output_path}")
    print(f"CPU logical cores: {CPU_COUNT}")

    # Load model (weights must be pre-downloaded during prep step)
    print(f"Loading model: {MODEL_PATH}")
    device = "cuda" if _cuda_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, running on CPU (will be slow)")

    model = YOLO(MODEL_PATH)
    model.to(device)
    # Warm up model with a dummy frame
    _warmup(model, device)

    # Determine GPU inference batch size based on available device VRAM
    gpu_batch_size = _pick_batch_size(device)
    print(f"Device: {device}  |  YOLO inference batch size: {gpu_batch_size}")

    # Open output file with line buffering
    with open(output_path, "w", buffering=1) as out_file:
        total_videos = 0
        total_errors = 0

        for pq_idx, parquet_path in enumerate(parquet_files, 1):
            print(f"[{pq_idx}/{len(parquet_files)}] {parquet_path.name}")

            try:
                pf      = pq.ParquetFile(parquet_path)
                row_idx = 0

                # Stream 10 rows at a time — only 'mp4' column (flat RAM usage)
                for batch in pf.iter_batches(batch_size=10, columns=["mp4"]):
                    for i in range(batch.num_rows):
                        video_bytes = batch.column("mp4")[i].as_py()
                        video_id    = f"{parquet_path.name}:{row_idx}"

                        process_video(
                            model,
                            video_bytes,
                            video_id,
                            out_file,
                            batch_size=gpu_batch_size,
                        )
                        total_videos += 1

                        del video_bytes
                        row_idx += 1

            except Exception as exc:
                print(f"  ERROR reading {parquet_path.name}: {exc}", file=sys.stderr)
                total_errors += 1

    print(f"\nDone. Processed {total_videos} video(s), {total_errors} file error(s).")
    print(f"Results written to: {output_path}")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _warmup(model: YOLO, device: str) -> None:
    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    model([dummy], imgsz=IMGSZ, conf=CONF, classes=TARGET_INDICES_LIST, verbose=False)


def _pick_batch_size(device: str) -> int:
    if device == "cpu":
        return 8

    try:
        import torch
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb >= 20:
            return 32   # RTX 4090 (24 GB), RTX 3090 (24 GB)
        elif vram_gb >= 12:
            return 16   # T4 (16 GB)
        elif vram_gb >= 6:
            return 8
        else:
            return 4
    except Exception:
        return 8


if __name__ == "__main__":
    main()
