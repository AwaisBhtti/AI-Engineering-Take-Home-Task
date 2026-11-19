"""
run.py — Video Indexing at Speed
AI Engineering Take-Home

Reads Parquet files from a folder, extracts MP4 video bytes, decodes frames,
runs YOLO11l object detection, and writes per-second results to a JSONL file.

Usage:
    python run.py --input <folder of .parquet files> --output results.jsonl
"""

# Disable ultralytics telemetry and update checks BEFORE any imports.
# run.py is executed in an offline sandbox — any network call will hang or fail.
import os
os.environ["YOLO_TELEMETRY"] = "0"
os.environ["ULTRALYTICS_OFFLINE"] = "1"

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

MODEL_PATH   = "yolo11l.pt"
IMGSZ        = 640
CONF         = 0.4
IOU          = 0.7   # Ultralytics default; listed here for clarity

# Exact class names as specified in the brief (section 1 and section 4)
TARGET_CLASSES = {"person", "car", "dog", "laptop", "cell phone"}

# COCO class indices for the five target classes (YOLO11l is COCO-pretrained)
# Filtering by index is faster than filtering by name string inside the loop.
# Verified against the official COCO 80-class list.
COCO_IDX_TO_NAME = {
    0:  "person",
    2:  "car",
    16: "dog",
    63: "laptop",
    67: "cell phone",
}
TARGET_INDICES = set(COCO_IDX_TO_NAME.keys())

# NVDEC hardware codec map: PyAV codec name → cuvid decoder name
NVDEC_MAP = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "vp9":  "vp9_cuvid",
    "av1":  "av1_cuvid",
}

# OpenCV rotation codes for common rotation tag values (degrees clockwise)
ROTATE_MAP = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

# ---------------------------------------------------------------------------
# Hardware-aware concurrency (no hardcoded numbers)
# ---------------------------------------------------------------------------

CPU_COUNT = os.cpu_count() or 1
# Leave one logical core for the main/GPU thread; use the rest for I/O work.
IO_WORKERS = max(1, CPU_COUNT - 1)


# ---------------------------------------------------------------------------
# Video decoding helpers
# ---------------------------------------------------------------------------

def _get_rotation(stream) -> int:
    """
    Read rotation degrees (clockwise) from the MP4 container metadata.

    The rotation tag is embedded inside the video file itself — in the moov
    box / track header transformation matrix — NOT in the Parquet json column.
    PyAV exposes it via two paths:
      1. stream.metadata["rotate"] — integer string, used by older encoders
      2. stream.side_data DISPLAYMATRIX — transformation matrix, used by newer
         encoders (e.g. most smartphones)
    """
    # Method 1: simple metadata tag
    rotate_tag = stream.metadata.get("rotate")
    if rotate_tag:
        try:
            return int(rotate_tag) % 360
        except ValueError:
            pass

    # Method 2: display matrix side data
    for sd in stream.side_data:
        if sd.type == "DISPLAYMATRIX":
            try:
                # av.video.transform.display_matrix_rotation returns degrees
                # counter-clockwise, so negate it for clockwise convention.
                deg = -round(av.video.transform.display_matrix_rotation(sd))
                return deg % 360
            except Exception:
                pass

    return 0


def _open_container(video_bytes: bytes) -> av.container.InputContainer:
    """
    Open a video container, trying NVDEC hardware decode first.

    NVDEC runs on a dedicated fixed-function chip on the GPU and does NOT
    compete with YOLO for CUDA cores. If NVDEC is not available (e.g. the
    system FFmpeg was not compiled with NVDEC support), falls back to CPU
    decode transparently.
    """
    # Probe codec without opening a full decode session
    probe_buf = io.BytesIO(video_bytes)
    try:
        probe = av.open(probe_buf)
        codec_name = probe.streams.video[0].codec_context.name
        probe.close()
    except Exception:
        # Unreadable container — let the caller handle it
        return av.open(io.BytesIO(video_bytes))

    hw_codec = NVDEC_MAP.get(codec_name)
    if hw_codec:
        try:
            container = av.open(io.BytesIO(video_bytes),
                                options={"c:v": hw_codec})
            # Force codec initialisation to catch NVDEC errors early
            _ = container.streams.video[0].codec_context.name
            return container
        except (av.AVError, Exception):
            pass  # NVDEC unavailable — fall through to CPU

    return av.open(io.BytesIO(video_bytes))


def decode_frames(video_bytes: bytes):
    """
    Decode all frames from a video stored as raw bytes.

    Yields (second_index: int, frame_bgr: np.ndarray) for every frame.
    - second_index = floor(frame_timestamp_in_seconds), counting from the
      timestamp of the very first decoded frame (which may not be 0.0 for
      some containers — we normalise it).
    - frame_bgr is in BGR colour order as required by the brief and expected
      by Ultralytics YOLO (which uses cv2 conventions internally).
    - Rotation (from the MP4 container moov box) is applied before yielding.
    """
    container = _open_container(video_bytes)
    try:
        video_streams = container.streams.video
        if not video_streams:
            raise ValueError("No video stream found")

        stream   = video_streams[0]
        rotation = _get_rotation(stream)
        rotate_code = ROTATE_MAP.get(rotation)

        first_ts = None  # will be set on first frame

        for packet in container.demux(stream):
            try:
                for frame in packet.decode():
                    if frame.pts is None:
                        continue

                    ts = float(frame.pts * stream.time_base)

                    # Normalise: timestamp relative to first decoded frame
                    if first_ts is None:
                        first_ts = ts
                    ts -= first_ts

                    if ts < 0:
                        ts = 0.0

                    second = math.floor(ts)
                    img    = frame.to_ndarray(format="bgr24")

                    if rotate_code is not None:
                        img = cv2.rotate(img, rotate_code)

                    yield second, img

            except av.AVError:
                # Skip corrupt packets; continue with the rest of the video
                continue
    finally:
        container.close()


# ---------------------------------------------------------------------------
# Per-second aggregation
# ---------------------------------------------------------------------------

def _floor_median(values: list) -> int:
    """
    Compute the median of a list of non-negative integers, rounding DOWN
    when the result falls exactly halfway between two integers.

    The brief specifies: "the median of the number of people found in each
    frame of that second, rounded down when it falls halfway between two
    numbers."

    Python's statistics.median() averages the two middle values for even-
    length lists (e.g. [2, 3] → 2.5). We must floor that result.
    """
    n = len(values)
    if n == 0:
        return 0
    s = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        # Odd number of values — exact median
        return s[mid]
    else:
        # Even number — average of two middle values, then floor
        return (s[mid - 1] + s[mid]) // 2  # integer floor division


def aggregate_second(frame_detections: list) -> tuple:
    """
    Aggregate per-frame detection results into per-second output values.

    Args:
        frame_detections: list of sets, one per sampled frame in this second.
                          Each set contains the class NAMES of detected objects
                          (filtered to TARGET_CLASSES, with person repeated
                          once per detected person for counting).

    Returns:
        (objects: list[str], people_count: int)
        - objects: sorted list of TARGET_CLASSES present in >= 50% of frames
        - people_count: floor(median(people per frame))

    Rules (brief section 5):
        An object is PRESENT if detected in >= 50% of that second's frames.
        People count = floor(median(person count per frame)).
    """
    n = len(frame_detections)
    if n == 0:
        return [], 0

    # Count detections per class across all frames
    class_frame_counts = {cls: 0 for cls in TARGET_CLASSES}
    people_per_frame   = []

    for frame_set in frame_detections:
        # frame_set is a dict: {class_name: count} for that frame
        people_in_frame = frame_set.get("person", 0)
        people_per_frame.append(people_in_frame)

        for cls in TARGET_CLASSES:
            if cls in frame_set:
                class_frame_counts[cls] += 1

    # Object present if in >= 50% of frames (>= because 50% counts as present)
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
    Run YOLO11l on a batch of BGR frames.

    Args:
        model:  loaded YOLO model (already on GPU)
        frames: list of np.ndarray images in BGR format

    Returns:
        List of dicts (one per frame): {class_name: count, ...}
        Only TARGET_CLASSES entries are included; absent classes are omitted.
    """
    if not frames:
        return []

    results = model(
        frames,
        imgsz=640,
        conf=CONF,
        iou=IOU,
        verbose=False,
        stream=False,
    )

    per_frame = []
    for result in results:
        frame_counts = {}
        if result.boxes is not None and len(result.boxes):
            cls_indices = result.boxes.cls.int().tolist()
            for idx in cls_indices:
                if idx in TARGET_INDICES:
                    name = COCO_IDX_TO_NAME[idx]
                    frame_counts[name] = frame_counts.get(name, 0) + 1
        per_frame.append(frame_counts)

    return per_frame


# ---------------------------------------------------------------------------
# Single-video processing
# ---------------------------------------------------------------------------

def process_video(
    model:       YOLO,
    video_bytes: bytes,
    video_id:    str,
    out_file,
    batch_size:  int = 32,
) -> None:
    """
    Process one video end-to-end and write output lines to out_file.

    For each decoded frame we:
      1. Buffer frames into GPU inference batches of `batch_size`.
      2. Run YOLO on the full batch at once for throughput.
      3. Accumulate per-frame detection dicts into per-second buckets.
      4. When the second index changes (or at end of video), aggregate and
         write one JSONL line per completed second.

    On any exception mid-video:
      - Lines already written for completed seconds are kept.
      - One error line is appended for this video_id.

    The brief requires writing a line for every second up to and INCLUDING
    the second that holds the last frame, even if that second is partial.
    """
    # Accumulator: second_index -> list of per-frame detection dicts
    second_buckets: dict = {}   # {second: [frame_dict, ...]}
    written_seconds: set = set()
    last_second:    int  = -1

    def _write_second(sec: int) -> None:
        """Aggregate and write one JSONL line for `sec`."""
        frame_dets = second_buckets.pop(sec, [])
        objects, people = aggregate_second(frame_dets)
        line = {
            "video":   video_id,
            "second":  sec,
            "objects": objects,
            "people":  people,
        }
        out_file.write(json.dumps(line) + "\n")
        written_seconds.add(sec)

    try:
        # --- Collect all frames, batch them for YOLO ---
        # We buffer (second, frame) pairs and flush complete seconds as we go.
        frame_buffer:  list = []   # [(second, bgr_array), ...]
        second_buffer: list = []   # parallel second indices

        for second, frame in decode_frames(video_bytes):
            frame_buffer.append(frame)
            second_buffer.append(second)
            last_second = max(last_second, second)

            # Run inference when buffer is full
            if len(frame_buffer) >= batch_size:
                detections = run_inference(model, frame_buffer)
                for sec, det in zip(second_buffer, detections):
                    second_buckets.setdefault(sec, []).append(det)
                frame_buffer.clear()
                second_buffer.clear()

        # Flush remaining frames
        if frame_buffer:
            detections = run_inference(model, frame_buffer)
            for sec, det in zip(second_buffer, detections):
                second_buckets.setdefault(sec, []).append(det)

        # Write all complete seconds in order
        if second_buckets:
            # Fill any gaps between 0 and last_second (seconds with no frames
            # still need a line: {"objects": [], "people": 0})
            for sec in range(last_second + 1):
                if sec not in second_buckets:
                    second_buckets[sec] = []
            for sec in sorted(second_buckets.keys()):
                _write_second(sec)

        elif last_second == -1:
            # Video had no decodeable frames at all
            raise ValueError("no frames decoded")

        out_file.flush()

    except Exception as exc:
        # Keep lines already written; add one error line for this video
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

    input_dir  = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    # Validate input directory
    if not input_dir.is_dir():
        print(f"ERROR: --input must be a directory, got: {input_dir}", file=sys.stderr)
        sys.exit(1)

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: No .parquet files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(parquet_files)} parquet file(s) in {input_dir}")
    print(f"Output: {output_path}")
    print(f"CPU logical cores: {CPU_COUNT}  |  IO workers: {IO_WORKERS}")

    # Load model (weights must be pre-downloaded during prep step)
    print(f"Loading model: {MODEL_PATH}")
    device = "cuda" if _cuda_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, running on CPU (will be slow)")

    model = YOLO(MODEL_PATH)
    model.to(device)
    # Warm up the model with a dummy frame so the first real video isn't delayed
    _warmup(model, device)

    # Determine GPU inference batch size based on available device
    # RTX 4090 (24 GB): batch=32 comfortably fits at imgsz=640
    # Colab T4 (16 GB): batch=16 is safe
    # CPU: batch=8 to avoid excessive RAM use
    gpu_batch_size = _pick_batch_size(device)
    print(f"Device: {device}  |  YOLO inference batch size: {gpu_batch_size}")

    # Open output file in write mode (creates or truncates)
    with open(output_path, "w", buffering=1) as out_file:
        total_videos = 0
        total_errors = 0

        for pq_idx, parquet_path in enumerate(parquet_files, 1):
            print(f"[{pq_idx}/{len(parquet_files)}] {parquet_path.name}")

            try:
                pf      = pq.ParquetFile(parquet_path)
                row_idx = 0

                # Stream 10 rows at a time — only the 'mp4' column.
                # This keeps RAM usage flat (~150 MB per batch of 10 videos)
                # regardless of Parquet file size (~0.5 GB each).
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

                        # Explicitly release video bytes before reading next
                        del video_bytes
                        row_idx += 1

            except Exception as exc:
                # Entire Parquet file unreadable — log and continue
                print(f"  ERROR reading {parquet_path.name}: {exc}", file=sys.stderr)
                total_errors += 1

    print(f"\nDone. Processed {total_videos} video(s), {total_errors} file error(s).")
    print(f"Results written to: {output_path}")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    """Check CUDA without importing torch at module level (saves startup time)."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _warmup(model: YOLO, device: str) -> None:
    """
    Run one dummy inference to initialise CUDA kernels and allocate GPU memory.
    This prevents the first real video from suffering a cold-start latency spike.
    """
    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    model([dummy], imgsz=IMGSZ, conf=CONF, verbose=False)


def _pick_batch_size(device: str) -> int:
    """
    Choose a safe YOLO inference batch size based on available GPU memory.
    Scales automatically — no hardcoded values for specific machines.
    """
    if device == "cpu":
        return 8

    try:
        import torch
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if vram_gb >= 20:
            return 32   # RTX 4090 (24 GB), RTX 3090 (24 GB), A100 (40/80 GB)
        elif vram_gb >= 12:
            return 16   # RTX 3080 (12 GB), T4 (16 GB)
        elif vram_gb >= 6:
            return 8    # GTX 1660 (6 GB), RTX 3060 (12 GB)
        else:
            return 4
    except Exception:
        return 8        # safe default if memory query fails


if __name__ == "__main__":
    main()

