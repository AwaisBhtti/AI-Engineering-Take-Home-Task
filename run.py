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
import queue
import sys
import threading

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

# Frame subsampling: at most this many frames are sent to YOLO per second of
# video. Set to a number >= native fps (e.g. 999) to disable and run every
# frame. This is the ONLY thing you need to change to tune speed vs. accuracy.
SAMPLE_FPS   = 15

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

# Set once in main() based on device: 16 enables FP16 (Tensor Core) inference
# on CUDA via Ultralytics' 'quantize' argument (the 'half=True' flag is
# deprecated). Left None on CPU, where FP16 isn't supported/faster.
QUANTIZE = None


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


def decode_frames(video_bytes: bytes, sample_fps: int):
    """
    Decode frames from a video stored as raw bytes in a single pass, keeping
    only the subsampled subset the caller actually needs.

    Yields (second_index: int, frame_bgr: np.ndarray) for each KEPT frame —
    at most sample_fps per second, picked by dividing each second into
    sample_fps equal time-slots and keeping the first frame in each slot.
    The subsampling decision happens BEFORE the expensive BGR conversion /
    rotation, so a dropped frame costs only cheap PTS arithmetic, not a
    swscale color-convert + cv2.rotate on data that's about to be discarded.

    - Uses exact Fraction arithmetic on integer PTS differences to prevent float drift.
    - Clamps non-monotonic PTS jumps (and the resulting frac, see below).
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
        frame_count = 0

        # Subsampling state: each second is divided into sample_fps equal
        # time-slots; we keep only the first frame seen in each slot.
        sample_second = -1
        sample_slot   = -1

        # Precompute stream duration in seconds if available
        stream_duration_sec = None
        if stream.duration is not None and stream.time_base is not None:
            try:
                stream_duration_sec = float(stream.duration * stream.time_base)
            except Exception:
                pass

        def _process_frame(frame):
            nonlocal rotate_code, rotation_determined, first_pts, first_time, last_second, frame_count
            nonlocal sample_second, sample_slot

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

            # Guard against garbage / corrupted PTS forward jumps:
            # 1. A single step jumping forward by > 60s (normal frame intervals are ~0.033s)
            if frame_count > 0 and (second - last_second) > 60:
                return None

            # 2. Timestamps far beyond declared stream duration
            if stream_duration_sec is not None and second > (stream_duration_sec + 30):
                return None

            # Non-monotonic PTS guard: prevent backward jumps
            if second < last_second:
                second = last_second

            last_second = second
            frame_count += 1

            # Fractional position within the second (0.0-1.0), used only to
            # pick evenly-spaced samples — does not affect second boundaries.
            # Clamped because the backward-jump guard above can force `second`
            # upward without changing `rel`, which would otherwise make this
            # negative (and silently corrupt the slot bookkeeping below).
            frac = float(rel) - second
            frac = max(0.0, min(0.999999, frac))

            if second != sample_second:
                sample_second = second
                sample_slot = -1
            slot = int(frac * sample_fps)
            # <= rather than == : after the backward-jump clamp above, a
            # frame's "true" position can still map to a slot at or below one
            # we've already kept for this second, even post-clamp. == would
            # let such a frame through and reset sample_slot backward,
            # causing every later frame in the second to be over-sampled;
            # <= rejects it like any other already-covered slot.
            if slot <= sample_slot:
                return None  # subsampled out — never pay for ndarray/rotate
            sample_slot = slot

            img = frame.to_ndarray(format="bgr24")
            if rotate_code is not None:
                img = cv2.rotate(img, rotate_code)

            return second, img

        # Demux packets with per-packet AVError recovery.
        # Note: Do not skip packet.size == 0; PyAV's empty EOF packet triggers internal B-frame flushing.
        for packet in container.demux(stream):
            try:
                for frame in packet.decode():
                    res = _process_frame(frame)
                    if res is not None:
                        yield res
            except av.AVError:
                continue

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
        quantize=QUANTIZE,
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
#
# Decode runs in a background thread that pushes subsampled frames onto a
# bounded queue; the main thread pulls batches off that queue and runs GPU
# inference. This overlaps CPU decode of the *next* batch with GPU inference
# of the *current* one, instead of doing them strictly one after another.
# Per-video batching and write semantics are unchanged from the synchronous
# version — only *where* decode happens changed, not when results get
# written or how a mid-video error is handled.
# ---------------------------------------------------------------------------

def _decode_worker(
    video_bytes: bytes,
    frame_q:     "queue.Queue",
    stop_event:  "threading.Event",
) -> None:
    """
    Runs in a background thread. Decodes one video (decode_frames already
    applies subsampling before handing frames back, so no filtering happens
    here) and pushes results onto frame_q as ("frame", second, frame_bgr)
    items, followed by exactly one ("done", last_seen_second,
    error_msg_or_None) sentinel.

    Never raises — any decode exception is converted into the sentinel's
    error_msg so the main thread doesn't need to catch across threads.

    stop_event: set by the consumer if it gives up early (e.g. an inference
    failure mid-video). Checked before every enqueue attempt, and enqueues
    use a bounded timeout rather than a plain blocking put, so this thread
    can never be left permanently blocked pushing into a queue nobody is
    draining anymore.
    """
    def _try_put(item) -> bool:
        while not stop_event.is_set():
            try:
                frame_q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    last_seen_second = -1
    error_msg = None

    try:
        for second, frame in decode_frames(video_bytes, SAMPLE_FPS):
            if stop_event.is_set():
                return  # consumer already gave up; stop decoding early
            last_seen_second = max(last_seen_second, second)
            if not _try_put(("frame", second, frame)):
                return  # stop_event was set while we were waiting to enqueue

        if last_seen_second < 0:
            error_msg = "no frames decoded"
    except Exception as exc:
        error_msg = str(exc).replace("\n", " ")[:200]

    _try_put(("done", last_seen_second, error_msg))


def process_video(
    model:       YOLO,
    video_bytes: bytes,
    video_id:    str,
    out_file,
    batch_size:  int,  # always supplied by main() from _pick_batch_size(), no fallback here
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

    # Bounded so the decoder thread can run a few batches ahead of inference
    # (giving the overlap room to work) without buffering an entire video's
    # frames in memory if inference ever falls behind decode.
    frame_q: "queue.Queue" = queue.Queue(maxsize=batch_size * 4)
    stop_event = threading.Event()
    decoder_thread = threading.Thread(
        target=_decode_worker, args=(video_bytes, frame_q, stop_event), daemon=True
    )
    decoder_thread.start()

    last_seen_second = -1
    error_msg = None
    exited_via_done = False

    try:
        while True:
            kind, a, b = frame_q.get()
            if kind == "frame":
                second, frame = a, b
                frame_buffer.append(frame)
                second_buffer.append(second)
                if len(frame_buffer) >= batch_size:
                    _flush_buffer()
                    _emit_ready_seconds(second)
            else:  # "done"
                last_seen_second, error_msg = a, b
                exited_via_done = True
                break
    except Exception as exc:
        # e.g. CUDA OOM or a malformed frame batch inside _flush_buffer().
        # Previously this propagated out of process_video uncaught, which
        # (a) skipped writing anything at all for this video, (b) aborted
        # the rest of the current parquet file's videos too, since the only
        # try/except around this call is at the parquet-file level in
        # main(), and (c) could leave the decoder thread permanently
        # blocked trying to enqueue into a queue nobody drains anymore.
        # Caught here instead: drop the in-flight batch (don't retry it),
        # record the error, and let the normal error-handling path below
        # write everything that completed before it.
        error_msg = str(exc).replace("\n", " ")[:200]
        if second_buckets:
            last_seen_second = max(last_seen_second, max(second_buckets.keys()))
        if second_buffer:
            last_seen_second = max(last_seen_second, max(second_buffer))
        frame_buffer.clear()
        second_buffer.clear()
    finally:
        # Tell the decoder thread to stop (harmless/no-op if it already
        # finished) and bound the wait — _try_put() in _decode_worker checks
        # this every ≤0.5s even mid-enqueue, so this should return promptly.
        stop_event.set()
        decoder_thread.join(timeout=5.0)

    if exited_via_done:
        # Flush whatever's left in the buffer. If this itself raises (e.g.
        # the GPU call on the tail batch fails), treat it the same as a
        # decode-time error: everything already written stands, and we drop
        # only the unresolved tail rather than the whole video.
        try:
            _flush_buffer()
        except Exception as exc:
            if error_msg is None:
                error_msg = str(exc).replace("\n", " ")[:200]

    if error_msg is None:
        # Success: emit every second from 0 through last_seen_second inclusive.
        if last_seen_second >= 0:
            while next_second_to_write <= last_seen_second:
                _write_second(next_second_to_write)
                next_second_to_write += 1
        else:
            error_msg = "no frames decoded"  # safety net; _decode_worker already sets this

    if error_msg is not None:
        # Error: emit only seconds strictly before last_seen_second (the
        # second in progress when the error occurred may be incomplete),
        # then exactly one error line (section 4 requirement).
        if last_seen_second > 0:
            while next_second_to_write < last_seen_second:
                _write_second(next_second_to_write)
                next_second_to_write += 1
        out_file.write(json.dumps({"video": video_id, "error": error_msg}) + "\n")

    out_file.flush()


# ---------------------------------------------------------------------------
# Resume support (opt-in via --resume; irrelevant to the grader's fresh VM)
# ---------------------------------------------------------------------------

def _scan_resume_state(output_path: "pathlib.Path", parquet_files: list) -> tuple:
    """
    Reads an existing --output file from a previous, interrupted run and
    decides which .parquet files were fully completed.

    A file only counts as fully completed if the highest row index seen for
    it in the existing output equals (num_rows_in_that_file - 1) — every row
    has at least one line (result or error) written. Any file that is only
    partially present has ALL of its lines discarded here and is fully
    reprocessed from scratch on this run, so no video can ever end up with
    duplicate or half-written-then-complete entries in the final file.

    Returns (kept_lines, done_file_names).
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return [], set()

    num_rows_by_file = {}
    for p in parquet_files:
        try:
            num_rows_by_file[p.name] = pq.ParquetFile(p).metadata.num_rows
        except Exception:
            num_rows_by_file[p.name] = None  # unreadable right now -> never counts as "done"

    max_row_seen:  dict = {}
    lines_by_file: dict = {}

    with open(output_path, "r") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            try:
                video_field = json.loads(line)["video"]
                file_name, row_str = video_field.rsplit(":", 1)
                row_idx = int(row_str)
            except Exception:
                continue  # malformed line (e.g. truncated mid-write by a kill) -> drop it
            lines_by_file.setdefault(file_name, []).append(line)
            if row_idx > max_row_seen.get(file_name, -1):
                max_row_seen[file_name] = row_idx

    kept_lines: list = []
    done_files: set = set()
    for file_name, lines in lines_by_file.items():
        total = num_rows_by_file.get(file_name)
        if total is not None and max_row_seen.get(file_name, -1) == total - 1:
            kept_lines.extend(lines)
            done_files.add(file_name)
        # else: incomplete, or file no longer present -> discard, reprocess fully below

    return kept_lines, done_files


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previous run: skip .parquet files that are already fully "
            "written to --output, and cleanly redo (from scratch) any file that "
            "was only partially done when the previous run stopped. Off by "
            "default, so the brief's required invocation is unaffected — this "
            "is only for your own multi-session runs (e.g. Kaggle/Colab "
            "free-tier time limits), not for grading."
        ),
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

    # FP16 (Tensor Core) inference — only on CUDA, not supported/faster on CPU.
    global QUANTIZE
    QUANTIZE = 16 if device == "cuda" else None

    # Warm up model with a dummy frame (same quantize setting used at inference time)
    _warmup(model, device)

    # Determine GPU inference batch size based on available device VRAM
    gpu_batch_size = _pick_batch_size(device)
    print(f"Device: {device}  |  YOLO inference batch size: {gpu_batch_size}  |  "
          f"quantize: {QUANTIZE}  |  Subsample: {SAMPLE_FPS} fps/sec")

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
    model([dummy], imgsz=IMGSZ, conf=CONF, classes=TARGET_INDICES_LIST, quantize=QUANTIZE, verbose=False)


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