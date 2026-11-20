"""Index objects and people per second in videos stored in Parquet files."""

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
import torch
from ultralytics import YOLO


MODEL_PATH = pathlib.Path(__file__).resolve().parent / "yolo11l.pt"
IMGSZ = 640
CONF = 0.4
IOU = 0.7
SAMPLE_FPS = 15

COCO_IDX_TO_NAME = {
    0: "person",
    2: "car",
    16: "dog",
    63: "laptop",
    67: "cell phone",
}
TARGET_INDICES = list(COCO_IDX_TO_NAME)
TARGET_CLASSES = set(COCO_IDX_TO_NAME.values())

ROTATE_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def get_rotation_code(stream, frame):
    rotation = getattr(frame, "rotation", None)
    if rotation:
        return ROTATE_MAP.get((-int(round(rotation))) % 360)

    metadata = getattr(stream, "metadata", {}) or {}
    tag = (
        metadata.get("rotate")
        or metadata.get("ROTATION")
        or metadata.get("rotation")
    )
    if tag:
        try:
            return ROTATE_MAP.get(int(tag) % 360)
        except (ValueError, TypeError):
            pass

    side_data = getattr(stream, "side_data", None)
    if isinstance(side_data, dict):
        matrix = side_data.get("DISPLAYMATRIX")
        if matrix is not None:
            try:
                return ROTATE_MAP.get((-int(round(float(matrix)))) % 360)
            except (ValueError, TypeError):
                pass

    return None

def decode_frames(video_bytes, sample_fps):
    container = av.open(io.BytesIO(video_bytes))
    try:
        if not container.streams.video:
            raise ValueError("No video stream found")

        stream = container.streams.video[0]
        try:
            stream.thread_type = "AUTO"
        except Exception:
            pass

        rotation_checked = False
        rotation_code = None
        first_pts = None
        first_time = None
        last_second = 0
        sample_second = -1
        sample_slot = -1

        for packet in container.demux(stream):
            try:
                frames = packet.decode()
            except av.error.FFmpegError:
                continue

            for frame in frames:
                if not rotation_checked:
                    rotation_code = get_rotation_code(stream, frame)
                    rotation_checked = True

                if frame.pts is not None:
                    if first_pts is None:
                        first_pts = frame.pts
                    rel = (frame.pts - first_pts) * stream.time_base
                elif frame.time is not None:
                    if first_time is None:
                        first_time = float(frame.time)
                    rel = float(frame.time) - first_time
                else:
                    continue

                second = max(0, math.floor(rel))
                second = max(second, last_second)
                last_second = second

                fraction = max(0.0, min(0.999999, float(rel) - second))
                if second != sample_second:
                    sample_second = second
                    sample_slot = -1

                slot = int(fraction * sample_fps)
                if slot <= sample_slot:
                    continue
                sample_slot = slot

                image = frame.to_ndarray(format="bgr24")
                if rotation_code is not None:
                    image = cv2.rotate(image, rotation_code)
                yield second, image
    finally:
        container.close()

def aggregate_second(frame_detections):
    if not frame_detections:
        return [], 0

    class_frame_counts = {name: 0 for name in TARGET_CLASSES}
    people_counts = []

    for detections in frame_detections:
        people_counts.append(detections.get("person", 0))
        for name in detections:
            class_frame_counts[name] += 1

    n = len(frame_detections)
    objects = sorted(
        name for name, count in class_frame_counts.items() if 2 * count >= n
    )

    people_counts.sort()
    people = (people_counts[(n - 1) // 2] + people_counts[n // 2]) // 2
    return objects, people


def run_inference(model, frames, quantize):
    results = model(
        frames,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        classes=TARGET_INDICES,
        quantize=quantize,
        verbose=False,
        stream=False,
    )
    if len(results) != len(frames):
        raise RuntimeError("YOLO returned an unexpected number of frame results")

    detections = []
    for result in results:
        counts = {}
        if result.boxes is not None and len(result.boxes):
            for index in result.boxes.cls.int().tolist():
                name = COCO_IDX_TO_NAME.get(index)
                if name is not None:
                    counts[name] = counts.get(name, 0) + 1
        detections.append(counts)

    return detections


def error_text(exc):
    return str(exc).replace("\n", " ")[:200] or type(exc).__name__


def decode_worker(video_bytes, frame_queue, stop_event):
    def enqueue(item):
        while not stop_event.is_set():
            try:
                frame_queue.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    last_second = -1
    error = None

    try:
        for second, frame in decode_frames(video_bytes, SAMPLE_FPS):
            if not enqueue(("frame", second, frame)):
                return
            last_second = second

        if last_second < 0:
            error = "no frames decoded"
    except Exception as exc:
        error = error_text(exc)

    enqueue(("done", last_second, error))


def process_video(model, video_bytes, video_id, output, batch_size, quantize):
    def write_error(message):
        output.write(json.dumps({"video": video_id, "error": message}) + "\n")

    if not video_bytes:
        write_error("empty or missing video bytes")
        return

    buckets = {}
    next_second = 0
    frames = []
    seconds = []

    def emit_before(stop_second):
        nonlocal next_second
        while next_second < stop_second:
            objects, people = aggregate_second(buckets.pop(next_second, []))
            row = {
                "video": video_id,
                "second": next_second,
                "objects": objects,
                "people": people,
            }
            output.write(json.dumps(row) + "\n")
            next_second += 1

    def infer_pending():
        if not frames:
            return

        # Stage all results before committing any part of this batch.
        detections = run_inference(model, frames, quantize)
        for second, counts in zip(seconds, detections):
            buckets.setdefault(second, []).append(counts)
        frames.clear()
        seconds.clear()

    frame_queue = queue.Queue(maxsize=batch_size * 4)
    stop_event = threading.Event()
    worker = threading.Thread(
        target=decode_worker,
        args=(video_bytes, frame_queue, stop_event),
        daemon=True,
    )
    worker.start()

    try:
        while True:
            kind, second, value = frame_queue.get()

            if kind == "frame":
                frames.append(value)
                seconds.append(second)
                if len(frames) < batch_size:
                    continue

            # Run a full batch, or flush the final partial batch on "done".
            try:
                infer_pending()
            except Exception as exc:
                stop_event.set()
                # The first pending frame can share a second with an earlier
                # successful batch. Only earlier seconds are complete.
                emit_before(seconds[0])
                write_error(error_text(exc))
                break

            if kind == "frame":
                emit_before(second)
                continue

            # The decoder sends its last sampled second and an optional error.
            if value is None:
                emit_before(second + 1)
            else:
                emit_before(second)
                write_error(value)
            break
    finally:
        stop_event.set()
        worker.join(timeout=5.0)

    output.flush()


def pick_batch_size(device):
    if device == "cpu":
        return 8

    try:
        vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        return 8

    if vram_gib >= 20:
        return 32
    if vram_gib >= 12:
        return 16
    if vram_gib >= 6:
        return 8
    return 4


def main():
    parser = argparse.ArgumentParser(
        description="Index objects in video stored in Parquet files using YOLO11l."
    )
    parser.add_argument("--input", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()

    if not args.input.is_dir():
        parser.error(f"Input directory does not exist: {args.input}")

    parquet_files = sorted(args.input.glob("*.parquet"))
    if not parquet_files:
        parser.error(f"No .parquet files found in {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantize = 16 if device == "cuda" else None
    batch_size = pick_batch_size(device)

    print(f"Found {len(parquet_files)} Parquet file(s)")
    print(f"Loading {MODEL_PATH}")
    print(f"Device: {device} | Batch size: {batch_size} | Sample FPS: {SAMPLE_FPS}")

    model = YOLO(str(MODEL_PATH))
    model.to(device)

    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    model(
        [dummy],
        imgsz=IMGSZ,
        conf=CONF,
        classes=TARGET_INDICES,
        quantize=quantize,
        verbose=False,
    )

    total_videos = 0
    file_errors = 0

    with args.output.open("w", encoding="utf-8", buffering=1) as output:
        for number, path in enumerate(parquet_files, 1):
            print(f"[{number}/{len(parquet_files)}] {path.name}")
            try:
                parquet = pq.ParquetFile(path)
                row_index = 0

                for batch in parquet.iter_batches(batch_size=10, columns=["mp4"]):
                    column = batch.column(0)
                    for item in column:
                        video_id = f"{path.name}:{row_index}"
                        process_video(
                            model,
                            item.as_py(),
                            video_id,
                            output,
                            batch_size,
                            quantize,
                        )
                        row_index += 1
                        total_videos += 1
            except Exception as exc:
                print(f"Error processing {path.name}: {exc}", file=sys.stderr)
                file_errors += 1

    print(f"Processed {total_videos} video(s), {file_errors} file-level error(s).")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()