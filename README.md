# Video Indexing at Speed

## System packages

```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
```

## Install

```bash
pip install -r requirements.txt
```

## Prepare (run once, has internet access, not timed)

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11l.pt')"
```

## Run

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

