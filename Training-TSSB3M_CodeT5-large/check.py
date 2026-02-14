import os
from pathlib import Path

extract_dir = Path("tssb_data/extracted")
print(f"Exists: {extract_dir.exists()}")

for root, dirs, files in os.walk(extract_dir):
    for file in files:
        if file.endswith('.jsonl.gz'):
            print(Path(root) / file)