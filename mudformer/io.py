"""Read JSONL records without splitting Unicode separators inside JSON strings."""
from pathlib import Path
import json

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]
