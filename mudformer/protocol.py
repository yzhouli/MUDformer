"""User-stratified 8:1:1 splitting and checkpoint protocol identity."""
from pathlib import Path
import hashlib
import json
import random

RATIOS = (0.8, 0.1, 0.1)
NAMES = ('train', 'validation', 'test')

def records(path):
    data = json.loads(Path(path).read_text())
    return data['records'] if isinstance(data, dict) else data

def stratified_split(rows, seed=42):
    ids = [str(r['user_id']) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate user IDs')
    if set(r['label'] for r in rows) != {0, 1}:
        raise ValueError('Expected binary user labels 0 and 1')
    sizes = [sum(r['label'] == label for r in rows) for label in (0, 1)]
    if min(sizes) < 10:
        raise ValueError('Each label requires at least 10 users for an 8:1:1 split')
    target = [int(len(rows) * ratio) for ratio in RATIOS]
    order = sorted(range(3), key=lambda i: (-(len(rows)*RATIOS[i]-target[i]), i))
    for i in order[:len(rows)-sum(target)]:
        target[i] += 1
    allocation = [[int(n*ratio) for ratio in RATIOS] for n in sizes]
    while sum(map(sum, allocation)) < len(rows):
        candidates = [(sizes[label]*RATIOS[i]-allocation[label][i], -label, -i, label, i)
                      for label in (0,1) for i in range(3)
                      if sum(allocation[label]) < sizes[label]
                      and sum(row[i] for row in allocation) < target[i]]
        _, _, _, label, i = max(candidates)
        allocation[label][i] += 1
    result = {name: [] for name in NAMES}
    for label in (0, 1):
        group = sorted((r for r in rows if r['label'] == label), key=lambda r: str(r['user_id']))
        random.Random(f'{seed}:{label}').shuffle(group)
        start = 0
        for name, count in zip(NAMES, allocation[label]):
            result[name].extend(group[start:start+count]); start += count
    return {k: sorted(v, key=lambda r: str(r['user_id'])) for k,v in result.items()}

def validate_splits(train, validation, test):
    groups = [records(p) for p in (train, validation, test)]
    seen = set()
    for name, group in zip(NAMES, groups):
        ids = [str(r['user_id']) for r in group]
        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError(f'Empty, duplicated or overlapping users in {name}')
        if set(r['label'] for r in group) != {0, 1}:
            raise ValueError(f'Both classes are required in {name}')
        seen.update(ids)
    # Integer class-wise rounding can change each split by at most two users.
    for label in (0, 1):
        counts = [sum(r['label'] == label for r in g) for g in groups]
        n = sum(counts)
        if any(abs(c - n * ratio) > 1.01 for c, ratio in zip(counts, RATIOS)):
            raise ValueError('Manifests do not implement a stratified 8:1:1 split')
    payload = {name: sorted(group, key=lambda r: str(r['user_id'])) for name,group in zip(NAMES, groups)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def check_resume(checkpoint, identity):
    if checkpoint.get('split_identity') != identity:
        raise ValueError('Checkpoint does not match this 8:1:1 split; use a new output directory. Bundled inference weights are not resume states.')
