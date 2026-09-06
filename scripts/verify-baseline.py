#!/usr/bin/env python3
"""Verify the recipe, patched source, or original model without loading weights."""
import argparse
import hashlib
import json
from pathlib import Path

repo = Path(__file__).resolve().parent.parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path)
p.add_argument('--model', type=Path)
a = p.parse_args()

def verify(root, manifest):
    entries = json.loads(manifest.read_text())
    for name, entry in entries.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise SystemExit(f'Path escapes verification root: {name}')
        expected = entry if isinstance(entry, str) else entry['sha256']
        if isinstance(entry, dict) and path.stat().st_size != entry['size']:
            raise SystemExit(f'Size mismatch: {name}')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise SystemExit(f'SHA-256 mismatch: {name}')
    print(f'PASS: {len(entries)} files matched {manifest.name}', flush=True)

verify(repo, repo / 'baseline/recipe-sha256.json')
if a.source:
    verify(a.source, repo / 'patches/patched-source-sha256.json')
if a.model:
    verify(a.model, repo / 'baseline/model-sha256.json')
