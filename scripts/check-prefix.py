#!/usr/bin/env python3
"""Copyable three-needle cold/warm check using only the Python standard library."""
import argparse
import json
from pathlib import Path
import re
import time
import urllib.request
import uuid

p = argparse.ArgumentParser()
p.add_argument('--base-url', default='http://127.0.0.1:8092')
p.add_argument('--tokens', type=int, default=20000)
p.add_argument('--out', type=Path)
a = p.parse_args()
base = a.base_url.rstrip('/')
model = 'nvidia/Qwen3.8-Flash-Next-NVFP4'
keys = {'north': 'NORTH_COPPER_7813', 'central': 'CENTRAL_MAPLE_2946', 'south': 'SOUTH_QUARTZ_6502'}
out = a.out or Path('prefix-check-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
out.mkdir(parents=True, exist_ok=False)

def call(path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + path, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=1200) as response:
        return json.load(response)

def metrics():
    with urllib.request.urlopen(base + '/metrics', timeout=10) as response:
        return response.read().decode()

def total(text, name):
    pattern = re.compile(r'^' + re.escape(name) + r'(?:\{[^}]*\})?\s+([-+0-9.eE]+)$', re.M)
    return sum(float(m) for m in pattern.findall(text))

def payload(n):
    lines = [f'BACKGROUND record {i:05d}: part AX-{i*31+9}; shelf {i%23}; quantity {i%47+5}; lot B{i*19+4}; alloy steel.' for i in range(n)]
    for position, site in reversed(list(zip((n//16, n//2, n*15//16), keys))):
        lines.insert(position, f'AUDITED VAULT KEY for {site}: {keys[site]}.')
    return {'model': model, 'messages': [{'role': 'user', 'content': '\n'.join(lines) + '\nReturn only JSON with all three exact audited vault keys: north, central, south.'}],
            'temperature': 0, 'seed': 42, 'max_tokens': 1024,
            'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'medium'}}

catalog = call('/v1/models')
entry = next(x for x in catalog['data'] if x['id'] == model)
assert 5000 <= a.tokens <= entry['max_model_len'] - 2048, 'Choose a target from5000 to context minus2048'
lo, hi = 1, a.tokens // 20
selected = None
while lo <= hi:
    mid = (lo + hi) // 2
    request = payload(mid)
    count = call('/tokenize', request)['count']
    if count <= a.tokens:
        selected = request
        lo = mid + 1
    else:
        hi = mid - 1
assert selected
selected['cache_salt'] = uuid.uuid4().hex
(out / 'request.json').write_text(json.dumps(selected, indent=2))
results = []
for label in ('cold', 'warm'):
    before = metrics()
    assert total(before, 'vllm:num_requests_running') == 0 and total(before, 'vllm:num_requests_waiting') == 0, 'Run on an idle endpoint'
    started = time.monotonic()
    response = call('/v1/chat/completions', selected)
    wall = time.monotonic() - started
    for _ in range(15):
        after = metrics()
        generated = total(after, 'vllm:request_generation_tokens_sum') - total(before, 'vllm:request_generation_tokens_sum')
        if generated >= response['usage']['completion_tokens']:
            break
        time.sleep(1)
    assert generated == response['usage']['completion_tokens'], 'Counter isolation failed'
    content = (response['choices'][0]['message'].get('content') or '').strip()
    content = content.removeprefix('```json').removesuffix('```').strip()
    answer_ok = json.loads(content) == keys
    def delta(name):
        return total(after, name) - total(before, name)
    record = {'condition': label, 'usage': response['usage'], 'wall_seconds': wall,
              'prefill_seconds': delta('vllm:request_prefill_time_seconds_sum'),
              'ttft_seconds': delta('vllm:time_to_first_token_seconds_sum'),
              'cache_hit_tokens': delta('vllm:prefix_cache_hits_total'), 'answer_pass': answer_ok}
    results.append(record)
    (out / f'{label}-response.json').write_text(json.dumps(response, indent=2))
    (out / f'{label}-metrics-before.txt').write_text(before)
    (out / f'{label}-metrics-after.txt').write_text(after)
    (out / 'results.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(record), flush=True)
    assert answer_ok, 'Incorrect needle answer'
    assert record['cache_hit_tokens'] == 0 if label == 'cold' else record['cache_hit_tokens'] > response['usage']['prompt_tokens'] * .5, 'Unexpected cache reuse'
print('PASS: correct cold/warm retrieval and positive prefix reuse. Artifacts:', out)
