#!/usr/bin/env python3
"""Quality and speed harness for the OpenAI-compatible vLLM endpoint. Stdlib only.

nll   : prompt NLL on fixed prose/code at several lengths, run twice (determinism)
gsm   : GSM8K first N, greedy, thinking off, exact numeric match
greedy: greedy completions of fixed prompts; saved for cross-config comparison
speed : streaming decode/prefill tok/s at given prompt lengths, greedy + sampled
"""
import argparse, json, re, time, urllib.request, uuid, hashlib
from pathlib import Path

HERE = Path(__file__).parent
MODEL = 'nvidia/Qwen3.8-Flash-Next-NVFP4'


def call(base, path, payload=None, timeout=1800):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def metric_totals(base):
    with urllib.request.urlopen(base + '/metrics', timeout=10) as r:
        text = r.read().decode()
    def tot(name):
        pat = re.compile(r'^' + re.escape(name) + r'(?:\{[^}]*\})?\s+([-+0-9.eE]+)$', re.M)
        return sum(float(m) for m in pat.findall(text))
    return {'drafts': tot('vllm:spec_decode_num_drafts_total'),
            'draft_tokens': tot('vllm:spec_decode_num_draft_tokens_total'),
            'accepted': tot('vllm:spec_decode_num_accepted_tokens_total')}


def tokenize(base, text):
    return call(base, '/tokenize', {'model': MODEL, 'prompt': text, 'add_special_tokens': False})['tokens']


def detok(base, toks):
    return call(base, '/detokenize', {'model': MODEL, 'tokens': toks})['prompt']


def nll(base, lengths, reps):
    out = {}
    for name in ('prose', 'code'):
        toks = tokenize(base, (HERE / f'{name}.txt').read_text())
        for n in lengths:
            if n > len(toks):
                continue
            prompt = detok(base, toks[:n])
            runs = []
            for r in range(reps):
                t = time.time()
                d = call(base, '/v1/completions', {'model': MODEL, 'prompt': prompt, 'max_tokens': 1,
                         'temperature': 0, 'prompt_logprobs': 0, 'cache_salt': uuid.uuid4().hex})
                lps = [list(x.values())[0]['logprob'] for x in d['choices'][0]['prompt_logprobs'] if x]
                # the scored token is the prompt token; pick matching entry when top-1 differs
                runs.append({'mean_nll': -sum(lps) / len(lps), 'n': len(lps), 'seconds': time.time() - t,
                             'hash': hashlib.sha1(json.dumps([round(x, 3) for x in lps]).encode()).hexdigest()[:12],
                             'lps': lps})
            key = f'{name}-{n}'
            diffs = [max(abs(a - b) for a, b in zip(runs[0]['lps'], x['lps'])) for x in runs[1:]]
            out[key] = {'mean_nll': [x['mean_nll'] for x in runs], 'n': runs[0]['n'],
                        'max_abs_lp_diff_between_reps': diffs, 'seconds': [x['seconds'] for x in runs]}
            print(key, json.dumps(out[key]), flush=True)
    return out


def num(s):
    m = re.findall(r'-?\d[\d,]*\.?\d*', s.replace('$', ''))
    return m[-1].replace(',', '').rstrip('.') if m else None


def gsm(base, n, effort):
    rows = [json.loads(l) for l in (HERE / 'gsm8k.jsonl').read_text().splitlines()[:n]]
    ok, recs, t0 = 0, [], time.time()
    kw = {'enable_thinking': False} if effort == 'off' else {'enable_thinking': True, 'reasoning_effort': effort}
    for i, r in enumerate(rows):
        gold = r['answer'].split('####')[-1].strip().replace(',', '')
        d = call(base, '/v1/chat/completions', {'model': MODEL, 'temperature': 0, 'seed': 0, 'max_tokens': 8192,
                 'messages': [{'role': 'user', 'content': r['question'] + '\nSolve step by step. End with "Answer: <number>".'}],
                 'chat_template_kwargs': kw})
        text = d['choices'][0]['message']['content'] or ''
        m = re.search(r'Answer:\s*\$?(-?[\d,]*\.?\d+)', text)
        pred = (m.group(1).replace(',', '') if m else num(text))
        good = pred is not None and float(pred) == float(gold)
        ok += good
        recs.append({'i': i, 'gold': gold, 'pred': pred, 'ok': good, 'tokens': d['usage']['completion_tokens'],
                     'sha': hashlib.sha1(text.encode()).hexdigest()[:12]})
    res = {'n': n, 'effort': effort, 'correct': ok, 'acc': ok / n, 'seconds': time.time() - t0, 'items': recs}
    print('gsm', effort, ok, '/', n, f'{res["seconds"]:.0f}s', flush=True)
    return res


GREEDY_PROMPTS = [
    'Explain how a gated DeltaNet layer updates its recurrent state, in about 300 words.',
    'Write a Python function that parses an ISO-8601 duration string into seconds, with tests.',
    'A press brake bends 6mm mild steel over a 50mm V-die. Estimate the tonnage per meter and show the formula.',
    'Summarize the plot of Pride and Prejudice in 10 bullet points.',
    'Write a bash script that watches a directory and rsyncs new files to a remote host, with logging.',
    'What are the tradeoffs between NVFP4 W4A4 and W4A16 quantization for MoE inference? Be precise.',
    'Translate into French and then back into English: "The lighthouse keeps its light even when no ship passes."',
    'Prove that the square root of 2 is irrational.',
]


def greedy(base, max_tokens):
    outs = []
    for p in GREEDY_PROMPTS:
        d = call(base, '/v1/chat/completions', {'model': MODEL, 'temperature': 0, 'seed': 0, 'max_tokens': max_tokens,
                 'messages': [{'role': 'user', 'content': p}], 'chat_template_kwargs': {'enable_thinking': False}})
        outs.append(d['choices'][0]['message']['content'])
    print('greedy', [hashlib.sha1(o.encode()).hexdigest()[:8] for o in outs], flush=True)
    return outs


def stream(base, payload):
    payload = dict(payload, stream=True, stream_options={'include_usage': True})
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    t0, first, usage = time.time(), None, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith('data:') or line == 'data: [DONE]':
                continue
            ch = json.loads(line[5:])
            if ch.get('usage'):
                usage = ch['usage']
            if first is None and ch.get('choices'):
                dl = ch['choices'][0].get('delta', {})
                if dl.get('content') or dl.get('reasoning_content') or dl.get('reasoning'):
                    first = time.time()
    end = time.time()
    return t0, first, end, usage


def speed(base, lengths, out_tokens, reps):
    toks = tokenize(base, (HERE / 'prose.txt').read_text())
    res = []
    modes = {'greedy': {'temperature': 0},
             'sampled': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20}}
    for n in lengths:
        for mode, samp in modes.items():
            for r in range(reps):
                text = detok(base, toks[:n]) if n > 64 else 'Tell me a long story about a lighthouse keeper.'
                before = metric_totals(base)
                t0, first, end, usage = stream(base, dict(samp, model=MODEL, max_tokens=out_tokens, min_tokens=out_tokens,
                    ignore_eos=True, cache_salt=uuid.uuid4().hex,
                    messages=[{'role': 'user', 'content': text + '\n\nContinue this text in the same style.'}],
                    chat_template_kwargs={'enable_thinking': False}))
                after = metric_totals(base)
                d = {k: after[k] - before[k] for k in after}
                ct, pt = usage['completion_tokens'], usage['prompt_tokens']
                rec = {'prompt_tokens': pt, 'mode': mode, 'completion_tokens': ct,
                       'ttft': first - t0, 'prefill_tps': pt / (first - t0),
                       'decode_tps': (ct - 1) / (end - first),
                       'mean_accept_len': (1 + d['accepted'] / d['drafts']) if d['drafts'] else None,
                       'draft_accept_rate': (d['accepted'] / d['draft_tokens']) if d['draft_tokens'] else None}
                res.append(rec)
                print('speed', json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:8092')
    ap.add_argument('--label', required=True)
    ap.add_argument('--tests', default='nll,gsm,greedy,speed')
    ap.add_argument('--nll-lengths', default='4096,32768,131072')
    ap.add_argument('--nll-reps', type=int, default=2)
    ap.add_argument('--gsm-n', type=int, default=100)
    ap.add_argument('--gsm-effort', default='off')
    ap.add_argument('--speed-lengths', default='32,30000')
    ap.add_argument('--speed-out', type=int, default=1024)
    ap.add_argument('--speed-reps', type=int, default=2)
    a = ap.parse_args()
    base = a.base.rstrip('/')
    info = {'label': a.label, 'time': time.strftime('%F %T'), 'version': call(base, '/version')}
    tests = a.tests.split(',')
    if 'nll' in tests:
        info['nll'] = nll(base, [int(x) for x in a.nll_lengths.split(',')], a.nll_reps)
    if 'gsm' in tests:
        info['gsm'] = gsm(base, a.gsm_n, a.gsm_effort)
    if 'greedy' in tests:
        info['greedy'] = greedy(base, 512)
    if 'speed' in tests:
        info['speed'] = speed(base, [int(x) for x in a.speed_lengths.split(',')], a.speed_out, a.speed_reps)
    p = HERE / 'results' / f'{a.label}.json'
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(info, indent=1))
    print('wrote', p)
