#!/usr/bin/env python3
"""Render the ten final charts from packaged data. No inference or live system access."""
from pathlib import Path
import argparse,json,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
REPO=Path(__file__).resolve().parent.parent
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('rendered-charts'));args=parser.parse_args()
O=args.out;O.mkdir(parents=True,exist_ok=True)
data=json.loads((REPO/'reports/final/chart_data.json').read_text())
raw=data['raw_prefix'];managed=data['managed_prefix'];memory=data['memory'];state=data['promotion'];sweep=data['historical_30k']
baseline={'phase':data['historical_mtp']['baseline']};mtp={'phase':data['historical_mtp']['mtp2']}
raw_boundary=data['raw_boundary'];managed_boundary=data['managed_boundary'];integration=data['opencode']
command=(REPO/'reports/final/production-prefix-command.sh').read_text()
setup_text,serve_text=command.split('exec ',1);serve_text='exec '+serve_text
BG = '#f7f7f5'; INK = '#171717'; GRAY = '#606060'; MUTED = '#929292'; GRID = '#d7d7d4'
plt.rcParams.update({'font.family': 'DejaVu Sans Mono', 'font.size': 12, 'text.color': INK,
                     'axes.facecolor': BG, 'figure.facecolor': BG, 'savefig.facecolor': BG,
                     'axes.edgecolor': GRID, 'pdf.fonttype': 42})
pdf = PdfPages(O / 'Qwen38_Final_Production_Atlas.pdf'); figures = []
def page(title, subtitle, height=9):
    f = plt.figure(figsize=(16, height), dpi=150)
    f.text(.03, 1-.48/height, title, fontsize=22, weight='bold', va='top')
    f.text(.03, 1-.94/height, subtitle, fontsize=10.5, color=GRAY, va='top')
    return f
def save(f, name, foot):
    f.text(.03, .025, foot, fontsize=8, color=MUTED)
    for ext in ('png', 'svg'):
        f.savefig(O / (name + '.' + ext), dpi=150)
    pdf.savefig(f); plt.close(f); figures.append((name, foot))
def rows(f, items, y=.80, step=.095, size=11):
    for title, body in items:
        f.text(.035, y, title, fontsize=size, weight='bold', va='top')
        f.text(.235, y, body, fontsize=size, color=GRAY, va='top', linespacing=1.45)
        y -= step
def axes_style(ax):
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    for k in ('top', 'right', 'left'):
        ax.spines[k].set_visible(False)

f = page('Qwen3.8 Flash Next: final DGX production', f"Verified {state['time']} MDT · original NVIDIA checkpoint · prefix caching ON")
rows(f, [
    ('Engine', 'vLLM 0.28.1rc1.dev442+g7fbd44cbe.d20260905\nbase 7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719 + documented local patches'),
    ('Runtime', 'PyTorch 2.13.0 · FlashInfer 0.6.18 · Transformers 5.16.1\nPython 3.12 · driver 580.173.02 / CUDA 13.0 · kernel 6.17.0-1031-nvidia'),
    ('Hardware', 'One GB10 · 20 ARM CPU threads · 121.6 GiB OS-visible unified memory\nTP1 · eager execution · original PLE FP8 mmap'),
    ('Geometry', '262,144 context · 131,072 output ceiling within remaining context\n8 GiB BF16 KV · batch 2,048 · one sequence · MTP2'),
    ('Architecture', '125B main / 6B active + 51B n-gram embeddings + 4B MTP (NVIDIA card)\n48 layers: 36 Gated DeltaNet + 12 sparse attention; 512 experts / 10 selected'),
    ('Cache', 'Automatic prefix reuse ON · recurrent checkpoint grid 1,600 tokens\nalign mode · retention interval 1,600 · separate QSA scratch ring 8'),
    ('OpenCode', 'Exact served ID: nvidia/Qwen3.8-Flash-Next-NVFP4 · port 8092\nThinking medium; low/xhigh available · one image, video disabled'),
], step=.098)
save(f, '01_Final_Runtime_and_Model', 'Final live configuration · no checkpoint conversion · original NVIDIA mixed NVFP4 / FP8 / BF16 preserved')

f = page('MTP2: the measured selection behind production', 'Historical cache-OFF comparison · 47,643 input · identical request and sampling · 131,072 configured context')
for pos, key, title, limit in [([.14,.24,.32,.52], 'decode_tps', 'Decode tokens / s', 34), ([.64,.24,.30,.52], 'prefill_tps', 'Prefill tokens / s', 2800)]:
    ax = f.add_axes(pos); vals = [baseline['phase'][key], mtp['phase'][key]]
    ax.barh([1,0], vals, color=[GRAY, INK], height=.45); ax.set_yticks([1,0], ['No MTP', 'MTP2'])
    ax.set_xlim(0,limit); ax.set_title(title, loc='left', pad=24); axes_style(ax)
    for y,v in zip([1,0], vals): ax.text(v+limit*.025,y,f'{v:,.2f}',va='center',weight='bold')
f.text(.14,.15,'Decode +62.2%  |  Prefill −17.8%',fontsize=17,weight='bold')
f.text(.14,.09,'One run per setting; includes reasoning. Historical selection evidence, not a new APC decode test.',fontsize=10,color=GRAY)
save(f, '02_MTP2_Selection_Historical', 'Retained measurements, freshly rendered · no MTP3 result claimed · outputs differ, request dictionaries match')

f = page('Prefix cache: 250K prefill in about 1.5 seconds', 'Measured cold / warm pairs · thinking medium · exact answers verified · prefill timing, not decode speed')
ax=f.add_axes([.17,.24,.68,.54]); pairs=[]
for i,t in enumerate((20000,64000,250000)):
    c=next(x for x in raw if x['label']==f'{t}-cold'); w=next(x for x in raw if x['label']==f'{t}-warm'); pairs.append((t,c,w))
    for off,d,col in ((-.17,c,GRAY),(.17,w,INK)):
        v=d['prefill_seconds'];ax.barh(i+off,v,height=.27,color=col);ax.text(v+1.5,i+off,f'{v:.2f}s',va='center',fontsize=11)
ax.set_yticks([0,1,2],['20K input','64K input','250K input']);ax.invert_yaxis();ax.set_xlim(0,190);axes_style(ax)
ax.set_xlabel('Server prefill seconds · gray: uncached · black: prefix reused')
f.text(.035,.145,'Reused tokens: '+ ' / '.join(f"{w['cache_hit_tokens']:,.0f}" for _,c,w in pairs),fontsize=14,weight='bold')
f.text(.035,.085,'Cold controls use unique cache_salt namespaces. First cold request includes first-use effects.\nEarlier-prefix edits and eviction can still cause misses; speedup applies to these repeated prompts.',fontsize=10,color=GRAY)
save(f, '03_Prefix_Cache_Cold_vs_Warm', 'Single sequential samples · raw API proof · same 8 GiB allocation · no extrapolated throughput claims')

f = page('250K needles: cold, warm, changed, and restored', 'Three audited keys embedded at roughly 6%, 50%, and 94% of the background records')
ax=f.add_axes([.08,.66,.84,.11]);ax.hlines(0,0,100,color=GRID,linewidth=10)
for pos,label in [(6.25,'north'),(50,'central'),(93.75,'south')]:
    ax.plot(pos,0,'o',color=INK,markersize=10);ax.text(pos,.18,label,ha='center',fontsize=11)
ax.set_xlim(-3,103);ax.set_ylim(-.3,.45);ax.axis('off')
checks=[x for x in raw if x['label'].startswith('250000-')]
table=[['Condition','Input tokens','Cache hit tokens','Answer']]
for x in checks:table.append([x['label'].replace('250000-',''),f"{x['usage']['prompt_tokens']:,}",f"{x['cache_hit_tokens']:,.0f}",'PASS' if x['answer_pass'] else 'FAIL'])
ax=f.add_axes([.07,.27,.86,.34]);ax.axis('off');tab=ax.table(cellText=table,loc='center',cellLoc='left',colWidths=[.40,.2,.23,.17]);tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,2.0)
for (i,j),cell in tab.get_celld().items():cell.set_facecolor('#e7e7e3' if i==0 else BG);cell.set_edgecolor(GRID)
f.text(.035,.16,f"Near-full boundary also passed: raw {raw_boundary['usage']['prompt_tokens']:,}, managed {managed_boundary['usage']['prompt_tokens']:,} input tokens.",fontsize=13,weight='bold')
f.text(.035,.09,'Suffix instruction asks for north/south only, then returns to all three.\nThis proves these retrieval cases; it is not exhaustive long-context reasoning or a multi-needle benchmark score.',fontsize=10,color=GRAY)
save(f, '04_250K_Needle_Qualification', 'Five 250K request variants plus separate raw/managed near-full context checks · thinking remains enabled')

f=page('Memory: prefix reuse inside the same 8 GiB pool','Unified-memory snapshots overlap; do not add GPU-visible and system-used measurements')
vals=[memory['total_gib'],memory['used_gib'],memory['engine_gib']]
ax=f.add_axes([.30,.56,.59,.20]);ax.barh([2,1,0],vals,color=['#d7d7d4',GRAY,INK],height=.55)
ax.set_yticks([2,1,0],['OS-visible total','Post-test total used','Engine GPU-visible']);ax.set_xlim(0,138);axes_style(ax)
for y,v in zip([2,1,0],vals):ax.text(v+1,y,f'{v:.1f} GiB',va='center')
ax.set_xlabel('GiB · overlapping accounting domains')
rows(f,[('KV allocation','8 GiB BF16 · 282,420 advertised cache tokens · max_num_seqs 1\n1.08× capacity ratio is not tested concurrency'),('Checkpoints','Retention every 1,600 tokens stays inside the preallocated pool.\nDenser retention changes cache eviction competition; cold requests still cost prefill.'),('PLE / pressure',f"47.684 GiB original FP8 PLE on disk; selected rows gathered through mmap.\nPost-test swap {memory['swap_percent']:.1f}%; memory headroom remains tight at full context.")],y=.41,step=.115)
save(f,'05_Memory_and_Cache',f"Live post-test nvitop/Python snapshot {memory['time']} · file-backed PLE size is not resident-memory size")

f=page('Source fixes: what makes this deployment work','Original weights retained · official upstream sources for cache patches · open PRs are still local backports',height=10)
rows(f,[('PLE mmap / stride','Preserved eager-TP1 original-FP8 mmap reader. PR55375 handles\nnon-contiguous state-index strides during speculative prefill/decode.'),('MTP ModelOpt','Remap draft-local layer0 to runtime layer48; preserve other mappings.\nDispatch FP8_BLOCK_SCALES experts to block-FP8 MoE; main NVFP4 stays intact.'),('PR53798','Bind actual MambaSpec before the first request; restore recurrent state\nusing its 1,600-token block size, not the generic attention block size.'),('PR54076','Materialize each crossed recurrent boundary. Local adaptation unwraps\nUniformTypeKVCacheSpecs for actual GDN and PLE groups; regression added.'),('PR55390 / 54713','Annotate the hybrid MTP group, then retain the lower checkpoint reachable\nafter speculative lookup drops a block. Warning alone did not prove in-GPU misses.'),('Retention policy','--mamba-cache-mode align --prefix-cache-retention-interval 1600\nFixes the measured shorter-branch miss left by sparse retention0.'),('Client integration','Output131072; max_tokens null; thinking medium/low/xhigh.\nPreserve 1,200,000 ms general/header/chunk timeouts and provider options.'),('Validation','201 core/worker +186 cache/scheduler tests passed; source pre-commit passed.\nTwo PP2 cases unavailable on one GPU. Earlier16 engine +2 wrapper tests retained.')],y=.84,step=.095,size=10.5)
save(f,'06_Fixes_and_Provenance','Official vllm-project/vllm PR53798,54076,55390,54713 were open when fetched · exact patches and hashes included')

f=page('Final production: complete launch command','Captured managed argv and allowlisted environment · all paths anonymized',height=9)
f.text(.035,.86,'ENVIRONMENT',fontsize=12,weight='bold');f.text(.46,.86,'SERVER',fontsize=12,weight='bold')
f.text(.035,.825,setup_text,fontsize=8.8,va='top',linespacing=1.45)
f.text(.46,.825,serve_text,fontsize=8.8,va='top',linespacing=1.45)
f.text(.035,.37,'Existing production endpoint: port8092\n\nRun the environment block first.\nDo not start a second large engine.\nUse the wrapper lifecycle for normal restarts.\n\nRaw qualified command also included.\nMAX_JOBS=1 and PLE mmap stay enabled.',fontsize=10,color=GRAY,linespacing=1.6)
save(f,'07_Final_Production_Command','Editable shell files included · command is captured configuration, not an instruction to start another concurrent model')

f=page('Validation: raw engine through actual OpenCode','Passed before final report generation · no inference submitted merely to redraw charts')
rows(f,[('Direct prefix suite',f'{len(raw)} requests:20K/64K/250K cold, repeat, changed suffix, return;\ninterleaved A/B/A, automatic inventory tool call and tool-result continuation.'),('State / boundary','Observed GPU state seeds match computed_tokens /1600 −1.\nRaw and managed260,834-token three-needle retrieval passed.'),('Managed engine',f'{len(managed)} cache requests passed; actual argv matches wrapper command.\nThinking, retrieval, tools, tool result, vision and prose passed.'),('TUI / OpenCode',f"Actual TUI warm action and streaming bash tool roundtrip passed.\nOpenCode aggregate cache hits: {integration['prefix_cache_hit_tokens']:,.0f} tokens over {integration['requests']} requests."),('Production settings','262,144 context;131,072 output ceiling; BF16 KV; MTP2; APC ON.\nThinking-only variants;20-minute transport timeouts; mirrors synchronized.'),('Limits','No extended soak, MTP3, concurrency>1, bitwise hidden-state proof,\nor131,072-token generated answer. Exact checks passed; broader claims are untested.')],step=.115,size=10.5)
save(f,'08_Production_Validation','OpenCode test uses a stable synthetic reference prefix and actual streamed tool execution · raw and managed proof remain distinct')

f=page('Decode through 30K: today’s measured baseline','Historical pre-APC sweep · 262,144 configured context · MTP2 · BF16 KV · medium thinking')
ax=f.add_axes([.16,.27,.70,.49]);values=[x['decode_tps'] for x in sweep]
ax.barh(range(len(sweep)),values,color=INK,height=.52);ax.set_yticks(range(len(sweep)),[f"{x['prompt_tokens']:,} input" for x in sweep]);ax.invert_yaxis();ax.set_xlim(0,40);axes_style(ax)
for i,v in enumerate(values):ax.text(v+.4,i,f'{v:.2f} t/s',va='center')
ax.set_xlabel('Decode tokens per second · excludes first token · includes reasoning')
f.text(.035,.16,'One warmed 256-token sample per length; ignore_eos was enabled.',fontsize=13,weight='bold')
f.text(.035,.095,'These are retained cache-OFF measurements, not a fresh prefix-enabled speed sweep.\nFixed-length continuation can affect speculative acceptance; this is not an answer-quality benchmark.',fontsize=10,color=GRAY)
save(f,'09_30K_Decode_Historical','Cold30K prefill measured14.62s before APC · current prefix savings are measured separately in figure03')
aa = json.loads((REPO/'provenance/aa-one-effort.json').read_text())
f=page('Artificial Analysis: where the model sits', 'Intelligence Index v4.2 | leading model families | highest score only | one entry per model',height=13)
ax=f.add_axes([.39,.19,.52,.69]);labels=[]
for i,row in enumerate(aa['rows']):
 name=row['name'];effort=re.search(r'\(([^)]+)\)',name)
 setting=effort.group(1) if effort else 'unspecified'
 setting=setting.replace('Adaptive Reasoning, ','').replace(' Effort','').replace(', Default Fallback','*').replace(', Opus 4.8 Fallback','*')
 labels.append(row['model'])
 ax.barh(i,row['score'],height=.62,color=INK if row['slug']=='qwen3-8-flash-next' else '#929292')
 ax.text(row['score']+.5,i,f"{row['score']:.1f}",va='center',fontsize=10)
ax.set_yticks(range(len(labels)),labels,fontsize=9.5);ax.invert_yaxis();ax.set_xlim(0,64);axes_style(ax);ax.set_xlabel('AA Intelligence Index | higher is better')
f.text(.035,.125,'Qwen3.8-Flash-Next: 45.6',fontsize=12,weight='bold')
f.text(.035,.075,'Highest-scoring result per family across effort settings and older versions.\nSelected effort and evaluation settings are recorded in the source data.',fontsize=9,color=GRAY)
save(f,'10_AA_Intelligence_Index','Artificial Analysis authenticated API | retrieved '+aa['retrieved_at']+' | Artificial Analysis model evaluation')
pdf.close()
print("Rendered",len(figures),"charts in",O)
