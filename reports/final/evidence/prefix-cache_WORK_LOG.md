# Prefix-cache repair work log : September5

Baseline: production262K, BF16 KV8GiB, MTP2, eagerTP1, batch2048,seq1,
thinking medium, originalNVFP4/FP8/BF16 checkpoint, prefix off.

Candidate sources are official vllm-project/vllm PRs, all OPEN/unmerged:
53798 at243dfa6df183eafcc092f9812ebcbd9da2d3b98e: initialize Mamba cache spec before
requests and seed resumed state column in its actual block units.
54076 at53930fe85c9e5dd78ec8cb7129c0210da40c843f: align scheduled prefill chunks to
Mamba state grid and materialize every crossed state boundary.
55390 at28bc017bbe359f4c17793ee8a103c5d623c8f75f: identify trailing MTP draft group.

Correction to earlier research:55390 explicitly scopes the warning's zero-hit
claim to external offload. In-GPU MambaManager ignores drop_eagle_block; the
warning alone does not prove in-GPU cache hits are impossible. Actual state
indexing and chunk alignment are the first concrete local blockers to repair.

All patch files, GitHub metadata, and original affected files saved here.
No checkpoint changes. Source changes do not alter the already-loaded server.
Regression testing precedes a direct-engine candidate launch and real cold/warm
correctness, state-index, counter and latency qualification.

Candidate adaptation and tests:
-53798/54076/55390 applied cleanly; preserved original files and patch provenance.
-54076 scheduler grid extraction now unwraps UniformTypeKVCacheSpecs through
  iter_layer_specs; added regression for this mixed-state grouping.
-Updated upstream test stubs for current use_eagle_block_drop, and changed the
  old two-boundary expectation to explicitly check consecutive state stops.
-Opt-in QWEN_PREFIX_AUDIT logs group geometry and actual GPU state seed at resume;
  default off, no semantics change.
-First suite: GPU OOM from second CUDA process beside full production; also
  missing root logging fixture and wrong HF cache permissions. No engine crash.
-Stopped idle production for authorized work. Reran with free GPU, intended
  writable HF_HOME and isolated standard logging fixture:201 passed; two PP2
  tests require more local GPUs and were deselected after their platform failure.
-Partial-prefix suite:44 passed. Ruff lint/format passed.
-Raw candidate launched17:02:52 on8099,262K,8GiB BF16,MTP2,enable-prefix-caching,
  mamba-cache-mode align. No provider/profile mutation before qualification.

First raw candidate evidence (attempt1/): startup passed with282420-token
capacity from8GiB. Live groups use1600-token recurrent blocks and8-token QSA
ring; UniformType unwrapping was necessary. Cold19984-token answer passed;
repeat answer passed but0 hit tokens and10.063s prefill: caching FAIL.
Development reset endpoint404 caused an earlier harness stop; switched to
unique cache_salt namespaces for cold controls without enabling dev endpoints.

Added official PR54713 atd3bdc5f7cc6011f0ad30b80b417aa6cdbb4217ab: retain the
state one scheduler alignment unit below the reachable boundary whenever any
group drops MTP lookahead, even if the recurrent group itself is not draft.
Local patch context preserves the newer logger and dcp_world_size argument.
186 cache/scheduler tests passed; all precommit passed. Restarted direct raw
candidate with all four fixes; retained first attempt's logs/results.

Second candidate: real hits verified (20K:17600,64K:60800), all answers correct. Sparse retention0 still misses when a new cold changed suffix crosses the64000 boundary and the next query returns to63989 tokens. Preserved attempt2. Selecting official CLI prefix-cache-retention-interval1600 (one observed scheduler block, dense recurrent checkpoints within the same fixed8GiB pool) for robust branching; no additional source changes. This is a retention-policy miss, not answer corruption.

Third candidate retained checkpoints1600:20K and64K full sequences PASS, including previously missing63989-after64008 branch (60800 hits,1.728s prefill).250K cold needle retrieval PASS249986input/211output/164reasoning,157.903s prefill. Warm PASS248000hits/1.450s prefill/1.682sTTFT. Changed250005-input suffix also PASS248000hits/1.459s. Remaining checks running. User explicitly requested250K needles while this suite was active; confirmed three depths plus near-full boundary.

Full260834-token raw retrieval answered all3 keys correctly in163.306s (224completion/163reasoning). Harness falsely failed because leading whitespace preceded a supported JSON fence. Preserved original result; fixed normalization, revalidated saved response, recorded exact candidate PID2668226. Promotion resumes from that proof without replaying the expensive request.
