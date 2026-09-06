# Copyable examples

- `portable-check-results.json`: live PASS from the standalone public checker.
- `chat-request.json`: small thinking-enabled OpenAI-compatible request.
- `needle-250k-request.json`: the synthetic249,986-token request used in the
  raw qualification. Send it only when the 262K server is idle.
- `needle-250k-cold.json` and `needle-250k-warm.json`: measured usage, timing,
  cache counters and the final answer. Generated reasoning text is omitted.
- `inventory-tool-request.json`: synthetic inventory task and tool schema.
- `inventory-tool-result.json`: the correct generated tool arguments.
- `opencode-tool-result.json`: scoped integration result, including cache hits.

```bash
curl -fsS http://127.0.0.1:8092/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @examples/needle-250k-request.json
```

To reproduce the measured cold/warm procedure, use `scripts/check-prefix.py`.
It creates a new hash namespace for the cold control and saves both responses.
Simply sending an archived request may hit existing cache, so it is not by
itself a controlled cold measurement.

The tool example describes an inventory submission. It does not connect to an
inventory system or execute an external action. OpenCode's integration test ran
only `printf QWEN38_TOOL_OK`, then checked the returned final marker.
