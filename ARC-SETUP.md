# ARC Setup Notes (private fork)

How the working pipeline was configured on the Mac Mini (31 Oct 2026) to get
stages 10–12 producing real, topic-aligned experiment code and metrics.

## The problem this solves

Stage 10 (CODE_GENERATION) is the hard stage: weaker proxy models
(glm-5.3-flash, kimi-k3) either never produced topic-aligned code or
half-broke the run. Two further traps appeared even on a frontier model:

1. **Thinking eats the token budget.** Anthropic extended thinking counts
   against `max_tokens`. With `thinking=adaptive` + `effort=high`, a 16k
   budget was consumed entirely by thinking (15,999 thinking tokens, 0 text
   chars) — the pipeline then saw an "empty generation". Fixed in the
   adapter: thinking calls are streamed with a 64k `max_tokens` floor, and
   any call that stops at `max_tokens` with zero text is retried once with a
   doubled budget (cap 128k).
2. **Silent placeholder fallback.** A codegen client without a usable key
   used to silently produce the built-in placeholder experiment ("runs when
   LLM code generation fails"), which stage 12 then "ran" in 0.05s. Empty
   loopback keys now fail loudly; empty max_tokens responses are surfaced.

## Configuration

`config.yaml` is gitignored (it references key paths). Copy
`config.example.yaml`:

```bash
cp config.example.yaml config.yaml
```

Key settings (already set in the example):

- Primary LLM: Morpheus local proxy `http://127.0.0.1:8083/v1`,
  model `glm-5.3-flash`, fallback `deepseek-v4-flash`. All non-codegen
  stages run here — cheap and good enough.
- Stage 10 codegen override (needs `codegen_provider: "anthropic"` to take
  effect):
  - `codegen_model: "claude-sonnet-5-5"`
  - `codegen_api_key_file: ~/.credentials/<key-file>` — the key is read
    from this file at run time. Never put the key itself into config,
    env files, or logs.
  - `codegen_thinking: "adaptive"`, `codegen_effort: "high"`,
    `codegen_refusal_fallback: true`
- Do **not** use `kimi-k3` / `qwen-3.8-max` from the Morpheus proxy for
  codegen: kimi-k3 has a single slot shared with the chat agent and locks
  ~253 MOR per 10-min session (it also caused 503s on the chat lane).

## Secrets

`~/.config/researchclaw/secrets.env` holds the proxy/API keys as env vars
(loaded by the launcher, never echoed). The Anthropic codegen key lives in
its own file referenced by `codegen_api_key_file`.

## Running a stages 10→12 rerun

Use tmux, not nohup — nohup-launched children died with the launching
session on this host; tmux sessions survive:

```bash
tmux new-session -d -s arc './scripts/run_stage10_to_12.sh'
tail -f /tmp/arc-run.log        # progress
cat /tmp/arc-run.status         # START / CHILD_PID / EXIT lines
```

Edit `LOG`/`STATUS`/`--output artifacts/<RUN_DIR>` in the script for your
run. Expected stage-10 duration with sonnet thinking: ~40 min (13 LLM
calls, 5 sandbox validation runs is typical). Success looks like:
`Pipeline complete: 3/3 stages done, 0 failed` and `EXIT=0`.

## Smoke-test result (31 Oct 2026, run rc-20261031-033646-1a9bb3)

- Stage 10: 2570s, best_score 1.0, topic-aligned code (RiskWeightedSPRT,
  CommitRevealBeacon, NetworkSimulator, 5 ablations)
- Stage 12: real metrics across seeds — RiskWeightedSPRT 0.786 vs uniform
  sampling 0.704; NoVerification baseline 0.0; ablations degrade sensibly
- Artifacts: `artifacts/<RUN_DIR>/stage-10..12/` + `deliverables/`
