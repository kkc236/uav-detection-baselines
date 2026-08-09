# GLGM-v2 Server Runbook

## Authority

- Legacy Screen30: `/home/ubuntu/glgm` (read-only reference)
- GLGM-v2 source: `/home/ubuntu/glgm-v2/source/ultralytics-main`
- GLGM-v2 campaign: `/home/ubuntu/glgm-v2/work/campaign-v1`
- Dataset: authoritative VisDrone train 6471 / val 548 audit
- Checkpoints: server-local only; publication is forbidden without separate exact-subset approval

The source installer records before/after SHA-256 values and refuses ambiguous import anchors. Every pair creates a fresh work root and verifies the data inventory, runtime, source tree, paired public initialization, finite training state, independent best/last evaluation, and artifact checksums.

## Two-GPU Scheduling

Each GPU runs one complete pair sequentially. A Control arm and its candidate arm never use different physical GPUs, and no run uses DDP.

| Wave | GPU 0 | GPU 1 |
|---|---|---|
| Smoke2/Screen10 1 | V2 pair | V3 pair |
| Smoke2/Screen10 2 | V4 pair | V5 pair |
| Screen30 | candidate A pair | candidate B pair |
| Formal100 1 | seed0 pair | seed1 pair |
| Formal100 2 | seed2 pair | seed3 pair |

## Hourly Checks

1. Strict SSH host-key verification and supervisor PID/state.
2. GPU UUID, utilization, memory, temperature, power and owning process.
3. Active stage, variant, seed, arm, completed epoch and finite latest metrics.
4. `results.csv` row continuity and required checkpoint presence.
5. Dataset/source/runtime hashes and new audit events.
6. Free disk space and unexpected processes or authority reuse.

The monitor may retry recoverable connectivity checks. It must not resume one arm, reuse an incomplete work root, relax a Screen gate, mix old checkpoints, or publish `.pt` files.
