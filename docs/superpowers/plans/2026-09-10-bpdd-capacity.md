# BPDD capacity implementation plan

> Execute inline with executing-plans, TDD and fresh verification. User approved implementation and deployment on 2026-09-10; no additional deployment approval is required.

**Goal:** deploy paired local-expert-only and local-expert-plus-distillation Formal100 experiments without disturbing previous jobs.

**Architecture:** retain the six-layer LRS-GFDR graph and initialization; expose its last hidden query, sample P2/P3/P4 around normal-query boxes, and refine distributions and classes using two attention blocks. Direct expert supervision reuses the final assignment. Detached expert targets teach consistent early/final queries only when measurably better. The expert stays at inference.

**Tech Stack:** pinned Ultralytics 8.4.90, PyTorch, existing Formal100 MuSGD/AMP launcher and Paramiko operator transport.

## Tasks

- [ ] Add `tests/test_bpdd_capacity.py`: zero-output identity; RNG isolation; known-coordinate sampling; invalid/empty input; actual feature/query gradients after head activation; detached teacher; consistent assignment filtering; paired shared initialization; train/eval prediction and pickle roundtrip.
- [ ] Run `C:/uav_env/Scripts/python.exe -m pytest tests/test_bpdd_capacity.py -q` and verify the missing module is the only initial failure.
- [ ] Implement `src/bpdd_capacity.py`: 41 points/scale, validity masking, 256-wide two-block local attention, 1024-wide FFN, zero residual output, private seed 30000, three feature projections. Use query chunks to limit peak intermediates; preserve gradient through features but detach sampling coordinates.
- [ ] Implement `src/bpdd_capacity_loss.py`: joint detached teacher, common-support localization KL, training-only bounded absolute NLL improvement weights and geometric non-degradation gate; matched classification Bernoulli KL with label-quality gate. Normalize per source then mean across layers, independent weights. Record coverage, teacher advantage, support bounds and losses, including zeros.
- [ ] Expose `last_output` in `src/fdr_head.py` without changing base computation/state. Implement `src/rtdetr_bpdd_capacity.py`: explicit stock graph execution with local feature retention, teacher direct loss and independent initialization loader. Initial keys may only differ by `expert.` prefix. Separate expert gradient clip group and sparse per-loss gradients.
- [ ] Add YAML with `capacity` options and launcher `scripts/train_bpdd_capacity.py`. Keep Formal100 settings, shared artifact and seeds; two arms differ only in distillation toggle. Save launch authority, param counts, runtime JSONL and completion status. Warm up KD from epoch 10 to 20; fixed localization/classification peak weights 0.15/0.10 and tau=0.1 as declared screening defaults, not previously optimized constants. Probe quality before formal launch; quality gates remain active during training.
- [ ] Local unit/integration regression, then remote CUDA forward/backward with normal/tiny/empty inputs, 640x640 batch8 and pinned optimizer. Run a short actual dataset probe in an isolated output directory; reject nonfinite loss/gradients or memory failure. Record latency/memory, without claiming short-run AP predicts Formal100.
- [ ] Freeze source into a commit/archive, upload create-only remote source directory, run reproducible tests and preflight. Start paired sequential queue, KD candidate first then expert-only control; do not overwrite earlier results or kill unrelated jobs. Confirm real batches and runtime records, not just a shell PID.
- [ ] Save the verified deployment record and concrete monitoring paths locally. Report what is running, what is queued, validation evidence and remaining accuracy uncertainty.

## Design adjudication

Success-priority implementation keeps common support/KL, adds a separate classifier teacher, and avoids changing backbone scale or adding FIA. Compared with the proposal, tau is a declared fixed screening hyperparameter instead of estimating it from an untrained teacher; normalization and loss gradients are measured in preflight before deployment. Teacher benefit is monitored unconditionally as well as after gates. A failed preflight is repaired and rerun before any formal job. Only the new candidate tests/configs/source are committed.
