# SQDA-SGC RT-DETR-L Implementation and Deployment Plan

> **For Codex:** Execute this plan task by task. Use test-driven development for every behavior change, keep the stock RT-DETR decoder/head/loss/matcher/postprocess untouched, and stop the experiment when a hard gate fails.

**Goal:** Implement the trainable SQDA-SGC query adapter on the mature RT-DETR-L VisDrone baseline, prove exact identity compatibility, deploy it to the provided RTX 4090 server, and start the pre-registered G1 frozen-module experiment.

**Architecture:** A standalone `SQDASGCAdapter` consumes the native last 300 RT-DETR queries, their detached normalized proposal boxes, and raw stride-4 C2 features. The adapter samples center, four boundary, and read-only outer-context roles; fuses semantic and geometric evidence through one bounded group-gated residual; and writes back only to the same 300 query slots. A model wrapper captures C2 and intercepts the stock deformable decoder immediately after Top-300 selection. Denoising-query prefixes and every stock detection component remain unchanged.

**Tech stack:** Python 3.10, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, Ultralytics 8.4.90, pytest, VisDrone2019-DET.

**Authoritative substrate:** `codex/matched-baseline@b08bc2ac`, mature baseline SHA256 `54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B`. The ACR-EG/GCQF design and all associated checkpoints, optimizers, query changes, losses, and resume schedules are explicitly out of scope.

---

## Task 1: Implement the pure SQDA-SGC module

**Files:**

- Create: `src/sqda_sgc.py`
- Create: `tests/test_sqda_sgc.py`

### Step 1: Write failing unit tests

Cover these frozen invariants:

- input/output shape `[B,300,256]`, dtype, and device;
- exact 20-point role layout: C=4, L/R/T/B=2 each, O=8;
- `max(w/2,1/W2)` and `max(h/2,1/H2)` sampling radius;
- zero-initialized point-offset output layers with `0.1u` bound;
- reference boxes are detached from autograd;
- invalid points are masked, all-invalid write roles produce zero residual, and no NaN appears;
- read-only context cannot enter the fusion tensor;
- `lambda_ctx` lies in `(0,0.25)`, initializes at `0.05`, and produces `c_sem` in `(0.75,1)`;
- invalid context forces `c_sem=1`;
- group gate has 16 values and expands to 256 channels;
- LayerScale lies in `(0,0.05)` and initializes to `1e-3`;
- disabled mode and identity override return the input bit-for-bit;
- parameter count remains below one million.

Run:

```powershell
pytest tests/test_sqda_sgc.py -q
```

Expected: tests fail because the module does not exist.

### Step 2: Implement geometry and sampling

Implement:

- `SQDASGCConfig` dataclass with the frozen dimensions and bounds;
- sinusoidal box positional encoding;
- shared role-conditioned offset MLP with zero-initialized final projections;
- vectorized `grid_sample(..., align_corners=False, padding_mode="zeros")`;
- explicit validity mask based on unclipped coordinates;
- shared C2 projector `128 -> 128 -> 256`;
- role-wise masked point attention.

### Step 3: Implement fusion and bounded writeback

Implement:

- semantic center descriptor;
- geometry attention across L/R/T/B descriptors;
- read-only outer-context reliability;
- the safe context modulation formula;
- 16-group gate from query, semantic/geometric differences, interactions, and box geometry;
- one fusion projection and one bounded residual;
- diagnostics detached from the training graph.

### Step 4: Run the unit tests

```powershell
pytest tests/test_sqda_sgc.py -q
```

Expected: all module tests pass.

### Step 5: Commit

```powershell
git add src/sqda_sgc.py tests/test_sqda_sgc.py
git commit -m "feat: implement SQDA-SGC query adapter"
```

---

## Task 2: Integrate after Top-300 selection without changing RT-DETR

**Files:**

- Create: `src/rtdetr_sqda_sgc.py`
- Create: `tests/test_rtdetr_sqda_sgc_integration.py`

### Step 1: Write failing integration tests

Use a small stock RT-DETR-L model fixture and assert:

- the adapter is registered outside the stock `model` sequential tree;
- a forward hook captures raw stride-4 C2 from stock layer 1;
- a pre-hook on `RTDETRDecoder.decoder` modifies only the final 300 query embeddings;
- a denoising-query prefix is unchanged;
- reference logits, encoded features, spatial shapes, heads, and masks are unchanged;
- identity override gives bitwise-equal stock outputs for eval and train forwards;
- adapter parameters appear in `state_dict`;
- hooks are installed exactly once and stale C2 cannot be reused;
- malformed query counts or missing C2 fail closed.

Run:

```powershell
pytest tests/test_rtdetr_sqda_sgc_integration.py -q
```

Expected: tests fail because the wrapper does not exist.

### Step 2: Implement the model wrapper

Implement `SQDASGCDetectionModel` as a stock `RTDETRDetectionModel` subclass:

1. construct the stock model;
2. attach `sqda_sgc` as a trainable child;
3. capture layer-1 C2 for the current forward only;
4. at the stock deformable decoder pre-hook, take the final 300 embeddings and final 300 reference logits;
5. apply `sigmoid().detach()` to reference boxes;
6. invoke the adapter;
7. concatenate any denoising prefix unchanged;
8. clear transient state in both success and failure paths.

Do not override or fork the stock detection decoder, loss, matcher, head, validator, or postprocess.

### Step 3: Implement strict mature-baseline loading

Add a loader that:

- accepts only a checkpoint containing an `ema` or `model` `nn.Module`;
- copies all matching stock model keys;
- permits missing keys only under `sqda_sgc.*`;
- rejects missing/unexpected stock keys and incompatible shapes;
- records the source checkpoint path and SHA in run metadata.

### Step 4: Run integration tests

```powershell
pytest tests/test_rtdetr_sqda_sgc_integration.py -q
```

Expected: all integration tests pass.

### Step 5: Commit

```powershell
git add src/rtdetr_sqda_sgc.py tests/test_rtdetr_sqda_sgc_integration.py
git commit -m "feat: integrate SQDA-SGC into stock RT-DETR"
```

---

## Task 3: Implement frozen-module training and hard-gate validation

**Files:**

- Create: `scripts/train_rtdetr_sqda_sgc.py`
- Create: `scripts/verify_sqda_sgc_g0.py`
- Create: `tests/test_sqda_sgc_training.py`

### Step 1: Write failing trainer tests

Assert:

- every stock parameter is frozen and every SQDA-SGC parameter is trainable;
- stock buffers and parameters remain unchanged after one optimizer step;
- the optimizer contains SQDA-SGC parameters only;
- matrix weights use exact weight decay `1e-4`;
- bias, normalization, LayerScale, context scalar, and other scalar parameters use zero decay;
- AdamW uses LR `1e-4` and betas `(0.9,0.999)`;
- only module gradients are clipped to global norm `0.1`;
- all trainable branches receive finite nonzero gradients after one synthetic backward;
- checkpoint/EMA state contains SQDA-SGC;
- G1 and G2 both load fresh from the same mature checkpoint;
- the formal CLI rejects epochs outside `{3,10}`, nonzero seed, changed image size, changed query count, and resume mode.

Run:

```powershell
pytest tests/test_sqda_sgc_training.py -q
```

Expected: tests fail because the trainer and CLI do not exist.

### Step 2: Implement the frozen trainer

Subclass `RTDETRTrainer` and:

- freeze stock layers through Ultralytics' freeze configuration;
- validate freeze state after setup;
- build an explicit module-only AdamW optimizer;
- override the optimizer step with module-only `clip_grad_norm_(...,0.1)`;
- use constant LR after 0.5-epoch linear warmup;
- compare stock parameter and buffer hashes before/after the smoke step.

### Step 3: Implement the formal CLI

Expose only pre-registered choices:

```text
--gate g1|g2
--checkpoint PATH
--data PATH
--project PATH
--device 0
--workers N
```

Fix internally:

```text
seed=0 deterministic=True imgsz=640 batch=8
epochs=3 for G1, epochs=10 for G2
optimizer=AdamW lr0=1e-4 lrf=1.0
warmup_epochs=0.5 warmup_bias_lr=0 cos_lr=False
val=True save=True save_period=1 max_det=300
```

Write a JSON run manifest before training containing git SHA, dependency versions, dataset path/hash/counts, baseline SHA, model parameter counts, frozen/trainable key lists, and all arguments.

### Step 4: Implement G0 exact-equivalence verification

Load the mature checkpoint twice:

- stock RT-DETR-L;
- SQDA-SGC wrapper with identity override.

On the same fixed validation batch, assert:

- maximum absolute tensor difference is zero;
- query count/order and reference boxes are identical;
- NMS-off decoded outputs and validator metrics are identical within serialization precision.

The script exits nonzero on any mismatch.

### Step 5: Run trainer tests

```powershell
pytest tests/test_sqda_sgc_training.py -q
```

Expected: all trainer tests pass.

### Step 6: Commit

```powershell
git add scripts/train_rtdetr_sqda_sgc.py scripts/verify_sqda_sgc_g0.py tests/test_sqda_sgc_training.py
git commit -m "feat: add frozen SQDA-SGC training gates"
```

---

## Task 4: Add dataset, baseline, and Top-300 preflight diagnostics

**Files:**

- Create: `scripts/prepare_sqda_sgc_server.py`
- Create: `scripts/diagnose_sqda_top300.py`
- Create: `tests/test_sqda_sgc_preflight.py`

### Step 1: Write failing preflight tests

Assert:

- SHA256 verification rejects a one-byte checkpoint change;
- dataset validation requires exactly 6,471 train images/labels and 548 validation images/labels;
- every validation label is parseable and in range;
- generated dataset YAML resolves to the actual `/root/data/uav/datasets/VisDrone` tree;
- proposal recall is class-agnostic and uses the stock Top-300 reference boxes;
- reported bins and thresholds are fixed before training;
- recoverability statistics are deterministic for a synthetic example.

Run:

```powershell
pytest tests/test_sqda_sgc_preflight.py -q
```

Expected: tests fail because the tools do not exist.

### Step 2: Implement server preparation

The preparation tool must:

- create only scoped directories under `/root/data/uav`;
- install pinned dependencies from preferred mainland mirrors, with official PyTorch fallback;
- clone the official GitHub repository and checkout the SQDA-SGC branch;
- download the mature baseline asset and verify its exact SHA256;
- download/prepare VisDrone with the repository converter;
- verify exact file counts and produce a deterministic validation signature;
- generate a path-correct dataset YAML and run a one-batch stock validation smoke test.

No password, token, or SSH credential may be written to disk or logs.

### Step 3: Implement Top-300 diagnostic

For the fixed validation set, report:

- class-agnostic proposal recall at IoU 0.3, 0.5, and 0.7;
- COCO small/medium/large bins, plus tiny `<16²` as an explicitly diagnostic-only bin;
- missed-object recoverability based on proposal IoU versus the stock baseline's final detections;
- JSON output with dataset, checkpoint, and git identities.

### Step 4: Run preflight tests

```powershell
pytest tests/test_sqda_sgc_preflight.py -q
```

Expected: all preflight tests pass.

### Step 5: Commit

```powershell
git add scripts/prepare_sqda_sgc_server.py scripts/diagnose_sqda_top300.py tests/test_sqda_sgc_preflight.py
git commit -m "feat: add SQDA-SGC deployment preflight"
```

---

## Task 5: Run local verification

### Step 1: Run focused tests

```powershell
pytest tests/test_sqda_sgc.py tests/test_rtdetr_sqda_sgc_integration.py tests/test_sqda_sgc_training.py tests/test_sqda_sgc_preflight.py -q
```

Expected: all focused tests pass.

### Step 2: Run full regression tests

```powershell
pytest -q
```

Expected: all repository tests pass or any unrelated pre-existing failure is recorded with evidence.

### Step 3: Run static and repository checks

```powershell
python -m compileall src scripts
git diff --check
git status --short
```

Expected: compilation and whitespace checks succeed; status lists only planned uncommitted run metadata, if any.

---

## Task 6: Deploy to the RTX 4090 server

Remote root: `/root/data/uav`.

### Step 1: Record remote preflight

Record:

- GPU name, driver, and memory;
- free data-disk space;
- Python/conda/git versions;
- absence of an existing training process.

Expected: one idle RTX 4090 with sufficient memory and at least 80 GB free on `/root/data`.

### Step 2: Push and clone the exact implementation

Push `codex/sqda-sgc` to the official GitHub repository, then on the server:

```bash
git clone --branch codex/sqda-sgc --single-branch \
  https://github.com/kkc236/uav-detection-baselines \
  /root/data/uav/sqda-sgc
```

Record the local and remote commit SHA and require equality.

### Step 3: Create the pinned environment with mirror priority

Create `/root/data/uav/venv` using:

```bash
python -m venv /root/data/uav/venv
/root/data/uav/venv/bin/pip install -U pip \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
/root/data/uav/venv/bin/pip install \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://mirrors.aliyun.com/pytorch-wheels/cu121
/root/data/uav/venv/bin/pip install -r requirements.txt \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

If the PyTorch mirror lacks an exact wheel, use the official `https://download.pytorch.org/whl/cu121` index and record the fallback.

### Step 4: Prepare and validate immutable inputs

Run the server preparation tool. Hard requirements:

- baseline SHA matches exactly;
- train/val counts are 6,471/548 with matching labels;
- dataset YAML resolves to the server path;
- stock one-batch validation succeeds;
- no ACR-EG/GCQF checkpoint or code path is referenced.

### Step 5: Run all tests remotely

```bash
cd /root/data/uav/sqda-sgc
/root/data/uav/venv/bin/pytest \
  tests/test_sqda_sgc.py \
  tests/test_rtdetr_sqda_sgc_integration.py \
  tests/test_sqda_sgc_training.py \
  tests/test_sqda_sgc_preflight.py -q
```

Expected: all tests pass on CUDA-capable dependencies.

---

## Task 7: Execute gates and start G1

### Step 1: Run G0 exact identity

```bash
/root/data/uav/venv/bin/python scripts/verify_sqda_sgc_g0.py \
  --checkpoint /root/data/uav/checkpoints/matched-baseline-best-epoch-0100.pt \
  --data /root/data/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml \
  --device 0
```

Hard gate: exit code zero and exact fixed-batch identity. Failure stops all training.

### Step 2: Run Top-300 diagnostics

```bash
/root/data/uav/venv/bin/python scripts/diagnose_sqda_top300.py \
  --checkpoint /root/data/uav/checkpoints/matched-baseline-best-epoch-0100.pt \
  --data /root/data/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml \
  --device 0 \
  --output /root/data/uav/runs/sqda-sgc/top300-diagnostic.json
```

Record proposal recall and recoverability. If the diagnostic shows that the native Top-300 set does not contain recoverable missed small objects, do not claim an 80% conditional success probability and stop before G2.

### Step 3: Start G1 in a persistent remote process

```bash
nohup /root/data/uav/venv/bin/python -u scripts/train_rtdetr_sqda_sgc.py \
  --gate g1 \
  --checkpoint /root/data/uav/checkpoints/matched-baseline-best-epoch-0100.pt \
  --data /root/data/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml \
  --project /root/data/uav/runs/sqda-sgc \
  --device 0 \
  --workers 8 \
  > /root/data/uav/runs/sqda-sgc/g1-console.log 2>&1 &
```

Save the PID and command manifest. Confirm after startup:

- one intended training process;
- GPU memory allocation is stable;
- loss is finite;
- module gradients are finite;
- stock parameter/buffer hash remains unchanged;
- checkpoint contains `sqda_sgc.*`.

### Step 4: Evaluate every G1 epoch against the immutable baseline

Use the same stock validator settings and fixed seed. Record P, R, mAP50, mAP50-95, and native AP-small for:

- mature baseline;
- G1 epoch 1;
- G1 epoch 2;
- G1 epoch 3.

G1 passes only if at least one checkpoint has:

```text
Delta P >= 0
Delta R >= 0
Delta mAP50 >= 0
Delta mAP50-95 >= 0
Delta AP-small >= +0.2 percentage points
```

No threshold retuning, multi-seed averaging, cherry-picking a different dataset split, or inference-time algorithm is allowed.

### Step 5: Start G2 only after a documented G1 pass

G2 must reload the same mature baseline from scratch and train for exactly 10 epochs:

```bash
nohup /root/data/uav/venv/bin/python -u scripts/train_rtdetr_sqda_sgc.py \
  --gate g2 \
  --checkpoint /root/data/uav/checkpoints/matched-baseline-best-epoch-0100.pt \
  --data /root/data/uav/protocols/tsgr-p2-e1/source-VisDrone-full.yaml \
  --project /root/data/uav/runs/sqda-sgc \
  --device 0 \
  --workers 8 \
  > /root/data/uav/runs/sqda-sgc/g2-console.log 2>&1 &
```

Do not start this command when any G0, diagnostic, or G1 hard gate fails.

