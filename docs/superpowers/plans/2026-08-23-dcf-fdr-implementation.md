# DCF-FDR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one optional, zero-initialized Distribution-Conditioned Feedback adapter to Clean FDR and publish the tested code/configuration to the configured GitHub repository for later training.

**Architecture:** A shared adapter converts the detached preceding cumulative 4-by-33 probability distribution into a 256-dimensional residual feature. Decoder layer 1 remains unchanged; layers 2-6 add the adapter output only to the FDR box-head input, leaving classification, decoder queries, matching, losses, decoding, and evidence tensors unchanged.

**Tech Stack:** Python, PyTorch, Ultralytics RT-DETR, pytest, YAML, Git.

---

### Task 1: Specify the DCF adapter contract with failing unit tests

**Files:**
- Modify: `tests/test_fdr_head.py`
- Test: `tests/test_fdr_head.py`

- [ ] **Step 1: Add failing tests for the requested adapter API**

Add tests importing `DistributionConditionedFeedback` and asserting shape,
zero-initialized output, detached input gradient, deterministic private
initialization, and rejection of malformed `[B,Q,4,33]` inputs.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `pytest -q tests/test_fdr_head.py -k distribution_conditioned_feedback`

Expected: collection failure because `DistributionConditionedFeedback` does not
exist.

- [ ] **Step 3: Implement the minimal adapter**

Modify `src/fdr_head.py` with one module containing a shared `Linear(33,16)`,
SiLU, and zero-initialized `Linear(64,hidden_dim)`. Normalize input logits with
softmax after detach and validate rank and edge/bin dimensions.

- [ ] **Step 4: Re-run focused tests and verify GREEN**

Run: `pytest -q tests/test_fdr_head.py -k distribution_conditioned_feedback`

Expected: all selected tests pass.

### Task 2: Specify decoder integration with failing tests

**Files:**
- Modify: `tests/test_fdr_head.py`
- Modify: `src/fdr_head.py`
- Test: `tests/test_fdr_head.py`

- [ ] **Step 1: Add failing decoder-behavior tests**

Add tests proving DCF-off matches the existing path, DCF-on zero initialization
is output-equivalent, the shared adapter is called only for layers 2-6, and a
nonzero adapter changes corner logits while leaving the score-head call path
structurally untouched.

- [ ] **Step 2: Run the decoder-focused tests and verify RED**

Run: `pytest -q tests/test_fdr_head.py -k dcf_decoder`

Expected: failures because the decoder has no DCF switch or adapter path.

- [ ] **Step 3: Implement minimal decoder integration**

Extend `FDRDeformableTransformerDecoder` and `from_stock` with an optional
shared adapter. Before each layer-2-to-6 box-head call, compute feedback from
the already accumulated preceding logits and add it to
`output + output_detach`. Do not alter the score-head input or evidence stack.

- [ ] **Step 4: Re-run decoder-focused tests and verify GREEN**

Run: `pytest -q tests/test_fdr_head.py -k dcf_decoder`

Expected: all selected tests pass.

### Task 3: Add the declarative option and Clean FDR+DCF configuration

**Files:**
- Modify: `tests/test_rtdetr_fdr.py`
- Modify: `src/fdr_head.py`
- Create: `configs/rtdetr-l-dcf-fdr.yaml`
- Test: `tests/test_rtdetr_fdr.py`

- [ ] **Step 1: Add failing configuration tests**

Extend the declarative-config helper with `distribution_feedback`. Assert that
the default is false, the DCF YAML enables it, the YAML simultaneously fixes
`preliminary_box=false`, `supervise_pre_boxes=false`,
`supervise_dn_fdr=false`, and `edge_adaptive_fgl=false`, and public RNG state
is unchanged by enabled model construction.

- [ ] **Step 2: Run focused configuration tests and verify RED**

Run: `pytest -q tests/test_rtdetr_fdr.py -k "distribution_feedback or dcf_config"`

Expected: failures because the option and YAML do not exist.

- [ ] **Step 3: Implement the option and YAML**

Add `distribution_feedback: false` to `_OPTION_DEFAULTS`; create the adapter
with the private initialization contract only when enabled; expose the resolved
option through `fdr_options`; add a YAML copied from the frozen RT-DETR-L FDR
graph with the exact Clean FDR loss switches and DCF enabled.

- [ ] **Step 4: Re-run focused tests and verify GREEN**

Run: `pytest -q tests/test_rtdetr_fdr.py -k "distribution_feedback or dcf_config"`

Expected: all selected tests pass.

### Task 4: Add a待跑 launcher contract

**Files:**
- Create: `scripts/train_dcf_fdr.py`
- Create: `tests/test_train_dcf_fdr.py`

- [ ] **Step 1: Add a failing launcher test**

Assert that the launcher binds `configs/rtdetr-l-dcf-fdr.yaml`, reuses the
frozen FDR training settings, assigns a distinct `formal-seed0-dcf-fdr-v1`
identity, and does not auto-launch when imported.

- [ ] **Step 2: Run the launcher test and verify RED**

Run: `pytest -q tests/test_train_dcf_fdr.py`

Expected: collection failure because the launcher does not exist.

- [ ] **Step 3: Implement the minimal launcher**

Follow the existing `train_equal.py`/FDR launcher pattern, changing only the
config and run identity. Keep server execution explicit and avoid embedding
credentials.

- [ ] **Step 4: Re-run the launcher test and verify GREEN**

Run: `pytest -q tests/test_train_dcf_fdr.py`

Expected: all tests pass.

### Task 5: Run regression verification and adversarial code review

**Files:**
- Modify if required: files from Tasks 1-4 only

- [ ] **Step 1: Run focused FDR suites**

Run: `pytest -q tests/test_fdr_head.py tests/test_rtdetr_fdr.py tests/test_train_dcf_fdr.py`

Expected: zero failures.

- [ ] **Step 2: Run the complete test suite**

Run: `pytest -q`

Expected: zero failures; any environment-only skips are recorded.

- [ ] **Step 3: Perform adversarial diff checks**

Run: `git diff --check` and inspect `git diff -- src/fdr_head.py configs/rtdetr-l-dcf-fdr.yaml scripts/train_dcf_fdr.py tests`.

Confirm that score heads, matcher/loss paths, output tensor contracts, and
non-DCF configurations are unchanged; DCF uses detached preceding cumulative
logits and one shared adapter; no credentials or unrelated user files enter the
diff.

### Task 6: Commit and push the待跑 implementation

**Files:**
- Add only the reviewed DCF design, plan, source, config, launcher, and tests.

- [ ] **Step 1: Verify repository identity and staged scope**

Run: `git status --short`, `git diff --stat`, and `git diff --cached --check`
after staging explicit paths.

- [ ] **Step 2: Commit**

Run: `git commit -m "feat: add distribution-conditioned FDR refinement"`

- [ ] **Step 3: Push the current branch**

Run: `git push origin codex/ap-fdr-integrated-redesign`

Expected: the commit is available on the configured GitHub remote for later
server training. Do not launch a training run in this task.
