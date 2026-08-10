# B-Conference Paper Materials Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one canonical Chinese Markdown that organizes the latest paper-ready FDR/BPDD evidence and moves all failed experiments into a clearly separated appendix.

**Architecture:** Build a small immutable evidence index first, then generate the narrative only from repository and Release evidence. Preserve all historical files in place and use the master document to classify them as current, superseded, or historical-only.

**Tech Stack:** Markdown, JSON, CSV, Git, GitHub Releases, SHA256, PowerShell, strict SSH.

---

## File map

- Create: `docs/CCF_B_PAPER_MATERIALS_MASTER_2026-08-10_ZH.md` — sole recommended reading entry for paper preparation.
- Create: `evidence/paper-master-2026-08-10/authority-index.json` — machine-readable current-result and claim authority.
- Create: `evidence/paper-master-2026-08-10/document-status.csv` — current/superseded/historical classification of relevant GitHub documents.
- Create: `evidence/paper-master-2026-08-10/SHA256SUMS.txt` — hashes of the master document and evidence index.

### Task 1: Inventory current GitHub evidence

**Files:**
- Create: `evidence/paper-master-2026-08-10/document-status.csv`

- [ ] **Step 1: Enumerate research-facing files**

Run `rg --files docs research evidence reports configs src | sort` and collect files relating to Control, FDR, BPDD, YAML modularization, LPR, IBER/Boundary, quality reranking, PFCR, FrequencyCM, SCADS, and GLGM.

Expected: every file referenced by the final report exists in the branch or has an explicit GitHub Release URL.

- [ ] **Step 2: Assign document authority status**

Write CSV columns `path,status,topic,reason,replaced_by`. Use only:

```text
current          latest evidence that may support paper claims
superseded       previously valid but replaced by a newer authority
historical-only  failed attempt or process record retained for the appendix
```

The strict Control/FDR report, FDR method guide, BPDD authority, BPDD design/protocol, and new master report references are `current`. Older handoffs and stale preliminary comparisons are `superseded`. Failed module records are `historical-only`.

- [ ] **Step 3: Verify paths**

For every repository path in the CSV, run `Test-Path`; reject missing entries. For Release-only evidence, verify the HTTP asset returns a successful response.

Expected: prints `DOCUMENT_INDEX_OK` with zero missing local paths and zero unavailable current Release assets.

### Task 2: Freeze the current paper authority

**Files:**
- Create: `evidence/paper-master-2026-08-10/authority-index.json`

- [ ] **Step 1: Record the immutable experiment environment**

Include RT-DETR-L, Ultralytics 8.4.90, RTX 4090 24GB, driver 550.142, Python 3.10.12, PyTorch 2.5.1+cu121, Torchvision 0.20.1+cu121, CUDA 12.1, VisDrone 6471/548 images, ten classes, dataset SHA256 `FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB`, scratch initialization, batch 8, MuSGD, AMP scale 128, 300 queries, `NMS=false`, and `max_det=300`.

- [ ] **Step 2: Record strict Control and FDR100 results**

Store the same-evaluator results:

```text
Control: P 0.46761, R 0.41731, AP50 0.38663, AP75 0.21302, mAP 0.21911
FDR:     P 0.5691126151072722, R 0.49277710639408445,
         AP50 0.48468375790335755, AP75 0.29252552290080275,
         mAP 0.289659749097641
```

Mark this as strict full-data seed0 100-epoch same-evaluator evidence.

- [ ] **Step 3: Record BPDD evidence and comparison boundary**

Store Screen30 paired deltas `mAP +0.00189`, `AP75 +0.001846`, and tail-three mAP `+0.000557`. Store Formal100 independent metrics `P 0.5706336342276676`, `R 0.494464137092214`, `F1 0.5298252895497555`, `AP50 0.4864074913692349`, `AP75 0.29809564823246976`, and `mAP 0.29225786956881444`.

Set:

```json
{
  "formal_comparison_kind": "preliminary-cross-authority",
  "fresh_formal_pair": false,
  "strict_pair_required_for_final_paper": true,
  "multi_seed_complete": false
}
```

- [ ] **Step 4: Record method and originality authority**

Declare FDR as D-FINE-derived structural adaptation, not an original base formula. Declare BPDD only as the six-part narrow combination in `research/bpdd/BPDD_AUTHORITY.md`. Record that a third successful original module is not yet frozen.

- [ ] **Step 5: Validate numeric consistency**

Recompute all Control→FDR and FDR→BPDD deltas from stored raw metrics. Require absolute calculation error below `1e-12` for full-precision values and preserve an explicit rounding field for five-decimal legacy Control values.

Expected: prints `PAPER_AUTHORITY_OK`.

### Task 3: Write the canonical paper-materials Markdown

**Files:**
- Create: `docs/CCF_B_PAPER_MATERIALS_MASTER_2026-08-10_ZH.md`

- [ ] **Step 1: Write the executive status and reading guide**

State the current strongest result, the exact paper-readiness boundary, and that this file supersedes narrative conclusions in older handoffs without deleting them.

- [ ] **Step 2: Write the paper-ready method narrative**

Cover the problem motivation, stock RT-DETR-L path, FDR structure/FGL/YAML integration, BPDD future-only softmin teacher/better-only reliability/KL loss/final-match reuse, and zero-inference-branch property. Link every implementation claim to source files.

- [ ] **Step 3: Write all current result tables**

Include Control versus FDR overall, scale and ten-class results; FDR versus BPDD Screen30 and preliminary Formal100; BPDD scale and class tables; parameter/GFLOPs/checkpoint evidence; and a separate column identifying strict, preliminary, or pending evidence status.

- [ ] **Step 4: Write submission materials**

Provide a conservative Chinese abstract fact block, three contribution slots, method-section outline, experiment-section outline, required figures, required tables, ablation matrix, and a claim-allowed/claim-forbidden table.

- [ ] **Step 5: Write the failed-attempt appendix**

For LPR/LPR-G, ACR/IBER/Boundary, quality oracle/OAR/PFCR, FrequencyCM, SCADS, and GLGM, record the tested hypothesis, decisive evidence, failure mode, and what it taught the successful FDR/BPDD direction. Do not promote any failed result into the main contribution list.

- [ ] **Step 6: Write the remaining-work checklist**

List fresh paired FDR/BPDD Formal100, seeds 1/2, final FP16 latency, the unresolved third original module, external-dataset/generalization evidence, and final full ablation as pending.

### Task 4: Verify and publish

**Files:**
- Create: `evidence/paper-master-2026-08-10/SHA256SUMS.txt`

- [ ] **Step 1: Hash current artifacts**

Generate uppercase SHA256 entries in lexical order for the master Markdown, authority JSON, and document-status CSV.

- [ ] **Step 2: Run integrity checks**

Check JSON parsing, CSV path existence, exact delta recomputation, Markdown link targets, placeholder patterns, contradictory completion claims, and `git diff --check`.

Expected: all checks exit `0`; no current claim lacks an evidence pointer.

- [ ] **Step 3: Run BPDD regression tests**

Run `PYTHONPATH=. /data/uav/venvs/iber-be-v1/bin/python -m pytest -q -k bpdd` on the verified server environment when SSH is available. If SSH remains temporarily closed, use the last freshly verified `129 passed` result and label it with its commit rather than fabricating a new run.

- [ ] **Step 4: Commit and push**

```bash
git add docs/CCF_B_PAPER_MATERIALS_MASTER_2026-08-10_ZH.md \
        evidence/paper-master-2026-08-10
git commit -m "docs: publish B-conference paper materials master"
git push origin codex/bpdd-fdr
```

- [ ] **Step 5: Verify GitHub state**

Require local HEAD to equal `git ls-remote origin refs/heads/codex/bpdd-fdr`. Open the GitHub Markdown and raw authority JSON URLs and confirm the current files are accessible.
