# BPDD Key Evidence Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish one evidence-backed Chinese BPDD method report and the minimum machine-readable evidence needed to reproduce every reported conclusion.

**Architecture:** Keep the narrative in one canonical Markdown file and place immutable lightweight evidence in a dedicated evidence directory. Link large checkpoints to the existing GitHub Release instead of committing them, and label the existing FDR100 comparison as preliminary cross-authority evidence rather than a fresh Formal100 pair.

**Tech Stack:** Markdown, JSON, CSV, SHA256, Git, GitHub Releases, PowerShell, SSH with pinned host fingerprint.

---

## File map

- Create: `docs/BPDD_FDR_METHOD_FORMAL100_REPORT_2026-08-10_ZH.md` — canonical method, experiment, result, and risk report.
- Create: `evidence/bpdd-formal-848f00cb/independent-eval.json` — exact 548-image evaluation payload already published in the Release.
- Create: `evidence/bpdd-formal-848f00cb/screen30-gate2.json` — frozen paired Screen30 decision and deltas.
- Create: `evidence/bpdd-formal-848f00cb/formal100-summary.json` — concise final metrics, preliminary FDR reference, deltas, and claim boundary.
- Create: `evidence/bpdd-formal-848f00cb/publication-status.json` — 100/100 publication verification and Release identifiers.
- Create: `evidence/bpdd-formal-848f00cb/SHA256SUMS.txt` — hashes for all committed evidence files.

### Task 1: Acquire and freeze lightweight evidence

**Files:**
- Create: `evidence/bpdd-formal-848f00cb/independent-eval.json`
- Create: `evidence/bpdd-formal-848f00cb/screen30-gate2.json`
- Create: `evidence/bpdd-formal-848f00cb/publication-status.json`

- [ ] **Step 1: Download the already-published independent evaluation**

Run:

```powershell
Invoke-WebRequest `
  -Uri 'https://github.com/kkc236/uav-detection-baselines/releases/download/bpdd-formal-848f00cb-live/bpdd-formal-848f00cb-independent-eval.json' `
  -OutFile 'evidence/bpdd-formal-848f00cb/independent-eval.json'
```

Expected: a JSON object with `processed_images=548`, `metrics.map=0.29225786956881444`, and an exact-final EMA checkpoint SHA256.

- [ ] **Step 2: Retrieve Screen30 Gate2 and publication status**

Use strict SSH with the pinned known-host file to copy:

```text
/data/uav/runs/bpdd-gate-848f00cb/gate2.json
/data/uav/logs/bpdd-formal-848f00cb/bpdd-exact-sync-status.json
```

If SSH is temporarily closed, retrieve the corresponding lightweight assets from GitHub Release metadata; do not synthesize missing fields.

Expected Gate2 values: final mAP delta `0.00189`, final AP75 delta `0.001846`, tail-three mAP delta `0.000557`, with a passed decision. Expected publication state: completed epoch `100`, ledger records `100`, queued epochs `100`, state `verified`.

- [ ] **Step 3: Validate machine-readable evidence**

Run a PowerShell `ConvertFrom-Json` check that asserts the independent evaluation image count, dataset SHA256, checkpoint publication flag, and final metrics are present and finite.

Expected: command exits `0` and prints `BPDD_EVIDENCE_OK`.

- [ ] **Step 4: Commit frozen source evidence**

```bash
git add evidence/bpdd-formal-848f00cb/independent-eval.json \
        evidence/bpdd-formal-848f00cb/screen30-gate2.json \
        evidence/bpdd-formal-848f00cb/publication-status.json
git commit -m "evidence: freeze BPDD screen and Formal100 results"
```

### Task 2: Build the concise comparison evidence

**Files:**
- Create: `evidence/bpdd-formal-848f00cb/formal100-summary.json`

- [ ] **Step 1: Compute exact global deltas**

Use the independent BPDD metrics and the existing strict FDR100 authority:

```text
FDR:  P 0.5691126151072722, R 0.49277710639408445,
      AP50 0.48468375790335755, AP75 0.29252552290080275,
      mAP 0.289659749097641
BPDD: P 0.5706336342276676, R 0.494464137092214,
      AP50 0.4864074913692349, AP75 0.29809564823246976,
      mAP 0.29225786956881444
```

Expected exact deltas: Precision `0.00152101912039537`, Recall `0.00168703069812953`, AP50 `0.00172373346587734`, AP75 `0.00557012533166701`, mAP `0.00259812047117347`.

- [ ] **Step 2: Write the summary with an explicit comparison boundary**

The JSON must include `comparison_kind="preliminary-cross-authority"`, `fresh_formal_pair=false`, and `strict_pair_required_for_final_paper=true`. It must also include Screen30 paired results and the exact checkpoint/EMA hashes from `independent-eval.json`.

- [ ] **Step 3: Cross-check all summary values**

Run a PowerShell assertion that recomputes every delta from the two metric objects and rejects a difference greater than `1e-12`.

Expected: command exits `0` and prints `BPDD_SUMMARY_OK`.

### Task 3: Write the canonical Chinese report

**Files:**
- Create: `docs/BPDD_FDR_METHOD_FORMAL100_REPORT_2026-08-10_ZH.md`

- [ ] **Step 1: Document the method and code mapping**

Cover motivation, six-layer cumulative FDR inputs, future-only softmin teacher, exact adjacent-bin FGL proper score, better-only reliability gate, KL distillation, stock final Hungarian assignment reuse, training-only isolation, and YAML options. Link to `src/bpdd_loss.py`, `src/rtdetr_fdr_bpdd.py`, `src/bpdd_protocol.py`, and `research/bpdd/BPDD_AUTHORITY.md`.

- [ ] **Step 2: Document protocol and evidence**

Include the frozen Ultralytics RT-DETR-L/VisDrone configuration, dataset SHA256, seed0/full-data/100-epoch status, Screen30 pair, Formal100 independent validation, checkpoint and EMA SHA256, and GitHub Release URLs.

- [ ] **Step 3: Document results and adversarial analysis**

Include overall metrics, scale metrics, ten-class AP/AP50/AP75, training endpoint versus independent evaluation, the preliminary FDR comparison, zero-inference-overhead explanation, positive findings, negative findings, and the exact work still required for paper-final evidence.

- [ ] **Step 4: Document claim boundaries**

Explicitly forbid claims that BPDD invented self-distillation, later-to-earlier supervision, distribution distillation, FDR, or GO-LSD. Limit the narrow contribution to the repository-authorized six-part combination in `BPDD_AUTHORITY.md`.

### Task 4: Hash, verify, and publish

**Files:**
- Create: `evidence/bpdd-formal-848f00cb/SHA256SUMS.txt`

- [ ] **Step 1: Generate deterministic SHA256 records**

Hash the four JSON files and the canonical Markdown report. Store uppercase hashes followed by repository-relative paths in lexical order.

- [ ] **Step 2: Verify report integrity**

Run:

```powershell
rg -n 'TBD|TODO|PLACEHOLDER|待定|成功率保证|fresh paired Formal100 已完成' `
  docs/BPDD_FDR_METHOD_FORMAL100_REPORT_2026-08-10_ZH.md `
  evidence/bpdd-formal-848f00cb
git diff --check
```

Expected: no unsupported placeholder or completion claim; `git diff --check` exits `0`.

- [ ] **Step 3: Run relevant regression tests**

Run on the verified 4090 environment:

```bash
PYTHONPATH=. /data/uav/venvs/iber-be-v1/bin/python -m pytest -q -k bpdd
```

Expected: all BPDD-selected tests pass with zero failures.

- [ ] **Step 4: Commit and push**

```bash
git add docs/BPDD_FDR_METHOD_FORMAL100_REPORT_2026-08-10_ZH.md \
        evidence/bpdd-formal-848f00cb
git commit -m "docs: publish BPDD method and Formal100 evidence"
git push origin codex/bpdd-fdr
```

- [ ] **Step 5: Verify remote publication**

Compare local `git rev-parse HEAD` with `git ls-remote origin refs/heads/codex/bpdd-fdr`, then open the raw GitHub Markdown and JSON URLs. Expected: local and remote OIDs are identical and all lightweight evidence is downloadable.
