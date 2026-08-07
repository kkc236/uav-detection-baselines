# RT-DETR Audit Report Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the 2026-08-08 RT-DETR research audit report so every SCADS/FDR statement matches the latest uploaded GitHub evidence and the newly frozen exploratory decision boundary.

**Architecture:** Preserve the existing report structure and evidence hierarchy. Revise only claims affected by the latest SCADS Screen30 audit: separate the failed preregistered mechanism Gate from the positive detector screen, add missing oracle/router/tail-window evidence, and define a prospective exploratory Formal100 path without claiming that such a run has started.

**Tech Stack:** UTF-8 Markdown, Git, PowerShell, GitHub remote refs and immutable CSV/JSON evidence.

---

### Task 1: Freeze the remote evidence snapshot

**Files:**
- Read: `reports/scads-screen-v1/gate-report.json` at `origin/codex/scads-publisher-fix@51b8e38d`
- Read: `reports/scads-screen-v1/core-metrics.csv` at `origin/codex/scads-publisher-fix@51b8e38d`
- Read: `reports/scads-screen-v1/training-windows.csv` at `origin/codex/scads-publisher-fix@51b8e38d`
- Read: `reports/scads-screen-v1/mechanism-metrics.csv` at `origin/codex/scads-publisher-fix@51b8e38d`
- Read: `reports/scads-screen-v1/efficiency-metrics.csv` at `origin/codex/scads-publisher-fix@51b8e38d`

- [ ] **Step 1: Confirm remote tips**

Run:

```powershell
git fetch --prune --tags origin
git for-each-ref --sort=-committerdate --format='%(committerdate:iso8601)|%(objectname:short)|%(refname:short)|%(subject)' refs/remotes/origin
```

Expected: SCADS implementation at `c93855fe`, final report at `51b8e38d`, and no later SCADS Formal100 branch or commit.

- [ ] **Step 2: Confirm immutable Gate identity**

Run:

```powershell
git show origin/codex/scads-publisher-fix:reports/scads-screen-v1/SHA256SUMS.txt
```

Expected: `gate-report.json` SHA-256 equals `7F86BD000CC12B8069941709BB8E04C8EF3F6E4E3A22F5B700DF52F92004002E`.

### Task 2: Rewrite the SCADS evidence interpretation

**Files:**
- Modify: `C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\RTDETR_导师审计版研究进展与证据报告_2026-08-08.md`

- [ ] **Step 1: Revise the executive summary and timeline**

Replace the absolute statement that SCADS is simply a failed method with a two-layer conclusion:

```text
The preregistered 50% tiny-saturation mechanism Gate failed and remains failed.
The paired detector screen is consistently positive and supports a separately preregistered exploratory Formal100 if the research objective is stable detector gain.
No SCADS Formal100 upload exists as of the evidence snapshot.
```

- [ ] **Step 2: Expand the detector and training-window tables**

Add the exact unified-evaluation delta `+0.002401663995` and distinguish it from the epoch-30 training log delta `+0.00256`. Add tail-3 mAP/AP50/AP75 deltas `+0.00211`, `+0.004623333333`, and `+0.001938544168`.

- [ ] **Step 3: Add mechanism ceiling evidence**

Record:

```text
fixed-base tiny edge saturation = 0.235525682569
adaptive tiny edge saturation = 0.163766205311
oracle tiny edge saturation = 0.121015362442
adaptive relative reduction = 30.4677929%
oracle relative reduction = 48.6190376%
50% target rate = 0.117762841284
```

Explain that the oracle itself misses the frozen threshold, so the old Gate cannot be relabelled as passed, but the threshold is not a valid reason to erase positive detector evidence.

- [ ] **Step 4: Add router diagnostics and efficiency boundary**

Record route accuracy `0.3807524`, balanced accuracy `0.4009534`, entropy `0.9741861`, wide-overflow rate `0.1511377`, and the `+5.62861%` median-latency increase. State that the router is nonconstant but weakly aligned with oracle targets.

### Task 3: Update claims, next actions, and final audit opinion

**Files:**
- Modify: `C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\RTDETR_导师审计版研究进展与证据报告_2026-08-08.md`

- [ ] **Step 1: Update allowed and forbidden claims**

Allowed: positive SCADS Screen30 detector evidence, unchanged old Gate failure, and exploratory long-training eligibility under a newly frozen objective.

Forbidden: claiming the old Gate passed, claiming 50% saturation relief was achieved, claiming SCADS Formal100 has started or completed, or comparing a new SCADS run to a non-paired historical FDR checkpoint as a strict causal result.

- [ ] **Step 2: Replace the P4 action**

Set the next action to:

```text
Freeze a new exploratory paired Formal100 protocol whose primary criteria are positive final mAP50-95, positive tail-3 mAP50-95, and positive final AP75; run fresh full-data seed0 FDR and SCADS arms from the same authority. Keep the old 50% mechanism Gate in the record as failed.
```

- [ ] **Step 3: Revise the final audit paragraph**

Position FDR as the mature main-module candidate and SCADS as a promising small-module candidate with positive detector evidence, incomplete mechanism confirmation, and no uploaded Formal100 result yet.

### Task 4: Verify the final report

**Files:**
- Verify: `C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\RTDETR_导师审计版研究进展与证据报告_2026-08-08.md`

- [ ] **Step 1: Check contradictory phrases**

Run:

```powershell
rg -n 'SCADS.*不得启动|SCADS.*严格失败|SCADS.*Formal100.*已启动|SCADS.*Formal100.*已完成|formal100_eligible=true' 'C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\RTDETR_导师审计版研究进展与证据报告_2026-08-08.md'
```

Expected: no unsupported positive Formal100 claim and no unconditional prohibition that contradicts the new exploratory path.

- [ ] **Step 2: Check required evidence values**

Run:

```powershell
rg -n '0\.002401|0\.00256|0\.00211|48\.619|30\.47|0\.380752|0\.400953|5\.63%' 'C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\RTDETR_导师审计版研究进展与证据报告_2026-08-08.md'
```

Expected: every required detector, mechanism, router, and efficiency value appears in a correctly labelled context.

- [ ] **Step 3: Check Markdown and encoding**

Run:

```powershell
$f='C:\Users\16946\Documents\OBJECTIVE CHECK PAPER\RTDETR_导师审计版研究进展与证据报告_2026-08-08.md'
Get-Content -LiteralPath $f -Raw -Encoding UTF8 | Out-Null
rg -n '^#{1,6} ' $f
```

Expected: UTF-8 read succeeds and heading levels remain ordered within the original section hierarchy.
