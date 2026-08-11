# ICASSP 2027 Chinese Paper Template Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成一份以 FDR、BPDD、RA-GLGM 为三个成立贡献的 ICASSP 2027 中文详细论文底稿，并严格区分冻结结果、初步结果和成功情景预估。

**Architecture:** 以现有证据文档为唯一事实来源，先构建证据与写作声明映射，再生成问题驱动的完整中文正文模板。模板使用统一结果令牌维护尚未冻结的 RA-GLGM 与 Full Model 数值，并通过自动扫描阻止未标记数字、缺失章节和证据边界混淆。

**Tech Stack:** Markdown、PowerShell、Git、现有实验 JSON/Markdown 证据。

---

## File Structure

- Create: `docs/ICASSP2027_FDR_BPDD_RA_GLGM_CHINESE_PAPER_TEMPLATE_ZH.md` — 中文论文正文、公式、表格、图示说明和可替换结果令牌。
- Create: `docs/ICASSP2027_FDR_BPDD_RA_GLGM_EVIDENCE_MAP_ZH.md` — 每项声明到本地证据、状态和可用措辞的映射。
- Read: `docs/CCF_B_PAPER_MATERIALS_MASTER_2026-08-10_ZH.md` — 总体研究材料和失败实验边界。
- Read: `docs/FDR_RTDETR_L_STRICT_CONTROL_AND_RESULTS_2026-08-09_ZH.md` — FDR严格对照结果。
- Read: `research/bpdd/BPDD_AUTHORITY.md` — BPDD定义与机制权威。
- Read: `docs/ICASSP2027_FDR_BPDD_RA_GLGM_ADVERSARIAL_AUDIT_2026-08-10_ZH.md` — 审稿风险和硬门槛。
- Read: `docs/superpowers/specs/2026-08-11-icassp2027-chinese-paper-template-design.md` — 已批准结构规范。

### Task 1: Build the Evidence Map

**Files:**
- Create: `docs/ICASSP2027_FDR_BPDD_RA_GLGM_EVIDENCE_MAP_ZH.md`
- Read: `docs/CCF_B_PAPER_MATERIALS_MASTER_2026-08-10_ZH.md`
- Read: `docs/FDR_RTDETR_L_STRICT_CONTROL_AND_RESULTS_2026-08-09_ZH.md`
- Read: `research/bpdd/BPDD_AUTHORITY.md`

- [ ] **Step 1: Extract the frozen FDR and BPDD values**

Run:

```powershell
rg -n "0\.21911|0\.28966|0\.2922579|0\.002598|0\.005570|Precision|Recall|AP50|AP75|mAP" docs/FDR_RTDETR_L_STRICT_CONTROL_AND_RESULTS_2026-08-09_ZH.md docs/CCF_B_PAPER_MATERIALS_MASTER_2026-08-10_ZH.md research/bpdd/BPDD_AUTHORITY.md
```

Expected: the output contains the strict Control/FDR comparison and the BPDD preliminary cross-authority comparison without introducing numbers from screenshots alone.

- [ ] **Step 2: Create the evidence map**

Use `apply_patch` to create a table with these columns:

```markdown
| Claim ID | Component | Planned claim | Evidence status | Numeric source | Allowed wording | Forbidden wording |
|---|---|---|---|---|---|---|
| C-FDR-1 | FDR | Continuous box regression is replaced by cumulative four-side distributions | Frozen | FDR authority and source commit | structural adaptation | original invention of FDR |
| C-FDR-2 | FDR | mAP improves from 0.21911 to 0.28966 | Frozen | strict Control100/FDR100 | strict seed0 result | universal gain |
| C-BPDD-1 | BPDD | target-wise decoder trajectory supervision improves AP75 | Preliminary | BPDD100 versus existing FDR authority | preliminary positive evidence | strict paired Formal100 conclusion |
| C-RA-1 | RA-GLGM | scale-routed P3 enhancement adds about 0.5 pp mAP | Planning estimate | successful-scenario assumption | expected contribution used to organize the draft | measured result |
| C-FULL-1 | Full | three components are complementary | Pending interaction proof | Full ablation matrix | hypothesis to be tested | verified synergy |
```

- [ ] **Step 3: Verify evidence-state completeness**

Run:

```powershell
$p='docs/ICASSP2027_FDR_BPDD_RA_GLGM_EVIDENCE_MAP_ZH.md'
$t=Get-Content -Raw -Encoding UTF8 $p
@('Frozen','Preliminary','Planning estimate','Pending interaction proof') | ForEach-Object { if (-not $t.Contains($_)) { throw "missing evidence state: $_" } }
```

Expected: exit code 0.

- [ ] **Step 4: Commit the evidence map**

```powershell
git add docs/ICASSP2027_FDR_BPDD_RA_GLGM_EVIDENCE_MAP_ZH.md
git commit -m "docs: map evidence for ICASSP paper template"
```

### Task 2: Draft the Front Matter and Problem-Driven Narrative

**Files:**
- Create: `docs/ICASSP2027_FDR_BPDD_RA_GLGM_CHINESE_PAPER_TEMPLATE_ZH.md`
- Read: `docs/ICASSP2027_FDR_BPDD_RA_GLGM_EVIDENCE_MAP_ZH.md`

- [ ] **Step 1: Create the title, abstract, keywords, and introduction**

Use `apply_patch` to write:

```markdown
# 面向无人机小目标检测的尺度感知细粒度分布回归与渐进式解码监督

> English title candidate: Scale-Aware Fine-Grained Distribution Regression with Progressive Decoder Distillation for UAV Small Object Detection

## 摘要

本文从边界表示、解码优化与尺度表征三个互补维度改进 RT-DETR-L。摘要中的 FDR 数值引用冻结结果；BPDD 使用“初步结果表明”；RA-GLGM 使用“成功情景预估约带来 0.5 pp 增益”，并明确最终稿必须替换为严格实验值。

## 1 引言

### 场景痛点
### 现有实时 DETR 的局限
### 统一解决思路
### 本文贡献
```

贡献段必须包含且只包含以下三项：

1. 面向 Ultralytics RT-DETR-L 的细粒度分布定位适配；
2. 目标级渐进式 Decoder 蒸馏与动态轨迹监督；
3. 面向高分辨率层的尺度路由局部—全局增强模块。

- [ ] **Step 2: Add evidence annotations**

Every paragraph containing a result must end with one of these source comments:

```markdown
<!-- EVIDENCE: FROZEN -->
<!-- EVIDENCE: PRELIMINARY -->
<!-- EVIDENCE: PLANNING_ESTIMATE -->
<!-- EVIDENCE: PENDING -->
```

- [ ] **Step 3: Verify the narrative structure**

Run:

```powershell
$p='docs/ICASSP2027_FDR_BPDD_RA_GLGM_CHINESE_PAPER_TEMPLATE_ZH.md'
$t=Get-Content -Raw -Encoding UTF8 $p
@('## 摘要','## 1 引言','边界表示','解码优化','尺度表征','<!-- EVIDENCE:') | ForEach-Object { if (-not $t.Contains($_)) { throw "missing narrative element: $_" } }
```

Expected: exit code 0.

- [ ] **Step 4: Commit the front matter**

```powershell
git add docs/ICASSP2027_FDR_BPDD_RA_GLGM_CHINESE_PAPER_TEMPLATE_ZH.md
git commit -m "docs: draft ICASSP paper narrative"
```

### Task 3: Write the Method Section

**Files:**
- Modify: `docs/ICASSP2027_FDR_BPDD_RA_GLGM_CHINESE_PAPER_TEMPLATE_ZH.md`
- Read: `research/bpdd/BPDD_AUTHORITY.md`
- Read: `docs/FDR_YAML_DECLARATIVE_MODULE.md`

- [ ] **Step 1: Add the overall architecture**

Describe the exact data flow:

```text
Image -> Backbone -> P3 RA-GLGM -> Hybrid Encoder -> Transformer Decoder
      -> preliminary box -> six-stage cumulative FDR -> final prediction
Decoder stage boxes + train GT -> BPDD teacher selection -> training-only loss
```

State explicitly that BPDD is absent at inference.

- [ ] **Step 2: Add the FDR subsection**

Include equations for distribution logits, Integral expectation, cumulative refinement, FGL, and preliminary-box supervision. Attribute the base FDR/FGL mechanism to D-FINE and frame the contribution as RT-DETR-L adaptation.

- [ ] **Step 3: Add the BPDD subsection**

Define per-target decoder candidates, quality utility, same-target validity, dynamic teacher selection, quality gate, detached teacher, and weighted distillation loss. Keep stock matching and detector losses unchanged.

- [ ] **Step 4: Add the RA-GLGM subsection**

Define P3 input, local branch, global context branch, true-scale router, bounded gate, and identity-initialized residual:

```math
Y = X + \alpha \cdot g_s(X, s) \odot \Phi_{lg}(X), \qquad \alpha(0)=0.
```

Explain that RA-GLGM is assumed successful for manuscript planning while final numeric claims remain replaceable.

- [ ] **Step 5: Add the unified objective**

Use a single equation:

```math
\mathcal{L}=\mathcal{L}_{RT-DETR}+\lambda_{fgl}\mathcal{L}_{FGL}+\lambda_{pre}\mathcal{L}_{pre}+\lambda_{bpdd}\mathcal{L}_{BPDD}.
```

RA-GLGM has no separate loss unless the final implementation contains a verified auxiliary objective.

- [ ] **Step 6: Verify method boundaries**

Run:

```powershell
$p='docs/ICASSP2027_FDR_BPDD_RA_GLGM_CHINESE_PAPER_TEMPLATE_ZH.md'
$t=Get-Content -Raw -Encoding UTF8 $p
@('D-FINE','BPDD','training-only','RA-GLGM','P3','\mathcal{L}_{BPDD}') | ForEach-Object { if (-not $t.Contains($_)) { throw "missing method boundary: $_" } }
```

Expected: exit code 0.

- [ ] **Step 7: Commit the method section**

```powershell
git add docs/ICASSP2027_FDR_BPDD_RA_GLGM_CHINESE_PAPER_TEMPLATE_ZH.md
git commit -m "docs: add ICASSP method template"
```

### Task 4: Add Experiments, Tables, and Case Design

**Files:**
- Modify: `docs/ICASSP2027_FDR_BPDD_RA_GLGM_CHINESE_PAPER_TEMPLATE_ZH.md`
- Read: `docs/EXPERIMENT_CONTROL_PROTOCOL.md`
- Read: `docs/FDR_RTDETR_L_STRICT_CONTROL_AND_RESULTS_2026-08-09_ZH.md`

- [ ] **Step 1: Add the fixed experimental protocol**

Include Ultralytics 8.4.90, RTX 4090, Python 3.10.12, PyTorch 2.5.1+cu121, VisDrone train/val, imgsz 640, batch 8, MuSGD, AMP scale 128, seed0, pretrained false, 100 epochs, query/max_det 300, and NMS false.

- [ ] **Step 2: Add the main results table**

Use rows for RT-DETR-L, RT-DETR-L+FDR, FDR+BPDD, FDR+RA-GLGM, and Full Model. Fill the strict Control/FDR values, mark BPDD as preliminary, and use these explicit planning tokens for future replacement:

```text
{{RA_MAP}}, {{RA_DELTA_MAP}}, {{FULL_MAP}}, {{FULL_DELTA_MAP}},
{{FULL_AP50}}, {{FULL_AP75}}, {{FULL_PRECISION}}, {{FULL_RECALL}}
```

The prose may state the planning hypothesis `{{RA_DELTA_MAP}} ≈ +0.5 pp`, but must not label it as measured.

- [ ] **Step 3: Add the ablation table**

Use the five-row matrix from the design spec and add columns for AP50, AP75, mAP50-95, Params, GFLOPs, and latency.

- [ ] **Step 4: Add scale, class, and efficiency tables**

Create separate compact tables for APtiny/APsmall/APmedium/APlarge, ten VisDrone categories, and deployment cost. Every unknown entry uses a named token rather than a fabricated value.

- [ ] **Step 5: Add qualitative case protocol**

Define deterministic case selection by fixed category and error-difference rules for tiny pedestrian, dense vehicles, occlusion, and scale variation. Require GT/RT-DETR-L/FDR/Full side-by-side boxes and include one failure case.

- [ ] **Step 6: Add the conclusion template**

Summarize the three contributions and state that final claims are conditional on replacing all result tokens with frozen evidence.

- [ ] **Step 7: Verify all required experiment blocks**

Run:

```powershell
$p='docs/ICASSP2027_FDR_BPDD_RA_GLGM_CHINESE_PAPER_TEMPLATE_ZH.md'
$t=Get-Content -Raw -Encoding UTF8 $p
@('主结果','消融','APtiny','APsmall','10类','GFLOPs','延迟','失败案例','{{RA_MAP}}','{{FULL_MAP}}') | ForEach-Object { if (-not $t.Contains($_)) { throw "missing experiment block: $_" } }
```

Expected: exit code 0.

- [ ] **Step 8: Commit the experiment template**

```powershell
git add docs/ICASSP2027_FDR_BPDD_RA_GLGM_CHINESE_PAPER_TEMPLATE_ZH.md
git commit -m "docs: add ICASSP experiment template"
```

### Task 5: Final Consistency and Format Verification

**Files:**
- Verify: `docs/ICASSP2027_FDR_BPDD_RA_GLGM_CHINESE_PAPER_TEMPLATE_ZH.md`
- Verify: `docs/ICASSP2027_FDR_BPDD_RA_GLGM_EVIDENCE_MAP_ZH.md`

- [ ] **Step 1: Verify UTF-8 and Markdown hygiene**

Run:

```powershell
$files=@(
  'docs/ICASSP2027_FDR_BPDD_RA_GLGM_CHINESE_PAPER_TEMPLATE_ZH.md',
  'docs/ICASSP2027_FDR_BPDD_RA_GLGM_EVIDENCE_MAP_ZH.md'
)
$utf8=[System.Text.UTF8Encoding]::new($false,$true)
foreach($p in $files){
  $t=[System.IO.File]::ReadAllText((Resolve-Path $p),$utf8)
  if($t.Contains([char]0xFFFD)){throw "replacement character: $p"}
  if([regex]::IsMatch($t,'(?m)[ \t]+$')){throw "trailing whitespace: $p"}
}
git diff --check
```

Expected: no exception and no whitespace errors.

- [ ] **Step 2: Verify claim boundaries**

Run:

```powershell
$p='docs/ICASSP2027_FDR_BPDD_RA_GLGM_CHINESE_PAPER_TEMPLATE_ZH.md'
$t=Get-Content -Raw -Encoding UTF8 $p
if($t -match '原创.{0,12}(FDR|FGL|Integral)'){throw 'overclaiming D-FINE mechanism'}
if(-not $t.Contains('PLANNING_ESTIMATE')){throw 'missing planning-estimate marker'}
if(-not $t.Contains('{{RA_DELTA_MAP}}')){throw 'missing RA replacement token'}
```

Expected: exit code 0.

- [ ] **Step 3: Produce a token inventory**

Run:

```powershell
$p='docs/ICASSP2027_FDR_BPDD_RA_GLGM_CHINESE_PAPER_TEMPLATE_ZH.md'
[regex]::Matches((Get-Content -Raw -Encoding UTF8 $p),'\{\{[A-Z0-9_]+\}\}') | ForEach-Object Value | Sort-Object -Unique
```

Expected: every unresolved result appears as a named token and no anonymous blank cells remain.

- [ ] **Step 4: Commit final verification adjustments**

```powershell
git add docs/ICASSP2027_FDR_BPDD_RA_GLGM_CHINESE_PAPER_TEMPLATE_ZH.md docs/ICASSP2027_FDR_BPDD_RA_GLGM_EVIDENCE_MAP_ZH.md
git commit -m "docs: finalize ICASSP Chinese paper template"
```

## Completion Criteria

- The Chinese draft has every ICASSP paper section and can be translated paragraph by paragraph.
- FDR, BPDD, and RA-GLGM are presented as three successful planned contributions.
- FDR uses frozen strict values; BPDD remains visibly preliminary until strict pairing; RA-GLGM uses a visible `about +0.5 pp` planning scenario.
- No planning estimate is indistinguishable from measured evidence.
- Main, ablation, scale, class, efficiency, and qualitative evaluation structures are present.
- All files pass UTF-8, whitespace, token, and claim-boundary checks.
