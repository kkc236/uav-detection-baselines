# No-Server Correctness and Material Deployment Design

## 1. Objective

Prepare the current LRS-FDR research line for later GPU execution without claiming
that Formal100 training has been completed. The deployment has five bounded goals:

1. make feasible FDR decoding numerically positive under FP32, FP16, and BF16;
2. make the AC-BPDD better-only decision compare source and teacher in the same
   stable log-probability domain;
3. add the missing `f*` arm, defined as LRS-FDR plus the revised feasible decoder
   and no BPDD or FIA;
4. verify the revised code with synthetic, unit, integration, and local dry-run
   checks that do not require a training server;
5. publish runnable source in the code repository and a self-contained
   reproducibility snapshot in the private material repository.

Formal VisDrone or UAVDT accuracy results are explicitly outside this deployment.

## 2. Considered approaches

### A. Minimal in-place correctness repair (selected)

Keep the current method graph, fix the two demonstrated numerical defects, add
the missing comparator, and extend the existing launcher and authority schemas.
This minimizes method-identity drift and preserves the value of later paired
experiments.

### B. Replace the feasible projection and BPDD gate with new learned modules

This could create smoother constraints or a joint geometric teacher, but it would
introduce a new method generation and invalidate more existing evidence. It is not
appropriate before the current hypotheses are tested.

### C. Leave code unchanged and document the limitations

This preserves historical identity but knowingly permits zero decoded extents
under low precision and false better-teacher activation. It is unsuitable for
future Formal100 execution.

## 3. Source and repository boundaries

- Authoritative executable source remains in `uav-detection-baselines` on a new
  `codex/` branch derived from `codex/fdr-feasible-geometry-uavdt`.
- The private material repository does not become a second development tree. It
  receives:
  - an immutable source archive or patch for the deployed source commit;
  - source commit, tree, branch, remote, and SHA-256 metadata;
  - the design and implementation plan;
  - test commands and captured machine-readable results;
  - revised method/protocol notes and the four-arm experiment matrix.
- The material snapshot must be sufficient to identify and reconstruct the exact
  source, while avoiding a silently diverging editable copy of the whole codebase.

## 4. Feasible-decoding contract

### 4.1 Required behavior

For every FDR decode path, the final normalized `cxcywh` tensor consumed by box
losses must have finite width and height strictly greater than zero in FP32, FP16,
and BF16 execution. Feasible inputs retain the existing decoded values within the
precision-appropriate tolerance. Horizontal correction preserves horizontal
center; vertical correction preserves vertical center.

### 4.2 Numerical design

Integral output, extent repair, and conversion from edge distances to normalized
box coordinates are evaluated in FP32 when the incoming distribution tensor uses
FP16 or BF16. The repaired box remains FP32 through the geometry-sensitive loss
interface instead of being cast back before the positivity assertion. The minimum
extent is enforced in final normalized box coordinates, not only in the signed
FDR-distance coordinate system.

The implementation exposes both raw infeasible counts and repaired final minimum
width/height. Straight-through gradients may be retained only if tests show finite,
non-zero gradients for invalid inputs. This is a numerical containment mechanism,
not a guarantee that raw distributions learn feasibility; persistent raw-invalid
rates remain a runtime warning and publication diagnostic.

### 4.3 Non-goals

- no learned feasibility head;
- no change to FGL targets, Integral bins, or D-FINE-derived projection values;
- no claim that the repair is an independent paper contribution;
- no retrospective reuse of old LRS results as results of the revised decoder.

## 5. AC-BPDD log-domain contract

Source distributions, future candidates, and the mixed teacher are evaluated from
stable log probabilities. A future teacher whose true interpolated target-edge
NLL is worse than the source cannot receive positive reliability because of an
epsilon clamp. Epsilon may protect a final arithmetic operation, but it must not
replace a finite log probability with `log(epsilon)` before the comparison.

The gate retains its current narrow meaning: it accepts lower interpolated
target-edge NLL, not necessarily higher decoded IoU or GIoU. Runtime and paper
text must use that definition. Assignment consistency remains a conservative
candidate filter; stable-match and active-edge coverage must be reported before
claiming an AC-BPDD contribution.

No geometric gate, new teacher topology, DN-query BPDD, or temperature search is
introduced in this deployment.

## 6. Revised experiment identities

All current-system arms share the revised feasible decoder:

| Arm | Method identity |
| --- | --- |
| `f` | LRS-FDR + feasible geometry |
| `g` | `f` + AC-BPDD |
| `h` | `f` + FIA |
| `i` | `f` + AC-BPDD + FIA |

The new `f` is the previously described `f*`; the launcher uses `f` because arm
letters are public protocol identifiers. The primary contrasts are `g-f`, `h-f`,
`i-h`, and `i-g`. The interaction is `(i-h)-(g-f)`. Comparisons against the old
pre-repair LRS run are historical only.

The VisDrone launcher must accept `f`, produce a distinct immutable authority
identity, use the same initial-state and frozen public settings as `g/h/i`, and
support dry-run without constructing the trainer. No local run will be labeled a
Formal100 result.

## 7. Verification strategy

### 7.1 Test-first regressions

Before production changes, tests must demonstrate both current failures:

- extreme invalid distances decode to zero width/height in FP16 or BF16;
- a teacher with true NLL 40 can be accepted over a source with true NLL 20
  because the teacher path is clamped to `log(1e-6)`.

Each regression is then required to pass after its isolated implementation fix.

### 7.2 Required local checks

- focused feasible-geometry tests, including dtype and gradient cases;
- focused pure AC-BPDD tests, including extreme logits and no-teacher behavior;
- FDR/BPDD/FIA integration tests;
- protocol and launcher tests for all four arms;
- CPU dry-run of `f/g/h/i` using temporary synthetic authority inputs where the
  existing test harness supports them;
- full repository test suite if its runtime and installed optional dependencies
  permit it; otherwise every skipped or unavailable group is listed explicitly.

The installed default Python environment is CPU-only. The local RTX 4070 Laptop
GPU is therefore not used unless a separate environment is deliberately created;
creating that environment is not required for this source deployment.

## 8. Material-repository package

The material update creates one dated deployment directory containing:

- `SOURCE_AUTHORITY.json` with exact Git identities and file hashes;
- `TEST_REPORT.json` with commands, exit codes, pass/fail/skip counts, Python and
  PyTorch environment, and the explicit statement that no Formal100 ran;
- the source patch or archive and its SHA-256;
- the approved design and implementation plan;
- a Chinese deployment note explaining corrected claims, the `f/g/h/i` matrix,
  unresolved GPU work, and prohibited reuse of pre-repair numbers.

Existing historical results remain untouched. Current entry documents gain a
pointer to the new deployment package and may not relabel legacy BPDD/FIA numbers.

## 9. Commit and publication sequence

1. Commit this design alone.
2. Commit the implementation plan alone.
3. Implement each defect with red-green tests and focused commits.
4. Add the `f` protocol arm and its tests in a separate commit.
5. Run fresh verification and generate immutable deployment evidence.
6. Push the source branch to the code remote.
7. Commit the reproducibility package on the material freeze branch and push it
   to the private-material remote.
8. Verify both remote branch heads with `git ls-remote`.

No force push, history rewrite, tag replacement, checkpoint deletion, or formal
performance claim is authorized.

## 10. Acceptance criteria

- the two adversarial regressions fail before and pass after their fixes;
- low-precision final widths and heights are finite and strictly positive;
- a truly worse teacher has exactly zero BPDD reliability;
- `f/g/h/i` identities are distinct and dry-run-valid;
- all relevant tests pass, with unavailable checks disclosed;
- both working trees are clean after their commits;
- the source and material remote heads equal the verified local commits;
- the material package states that performance remains pending GPU execution.
