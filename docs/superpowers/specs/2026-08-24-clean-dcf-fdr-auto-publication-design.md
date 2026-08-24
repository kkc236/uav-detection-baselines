# Clean FDR / DCF-FDR Formal100 Automatic Publication Design

Date: 2026-08-24

## 1. Objective

After the running DCF-FDR seed-0 Formal100 arm exits successfully, automatically
validate, package, publish, and remotely verify the complete Clean FDR versus
DCF-FDR ablation evidence. The publication target is the private repository
`kkc236/icassp2027-fdr-bpdd-fia-material`.

The publication must preserve both model-selection and training-end checkpoints
for both arms without putting large binaries in Git history.

## 2. Frozen experiment identity

- Source repository: `kkc236/uav-detection-baselines`
- Source branch: `codex/ap-fdr-integrated-redesign`
- Source commit: `ec4e2a463db7a53f7c4c9c4bc9edabdf5c39f40b`
- Dataset: `/data/uav/datasets/VisDrone`
- Frozen initial state:
  `/data/uav/protocols/fdr-d97e1eb7/initial-state.pt`
- Frozen initial-state SHA-256:
  `51aab2eb3fb7d123501c69c7b8dc90ff3ea0b9344a108edeef2c7d6dcdbb742d`
- Clean run root: `/data/uav/runs/dcf-fdr-ec4e2a46-clean`
- DCF run root: `/data/uav/runs/dcf-fdr-ec4e2a46-dcf`
- Expected epochs per arm: exactly 100, continuous

No publication process may modify these run directories or their artifacts.

## 3. Completion and validity gates

The watcher waits for the existing two-arm training chain to terminate. It may
publish only if all of the following are true:

1. The chain exits with status zero and no DCF trainer remains alive.
2. Both `results.csv` files contain exactly 100 continuous epochs.
3. Each arm contains readable `args.yaml`, authority JSON, `best.pt`, `last.pt`,
   and its complete training log.
4. Both checkpoints deserialize successfully on CPU and contain a model or EMA
   state suitable for later evaluation.
5. The source commit, dataset/run identity, seed, epoch count, and frozen initial
   state match this design.
6. The logs contain no terminal traceback, CUDA out-of-memory, non-finite-loss,
   or no-space-left failure.

Any failed gate blocks remote mutation and writes a local failure report. A
scientifically unfavorable DCF result does not block publication; negative and
null results are evidence and must be preserved honestly.

## 4. Derived evidence

The finalizer produces deterministic lightweight artifacts:

- untouched `results.csv`, `args.yaml`, and authority JSON for both arms;
- compressed complete training logs;
- per-arm completion manifests with byte sizes and SHA-256 hashes;
- best-mAP checkpoint metrics and final-epoch metrics;
- per-metric peak epoch/value for precision, recall, AP50, and mAP50-95;
- a 100-row aligned epoch comparison CSV;
- a canonical comparison JSON containing DCF minus Clean deltas;
- a concise Markdown result report that separates facts from interpretation;
- a release asset manifest binding every large asset to its SHA-256 and byte
  size.

The decision field is computed without rounding:

- `passed_nonnegative` if DCF best mAP50-95 is at least Clean best mAP50-95;
- `failed_negative` otherwise.

Precision, recall, and AP50 are reported as secondary health indicators and do
not silently override this registered primary rule.

## 5. GitHub layout

### 5.1 Lightweight Git evidence

An isolated checkout of the private material repository receives one new,
experiment-specific directory:

`experiments/clean-dcf-fdr-formal100-seed0-20260824/`

It contains the lightweight artifacts from Section 4. The publisher creates one
dedicated result commit on `main` and never rewrites history or edits unrelated
materials. If remote `main` advances, the isolated checkout fetches and rebases
the new result commit before retrying the push.

### 5.2 Large Release evidence

Release tag:

`clean-dcf-fdr-formal100-seed0-20260824`

The private GitHub Release contains these four checkpoints as separate assets:

- `clean-fdr-seed0-formal100-best.pt`
- `clean-fdr-seed0-formal100-last.pt`
- `dcf-fdr-seed0-formal100-best.pt`
- `dcf-fdr-seed0-formal100-last.pt`

It also contains the lightweight evidence bundle and release manifest. The
approximately 199 MB checkpoint files remain below GitHub Release's 2 GiB
per-asset limit. They are never added to Git history or Git LFS.

## 6. Authentication and remote verification

The publisher reads the GitHub credential only from the existing mode-600 token
file. The token must not appear in command arguments, remote URLs, logs,
manifests, commits, shell history, or process listings.

Before publication, the API must confirm that the destination repository is
private. After upload, the publisher re-queries GitHub and verifies:

- repository and tag identity;
- result commit existence on remote `main`;
- every expected Release asset name;
- every remote asset byte size;
- the downloadable manifest's expected hashes;
- the Release URL and result commit SHA recorded in local publication status.

Publication is successful only after all remote checks pass.

## 7. Retry, idempotency, and retention

- Network and GitHub failures retry with bounded exponential backoff.
- Rerunning the publisher is idempotent: identical assets are skipped; a
  same-name wrong-size asset is replaced and reverified.
- Existing unrelated releases, assets, branches, commits, runs, and checkpoints
  are never deleted.
- Local Clean/DCF artifacts remain on the server after successful publication.
- The token file remains in place; the publisher neither prints nor deletes a
  shared credential.
- The server is not shut down automatically.

After all retries fail, the watcher writes `publication-failed.json` with a
sanitized error and exits nonzero. A later manual rerun resumes publication from
the existing staged evidence.

## 8. Deployment and verification

The finalizer and watcher are versioned in the source repository and covered by
tests for completion gates, continuous epochs, metric extraction, deterministic
manifests, checkpoint validation, secret exclusion, idempotent asset handling,
and remote-verification logic.

Deployment uses a detached server process whose parent is PID 1, writes a local
watcher log, and records its PID. A dry-run must pass against the live Clean
artifacts and the still-incomplete DCF arm must be rejected before the watcher is
armed. Deployment verification checks that the watcher is alive, bound to the
expected source commit and paths, and waiting without interfering with DCF
training.
