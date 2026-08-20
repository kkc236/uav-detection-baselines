# AP-FDR Auto Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish complete AP-FDR Formal100 ablation evidence to a private GitHub Release after both sequential training runs succeed.

**Architecture:** A standalone Python publisher validates both run directories, builds deterministic manifests and tar archives, and publishes idempotent Release assets through the GitHub REST API. A small shell watcher waits for the existing `all.completed` marker and invokes the publisher without modifying the active training supervisor.

**Tech Stack:** Python 3.10, pytest, requests, POSIX shell, GitHub Releases REST API.

---

### Task 1: Validation and manifest core

**Files:**
- Create: `scripts/publish_ap_fdr_ablation.py`
- Create: `tests/test_publish_ap_fdr_ablation.py`

- [ ] **Step 1: Write failing tests for complete and incomplete runs**

Create fixtures with `results.csv`, `args.yaml`, weights, logs, dry-run and authority records. Assert that 99 epochs or a missing artifact raises `PublicationGateError`, while two 100-epoch fixtures return sorted manifests with SHA-256 and byte counts.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/test_publish_ap_fdr_ablation.py -q`

Expected: collection fails because `scripts.publish_ap_fdr_ablation` does not exist.

- [ ] **Step 3: Implement the minimum validation and manifest functions**

Implement `completed_epochs`, `sha256_file`, `validate_variant`, `build_publication_manifest`, and atomic JSON writing. Accept only exactly 100 continuous epoch identifiers in either zero-based `0..99` or one-based `1..100` form.

- [ ] **Step 4: Run the focused tests and confirm GREEN**

Run: `python -m pytest tests/test_publish_ap_fdr_ablation.py -q`

Expected: all focused tests pass.

### Task 2: Archive and GitHub Release client

**Files:**
- Modify: `scripts/publish_ap_fdr_ablation.py`
- Modify: `tests/test_publish_ap_fdr_ablation.py`

- [ ] **Step 1: Write failing tests for deterministic asset names and idempotent upload**

Use a fake session to assert: absent assets upload once; same-name/same-size assets skip; same-name/different-size assets delete then upload; API errors raise without creating a success record.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/test_publish_ap_fdr_ablation.py -q`

Expected: upload/client tests fail because the API functions do not exist.

- [ ] **Step 3: Implement archives, Release creation and verified uploads**

Add `build_variant_archive`, `github_session`, `get_or_create_release`, `upload_asset`, `verify_assets`, and CLI arguments for base directory, token file, repository, tag, source commit and check-only mode. Write `publication-status.json` only after all remote assets match expected sizes.

- [ ] **Step 4: Run focused and publisher-related regression tests**

Run: `python -m pytest tests/test_publish_ap_fdr_ablation.py tests/test_result_publisher.py -q`

Expected: all selected tests pass.

### Task 3: Watcher and deployment contract

**Files:**
- Create: `scripts/watch_and_publish_ap_fdr_ablation.sh`
- Modify: `tests/test_publish_ap_fdr_ablation.py`

- [ ] **Step 1: Write a failing static contract test for the watcher**

Assert that the watcher waits for `all.completed`, uses the pinned repository/tag/source commit, does not contain a shutdown command, and invokes the publisher with a token-file argument rather than an inline token.

- [ ] **Step 2: Run the watcher test and confirm RED**

Run: `python -m pytest tests/test_publish_ap_fdr_ablation.py -q`

Expected: watcher contract test fails because the shell script is absent.

- [ ] **Step 3: Implement the watcher**

Poll every 60 seconds, exit early if `publication-succeeded` exists, invoke the publisher after `all.completed`, record output in `logs/auto-publication.log`, create `publication-succeeded` only on exit code zero, and retry a failed publication up to five times with increasing delays.

- [ ] **Step 4: Run the focused tests and shell syntax check**

Run: `python -m pytest tests/test_publish_ap_fdr_ablation.py -q`

Run: `bash -n scripts/watch_and_publish_ap_fdr_ablation.sh`

Expected: tests pass and shell syntax exits zero.

### Task 4: Integrate and deploy

**Files:**
- Modify only through deployment: `/home/ubuntu/ap-fdr-ablation/source-ebb349ae`
- Create on server: `/home/ubuntu/ap-fdr-ablation/github_token`
- Create on server: `/home/ubuntu/ap-fdr-ablation/publication-watcher.pid`

- [ ] **Step 1: Run the full source regression suite**

Run: `python -m pytest -q`

Expected: all repository tests pass.

- [ ] **Step 2: Commit and push `codex/ap-fdr-auto-publish`**

Commit only the design, plan, publisher, watcher, and publisher tests. Push the branch to `kkc236/uav-detection-baselines`.

- [ ] **Step 3: update the server checkout without touching the running process**

Transfer a Git bundle containing the new branch, fetch it into the existing server checkout, and check out the publisher files without changing the loaded Python process or active run directory.

- [ ] **Step 4: install the credential and validate check-only behavior**

Create the token file with mode `0600`. Run publisher `--check-only`; expected result is a nonzero incomplete gate while training is in progress and no Release mutation occurs.

- [ ] **Step 5: test GitHub authentication without mutation**

Call `GET /repos/kkc236/icassp2027-fdr-bpdd-fia-material`; require HTTP 200 and `private=true` without logging the token.

- [ ] **Step 6: launch and verify the watcher**

Start with `nohup`, disconnect and reconnect SSH, verify watcher PID is reparented and alive, confirm the original training PID remains alive, inspect log paths, and ensure no success marker exists before completion.
