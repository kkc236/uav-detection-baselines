# PR-IRA Late-Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a protocol-bound Formal100 epoch-61 private-parameter freeze to PR-IRA while leaving Screen30, forward inference, FDR, BPDD, and all public optimizer behavior unchanged.

**Architecture:** Encode private-update windows in the immutable PR-IRA protocol and expose one fail-closed pure helper for schedule decisions. Extend the trainer's existing identity-gradient suppression so both identity and late-freeze epochs set only PR-IRA private gradients to `None` after BPDD firewall subtraction and AMP unscale.

**Tech Stack:** Python 3.10, PyTorch 2.5.1, Ultralytics 8.4.90, pytest, YAML, Git.

---

## File map

- `src/pr_ira_protocol.py`: immutable schedule fields, protocol hash, and pure update-window validator.
- `src/rtdetr_fdr_bpdd_pr_ira.py`: trainer-side private-gradient suppression using the protocol helper.
- `tests/test_pr_ira_protocol.py`: protocol shape, boundary, type, and fail-closed tests.
- `tests/test_pr_ira_trainer_contract.py`: gradient, optimizer momentum/decay, and public/private isolation tests.
- `docs/superpowers/specs/2026-08-14-pr-ira-late-freeze-design.md`: scientific and engineering design authority.

### Task 1: Freeze private-update windows in the protocol

**Files:**
- Modify: `tests/test_pr_ira_protocol.py`
- Modify: `src/pr_ira_protocol.py`

- [ ] **Step 1: Write failing protocol-shape and boundary tests**

Add imports and tests that require the exact schedule fields and boundary behavior:

```python
from src.pr_ira_protocol import pr_ira_private_update_enabled


@pytest.mark.parametrize(
    ("epoch", "epochs", "expected"),
    [
        (3, 30, False),
        (4, 30, True),
        (30, 30, True),
        (10, 100, False),
        (11, 100, True),
        (60, 100, True),
        (61, 100, False),
        (100, 100, False),
    ],
)
def test_private_update_window_is_frozen(
    epoch: int,
    epochs: int,
    expected: bool,
) -> None:
    assert pr_ira_private_update_enabled(epoch, epochs) is expected


@pytest.mark.parametrize(
    ("epoch", "epochs"),
    [(0, 30), (31, 30), (0, 100), (101, 100), (1, 50)],
)
def test_private_update_window_rejects_unknown_or_out_of_range_protocols(
    epoch: int,
    epochs: int,
) -> None:
    with pytest.raises(ValueError):
        pr_ira_private_update_enabled(epoch, epochs)


@pytest.mark.parametrize("value", [True, False, 1.0, "1"])
def test_private_update_window_rejects_non_integer_epoch_types(value: object) -> None:
    with pytest.raises(TypeError):
        pr_ira_private_update_enabled(value, 30)  # type: ignore[arg-type]
```

Update the exact `PR_IRA_PROTOCOL["pr_ira"]` assertion to include `private_update` and `private_frozen` fields from the design.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
python -m pytest tests/test_pr_ira_protocol.py -q
```

Expected: collection or assertion failures because `pr_ira_private_update_enabled` and the new protocol fields do not exist.

- [ ] **Step 3: Implement the fail-closed protocol helper**

In `src/pr_ira_protocol.py`, add the exact schedule fields and this behavior:

```python
def pr_ira_private_update_enabled(epoch: int, epochs: int) -> bool:
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        raise TypeError("epoch must be an integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int):
        raise TypeError("epochs must be an integer")
    if epochs == 30:
        schedule_name = "screen30"
    elif epochs == 100:
        schedule_name = "formal100"
    else:
        raise ValueError("PR-IRA private-update protocol supports only 30 or 100 epochs")
    if epoch < 1 or epoch > epochs:
        raise ValueError("epoch is outside the frozen PR-IRA schedule")
    start, end = PR_IRA_PROTOCOL["pr_ira"]["schedule"][schedule_name]["private_update"]
    return int(start) <= epoch <= int(end)
```

Export it through `__all__`.

- [ ] **Step 4: Run protocol tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_pr_ira_protocol.py -q
```

Expected: all protocol tests pass.

- [ ] **Step 5: Commit the protocol change**

```powershell
git add src/pr_ira_protocol.py tests/test_pr_ira_protocol.py
git commit -m "feat: freeze PR-IRA private update schedule"
```

### Task 2: Extend private-gradient suppression to the late-freeze phase

**Files:**
- Modify: `tests/test_pr_ira_trainer_contract.py`
- Modify: `src/rtdetr_fdr_bpdd_pr_ira.py`

- [ ] **Step 1: Write failing trainer boundary tests**

Replace the identity-only test with a stage-aware test that sets `trainer.epoch` and `trainer.epochs`, then verifies:

```python
@pytest.mark.parametrize(
    ("epoch_zero_based", "epochs", "suppressed"),
    [(2, 30, True), (3, 30, False), (29, 30, False),
     (9, 100, True), (10, 100, False), (59, 100, False),
     (60, 100, True), (99, 100, True)],
)
def test_private_gradients_are_suppressed_outside_update_window(
    combined_model: FDRBPDDPRIRADetectionModel,
    epoch_zero_based: int,
    epochs: int,
    suppressed: bool,
) -> None:
    trainer = FDRBPDDPRIRATrainer.__new__(FDRBPDDPRIRATrainer)
    trainer.model = combined_model
    trainer.epoch = epoch_zero_based
    trainer.epochs = epochs
    private = combined_model.pr_ira_private_parameters()
    private_ids = {id(parameter) for parameter in private}
    public = next(
        parameter for parameter in combined_model.parameters()
        if parameter.requires_grad and id(parameter) not in private_ids
    )
    for parameter in private:
        parameter.grad = torch.ones_like(parameter)
    public.grad = torch.ones_like(public)

    assert trainer.suppress_pr_ira_inactive_gradients() is suppressed
    assert all((parameter.grad is None) is suppressed for parameter in private)
    assert public.grad is not None
```

In the same RED phase, add a real SGD regression test. Build momentum state with
one active update, snapshot one private parameter and its momentum buffer, move
the trainer to Formal100 epoch 61, assign fresh private/public gradients, invoke
the new suppression API, and perform another optimizer step:

```python
private_before = private_parameter.detach().clone()
private_momentum_before = optimizer.state[private_parameter]["momentum_buffer"].clone()
public_before = public_parameter.detach().clone()
assert trainer.suppress_pr_ira_inactive_gradients() is True
optimizer.step()
assert torch.equal(private_parameter, private_before)
assert torch.equal(
    optimizer.state[private_parameter]["momentum_buffer"],
    private_momentum_before,
)
assert not torch.equal(public_parameter, public_before)
```

- [ ] **Step 2: Run the boundary test and verify RED**

Run:

```powershell
python -m pytest tests/test_pr_ira_trainer_contract.py -q
```

Expected: failure because `suppress_pr_ira_inactive_gradients` does not exist;
without the new suppression behavior, momentum and weight decay would also move
the private tensor at epoch 61.

- [ ] **Step 3: Implement minimal stage-aware suppression**

Import `pr_ira_private_update_enabled` in `src/rtdetr_fdr_bpdd_pr_ira.py`, replace `suppress_pr_ira_identity_gradients` with:

```python
def suppress_pr_ira_inactive_gradients(self) -> bool:
    model = self._firewall_model()
    current_epoch = int(self.epoch) + 1
    if pr_ira_private_update_enabled(current_epoch, int(self.epochs)):
        return False
    for parameter in model.pr_ira_private_parameters():
        parameter.grad = None
    return True
```

Call this method in `optimizer_step` at the existing suppression point, after firewall subtraction and before global clipping.

- [ ] **Step 4: Run trainer tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_pr_ira_trainer_contract.py -q
```

Expected: all trainer-contract tests pass, including bit-exact private parameter
and momentum-state preservation at epoch 61 while the public parameter updates.

- [ ] **Step 5: Commit the trainer change**

```powershell
git add src/rtdetr_fdr_bpdd_pr_ira.py tests/test_pr_ira_trainer_contract.py
git commit -m "feat: freeze PR-IRA private state after epoch 60"
```

### Task 3: Run complete verification and publish the amended authority

**Files:**
- Verify: `src/pr_ira_protocol.py`
- Verify: `src/rtdetr_fdr_bpdd_pr_ira.py`
- Verify: `tests/test_pr_ira_protocol.py`
- Verify: `tests/test_pr_ira_trainer_contract.py`
- Verify: all PR-IRA/FDR/BPDD tests

- [ ] **Step 1: Run focused regression tests**

```powershell
python -m pytest tests/test_pr_ira.py tests/test_pr_ira_protocol.py tests/test_pr_ira_trainer_contract.py -q
```

Expected: zero failures; the CUDA-only test may skip on a CPU host.

- [ ] **Step 2: Run the previously validated complete suite**

```powershell
python -m pytest tests -q
```

Expected: zero failures. Record the exact pass/skip/warning counts from this fresh run rather than reusing the previous `180 passed, 1 skipped` evidence.

- [ ] **Step 3: Audit scope and protocol hash changes**

```powershell
git diff --check
git status --short
git diff --stat HEAD~2..HEAD
```

Expected: no whitespace errors; only the four intended source/test files and two design/plan documents are tracked by this amendment. The two pre-existing unrelated untracked documents remain untouched.

- [ ] **Step 4: Commit documentation if it is not already committed**

```powershell
git add docs/superpowers/specs/2026-08-14-pr-ira-late-freeze-design.md docs/superpowers/plans/2026-08-14-pr-ira-late-freeze.md
git commit -m "docs: freeze PR-IRA late-stage stabilization design"
```

- [ ] **Step 5: Push the exact requested branch**

```powershell
git push origin codex/bpdd-ira-final-eval-d7200906
git ls-remote origin refs/heads/codex/bpdd-ira-final-eval-d7200906
git rev-parse HEAD
```

Expected: the remote branch OID exactly equals local `HEAD`.
