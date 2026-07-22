from pathlib import Path

import pytest
import torch
from torch import nn

from src.ebc_qp_protocol import (
    build_initial_state,
    dataset_signature,
    load_initial_state,
    select_hashed_subset,
    subset_signature,
    state_fingerprint,
    write_d2_subset,
)


def test_d2_subset_is_order_independent_exact_ten_percent_and_hash_locked(tmp_path: Path):
    images = []
    for index in range(20):
        image = tmp_path / "images" / "train" / f"image-{index:02d}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(str(index).encode("ascii"))
        images.append(image)

    first = select_hashed_subset(images, root=tmp_path, fraction=0.10)
    second = select_hashed_subset(reversed(images), root=tmp_path, fraction=0.10)

    assert len(first) == 2
    assert first == second
    assert subset_signature(first, root=tmp_path) == subset_signature(second, root=tmp_path)


def test_dataset_signature_covers_image_names_and_label_contents(tmp_path: Path):
    image = tmp_path / "images" / "train" / "one.jpg"
    label = tmp_path / "labels" / "train" / "one.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    before = dataset_signature(tmp_path)
    label.write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    after = dataset_signature(tmp_path)

    assert before["file_count"] == 2
    assert before["sha256"] != after["sha256"]


def test_d2_subset_file_is_stable_and_records_exact_signature(tmp_path: Path):
    image_root = tmp_path / "images" / "train"
    image_root.mkdir(parents=True)
    images = []
    for index in range(20):
        image = image_root / f"frame-{index:03d}.jpg"
        image.write_bytes(str(index).encode())
        images.append(image)

    record = write_d2_subset(reversed(images), root=tmp_path, output=tmp_path / "d2-train.txt", fraction=0.10)

    lines = (tmp_path / "d2-train.txt").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines == [str(path.resolve()) for path in select_hashed_subset(images, root=tmp_path, fraction=0.10)]
    assert record == {
        "count": 2,
        "fraction": 0.10,
        "sha256": subset_signature([Path(line) for line in lines], root=tmp_path),
    }


class _Control(nn.Module):
    def __init__(self):
        super().__init__()
        self.stock = nn.Linear(2, 2)


class _Method(_Control):
    def __init__(self):
        super().__init__()
        self.p2_adapter = nn.Linear(2, 1)


def test_initial_state_loads_identical_common_tensors_and_only_method_extras():
    torch.manual_seed(3)
    control = _Control()
    torch.manual_seed(7)
    method = _Method()
    artifact = build_initial_state(control.state_dict(), method.state_dict(), metadata={"seed": 0})

    restored_control = _Control()
    restored_method = _Method()
    load_initial_state(restored_control, artifact, include_innovation=False)
    load_initial_state(restored_method, artifact, include_innovation=True)

    for name, value in restored_control.state_dict().items():
        torch.testing.assert_close(value, restored_method.state_dict()[name])
        torch.testing.assert_close(value, artifact["common_state"][name])
    torch.testing.assert_close(
        restored_method.state_dict()["p2_adapter.weight"], artifact["innovation_state"]["p2_adapter.weight"]
    )
    assert artifact["metadata"] == {"seed": 0}
    assert artifact["fingerprints"]["common"]
    assert artifact["fingerprints"]["innovation"]


def test_initial_state_rejects_non_p2_method_parameters():
    control = _Control()
    method = _Method()
    method.register_parameter("unapproved", nn.Parameter(torch.ones(1)))

    with pytest.raises(ValueError, match="unapproved innovation state"):
        build_initial_state(control.state_dict(), method.state_dict(), metadata={})


def test_state_fingerprint_accepts_scalar_long_buffers():
    first = state_fingerprint({"num_batches_tracked": torch.tensor(0, dtype=torch.long)})
    second = state_fingerprint({"num_batches_tracked": torch.tensor(1, dtype=torch.long)})

    assert first != second
