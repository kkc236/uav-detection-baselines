from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Mapping

import torch


EXPECTED_DATASET_SHA256 = "FD92E9FF4B3B58FCDD5A32F7E770FC3398E566B627DB0E188CB5FF9F3B7BBDAB"
EXPECTED_DATASET_FILE_COUNT = 14_038
EXPECTED_CATEGORY_MAPPING_SHA256 = "1455A1F5D9FA9988799815B36CF2CE6D5044B5BD7CAD7CC614D6B5E5059EF2A6"
EXPECTED_SUBSET_SHA256 = "52660F55552FFD953E2EE26F55FD0A1CB14217DBBEA0F5F3B981C3514F8D93A0"
EXPECTED_SUBSET_FILE_SHA256 = "4BDEE4F03CC903422ADBBF4BD3511027628000DB578DEFC07DFE6E45F1E7CB60"
EXPECTED_SUBSET_COUNT = 647
EXPECTED_PARENT_ATTESTATION_SHA256 = {
    0: "6DB896E33CC524F49B079B248782BA16C206BE19C027888408ED6A8B2CCAE409",
    1: "98FB41D7B321B5BD7099D9670FB19CEBDD387D72173D2B894EE2DE4EEEF05E0F",
    2: "3DBAD2E1894933B1D9411C2B385A3B0E98C69C862E9349B477FAC0D1A6E702B4",
}
EXPECTED_INITIAL_STATE_SHA256 = {
    0: "3F4EC67E4C760E6FB70999CD45399E5DE8CC7CBB2C870F28F8A58AD945CE3F75",
    1: "670CAAD3BD105556F9BF8026B1B9E62141FE1E8DA6E848404BBDA3C640F3FEFB",
    2: "E48136B48C97985B5C32DEE79D9FCC66F122E51A85726E52B2AB22F7E2120B96",
}
EXPECTED_COMMON_FINGERPRINTS = {
    0: "0B968046FDC89BE5A31581C81F7335A9742BC422503428113637B1CC829F0FA0",
    1: "A73D3A57F5DCF3F62FA4B30329C32204E3A74BC57AA4FFEE577873D14F0A3D65",
    2: "1CCA2D745106F949268B3978722A415439623376D01C1D188B1450C6230AF1B2",
}
EXPECTED_PARENT_SOURCE_SHA256 = {
    "head.py": "5701116D86881827AC9E1E7462DFAA44C33937BD68E23324763459685729E06F",
    "rtdetr-l.yaml": "85716F626769CB5DDF00D59FCF6CAFB5814AAD196328100BDC7C93306F650E83",
    "tasks.py": "B00935C1851BB9CEA240985704C12E654E68B369F6C59DE20E45FA295CB79B92",
}
EXPECTED_UPSTREAM_SOURCE_SHA256 = {
    **EXPECTED_PARENT_SOURCE_SHA256,
    "data/augment.py": "697492FD23C9D763B99ED65F56CF4EE5457732E30CAA10618A20A1ED96DE4B64",
    "data/build.py": "80264B44C8C3C5049699E6548A9DCD8A5A9F5B2C66D9EC8BFF58132F9498D5AB",
    "data/dataset.py": "6DA06C274091A27A6DFF9A71F928D8E765549591A85C399C5226AC917EB6F9FD",
    "engine/trainer.py": "256EB7680A361308D8E5B55DEAF8148280DDEA1FC734663A3C46FA491388F0D8",
    "models/rtdetr/model.py": "8C9CD287FE44FFFAD540EEFF2249A77290B88C876EF713622319534C8685255C",
    "models/rtdetr/train.py": "7B13E6B1EB7F0962B76417ABDDBB44BC32EDB6030F5EEB0CAF6D56B091776E48",
    "models/utils/loss.py": "265483A64AAB9DD63E56B3FA1A864B838916DDDDECBA21DAFB2121FC513BBFAF",
    "models/utils/ops.py": "EF211FAFC112A715305A8070F0053FAC38FAF4D0F9F28BC04D91C9A112451778",
    "nn/modules/block.py": "20EDA06BE7AD7FEA69DB8161C91F6681371AA6AEDB1DCAE18DB1A004783E1CBD",
    "nn/modules/conv.py": "C802F36EA1596D8910F2B651D57877B50A4960857B97CBB5A20D44A1DBBAA774",
    "nn/modules/transformer.py": "5D6C6904FB773722EB858663DE4D8601731D0EFBDA734E2FA6BA687072C6748B",
    "optim/muon.py": "22EF96094E891696CCB8BB14C17C669A1E2E219698943F2E29AA55DB2812C1EE",
    "utils/loss.py": "61F2FF23A8A468D6423C38F7629DA8D66E71D5463F43A1CF98EA6F170BFB1019",
    "utils/torch_utils.py": "3C1A95BDCC98379A4506FFDBF7BC84F1475AA1A3B618AA2676FA81AE7707C807",
}
EXPECTED_ENVIRONMENT = {
    "python": "3.10.12",
    "torch": "2.5.1+cu121",
    "ultralytics": "8.4.90",
    "cuda": "12.1",
    "gpu": "NVIDIA GeForce RTX 4090",
}
FROZEN_STATE_MACHINE = (
    "PREFLIGHT_1",
    "MECHANISM_500",
    "SCREEN_10_PAIRED_S012",
    "SEED0_100_PAIRED",
    "SEED1_100_AND_SEED2_100_PAIRED",
    "PAPER_READY_OR_ASCV_LOC_STOP",
)
FROZEN_CROP_CONTRACT = {
    "protocol": "ascv-loc/crop-v2",
    "input_hw": [640, 640],
    "crop_hw": [384, 384],
    "containment_tolerance_px": 1e-6,
    "identity": "resolved_im_file.relative_to(resolved_dataset_root).as_posix()",
}
FROZEN_MECHANISM_GATE = {
    "successful_batches": 500,
    "optimizer_attempts": 106,
    "scientific_tail_window": [401, 500],
    "minimum_pairs_per_direction": 100,
    "minimum_batches_per_direction": 80,
    "mean_advantage_strictly_positive": True,
    "win_rate_strictly_greater_than": 0.5,
}
FROZEN_SCREEN_GATE = {
    "seeds": [0, 1, 2],
    "mAP_dC_wins_minimum": 2,
    "mAP_dC_mean_strictly_positive": True,
    "mAP_DID_wins_minimum": 2,
    "mAP_DID_mean_strictly_positive": True,
    "per_seed_treatment_C_over_control_C_minimum": 0.8,
    "nonnegative_mean_dC": ["AP-tiny-SBR", "tiny_recall", "AP75", "AP-large-SBR"],
}
FROZEN_FORMAL_THRESHOLDS = {
    "AP-tiny-SBR": 0.010,
    "mAP50-95": 0.003,
    "tiny_recall": 0.020,
    "AP75": -0.002,
    "AP-large-SBR": -0.005,
}

REPO_SOURCE_FILES = (
    "scripts/adjudicate_ascv_loc.py",
    "scripts/prepare_ascv_loc_protocol.py",
    "scripts/train_rtdetr_ascv_loc.py",
    "src/ascv_loc.py",
    "src/ascv_loc_adjudicator.py",
    "src/ascv_loc_cli.py",
    "src/ascv_loc_diagnostics.py",
    "src/ascv_loc_protocol.py",
    "src/ascv_loc_stage.py",
    "src/rtdetr_ascv_loc.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def training_batch_sha256(batch: Mapping) -> str:
    """Bind an ordered, already-transformed training batch without serializing it."""

    digest = hashlib.sha256()
    digest.update(b"ascv-loc/training-batch-v1\0")
    image_paths = batch.get("im_file")
    if not isinstance(image_paths, (list, tuple)):
        raise ValueError("training batch digest requires ordered im_file values")
    digest.update(json.dumps([str(value) for value in image_paths], separators=(",", ":")).encode("utf-8"))
    for name in ("img", "cls", "bboxes", "batch_idx"):
        tensor = batch.get(name)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"training batch digest requires tensor {name}")
        materialized = tensor.detach().cpu().contiguous()
        header = {
            "name": name,
            "dtype": str(materialized.dtype),
            "shape": list(materialized.shape),
        }
        digest.update(b"\0")
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(materialized.numpy().tobytes(order="C"))
    return digest.hexdigest().upper()


def state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest().upper()


def subset_signature(path: Path, *, root: Path) -> dict[str, int | str]:
    root = root.resolve()
    entries = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    digest = hashlib.sha256()
    seen: set[str] = set()
    for entry in entries:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            relative = candidate.resolve().relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"subset entry is outside dataset root: {entry}") from error
        if relative in seen:
            raise ValueError(f"duplicate subset entry: {relative}")
        seen.add(relative)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\n")
    return {"count": len(entries), "sha256": digest.hexdigest().upper()}


def require_clean_repo(repo_root: Path) -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        raise ValueError("ASCV source tree is not clean")


def repo_source_hashes(repo_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in REPO_SOURCE_FILES:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[relative] = sha256_file(path)
    return hashes


def source_bundle_sha256(hashes: Mapping[str, str]) -> str:
    payload = json.dumps(dict(hashes), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def validate_initial_state_artifact(artifact: dict, *, seed: int) -> None:
    if artifact.get("format_version") != 1:
        raise ValueError("initial-state scratch provenance format mismatch")
    metadata = artifact.get("metadata", {})
    expected_metadata = {
        "seed": seed,
        "dataset": {
            "file_count": EXPECTED_DATASET_FILE_COUNT,
            "sha256": EXPECTED_DATASET_SHA256,
        },
        "category_mapping_sha256": EXPECTED_CATEGORY_MAPPING_SHA256,
        "subset": {
            "count": EXPECTED_SUBSET_COUNT,
            "fraction": 0.1,
            "sha256": EXPECTED_SUBSET_SHA256,
        },
        "source_sha256": EXPECTED_PARENT_SOURCE_SHA256,
        "environment": EXPECTED_ENVIRONMENT,
        "control_parameters": 32_826_626,
        "innovation_seed": seed + 10_000,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"initial-state scratch provenance mismatch: {key}")
    common = artifact.get("common_state")
    if not isinstance(common, dict):
        raise ValueError("initial-state scratch provenance has no common_state")
    fingerprint = state_fingerprint(common)
    if artifact.get("fingerprints", {}).get("common") != fingerprint:
        raise ValueError("initial-state common fingerprint mismatch")
    if fingerprint != EXPECTED_COMMON_FINGERPRINTS[seed]:
        raise ValueError("initial-state scratch provenance fingerprint is not allowlisted")


def validate_parent_attestation(manifest: dict, seed: int) -> dict:
    lineage = manifest["parent_lineage"][str(seed)]
    path = Path(lineage["parent_protocol"]).resolve()
    actual_sha = sha256_file(path)
    if actual_sha != lineage["parent_protocol_sha256"]:
        raise ValueError("parent attestation checksum does not match protocol manifest")
    if actual_sha != EXPECTED_PARENT_ATTESTATION_SHA256[seed]:
        raise ValueError("parent attestation checksum is not allowlisted")
    record = json.loads(path.read_text(encoding="utf-8"))
    if int(record.get("seed", -1)) != seed:
        raise ValueError("parent attestation seed mismatch")
    if record.get("dataset") != {
        "file_count": EXPECTED_DATASET_FILE_COUNT,
        "sha256": EXPECTED_DATASET_SHA256,
    }:
        raise ValueError("parent attestation dataset signature mismatch")
    if record.get("category_mapping_sha256") != EXPECTED_CATEGORY_MAPPING_SHA256:
        raise ValueError("parent attestation category mapping mismatch")
    return record
