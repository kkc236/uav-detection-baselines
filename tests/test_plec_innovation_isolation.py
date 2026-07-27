import ast
from pathlib import Path

from src.gcmv_plec import PhasePreservingLocalEvidenceCanonicalizer


def test_plec_source_is_isolated_from_detector_and_future_modules():
    source_path = Path("src/gcmv_plec.py")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    forbidden_import_fragments = (
        "ultralytics.models.rtdetr",
        "gcmv_gglf",
        "gcmv_peg",
        "saded",
        "sbr_fusion",
        "query",
    )
    assert not any(
        fragment in imported
        for imported in imports
        for fragment in forbidden_import_fragments
    )
    assert "PVC" not in source
    assert "GRCA" not in source
    assert "QCVR" not in source


def test_plec_is_a_trainable_network_module():
    module = PhasePreservingLocalEvidenceCanonicalizer(channels=8)

    assert sum(parameter.numel() for parameter in module.parameters()) > 0
    assert all(parameter.requires_grad for parameter in module.parameters())
