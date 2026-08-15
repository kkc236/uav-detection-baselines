from pathlib import Path

from PIL import Image

from src.visdrone import convert_split, convert_visdrone_ignore_row, convert_visdrone_row


def test_convert_visdrone_row_skips_ignored_region():
    assert convert_visdrone_row("10,20,30,40,0,4,0,0", image_width=100, image_height=200) is None


def test_convert_visdrone_row_to_yolo_box():
    row = convert_visdrone_row("10,20,30,40,1,4,0,0", image_width=100, image_height=200)

    assert row == "3 0.250000 0.200000 0.300000 0.200000"


def test_convert_visdrone_ignore_row_preserves_ignored_box_without_class():
    row = convert_visdrone_ignore_row("10,20,30,40,0,4,0,0", image_width=100, image_height=200)

    assert row == "0.250000 0.200000 0.300000 0.200000"


def test_convert_visdrone_ignore_row_skips_detection_box():
    assert (
        convert_visdrone_ignore_row("10,20,30,40,1,4,0,0", image_width=100, image_height=200) is None
    )


def test_convert_split_writes_frozen_crlf_label_bytes(tmp_path: Path, monkeypatch):
    source = tmp_path / "VisDrone2019-DET-val"
    images = source / "images"
    annotations = source / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    Image.new("RGB", (100, 200)).save(images / "sample.jpg")
    (annotations / "sample.txt").write_text(
        "10,20,30,40,1,4,0,0\n"
        "20,30,10,10,1,2,0,0\n"
        "1,2,3,4,0,0,0,0\n"
        "5,6,7,8,0,0,0,0\n",
        encoding="utf-8",
    )

    def reject_platform_dependent_text_write(*_args, **_kwargs):
        raise AssertionError("frozen labels must be written as explicit bytes")

    monkeypatch.setattr(Path, "write_text", reject_platform_dependent_text_write)

    convert_split(tmp_path, "val")

    labels = (tmp_path / "labels" / "val" / "sample.txt").read_bytes()
    ignores = (tmp_path / "labels_ignore" / "val" / "sample.txt").read_bytes()
    assert b"\r\n" in labels and b"\n" not in labels.replace(b"\r\n", b"")
    assert b"\r\n" in ignores and b"\n" not in ignores.replace(b"\r\n", b"")
    assert not labels.endswith(b"\r\n")
    assert not ignores.endswith(b"\r\n")
