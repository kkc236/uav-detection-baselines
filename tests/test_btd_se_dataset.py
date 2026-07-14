from pathlib import Path

import numpy as np

from src.btd_se_dataset import append_ignored_boxes, ignore_label_path


def test_ignore_label_path_mirrors_images_split(tmp_path: Path):
    image = tmp_path / "VisDrone" / "images" / "train" / "000001.jpg"

    actual = ignore_label_path(image)

    assert actual == tmp_path / "VisDrone" / "labels_ignore" / "train" / "000001.txt"


def test_append_ignored_boxes_uses_negative_class_for_shared_geometric_transforms(tmp_path: Path):
    image = tmp_path / "VisDrone" / "images" / "train" / "000001.jpg"
    ignore_file = ignore_label_path(image)
    ignore_file.parent.mkdir(parents=True)
    ignore_file.write_text("0.25 0.20 0.30 0.20\n0.75 0.70 0.10 0.15\n", encoding="ascii")
    label = {
        "im_file": str(image),
        "cls": np.array([[3.0]], dtype=np.float32),
        "bboxes": np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
    }

    updated = append_ignored_boxes(label)

    np.testing.assert_array_equal(updated["cls"], np.array([[3.0], [-1.0], [-1.0]], dtype=np.float32))
    np.testing.assert_allclose(
        updated["bboxes"],
        np.array(
            [
                [0.5, 0.5, 0.2, 0.2],
                [0.25, 0.20, 0.30, 0.20],
                [0.75, 0.70, 0.10, 0.15],
            ],
            dtype=np.float32,
        ),
    )
