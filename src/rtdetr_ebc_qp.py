from __future__ import annotations

from pathlib import Path

import ultralytics
import ultralytics.nn.modules.head as ultralytics_head
import ultralytics.nn.tasks as ultralytics_tasks
from ultralytics.nn.tasks import RTDETRDetectionModel

from src.ebc_qp_config import EBCQPConfig, assert_ultralytics_source_lock
from src.ebc_qp_decoder import EBCQPDecoder, register_ebc_qp_decoder


def _assert_source_lock() -> None:
    package_root = Path(ultralytics.__file__).parent
    assert_ultralytics_source_lock(
        {
            "head.py": Path(ultralytics_head.__file__),
            "tasks.py": Path(ultralytics_tasks.__file__),
            "rtdetr-l.yaml": package_root / "cfg" / "models" / "rt-detr" / "rtdetr-l.yaml",
        }
    )


class EBCQPDetectionModel(RTDETRDetectionModel):
    def __init__(
        self,
        cfg: str | Path = "configs/rtdetr-l-ebc-qp.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        ebc_config: EBCQPConfig | None = None,
    ):
        _assert_source_lock()
        self.ebc_config = ebc_config or EBCQPConfig()
        original_decoder = ultralytics_tasks.RTDETRDecoder
        register_ebc_qp_decoder()
        try:
            super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        finally:
            ultralytics_tasks.RTDETRDecoder = original_decoder
        self.ebc_head.ebc_config = self.ebc_config

    @property
    def ebc_head(self) -> EBCQPDecoder:
        head = self.model[-1]
        if not isinstance(head, EBCQPDecoder):
            raise TypeError("model does not end in EBCQPDecoder")
        return head

    def set_ebc_progress(self, epoch: int) -> None:
        self.ebc_head.set_progress(epoch)
