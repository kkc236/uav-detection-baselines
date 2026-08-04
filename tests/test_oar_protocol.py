from decimal import Decimal

from src.oar_protocol import (
    OAR_EPOCHS,
    OAR_FEATURE_DIM,
    OAR_GAIN_RECOVERY,
    OAR_HIDDEN_DIM,
    OAR_K_GRID,
    OAR_MAP_GATE,
    OAR_MAX_DET,
    OAR_NUM_CLASSES,
    OAR_NUM_QUERIES,
    OAR_PAIR_CAP,
)


def test_oar_constants_are_frozen() -> None:
    assert OAR_K_GRID == (20, 40, 60, 100)
    assert OAR_GAIN_RECOVERY == Decimal("0.90")
    assert OAR_MAP_GATE == Decimal("0.0050")
    assert OAR_EPOCHS == 20
    assert OAR_PAIR_CAP == 2647
    assert OAR_NUM_CLASSES == 10
    assert OAR_NUM_QUERIES == 300
    assert OAR_MAX_DET == 300
    assert OAR_HIDDEN_DIM == 256
    assert OAR_FEATURE_DIM == 276
