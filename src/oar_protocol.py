"""Frozen constants for the Objective-Aligned Reranker protocol."""

from decimal import Decimal


OAR_K_GRID = (20, 40, 60, 100)
OAR_GAIN_RECOVERY = Decimal("0.90")
OAR_MAP_GATE = Decimal("0.0050")
OAR_EPOCHS = 20
OAR_PAIR_CAP = 2647
OAR_NUM_CLASSES = 10
OAR_NUM_QUERIES = 300
OAR_MAX_DET = 300
OAR_HIDDEN_DIM = 256
OAR_FEATURE_DIM = 276
