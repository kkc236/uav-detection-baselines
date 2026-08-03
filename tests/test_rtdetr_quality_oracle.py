from __future__ import annotations

import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ultralytics.nn.modules.head import RTDETRDecoder

from src.rtdetr_quality_oracle import (
    ALPHA_GRID,
    DEV_COUNT,
    DEV_SPLIT_SALT,
    EXPECTED_DEV_SHA256,
    MAP_GAIN_THRESHOLD,
    QualityOracleCacheViolation,
    decide_quality_oracle,
    flattened_topk,
    load_quality_oracle_cache,
    oracle_topk,
    ordered_path_sha256,
    same_class_iou_quality,
    select_alpha,
    select_internal_dev,
    write_quality_oracle_cache,
)


_REAL_AUTHORIZED_RELATIVE_PATHS = tuple(
    """images/train/9999956_00000_d_0000048.jpg
images/train/0000308_01201_d_0000311.jpg
images/train/9999999_00057_d_0000015.jpg
images/train/0000136_00162_d_0000154.jpg
images/train/9999984_00000_d_0000018.jpg
images/train/9999966_00000_d_0000028.jpg
images/train/9999982_00000_d_0000202.jpg
images/train/9999998_00242_d_0000196.jpg
images/train/9999984_00000_d_0000048.jpg
images/train/9999997_00000_d_0000002.jpg
images/train/9999951_00000_d_0000079.jpg
images/train/9999982_00000_d_0000049.jpg
images/train/9999987_00000_d_0000005.jpg
images/train/9999999_00542_d_0000243.jpg
images/train/9999953_00000_d_0000178.jpg
images/train/9999955_00000_d_0000263.jpg
images/train/9999955_00000_d_0000056.jpg
images/train/9999955_00000_d_0000402.jpg
images/train/0000309_01801_d_0000342.jpg
images/train/9999984_00000_d_0000056.jpg
images/train/9999942_00000_d_0000215.jpg
images/train/0000071_00007_d_0000001.jpg
images/train/9999985_00000_d_0000016.jpg
images/train/0000076_04925_d_0000017.jpg
images/train/9999982_00000_d_0000102.jpg
images/train/9999984_00000_d_0000012.jpg
images/train/9999984_00000_d_0000118.jpg
images/train/9999998_00405_d_0000357.jpg
images/train/9999951_00000_d_0000012.jpg
images/train/9999999_00448_d_0000199.jpg
images/train/9999994_00000_d_0000052.jpg
images/train/9999956_00000_d_0000100.jpg
images/train/0000288_03601_d_0000802.jpg
images/train/9999998_00156_d_0000132.jpg
images/train/9999945_00000_d_0000056.jpg
images/train/9999953_00000_d_0000084.jpg
images/train/9999999_00566_d_0000255.jpg
images/train/9999998_00396_d_0000348.jpg
images/train/9999998_00357_d_0000309.jpg
images/train/9999945_00000_d_0000053.jpg
images/train/9999982_00000_d_0000042.jpg
images/train/9999977_00000_d_0000100.jpg
images/train/9999999_00155_d_0000062.jpg
images/train/9999982_00000_d_0000041.jpg
images/train/9999955_00000_d_0000084.jpg
images/train/9999962_00000_d_0000142.jpg
images/train/9999990_00000_d_0000013.jpg
images/train/9999982_00000_d_0000112.jpg
images/train/0000145_00001_d_0000001.jpg
images/train/9999945_00000_d_0000146.jpg
images/train/0000308_00001_d_0000305.jpg
images/train/9999953_00000_d_0000093.jpg
images/train/9999943_00000_d_0000069.jpg
images/train/0000165_05525_d_0000111.jpg
images/train/9999960_00000_d_0000024.jpg
images/train/9999977_00000_d_0000070.jpg
images/train/9999937_00000_d_0000069.jpg
images/train/9999945_00000_d_0000126.jpg
images/train/0000325_00801_d_0000666.jpg
images/train/0000339_00981_d_0000166.jpg
images/train/0000178_00393_d_0000008.jpg
images/train/9999945_00000_d_0000138.jpg
images/train/9999960_00000_d_0000169.jpg
images/train/9999981_00000_d_0000042.jpg
images/train/9999981_00000_d_0000005.jpg
images/train/9999945_00000_d_0000123.jpg
images/train/9999953_00000_d_0000137.jpg
images/train/9999942_00000_d_0000111.jpg
images/train/0000042_02421_d_0000076.jpg
images/train/9999969_00000_d_0000033.jpg
images/train/9999972_00000_d_0000040.jpg
images/train/9999984_00000_d_0000109.jpg
images/train/9999964_00000_d_0000074.jpg
images/train/0000261_00001_d_0000120.jpg
images/train/9999955_00000_d_0000306.jpg
images/train/9999999_00765_d_0000350.jpg
images/train/9999982_00000_d_0000002.jpg
images/train/9999945_00000_d_0000017.jpg
images/train/0000351_02941_d_0000534.jpg
images/train/9999982_00000_d_0000143.jpg
images/train/9999942_00000_d_0000237.jpg
images/train/9999965_00000_d_0000015.jpg
images/train/0000279_02401_d_0000596.jpg
images/train/9999937_00000_d_0000030.jpg
images/train/9999974_00000_d_0000006.jpg
images/train/9999984_00000_d_0000101.jpg
images/train/9999998_00196_d_0000157.jpg
images/train/9999942_00000_d_0000076.jpg
images/train/9999981_00000_d_0000127.jpg
images/train/0000290_00401_d_0000850.jpg
images/train/9999982_00000_d_0000235.jpg
images/train/9999964_00000_d_0000077.jpg
images/train/0000313_00001_d_0000436.jpg
images/train/9999982_00000_d_0000229.jpg
images/train/9999981_00000_d_0000021.jpg
images/train/9999984_00000_d_0000079.jpg
images/train/9999999_00177_d_0000073.jpg
images/train/9999965_00000_d_0000002.jpg
images/train/9999955_00000_d_0000020.jpg
images/train/9999960_00000_d_0000075.jpg
images/train/0000113_00064_d_0000075.jpg
images/train/9999943_00000_d_0000011.jpg
images/train/0000199_00603_d_0000164.jpg
images/train/0000172_01484_d_0000001.jpg
images/train/9999999_00624_d_0000283.jpg
images/train/9999972_00000_d_0000128.jpg
images/train/0000264_01401_d_0000201.jpg
images/train/9999969_00000_d_0000066.jpg
images/train/0000107_01673_d_0000053.jpg
images/train/9999956_00000_d_0000009.jpg
images/train/9999999_00867_d_0000397.jpg
images/train/0000279_05801_d_0000610.jpg
images/train/9999951_00000_d_0000120.jpg
images/train/9999999_00265_d_0000115.jpg
images/train/9999975_00000_d_0000017.jpg
images/train/9999955_00000_d_0000261.jpg
images/train/0000258_02669_d_0000078.jpg
images/train/9999970_00000_d_0000043.jpg
images/train/9999965_00000_d_0000033.jpg
images/train/0000309_03801_d_0000352.jpg
images/train/9999964_00000_d_0000020.jpg
images/train/9999951_00000_d_0000261.jpg
images/train/9999937_00000_d_0000014.jpg
images/train/9999955_00000_d_0000277.jpg
images/train/0000170_00801_d_0000001.jpg
images/train/9999950_00000_d_0000018.jpg
images/train/9999955_00000_d_0000210.jpg
images/train/9999982_00000_d_0000226.jpg
images/train/9999950_00000_d_1007223.jpg
images/train/9999955_00000_d_0000299.jpg
images/train/0000305_01201_d_0000219.jpg
images/train/9999990_00000_d_0000077.jpg
images/train/9999953_00000_d_0000075.jpg
images/train/9999956_00000_d_0000103.jpg
images/train/9999981_00000_d_0000057.jpg
images/train/9999998_00058_d_0000044.jpg
images/train/9999977_00000_d_0000082.jpg
images/train/9999982_00000_d_0000148.jpg
images/train/9999960_00000_d_0000108.jpg
images/train/9999997_00000_d_0000010.jpg
images/train/9999955_00000_d_0000330.jpg
images/train/0000288_04401_d_0000805.jpg
images/train/0000290_02801_d_0000861.jpg
images/train/9999943_00000_d_0000075.jpg
images/train/9999948_00000_d_0000006.jpg
images/train/9999972_00000_d_0000143.jpg
images/train/9999955_00000_d_0000414.jpg
images/train/9999990_00000_d_0000052.jpg
images/train/9999960_00000_d_0000096.jpg
images/train/0000239_11139_d_0000028.jpg
images/train/9999998_00372_d_0000324.jpg
images/train/9999982_00000_d_0000026.jpg
images/train/9999981_00000_d_0000106.jpg
images/train/9999937_00000_d_0000057.jpg
images/train/9999999_00141_d_0000055.jpg
images/train/0000352_01765_d_0000548.jpg
images/train/9999998_00100_d_0000080.jpg
images/train/9999942_00000_d_0000226.jpg
images/train/9999998_00218_d_0000174.jpg
images/train/9999937_00000_d_0000131.jpg
images/train/9999966_00000_d_0000011.jpg
images/train/9999982_00000_d_0000175.jpg
images/train/9999960_00000_d_0000016.jpg
images/train/9999999_00851_d_0000389.jpg
images/train/9999953_00000_d_0000057.jpg
images/train/9999999_00671_d_0000305.jpg
images/train/9999945_00000_d_0000128.jpg
images/train/9999942_00000_d_0000052.jpg
images/train/9999937_00000_d_0000045.jpg
images/train/9999972_00000_d_0000127.jpg
images/train/9999998_00384_d_0000336.jpg
images/train/9999945_00000_d_0000143.jpg
images/train/9999972_00000_d_0000085.jpg
images/train/9999955_00000_d_0000254.jpg
images/train/9999998_00025_d_0000018.jpg
images/train/9999991_00000_d_0000027.jpg
images/train/9999951_00000_d_0000133.jpg
images/train/9999999_00305_d_0000135.jpg
images/train/9999997_00000_d_0000069.jpg
images/train/0000201_00000_d_0000171.jpg
images/train/9999999_00781_d_0000357.jpg
images/train/9999987_00000_d_0000046.jpg
images/train/9999999_00217_d_0000091.jpg
images/train/9999937_00000_d_0000218.jpg
images/train/0000288_05201_d_0000808.jpg
images/train/9999998_00334_d_0000286.jpg
images/train/0000210_01251_d_0000227.jpg
images/train/9999951_00000_d_0000257.jpg
images/train/9999998_00333_d_0000285.jpg
images/train/9999962_00000_d_0000026.jpg
images/train/9999960_00000_d_0000073.jpg
images/train/9999982_00000_d_0000256.jpg
images/train/9999982_00000_d_0000124.jpg
images/train/9999951_00000_d_0000230.jpg
images/train/9999972_00000_d_0000111.jpg
images/train/9999953_00000_d_0000125.jpg
images/train/0000143_00281_d_0000049.jpg
images/train/9999991_00000_d_0000030.jpg
images/train/0000343_03137_d_0000291.jpg
images/train/0000309_03401_d_0000350.jpg
images/train/9999937_00000_d_0000103.jpg
images/train/9999953_00000_d_0000139.jpg
images/train/0000307_06601_d_0000301.jpg
images/train/0000293_02801_d_0000936.jpg
images/train/9999951_00000_d_0000027.jpg
images/train/9999955_00000_d_0000182.jpg
images/train/9999951_00000_d_0000191.jpg
images/train/9999998_00320_d_0000273.jpg
images/train/9999990_00000_d_0000018.jpg
images/train/9999998_00085_d_0000069.jpg
images/train/9999984_00000_d_0000059.jpg
images/train/9999982_00000_d_0000097.jpg
images/train/9999956_00000_d_0000050.jpg
images/train/9999998_00033_d_0000026.jpg
images/train/9999999_00604_d_0000274.jpg
images/train/9999998_00239_d_0000193.jpg
images/train/9999942_00000_d_0000206.jpg
images/train/9999972_00000_d_0000104.jpg
images/train/9999955_00000_d_0000203.jpg
images/train/0000156_00354_d_0000078.jpg
images/train/9999970_00000_d_0000010.jpg
images/train/9999955_00000_d_0000221.jpg
images/train/0000068_00571_d_0000003.jpg
images/train/0000239_09866_d_0000025.jpg
images/train/9999999_00869_d_0000398.jpg
images/train/9999937_00000_d_0000177.jpg
images/train/9999953_00000_d_0000066.jpg
images/train/9999991_00000_d_0000017.jpg
images/train/9999943_00000_d_0000047.jpg
images/train/9999972_00000_d_0000099.jpg
images/train/9999955_00000_d_0000246.jpg
images/train/9999974_00000_d_0000029.jpg
images/train/9999945_00000_d_0000080.jpg
images/train/9999998_00425_d_0000377.jpg
images/train/9999972_00000_d_0000139.jpg
images/train/9999966_00000_d_0000086.jpg
images/train/9999970_00000_d_0000024.jpg
images/train/9999969_00000_d_0000023.jpg
images/train/9999990_00000_d_0000063.jpg
images/train/9999937_00000_d_0000043.jpg
images/train/9999990_00000_d_0000026.jpg
images/train/9999999_00085_d_0000029.jpg
images/train/9999955_00000_d_0000102.jpg
images/train/9999999_00877_d_0000402.jpg
images/train/9999972_00000_d_0000074.jpg
images/train/9999998_00032_d_0000025.jpg
images/train/9999966_00000_d_0000039.jpg
images/train/9999962_00000_d_0000065.jpg
images/train/9999981_00000_d_0000099.jpg
images/train/9999998_00371_d_0000323.jpg
images/train/9999984_00000_d_0000084.jpg
images/train/9999998_00274_d_0000227.jpg
images/train/0000210_00261_d_0000224.jpg
images/train/9999977_00000_d_0000023.jpg
images/train/0000331_03201_d_0000857.jpg
images/train/9999945_00000_d_0000059.jpg
images/train/9999998_00383_d_0000335.jpg
images/train/9999942_00000_d_0000140.jpg
images/train/9999955_00000_d_0000123.jpg
images/train/9999998_00267_d_0000221.jpg
images/train/9999966_00000_d_0000048.jpg
images/train/9999984_00000_d_0000010.jpg
images/train/9999955_00000_d_0000201.jpg
images/train/9999960_00000_d_0000018.jpg
images/train/9999962_00000_d_0000136.jpg
images/train/0000258_00001_d_0000069.jpg
images/train/9999962_00000_d_0000071.jpg
images/train/0000264_04201_d_0000215.jpg
images/train/9999940_00000_d_0000011.jpg
images/train/0000239_01172_d_0000004.jpg
images/train/9999999_00213_d_0000089.jpg
images/train/9999966_00000_d_0000099.jpg
images/train/9999998_00014_d_0000010.jpg
images/train/9999984_00000_d_0000003.jpg
images/train/9999955_00000_d_0000110.jpg
images/train/9999950_00000_d_0000047.jpg
images/train/9999981_00000_d_0000109.jpg
images/train/9999972_00000_d_0000061.jpg
images/train/9999942_00000_d_0000187.jpg
images/train/0000007_04999_d_0000036.jpg
images/train/9999964_00000_d_0000082.jpg
images/train/9999982_00000_d_0000111.jpg
images/train/9999999_00761_d_0000348.jpg
images/train/9999953_00000_d_0000166.jpg
images/train/0000142_04858_d_0000047.jpg
images/train/9999994_00000_d_0000036.jpg
images/train/0000313_06001_d_0000466.jpg
images/train/9999965_00000_d_0000079.jpg
images/train/0000363_02157_d_0000793.jpg
images/train/9999945_00000_d_0000007.jpg
images/train/9999962_00000_d_0000122.jpg
images/train/0000331_00401_d_0000843.jpg
images/train/9999956_00000_d_0000115.jpg
images/train/9999977_00000_d_0000091.jpg
images/train/9999984_00000_d_0000123.jpg
images/train/9999942_00000_d_0000250.jpg
images/train/9999984_00000_d_0000045.jpg
images/train/9999953_00000_d_0000133.jpg
images/train/9999950_00000_d_0000039.jpg
images/train/9999998_00429_d_0000381.jpg
images/train/9999942_00000_d_0000244.jpg
images/train/9999974_00000_d_0000017.jpg
images/train/9999982_00000_d_0000137.jpg
images/train/9999943_00000_d_0000044.jpg
images/train/9999984_00000_d_0000108.jpg
images/train/0000252_00001_d_0000001.jpg
images/train/0000181_00000_d_0000028.jpg
images/train/9999998_00145_d_0000121.jpg
images/train/9999955_00000_d_0000332.jpg
images/train/9999997_00000_d_0000066.jpg
images/train/9999984_00000_d_0000001.jpg
images/train/0000068_03581_d_0000011.jpg
images/train/0000323_01801_d_0000641.jpg
images/train/9999955_00000_d_0000426.jpg
images/train/9999953_00000_d_0000003.jpg
images/train/9999953_00000_d_0000217.jpg
images/train/0000260_03359_d_0000115.jpg
images/train/0000199_00933_d_0000165.jpg
images/train/9999955_00000_d_0000251.jpg
images/train/9999955_00000_d_0000041.jpg
images/train/0000348_03725_d_0000425.jpg
images/train/9999999_00727_d_0000331.jpg
images/train/0000220_00001_d_0000001.jpg
images/train/9999998_00027_d_0000020.jpg
images/train/9999998_00049_d_0000039.jpg
images/train/0000126_02076_d_0000125.jpg
images/train/9999966_00000_d_0000037.jpg
images/train/9999945_00000_d_0000035.jpg
images/train/9999977_00000_d_0000068.jpg
images/train/9999999_00171_d_0000070.jpg
images/train/9999999_00081_d_0000027.jpg
images/train/9999951_00000_d_0000059.jpg
images/train/9999999_00159_d_0000064.jpg
images/train/9999955_00000_d_0000364.jpg
images/train/9999997_00000_d_0000060.jpg
images/train/9999981_00000_d_0000116.jpg
images/train/9999951_00000_d_0000129.jpg
images/train/0000304_00201_d_0000203.jpg
images/train/9999942_00000_d_0000159.jpg
images/train/9999999_00472_d_0000211.jpg
images/train/9999955_00000_d_0000112.jpg
images/train/0000281_00601_d_0000631.jpg
images/train/9999955_00000_d_0000101.jpg
images/train/0000076_04382_d_0000015.jpg
images/train/9999948_00000_d_0000013.jpg
images/train/9999981_00000_d_0000107.jpg
images/train/9999999_00759_d_0000347.jpg
images/train/0000288_00601_d_0000788.jpg
images/train/0000305_02401_d_0000225.jpg
images/train/9999956_00000_d_0000055.jpg
images/train/9999998_00065_d_0000050.jpg
images/train/9999998_00076_d_0000061.jpg
images/train/9999953_00000_d_0000040.jpg
images/train/9999964_00000_d_0000058.jpg
images/train/9999937_00000_d_0000132.jpg
images/train/0000366_08233_d_0000814.jpg
images/train/0000261_03701_d_0000133.jpg
images/train/9999960_00000_d_0000125.jpg
images/train/0000256_01110_d_0000021.jpg
images/train/9999937_00000_d_0000199.jpg
images/train/9999937_00000_d_0000022.jpg
images/train/9999984_00000_d_0000113.jpg
images/train/9999967_00000_d_0000061.jpg
images/train/9999982_00000_d_0000123.jpg
images/train/9999969_00000_d_0000063.jpg
images/train/0000167_00984_d_0000129.jpg
images/train/9999998_00319_d_0000272.jpg
images/train/9999962_00000_d_0000139.jpg
images/train/9999999_00161_d_0000065.jpg
images/train/9999994_00000_d_0000049.jpg
images/train/9999989_00000_d_0000012.jpg
images/train/0000342_05293_d_0000269.jpg
images/train/9999998_00389_d_0000341.jpg
images/train/9999994_00000_d_0000053.jpg
images/train/9999972_00000_d_0000017.jpg
images/train/9999951_00000_d_0000083.jpg
images/train/9999964_00000_d_0000009.jpg
images/train/9999998_00069_d_0000054.jpg
images/train/0000167_00384_d_0000126.jpg
images/train/0000182_00000_d_0000036.jpg
images/train/9999990_00000_d_0000017.jpg
images/train/9999951_00000_d_0000175.jpg
images/train/9999955_00000_d_0000280.jpg
images/train/9999953_00000_d_0000147.jpg
images/train/9999953_00000_d_0000205.jpg
images/train/9999945_00000_d_0000066.jpg
images/train/0000010_05149_d_0000057.jpg
images/train/9999969_00000_d_0000050.jpg
images/train/0000222_06100_d_0000019.jpg
images/train/9999955_00000_d_0000144.jpg
images/train/0000179_00149_d_0000014.jpg
images/train/9999955_00000_d_0000321.jpg
images/train/0000344_01177_d_0000299.jpg
images/train/9999937_00000_d_0000184.jpg
images/train/9999974_00000_d_0000022.jpg
images/train/9999945_00000_d_0000038.jpg
images/train/9999990_00000_d_0000065.jpg
images/train/9999962_00000_d_0000088.jpg
images/train/0000222_05666_d_0000018.jpg
images/train/9999951_00000_d_0000108.jpg
images/train/9999951_00000_d_0000226.jpg
images/train/9999972_00000_d_0000088.jpg
images/train/9999942_00000_d_0000186.jpg
images/train/0000126_02406_d_0000126.jpg
images/train/9999964_00000_d_0000005.jpg
images/train/9999955_00000_d_0000029.jpg
images/train/9999984_00000_d_0000166.jpg
images/train/0000305_01401_d_0000220.jpg
images/train/9999987_00000_d_0000057.jpg
images/train/9999966_00000_d_0000140.jpg
images/train/9999955_00000_d_0000230.jpg
images/train/9999972_00000_d_0000018.jpg
images/train/9999970_00000_d_0000025.jpg
images/train/9999955_00000_d_0000032.jpg
images/train/0000177_00000_d_0000001.jpg
images/train/9999990_00000_d_0000028.jpg
images/train/0000145_00401_d_0000001.jpg
images/train/9999960_00000_d_0000021.jpg
images/train/0000290_01401_d_0000855.jpg
images/train/9999962_00000_d_0000163.jpg
images/train/9999999_00079_d_0000026.jpg
images/train/0000197_02821_d_0000153.jpg
images/train/9999998_00297_d_0000250.jpg
images/train/9999977_00000_d_0000083.jpg
images/train/9999950_00000_d_0000081.jpg
images/train/9999981_00000_d_0000023.jpg
images/train/9999950_00000_d_0000005.jpg
images/train/9999987_00000_d_0000019.jpg
images/train/0000143_00081_d_0000048.jpg
images/train/9999953_00000_d_0000028.jpg
images/train/0000177_00570_d_0000004.jpg
images/train/9999966_00000_d_0000012.jpg
images/train/9999984_00000_d_0000150.jpg
images/train/9999999_00721_d_0000328.jpg
images/train/9999962_00000_d_0000064.jpg
images/train/0000257_00001_d_0000042.jpg
images/train/0000150_00230_d_0000068.jpg
images/train/9999985_00000_d_0000071.jpg
images/train/9999953_00000_d_0000204.jpg
images/train/9999999_00554_d_0000249.jpg
images/train/0000352_00393_d_0000541.jpg
images/train/9999994_00000_d_0000022.jpg
images/train/9999967_00000_d_0000052.jpg
images/train/9999982_00000_d_0000154.jpg
images/train/9999999_00163_d_0000066.jpg
images/train/9999942_00000_d_0000252.jpg
images/train/9999966_00000_d_0000105.jpg
images/train/9999955_00000_d_0000086.jpg
images/train/9999945_00000_d_0000131.jpg
images/train/0000351_03137_d_0000535.jpg
images/train/9999969_00000_d_0000026.jpg
images/train/9999951_00000_d_0000193.jpg
images/train/9999999_00544_d_0000244.jpg
images/train/9999985_00000_d_0000027.jpg
images/train/9999962_00000_d_0000084.jpg
images/train/9999955_00000_d_0000114.jpg
images/train/0000008_02999_d_0000042.jpg
images/train/9999937_00000_d_0000205.jpg
images/train/9999967_00000_d_0000047.jpg
images/train/9999942_00000_d_0000256.jpg
images/train/9999990_00000_d_0000002.jpg
images/train/9999951_00000_d_0000085.jpg
images/train/9999964_00000_d_0000075.jpg
images/train/9999984_00000_d_0000159.jpg
images/train/9999982_00000_d_0000133.jpg
images/train/0000195_00409_d_0000128.jpg
images/train/9999977_00000_d_0000065.jpg
images/train/9999945_00000_d_0000135.jpg
images/train/9999943_00000_d_0000067.jpg
images/train/0000339_00001_d_0000161.jpg
images/train/9999999_00558_d_0000251.jpg
images/train/0000182_01653_d_0000040.jpg
images/train/9999937_00000_d_0000191.jpg
images/train/9999965_00000_d_0000045.jpg
images/train/0000173_00401_d_0000001.jpg
images/train/0000225_02427_d_0000009.jpg
images/train/9999951_00000_d_0000302.jpg
images/train/9999962_00000_d_0000124.jpg
images/train/9999972_00000_d_0000157.jpg
images/train/0000223_01986_d_0000006.jpg
images/train/9999982_00000_d_0000072.jpg
images/train/9999987_00000_d_0000003.jpg
images/train/9999994_00000_d_0000111.jpg
images/train/9999943_00000_d_0000079.jpg
images/train/9999967_00000_d_0000053.jpg
images/train/9999966_00000_d_0000070.jpg
images/train/9999991_00000_d_0000014.jpg
images/train/0000279_03001_d_0000598.jpg
images/train/9999955_00000_d_0000419.jpg
images/train/0000308_00801_d_0000309.jpg
images/train/9999955_00000_d_0000255.jpg
images/train/9999951_00000_d_0000043.jpg
images/train/9999951_00000_d_0000263.jpg
images/train/9999940_00000_d_0000086.jpg
images/train/9999945_00000_d_0000042.jpg
images/train/9999998_00277_d_0000230.jpg
images/train/9999999_00689_d_0000312.jpg
images/train/9999955_00000_d_0000296.jpg
images/train/9999945_00000_d_0000134.jpg
images/train/9999955_00000_d_0000352.jpg
images/train/9999940_00000_d_0000097.jpg
images/train/9999999_00675_d_0000306.jpg
images/train/9999969_00000_d_0000055.jpg
images/train/0000243_01787_d_0000005.jpg
images/train/9999998_00077_d_0000062.jpg
images/train/9999945_00000_d_0000141.jpg
images/train/9999943_00000_d_0000019.jpg
images/train/0000165_03926_d_0000103.jpg
images/train/9999962_00000_d_0000086.jpg
images/train/0000197_01661_d_0000150.jpg
images/train/9999984_00000_d_0000035.jpg
images/train/0000257_00359_d_0000044.jpg
images/train/9999965_00000_d_0000021.jpg
images/train/9999998_00109_d_0000087.jpg
images/train/9999984_00000_d_0000088.jpg
images/train/9999999_00653_d_0000297.jpg
images/train/0000043_00500_d_0000077.jpg
images/train/0000180_00806_d_0000023.jpg
images/train/9999981_00000_d_0000102.jpg
images/train/9999964_00000_d_0000006.jpg
images/train/9999977_00000_d_0000078.jpg
images/train/9999990_00000_d_0000041.jpg
images/train/9999964_00000_d_0000037.jpg
images/train/9999998_00189_d_0000152.jpg
images/train/0000352_05881_d_0000569.jpg
images/train/9999999_00793_d_0000363.jpg
images/train/9999982_00000_d_0000152.jpg
images/train/0000171_00001_d_0000001.jpg
images/train/9999955_00000_d_0000065.jpg
images/train/9999994_00000_d_0000027.jpg
images/train/0000290_00801_d_0000852.jpg
images/train/9999960_00000_d_0000116.jpg
images/train/9999937_00000_d_0000166.jpg
images/train/9999951_00000_d_0000310.jpg
images/train/9999966_00000_d_0000023.jpg
images/train/9999987_00000_d_0000053.jpg
images/train/9999972_00000_d_0000060.jpg
images/train/0000313_01601_d_0000444.jpg
images/train/9999977_00000_d_0000089.jpg
images/train/9999945_00000_d_0000067.jpg
images/train/9999955_00000_d_0000089.jpg
images/train/9999998_00411_d_0000363.jpg
images/train/9999955_00000_d_0000008.jpg
images/train/0000342_05685_d_0000271.jpg
images/train/9999994_00000_d_0000073.jpg
images/train/0000273_00001_d_0000439.jpg
images/train/9999998_00392_d_0000344.jpg
images/train/9999937_00000_d_0000162.jpg
images/train/9999967_00000_d_0000043.jpg
images/train/9999998_00064_d_0000049.jpg
images/train/9999999_00189_d_0000078.jpg
images/train/9999972_00000_d_0000156.jpg
images/train/9999943_00000_d_0000032.jpg
images/train/9999965_00000_d_0000067.jpg
images/train/9999937_00000_d_0000080.jpg
images/train/9999972_00000_d_0000090.jpg
images/train/9999984_00000_d_0000126.jpg
images/train/9999999_00137_d_0000053.jpg
images/train/9999942_00000_d_0000122.jpg
images/train/9999937_00000_d_0000033.jpg
images/train/9999984_00000_d_0000106.jpg
images/train/9999953_00000_d_0000019.jpg
images/train/0000315_03001_d_0000516.jpg
images/train/0000204_01028_d_0000194.jpg
images/train/9999982_00000_d_0000045.jpg
images/train/0000339_03529_d_0000179.jpg
images/train/9999955_00000_d_0000362.jpg
images/train/9999956_00000_d_0000044.jpg
images/train/9999955_00000_d_0000300.jpg
images/train/9999981_00000_d_0000098.jpg
images/train/9999937_00000_d_0000192.jpg
images/train/9999953_00000_d_0000230.jpg
images/train/9999984_00000_d_0000034.jpg
images/train/0000352_02941_d_0000554.jpg
images/train/0000236_00001_d_0000001.jpg
images/train/9999951_00000_d_0000046.jpg
images/train/9999965_00000_d_0000056.jpg
images/train/9999960_00000_d_0000147.jpg
images/train/9999942_00000_d_0000149.jpg
images/train/9999981_00000_d_0000030.jpg
images/train/9999972_00000_d_0000054.jpg
images/train/9999942_00000_d_0000254.jpg
images/train/9999960_00000_d_0000089.jpg
images/train/0000315_01001_d_0000506.jpg
images/train/9999955_00000_d_0000127.jpg
images/train/9999977_00000_d_0000047.jpg
images/train/9999943_00000_d_0000062.jpg
images/train/0000225_04522_d_0000015.jpg
images/train/9999999_00618_d_0000281.jpg
images/train/9999955_00000_d_0000077.jpg
images/train/0000352_06665_d_0000573.jpg
images/train/9999942_00000_d_0000132.jpg
images/train/0000352_00785_d_0000543.jpg
images/train/9999955_00000_d_0000327.jpg
images/train/9999940_00000_d_0000059.jpg
images/train/0000180_00213_d_0000020.jpg
images/train/0000307_07001_d_0000303.jpg
images/train/9999998_00137_d_0000115.jpg
images/train/9999948_00000_d_0000033.jpg
images/train/0000339_01765_d_0000170.jpg
images/train/9999972_00000_d_0000057.jpg
images/train/9999942_00000_d_0000196.jpg
images/train/9999953_00000_d_0000171.jpg
images/train/9999990_00000_d_0000016.jpg
images/train/9999951_00000_d_0000137.jpg
images/train/9999937_00000_d_0000088.jpg
images/train/9999964_00000_d_0000029.jpg
images/train/9999994_00000_d_0000006.jpg
images/train/0000329_06801_d_0000799.jpg
images/train/9999994_00000_d_0000094.jpg
images/train/0000175_00001_d_0000001.jpg
images/train/0000134_01332_d_0000146.jpg
images/train/9999999_00715_d_0000325.jpg
images/train/0000209_00922_d_0000222.jpg
images/train/9999977_00000_d_0000027.jpg
images/train/0000201_01634_d_0000179.jpg
images/train/9999955_00000_d_0000274.jpg
images/train/9999955_00000_d_0000150.jpg
images/train/9999955_00000_d_0000140.jpg
images/train/9999955_00000_d_0000309.jpg
images/train/9999997_00000_d_0000029.jpg
images/train/0000342_04117_d_0000263.jpg
images/train/9999937_00000_d_0000090.jpg
images/train/9999956_00000_d_0000095.jpg
images/train/9999960_00000_d_0000013.jpg
images/train/0000169_01098_d_0000001.jpg
images/train/9999999_00219_d_0000092.jpg
images/train/9999990_00000_d_0000049.jpg
images/train/0000142_00458_d_0000025.jpg
images/train/9999998_00155_d_0000131.jpg
images/train/9999937_00000_d_0000105.jpg
images/train/0000181_00889_d_0000030.jpg
images/train/0000264_02001_d_0000204.jpg
images/train/9999994_00000_d_0000013.jpg
images/train/9999997_00000_d_0000007.jpg
images/train/9999960_00000_d_0000149.jpg
images/train/9999981_00000_d_0000011.jpg
images/train/9999974_00000_d_0000015.jpg
images/train/0000303_01601_d_0000191.jpg
images/train/9999945_00000_d_0000078.jpg
images/train/9999950_00000_d_0000071.jpg
images/train/9999984_00000_d_0000155.jpg
images/train/9999972_00000_d_0000029.jpg
images/train/0000165_02725_d_0000097.jpg
images/train/0000135_00118_d_0000149.jpg
images/train/9999999_00586_d_0000265.jpg
images/train/0000142_03658_d_0000041.jpg""".splitlines()
)


def _authority() -> dict[str, str]:
    return {
        "baseline_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "subset_sha256": "c" * 64,
        "runtime_amendment_sha256": "d" * 64,
        "source_commit": "E" * 40,
        "schema_sha256": "f" * 64,
        "dev_sha256": EXPECTED_DEV_SHA256.lower(),
    }


def _cache_records(prefix: str, count: int) -> list[dict[str, object]]:
    tensors = {
        "boxes": torch.full((300, 4), 0.5),
        "logits": torch.linspace(-2.0, 2.0, 3000).reshape(300, 10),
        "target_boxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        "target_classes": torch.tensor([9], dtype=torch.long),
    }
    return [
        {"image_id": f"{prefix}-{index:04d}", **tensors}
        for index in range(count)
    ]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _assert_byte_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert torch.equal(
        actual.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    )


def test_frozen_oracle_constants_are_exact() -> None:
    assert ALPHA_GRID == (0.25, 0.5, 1.0, 2.0)
    assert DEV_COUNT == 129
    assert DEV_SPLIT_SALT == b"rtdetr-quality-oracle-dev-v1\0"
    assert (
        EXPECTED_DEV_SHA256
        == "FCF8749BAADBA8BDDF5870F472BDE1E937156AFBCEEFDA9F96FED21FA6BB0514"
    )
    assert MAP_GAIN_THRESHOLD == Decimal("0.0050")


def test_same_class_iou_quality_uses_exact_per_class_maxima() -> None:
    boxes = torch.tensor(
        [
            [0.50, 0.50, 0.20, 0.20],
            [0.50, 0.50, 0.40, 0.40],
            [0.80, 0.80, 0.20, 0.20],
        ]
    )
    target_boxes = torch.tensor(
        [
            [0.50, 0.50, 0.20, 0.20],
            [0.20, 0.20, 0.20, 0.20],
            [0.80, 0.80, 0.20, 0.20],
        ]
    )
    target_classes = torch.tensor([1, 1, 2])

    quality = same_class_iou_quality(
        boxes, target_boxes, target_classes, num_classes=4
    )

    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.25, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    torch.testing.assert_close(quality, expected, rtol=0, atol=1e-6)
    assert quality.shape == (3, 4)
    assert torch.isfinite(quality).all()
    assert torch.all((quality >= 0) & (quality <= 1))


def test_same_class_iou_quality_returns_zeros_for_empty_targets() -> None:
    quality = same_class_iou_quality(
        torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]),
        torch.empty(0, 4),
        torch.empty(0, dtype=torch.long),
        num_classes=3,
    )

    torch.testing.assert_close(quality, torch.zeros(2, 3), rtol=0, atol=0)
    assert torch.isfinite(quality).all()


@pytest.mark.parametrize(
    ("boxes", "target_boxes", "target_classes", "num_classes", "error"),
    [
        (torch.zeros(1, 2, 4), torch.zeros(1, 4), torch.tensor([0]), 2, ValueError),
        (torch.zeros(2, 4), torch.zeros(1, 3), torch.tensor([0]), 2, ValueError),
        (torch.zeros(2, 4), torch.zeros(1, 4), torch.tensor([[0]]), 2, ValueError),
        (torch.zeros(2, 4), torch.zeros(1, 4), torch.empty(0, dtype=torch.long), 2, ValueError),
        (torch.zeros(2, 4), torch.full((1, 4), float("nan")), torch.tensor([0]), 2, ValueError),
        (torch.zeros(2, 4), torch.zeros(1, 4), torch.tensor([2]), 2, ValueError),
        (torch.zeros(2, 4), torch.zeros(1, 4), torch.tensor([0.0]), 2, TypeError),
        (torch.zeros(2, 4), torch.zeros(1, 4), torch.tensor([0]), 0, ValueError),
    ],
)
def test_same_class_iou_quality_strictly_validates_inputs(
    boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    num_classes: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        same_class_iou_quality(
            boxes, target_boxes, target_classes, num_classes=num_classes
        )


def test_flattened_topk_is_byte_exact_with_ultralytics_8_4_90() -> None:
    fake_head = SimpleNamespace(num_queries=4, nc=3)
    boxes = torch.linspace(0.01, 0.99, 2 * 4 * 4).reshape(2, 4, 4)
    scores = torch.tensor(
        [
            [[0.01, 0.92, 0.11], [0.81, 0.21, 0.71], [0.61, 0.31, 0.51], [0.41, 0.99, 0.91]],
            [[0.93, 0.02, 0.12], [0.22, 0.82, 0.72], [0.32, 0.62, 0.52], [0.42, 0.98, 0.88]],
        ]
    )

    expected = RTDETRDecoder.postprocess(fake_head, boxes, scores)
    actual = flattened_topk(boxes, scores, num_classes=3, max_det=4)

    _assert_byte_equal(actual, expected)


def test_production_shapes_use_default_top_300_contract() -> None:
    fake_head = SimpleNamespace(num_queries=300, nc=10)
    boxes = torch.linspace(0.001, 0.999, 1 * 300 * 4).reshape(1, 300, 4)
    logits = torch.linspace(-5.0, 5.0, 1 * 300 * 10).reshape(1, 300, 10)
    scores = logits.sigmoid()
    qualities = torch.linspace(0.1, 1.0, 1 * 300 * 10).reshape(1, 300, 10)

    expected_stock = RTDETRDecoder.postprocess(fake_head, boxes, scores)
    stock = flattened_topk(boxes, scores, num_classes=10)
    oracle = oracle_topk(
        boxes,
        logits,
        qualities,
        alpha=0.5,
        num_classes=10,
    )
    quality = same_class_iou_quality(
        boxes[0],
        boxes[0, :2],
        torch.tensor([0, 9]),
        num_classes=10,
    )

    _assert_byte_equal(stock, expected_stock)
    assert stock.shape == oracle.shape == (1, 300, 6)
    assert quality.shape == (300, 10)


def test_flattened_topk_keeps_duplicate_queries_for_different_classes() -> None:
    boxes = torch.tensor(
        [[[0.10, 0.20, 0.30, 0.40], [0.50, 0.60, 0.70, 0.80]]]
    )
    scores = torch.tensor([[[0.99, 0.98, 0.10], [0.97, 0.96, 0.95]]])

    selected = flattened_topk(boxes, scores, num_classes=3, max_det=4)

    expected = torch.tensor(
        [
            [
                [0.10, 0.20, 0.30, 0.40, 0.99, 0.0],
                [0.10, 0.20, 0.30, 0.40, 0.98, 1.0],
                [0.50, 0.60, 0.70, 0.80, 0.97, 0.0],
                [0.50, 0.60, 0.70, 0.80, 0.96, 1.0],
            ]
        ]
    )
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)
    assert selected.shape == (1, 4, 6)
    _assert_byte_equal(selected[0, 0, :4], boxes[0, 0])
    _assert_byte_equal(selected[0, 1, :4], boxes[0, 0])


@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0, 2.0])
def test_oracle_topk_uses_sigmoid_quality_power_then_flattened_topk(
    alpha: float,
) -> None:
    boxes = torch.tensor(
        [[[0.10, 0.20, 0.30, 0.40], [0.50, 0.60, 0.70, 0.80]]]
    )
    logits = torch.tensor([[[4.0, -4.0], [2.0, -2.0]]])
    qualities = torch.tensor([[[0.10, 0.50], [1.00, 0.25]]])
    expected_scores = logits.sigmoid() * qualities**alpha

    expected = flattened_topk(
        boxes, expected_scores, num_classes=2, max_det=3
    )
    actual = oracle_topk(
        boxes,
        logits,
        qualities,
        alpha=alpha,
        num_classes=2,
        max_det=3,
    )

    _assert_byte_equal(actual, expected)
    assert actual.shape == (1, 3, 6)


@pytest.mark.parametrize("alpha", [True, 0.0, 0.75, 4.0, float("nan")])
def test_oracle_topk_rejects_alpha_outside_frozen_grid(alpha: float) -> None:
    with pytest.raises(ValueError, match="ALPHA_GRID"):
        oracle_topk(
            torch.zeros(1, 1, 4),
            torch.zeros(1, 1, 1),
            torch.ones(1, 1, 1),
            alpha=alpha,
            num_classes=1,
            max_det=1,
        )


@pytest.mark.parametrize(
    ("boxes", "logits", "qualities", "error"),
    [
        (torch.zeros(1, 4), torch.zeros(1, 1, 1), torch.ones(1, 1, 1), ValueError),
        (torch.zeros(1, 1, 4), torch.zeros(1, 2, 1), torch.ones(1, 2, 1), ValueError),
        (torch.zeros(1, 1, 4), torch.zeros(1, 1, 2), torch.ones(1, 1, 1), ValueError),
        (torch.zeros(1, 1, 4), torch.full((1, 1, 1), float("inf")), torch.ones(1, 1, 1), ValueError),
        (torch.zeros(1, 1, 4), torch.zeros(1, 1, 1), torch.full((1, 1, 1), float("nan")), ValueError),
        (torch.zeros(1, 1, 4), torch.zeros(1, 1, 1), torch.full((1, 1, 1), 1.1), ValueError),
    ],
)
def test_oracle_topk_rejects_mismatched_or_nonfinite_inputs(
    boxes: torch.Tensor,
    logits: torch.Tensor,
    qualities: torch.Tensor,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        oracle_topk(
            boxes,
            logits,
            qualities,
            alpha=0.5,
            num_classes=1,
            max_det=1,
        )


@pytest.mark.parametrize(
    ("boxes", "scores"),
    [
        (torch.zeros(1, 4), torch.zeros(1, 1, 1)),
        (torch.zeros(1, 1, 4), torch.zeros(1, 1)),
        (torch.zeros(1, 2, 4), torch.zeros(1, 1, 1)),
        (torch.zeros(1, 1, 4), torch.zeros(1, 1, 2)),
        (torch.zeros(1, 1, 4), torch.full((1, 1, 1), float("nan"))),
    ],
)
def test_flattened_topk_rejects_invalid_shapes_or_scores(
    boxes: torch.Tensor, scores: torch.Tensor
) -> None:
    with pytest.raises(ValueError):
        flattened_topk(boxes, scores, num_classes=1, max_det=1)


def test_real_authorized_647_path_split_has_the_frozen_hash_and_order() -> None:
    root = Path("authorized-visdrone-root")
    paths = tuple(root / relative for relative in _REAL_AUTHORIZED_RELATIVE_PATHS)

    selected = select_internal_dev(paths, root=root)
    reversed_selected = select_internal_dev(tuple(reversed(paths)), root=root)

    assert len(paths) == len(set(paths)) == 647
    assert len(selected) == DEV_COUNT
    assert selected == reversed_selected
    assert ordered_path_sha256(selected, root=root) == EXPECTED_DEV_SHA256


def test_ordered_path_sha256_uses_relative_posix_utf8_lf_and_trailing_lf(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    paths = (root / "z" / "β.jpg", root / "a.jpg")
    expected = hashlib.sha256("z/β.jpg\na.jpg\n".encode("utf-8")).hexdigest().upper()

    assert ordered_path_sha256(paths, root=root) == expected


def test_internal_dev_rejects_wrong_count_duplicates_and_outside_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    paths = [root / "images" / "train" / f"{index:04d}.jpg" for index in range(647)]

    with pytest.raises(ValueError, match="647"):
        select_internal_dev(paths[:-1], root=root)

    duplicate = list(paths)
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="unique"):
        select_internal_dev(duplicate, root=root)

    outside = list(paths)
    outside[-1] = root.parent / "outside.jpg"
    with pytest.raises(ValueError, match="root"):
        select_internal_dev(outside, root=root)


def test_path_contract_rejects_malformed_values(tmp_path: Path) -> None:
    root = tmp_path / "root"
    paths = [root / "images" / "train" / f"{index:04d}.jpg" for index in range(647)]

    malformed_type: list[object] = list(paths)
    malformed_type[0] = "images/train/0000.jpg"
    with pytest.raises(TypeError, match="Path"):
        select_internal_dev(malformed_type, root=root)  # type: ignore[arg-type]

    malformed_name = list(paths)
    malformed_name[0] = root / "images" / "train" / "bad\nname.jpg"
    with pytest.raises(ValueError, match="path"):
        select_internal_dev(malformed_name, root=root)

    with pytest.raises(ValueError, match="unique"):
        ordered_path_sha256((paths[0], paths[0]), root=root)
    with pytest.raises(ValueError, match="root"):
        ordered_path_sha256((root.parent / "outside.jpg",), root=root)


def test_quality_cache_roundtrip_is_canonical_hashed_fsynced_and_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cache"
    authority = _authority()
    real_fsync = os.fsync
    fsync_calls: list[int] = []

    def tracked_fsync(file_descriptor: int) -> None:
        fsync_calls.append(file_descriptor)
        real_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", tracked_fsync)
    manifest = write_quality_oracle_cache(
        root,
        dev=_cache_records("dev", 129),
        val=_cache_records("val", 548),
        authority=authority,
    )

    assert len(fsync_calls) >= 3
    assert {path.name for path in root.iterdir()} == {"dev.pt", "val.pt", "manifest.json"}
    assert manifest["complete"] is True
    assert manifest["split_counts"] == {"dev": 129, "val": 548}
    assert manifest["authority"] == {
        **{name: value.upper() for name, value in authority.items() if name != "source_commit"},
        "source_commit": authority["source_commit"].lower(),
    }
    expected_manifest = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    assert (root / "manifest.json").read_bytes() == expected_manifest
    for artifact in manifest["artifacts"]:
        artifact_path = root / artifact["path"]
        assert artifact["bytes"] == artifact_path.stat().st_size
        assert artifact["sha256"] == _file_sha256(artifact_path)

    real_load = torch.load
    load_calls: list[dict[str, object]] = []

    def tracked_load(*args: object, **kwargs: object) -> object:
        load_calls.append(dict(kwargs))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    loaded = load_quality_oracle_cache(root, authority=authority)

    assert [call["weights_only"] for call in load_calls] == [True, True]
    assert [call["map_location"] for call in load_calls] == ["cpu", "cpu"]
    assert tuple(loaded) == ("dev", "val")
    assert len(loaded["dev"]) == 129
    assert len(loaded["val"]) == 548
    assert loaded["dev"][0]["image_id"] == "dev-0000"
    assert torch.equal(loaded["val"][-1]["logits"], torch.linspace(-2.0, 2.0, 3000).reshape(300, 10))


def test_quality_cache_root_is_strictly_create_only(tmp_path: Path) -> None:
    dev = _cache_records("dev", 129)
    val = _cache_records("val", 548)
    authority = _authority()
    empty_root = tmp_path / "already-exists"
    empty_root.mkdir()

    with pytest.raises(FileExistsError, match="cache root"):
        write_quality_oracle_cache(empty_root, dev=dev, val=val, authority=authority)

    root = tmp_path / "created"
    write_quality_oracle_cache(root, dev=dev, val=val, authority=authority)
    with pytest.raises(FileExistsError, match="cache root"):
        write_quality_oracle_cache(root, dev=dev, val=val, authority=authority)


def test_quality_cache_requires_exact_authority_and_normalizes_hashes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "cache"
    authority = _authority()
    write_quality_oracle_cache(
        root,
        dev=_cache_records("dev", 129),
        val=_cache_records("val", 548),
        authority=authority,
    )

    changed = dict(authority)
    changed["baseline_sha256"] = "0" * 64
    with pytest.raises(QualityOracleCacheViolation, match="authority.*baseline_sha256"):
        load_quality_oracle_cache(root, authority=changed)

    missing = dict(authority)
    missing.pop("schema_sha256")
    with pytest.raises(QualityOracleCacheViolation, match="authority schema"):
        load_quality_oracle_cache(root, authority=missing)

    invalid = dict(authority)
    invalid["source_commit"] = "not-a-commit"
    with pytest.raises(QualityOracleCacheViolation, match="source_commit"):
        load_quality_oracle_cache(root, authority=invalid)


def test_quality_cache_rejects_corruption_and_incomplete_manifest(tmp_path: Path) -> None:
    authority = _authority()
    corrupt_root = tmp_path / "corrupt"
    manifest = write_quality_oracle_cache(
        corrupt_root,
        dev=_cache_records("dev", 129),
        val=_cache_records("val", 548),
        authority=authority,
    )
    artifact_path = corrupt_root / manifest["artifacts"][0]["path"]
    with artifact_path.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(QualityOracleCacheViolation, match="bytes|sha256"):
        load_quality_oracle_cache(corrupt_root, authority=authority)

    incomplete_root = tmp_path / "incomplete"
    incomplete = write_quality_oracle_cache(
        incomplete_root,
        dev=_cache_records("dev", 129),
        val=_cache_records("val", 548),
        authority=authority,
    )
    incomplete["complete"] = False
    with (incomplete_root / "manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        json.dump(incomplete, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    with pytest.raises(QualityOracleCacheViolation, match="complete"):
        load_quality_oracle_cache(incomplete_root, authority=authority)


def test_quality_cache_revalidates_loaded_payload_schema(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    authority = _authority()
    manifest = write_quality_oracle_cache(
        root,
        dev=_cache_records("dev", 129),
        val=_cache_records("val", 548),
        authority=authority,
    )
    dev_artifact = next(
        artifact for artifact in manifest["artifacts"] if artifact["split"] == "dev"
    )
    artifact_path = root / dev_artifact["path"]
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    payload["records"][0]["unexpected"] = torch.tensor(1)
    torch.save(payload, artifact_path)
    dev_artifact["bytes"] = artifact_path.stat().st_size
    dev_artifact["sha256"] = _file_sha256(artifact_path)
    with (root / "manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")

    with pytest.raises(QualityOracleCacheViolation, match="record schema"):
        load_quality_oracle_cache(root, authority=authority)


def test_quality_cache_rejects_overlap_duplicates_and_wrong_counts(tmp_path: Path) -> None:
    authority = _authority()
    dev = _cache_records("dev", 129)
    val = _cache_records("val", 548)

    overlap = _cache_records("val", 548)
    overlap[0] = {**overlap[0], "image_id": dev[0]["image_id"]}
    with pytest.raises(QualityOracleCacheViolation, match="overlap"):
        write_quality_oracle_cache(
            tmp_path / "overlap", dev=dev, val=overlap, authority=authority
        )

    duplicate = list(dev)
    duplicate[1] = {**duplicate[1], "image_id": duplicate[0]["image_id"]}
    with pytest.raises(QualityOracleCacheViolation, match="unique"):
        write_quality_oracle_cache(
            tmp_path / "duplicate", dev=duplicate, val=val, authority=authority
        )

    with pytest.raises(QualityOracleCacheViolation, match="129"):
        write_quality_oracle_cache(
            tmp_path / "count", dev=dev[:-1], val=val, authority=authority
        )


@pytest.mark.parametrize(
    ("field", "unsafe"),
    [
        ("boxes", torch.zeros(4, 300).T),
        ("boxes", torch.zeros(300, 4, requires_grad=True)),
        ("boxes", torch.zeros(300, 4, dtype=torch.long)),
        ("logits", torch.full((300, 10), float("nan"))),
        ("target_boxes", torch.tensor([[1.1, 0.5, 0.2, 0.2]])),
        ("target_classes", torch.tensor([0.0])),
        ("target_classes", torch.tensor([10])),
    ],
)
def test_quality_cache_rejects_unsafe_tensors(
    tmp_path: Path, field: str, unsafe: torch.Tensor
) -> None:
    dev = _cache_records("dev", 129)
    dev[0] = {**dev[0], field: unsafe}

    with pytest.raises(QualityOracleCacheViolation):
        write_quality_oracle_cache(
            tmp_path / field,
            dev=dev,
            val=_cache_records("val", 548),
            authority=_authority(),
        )


def test_quality_cache_records_have_an_exact_schema(tmp_path: Path) -> None:
    dev = _cache_records("dev", 129)
    dev[0] = {**dev[0], "unexpected": torch.tensor(1)}

    with pytest.raises(QualityOracleCacheViolation, match="record schema"):
        write_quality_oracle_cache(
            tmp_path / "cache",
            dev=dev,
            val=_cache_records("val", 548),
            authority=_authority(),
        )


def test_select_alpha_uses_frozen_lexicographic_order_and_smaller_tie() -> None:
    tied = {
        alpha: {"map": 0.30, "ap75": 0.20, "ap50": 0.40, "precision": 0.9}
        for alpha in ALPHA_GRID
    }
    assert select_alpha(tied) == 0.25

    metrics = {
        0.25: {"map": 0.30, "ap75": 0.25, "ap50": 0.50},
        0.5: {"map": 0.31, "ap75": 0.10, "ap50": 0.10},
        1.0: {"map": 0.31, "ap75": 0.26, "ap50": 0.10},
        2.0: {"map": 0.31, "ap75": 0.26, "ap50": 0.51},
    }
    assert select_alpha(metrics) == 2.0


def test_select_alpha_requires_exact_grid_complete_finite_metrics() -> None:
    valid = {
        alpha: {"map": 0.30, "ap75": 0.20, "ap50": 0.40}
        for alpha in ALPHA_GRID
    }
    missing_alpha = dict(valid)
    missing_alpha.pop(2.0)
    with pytest.raises(ValueError, match="ALPHA_GRID"):
        select_alpha(missing_alpha)

    extra_alpha = {**valid, 4.0: valid[2.0]}
    with pytest.raises(ValueError, match="ALPHA_GRID"):
        select_alpha(extra_alpha)

    missing_metric = {alpha: dict(metrics) for alpha, metrics in valid.items()}
    missing_metric[0.5].pop("ap75")
    with pytest.raises(ValueError, match="ap75"):
        select_alpha(missing_metric)

    nonfinite = {alpha: dict(metrics) for alpha, metrics in valid.items()}
    nonfinite[1.0]["map"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        select_alpha(nonfinite)


def test_quality_gate_passes_at_exact_map_boundary_with_positive_ap75() -> None:
    decision = decide_quality_oracle(
        stock_map=0.20,
        stock_ap75=0.18,
        oracle_map=0.205,
        oracle_ap75=0.180001,
    )

    assert decision == {
        "status": "passed",
        "finite": True,
        "observed": {
            "stock_map": Decimal("0.2"),
            "stock_ap75": Decimal("0.18"),
            "oracle_map": Decimal("0.205"),
            "oracle_ap75": Decimal("0.180001"),
        },
        "deltas": {"map": Decimal("0.005"), "ap75": Decimal("0.000001")},
        "thresholds": {"map": Decimal("0.0050"), "ap75": Decimal("0")},
    }


@pytest.mark.parametrize(
    ("oracle_map", "oracle_ap75"),
    [
        (0.204999, 0.20),
        (0.21, 0.18),
    ],
)
def test_quality_gate_enforces_map_minimum_and_strict_ap75(
    oracle_map: float, oracle_ap75: float
) -> None:
    decision = decide_quality_oracle(
        stock_map=0.20,
        stock_ap75=0.18,
        oracle_map=oracle_map,
        oracle_ap75=oracle_ap75,
    )

    assert decision["status"] == "scientific_failed"
    assert decision["finite"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stock_map", float("nan")),
        ("stock_ap75", float("inf")),
        ("oracle_map", float("-inf")),
        ("oracle_ap75", float("nan")),
    ],
)
def test_quality_gate_rejects_nonfinite_inputs_as_engineering_invalid(
    field: str, value: float
) -> None:
    metrics = {
        "stock_map": 0.20,
        "stock_ap75": 0.18,
        "oracle_map": 0.21,
        "oracle_ap75": 0.19,
    }
    metrics[field] = value

    with pytest.raises(ValueError, match="finite"):
        decide_quality_oracle(**metrics)
