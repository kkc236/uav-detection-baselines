# RT-DETR Quality-Reordering Oracle Result

- Status: `passed`
- Selected alpha: `2.0`
- Stock mAP: `0.24164844987309864`
- Oracle mAP: `0.3973619055936227`
- mAP delta: `+0.15571345572052406`
- Stock AP75: `0.23916375458831637`
- Oracle AP75: `0.3883675963852108`
- AP75 delta: `+0.14920384179689443`
- Evidence source commit: `29d402a21beb54bfffed9d5bd79c7aeeef6afe31`
- Resume runner commit: `11be5d9a3bf9a6875ead791758f23f9d985bb9c1`

The result is a perfect ground-truth same-class quality-reordering upper bound. It
establishes substantial score-ordering headroom but is not itself a deployable model.
The next authorized stage is the frozen C0/C1/Q learnable-quality probe.

The native auxiliary-tuple path reproduced all AP authority metrics exactly. Precision,
which is not part of the scientific Gate, differed from the former adapter-path authority
by approximately `1.76e-9`; the immutable report records the bounded `1e-8` diagnostic
floating-point amendment. Official-validation inference was not repeated during resume.
