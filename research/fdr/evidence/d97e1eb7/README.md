# FDR d97e1eb7 paired screen evidence

This directory is an immutable extraction of the completed fixed-10% VisDrone,
seed-0, 30-epoch paired FDR/control screen.

- Model/training authority: `d97e1eb7f98414752a1c1f38287697db3f2a0679`
- Gate evaluator authority: `1cc64045560b69a07b9a8699019bc02fe298c488`
- Source archive SHA256: `8FF7BBE8845EC147B60C27082DCF88CA5EAFF5D89CA8A700CFB41A2E0671ED05`
- Engineering checks: all passed
- Gate2: passed
- Final mAP delta (FDR - control): `+0.01801`
- Tail-3 mean mAP delta: `+0.0156366667`
- Final AP75 delta: `+0.0154278605`

The full-data seed-0 FDR run was fresh-started after Gate2 passed. It does not
resume from either screen checkpoint.
