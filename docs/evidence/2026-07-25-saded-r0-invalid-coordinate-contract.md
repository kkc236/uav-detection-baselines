# SADED R0 invalidation: fused-box coordinate contract

Status: `INVALID_SOFTWARE_CONTRACT`

The first authoritative SADED R0 replay under source commit
`43360575240af80bb98eb9ee4a036f8b7cdb921c` is retained but is forbidden as
scientific evidence.

Retained server artifacts:

- route:
  `/home/ubuntu/saded-r0-43360575-20260724T193246Z`
- evaluation:
  `/home/ubuntu/saded-r0-eval-43360575-20260724T194620Z`
- route anchor SHA256:
  `0955a76e9e50f950515a8c4045e29e3eb07153d5dbd8b645a38bc3cdf2087901`

Reason:

`Detection.box` contains the actual post-fusion output coordinates, while
`global_xyxy` may retain only the fusion cluster seed coordinates. The invalid
implementation classified routed candidates by `global_xyxy` and evaluated
the unified output with `frozen_global=True`. A fused non-tiny output could
therefore pass the tiny-only invariant, and the evaluator could score a seed
box instead of the emitted box.

The defect was identified before the evaluation metrics or gate decision were
consumed. No metric from these directories may be reported, reused, or used to
change a scientific threshold.

Repair protocol:

1. reproduce both coordinate divergences with failing tests;
2. use `Detection.box` for routing scale and final metric coordinates;
3. validate both the actual box and authenticated provenance box;
4. keep all scientific constants and gates unchanged;
5. commit the repair and rerun route plus evaluation into new directories;
6. retain the invalid directories permanently.
