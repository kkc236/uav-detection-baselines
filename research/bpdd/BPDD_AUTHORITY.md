# BPDD research authority and claim boundary

BPDD is a parameter-free, training-only candidate on top of the validated FDR
RT-DETR-L detector. It is not a claim that self-distillation, localization
distribution distillation, later-to-earlier supervision, or better-teacher
gating was invented in this project.

The literature collision audit found no exact prior implementation of the full
combination, but it found direct precedents for every broad ingredient. The
narrow candidate contribution is therefore limited to:

1. an FDR edge distribution teacher made from only future decoder layers;
2. softmin mixing by the same two-bin GT proper score used by FGL;
3. evaluation of the actual mixture, not an average of component errors;
4. a detached better-only reliability weight;
5. one final stock Hungarian assignment, matched normal Queries only;
6. no inference branch, parameter, score change, matching union, or unmatched
   Query distillation.

The immutable machine-readable authority is `authority.json`. Its source,
dataset, FDR protocol, initial-state, and mature FDR checkpoint hashes must be
copied into every BPDD protocol manifest. The BPDD source commit and exact YAML
options receive separate hashes after implementation.

## No-go conditions for a paper claim

- BPDD fails to beat a final-layer teacher with the same better-only gate.
- The gain requires unmatched Queries, a matching union, or changed inference.
- The gain exists only at an intermediate/best checkpoint and disappears at
  epoch 100.
- The exact final checkpoint cannot be tied to the independent evaluator.
- A fresh same-source FDR control does not confirm the historical comparison.

Scientific failure must be preserved and reported. It cannot be converted into
success by changing thresholds after viewing validation results.
