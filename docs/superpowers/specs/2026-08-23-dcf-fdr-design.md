# DCF-FDR Single-Module Design

## Status and objective

This document freezes the design of one small internal FDR modification before
implementation. The method starts from the clean native-reference/no-DN FDR
configuration and adds exactly one module: Distribution-Conditioned Feedback
(DCF). It does not add a second mechanism, a new loss, score calibration,
distillation, feature enhancement, matching changes, or post-processing.

The primary claim tested by the experiment is deliberately narrow:

> Conditioning the next FDR residual head on the preceding complete boundary
> distribution preserves or slightly improves the stronger clean cumulative
> FDR obtained after removing the two unsupported historical mechanisms.

The principal gain is expected to come from simplifying AP-FDR by removing the
preliminary-reference path and extra DN-side FDR supervision. DCF is retained
when its on arm is non-negative relative to the exact DCF-off Clean FDR arm
under the frozen evaluation contract. No positive result is assumed in
advance, and DCF must not be credited with the gain caused by removing the two
historical mechanisms.

## Clean FDR authority

The direct baseline disables the two historical AP-FDR mechanisms and EAW:

- `preliminary_box: false`;
- `supervise_pre_boxes: false`;
- `supervise_dn_fdr: false`;
- `edge_adaptive_fgl: false`;
- `cumulative: true`.

It retains the native RT-DETR reference, six decoder layers, 33 bins per edge,
normal-query FGL, native classification/L1/GIoU losses, native RT-DETR
denoising losses, 300 queries, Hungarian matching, and the frozen
post-processing contract. EQuAL is not a valid clean baseline while EAW is
enabled.

Before DCF training, this exact clean arm must be evaluated or trained under
the same source, initialization, data, and protocol authority used by the DCF
arm.

## Audited limitation in the current implementation

The existing decoder predicts a distribution residual as:

```text
delta_corners = bbox_head[index](output + output_detach)
cumulative_corners = cumulative_corners + delta_corners
refined = Integral(cumulative_corners, initial_reference)
```

Consequently, the existing implementation already contains two forms of
cross-layer information:

1. the preceding decoder feature is reused through `output_detach`;
2. the preceding decoded box updates the next decoder reference.

DCF must therefore not be described as the first cross-layer feedback method.
Its defensible gap is narrower: the complete preceding 4-by-33 distribution
shape is not an input to the head that predicts the next distribution
residual. Cumulative addition preserves previous logits after the new residual
has been predicted, while the integral reference preserves mainly decoded
geometry. Neither operation explicitly conditions residual generation on
whether each preceding edge distribution is sharp, flat, asymmetric, or
multimodal.

The method distinguishes:

```text
existing FDR: distribution accumulation
DCF-FDR:      distribution-conditioned residual generation
```

## The single DCF module

Let `Z[l-1]` be the cumulative preceding logits with shape
`[batch, queries, 4, 33]`. DCF computes:

```text
P = softmax(stop_gradient(Z[l-1]), dim=-1)
edge = SharedLinear33x16(P)            # [B, Q, 4, 16]
edge = SiLU(edge)
state = flatten_edges(edge)            # [B, Q, 64]
feedback = ZeroInitLinear64x256(state) # [B, Q, 256]
```

The same DCF instance is shared by decoder layers 2 through 6. Layer 1 has no
preceding FDR distribution and receives zero feedback.

The current residual becomes:

```text
regression_input = output + output_detach + feedback
delta_corners = bbox_head[index](regression_input)
cumulative_corners = cumulative_corners + delta_corners
```

DCF is injected only into the distribution regression input. `score_head`
continues to consume the decoder's normal `output`, and its structure and
direct call contract remain unchanged. As in any iterative DETR box refinement,
changed boxes update later references and may therefore indirectly change later
decoder features and scores. Decoder-layer architecture, matcher inputs, FGL
targets, and returned tensor contracts remain unchanged.

The final `64 -> 256` projection is initialized to exact zeros. Thus DCF starts
as an identity addition, while gradients can first train the output projection
and subsequently the shared edge encoder. The adapter must use the existing
private RNG contract for every nonzero initialization and must not consume the
public training RNG stream.

The design fixes the edge bottleneck to 16 and does not expose a collection of
paper-tuned statistics, gates, temperatures, or layer-specific adapters. This
keeps DCF one module and limits the method's degrees of freedom.

## Why full probability distributions are used

DCF consumes normalized 33-bin probabilities instead of hand-crafted entropy,
variance, top-k, and inter-layer-change branches. A learned shared edge encoder
can retain whichever distribution shapes are useful without turning the
method into a collection of separately named components. Normalization also
prevents arbitrary cumulative-logit offsets from becoming an input signal.

The four edge embeddings are concatenated rather than averaged because left,
top, right, and bottom ambiguities need not be interchangeable after encoding.
The `33 -> 16` encoder weights are shared across edges to keep parameter cost
small and avoid four independently learned submodules.

Approximate added weights are:

```text
33 * 16 + 16 bias + 64 * 256 + 256 bias ~= 17.2K parameters
```

No additional prediction is returned and no additional loss is required.

## Gradient and compatibility contract

The preceding distribution is detached before softmax. DCF treats it as a
causal state for the next refinement rather than creating a second backward
path across decoder layers. This reduces the risk of early distributions
changing merely to encode a convenient hidden message for later heads and
keeps future BPDD gradients separable from DCF.

DCF must remain compatible with BPDD and FIA, but neither is enabled in the
standalone DCF ablation. BPDD may later consume the unchanged cumulative
corner-logit stack; FIA remains upstream and requires no DCF-specific branch.
Compatibility means graph and tensor-contract compatibility, not permission to
claim DCF gains from a combined run.

## Adversarial audit

### Objection: previous logits are already accumulated

This is the strongest objection. DCF is redundant if accumulation alone gives
the residual head all necessary information. The implementation shows it does
not: `bbox_head[index]` is evaluated before `delta_corners` is added to
`cumulative_corners`, and its inputs contain decoder features rather than the
preceding distribution tensor. DCF tests whether distribution shape is useful
at residual-generation time. If the controlled ablation is non-positive, the
redundancy objection wins and DCF is rejected.

### Objection: this is GFLv2/D-FINE LQE under another name

LQE maps distribution statistics to an absolute localization-quality score
and affects ranking or confidence. DCF maps a preceding distribution to a
feature used only to predict the next localization residual. It neither emits
a quality score nor changes classification scores. The paper must not call DCF
a quality estimator or confidence calibrator.

### Objection: this is a generic MLP adapter

The MLP itself is not claimed as novel. The method claim concerns the explicit
distribution-to-residual conditioning path inside cumulative FDR. A generic
query adapter or feature enhancement baseline would be outside this claim. The
paper must describe the information path and the lost-distribution-shape
motivation, not advertise the linear layers.

### Objection: `output_detach` already carries the same information

Previous decoder features can implicitly encode localization state, but they
are not constrained to preserve the normalized four-edge distribution shape.
DCF supplies this state explicitly. The on/off result decides whether the
explicit path adds information beyond `output_detach`; no assertion of
complementarity is made without a positive controlled result.

### Objection: the module may only sharpen boxes and reduce recall

DCF is a localization refinement method, not a guaranteed classification
recall mechanism. It may reduce localization-induced false negatives, but it
cannot promise recovery of low-classification-score queries. A result with
higher precision but lower recall and lower mAP is a rejection, not a success.

### Objection: stopping the gradient makes the method weak

Stop-gradient is intentional because DCF conditions later refinement on an
observed preceding state. The main test is whether that state improves the next
residual. A non-detached version is not part of the frozen primary method and
must not be substituted after seeing the formal result without starting a new
method audit.

### Objection: the contribution is too small

The intended venue strategy is a small, architecture-preserving improvement,
not a claim that DCF reinvents FDR. The contribution is publishable only when
paired with a clear controlled gain, negligible overhead, exact compatibility,
and the broader verified FDR/BPDD/FIA evidence. DCF alone must not be presented
as multiple innovations.

## Required implementation tests

Implementation is not accepted until tests demonstrate:

1. DCF-off is byte-identical to Clean FDR for the same state and input.
2. DCF-on with its zero-initialized projection is initially output-equivalent
   to Clean FDR.
3. Layer 1 does not call DCF; layers 2-6 use the shared instance.
4. DCF receives `[B,Q,4,33]` probabilities and returns `[B,Q,256]`.
5. The input distribution is detached and receives no gradient through DCF.
6. The DCF path changes corner logits after nonzero adapter weights are loaded.
7. Score heads retain their original structure and continue to receive decoder
   outputs rather than DCF-augmented regression inputs.
8. Public RNG state is unchanged by DCF construction.
9. Existing FDR, loss, checkpoint, BPDD, and FIA contract tests continue to
   pass.
10. DCF checkpoints load deterministically and DCF-off rejects or explicitly
    handles unexpected DCF weights rather than silently changing the graph.

## Experimental design and decision rule

The mandatory internal ablation has exactly two method rows:

| Method | Native reference | Extra DN-FDR supervision | EAW | DCF |
|---|---:|---:|---:|---:|
| Clean FDR | yes | no | no | no |
| DCF-FDR | yes | no | no | yes |

Both rows use the same source parent, initial state, seed, dataset signature,
optimizer, epochs, image size, batch, workers, checkpoint selection rule, and
exact val/test evaluator. Parameter and latency differences are reported.

The primary metric is validation mAP50-95. Recall, AP50, AP75, and tiny/small
AP diagnose the effect but do not replace the primary metric after results are
known. The simplification row, rather than DCF, carries the main expected gain.

DCF is retained only if all conditions hold:

- validation mAP50-95 is not lower than Clean FDR at the frozen checkpoint
  selection precision;
- no secondary metric exhibits a material regression that contradicts the
  paper's stated localization or recall interpretation;
- training completes without non-finite gradients or evidence-contract
  failures;
- inference overhead remains within the previously accepted 3% budget.

An exactly tied or marginally positive result is sufficient for the
non-negative internal ablation, but it supports only a conservative claim that
DCF preserves the simplified FDR performance. It does not support describing
DCF as the source of the main gain. A precision-only improvement accompanied
by lower mAP rejects DCF. Best-epoch cherry-picking across different selection
rules, evaluator changes, or post-hoc protocol changes is prohibited.

If DCF passes the standalone validation decision, run the frozen test
evaluation and only then consider DCF+BPDD and DCF+BPDD+FIA compatibility
experiments. Combined runs cannot rescue a failed standalone DCF result.

## Paper-safe wording

Permitted wording before results:

> We introduce a lightweight distribution-conditioned feedback adapter that
> exposes the preceding per-edge probability distributions to the next FDR
> residual head while preserving the original classification, matching,
> supervision, and decoding contracts.

Permitted wording only after a clearly positive ablation:

> Explicit distribution-conditioned residual generation improves the clean
> cumulative FDR baseline with negligible computational overhead.

Permitted wording after a numerical tie or marginal non-negative result:

> The lightweight distribution-conditioned adapter preserves the gain of the
> simplified FDR formulation without changing its detection contract.

Prohibited wording:

- DCF is the first cross-layer FDR mechanism;
- DCF guarantees recall recovery;
- the linear adapter itself is a new attention architecture;
- DCF is independent evidence if its standalone row is non-positive;
- DCF gains can be inferred from EQuAL, BPDD, or FIA combinations.

## Frozen conclusion

The proposed method is one module, one switch, and one direct ablation. The
main improvement is attributed to removal of the preliminary-reference and
extra DN-FDR mechanisms. DCF is a retained internal refinement only when its
controlled result is non-negative against the resulting
native-reference/no-DN/no-EAW Clean FDR baseline. If that comparison is
negative, the module is removed rather than expanded into additional branches.
