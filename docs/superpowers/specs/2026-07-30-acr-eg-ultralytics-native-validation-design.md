# ACR-EG Ultralytics Native Validation Design

## Goal

Use Ultralytics 8.4.90 detection metrics to evaluate the sealed mature
RT-DETR-L baseline and the final epoch-100 ACR-EG checkpoint on the same
548-image VisDrone validation set without retraining.

## Evidence boundary

The existing live evaluation uses the project SBR evaluator. Its predictions
are valid evidence, but its aggregate metrics are not Ultralytics native
metrics. Training also used `val=False`, so zero-valued validation columns in
`results.csv` are placeholders.

The native evaluation must use the same checkpoints as the completed live
evaluation:

- mature baseline SHA256
  `54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B`;
- final ACR-EG SHA256
  `66E0B8D27706CDA594BE657B20BFD01CAA536D90B7EA0A05EDC2FEEC11C6E2B4`,
  checkpoint epoch `99`.

## Architecture

Create a focused native-validation module that owns no detection or metric
formula. It will:

1. load and identity-check both checkpoint models through the sealed live
   checkpoint loader;
2. construct the existing deterministic global-plus-four-local-view
   validation dataset;
3. run the mature baseline with one global view and ACR-EG with one global
   plus four local views;
4. pass both outputs through Ultralytics 8.4.90
   `RTDETRValidator.postprocess()`, `update_metrics()` and `DetMetrics`;
5. extract macro Precision, Recall, AP50, AP75 and AP50-95 from the official
   `Metric` object;
6. write paired metrics, method-minus-baseline deltas, checkpoint identities,
   dataset identity and protocol to canonical JSON.

The ACR-EG arm must fail closed if `last_acr_eg_output` is missing after any
image. This prevents silent single-view fallback.

## Frozen protocol

- dataset: 548 VisDrone validation images;
- dataset signature:
  `A9A0C00DC640BCAAEFE9360F5E3B55382E74E169B5AEEF15EB1F0AE2A571228A`;
- Ultralytics: `8.4.90`;
- global input: `640`;
- local inputs: four `640` views for ACR-EG only;
- batch: `1`;
- workers: `0`;
- device: CUDA device `0`;
- AMP: enabled;
- confidence floor: `0.001`;
- max detections: `300`;
- NMS: not applied by the RT-DETR validator;
- plots, JSON export and text export: disabled.

Batch size and worker count are runtime controls and do not alter the metric
semantics. They are fixed here to match the completed paired live evaluation
and fit the five-view model on the local GPU.

## Metrics

All requested values come from Ultralytics `DetMetrics.box`:

- Precision: `mp`;
- Recall: `mr`;
- AP50: `map50`;
- AP75: `map75`;
- mAP50-95: `map`.

The result also records the ordinary Ultralytics results dictionary and
per-class summary for auditability.

## Tests

- protocol drift must fail closed;
- metric extraction must read all five requested values from an
  Ultralytics-compatible metric object;
- the paired arm must reject a silent stock fallback;
- result construction must calculate exact method-minus-baseline deltas;
- a real one-image CUDA smoke must pass before the 548-image run.

## Outputs

- `artifacts/acr-eg-ultralytics-native-final/evaluation.json`;
- `artifacts/acr-eg-ultralytics-native-final/checksums.sha256`;
- updated consolidated handoff with a clear distinction between SBR and
  Ultralytics native results.

