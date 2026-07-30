# ACR-EG Live Checkpoint Evaluation Design

## Goal

Measure whether the final YAML-integrated ACR-EG checkpoint exceeds the sealed mature RT-DETR-L baseline on the same 548-image VisDrone validation set, using the frozen SBR metric semantics.

## Evidence boundary

The existing `scripts/evaluate_acr_eg_integrated.py` evaluates a frozen Query cache and cannot measure a live integrated detector checkpoint. The current paired `ACREGDetectionModel.predict()` path also assumes Ultralytics' training-mode raw tuple. In evaluation mode the RT-DETR decoder returns `(postprocessed, raw)`, so the paired path rejects the otherwise valid inference output. This is an evaluation-interface defect; the completed training path and checkpoint tensors must not change.

## Architecture

1. Extend the ACR-EG RT-DETR adapter to normalize both Ultralytics output contracts:
   - training: `(dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta)`;
   - evaluation: `(postprocessed, raw_tuple)`.
2. Preserve the existing training return exactly. In evaluation, inject the learned retention logits into the raw decoder class scores and invoke the stock RT-DETR head postprocessor to return the standard `[B,300,6]` tensor.
3. Add a dedicated live evaluator that:
   - verifies the final checkpoint identity, 48 `acr_eg.*` state entries and SHA256;
   - verifies the mature baseline SHA256;
   - constructs the exact no-augmentation global plus four local views;
   - fails if the ACR-EG output is not produced for every image;
   - converts normalized predictions to original-image half-open `xyxy` pixels;
   - evaluates both arms with `src.sbr_metrics.evaluate_dataset` using `conf=0.001`, `max_det=300`, ignored regions and the frozen size bins;
   - records mAP50-95, AP50, AP75, AP-tiny, tiny recall, AP-medium, AP-large, latency, peak VRAM and all deltas in canonical JSON.

## Scientific controls

- Dataset: exactly 548 VisDrone val images, signature `A9A0C00DC640BCAAEFE9360F5E3B55382E74E169B5AEEF15EB1F0AE2A571228A`.
- Baseline checkpoint SHA256: `54CE60289DD34C6750B8BA5F7516EEFCF3AFEF6C174C6E4F3B1EF810C883099B`.
- Method and baseline use the same source images, image order, 640 global input, confidence threshold and SBR evaluator.
- A stock single-view fallback is a hard error, not a warning.
- Old cache metrics remain diagnostic only and are not copied into the live result.

## Tests

- A failing unit test demonstrates that an evaluation-mode `(postprocessed, raw)` output is rejected by the current normalizer.
- A unit test proves evaluation decoding applies the fused score tensor while preserving stock boxes.
- A unit test proves training-mode output is unchanged.
- Evaluator unit tests cover checkpoint identity, image-coordinate conversion, paired-view enforcement, metric delta construction and canonical result schema.
- A real one-image CUDA smoke precedes the 548-image run.

## Outputs

- `artifacts/acr-eg-live-final/evaluation.json`
- `artifacts/acr-eg-live-final/predictions-baseline.jsonl.gz`
- `artifacts/acr-eg-live-final/predictions-acr-eg.jsonl.gz`
- `artifacts/acr-eg-live-final/checksums.sha256`
- an updated handoff section that distinguishes live results from cache diagnostics.
