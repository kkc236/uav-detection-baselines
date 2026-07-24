# SBR Dual-Channel Scale Router Evidence Snapshot

This directory contains the compact, public-safe evidence snapshot used to
freeze the one-shot dual-channel cache replay.

`feasibility_metrics.json` records:

- the sealed Arm A and Arm C metrics;
- the completed coordinate-only guard result;
- the completed score-oracle result;
- the original five gate thresholds;
- the AP75 large-loss attribution counts; and
- the frozen dual-channel replay status.

The full VisDrone data, model weights, raw prediction cache, server paths, and
runtime manifests are intentionally excluded. The exact algorithm is specified
in:

`docs/superpowers/specs/2026-07-24-sbr-dual-channel-scale-router-design.md`

The dual-channel result is explicitly marked `FROZEN_NOT_RUN`. Existing
negative results must not be presented as results of the new route.
