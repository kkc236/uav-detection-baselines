# Quality Oracle Non-Gate Floating-Point Amendment

Date: 2026-08-04

The first immutable native-auxiliary oracle run reproduced the frozen stock authority
exactly for `map`, `ap50`, `ap75`, `ap_tiny`, `ap_small`, and `recall`. Its `precision`
was `0.5119369292841953` instead of the adapter-path authority value
`0.5119369275291381`, an absolute difference of approximately `1.76e-9`.

The quality-oracle design intentionally replaced the former `FrozenIBERAdapter` forward
path with the unmodified Ultralytics auxiliary decoder tuple. Re-evaluating the cached
official-validation records with both repository metric implementations produced the
same native-path value, while every AP metric remained byte-exact.

This amendment does not change the scientific Gate. `map`, `ap50`, `ap75`, `ap_tiny`,
and `ap_small` remain exact-authority checks. Only the non-Gate diagnostic metrics
`precision` and `recall` receive an absolute tolerance of `1e-8`. The observed deltas,
tolerance, and amendment status must be written into the immutable official report.
Any AP mismatch or diagnostic mismatch above `1e-8` remains an engineering failure.

The already-produced official-validation cache is reused; official-validation inference
must not be repeated.
