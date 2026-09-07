# AI4I Condition-Based Maintenance Risk Triage

- Release: `ai4i-risk-v3.0.0-20260907162125-9b01f38`
- Model: `hist_gradient_boosting` + sigmoid cross-fitted calibration
- Risk meaning: risk associated with the current operating snapshot.
- No future horizon, RUL, or failure time is implied.
- Failure-mode flags are target-derived diagnostic labels used only for offline slice analysis.
- RNF is an out-of-model limitation because the available sensor contract has no reliable precursor.
- Tool-wear failures require explicit monitoring; overall PR-AUC must not hide this slice.
