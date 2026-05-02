# Phase 37 Research: Loss-Adjacent Metal Lowering

## Current State

- `NLL_LOSS`, `CROSS_ENTROPY_LOSS`, `CROSS_ENTROPY_LOSS_INDICES`, and `CROSS_ENTROPY_LOSS_INDICES_GRAD` are explicit loss-adjacent matrix rows.
- Metal currently reports dense `NLL_LOSS` / `CROSS_ENTROPY_LOSS` as unsupported and index-target CE/NLL as `UNSUPPORTED_INDEX_SEMANTICS`.
- Forward `LOG_SOFTMAX`, reductions, elementwise ops, `GATHER`, and `TAKE_ALONG_AXIS` already have scoped Metal coverage.
- Phase 36 verified that `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` remain stable duplicate-index blockers.
- CPU loss semantics live in `TensorLossOps`, `LossSupport`, and CPU reduction kernels. Index-target CE has descriptors carrying class dimension, reduction, and optional ignore index.

## Implementation Implications

1. Dense loss lowering is the safest first native claim.
   - Dense `CROSS_ENTROPY_LOSS` can lower to `LOG_SOFTMAX` plus target multiplication and class reduction if the exact CPU formula and output shape are matched.
   - Dense `NLL_LOSS` can lower to negated target-weighted log-prob reduction if dense target semantics are locked.
   - The phase should start with `FLOAT32`, dense layout, supported rank/axis, and the current public reduction semantics.

2. Index-target forward support may be possible, but it must be scoped.
   - Forward `CROSS_ENTROPY_LOSS_INDICES` can reuse `LOG_SOFTMAX` plus `TAKE_ALONG_AXIS` for no-ignore/no-class-weight variants.
   - Ignore-index and class weights affect masking and denominator semantics, so they need separate gates.
   - Backward index-target support depends on scatter-style gradient construction and must not be claimed because Phase 36 keeps index-gradient blockers.

3. Reports need separate truth rows and target names.
   - Dense loss support must not mark index-target CE/NLL as native.
   - Index-target unsupported variants should surface `UNSUPPORTED_INDEX_SEMANTICS`, `UNSUPPORTED_IGNORE_INDEX`, `UNSUPPORTED_DUPLICATE_INDEX`, or bounds-specific details.

4. Training coverage should focus on boundary reduction evidence.
   - If forward dense loss stays on Metal but backward exits at an unsupported gradient primitive, that is a partial improvement and must be reported as such.
   - Avoid counting graph output or gradient publication as hidden internal CPU materialization.

## Recommended Wave Shape

- 37-01: lock dense loss formulas, shape/reduction gates, and planner truth.
- 37-02: implement scoped dense Metal lowering/execution and decide index-target forward admission versus stable rejection.
- 37-03: add backward/training integration gates and explicit unsupported index-gradient boundaries.
- 37-04: close coverage targets, reports, docs, and verification.

## Pitfalls

- Do not lower index-target CE to `gather_take_small` coverage and then claim index-gradient support.
- Do not ignore denominator changes for `MEAN` with ignore-index or class weights.
- Do not make `CROSS_ENTROPY_LOSS_INDICES_GRAD` native while `SCATTER_ADD` remains unsupported.
- Do not add broad matrix `SUPPORTED` rows before native execution and parity evidence exist.
