# ADR 0029: Stable finite softmax without unshifted EXP overflow

## Status

Accepted — 2026-10-13, following the owner decision recorded in
[Model 0033](../../planning/modules/model/tasks/0033-stable-softmax-overflow-semantics.md).
This record explains the decision; the root and incorporated foundational contract are authoritative.

## Context

The Model composite row previously prescribed unshifted `EXP`, sum, and division/subtraction.
For FLOAT32 `[1000, 1000]`, that literal evaluation overflows and SOFTMAX yields NaN, yet the
existing CPU route already subtracts the maximum and yields `[0.5, 0.5]`. The CPU tests used a
stable oracle, but that did not authorize the difference from Model's literal result class.
The existing CPU emitter also evaluated FLOAT32 and FLOAT16 arithmetic in double, contrary to
the Model's declared working sites.

## Decision drivers

- Keep useful finite probabilities and log-probabilities without incidental unshifted EXP
  overflow or underflow.
- Preserve exact maximum selection, contributor membership, special-value classes, and empty
  behavior; avoid extending the change to other composites by analogy.
- Keep the existing CPU admission of finite FLOAT32 extremes while correcting working precision.
- Prevent old generated softmax artifacts from being reused with changed method bodies.

## Options considered

| Option | Advantage | Cost |
|---|---|---|
| Keep the literal unshifted formula | No Model text change | Turns ordinary finite `[1000,1000]` into NaN; would regress the CPU stable route. |
| Declare one stable finite-slice formula | Retains useful finite outputs and the existing CPU algorithm shape | Requires explicit boundary for infinity slices and FLOAT32 working-site correction. |
| Allow either stable or literal finite output | Minimizes immediate implementation work | Leaves contradictory result classes, defeating one Model meaning. |

The options differ in observable result class, not merely in finite tolerance.

## Decision

For an all-finite nonempty slice, select the exact represented maximum, subtract it at the
declared working precision, sum each shifted exponential contributor once, and publish a
probability or shift-minus-log-sum. Kahan-style compensated summation may track rounding
residuals without adding a contributor. LOG_SOFTMAX is not computed by taking `log` of a rounded
SOFTMAX output. FLOAT64 works in FLOAT64; FLOAT32 and homogeneous low types work in FLOAT32,
with one final low narrowing.

The finite-slice rule does not rewrite non-finite slices. NaN, positive infinity, negative
infinity, and empty slices retain their prior Model results. In particular, a mixed
`[-infinity, 1000]` FLOAT32 slice retains the literal special-value path and may yield
`[+0, NaN]` for SOFTMAX. CPU continues to reject non-finite input before consumer writes.
Its pre-existing FLOAT64 and BFLOAT16 finite-shift guards remain; FLOAT32 keeps admitting finite
extremes even when its now-correct FLOAT32 shift becomes negative infinity. FLOAT16's finite
inputs cannot overflow a FLOAT32 shift.

The CPU softmax algorithm identity advances from 1 to 2, changing its structural fingerprint,
generated class name, and compatibility key. Global CPU generator schema 70 does not change
because this family-local append-only identity excludes old softmax artifacts. Other generated
families retain their keys. The Metal route catalog still marks SOFTMAX and LOG_SOFTMAX as
pending aggregate routes, with no executable first-class softmax body or compatible artifact to
invalidate; its identity and capability remain unchanged.

## Rationale

Maximum shifting preserves the exact real-valued softmax identity for finite inputs while
avoiding unrelated unshifted EXP overflow. A separate non-finite path keeps the earlier literal
result classes; it prevents a numerical stability decision from silently changing special
semantics. The family-local identity is the smallest compatible artifact invalidation surface.

## Consequences

### Positive

`[1000, 1000]` has finite normalized probabilities and finite log-probabilities. `[0, -1000]`
can have rounded probability zero and finite log-probability `-1000`. Generated and clean-Java
FLOAT32/FLOAT16 sites now agree with the Model working dtype.

### Negative and risks

Some previously double-evaluated FLOAT32/FLOAT16 CPU results can change last bits. The
old generated/direct performance evidence is historical until a new matched-shape comparison
is recorded. A backend that supports non-finite slices must distinguish the literal special
path from the stable finite path.

### Migration, testing, and follow-up

Old softmax class names and compatibility keys become incompatible misses; no migration reader
is added. Tests cover overflow/underflow, offsets, admitted FLOAT32 extremes, lower-precision
narrowing, value rejection before writes, and generated-versus-direct code shape. Attention's
guarded no-eligible and infinity-tie cases stay owned by attention. Categorical cross-entropy
already has its own max-shifted formula and remains unchanged; neither it nor LOG_SUM_EXP
inherits this exception.

## Related documentation

- [Root architecture invariants](../../../ARCHITECTURE.md#core-invariants)
- [Model composite numerical contract](../../architecture/contracts/foundational-modules.md#composite-inheritance)
- [Softmax API](../../api/tensor-api.md#softmax-expressions)
- [CPU softmax guide](../../backend-guide/cpu-backend.md#current-stable-softmax-family)
