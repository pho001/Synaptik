# Stable finite softmax and its boundaries

> [ARCHITECTURE.md](../../ARCHITECTURE.md) and its
> [foundational numerical contract](contracts/foundational-modules.md#composite-inheritance)
> are authoritative; this page explains the decision and its implementation boundary.

## Purpose and principle

Softmax turns one normalization slice into probabilities. For finite values, subtracting the
slice maximum before exponentiation avoids overflow that has nothing to do with the final
probability. Model owns that meaning. A backend owns whether it can execute a particular
dtype, layout, and runtime value domain.

## Mental model

```text
finite slice -> exact maximum -> working shifts -> shifted EXP sum -> probability/log-probability
non-finite slice -> separate inherited special-value rules
CPU route -> finite input validation -> generated three-pass body -> final publication
```

Read the first line as the Model finite algorithm, the second as a distinct Model branch, and
the third as CPU's narrower execution path. The working sites are FLOAT64 for FLOAT64 inputs and
FLOAT32 for FLOAT32 or either low storage type; a low output narrows once at publication.

## Responsibilities and boundaries

Model selects the exact represented maximum, the precise contributors and arithmetic sites,
and the NaN, infinity, and empty results. A compensated sum may correct roundoff but cannot
change the contributor set. CPU validates its supported value domain after producers run and
before any consumer write. Its generated body has no per-element semantic dispatch or allocation.
Other composites do not gain softmax's finite shift permission by analogy.

## Scenario that follows the boundary

For all-finite FLOAT32 `[1000, 1000]`, both working shifts are zero. The shifted exponential
sum is two, so SOFTMAX gives `[0.5, 0.5]` and LOG_SOFTMAX gives two values near `-ln(2)`.
For `[0, -1000]`, SOFTMAX may publish `[1, 0]` after underflow, while LOG_SOFTMAX publishes
`[0, -1000]`; the latter does not take `log` of the rounded zero.

## Counter-scenario and failure mode

Replacing the all-finite rule with literal unshifted `exp(1000)` creates infinity and then
`infinity/infinity = NaN`, changing a useful finite result class. Conversely, applying the
finite shift to a slice containing `-infinity` would silently change pre-existing special-value
classes. FLOAT32 `[-infinity, 1000]` retains the literal special path, yielding `[+0, NaN]`
for SOFTMAX. CPU rejects that input before output mutation instead of attempting the path.

## Implementation and testing consequences

CPU's FLOAT32 and FLOAT16 generated sites use FLOAT32 working locals, matching the clean-Java
oracle. Its family-local algorithm identity advances to version two, so previous softmax class
artifacts cannot be reused. Backend tests cover finite overflow/underflow, working-site
boundaries, direct generated shape, and CPU pre-write rejection. Attention's guarded rows and
loss formulas retain their own semantics and need no implementation change.

## Related decisions and documentation

- [ADR 0029](../design/decisions/0029-stable-finite-softmax.md)
- [Softmax API reference](../api/tensor-api.md#softmax-expressions)
- [CPU route guide](../backend-guide/cpu-backend.md#current-stable-softmax-family)
