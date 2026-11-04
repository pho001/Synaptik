# Stable [finite](../glossary.md#term-finite) [softmax](../glossary.md#softmax--log-softmax) and its boundaries

> [ARCHITECTURE.md](../../ARCHITECTURE.md) and its
> [foundational numerical contract](contracts/foundational-modules.md#composite-inheritance)
> are authoritative; this page explains the decision and its implementation boundary.

## Purpose and principle

[Softmax](../glossary.md#softmax--log-softmax) turns one [normalization](../glossary.md#term-normalization) [slice](../glossary.md#slice) into probabilities. For [finite](../glossary.md#term-finite) values, subtracting the
slice maximum before exponentiation avoids [overflow](../glossary.md#term-overflow) that has nothing to do with the final
probability. [Model](../glossary.md#term-model) owns that meaning. A [backend](../glossary.md#backend) owns whether it can execute a particular
[dtype](../glossary.md#data-type--datatype), [layout](../glossary.md#layout), and [runtime](../glossary.md#term-runtime) value domain.

A **normalization slice** is the group along the selected [axis](../glossary.md#term-axis) whose probabilities are computed
together. **Finite** excludes [infinity](../glossary.md#term-infinity) and [NaN](../glossary.md#term-nan) (“[not a number](../glossary.md#nan-infinity-and-signed-zero)”). **Overflow** exceeds the numeric
format's finite range; **[underflow](../glossary.md#term-underflow)** loses a very small result's representable magnitude.
“Working” [precision](../glossary.md#term-precision) is the format used during calculation, which can differ from the format
stored at the output. This numerical normalization is unrelated to [compiler](../glossary.md#term-compiler) normalization of
axes, identifiers or [graph](../glossary.md#graph) representation.

## [Mental model](../glossary.md#term-mental-model)

```text
finite slice -> exact maximum -> working shifts -> shifted EXP sum -> probability/log-probability
non-finite slice -> separate inherited special-value rules
CPU route -> finite input validation -> generated three-pass body -> final publication
```

Read the first line as the [Model](../glossary.md#term-model) [finite](../glossary.md#term-finite) algorithm, the second as a distinct Model branch, and
the third as [CPU](../glossary.md#term-cpu)'s narrower execution path. The working sites are [FLOAT64](../glossary.md#term-float64) for FLOAT64 inputs and
[FLOAT32](../glossary.md#term-float32) for FLOAT32 or either low storage type; a low output narrows once at [publication](../glossary.md#term-publication).

## Responsibilities and boundaries

[Model](../glossary.md#term-model) selects the exact represented maximum, the precise contributors and arithmetic sites,
and the [NaN](../glossary.md#term-nan), [infinity](../glossary.md#term-infinity), and empty results. A [compensated sum](../glossary.md#rounding-sites-fma-and-aggregate-floor) may correct roundoff but cannot
change the contributor set. [CPU](../glossary.md#term-cpu) validates its supported value domain after [producers](../glossary.md#term-producer) [run](../glossary.md#term-run) and
before any consumer write. Its generated body has no per-[element](../glossary.md#term-element) semantic [dispatch](../glossary.md#term-dispatch) or allocation.
Other composites do not gain [softmax](../glossary.md#softmax--log-softmax)'s [finite](../glossary.md#term-finite) shift permission by analogy.

A **compensated sum** keeps an extra correction for [rounding](../glossary.md#term-rounding) error; it is still summing the same
selected terms. It does not permit dropping terms or treating a non-finite [slice](../glossary.md#slice) as finite.

## Scenario that follows the boundary

For all-[finite](../glossary.md#term-finite) [FLOAT32](../glossary.md#term-float32) `[1000, 1000]`, both working shifts are zero. The shifted exponential
sum is two, so [SOFTMAX](../glossary.md#softmax--log-softmax) gives `[0.5, 0.5]` and LOG_SOFTMAX gives two values near `-ln(2)`.
For `[0, -1000]`, SOFTMAX may publish `[1, 0]` after [underflow](../glossary.md#term-underflow), while LOG_SOFTMAX publishes
`[0, -1000]`; the latter does not take [`log`](../glossary.md#term-log) of the rounded zero.

## Counter-scenario and failure mode

Replacing the all-[finite](../glossary.md#term-finite) rule with literal unshifted `exp(1000)` creates [infinity](../glossary.md#term-infinity) and then
`infinity/infinity = NaN`, changing a useful finite result class. Conversely, applying the
finite shift to a [slice](../glossary.md#slice) containing `-infinity` would silently change pre-existing special-value
classes. [FLOAT32](../glossary.md#term-float32) `[-infinity, 1000]` retains the literal special path, yielding `[+0, NaN]`
for [SOFTMAX](../glossary.md#softmax--log-softmax). [CPU](../glossary.md#term-cpu) rejects that input before output mutation instead of attempting the path.

## Implementation and testing consequences

[CPU](../glossary.md#term-cpu)'s [FLOAT32](../glossary.md#term-float32) and [FLOAT16](../glossary.md#term-float16) generated sites use FLOAT32 working locals, matching the clean-Java
[oracle](../glossary.md#term-test-oracle). Its family-local algorithm [identity](../glossary.md#term-identity-meanings) advances to version two, so previous [softmax](../glossary.md#softmax--log-softmax) class
[artifacts](../glossary.md#term-artifact) cannot be reused. [Backend](../glossary.md#backend) tests cover [finite](../glossary.md#term-finite) [overflow](../glossary.md#term-overflow)/[underflow](../glossary.md#term-underflow), working-site
boundaries, direct generated [shape](../glossary.md#shape), and CPU pre-write rejection. Attention's guarded rows and
[loss](../glossary.md#loss) formulas retain their own semantics and need no implementation change.

## Related decisions and documentation

- [ADR 0029](../design/decisions/0029-stable-finite-softmax.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span>
- [Softmax API reference](../reference/tensor/normalization.md#softmax-expressions)<span data-glossary-navigation="true"> ([Softmax glossary](../glossary.md#softmax--log-softmax); [API glossary](../glossary.md#term-application-programming-interface-api))</span>
- [CPU route guide](../backend-guide/cpu-backend.md#current-stable-softmax-family)<span data-glossary-navigation="true"> ([CPU glossary](../glossary.md#term-cpu))</span>
