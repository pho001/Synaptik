# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0029: Stable [finite](../../glossary.md#term-finite) [softmax](../../glossary.md#softmax--log-softmax) without unshifted [EXP](../../glossary.md#term-expx) [overflow](../../glossary.md#term-overflow)


## Editorial reading context

Softmax converts a nonempty slice of finite scores into normalized weights; log-softmax publishes their logarithms without first rounding probabilities. Subtracting the maximum prevents unshifted exponential overflow, but is not a rewrite permission for non-finite inputs. Model defines the formula; the CPU backend must separately qualify its actual working precision and guards.

Use the [CPU finite-domain realization](../../backend-guide/cpu-backend.md#current-stable-softmax-family) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

### Mathematical reading aid: the finite-slice calculation

For a nonempty slice of $n$ finite real scores $x_i$, with $1\le i\le n$, define
$m=\max_{1\le j\le n}x_j$ and $s=\sum_{j=1}^{n}\exp(x_j-m)$. The stable
real-number formulas are

$$
p_i=\frac{\exp(x_i-m)}{s},\qquad \ell_i=(x_i-m)-\log s.
$$

Here $p_i$ is a dimensionless probability and $\ell_i$ its log-probability. For
$(x_1,x_2)=(1000,1000)$, both shifts are zero, $s=2$, and $p_1=p_2=1/2$;
$\ell_1=\ell_2=-\log2$. Actual subtraction, exponentiation, accumulation and
publication use the dtype/rounding sites stated in the decision, not exact reals.
In particular, finite extreme FLOAT32 scores can overflow a shift to negative infinity.
This aid does not extend the finite rule to NaN, infinities or an empty slice.

## Status

Accepted — 2026-10-13, following the [owner](../../glossary.md#term-owner) decision recorded in
[Model 0033](../../planning/modules/model/tasks/0033-stable-softmax-overflow-semantics.md)<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model))</span>.
This record explains the decision; the root and incorporated foundational contract are authoritative.

## Context

The [Model](../../glossary.md#term-model) composite row previously prescribed unshifted [`EXP`](../../glossary.md#term-expx), sum, and division/subtraction.
For [FLOAT32](../../glossary.md#term-float32) `[1000, 1000]`, that literal [evaluation](../../glossary.md#term-evaluation) overflows and [SOFTMAX](../../glossary.md#softmax--log-softmax) yields [NaN](../../glossary.md#term-nan), yet the
existing [CPU](../../glossary.md#term-cpu) route already subtracts the maximum and yields `[0.5, 0.5]`. The CPU tests used a
stable [oracle](../../glossary.md#term-test-oracle), but that did not authorize the difference from Model's literal result class.
The existing CPU emitter also evaluated FLOAT32 and [FLOAT16](../../glossary.md#term-float16) arithmetic in double, contrary to
the Model's declared working sites.

## Decision drivers

- Keep useful finite probabilities and log-probabilities without incidental unshifted EXP
  overflow or underflow.<span data-glossary-navigation="true"> ([log glossary](../../glossary.md#term-log); [EXP glossary](../../glossary.md#term-expx); [overflow glossary](../../glossary.md#term-overflow); [underflow glossary](../../glossary.md#term-underflow))</span>
- Preserve exact maximum selection, contributor membership, special-value classes, and empty
  behavior; avoid extending the change to other composites by analogy.
- Keep the existing [CPU](../../glossary.md#term-cpu) [admission](../../glossary.md#term-admission) of [finite](../../glossary.md#term-finite) [FLOAT32](../../glossary.md#term-float32) extremes while correcting working [precision](../../glossary.md#term-precision).
- Prevent old generated [softmax](../../glossary.md#softmax--log-softmax) [artifacts](../../glossary.md#term-artifact) from being reused with changed method bodies.

## Options considered

| Option | Advantage | Cost |
|---|---|---|
| Keep the literal unshifted formula | No [Model](../../glossary.md#term-model) text change | Turns ordinary [finite](../../glossary.md#term-finite) `[1000,1000]` into [NaN](../../glossary.md#term-nan); would regress the [CPU](../../glossary.md#term-cpu) stable route. |
| Declare one stable finite-[slice](../../glossary.md#slice) formula | Retains useful finite outputs and the existing CPU algorithm [shape](../../glossary.md#shape) | Requires explicit boundary for [infinity](../../glossary.md#term-infinity) slices and [FLOAT32](../../glossary.md#term-float32) working-site correction. |
| Allow either stable or literal finite output | Minimizes immediate implementation work | Leaves contradictory result classes, defeating one Model meaning. |

The options differ in observable result class, not merely in finite [tolerance](../../glossary.md#term-tolerance).

## Decision

For an all-[finite](../../glossary.md#term-finite) nonempty [slice](../../glossary.md#slice), select the exact represented maximum, subtract it at the
declared working [precision](../../glossary.md#term-precision), sum each shifted exponential contributor once, and publish a
probability or shift-minus-[log](../../glossary.md#term-log)-sum. Kahan-style [compensated summation](../../glossary.md#term-compensated-summation) may track [rounding](../../glossary.md#term-rounding)
residuals without adding a contributor. LOG_SOFTMAX is not computed by taking `log` of a rounded
[SOFTMAX](../../glossary.md#softmax--log-softmax) output. [FLOAT64](../../glossary.md#term-float64) works in FLOAT64; [FLOAT32](../../glossary.md#term-float32) and [homogeneous](../../glossary.md#term-homogeneous) low types work in FLOAT32,
with one final low [narrowing](../../glossary.md#term-narrowing).

The finite-slice rule does not rewrite non-finite slices. [NaN](../../glossary.md#term-nan), [positive infinity](../../glossary.md#nan-infinity-and-signed-zero), [negative
infinity](../../glossary.md#nan-infinity-and-signed-zero), and empty slices retain their prior [Model](../../glossary.md#term-model) results. In particular, a mixed
`[-infinity, 1000]` FLOAT32 slice retains the literal special-value path and may yield
`[+0, NaN]` for SOFTMAX. [CPU](../../glossary.md#term-cpu) continues to reject non-finite input before consumer writes.
Its pre-existing FLOAT64 and [BFLOAT16](../../glossary.md#term-bfloat16) finite-shift guards remain; FLOAT32 keeps admitting finite
extremes even when its now-correct FLOAT32 shift becomes negative infinity. [FLOAT16](../../glossary.md#term-float16)'s finite
inputs cannot [overflow](../../glossary.md#term-overflow) a FLOAT32 shift.

The CPU softmax algorithm [identity](../../glossary.md#term-identity-meanings) advances from 1 to 2, changing its structural [fingerprint](../../glossary.md#term-fingerprint),
generated class name, and [compatibility key](../../glossary.md#cache-and-memoization). Global CPU generator [schema](../../glossary.md#term-schema) 70 does not change
because this family-local append-only identity excludes old softmax [artifacts](../../glossary.md#term-artifact). Other generated
families retain their keys. The [Metal](../../glossary.md#term-metal) route catalog still marks SOFTMAX and LOG_SOFTMAX as
pending aggregate routes, with no executable first-class softmax body or compatible artifact to
invalidate; its identity and capability remain unchanged.

## Rationale

Maximum shifting preserves the exact real-valued [softmax](../../glossary.md#softmax--log-softmax) [identity](../../glossary.md#term-identity-meanings) for [finite](../../glossary.md#term-finite) inputs while
avoiding unrelated unshifted [EXP](../../glossary.md#term-expx) [overflow](../../glossary.md#term-overflow). A separate non-finite path keeps the earlier literal
result classes; it prevents a [numerical stability](../../glossary.md#term-numerical-stability) decision from silently changing special
semantics. The family-local identity is the smallest compatible [artifact](../../glossary.md#term-artifact) invalidation surface.

## Consequences

### Positive

`[1000, 1000]` has [finite](../../glossary.md#term-finite) normalized probabilities and finite [log](../../glossary.md#term-log)-probabilities. `[0, -1000]`
can have rounded probability zero and finite log-probability `-1000`. Generated and clean-Java
[FLOAT32](../../glossary.md#term-float32)/[FLOAT16](../../glossary.md#term-float16) sites now agree with the [Model](../../glossary.md#term-model) working [dtype](../../glossary.md#data-type--datatype).

### Negative and risks

Some previously double-evaluated [FLOAT32](../../glossary.md#term-float32)/[FLOAT16](../../glossary.md#term-float16) [CPU](../../glossary.md#term-cpu) results can change last bits. The
old generated/direct performance evidence is historical until a new matched-[shape](../../glossary.md#shape) comparison
is recorded. A [backend](../../glossary.md#backend) that supports non-[finite](../../glossary.md#term-finite) slices must distinguish the literal special
path from the stable finite path.

### Migration, testing, and follow-up

Old [softmax](../../glossary.md#softmax--log-softmax) class names and compatibility keys become incompatible misses; no migration reader
is added. Tests cover [overflow](../../glossary.md#term-overflow)/[underflow](../../glossary.md#term-underflow), offsets, admitted [FLOAT32](../../glossary.md#term-float32) extremes, lower-[precision](../../glossary.md#term-precision)
[narrowing](../../glossary.md#term-narrowing), value rejection before writes, and generated-versus-direct code [shape](../../glossary.md#shape). Attention's
guarded no-eligible and [infinity](../../glossary.md#term-infinity)-tie cases stay owned by attention. Categorical cross-entropy
already has its own max-shifted formula and remains unchanged; neither it nor LOG_SUM_EXP
inherits this exception.

## Related documentation

- [Root architecture invariants](../../../ARCHITECTURE.md#core-invariants)
- [Model composite numerical contract](../../architecture/contracts/foundational-modules.md#composite-inheritance)<span data-glossary-navigation="true"> ([Model glossary](../../glossary.md#term-model))</span>
- [Softmax API](../../api/tensor-api.md#softmax-expressions)<span data-glossary-navigation="true"> ([Softmax glossary](../../glossary.md#softmax--log-softmax); [API glossary](../../glossary.md#term-application-programming-interface-api))</span>
- [CPU softmax guide](../../backend-guide/cpu-backend.md#current-stable-softmax-family)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu))</span>
