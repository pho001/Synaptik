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
CPU route -> bind checked storage -> generated finite/special-value branches -> publication
```

Read the first line as Model's finite algorithm, the second as a distinct Model branch, and
the third as CPU's current realization of both. FLOAT64 inputs use FLOAT64 working sites;
FLOAT32 and all four low storage formats use FLOAT32 work, with one final low narrowing.
The [current CPU family](../backend-guide/cpu-backend.md#current-stable-softmax-family)
has superseded the historical finite-input-only admission.

## Responsibilities and boundaries

Model selects the represented maximum, contributors, arithmetic sites, and NaN/infinity/empty
results. A compensated sum may correct roundoff but cannot change the contributor set.
CPU binds checked input/output storage before generated execution; the generated body handles
the current finite and non-finite branches without a separate finite-input validator.
It has no per-element semantic dispatch or allocation. Other composites do not gain softmax's
finite-shift permission by analogy.

A **compensated sum** keeps an extra correction for [rounding](../glossary.md#term-rounding) error; it is still summing the same
selected terms. It does not permit dropping terms or treating a non-finite [slice](../glossary.md#slice) as finite.

### What the finite formula calculates

Take a nonempty normalization slice of $n>0$ finite real inputs $x_1,\ldots,x_n$.
Let $m=\max_{1\le j\le n}x_j$ be its maximum and
$s=\sum_{j=1}^{n}\exp(x_j-m)$ its shifted exponential sum. For each index
$i\in\{1,\ldots,n\}$, the real-number softmax probability $p_i$ and log-probability $\ell_i$ are

$$
p_i=\frac{\exp(x_i-m)}{s},\qquad
\ell_i=(x_i-m)-\log s.
$$

Every shifted input is non-positive and at least one is zero, so $s\ge1$ in real arithmetic:
the denominator cannot be zero. For $x=(1000,1000)$, $m=1000$, $s=2$, $p=(1/2,1/2)$,
and both log-probabilities are $-\log2$. Subtracting the same maximum preserves relative
probabilities without first calculating the overflowing unshifted exponentials.

Actual execution rounds at the working sites described above. Tiny exponentials can underflow,
and finite subtraction itself can overflow for widely separated represented extremes; the
formula is not a promise of exact real-number output. It does not define the NaN, infinity or
empty-slice branches. Those remain separate inherited rules; an empty shape publishes no
elements, while a nonempty special-value slice follows its own result-class branch.


## Scenario that follows the boundary

For finite FLOAT32 $(1000,1000)$, both shifts are zero and the shifted exponential sum is two.
SOFTMAX therefore gives $(0.5,0.5)$; both LOG_SOFTMAX values are near $-\log2$.
For $(0,-1000)$, softmax may publish $(1,0)$ after underflow, while log-softmax publishes
$(0,-1000)$: it does not compute a logarithm of the rounded zero.

## Counter-scenario and failure mode

Replacing the finite rule with literal unshifted $\exp(1000)$ overflows in FLOAT32,
and dividing infinity by infinity yields NaN instead of the useful finite probabilities.
Conversely, a slice containing negative infinity but neither NaN nor positive infinity retains
the literal unshifted special path. FLOAT32 `[-infinity, 1000]` therefore yields `[+0, NaN]`
for SOFTMAX. The finite-input CPU checkpoint rejected that input before mutation; the current
generated route implements its Model result classes instead. This is a historical admission
change, not permission to shift every special-value slice.

## Implementation and testing consequences

The stable-finite implementation originally advanced the family-local algorithm identity to
version two and qualified finite overflow/underflow and working-site boundaries against clean
Java. That checkpoint's finite-only rejection is historical, not today's value domain.
Current [generated CPU code](../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuSoftmaxEmitter.java)
and the [reference algorithm](../../modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuSoftmaxReferenceKernel.java)
retain separate NaN, positive-infinity, negative-infinity and empty branches.
Their presence is source evidence here, not a claim that runtime checks were performed for this
editorial revision. Attention and losses retain their own formulas.

## Related decisions and documentation

- [ADR 0029](../design/decisions/0029-stable-finite-softmax.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span>
- [Softmax API reference](../reference/tensor/normalization.md#softmax-expressions)<span data-glossary-navigation="true"> ([Softmax glossary](../glossary.md#softmax--log-softmax); [API glossary](../glossary.md#term-application-programming-interface-api))</span>
- [CPU route guide](../backend-guide/cpu-backend.md#current-stable-softmax-family)<span data-glossary-navigation="true"> ([CPU glossary](../glossary.md#term-cpu))</span>
