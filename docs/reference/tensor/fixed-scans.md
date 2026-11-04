# Cumulative and fixed recurrent scans

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## Which API describes the repeated work?

Application or layer-building code calls `input.cumSum(axis)` for every prefix total, or the
static `RecurrentScan` methods for a fixed neural state update. Both receive Tensor descriptions,
retain exact input references, and return storage-free result descriptions. They do not read
length values or run the recurrence during construction. For a user-described step body rather
than a fixed cell, use [general Scan](general-scan.md#formals-actual-positions-and-source-closure).

For finite real values $x_0,\ldots,x_{D-1}$ along an axis of integer extent $D\geq0$,
forward inclusive sum describes $y_i=\sum_{j=0}^{i}x_j$ for $0\leq i<D$;
exclusive sum uses $\sum_{j=0}^{i-1}x_j$ with empty sum zero.
Thus `[1, 2, 3]` gives final inclusive value $1+2+3=6$, while the first exclusive
value is zero because no prior position exists. Floating accumulation is rounded under the
rules below, not unrestricted real reassociation; an empty axis has no output position.

**[Reader model](../../glossary.md#term-reader-model).** A **[scan](../../glossary.md#term-cumulative-scan)** visits an ordered [axis](../../glossary.md#term-axis) while carrying information from earlier
visited positions. A cumulative sum returns every partial total, unlike a reduction that returns
only the final total. **Inclusive** includes the current value; **exclusive** reports the total
before it. A **recurrence** updates a state at each step.
[Recurrent neural network](../../glossary.md#term-recurrent-neural-network-rnn) (**[RNN](../../glossary.md#neural-network-families)**) cells, gated recurrent units (**GRUs**), and [long short-term
memory](../../glossary.md#term-long-short-term-memory-lstm) (**[LSTM](../../glossary.md#neural-network-families)**) cells use different fixed state-update formulas. Their **hidden state** is the
remembered numeric [vector](../../glossary.md#term-vector); an LSTM additionally has a cell state. A **gate** controls how much
information passes through an update. **[Time-major](../../glossary.md#term-time-major)** places time on the first axis.
The valid-length vector says how many steps belong to each [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) row, not how much storage
the rectangular input has.

<a id="cumulative-scan-expressions"></a>
### Cumulative-[scan](../../glossary.md#term-cumulative-scan) expressions

The four current cumulative-scan methods preserve logical [Shape](../../glossary.md#shape) while accumulating along exactly
one input [axis](../../glossary.md#term-axis):

| Public form | Arithmetic | Inclusion | Direction |
|---|---|---|---|
| `cumSum(axis)` | Addition | Inclusive | Forward |
| `cumSum(axis, exclusive, reverse)` | Addition | Selected by `exclusive` | Selected by `reverse` |
| `cumProd(axis)` | Multiplication | Inclusive | Forward |
| `cumProd(axis, exclusive, reverse)` | Multiplication | Selected by `exclusive` | Selected by `reverse` |

An axis identifies the [dimension](../../glossary.md#dimension) whose positions contribute to each cumulative prefix. Forward
traversal visits that axis from lower to higher indices; reverse traversal visits it from higher
to lower indices without reversing the returned dimension or output positions. Inclusive mode
includes the current value. Exclusive mode omits it, so the first position visited has the
[operation](../../glossary.md#operation) [identity](../../glossary.md#term-identity-meanings): additive zero for `CUM_SUM` or multiplicative positive one for `CUM_PROD`.

For logical sum input `[1, 2, 3]`, the four modes retain their existing meanings:

| Call mode | Semantic result | Position-by-position interpretation |
|---|---|---|
| `cumSum(axis)` or `(false, false)` | `[1, 3, 6]` | Forward prefixes are `1`, `1 + 2`, and `1 + 2 + 3`. |
| `(true, false)` | `[0, 1, 3]` | Excluding each current value leaves the empty prefix, `1`, and `1 + 2`. |
| `(false, true)` | `[6, 5, 3]` | Reverse-inclusive contributions are `1 + 2 + 3`, `2 + 3`, and `3`. |
| `(true, true)` | `[5, 3, 0]` | Reverse-exclusive contributions are `2 + 3`, `3`, and the empty reverse prefix. |

For logical product input `[2, 3, 4]`, the corresponding four modes are:

| Call mode | Semantic result | Position-by-position interpretation |
|---|---|---|
| `cumProd(axis)` or `(false, false)` | `[2, 6, 24]` | Forward prefixes are `2`, `2 * 3`, and `2 * 3 * 4`. |
| `(true, false)` | `[1, 2, 6]` | Excluding each current value leaves positive one, `2`, and `2 * 3`. |
| `(false, true)` | `[24, 12, 4]` | Reverse-inclusive contributions are `2 * 3 * 4`, `3 * 4`, and `4`. |
| `(true, true)` | `[12, 4, 1]` | Reverse-exclusive contributions are `3 * 4`, `4`, and positive one. |

These values explain the requested operations. Current expression construction computes none of
them: it reads no [element](../../glossary.md#term-element) or [host storage](../../glossary.md#host-storage) and allocates no result storage. A zero-length selected
axis produces a zero-length result with no position at which to emit an identity. On a non-empty
exclusive scan, the identity appears at the first position in the selected traversal direction.

All four methods accept floating and signed-integral inputs, including [FP8](../../glossary.md#term-fp8);
[`BOOL`](../../glossary.md#data-type--datatype) is rejected without truthiness or conversion. A positive or negative caller axis is normalized
exactly once against the input `Shape`. Every valid result is a fresh unlabeled, storage-free
[Tensor](../../glossary.md#tensor) whose [descriptor](../../glossary.md#term-descriptor) retains the exact input Shape reference, [data type](../../glossary.md#data-type--datatype), and `requiresGrad`
eligibility, but uses unresolved [layout](../../glossary.md#layout) even when the input layout is resolved. Integral inputs
necessarily remain ineligible for [gradients](../../glossary.md#term-gradient) under the descriptor contract.

Integral cumulative product retains exact [INT32](../../glossary.md#data-type--datatype) or [INT64](../../glossary.md#data-type--datatype) type and means fixed-width two's-
complement modular multiplication. Floating cumulative product propagates [NaN](../../glossary.md#term-nan), makes zero times
[infinity](../../glossary.md#term-infinity) NaN, and determines zero and infinity signs by multiplication parity. Positive one is the
empty product used by a non-empty exclusive boundary. Arithmetic follows the declared
primitive/aggregate floors: [FLOAT32](../../glossary.md#term-float32) sites round to FLOAT32, low arithmetic works/accumulates
in FLOAT32 with final declared-result [narrowing](../../glossary.md#term-narrowing), and [FLOAT64](../../glossary.md#term-float64) retains its own format.
The constructor chooses no algorithm/[backend route](../../glossary.md#backend-route) and promises no [NaN payload](../../glossary.md#nan-infinity-and-signed-zero) or cross-[backend](../../glossary.md#backend) bits.

[Provenance](../../glossary.md#provenance) records the matching `CumulativeScanKind.CUM_SUM` or `CUM_PROD`, one
`CumulativeScanAttrs(normalizedAxis, exclusive, reverse)`, and exact ordered inputs `[input]`.
Repeated equivalent requests remain distinct expressions with fresh identities. Construction
defines no [model](../../glossary.md#term-model)-owned gradient rule, [compiler](../../glossary.md#term-compiler) [capture](../../glossary.md#term-capture-meanings) or [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), backend [lowering](../../glossary.md#lowering),
[runtime](../../glossary.md#term-runtime) behavior, or execution. Current package-private compiler [autograd](../../glossary.md#autograd) supports floating
`CUM_SUM` for all four exclusive/reverse combinations by retaining exclusivity and reversing the
scan direction. Floating `CUM_PROD` differentiation is also implemented by
[ReductionGradientRules](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionGradientRules.java):
zero-safe products and zero-count prefix scans separate zero and nonzero input contributions,
then reverse cumulative sums construct the [cotangent](../../glossary.md#cotangent). Integral scan roles remain non-[differentiable](../../glossary.md#derivative-and-chain-rule).

Failure behavior is local and deterministic:

- `BOOL` fails with `IllegalArgumentException` and message
  `input must have a numeric data type, but was BOOL` before axis [validation](../../glossary.md#term-validation).
- An axis outside `[-rank, rank - 1]` fails with `IndexOutOfBoundsException`; every scalar axis is
  invalid. These validation failures consume no Tensor identity.<span data-glossary-navigation="true"> ([scalar glossary](../../glossary.md#term-scalar))</span>
- Exhausted factory identity space fails only after descriptor, operation, and provenance
  metadata have been constructed.<span data-glossary-navigation="true"> ([metadata glossary](../../glossary.md#term-metadata))</span>

<a id="fixed-recurrent-scan-expressions"></a>
### Fixed recurrent-[scan](../../glossary.md#term-cumulative-scan) expressions

The public final field-free `model.tensor.RecurrentScan` type is an advanced low-level static
[Tensor](../../glossary.md#tensor)-expression namespace. Its exactly six methods construct one fixed [time-major](../../glossary.md#term-time-major) scan
[occurrence](../../glossary.md#term-occurrence); they are not [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) layers or [modules](../../glossary.md#term-module), an execution service, a [registry](../../glossary.md#term-registry), or a general
user-defined scan-body [API](../../glossary.md#term-application-programming-interface-api). `Tensor` has no recurrent receiver method.

| Family | [Bias](../../glossary.md#term-bias)-free static form | Biased static form | Ordered outputs |
|---|---|---|---|
| [RNN](../../glossary.md#term-recurrent-neural-network-rnn) [tanh](../../glossary.md#term-tanh) | `RecurrentScan.rnn(input, validLengths, initialHidden, inputWeight, hiddenWeight, direction)` | `RecurrentScan.rnn(input, validLengths, initialHidden, inputWeight, hiddenWeight, bias, direction)` | `[outputs, finalHidden]` |
| Reset-after [GRU](../../glossary.md#term-gated-recurrent-unit-gru) | `RecurrentScan.gru(input, validLengths, initialHidden, inputWeight, hiddenWeight, direction)` | `RecurrentScan.gru(input, validLengths, initialHidden, inputWeight, hiddenWeight, bias, direction)` | `[outputs, finalHidden]` |
| [LSTM](../../glossary.md#term-long-short-term-memory-lstm) | `RecurrentScan.lstm(input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight, direction)` | `RecurrentScan.lstm(input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight, bias, direction)` | `[outputs, finalHidden, finalCell]` |

The explicit first [argument](../../glossary.md#term-argument) is `input [time, batch, inputSize]`. Every [Shape](../../glossary.md#shape) is fully static; `inputSize` and
`hiddenSize` must be positive, while `time` and [`batch`](../../glossary.md#term-batch-machine-learning-and-hyperparameters) may be zero. `validLengths` is exactly
non-[gradient](../../glossary.md#term-gradient) `INT64[batch]`. Initial hidden and cell states are `[batch, hiddenSize]`. For gate
count `G`, input [weight](../../glossary.md#term-weight) is `[G * hiddenSize, inputSize]`, hidden weight is
`[G * hiddenSize, hiddenSize]`, and optional bias is `[G * hiddenSize]`; `G` is one, three, or four
for RNN, GRU, or LSTM respectively. The input, states, weights, and optional bias use one exact
common floating [data type](../../glossary.md#data-type--datatype), including either [FP8](../../glossary.md#term-fp8) type.

The fixed transition equations match the current NN cells. With `X = x @ transpose(inputWeight)`
plus the optional input-side bias and `H = h @ transpose(hiddenWeight)`:

The packed gate axis is ordered, not interchangeable. Consecutive blocks of `hiddenSize`
rows in both weight matrices—and `hiddenSize` elements in the optional bias—use:

- GRU: reset $r$, update $z$, candidate $n$, in that order.
- LSTM: input $i$, forget $f$, candidate $g$, output $o$ (**IFGO**), in that order.

Swapping blocks preserves the shapes but changes the recurrence. These are the orders
defined by [`RecurrentScanKind`](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/operation/recurrent/RecurrentScanKind.java).

For each finite real row/hidden coordinate, let $X$ and $H$ be the input and previous-hidden
projections defined above, and $h$ the previous hidden value. Subscripts select blocks of
the packed gate axis, not time positions. All products below are elementwise:

$$
h_{\mathrm{next}}=\tanh(X+H)\qquad\text{(RNN)}.
$$

For reset-after GRU, $r$ is the reset gate, $z$ the update gate and $n$ the candidate:

$$
r=\sigma(X_r+H_r),\quad z=\sigma(X_z+H_z),\quad
n=\tanh(X_n+rH_n),\quad h_{\mathrm{next}}=n+z(h-n).
$$

For LSTM, $i,f,o$ are input/forget/output gates, $g$ the candidate, and $c$ the previous cell:

$$
\begin{aligned}
i&=\sigma(X_i+H_i), & f&=\sigma(X_f+H_f),\\
g&=\tanh(X_g+H_g), & o&=\sigma(X_o+H_o),\\
c_{\mathrm{next}}&=fc+ig, &
h_{\mathrm{next}}&=o\tanh(c_{\mathrm{next}}).
\end{aligned}
$$

Here $\sigma$ denotes sigmoid. If $X=H=0$, the ideal RNN next hidden value is zero;
for an LSTM with $c=0$ and zero projections, $i=f=o=1/2$, $g=0$, so both next states
are zero. These real formulas explain gates, not an IEEE rewrite rule: working precision and
stored-state narrowing below still determine what a later state consumer reads.

There is no hidden-side bias, configurable [activation](../../glossary.md#term-activation), gate order, peephole, projection,
recurrent [dropout](../../glossary.md#dropout--inverted-dropout), arbitrary cell body, [callback](../../glossary.md#term-callback), nested [graph](../../glossary.md#graph), or region. `RecurrentScanKind`
has closed [signatures](../../glossary.md#term-signature): RNN and GRU accept five or six inputs and produce exactly two outputs;
LSTM accepts six or seven inputs and produces exactly three. The exact attribute class is
`RecurrentDirection`, whose only values are `FORWARD` and `REVERSE`.

[Provenance](../../glossary.md#provenance) retains the following exact input order:

```text
RNN/GRU without bias: [input, validLengths, initialHidden, inputWeight, hiddenWeight]
RNN/GRU with bias:    [input, validLengths, initialHidden, inputWeight, hiddenWeight, bias]
LSTM without bias:   [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight]
LSTM with bias:      [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight, bias]
```

One call creates one fresh [identity](../../glossary.md#term-identity-meanings)-distinct flat [`TensorProducer`](../../glossary.md#producer--tensorproducer). `RecurrentScanResult` exposes
the [producer](../../glossary.md#term-producer)'s [canonical](../../glossary.md#term-canonical-form) wrappers at positions zero and one; `LstmRecurrentScanResult` exposes
positions zero through two. The dense output [descriptor](../../glossary.md#term-descriptor) is `[time, batch, hiddenSize]`; final
state descriptors are `[batch, hiddenSize]`. Their [layouts](../../glossary.md#layout) are unresolved, they are unlabeled and
storage-free, and their `requiresGrad` value is the OR of the floating input, state, weight, and
optional-bias roles. Valid lengths never contribute to gradient eligibility. No projection,
gate, [mask](../../glossary.md#term-boolean-mask), step, [padding](../../glossary.md#padding), or zero-fill intermediate Tensor is constructed.

For original batch row `b` with [runtime](../../glossary.md#term-runtime) length `L[b]`, `FORWARD` consumes [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix)
`0 .. L[b]-1`; `REVERSE` consumes `L[b]-1 .. 0`, reversing only the valid prefix. Both directions return
dense outputs aligned to original time positions, with exact [positive zero](../../glossary.md#nan-infinity-and-signed-zero) at `t >= L[b]`, plus
explicit final states. A zero-length row returns its initial states semantically. A zero-time
input has an empty dense output and requires every runtime length to be zero. These are fixed
[operation](../../glossary.md#operation) semantics, not values computed by [Model](../../glossary.md#term-model) construction.

Construction never reads `validLengths` storage, so it cannot validate [scalar](../../glossary.md#term-scalar) bounds or infer
active rows. It validates nulls in declaration order, then direction, floating input and static
[rank](../../glossary.md#term-rank)-three Shape, positive input width, the valid-length descriptor, state descriptors, packed
weight extents, optional bias, and exact common data type before allocating output identities.
Descriptor or arithmetic-[overflow](../../glossary.md#term-overflow) failures therefore consume no Tensor ID. Identifier exhaustion
may consume earlier output IDs without returning a partial result, following the shared canonical
multi-output factory contract.

Current generic [graph capture](../../glossary.md#capture-graph-ports-and-reconstruction) can translate the shared producer to one flat [node](../../glossary.md#node) with the same
ordered inputs and outputs. Current package-private [Compiler](../../glossary.md#term-compiler) forward-only [inference](../../glossary.md#term-inference) independently
revalidates the fully static descriptor contract, while both backward-capable modes reject a
complete forward inventory containing recurrence before [derivative](../../glossary.md#term-derivative) Tensor allocation.

[CPU](../../glossary.md#term-cpu) scalar execution currently supports all six signatures and both directions for [homogeneous](../../glossary.md#term-homogeneous)
floating types, including all four low types, with canonical dense outputs. Every input role, including [INT64](../../glossary.md#data-type--datatype)
lengths, accepts resolved nonnegative-[stride](../../glossary.md#term-stride)/offset reads: dense, transposed, sliced/gapped,
overlapping and zero-stride repeated layouts. Non-[singleton](../../glossary.md#term-singleton) repetition requires Model's [view](../../glossary.md#term-view-meanings) flag.
Negative and unresolved storage geometry is unsupported. Every [dimension](../../glossary.md#dimension), packed gate [extent](../../glossary.md#term-extent),
logical count and referenced storage span must fit `int`; fresh [Engine](../../glossary.md#term-engine) [bindings](../../glossary.md#term-binding) must retain the
full [referenced span](../../glossary.md#referenced-element-span), not merely the logical count. Empty layouts have span zero regardless of
unused offset.

CPU low fixed recurrence includes [BFLOAT16](../../glossary.md#term-bfloat16), [FLOAT16](../../glossary.md#term-float16) and both [OFP8](../../glossary.md#term-ofp8) types under the same
layout, geometry and actual derivative guards; no separate executor or [AD](../../glossary.md#autograd) is introduced.

[FLOAT64](../../glossary.md#term-float64) work/accumulation is double without [denormal](../../glossary.md#subnormal-daz-ftz-overflow-and-underflow) flushing. Low types decode exactly and use
[FLOAT32](../../glossary.md#term-float32) work/accumulation without intermediate low gate or projection [narrowing](../../glossary.md#term-narrowing). Hidden/cell
[publication](../../glossary.md#term-publication) narrows once [ties-to-even](../../glossary.md#term-ties-to-even), and state consumers read the stored value, including LSTM
hidden's consumption of its newly stored cell. Each [run](../../glossary.md#term-run) validates the complete logically strided
length [vector](../../glossary.md#term-vector) before any scan output write. Invalid lengths leave all scan siblings unchanged;
repair them and reuse the same preparation. These rules do not authorize backward execution,
[serialization](../../glossary.md#term-serialization) of prepared loops, or support for adjacent operations.

[`RnnSequence`, `GruSequence`, and `LstmSequence`](../nn/recurrent-layers.md#current-nn-recurrent-composition-contract)
retain construction-time Java `long[]` lengths, static [unrolling](../../glossary.md#term-unrolling), and compact outputs. Their
separate `forwardScan` methods retain fixed inference semantics. NN0033 additionally composes
[general Scan](../../glossary.md#general-differentiable-scan) through RNN `forwardGeneralScan` and [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) `rnnScan`, with independently qualified
first-order training and v5 [persistence](../../glossary.md#term-persistence). Neither redirects static containers or adds fixed-scan [BPTT](../../glossary.md#term-backpropagation-through-time-bptt).


---

[Previous topic](masked-index-reductions.md) · [Next topic](normalization.md)
