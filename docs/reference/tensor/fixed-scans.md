# Cumulative and fixed recurrent scans

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

<a id="cumulative-scan-expressions"></a>
### Cumulative-scan expressions

The four current cumulative-scan methods preserve logical Shape while accumulating along exactly
one input axis:

| Public form | Arithmetic | Inclusion | Direction |
|---|---|---|---|
| `cumSum(axis)` | Addition | Inclusive | Forward |
| `cumSum(axis, exclusive, reverse)` | Addition | Selected by `exclusive` | Selected by `reverse` |
| `cumProd(axis)` | Multiplication | Inclusive | Forward |
| `cumProd(axis, exclusive, reverse)` | Multiplication | Selected by `exclusive` | Selected by `reverse` |

An axis identifies the dimension whose positions contribute to each cumulative prefix. Forward
traversal visits that axis from lower to higher indices; reverse traversal visits it from higher
to lower indices without reversing the returned dimension or output positions. Inclusive mode
includes the current value. Exclusive mode omits it, so the first position visited has the
operation identity: additive zero for `CUM_SUM` or multiplicative positive one for `CUM_PROD`.

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
them: it reads no element or host storage and allocates no result storage. A zero-length selected
axis produces a zero-length result with no position at which to emit an identity. On a non-empty
exclusive scan, the identity appears at the first position in the selected traversal direction.

All four methods accept floating and signed-integral inputs, including FP8;
`BOOL` is rejected without truthiness or conversion. A positive or negative caller axis is normalized
exactly once against the input `Shape`. Every valid result is a fresh unlabeled, storage-free
Tensor whose descriptor retains the exact input Shape reference, data type, and `requiresGrad`
eligibility, but uses unresolved layout even when the input layout is resolved. Integral inputs
necessarily remain ineligible for gradients under the descriptor contract.

Integral cumulative product retains exact INT32 or INT64 type and means fixed-width two's-
complement modular multiplication. Floating cumulative product propagates NaN, makes zero times
infinity NaN, and determines zero and infinity signs by multiplication parity. Positive one is the
empty product used by a non-empty exclusive boundary. Arithmetic follows the declared
primitive/aggregate floors: FLOAT32 sites round to FLOAT32, low arithmetic works/accumulates
in FLOAT32 with final declared-result narrowing, and FLOAT64 retains its own format.
The constructor chooses no algorithm/backend route and promises no NaN payload or cross-backend bits.

Provenance records the matching `CumulativeScanKind.CUM_SUM` or `CUM_PROD`, one
`CumulativeScanAttrs(normalizedAxis, exclusive, reverse)`, and exact ordered inputs `[input]`.
Repeated equivalent requests remain distinct expressions with fresh identities. Construction
defines no model-owned gradient rule, compiler capture or canonicalization, backend lowering,
runtime behavior, or execution. Current package-private compiler autograd supports floating
`CUM_SUM` for all four exclusive/reverse combinations by retaining exclusivity and reversing the
scan direction. Floating `CUM_PROD` differentiation is also implemented by
[ReductionGradientRules](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionGradientRules.java):
zero-safe products and zero-count prefix scans separate zero and nonzero input contributions,
then reverse cumulative sums construct the cotangent. Integral scan roles remain non-differentiable.

Failure behavior is local and deterministic:

- `BOOL` fails with `IllegalArgumentException` and message
  `input must have a numeric data type, but was BOOL` before axis validation.
- An axis outside `[-rank, rank - 1]` fails with `IndexOutOfBoundsException`; every scalar axis is
  invalid. These validation failures consume no Tensor identity.
- Exhausted factory identity space fails only after descriptor, operation, and provenance
  metadata have been constructed.

<a id="fixed-recurrent-scan-expressions"></a>
### Fixed recurrent-scan expressions

The public final field-free `model.tensor.RecurrentScan` type is an advanced low-level static
Tensor-expression namespace. Its exactly six methods construct one fixed time-major scan
occurrence; they are not NN layers or modules, an execution service, a registry, or a general
user-defined scan-body API. `Tensor` has no recurrent receiver method.

| Family | Bias-free static form | Biased static form | Ordered outputs |
|---|---|---|---|
| RNN tanh | `RecurrentScan.rnn(input, validLengths, initialHidden, inputWeight, hiddenWeight, direction)` | `RecurrentScan.rnn(input, validLengths, initialHidden, inputWeight, hiddenWeight, bias, direction)` | `[outputs, finalHidden]` |
| Reset-after GRU | `RecurrentScan.gru(input, validLengths, initialHidden, inputWeight, hiddenWeight, direction)` | `RecurrentScan.gru(input, validLengths, initialHidden, inputWeight, hiddenWeight, bias, direction)` | `[outputs, finalHidden]` |
| LSTM | `RecurrentScan.lstm(input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight, direction)` | `RecurrentScan.lstm(input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight, bias, direction)` | `[outputs, finalHidden, finalCell]` |

The explicit first argument is `input [time, batch, inputSize]`. Every Shape is fully static; `inputSize` and
`hiddenSize` must be positive, while `time` and `batch` may be zero. `validLengths` is exactly
non-gradient `INT64[batch]`. Initial hidden and cell states are `[batch, hiddenSize]`. For gate
count `G`, input weight is `[G * hiddenSize, inputSize]`, hidden weight is
`[G * hiddenSize, hiddenSize]`, and optional bias is `[G * hiddenSize]`; `G` is one, three, or four
for RNN, GRU, or LSTM respectively. The input, states, weights, and optional bias use one exact
common floating data type, including either FP8 type.

The fixed transition equations match the current NN cells. With `X = x @ transpose(inputWeight)`
plus the optional input-side bias and `H = h @ transpose(hiddenWeight)`:

```text
RNN_TANH:
  nextHidden = tanh(X + H)

GRU_RESET_AFTER, packed reset/update/candidate order:
  r = sigmoid(X_r + H_r)
  z = sigmoid(X_z + H_z)
  n = tanh(X_n + r * H_n)
  nextHidden = n + z * (h - n)

LSTM, packed input/forget/candidate/output (IFGO) order:
  i = sigmoid(X_i + H_i)
  f = sigmoid(X_f + H_f)
  g = tanh(X_g + H_g)
  o = sigmoid(X_o + H_o)
  nextCell = f * cell + i * g
  nextHidden = o * tanh(nextCell)
```

There is no hidden-side bias, configurable activation, gate order, peephole, projection,
recurrent dropout, arbitrary cell body, callback, nested graph, or region. `RecurrentScanKind`
has closed signatures: RNN and GRU accept five or six inputs and produce exactly two outputs;
LSTM accepts six or seven inputs and produces exactly three. The exact attribute class is
`RecurrentDirection`, whose only values are `FORWARD` and `REVERSE`.

Provenance retains the following exact input order:

```text
RNN/GRU without bias: [input, validLengths, initialHidden, inputWeight, hiddenWeight]
RNN/GRU with bias:    [input, validLengths, initialHidden, inputWeight, hiddenWeight, bias]
LSTM without bias:   [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight]
LSTM with bias:      [input, validLengths, initialHidden, initialCell, inputWeight, hiddenWeight, bias]
```

One call creates one fresh identity-distinct flat `TensorProducer`. `RecurrentScanResult` exposes
the producer's canonical wrappers at positions zero and one; `LstmRecurrentScanResult` exposes
positions zero through two. The dense output descriptor is `[time, batch, hiddenSize]`; final
state descriptors are `[batch, hiddenSize]`. Their layouts are unresolved, they are unlabeled and
storage-free, and their `requiresGrad` value is the OR of the floating input, state, weight, and
optional-bias roles. Valid lengths never contribute to gradient eligibility. No projection,
gate, mask, step, padding, or zero-fill intermediate Tensor is constructed.

For original batch row `b` with runtime length `L[b]`, `FORWARD` consumes coordinates
`0 .. L[b]-1`; `REVERSE` consumes `L[b]-1 .. 0`, reversing only the valid prefix. Both directions return
dense outputs aligned to original time positions, with exact positive zero at `t >= L[b]`, plus
explicit final states. A zero-length row returns its initial states semantically. A zero-time
input has an empty dense output and requires every runtime length to be zero. These are fixed
operation semantics, not values computed by Model construction.

Construction never reads `validLengths` storage, so it cannot validate scalar bounds or infer
active rows. It validates nulls in declaration order, then direction, floating input and static
rank-three Shape, positive input width, the valid-length descriptor, state descriptors, packed
weight extents, optional bias, and exact common data type before allocating output identities.
Descriptor or arithmetic-overflow failures therefore consume no Tensor ID. Identifier exhaustion
may consume earlier output IDs without returning a partial result, following the shared canonical
multi-output factory contract.

Current generic graph capture can translate the shared producer to one flat node with the same
ordered inputs and outputs. Current package-private Compiler forward-only inference independently
revalidates the fully static descriptor contract, while both backward-capable modes reject a
complete forward inventory containing recurrence before derivative Tensor allocation.

CPU scalar execution currently supports all six signatures and both directions for homogeneous
floating types, including all four low types, with canonical dense outputs. Every input role, including INT64
lengths, accepts resolved nonnegative-stride/offset reads: dense, transposed, sliced/gapped,
overlapping and zero-stride repeated layouts. Non-singleton repetition requires Model's view flag.
Negative and unresolved storage geometry is unsupported. Every dimension, packed gate extent,
logical count and referenced storage span must fit `int`; fresh Engine bindings must retain the
full referenced span, not merely the logical count. Empty layouts have span zero regardless of
unused offset.

CPU low fixed recurrence includes BFLOAT16, FLOAT16 and both OFP8 types under the same
layout, geometry and actual derivative guards; no separate executor or AD is introduced.

FLOAT64 work/accumulation is double without denormal flushing. Low types decode exactly and use
FLOAT32 work/accumulation without intermediate low gate or projection narrowing. Hidden/cell
publication narrows once ties-to-even, and state consumers read the stored value, including LSTM
hidden's consumption of its newly stored cell. Each run validates the complete logically strided
length vector before any scan output write. Invalid lengths leave all scan siblings unchanged;
repair them and reuse the same preparation. These rules do not authorize backward execution,
serialization of prepared loops, or support for adjacent operations.

[`RnnSequence`, `GruSequence`, and `LstmSequence`](../nn/recurrent-layers.md#current-nn-recurrent-composition-contract)
retain construction-time Java `long[]` lengths, static unrolling, and compact outputs. Their
separate `forwardScan` methods retain fixed inference semantics. NN0033 additionally composes
general Scan through RNN `forwardGeneralScan` and F32/F64 `rnnScan`, with independently qualified
first-order training and v5 persistence. Neither redirects static containers or adds fixed-scan BPTT.


---

[Previous topic](masked-index-reductions.md) · [Next topic](normalization.md)
