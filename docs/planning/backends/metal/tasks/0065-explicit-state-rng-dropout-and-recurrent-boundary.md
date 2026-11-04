# Task 0065: Explicit-State [RNG](../../../../glossary.md#graph-rng-state--graphrngstate), [Dropout](../../../../glossary.md#dropout--inverted-dropout), and Recurrent Boundary

## Status

Complete — approved plan `9ceee2b53c88a8c87ca4b8a42cdd2c92682074c9` was implemented by
`05d83074` plus correctness remediation `ed7a3369`. The cutover adds exact both-profile
`INITIAL_STATE` and [accelerator](../../../../glossary.md#term-accelerator) [FLOAT32](../../../../glossary.md#term-float32) [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout), while [eager](../../../../glossary.md#term-eager-evaluation) [distributions](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) and all recurrent rows
remain [fail-closed](../../../../glossary.md#term-fail-closed). Native package, complete [Metal](../../../../glossary.md#term-metal), focused [Compiler](../../../../glossary.md#term-compiler), [conformance](../../../../glossary.md#term-conformance), public [Engine](../../../../glossary.md#term-engine),
[Javadoc](../../../../glossary.md#term-javadoc)/architecture, full repository test/build, and documentation/diff [validation](../../../../glossary.md#term-validation) passed after
remediation. Independent cumulative code/evidence/documentation and security/determinism reviews
returned `APPROVE` with zero unresolved P0/P1/P2.

## Change class

Class C — the implementation adds the first [Metal](../../../../glossary.md#term-metal) explicit-state random execution, zero-feed
native creation, one genuine three-output custom [operation](../../../../glossary.md#operation), saved-[mask](../../../../glossary.md#term-boolean-mask) [gradient](../../../../glossary.md#term-gradient) use, private
generator [identity](../../../../glossary.md#term-identity-meanings), and public [Engine](../../../../glossary.md#term-engine) behavior. It advertises no recurrent row: all three recurrent
kinds remain [fail-closed](../../../../glossary.md#term-fail-closed) for explicit elementary-function, recurrence, and [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) reasons.
Independent cumulative code/evidence/documentation and security review was the final completion
gate and returned `APPROVE` after correctness remediation.

## Goal

Implement the maximal deterministic domain currently provable from [Model](../../../../glossary.md#term-model) and [Compiler](../../../../glossary.md#term-compiler) contracts for
`INITIAL_STATE` and explicit-state [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout): exact raw key/counter [materialization](../../../../glossary.md#materialization) under both
profiles and [accelerator](../../../../glossary.md#term-accelerator) [FLOAT32](../../../../glossary.md#term-float32) dropout through one versioned [Metal](../../../../glossary.md#term-metal)-private counter algorithm and
one fixed custom-program route. Preserve exact draw membership, [canonical](../../../../glossary.md#term-canonical-form) [mask](../../../../glossary.md#term-boolean-mask), positive-zero drop,
counter advancement, replay, branching, chaining, concurrent-session isolation, saved-mask
backward use, and [evaluation](../../../../glossary.md#term-evaluation) bypass. Route selection must occur before resource creation and may
use no timing, [autotuning](../../../../glossary.md#autotuning--model-autotuning), [cache](../../../../glossary.md#term-cache) winner, input value, retry, fallback, hidden generator, or [host](../../../../glossary.md#term-host)
repair.

Audit `RNN_TANH`, `GRU_RESET_AFTER`, and [`LSTM`](../../../../glossary.md#term-long-short-term-memory-lstm) completely, including both directions, optional
[bias](../../../../glossary.md#term-bias), [runtime](../../../../glossary.md#term-runtime) valid lengths, output states, recursive numerical sites, and Compiler [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt). Do not
make any recurrent [occurrence](../../../../glossary.md#term-occurrence) production-capable without the missing all-domain elementary proof
and [gradient](../../../../glossary.md#term-gradient) boundary. Audit [eager](../../../../glossary.md#term-eager-evaluation) normal, uniform, integral, and [Bernoulli](../../../../glossary.md#term-bernoulli-distribution) factories, but do not
invent Metal [graph](../../../../glossary.md#graph) operations for host-created [leaf](../../../../glossary.md#term-leaf) values.

## [Scope](../../../../glossary.md#term-scope)

### Exact audited 115-kind [registry](../../../../glossary.md#term-registry) boundary

The current registry has exactly five [operation](../../../../glossary.md#operation) rows in [scope](../../../../glossary.md#term-scope):

| Wire/kind | [Arity](../../../../glossary.md#term-arity) and attribute wire | Exact [Model](../../../../glossary.md#term-model) role | Pre-cutover [Metal](../../../../glossary.md#term-metal) state | Task-0065 decision |
|---|---|---|---|---|
| 101 [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout) | 2 inputs / 3 outputs; 33 `DROPOUT` | `[value,state] -> [output,keepMask,nextState]`; one raw [binary64](../../../../glossary.md#floating-point-precision-and-rounding) probability word | non-executable; [MPSGraph](../../../../glossary.md#term-mpsgraph) `UNAVAILABLE / MU_RNG`; custom `PENDING / CP_STATE` | [accelerator](../../../../glossary.md#term-accelerator) [FLOAT32](../../../../glossary.md#term-float32) custom route |
| 102 `INITIAL_STATE` | 0 inputs / 1 output; 34 `GRAPH_RNG_STATE` | exact raw `INT64 Shape[2]` key/counter [materialization](../../../../glossary.md#materialization) | non-executable; MPSGraph `UNAVAILABLE / MU_RNG`; custom `PENDING / CP_STATE` | exact both-profile custom route |
| 103 `RNN_TANH` | 5..6 inputs / 2 outputs; 35 `RECURRENT_DIRECTION` | fixed [tanh](../../../../glossary.md#term-tanh) recurrence, optional input-side [bias](../../../../glossary.md#term-bias), dense outputs and final hidden | non-executable; MPSGraph `UNAVAILABLE / MU_RECURRENT`; custom `PENDING / CP_STATE` | remain blocked |
| 104 `GRU_RESET_AFTER` | 5..6 inputs / 2 outputs; 35 `RECURRENT_DIRECTION` | reset/update/candidate recurrence, optional packed input-side bias | same unavailable/pending state | remain blocked |
| 105 [`LSTM`](../../../../glossary.md#term-long-short-term-memory-lstm) | 6..7 inputs / 3 outputs; 35 `RECURRENT_DIRECTION` | input/forget/candidate/output recurrence, hidden and cell final states | same unavailable/pending state | remain blocked |

Attribute wire 33 contains exactly one raw `double` word and accepts every [finite](../../../../glossary.md#term-finite) numerical value in
`[0,1)`, including both signed zeros. Wire 34 contains exactly the raw key then counter words and
accepts every 64-bit pattern. Wire 35 contains exactly `FORWARD=1` or `REVERSE=2`. [Schema](../../../../glossary.md#term-schema) 15 already
encodes zero inputs, variable input counts, three outputs, all six [carrier](../../../../glossary.md#term-storage-carrier) types, and the complete
attribute grammar. Type wires remain FLOAT32 `1`, [INT32](../../../../glossary.md#data-type--datatype) `2`, [BOOL](../../../../glossary.md#data-type--datatype) `3`, [FLOAT64](../../../../glossary.md#term-float64) `4`, [BFLOAT16](../../../../glossary.md#term-bfloat16) `5`,
and [INT64](../../../../glossary.md#data-type--datatype) `6`. These existing wires are representation, not capability.

The current native [signature](../../../../glossary.md#term-signature) tables accept numeric operation positions 101..105, but the native
operation enum intentionally jumps from 100 to 106 and has no executable selector [constants](../../../../glossary.md#term-constant) for
these five positions. Task 0065 may name native custom constants only for existing wires 101 and
102. It must not add a Model kind, operation wire, attribute wire, [data type](../../../../glossary.md#data-type--datatype), route wire, public [RNG](../../../../glossary.md#graph-rng-state--graphrngstate)
algorithm enum, or recurrent [alias](../../../../glossary.md#aliasing-and-arena).

### Random-[distribution](../../../../glossary.md#term-distribution) inventory and [ownership](../../../../glossary.md#term-ownership-meanings)

The [graph](../../../../glossary.md#graph)-random inventory is exactly `INITIAL_STATE` plus [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout). There is no graph [operation](../../../../glossary.md#operation)
for normal, general uniform, bounded integral, or [Bernoulli](../../../../glossary.md#term-bernoulli-distribution) [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters) in the 115-kind [registry](../../../../glossary.md#term-registry).
`TensorRandoms` instead exposes [eager](../../../../glossary.md#term-eager-evaluation) `randomNormal`, `randomUniform`, two `randomInt` overloads,
and `randomBernoulli`. Each consumes a caller-owned `RandomGenerator` on the [host](../../../../glossary.md#term-host) and returns a
[provenance](../../../../glossary.md#provenance)-free [leaf](../../../../glossary.md#term-leaf). [Metal](../../../../glossary.md#term-metal) may receive the resulting represented values through ordinary ingress;
it does not own their generator, distribution mapping, [seed](../../../../glossary.md#term-seed), counter, replay, or sampling.

The eager mappings are exact host-factory contracts, not candidate Metal [distributions](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings):

| Factory | Caller-generator consumption and mapping | Result [carrier](../../../../glossary.md#term-storage-carrier) conversion |
|---|---|---|
| `randomNormal` | one `nextGaussian()` per logical [row-major](../../../../glossary.md#term-row-major) [element](../../../../glossary.md#term-element); [binary64](../../../../glossary.md#floating-point-precision-and-rounding) multiplication `gaussian * standardDeviation` followed by binary64 addition of [`mean`](../../../../glossary.md#term-mean), with no [fused](../../../../glossary.md#specialization-and-fusion) site | direct [FLOAT64](../../../../glossary.md#term-float64), one [narrowing](../../../../glossary.md#term-narrowing) conversion for [FLOAT32](../../../../glossary.md#term-float32), or FLOAT32 followed by [BFLOAT16](../../../../glossary.md#term-bfloat16) conversion |
| `randomUniform` | one unchanged `nextDouble(lowerBound, upperBound)` call per logical row-major element | direct FLOAT64, one narrowing conversion for FLOAT32, or FLOAT32 followed by BFLOAT16 conversion |
| `randomInt` with `int` bounds | one unchanged `nextInt(origin, bound)` call per logical row-major element | direct [INT32](../../../../glossary.md#data-type--datatype) |
| `randomInt` with `long` bounds | one unchanged `nextLong(origin, bound)` call per logical row-major element | direct [INT64](../../../../glossary.md#data-type--datatype) |
| `randomBernoulli` | one `nextDouble()` per logical row-major element; true exactly when the draw is less than the closed-domain probability | [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) |

Task 0065 does not reproduce, wrap, seed, or promise any of those caller-generator sequences.

Task 0065 therefore adds no entropy source and makes no generic distribution claim. The caller's
raw graph key is the only seed/domain word for dropout, and the raw graph counter is the next
abstract logical draw position. Neither word is normalized, randomized, secret, or implicitly
initialized. The implementation is not cryptographic and makes no distinct-key stream-separation,
statistical certification, cross-[backend](../../../../glossary.md#backend) bitstream, cross-version replay, or [serialization](../../../../glossary.md#term-serialization) promise.

### Exact initial-state production domain

Admit `GraphRngKind.INITIAL_STATE` under both `STRICT_IEEE` and [`ACCELERATOR`](../../../../glossary.md#term-accelerator) only when the sole
output is [canonical](../../../../glossary.md#term-canonical-form) dense, zero-offset, non-[view](../../../../glossary.md#term-view-meanings), non-[gradient](../../../../glossary.md#term-gradient) [INT64](../../../../glossary.md#data-type--datatype) `Shape.of(2)`. The [kernel](../../../../glossary.md#kernel)
writes key bits to logical [lane](../../../../glossary.md#term-lane) zero and counter bits to lane one exactly. It performs no [hash](../../../../glossary.md#term-hash),
draw, key expansion, counter advancement, [host](../../../../glossary.md#term-host) [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters), or persistent-state mutation.

A one-[node](../../../../glossary.md#node) initializer is a legal zero-feed/one-[target](../../../../glossary.md#term-target) program. Java program [validation](../../../../glossary.md#term-validation), native
image validation, custom-program creation, resource declaration, finalization, [binding](../../../../glossary.md#term-binding), and
invocation must accept that [topology](../../../../glossary.md#term-topology) without a dummy input. Native creation must permit
`feed_count == 0` with the corresponding pointer absent while retaining positive value, node, and
target counts. Every generic Java/native node walk must consume the declared input/output arrays
and must not call a first-input accessor on the zero-input row.

### [Metal](../../../../glossary.md#term-metal)-private [dropout](../../../../glossary.md#dropout--inverted-dropout) generator

Select the [immutable](../../../../glossary.md#term-immutable) [backend](../../../../glossary.md#backend)-private configuration
`SYNAPTIK_METAL_SPLITMIX64_COUNTER_V1`. It deliberately uses the same reviewed integer [permutation](../../../../glossary.md#permutation)
as the [CPU](../../../../glossary.md#term-cpu)-private precedent but has a distinct Metal [identity](../../../../glossary.md#term-identity-meanings) and creates no portable bitstream
contract. Unsigned arithmetic is modulo `2^64`:

```text
KEY_BIAS = 0x9e3779b97f4a7c15
MIX_MULTIPLIER_1 = 0xbf58476d1ce4e5b9
MIX_MULTIPLIER_2 = 0x94d049bb133111eb

mix64(z):
  z = (z ^ (z >>> 30)) * MIX_MULTIPLIER_1
  z = (z ^ (z >>> 27)) * MIX_MULTIPLIER_2
  return z ^ (z >>> 31)

keyOffset(key) = mix64(key + KEY_BIAS)
word(key, counter, i) = mix64(counter + i + keyOffset(key))
uniform53(word) = (word >>> 11) * 2^-53
```

For fixed key, `mix64` is a permutation of 64-bit words; no [mutable](../../../../glossary.md#term-mutable) stream cursor is needed. Logical
[row-major](../../../../glossary.md#term-row-major) ordinal `i`, not physical address, [threadgroup](../../../../glossary.md#term-threadgroup), grid [decomposition](../../../../glossary.md#term-decomposition), [dispatch](../../../../glossary.md#term-dispatch) order, or
session identity, selects the draw. The [kernel](../../../../glossary.md#kernel) computes its word directly from explicit input state
and `i`. Scheduling and concurrent invocation therefore cannot change a result.

[Mask](../../../../glossary.md#term-boolean-mask) membership uses the exact 53-bit numerator `n = word >>> 11`, in
`[0, 2^53 - 1]`. For the numerical value of the raw [binary64](../../../../glossary.md#floating-point-precision-and-rounding) probability bits, define the
[normative](../../../../glossary.md#term-normative) unsigned-64 [metadata](../../../../glossary.md#term-metadata) value
`threshold(p) = ceil(p * 2^53)`, evaluated exactly rather than through [floating-point](../../../../glossary.md#term-floating-point)
multiplication. Both signed zeros map to zero; every positive probability below `2^-53` maps to
one; and the complete legal domain maps into `[0, 2^53 - 1]`. Keep exactly when
`n >= threshold(p)`, including equality. This is exactly the [Model](../../../../glossary.md#term-model) predicate
`uniform53 >= probability`.

Preparation derives the threshold by integer decomposition of the raw binary64 sign, [exponent](../../../../glossary.md#term-exponent), and
[significand](../../../../glossary.md#term-significand), without floating-point [rounding](../../../../glossary.md#term-rounding). Native [validation](../../../../glossary.md#term-validation) independently recomputes the same
integer from the [raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) and authenticates the supplied value. No [FLOAT32](../../../../glossary.md#term-float32) [narrowing](../../../../glossary.md#term-narrowing) participates
in membership. Signed probability zero creates an all-true mask but still consumes every draw.
Each mask byte is exactly zero or one. Independent threshold oracles include:

| Probability raw bits | Exact value class | `threshold(p)` |
|---|---|---|
| `0000000000000000` | [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) | `0` |
| `8000000000000000` | [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero) | `0` |
| `0000000000000001` | least positive binary64 [subnormal](../../../../glossary.md#term-subnormal) | `1` |
| `3c90000000000000` | `2^-54` | `1` |
| `3ca0000000000000` | `2^-53` | `1` |
| `3ca0000000000001` | next binary64 value above `2^-53` | `2` |
| `3fe0000000000000` | `1/2` | `2^52` |
| `3fefffffffffffff` | largest binary64 value below one | `2^53 - 1` |

Independent [oracle](../../../../glossary.md#term-test-oracle) tests must derive, without production helpers, at least these fixed word [vectors](../../../../glossary.md#term-vector):

| Key | Counter | Word | Top 53 bits |
|---|---|---|---|
| `0000000000000000` | `0000000000000000` | `48218226ff3cd4bf` | `09043044dfe79a` |
| `0000000000000000` | `0000000000000001` | `ea8568d2e45fd6cb` | `1d50ad1a5c8bfa` |
| `0000000000000001` | `0000000000000000` | `dce423fc82c0d5b8` | `1b9c847f90581a` |
| `ffffffffffffffff` | `ffffffffffffffff` | `e8ba9f99ca933538` | `1d1753f3395266` |
| `0000000000001234` | `0000000000000007` | `3e4cf5a0c9489779` | `07c99eb4192912` |

The generator name, [constants](../../../../glossary.md#term-constant), integer threshold mapping, FLOAT32 scaling policy, mask policy, and
state-write policy are part of backend-local identity 21. A later algorithm or policy must advance
identity; it must not silently reuse V1.

### Exact [dropout](../../../../glossary.md#dropout--inverted-dropout) production domain

Admit `DropoutKind.DROPOUT` only under [`ACCELERATOR`](../../../../glossary.md#term-accelerator) when all of the following hold:

- inputs are ordered `[value,state]`; outputs are ordered `[output,keepMask,nextState]`; all five
  graph values are distinct;<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- value and output are FLOAT32 with the same fully static Shape and equal gradient eligibility;
  keep mask is canonical BOOL with that exact Shape and no gradient; state and next state are
  non-gradient INT64 <span data-glossary-navigation="true"> ([INT64 glossary](../../../../glossary.md#data-type--datatype))</span>`Shape.of(2)`;
- every boundary has [canonical](../../../../glossary.md#term-canonical-form) dense, zero-offset, non-[view](../../../../glossary.md#term-view-meanings) [layout](../../../../glossary.md#layout). The [Shape](../../../../glossary.md#shape) [rank](../../../../glossary.md#term-rank) is `0..16`, every
  present extent is in <span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent))</span>`1..UINT32_MAX`, and the logical [element](../../../../glossary.md#term-element) count is in `1..UINT32_MAX`. Scalar
  rank zero therefore consumes one draw; dynamic and zero-extent Shapes remain false;<span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar))</span>
- every element count, stride, logical ordinal, grid coordinate, byte count, byte offset, physical
  span, and native <span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`NSUInteger` conversion passes checked Java and native bounds before pipeline or
  buffer creation; and<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>
- the probability is the complete [Model](../../../../glossary.md#term-model) domain of [finite](../../../../glossary.md#term-finite) [binary64](../../../../glossary.md#floating-point-precision-and-rounding) values in `[0,1)`, including both
  signed zeros. Capability must not select a fixed probability, key, counter, Shape, corpus, or
  mask outcome.

For each kept value, first derive the complement bits exactly as the [FLOAT32](../../../../glossary.md#term-float32) conversion of the
binary64 expression `1.0d - probability`; the probability itself must not be narrowed before that
subtraction. Then execute the current Model accelerator formula at the remaining typed sites:
FLOAT32 one divided by that typed complement, followed by FLOAT32 input multiplication. The raw
binary64 attribute remains the exact membership [leaf](../../../../glossary.md#term-leaf). Use the already qualified accelerator
primitive rules, including [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz), and preserve the sites and association. Do not pre-narrow the
probability, replace the formula with input division, use a reciprocal approximation, fuse the
algebra, reduce [precision](../../../../glossary.md#term-precision), or accept a final-output [tolerance](../../../../glossary.md#term-tolerance). Java and native [metadata](../../../../glossary.md#term-metadata) must [carry](../../../../glossary.md#term-carry)
the raw probability bits, exact integer threshold, and derived FLOAT32 complement bits for
independent [validation](../../../../glossary.md#term-validation).

A dropped value is exact positive FLOAT32 zero even for [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero), [NaN](../../../../glossary.md#term-nan), or [infinity](../../../../glossary.md#term-infinity). A kept
[signed zero](../../../../glossary.md#term-signed-zero) and infinity retain the formula-required sign; kept NaN remains NaN without a [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication)
promise. Ordinary finite, [subnormal](../../../../glossary.md#term-subnormal), [overflow](../../../../glossary.md#term-overflow), [underflow](../../../../glossary.md#term-underflow), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), NaN, and infinity outputs are
tested against an independent three-site [oracle](../../../../glossary.md#term-test-oracle), not [CPU](../../../../glossary.md#term-cpu)'s older finite-order emitter. Probability
values immediately around binary64 and [binary32](../../../../glossary.md#floating-point-precision-and-rounding) threshold/conversion boundaries, including the
largest value below one, are mandatory cases.

Load key and counter from state [lanes](../../../../glossary.md#term-lane) zero and one. The next state writes the exact unchanged key
and `counter + N mod 2^64`, where `N` is the static logical element count. Advancement is independent
of probability, input values, [mask](../../../../glossary.md#term-boolean-mask), value type, execution order, grid, session, and [gradient](../../../../glossary.md#term-gradient) use.
Counter wrap is valid. Branching the same state intentionally reuses an interval; chaining the
returned next state selects the subsequent interval.

`STRICT_IEEE` FLOAT32 remains false because current [Metal](../../../../glossary.md#term-metal) primitive arithmetic does not preserve the
complete strict subnormal/[rounding](../../../../glossary.md#term-rounding) set. [FLOAT64](../../../../glossary.md#term-float64) dropout remains false because the [GPU](../../../../glossary.md#term-gpu) route has no
proved FLOAT64 arithmetic. [BFLOAT16](../../../../glossary.md#term-bfloat16) remains false because it has no accelerator relaxation and no
direct [correctly rounded](../../../../glossary.md#nan-infinity-and-signed-zero) scaling/conversion proof. Integral and [BOOL](../../../../glossary.md#data-type--datatype) values are Model-illegal.
General affine/view, unresolved, empty, negative/zero-[stride](../../../../glossary.md#term-stride), in-place, aliased, and over-limit
occurrences remain false.

### Fixed custom route, outputs, and [lifecycle](../../../../glossary.md#lifecycle)

Add one Task-0065 native source/header and append it unconditionally to the existing fixed
custom-program library. Add native custom [operation](../../../../glossary.md#operation) [constants](../../../../glossary.md#term-constant) only for wires 101 and 102. One
initializer thread writes the two raw words. One [dropout](../../../../glossary.md#dropout--inverted-dropout) thread owns each value/[mask](../../../../glossary.md#term-boolean-mask) position; thread
zero additionally writes the two next-state words exactly once. No random-word [buffer](../../../../glossary.md#term-buffer), replay
buffer, mask staging buffer, scratch, [atomic](../../../../glossary.md#resource-transaction-and-rollback), reduction pass, persistent counter, per-thread
generator, [host](../../../../glossary.md#term-host) [sampling](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters)/output/mask/state calculation or repair, or second route exists. Cold Java
derivation of the authenticated threshold and complement [metadata](../../../../glossary.md#term-metadata) above is permitted; it is not
execution or sampling.

Before [dispatch](../../../../glossary.md#term-dispatch), cold Java and native [validation](../../../../glossary.md#term-validation) independently authenticate kind, attribute bits,
[cardinality](../../../../glossary.md#term-cardinality), ordered value roles, type, [Shape](../../../../glossary.md#shape), [canonical](../../../../glossary.md#term-canonical-form) [layout](../../../../glossary.md#layout), [gradient](../../../../glossary.md#term-gradient) flags, [UINT32](../../../../glossary.md#index-mappings-and-cost-dominance) and byte
bounds, value [identity](../../../../glossary.md#term-identity-meanings), resource access, and exact metadata. Reject every output span against both
input spans and reject every output/output pair before mutation. Input/input overlap may remain
legal because both are read-only, but it creates no semantic [alias](../../../../glossary.md#aliasing-and-arena). A failed validation writes no
state, mask, or value.

Every [partition](../../../../glossary.md#partition) containing wire 101 or 102 selects `CUSTOM_PROGRAM` only. It may compose the random
[kernels](../../../../glossary.md#kernel) with already approved custom or nested [MPSGraph](../../../../glossary.md#term-mpsgraph) steps in stable [topological order](../../../../glossary.md#term-topological-order), including
`INITIAL_STATE -> DROPOUT`, chained dropouts, and generated backward consumers. Candidate generation
must never offer MPSGraph for either row. Route choice never consults probability, key/counter,
input payloads, timing, benchmark data, [cache](../../../../glossary.md#term-cache) winners, [runtime](../../../../glossary.md#term-runtime) failure, or retry.

Prepared executables remain [immutable](../../../../glossary.md#term-immutable) and reusable. Each [bound invocation](../../../../glossary.md#bound-invocation--boundinvocation) owns ordinary [run](../../../../glossary.md#term-run)-local
buffers; no [RNG](../../../../glossary.md#graph-rng-state--graphrngstate) field enters session, [`RunState`](../../../../glossary.md#run-state--runstate), Runtime, [Trace](../../../../glossary.md#trace), or the prepared recipe. Equal
explicit inputs and the same identity/configuration replay equal mask, output, and next state.
Repeated calls in one session, independent sessions, and concurrent invocations have no counter
interaction. Close, [lease](../../../../glossary.md#term-lease), [rollback](../../../../glossary.md#term-rollback), suppression, tracing, and [synchronous](../../../../glossary.md#term-synchronous) invocation retain the
existing lifecycle.

### Training, [evaluation](../../../../glossary.md#term-evaluation), saved [mask](../../../../glossary.md#term-boolean-mask), and [gradients](../../../../glossary.md#term-gradient)

[`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout) always denotes training dropout. There is no [Metal](../../../../glossary.md#term-metal) training flag and no evaluation [kernel](../../../../glossary.md#kernel).
The current [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) evaluation path returns the original value and state without constructing wire 101,
so it consumes no draw and advances no state. Tests must prove this [graph](../../../../glossary.md#graph)-level absence rather than
turn probability into zero or [dispatch](../../../../glossary.md#term-dispatch) a disabled dropout.

[Compiler](../../../../glossary.md#term-compiler) already [captures](../../../../glossary.md#term-capture-meanings) all three forward outputs and constructs the sole first-order input
[cotangent](../../../../glossary.md#cotangent) from the exact same-[occurrence](../../../../glossary.md#term-occurrence) slot-one mask:

```text
dInput = where(mask, gradient / typed(1.0d - probability), positiveZero)
```

Task 0065 changes no Compiler formula. The forward kernel must publish mask and next state as
ordinary live internal values even when only output slot zero is public. Resource [planning](../../../../glossary.md#term-planning) keeps the
mask alive until the generated backward `WHERE` consumes it and never resamples, reconstructs, or
infers it from output values. State input/output and mask are non-[differentiable](../../../../glossary.md#derivative-and-chain-rule).

Admit legal `requiresGrad=true` [FLOAT32](../../../../glossary.md#term-float32) forward [metadata](../../../../glossary.md#term-metadata) as well as no-grad metadata. End-to-end
backward succeeds only when the existing [accelerator](../../../../glossary.md#term-accelerator) [scalar](../../../../glossary.md#term-scalar)-DIV, exact [BOOL](../../../../glossary.md#data-type--datatype) `WHERE`, zero [constant](../../../../glossary.md#term-constant),
reduction/[seed](../../../../glossary.md#term-seed), and any requested surrounding graph are independently capable. Public [Engine](../../../../glossary.md#term-engine) proof
must exercise a bounded primitive-closed dropout gradient and verify the selected mask, positive-zero
dropped cotangent, typed scale, and absence of state differentiation. Unsupported strict or [carrier](../../../../glossary.md#term-storage-carrier)
cases fail during Planning, before native resource creation.

### Exact recurrent contract audit

All three recurrent rows consume a [time-major](../../../../glossary.md#term-time-major) floating input `[T,N,I]`, [runtime](../../../../glossary.md#term-runtime) non-[gradient](../../../../glossary.md#term-gradient) [INT64](../../../../glossary.md#data-type--datatype)
valid lengths `[N]`, initial hidden `[N,H]`, packed input [weight](../../../../glossary.md#term-weight) `[G*H,I]`, packed hidden weight
`[G*H,H]`, and optional packed input-side [bias](../../../../glossary.md#term-bias) `[G*H]`; [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm) additionally consumes initial cell
`[N,H]`. Gate count `G` is one, three, or four. [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../../../glossary.md#term-gated-recurrent-unit-gru) output `[outputs[T,N,H],finalHidden[N,H]]`;
LSTM additionally outputs `finalCell[N,H]`. `FORWARD` traverses `0..L-1`; `REVERSE` traverses the
valid prefix `L-1..0`. Every padded output is exact [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), zero length returns the exact
initial state, and the complete length [vector](../../../../glossary.md#term-vector) must validate in `[0,T]` before any output mutation.
Both directions and biased/bias-free [signatures](../../../../glossary.md#term-signature) are current; bidirectionality, stacking,
configurable [activations](../../../../glossary.md#term-activation), projection, peepholes, recurrent [dropout](../../../../glossary.md#dropout--inverted-dropout), and [graph](../../../../glossary.md#graph) regions are absent.

The current [schema](../../../../glossary.md#term-schema)-15 [cardinality](../../../../glossary.md#term-cardinality)/type/attribute image can represent every recurrent signature, and
current generic multi-output [publication](../../../../glossary.md#term-publication) can represent its sibling states. Those historical Task-
0037 representation blockers are resolved. They do not create execution support. The recurrent
production subset remains empty for independent reasons:

1. installed MPSGraph recurrent selectors have no Model runtime-length input, atomic all-length
   validation, skipped padded recurrence, exact positive-zero padding contract, or exact final-state
   output roles. Training intermediates are not substitutes;<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [Model glossary](../../../../glossary.md#term-model); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [validation glossary](../../../../glossary.md#term-validation); [padding glossary](../../../../glossary.md#padding))</span>
2. no current native custom recurrent loop validates all lengths before writing and publishes the
   exact two/three output sets for both directions and optional bias;
3. `RNN_TANH` requires a complete [accelerator](../../../../glossary.md#term-accelerator) [FLOAT32](../../../../glossary.md#term-float32) [`TANH`](../../../../glossary.md#term-tanh) site proof. The direct historical
   route has invalid negative-zero/NaN behavior, and custom <span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan))</span>`TANH` remains pending without an
   all-binary32 recursive-floor certificate;<span data-glossary-navigation="true"> ([binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions))</span>
4. GRU and LSTM additionally require stable sigmoid branches and therefore the missing certified
   exponential bridge retained by Blocked Task 0053, plus their candidate/cell <span data-glossary-navigation="true"> ([sigmoid glossary](../../../../glossary.md#term-sigmoid))</span>`TANH` sites; and
5. [Compiler](../../../../glossary.md#term-compiler) rejects every recurrent [occurrence](../../../../glossary.md#term-occurrence) in `FORWARD_AND_BACKWARD` and `TRAINING_STEP` before
   derivative allocation because BPTT, saved-gate policy, and recurrent gradient formulas do not
   exist.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative))</span>

Do not advertise a zero-time, zero-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters), all-zero-length, single-cell, fixed-length, selected-value,
or full-length-only special case. Capability cannot inspect valid-length payloads, and a no-work
special case would evade rather than implement the recurrence. [FLOAT64](../../../../glossary.md#term-float64), [BFLOAT16](../../../../glossary.md#term-bfloat16), strict FLOAT32,
dynamic [Shapes](../../../../glossary.md#shape), [views](../../../../glossary.md#term-view-meanings), in-place execution, all gradients, and every recurrent row remain false.
Unblocking requires a separately reviewed task after both a complete elementary proof and the
Compiler [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) decision; schema presence and sampled [device](../../../../glossary.md#term-device) output are insufficient.

### Catalog, [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), and [lifecycle](../../../../glossary.md#lifecycle) deltas

- Mark exactly wires 101 and 102 structurally executable: `99/16 -> 101/14`. Wires 103..105 remain
  non-executable.
- Production capability becomes `81/34 -> 83/32`. `INITIAL_STATE` has one exact both-profile domain;
  [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout) has the [accelerator](../../../../glossary.md#term-accelerator) [FLOAT32](../../../../glossary.md#term-float32) domain above. Recurrent kinds remain false.
- Keep the [MPSGraph](../../../../glossary.md#term-mpsgraph) catalog exactly `75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE`; structural catalog
  state is not route authorization.
- Add custom reason `CA_0065`, change exactly the two RNG rows from pending to available, and produce
  custom catalog <span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>`60 AVAILABLE / 55 PENDING / 0 UNAVAILABLE-WITH-PROOF`. Recurrent rows retain their
  current pending entry and the explicit blockers above.
- Keep schema 15, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, type wires `1..6`, [operation](../../../../glossary.md#operation) wires `1..115`, attribute wires `0..41`, route
  wires `1..3`, generic cardinality/image grammar, and exactly thirteen exports. Filling native
  selector names for existing positions 101 and 102 is not a new wire or export.<span data-glossary-navigation="true"> ([cardinality glossary](../../../../glossary.md#term-cardinality))</span>
- Advance workload-signature, exact-default-policy, candidate-schema, compatibility-schema,
  route-policy, and codec identities together <span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [codec glossary](../../../../glossary.md#term-codec))</span>`20 -> 21`. Identity 20 and every older identity
  reject with no alias, migration reader, fallback library, or shim.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Preserve stable value/feed/target order, transactional finalization, synchronous invocation,
  internal/output liveness, physical-span authentication, leases, reuse, concurrency/session
  isolation, close ordering, trace identity, and the profile-free Runtime/Trace boundary.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [synchronous glossary](../../../../glossary.md#term-synchronous); [liveness glossary](../../../../glossary.md#term-liveness); [authentication glossary](../../../../glossary.md#term-authentication); [leases glossary](../../../../glossary.md#term-lease); [concurrency glossary](../../../../glossary.md#term-concurrency); [trace glossary](../../../../glossary.md#trace); [Runtime glossary](../../../../glossary.md#term-runtime))</span>

## Provable domains versus explicit blockers

The provable cutover is exactly:

- exact raw zero-input `INITIAL_STATE` under both profiles for every key/counter bit pair; and
- [accelerator](../../../../glossary.md#term-accelerator) [FLOAT32](../../../../glossary.md#term-float32) [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout) over the complete binary64 probability and explicit-state domains,
  canonical positive static rank-<span data-glossary-navigation="true"> ([binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [canonical glossary](../../../../glossary.md#term-canonical-form); [rank glossary](../../../../glossary.md#term-rank))</span>`0..16` [Shapes](../../../../glossary.md#shape) bounded by `UINT32_MAX`, with no-grad or legal
  gradient metadata and exact mask/next-state outputs.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>

Remain explicitly blocked:

- all [graph](../../../../glossary.md#graph)-random [distributions](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) other than dropout because no such [Model](../../../../glossary.md#term-model) [operation](../../../../glossary.md#operation) or wire exists;
- empty/dynamic Shapes, affine views, aliases, over-limit geometry, strict dropout, FLOAT64/BFLOAT16
  dropout, attention dropout, hidden/default/session RNG, public algorithm selection, entropy,
  cryptographic/statistical claims, split/fold-in/jump, serialization, and cross-backend/version
  bitstream identity;<span data-glossary-navigation="true"> ([views glossary](../../../../glossary.md#term-view-meanings); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [FLOAT64 glossary](../../../../glossary.md#term-float64); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [serialization glossary](../../../../glossary.md#term-serialization); [backend glossary](../../../../glossary.md#backend); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- all `RNN_TANH`, `GRU_RESET_AFTER`, and [`LSTM`](../../../../glossary.md#term-long-short-term-memory-lstm) occurrences, both directions, both bias forms, every
  type/profile/layout, every special case, and every recurrent gradient; and<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias))</span>
- direct MPSGraph random/recurrent selectors, host sampling/repair, runtime mode switches, timing,
  autotuning, benchmark choice, cache winners, retry, and fallback.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [host glossary](../../../../glossary.md#term-host); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [cache glossary](../../../../glossary.md#term-cache))</span>

If implementation needs another [carrier](../../../../glossary.md#term-storage-carrier), profile, [layout](../../../../glossary.md#layout), probability [narrowing](../../../../glossary.md#term-narrowing), [distribution](../../../../glossary.md#term-distribution),
[schema](../../../../glossary.md#term-schema) field, [ABI](../../../../glossary.md#term-application-binary-interface-abi) export, 64-bit grid, [mutable](../../../../glossary.md#term-mutable) [RNG state](../../../../glossary.md#graph-rng-state--graphrngstate), recurrent special case, elementary
assumption, [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) formula, or [runtime](../../../../glossary.md#term-runtime) route choice, stop for a new independently reviewed plan rather
than narrowing or silently changing this one.

## Non-goals

No [Model](../../../../glossary.md#term-model), [Compiler](../../../../glossary.md#term-compiler), public [Tensor](../../../../glossary.md#tensor), [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), Config, [Planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Trace](../../../../glossary.md#trace), or [Engine](../../../../glossary.md#term-engine) [API](../../../../glossary.md#term-application-programming-interface-api)
change; no [eager](../../../../glossary.md#term-eager-evaluation) `TensorRandoms` change; no [operation](../../../../glossary.md#operation)/attribute/type/route wire; no [ABI](../../../../glossary.md#term-application-binary-interface-abi) export; no
portable or public [RNG](../../../../glossary.md#graph-rng-state--graphrngstate) algorithm; no general affine or empty [kernel](../../../../glossary.md#kernel); no attention [dropout](../../../../glossary.md#dropout--inverted-dropout); no
recurrent execution or [gradient](../../../../glossary.md#term-gradient); no benchmark; no selector rehabilitation; and no rewrite of
historical blocked evidence. Task 0065 supersedes only the future-route conclusion of historical
Task 0035 by choosing a separately specified custom RNG route. Task 0037 remains Blocked.

## Contracts

- [`ARCHITECTURE.md` — Core lifecycle and invariants](../../../../../ARCHITECTURE.md#core-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
  — Model owns profile-indexed meaning; backend support is fail-closed and route choice is cold.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
  — dropout/recurrent formulas recurse through exact guards and primitive/aggregate/elementary
  floors; masks, state, traversal, and publication remain exact.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [masks glossary](../../../../glossary.md#term-boolean-mask); [publication glossary](../../../../glossary.md#term-publication))</span>
- [Fixed recurrent scan](../../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions)<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
  — exact signatures, directions, valid lengths, traversal, padding, states, and BPTT boundary.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [padding glossary](../../../../glossary.md#padding); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt))</span>
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>
  and [Numerical-profile backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity)<span data-glossary-navigation="true"> ([backend identity glossary](../../../../glossary.md#backend-identity--backendid))</span>
  — truthful occurrence capability, fixed routes, private identity, and no fallback.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>.
- [Model](../../../../glossary.md#term-model) [explicit graph RNG state](../../../modules/model/tasks/0019b-explicit-graph-rng-state-foundation.md)<span data-glossary-navigation="true"> ([graph RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>,
  [dropout](../../../modules/model/tasks/0019b1-explicit-graph-dropout-construction.md), and
  [fixed recurrent scans](../../../modules/model/tasks/0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md)
  own operation meaning and public construction.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- [Compiler](../../../../glossary.md#term-compiler) [stochastic gradients](../../../modules/compiler/tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient))</span>
  and [recurrent forward/BPTT boundary](../../../modules/compiler/tasks/0006a-fixed-recurrent-scan-forward-adoption-and-bptt-boundary.md)
  own mask reuse and fail-closed recurrent backward behavior.
- [CPU](../../../../glossary.md#term-cpu) [explicit-state RNG/dropout](../../cpu/tasks/0006d-portable-explicit-state-rng-and-dropout.md)<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
  is exact private precedent only; its route and identity do not authorize Metal.

## Dependencies and integration

- Depends on: Task 0064 Complete through `3df362e1`; current Model/Compiler contracts named above;
  completed Tasks 0055/0056/0057/0058/0059/0063 multi-type, generic-cardinality, custom-program,
  BOOL/WHERE, primitive, layout, INT64, multi-output, lifecycle, package, and public Engine
  foundations; retained Tasks 0035/0037 evidence; and Task 0053's current elementary proof blocker.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [cardinality glossary](../../../../glossary.md#term-cardinality); [BOOL glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [INT64 glossary](../../../../glossary.md#data-type--datatype); [lifecycle glossary](../../../../glossary.md#lifecycle); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Supersedes only historical Task 0035's future-route conclusion. Its direct MPSGraph/Philox
  mismatch remains evidence. Task 0037 and Task 0053 remain Blocked.<span data-glossary-navigation="true"> ([MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [Philox glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Conflicts with: every concurrent Metal capability/schema/native/custom-source/catalog/candidate/
  route/preparation/package/shared-document scope, random/recurrent work, and resumed Task-0053
  production.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [schema glossary](../../../../glossary.md#term-schema); [scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None.
- Common base revision: `3df362e1e2acda3d1c333c34773975977068e9d4`.
- Integration order: this independent plan review, then one serial RNG/dropout proof, kernel,
  capability, preparation, execution, test, and documentation cutover; no production edit before
  plan approval.<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [kernel glossary](../../../../glossary.md#kernel))</span>
- Integration validation: native build/sign/package/verification, complete Metal, focused Compiler,
  conformance, source-backed Model/Compiler contract tests, CPU-free public Engine forward/backward/
  replay/negative evidence, Javadoc/architecture/docs, exact count/identity/export audits, and
  independent cumulative Class C review.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): Task-0065 coordinator.

## Files and symbols

Implementation [owners](../../../../glossary.md#term-owner):

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNegPartitionPreparer`,
  `MetalNegPreparationPlan`, `MetalNativeApi`, `MetalOperationRouteCatalog`, custom route candidate/
  tuning/codec identity owners, and affected package/type Javadocs;<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- native foundation [operation](../../../../glossary.md#operation)/[preflight](../../../../glossary.md#term-preflight)/[metadata](../../../../glossary.md#term-metadata)/pipeline/step/[binding](../../../../glossary.md#term-binding) code plus new
  `synaptik_task0065_rng_dropout_kernels.h`, with no new export or [schema](../../../../glossary.md#term-schema) field;
- focused capability/catalog/schema/zero-feed/malformed/prepared/native/raw-word/result-set tests,
  including allocation-free exact-`UINT32_MAX` and one-past/product/span rejection parity;
- existing Model/Compiler stochastic and recurrent contract suites, Metal conformance, and a
  dedicated CPU-free Engine Metal integration for forward, replay, chaining, saved-mask backward,
  training/evaluation distinction, reuse, concurrency, sessions, and exclusions; and<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [CPU glossary](../../../../glossary.md#term-cpu); [mask glossary](../../../../glossary.md#term-boolean-mask); [evaluation glossary](../../../../glossary.md#term-evaluation); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Metal/backend/native guides, public API scope/capability/preparer status, targeted glossary and
  architecture identity claims, this brief, master plan, and roadmap after behavior stabilizes.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [scope glossary](../../../../glossary.md#term-scope))</span>

No [Compiler](../../../../glossary.md#term-compiler) or [Model](../../../../glossary.md#term-model) production edit was required. The original [planning](../../../../glossary.md#term-planning)-only revision changed
exactly this brief, the [Metal](../../../../glossary.md#term-metal) master plan, and the roadmap; implementation then changed only the
approved Metal Java/native/test, public [Engine](../../../../glossary.md#term-engine) integration, and current documentation scopes.

## Acceptance criteria

1. The five-row audit and tests retain exact wires, attributes, arities, directions, carriers,
   profiles, layouts, gradient metadata, output roles, probability/state bits, Shape/UINT32/native-
   size bounds, and blocker decisions. Eager distributions remain leaf factories, not graph rows.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [gradient glossary](../../../../glossary.md#term-gradient); [metadata glossary](../../../../glossary.md#term-metadata); [Shape glossary](../../../../glossary.md#shape); [UINT32 glossary](../../../../glossary.md#index-mappings-and-cost-dominance); [Eager glossary](../../../../glossary.md#term-eager-evaluation); [distributions glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [leaf glossary](../../../../glossary.md#term-leaf); [graph glossary](../../../../glossary.md#graph))</span>
2. Exactly wires 101 and 102 become structurally executable/custom-available and occurrence-capable
   only in the domains above. Wires 103..105 and every excluded occurrence reject before resource
   creation.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
3. V1 word [vectors](../../../../glossary.md#term-vector), the exact `ceil(p * 2^53)` integer threshold and boundary vectors, canonical
   mask, positive-zero drops, post-binary64-subtraction FLOAT32 complement and accelerator scaling,
   one-draw-per-logical-ordinal mapping, modulo advancement, scalar behavior, replay, branching,
   chaining, reuse, sessions, and concurrency match independent oracles. No hidden mutable state
   exists.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [mask glossary](../../../../glossary.md#term-boolean-mask); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [FLOAT32 glossary](../../../../glossary.md#term-float32); [accelerator glossary](../../../../glossary.md#term-accelerator); [scalar glossary](../../../../glossary.md#term-scalar); [concurrency glossary](../../../../glossary.md#term-concurrency); [mutable glossary](../../../../glossary.md#term-mutable))</span>
4. Zero-feed initializer creation and ordered five-boundary dropout binding work without dummy
   values. Every output is distinct, hidden mask/next state remain live, complete overlaps reject
   before mutation, and one next-state write occurs per invocation.<span data-glossary-navigation="true"> ([dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [binding glossary](../../../../glossary.md#term-binding))</span>
5. Training dropout always samples; evaluation constructs no dropout row and consumes no state.
   First-order backward consumes the exact same-occurrence mask, never resamples, and keeps state and
   mask non-differentiable.<span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [evaluation glossary](../../../../glossary.md#term-evaluation); [differentiable glossary](../../../../glossary.md#derivative-and-chain-rule))</span>
6. Every admitted [partition](../../../../glossary.md#partition) has only `CUSTOM_PROGRAM`; MPSGraph family selectors, host sampling or
   repair, retry, fallback, timing, autotuning, payload decisions, and cache winners are absent.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [sampling glossary](../../../../glossary.md#term-sampling-machine-learning-and-hyperparameters); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [cache glossary](../../../../glossary.md#term-cache))</span>
7. Counts become `83/32` capability, `101/14` structural, `75/35/5` [MPSGraph](../../../../glossary.md#term-mpsgraph), and `60/55/0` custom;
   schema 15, ABI 5, and thirteen exports remain; all backend-local identities are 21 and identity
   20 rejects.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [backend glossary](../../../../glossary.md#backend); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
8. Direct packaged-native/prepared-Metal evidence proves raw initializer publication under both
   profiles. CPU-free packaged-Metal Engine proves accelerator initializer use through<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [publication glossary](../../../../glossary.md#term-publication); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>
   `INITIAL_STATE -> DROPOUT`, forward raw special values, repeat/chained state behavior, gradient
   mask reuse, accelerator-only dropout, prepared reuse, concurrent/independent sessions, and every
   named public early-failure boundary without skip or CPU fallback.
9. RNN/GRU/LSTM both directions, bias variants, valid-length payloads, positive-zero padding, final
   states, elementary sites, zero/no-work special cases, and all gradients remain visibly
   fail-closed; schema vocabulary is never reported as recurrent execution.<span data-glossary-navigation="true"> ([RNN glossary](../../../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [bias glossary](../../../../glossary.md#term-bias); [padding glossary](../../../../glossary.md#padding); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
10. Native/package, complete Metal, focused Compiler, conformance, dedicated Engine, Javadoc,
    architecture, Markdown/link/diff checks pass; independent cumulative Class C review has no
    unresolved P0/P1/P2.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [conformance glossary](../../../../glossary.md#term-conformance); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## [Validation](../../../../glossary.md#term-validation) after implementation

```bash
./native/metal-macos-arm64/build.sh
codesign --force --sign - --identifier io.github.pho001.synaptik.metal.foundation native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/package-local.sh native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
./native/metal-macos-arm64/verify-package.sh native/metal-macos-arm64/build/package-v1/macos-arm64
./gradlew :backends:metal:verifyMetalNativePackage :backends:metal:metalNativeLocalZip \
  -PsynaptikMetalNativePackage="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64"
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test
./gradlew :modules:compiler:test --tests '*Dropout*' --tests '*Stochastic*' --tests '*Recurrent*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*RandomDropoutMetalIntegrationTest*'
./gradlew :backends:metal:javadoc :testing:architecture-tests:test
./gradlew test
./gradlew build
git diff --check
```

No timing, benchmark, or random-quality command is authorized. Complete packaged [Metal](../../../../glossary.md#term-metal), focused
[Compiler](../../../../glossary.md#term-compiler) contracts, [conformance](../../../../glossary.md#term-conformance), public [Engine](../../../../glossary.md#term-engine), architecture/[Javadoc](../../../../glossary.md#term-javadoc), and repository validation
close the implementation.

## Documentation and review impact

Implementation must update every current capability/count/[identity](../../../../glossary.md#term-identity-meanings)/random/[dropout](../../../../glossary.md#dropout--inverted-dropout)/recurrent
statement and describe only the exact bounded domains above. Public/[backend](../../../../glossary.md#backend)/native/[Javadoc](../../../../glossary.md#term-javadoc) wording
must name [graph](../../../../glossary.md#graph) versus [eager](../../../../glossary.md#term-eager-evaluation) randomness, raw state and V1 private identity, threshold and scaling
policies, no entropy/portable stream claim, train/[evaluation](../../../../glossary.md#term-evaluation) distinction, [gradient](../../../../glossary.md#term-gradient) [mask](../../../../glossary.md#term-boolean-mask) reuse,
zero-feed and multi-output [lifecycle](../../../../glossary.md#lifecycle), fixed custom route, and all recurrent blockers. Historical
tasks remain [immutable](../../../../glossary.md#term-immutable) evidence records.

Independent plan review precedes every production edit. Independent final review must inspect the
full plan-to-source proof, raw word and threshold arithmetic, probability conversion and scaling
sites, counter wrap, logical ordinal mapping, [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) bytes, zero-feed safety, output
[liveness](../../../../glossary.md#term-liveness)/[alias](../../../../glossary.md#aliasing-and-arena) rejection, single-writer state, session/[concurrency](../../../../glossary.md#term-concurrency) isolation, route exclusivity,
identity invalidation, generated-gradient mask identity, recurrent [fail-closed](../../../../glossary.md#term-fail-closed) boundary, public
[Engine](../../../../glossary.md#term-engine) behavior, documentation, and changed-path [scope](../../../../glossary.md#term-scope). Passing worker evidence may be reused unless
remediation changes executable behavior.

## [Planning](../../../../glossary.md#term-planning) result

The audited maximal cutover is two fixed custom explicit-state [RNG](../../../../glossary.md#graph-rng-state--graphrngstate) rows, not a generic
random-[distribution](../../../../glossary.md#term-distribution) or recurrent implementation. Both-profile zero-input `INITIAL_STATE` publishes
raw key/counter bits; [accelerator](../../../../glossary.md#term-accelerator) [canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout) publishes value, saved [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask), and
next state using the private `SYNAPTIK_METAL_SPLITMIX64_COUNTER_V1` replay boundary. Exact threshold,
scaling, counter wrap, [target](../../../../glossary.md#term-target)-subset [liveness](../../../../glossary.md#term-liveness), saved-mask backward, [evaluation](../../../../glossary.md#term-evaluation) bypass, reuse, and
session/[concurrency](../../../../glossary.md#term-concurrency) isolation are implemented and proven without [Compiler](../../../../glossary.md#term-compiler) or [Model](../../../../glossary.md#term-model) production
changes.

Production is now `83/32` capability and `101/14` structural execution; catalogs are `75/35/5`
[MPSGraph](../../../../glossary.md#term-mpsgraph) and `60/55/0` custom. [Schema](../../../../glossary.md#term-schema) 15, [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, and thirteen exports remain fixed; all
[backend](../../../../glossary.md#backend)-local [identities](../../../../glossary.md#term-identity-meanings) are 21 and [identity](../../../../glossary.md#term-identity-meanings) 20 fails closed. Recurrent execution remains empty
pending a complete [TANH](../../../../glossary.md#term-tanh)/[EXP](../../../../glossary.md#term-expx)/[SIGMOID](../../../../glossary.md#term-sigmoid) proof and Compiler [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) decision.
