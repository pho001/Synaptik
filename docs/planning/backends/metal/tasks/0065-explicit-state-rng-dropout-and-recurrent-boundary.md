# Task 0065: Explicit-State RNG, Dropout, and Recurrent Boundary

## Status

Complete — approved plan `9ceee2b53c88a8c87ca4b8a42cdd2c92682074c9` was implemented by
`05d83074` plus correctness remediation `ed7a3369`. The cutover adds exact both-profile
`INITIAL_STATE` and accelerator FLOAT32 `DROPOUT`, while eager distributions and all recurrent rows
remain fail-closed. Native package, complete Metal, focused Compiler, conformance, public Engine,
Javadoc/architecture, full repository test/build, and documentation/diff validation passed after
remediation. Independent cumulative code/evidence/documentation and security/determinism reviews
returned `APPROVE` with zero unresolved P0/P1/P2.

## Change class

Class C — the implementation adds the first Metal explicit-state random execution, zero-feed
native creation, one genuine three-output custom operation, saved-mask gradient use, private
generator identity, and public Engine behavior. It advertises no recurrent row: all three recurrent
kinds remain fail-closed for explicit elementary-function, recurrence, and BPTT reasons.
Independent cumulative code/evidence/documentation and security review was the final completion
gate and returned `APPROVE` after correctness remediation.

## Goal

Implement the maximal deterministic domain currently provable from Model and Compiler contracts for
`INITIAL_STATE` and explicit-state `DROPOUT`: exact raw key/counter materialization under both
profiles and accelerator FLOAT32 dropout through one versioned Metal-private counter algorithm and
one fixed custom-program route. Preserve exact draw membership, canonical mask, positive-zero drop,
counter advancement, replay, branching, chaining, concurrent-session isolation, saved-mask
backward use, and evaluation bypass. Route selection must occur before resource creation and may
use no timing, autotuning, cache winner, input value, retry, fallback, hidden generator, or host
repair.

Audit `RNN_TANH`, `GRU_RESET_AFTER`, and `LSTM` completely, including both directions, optional
bias, runtime valid lengths, output states, recursive numerical sites, and Compiler BPTT. Do not
make any recurrent occurrence production-capable without the missing all-domain elementary proof
and gradient boundary. Audit eager normal, uniform, integral, and Bernoulli factories, but do not
invent Metal graph operations for host-created leaf values.

## Scope

### Exact audited 115-kind registry boundary

The current registry has exactly five operation rows in scope:

| Wire/kind | Arity and attribute wire | Exact Model role | Pre-cutover Metal state | Task-0065 decision |
|---|---|---|---|---|
| 101 `DROPOUT` | 2 inputs / 3 outputs; 33 `DROPOUT` | `[value,state] -> [output,keepMask,nextState]`; one raw binary64 probability word | non-executable; MPSGraph `UNAVAILABLE / MU_RNG`; custom `PENDING / CP_STATE` | accelerator FLOAT32 custom route |
| 102 `INITIAL_STATE` | 0 inputs / 1 output; 34 `GRAPH_RNG_STATE` | exact raw `INT64 Shape[2]` key/counter materialization | non-executable; MPSGraph `UNAVAILABLE / MU_RNG`; custom `PENDING / CP_STATE` | exact both-profile custom route |
| 103 `RNN_TANH` | 5..6 inputs / 2 outputs; 35 `RECURRENT_DIRECTION` | fixed tanh recurrence, optional input-side bias, dense outputs and final hidden | non-executable; MPSGraph `UNAVAILABLE / MU_RECURRENT`; custom `PENDING / CP_STATE` | remain blocked |
| 104 `GRU_RESET_AFTER` | 5..6 inputs / 2 outputs; 35 `RECURRENT_DIRECTION` | reset/update/candidate recurrence, optional packed input-side bias | same unavailable/pending state | remain blocked |
| 105 `LSTM` | 6..7 inputs / 3 outputs; 35 `RECURRENT_DIRECTION` | input/forget/candidate/output recurrence, hidden and cell final states | same unavailable/pending state | remain blocked |

Attribute wire 33 contains exactly one raw `double` word and accepts every finite numerical value in
`[0,1)`, including both signed zeros. Wire 34 contains exactly the raw key then counter words and
accepts every 64-bit pattern. Wire 35 contains exactly `FORWARD=1` or `REVERSE=2`. Schema 15 already
encodes zero inputs, variable input counts, three outputs, all six carrier types, and the complete
attribute grammar. Type wires remain FLOAT32 `1`, INT32 `2`, BOOL `3`, FLOAT64 `4`, BFLOAT16 `5`,
and INT64 `6`. These existing wires are representation, not capability.

The current native signature tables accept numeric operation positions 101..105, but the native
operation enum intentionally jumps from 100 to 106 and has no executable selector constants for
these five positions. Task 0065 may name native custom constants only for existing wires 101 and
102. It must not add a Model kind, operation wire, attribute wire, data type, route wire, public RNG
algorithm enum, or recurrent alias.

### Random-distribution inventory and ownership

The graph-random inventory is exactly `INITIAL_STATE` plus `DROPOUT`. There is no graph operation
for normal, general uniform, bounded integral, or Bernoulli sampling in the 115-kind registry.
`TensorRandoms` instead exposes eager `randomNormal`, `randomUniform`, two `randomInt` overloads,
and `randomBernoulli`. Each consumes a caller-owned `RandomGenerator` on the host and returns a
provenance-free leaf. Metal may receive the resulting represented values through ordinary ingress;
it does not own their generator, distribution mapping, seed, counter, replay, or sampling.

The eager mappings are exact host-factory contracts, not candidate Metal distributions:

| Factory | Caller-generator consumption and mapping | Result carrier conversion |
|---|---|---|
| `randomNormal` | one `nextGaussian()` per logical row-major element; binary64 multiplication `gaussian * standardDeviation` followed by binary64 addition of `mean`, with no fused site | direct FLOAT64, one narrowing conversion for FLOAT32, or FLOAT32 followed by BFLOAT16 conversion |
| `randomUniform` | one unchanged `nextDouble(lowerBound, upperBound)` call per logical row-major element | direct FLOAT64, one narrowing conversion for FLOAT32, or FLOAT32 followed by BFLOAT16 conversion |
| `randomInt` with `int` bounds | one unchanged `nextInt(origin, bound)` call per logical row-major element | direct INT32 |
| `randomInt` with `long` bounds | one unchanged `nextLong(origin, bound)` call per logical row-major element | direct INT64 |
| `randomBernoulli` | one `nextDouble()` per logical row-major element; true exactly when the draw is less than the closed-domain probability | canonical BOOL |

Task 0065 does not reproduce, wrap, seed, or promise any of those caller-generator sequences.

Task 0065 therefore adds no entropy source and makes no generic distribution claim. The caller's
raw graph key is the only seed/domain word for dropout, and the raw graph counter is the next
abstract logical draw position. Neither word is normalized, randomized, secret, or implicitly
initialized. The implementation is not cryptographic and makes no distinct-key stream-separation,
statistical certification, cross-backend bitstream, cross-version replay, or serialization promise.

### Exact initial-state production domain

Admit `GraphRngKind.INITIAL_STATE` under both `STRICT_IEEE` and `ACCELERATOR` only when the sole
output is canonical dense, zero-offset, non-view, non-gradient INT64 `Shape.of(2)`. The kernel
writes key bits to logical lane zero and counter bits to lane one exactly. It performs no hash,
draw, key expansion, counter advancement, host sampling, or persistent-state mutation.

A one-node initializer is a legal zero-feed/one-target program. Java program validation, native
image validation, custom-program creation, resource declaration, finalization, binding, and
invocation must accept that topology without a dummy input. Native creation must permit
`feed_count == 0` with the corresponding pointer absent while retaining positive value, node, and
target counts. Every generic Java/native node walk must consume the declared input/output arrays
and must not call a first-input accessor on the zero-input row.

### Metal-private dropout generator

Select the immutable backend-private configuration
`SYNAPTIK_METAL_SPLITMIX64_COUNTER_V1`. It deliberately uses the same reviewed integer permutation
as the CPU-private precedent but has a distinct Metal identity and creates no portable bitstream
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

For fixed key, `mix64` is a permutation of 64-bit words; no mutable stream cursor is needed. Logical
row-major ordinal `i`, not physical address, threadgroup, grid decomposition, dispatch order, or
session identity, selects the draw. The kernel computes its word directly from explicit input state
and `i`. Scheduling and concurrent invocation therefore cannot change a result.

Mask membership uses the exact 53-bit numerator `n = word >>> 11`, in
`[0, 2^53 - 1]`. For the numerical value of the raw binary64 probability bits, define the
normative unsigned-64 metadata value
`threshold(p) = ceil(p * 2^53)`, evaluated exactly rather than through floating-point
multiplication. Both signed zeros map to zero; every positive probability below `2^-53` maps to
one; and the complete legal domain maps into `[0, 2^53 - 1]`. Keep exactly when
`n >= threshold(p)`, including equality. This is exactly the Model predicate
`uniform53 >= probability`.

Preparation derives the threshold by integer decomposition of the raw binary64 sign, exponent, and
significand, without floating-point rounding. Native validation independently recomputes the same
integer from the raw bits and authenticates the supplied value. No FLOAT32 narrowing participates
in membership. Signed probability zero creates an all-true mask but still consumes every draw.
Each mask byte is exactly zero or one. Independent threshold oracles include:

| Probability raw bits | Exact value class | `threshold(p)` |
|---|---|---|
| `0000000000000000` | positive zero | `0` |
| `8000000000000000` | negative zero | `0` |
| `0000000000000001` | least positive binary64 subnormal | `1` |
| `3c90000000000000` | `2^-54` | `1` |
| `3ca0000000000000` | `2^-53` | `1` |
| `3ca0000000000001` | next binary64 value above `2^-53` | `2` |
| `3fe0000000000000` | `1/2` | `2^52` |
| `3fefffffffffffff` | largest binary64 value below one | `2^53 - 1` |

Independent oracle tests must derive, without production helpers, at least these fixed word vectors:

| Key | Counter | Word | Top 53 bits |
|---|---|---|---|
| `0000000000000000` | `0000000000000000` | `48218226ff3cd4bf` | `09043044dfe79a` |
| `0000000000000000` | `0000000000000001` | `ea8568d2e45fd6cb` | `1d50ad1a5c8bfa` |
| `0000000000000001` | `0000000000000000` | `dce423fc82c0d5b8` | `1b9c847f90581a` |
| `ffffffffffffffff` | `ffffffffffffffff` | `e8ba9f99ca933538` | `1d1753f3395266` |
| `0000000000001234` | `0000000000000007` | `3e4cf5a0c9489779` | `07c99eb4192912` |

The generator name, constants, integer threshold mapping, FLOAT32 scaling policy, mask policy, and
state-write policy are part of backend-local identity 21. A later algorithm or policy must advance
identity; it must not silently reuse V1.

### Exact dropout production domain

Admit `DropoutKind.DROPOUT` only under `ACCELERATOR` when all of the following hold:

- inputs are ordered `[value,state]`; outputs are ordered `[output,keepMask,nextState]`; all five
  graph values are distinct;
- value and output are FLOAT32 with the same fully static Shape and equal gradient eligibility;
  keep mask is canonical BOOL with that exact Shape and no gradient; state and next state are
  non-gradient INT64 `Shape.of(2)`;
- every boundary has canonical dense, zero-offset, non-view layout. The Shape rank is `0..16`, every
  present extent is in `1..UINT32_MAX`, and the logical element count is in `1..UINT32_MAX`. Scalar
  rank zero therefore consumes one draw; dynamic and zero-extent Shapes remain false;
- every element count, stride, logical ordinal, grid coordinate, byte count, byte offset, physical
  span, and native `NSUInteger` conversion passes checked Java and native bounds before pipeline or
  buffer creation; and
- the probability is the complete Model domain of finite binary64 values in `[0,1)`, including both
  signed zeros. Capability must not select a fixed probability, key, counter, Shape, corpus, or
  mask outcome.

For each kept value, first derive the complement bits exactly as the FLOAT32 conversion of the
binary64 expression `1.0d - probability`; the probability itself must not be narrowed before that
subtraction. Then execute the current Model accelerator formula at the remaining typed sites:
FLOAT32 one divided by that typed complement, followed by FLOAT32 input multiplication. The raw
binary64 attribute remains the exact membership leaf. Use the already qualified accelerator
primitive rules, including DAZ/FTZ, and preserve the sites and association. Do not pre-narrow the
probability, replace the formula with input division, use a reciprocal approximation, fuse the
algebra, reduce precision, or accept a final-output tolerance. Java and native metadata must carry
the raw probability bits, exact integer threshold, and derived FLOAT32 complement bits for
independent validation.

A dropped value is exact positive FLOAT32 zero even for negative zero, NaN, or infinity. A kept
signed zero and infinity retain the formula-required sign; kept NaN remains NaN without a payload
promise. Ordinary finite, subnormal, overflow, underflow, signed-zero, NaN, and infinity outputs are
tested against an independent three-site oracle, not CPU's older finite-order emitter. Probability
values immediately around binary64 and binary32 threshold/conversion boundaries, including the
largest value below one, are mandatory cases.

Load key and counter from state lanes zero and one. The next state writes the exact unchanged key
and `counter + N mod 2^64`, where `N` is the static logical element count. Advancement is independent
of probability, input values, mask, value type, execution order, grid, session, and gradient use.
Counter wrap is valid. Branching the same state intentionally reuses an interval; chaining the
returned next state selects the subsequent interval.

`STRICT_IEEE` FLOAT32 remains false because current Metal primitive arithmetic does not preserve the
complete strict subnormal/rounding set. FLOAT64 dropout remains false because the GPU route has no
proved FLOAT64 arithmetic. BFLOAT16 remains false because it has no accelerator relaxation and no
direct correctly rounded scaling/conversion proof. Integral and BOOL values are Model-illegal.
General affine/view, unresolved, empty, negative/zero-stride, in-place, aliased, and over-limit
occurrences remain false.

### Fixed custom route, outputs, and lifecycle

Add one Task-0065 native source/header and append it unconditionally to the existing fixed
custom-program library. Add native custom operation constants only for wires 101 and 102. One
initializer thread writes the two raw words. One dropout thread owns each value/mask position; thread
zero additionally writes the two next-state words exactly once. No random-word buffer, replay
buffer, mask staging buffer, scratch, atomic, reduction pass, persistent counter, per-thread
generator, host sampling/output/mask/state calculation or repair, or second route exists. Cold Java
derivation of the authenticated threshold and complement metadata above is permitted; it is not
execution or sampling.

Before dispatch, cold Java and native validation independently authenticate kind, attribute bits,
cardinality, ordered value roles, type, Shape, canonical layout, gradient flags, UINT32 and byte
bounds, value identity, resource access, and exact metadata. Reject every output span against both
input spans and reject every output/output pair before mutation. Input/input overlap may remain
legal because both are read-only, but it creates no semantic alias. A failed validation writes no
state, mask, or value.

Every partition containing wire 101 or 102 selects `CUSTOM_PROGRAM` only. It may compose the random
kernels with already approved custom or nested MPSGraph steps in stable topological order, including
`INITIAL_STATE -> DROPOUT`, chained dropouts, and generated backward consumers. Candidate generation
must never offer MPSGraph for either row. Route choice never consults probability, key/counter,
input payloads, timing, benchmark data, cache winners, runtime failure, or retry.

Prepared executables remain immutable and reusable. Each bound invocation owns ordinary run-local
buffers; no RNG field enters session, `RunState`, Runtime, Trace, or the prepared recipe. Equal
explicit inputs and the same identity/configuration replay equal mask, output, and next state.
Repeated calls in one session, independent sessions, and concurrent invocations have no counter
interaction. Close, lease, rollback, suppression, tracing, and synchronous invocation retain the
existing lifecycle.

### Training, evaluation, saved mask, and gradients

`DROPOUT` always denotes training dropout. There is no Metal training flag and no evaluation kernel.
The current NN evaluation path returns the original value and state without constructing wire 101,
so it consumes no draw and advances no state. Tests must prove this graph-level absence rather than
turn probability into zero or dispatch a disabled dropout.

Compiler already captures all three forward outputs and constructs the sole first-order input
cotangent from the exact same-occurrence slot-one mask:

```text
dInput = where(mask, gradient / typed(1.0d - probability), positiveZero)
```

Task 0065 changes no Compiler formula. The forward kernel must publish mask and next state as
ordinary live internal values even when only output slot zero is public. Resource planning keeps the
mask alive until the generated backward `WHERE` consumes it and never resamples, reconstructs, or
infers it from output values. State input/output and mask are non-differentiable.

Admit legal `requiresGrad=true` FLOAT32 forward metadata as well as no-grad metadata. End-to-end
backward succeeds only when the existing accelerator scalar-DIV, exact BOOL `WHERE`, zero constant,
reduction/seed, and any requested surrounding graph are independently capable. Public Engine proof
must exercise a bounded primitive-closed dropout gradient and verify the selected mask, positive-zero
dropped cotangent, typed scale, and absence of state differentiation. Unsupported strict or carrier
cases fail during Planning, before native resource creation.

### Exact recurrent contract audit

All three recurrent rows consume a time-major floating input `[T,N,I]`, runtime non-gradient INT64
valid lengths `[N]`, initial hidden `[N,H]`, packed input weight `[G*H,I]`, packed hidden weight
`[G*H,H]`, and optional packed input-side bias `[G*H]`; LSTM additionally consumes initial cell
`[N,H]`. Gate count `G` is one, three, or four. RNN/GRU output `[outputs[T,N,H],finalHidden[N,H]]`;
LSTM additionally outputs `finalCell[N,H]`. `FORWARD` traverses `0..L-1`; `REVERSE` traverses the
valid prefix `L-1..0`. Every padded output is exact positive zero, zero length returns the exact
initial state, and the complete length vector must validate in `[0,T]` before any output mutation.
Both directions and biased/bias-free signatures are current; bidirectionality, stacking,
configurable activations, projection, peepholes, recurrent dropout, and graph regions are absent.

The current schema-15 cardinality/type/attribute image can represent every recurrent signature, and
current generic multi-output publication can represent its sibling states. Those historical Task-
0037 representation blockers are resolved. They do not create execution support. The recurrent
production subset remains empty for independent reasons:

1. installed MPSGraph recurrent selectors have no Model runtime-length input, atomic all-length
   validation, skipped padded recurrence, exact positive-zero padding contract, or exact final-state
   output roles. Training intermediates are not substitutes;
2. no current native custom recurrent loop validates all lengths before writing and publishes the
   exact two/three output sets for both directions and optional bias;
3. `RNN_TANH` requires a complete accelerator FLOAT32 `TANH` site proof. The direct historical
   route has invalid negative-zero/NaN behavior, and custom `TANH` remains pending without an
   all-binary32 recursive-floor certificate;
4. GRU and LSTM additionally require stable sigmoid branches and therefore the missing certified
   exponential bridge retained by Blocked Task 0053, plus their candidate/cell `TANH` sites; and
5. Compiler rejects every recurrent occurrence in `FORWARD_AND_BACKWARD` and `TRAINING_STEP` before
   derivative allocation because BPTT, saved-gate policy, and recurrent gradient formulas do not
   exist.

Do not advertise a zero-time, zero-batch, all-zero-length, single-cell, fixed-length, selected-value,
or full-length-only special case. Capability cannot inspect valid-length payloads, and a no-work
special case would evade rather than implement the recurrence. FLOAT64, BFLOAT16, strict FLOAT32,
dynamic Shapes, views, in-place execution, all gradients, and every recurrent row remain false.
Unblocking requires a separately reviewed task after both a complete elementary proof and the
Compiler BPTT decision; schema presence and sampled device output are insufficient.

### Catalog, schema, identity, and lifecycle deltas

- Mark exactly wires 101 and 102 structurally executable: `99/16 -> 101/14`. Wires 103..105 remain
  non-executable.
- Production capability becomes `81/34 -> 83/32`. `INITIAL_STATE` has one exact both-profile domain;
  `DROPOUT` has the accelerator FLOAT32 domain above. Recurrent kinds remain false.
- Keep the MPSGraph catalog exactly `75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE`; structural catalog
  state is not route authorization.
- Add custom reason `CA_0065`, change exactly the two RNG rows from pending to available, and produce
  custom catalog `60 AVAILABLE / 55 PENDING / 0 UNAVAILABLE-WITH-PROOF`. Recurrent rows retain their
  current pending entry and the explicit blockers above.
- Keep schema 15, ABI 5, type wires `1..6`, operation wires `1..115`, attribute wires `0..41`, route
  wires `1..3`, generic cardinality/image grammar, and exactly thirteen exports. Filling native
  selector names for existing positions 101 and 102 is not a new wire or export.
- Advance workload-signature, exact-default-policy, candidate-schema, compatibility-schema,
  route-policy, and codec identities together `20 -> 21`. Identity 20 and every older identity
  reject with no alias, migration reader, fallback library, or shim.
- Preserve stable value/feed/target order, transactional finalization, synchronous invocation,
  internal/output liveness, physical-span authentication, leases, reuse, concurrency/session
  isolation, close ordering, trace identity, and the profile-free Runtime/Trace boundary.

## Provable domains versus explicit blockers

The provable cutover is exactly:

- exact raw zero-input `INITIAL_STATE` under both profiles for every key/counter bit pair; and
- accelerator FLOAT32 `DROPOUT` over the complete binary64 probability and explicit-state domains,
  canonical positive static rank-`0..16` Shapes bounded by `UINT32_MAX`, with no-grad or legal
  gradient metadata and exact mask/next-state outputs.

Remain explicitly blocked:

- all graph-random distributions other than dropout because no such Model operation or wire exists;
- empty/dynamic Shapes, affine views, aliases, over-limit geometry, strict dropout, FLOAT64/BFLOAT16
  dropout, attention dropout, hidden/default/session RNG, public algorithm selection, entropy,
  cryptographic/statistical claims, split/fold-in/jump, serialization, and cross-backend/version
  bitstream identity;
- all `RNN_TANH`, `GRU_RESET_AFTER`, and `LSTM` occurrences, both directions, both bias forms, every
  type/profile/layout, every special case, and every recurrent gradient; and
- direct MPSGraph random/recurrent selectors, host sampling/repair, runtime mode switches, timing,
  autotuning, benchmark choice, cache winners, retry, and fallback.

If implementation needs another carrier, profile, layout, probability narrowing, distribution,
schema field, ABI export, 64-bit grid, mutable RNG state, recurrent special case, elementary
assumption, BPTT formula, or runtime route choice, stop for a new independently reviewed plan rather
than narrowing or silently changing this one.

## Non-goals

No Model, Compiler, public Tensor, NN, Config, Planning, Prepare, Runtime, Trace, or Engine API
change; no eager `TensorRandoms` change; no operation/attribute/type/route wire; no ABI export; no
portable or public RNG algorithm; no general affine or empty kernel; no attention dropout; no
recurrent execution or gradient; no benchmark; no selector rehabilitation; and no rewrite of
historical blocked evidence. Task 0065 supersedes only the future-route conclusion of historical
Task 0035 by choosing a separately specified custom RNG route. Task 0037 remains Blocked.

## Contracts

- [`ARCHITECTURE.md` — Core lifecycle and invariants](../../../../../ARCHITECTURE.md#core-lifecycle)
  — Model owns profile-indexed meaning; backend support is fail-closed and route choice is cold.
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
  — dropout/recurrent formulas recurse through exact guards and primitive/aggregate/elementary
  floors; masks, state, traversal, and publication remain exact.
- [Fixed recurrent scan](../../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions)
  — exact signatures, directions, valid lengths, traversal, padding, states, and BPTT boundary.
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
  and [Numerical-profile backend identity](../../../../architecture/contracts/backend-execution.md#numerical-profile-backend-identity)
  — truthful occurrence capability, fixed routes, private identity, and no fallback.
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle).
- Model [explicit graph RNG state](../../../modules/model/tasks/0019b-explicit-graph-rng-state-foundation.md),
  [dropout](../../../modules/model/tasks/0019b1-explicit-graph-dropout-construction.md), and
  [fixed recurrent scans](../../../modules/model/tasks/0025e-fixed-recurrent-scan-semantic-family-and-tensor-expressions.md)
  own operation meaning and public construction.
- Compiler [stochastic gradients](../../../modules/compiler/tasks/0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)
  and [recurrent forward/BPTT boundary](../../../modules/compiler/tasks/0006a-fixed-recurrent-scan-forward-adoption-and-bptt-boundary.md)
  own mask reuse and fail-closed recurrent backward behavior.
- CPU [explicit-state RNG/dropout](../../cpu/tasks/0006d-portable-explicit-state-rng-and-dropout.md)
  is exact private precedent only; its route and identity do not authorize Metal.

## Dependencies and integration

- Depends on: Task 0064 Complete through `3df362e1`; current Model/Compiler contracts named above;
  completed Tasks 0055/0056/0057/0058/0059/0063 multi-type, generic-cardinality, custom-program,
  BOOL/WHERE, primitive, layout, INT64, multi-output, lifecycle, package, and public Engine
  foundations; retained Tasks 0035/0037 evidence; and Task 0053's current elementary proof blocker.
- Supersedes only historical Task 0035's future-route conclusion. Its direct MPSGraph/Philox
  mismatch remains evidence. Task 0037 and Task 0053 remain Blocked.
- Conflicts with: every concurrent Metal capability/schema/native/custom-source/catalog/candidate/
  route/preparation/package/shared-document scope, random/recurrent work, and resumed Task-0053
  production.
- Parallel group: None.
- Common base revision: `3df362e1e2acda3d1c333c34773975977068e9d4`.
- Integration order: this independent plan review, then one serial RNG/dropout proof, kernel,
  capability, preparation, execution, test, and documentation cutover; no production edit before
  plan approval.
- Integration validation: native build/sign/package/verification, complete Metal, focused Compiler,
  conformance, source-backed Model/Compiler contract tests, CPU-free public Engine forward/backward/
  replay/negative evidence, Javadoc/architecture/docs, exact count/identity/export audits, and
  independent cumulative Class C review.
- Shared-document integration owner: Task-0065 coordinator.

## Files and symbols

Implementation owners:

- `MetalCapabilityProvider`, `MetalMpsGraphProgram`, `MetalNegPartitionPreparer`,
  `MetalNegPreparationPlan`, `MetalNativeApi`, `MetalOperationRouteCatalog`, custom route candidate/
  tuning/codec identity owners, and affected package/type Javadocs;
- native foundation operation/preflight/metadata/pipeline/step/binding code plus new
  `synaptik_task0065_rng_dropout_kernels.h`, with no new export or schema field;
- focused capability/catalog/schema/zero-feed/malformed/prepared/native/raw-word/result-set tests,
  including allocation-free exact-`UINT32_MAX` and one-past/product/span rejection parity;
- existing Model/Compiler stochastic and recurrent contract suites, Metal conformance, and a
  dedicated CPU-free Engine Metal integration for forward, replay, chaining, saved-mask backward,
  training/evaluation distinction, reuse, concurrency, sessions, and exclusions; and
- Metal/backend/native guides, public API scope/capability/preparer status, targeted glossary and
  architecture identity claims, this brief, master plan, and roadmap after behavior stabilizes.

No Compiler or Model production edit was required. The original planning-only revision changed
exactly this brief, the Metal master plan, and the roadmap; implementation then changed only the
approved Metal Java/native/test, public Engine integration, and current documentation scopes.

## Acceptance criteria

1. The five-row audit and tests retain exact wires, attributes, arities, directions, carriers,
   profiles, layouts, gradient metadata, output roles, probability/state bits, Shape/UINT32/native-
   size bounds, and blocker decisions. Eager distributions remain leaf factories, not graph rows.
2. Exactly wires 101 and 102 become structurally executable/custom-available and occurrence-capable
   only in the domains above. Wires 103..105 and every excluded occurrence reject before resource
   creation.
3. V1 word vectors, the exact `ceil(p * 2^53)` integer threshold and boundary vectors, canonical
   mask, positive-zero drops, post-binary64-subtraction FLOAT32 complement and accelerator scaling,
   one-draw-per-logical-ordinal mapping, modulo advancement, scalar behavior, replay, branching,
   chaining, reuse, sessions, and concurrency match independent oracles. No hidden mutable state
   exists.
4. Zero-feed initializer creation and ordered five-boundary dropout binding work without dummy
   values. Every output is distinct, hidden mask/next state remain live, complete overlaps reject
   before mutation, and one next-state write occurs per invocation.
5. Training dropout always samples; evaluation constructs no dropout row and consumes no state.
   First-order backward consumes the exact same-occurrence mask, never resamples, and keeps state and
   mask non-differentiable.
6. Every admitted partition has only `CUSTOM_PROGRAM`; MPSGraph family selectors, host sampling or
   repair, retry, fallback, timing, autotuning, payload decisions, and cache winners are absent.
7. Counts become `83/32` capability, `101/14` structural, `75/35/5` MPSGraph, and `60/55/0` custom;
   schema 15, ABI 5, and thirteen exports remain; all backend-local identities are 21 and identity
   20 rejects.
8. Direct packaged-native/prepared-Metal evidence proves raw initializer publication under both
   profiles. CPU-free packaged-Metal Engine proves accelerator initializer use through
   `INITIAL_STATE -> DROPOUT`, forward raw special values, repeat/chained state behavior, gradient
   mask reuse, accelerator-only dropout, prepared reuse, concurrent/independent sessions, and every
   named public early-failure boundary without skip or CPU fallback.
9. RNN/GRU/LSTM both directions, bias variants, valid-length payloads, positive-zero padding, final
   states, elementary sites, zero/no-work special cases, and all gradients remain visibly
   fail-closed; schema vocabulary is never reported as recurrent execution.
10. Native/package, complete Metal, focused Compiler, conformance, dedicated Engine, Javadoc,
    architecture, Markdown/link/diff checks pass; independent cumulative Class C review has no
    unresolved P0/P1/P2.

## Validation after implementation

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

No timing, benchmark, or random-quality command is authorized. Complete packaged Metal, focused
Compiler contracts, conformance, public Engine, architecture/Javadoc, and repository validation
close the implementation.

## Documentation and review impact

Implementation must update every current capability/count/identity/random/dropout/recurrent
statement and describe only the exact bounded domains above. Public/backend/native/Javadoc wording
must name graph versus eager randomness, raw state and V1 private identity, threshold and scaling
policies, no entropy/portable stream claim, train/evaluation distinction, gradient mask reuse,
zero-feed and multi-output lifecycle, fixed custom route, and all recurrent blockers. Historical
tasks remain immutable evidence records.

Independent plan review precedes every production edit. Independent final review must inspect the
full plan-to-source proof, raw word and threshold arithmetic, probability conversion and scaling
sites, counter wrap, logical ordinal mapping, canonical BOOL bytes, zero-feed safety, output
liveness/alias rejection, single-writer state, session/concurrency isolation, route exclusivity,
identity invalidation, generated-gradient mask identity, recurrent fail-closed boundary, public
Engine behavior, documentation, and changed-path scope. Passing worker evidence may be reused unless
remediation changes executable behavior.

## Planning result

The audited maximal cutover is two fixed custom explicit-state RNG rows, not a generic
random-distribution or recurrent implementation. Both-profile zero-input `INITIAL_STATE` publishes
raw key/counter bits; accelerator canonical FLOAT32 `DROPOUT` publishes value, saved BOOL mask, and
next state using the private `SYNAPTIK_METAL_SPLITMIX64_COUNTER_V1` replay boundary. Exact threshold,
scaling, counter wrap, target-subset liveness, saved-mask backward, evaluation bypass, reuse, and
session/concurrency isolation are implemented and proven without Compiler or Model production
changes.

Production is now `83/32` capability and `101/14` structural execution; catalogs are `75/35/5`
MPSGraph and `60/55/0` custom. Schema 15, ABI 5, and thirteen exports remain fixed; all
backend-local identities are 21 and identity 20 fails closed. Recurrent execution remains empty
pending a complete TANH/EXP/SIGMOID proof and Compiler BPTT decision.
