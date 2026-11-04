# Task 0006D: Portable Explicit-State [RNG](../../../../glossary.md#graph-rng-state--graphrngstate) and [Dropout](../../../../glossary.md#dropout--inverted-dropout) Coverage

## Status

Complete

## Goal

Add the next executable [CPU](../../../../glossary.md#term-cpu) frontier after completed CPU 0006C: exactly one fully static,
resolved-[layout](../../../../glossary.md#layout) current [Model](../../../../glossary.md#term-model) `INITIAL_STATE` or [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout) [occurrence](../../../../glossary.md#term-occurrence) through the portable
generated-[kernel](../../../../glossary.md#kernel) route.

The task materializes the [opaque](../../../../glossary.md#term-opaque) two-word initial state without drawing randomness and executes
[FLOAT64](../../../../glossary.md#term-float64)/[FLOAT32](../../../../glossary.md#term-float32) training dropout with one explicitly versioned CPU-private counter-based
generator. Dropout consumes one logical draw per [row-major](../../../../glossary.md#term-row-major) logical input [element](../../../../glossary.md#term-element), writes the
dropped output, [canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) keep [mask](../../../../glossary.md#term-boolean-mask), and exact next state, and replays independently of
[scalar](../../../../glossary.md#term-scalar)/parallel chunking. It adds no hidden [mutable](../../../../glossary.md#term-mutable) [RNG state](../../../../glossary.md#graph-rng-state--graphrngstate) and makes no Model or cross-[backend](../../../../glossary.md#backend)
bitstream promise.

This is one cohesive implementation session. Both occurrences share the same explicit state
representation, random-family [lowering](../../../../glossary.md#lowering), generated emitter, multi-output [binding](../../../../glossary.md#term-binding), replay tests,
and [schema](../../../../glossary.md#term-schema) transition. Splitting initialization from dropout would duplicate the state execution
boundary without reducing architectural risk; the bounded one-[node](../../../../glossary.md#node) [scope](../../../../glossary.md#term-scope) and path ceiling below
remain comparable to completed CPU 0006C.

## [Scope](../../../../glossary.md#term-scope)

### Exact [occurrence](../../../../glossary.md#term-occurrence) and type [matrix](../../../../glossary.md#term-matrix-meanings)

- Admit exactly one [CPU](../../../../glossary.md#term-cpu)-owned [node](../../../../glossary.md#node) whose kind is either:
  - `GraphRngKind.INITIAL_STATE` with zero inputs and one [INT64](../../../../glossary.md#data-type--datatype) `Shape.of(2)` output; or
  - `DropoutKind.DROPOUT` with ordered inputs `[value, state]` and ordered outputs
    `[output, keepMask, nextState]`.
- Require fully static Shapes and resolved layouts with non-negative storage offsets and strides.
  Every writable output layout is statically proved injective. The dropout state input is also
  injective so logical lane zero and lane one denote distinct key and counter words.<span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride); [injective glossary](../../../../glossary.md#term-injective))</span>
- Support dropout value/output only for FLOAT64 and FLOAT32. BFLOAT16 remains fail-closed because
  the current CPU carrier and represented-addition support does not establish a direct,
  correctly-rounded BFLOAT16 division/conversion contract for inverted scaling. INT32, INT64,
  and BOOL dropout values remain ineligible under the Model floating-only contract.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [carrier glossary](../../../../glossary.md#term-storage-carrier); [INT32 glossary](../../../../glossary.md#data-type--datatype); [BOOL glossary](../../../../glossary.md#data-type--datatype); [Model glossary](../../../../glossary.md#term-model))</span>
- `INITIAL_STATE` writes logical state lane zero as the exact key bits and lane one as the exact
  counter bits from `GraphRngStateAttrs`, with no signed interpretation, normalization, hashing,
  random draw, or counter advancement.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>
- [`DROPOUT`](../../../../glossary.md#dropout--inverted-dropout) loads state [lane](../../../../glossary.md#term-lane) zero as key and lane one as counter, retains key exactly, and writes
  `counter + N` modulo `2^64`, where `N` is the checked static logical value [element](../../../../glossary.md#term-element) count.

### [CPU](../../../../glossary.md#term-cpu)-private generator configuration and test [vectors](../../../../glossary.md#term-vector)

Select the [immutable](../../../../glossary.md#term-immutable) configuration named `SYNAPTIK_CPU_SPLITMIX64_COUNTER_V1`. It is a CPU-private
counter-to-word [permutation](../../../../glossary.md#permutation), not a public algorithm identifier or [backend](../../../../glossary.md#backend)-neutral serialized
contract. All arithmetic below is Java `long` arithmetic modulo `2^64`; `>>>` is unsigned shift:

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
uniform53(word) = (word >>> 11) * 0x1.0p-53
```

`mix64` is a [bijection](../../../../glossary.md#term-bijection) on 64-bit words, so for a fixed key each distinct counter position maps to
one distinct 64-bit word across the full period. The key selects a translated, mixed CPU stream;
this task does not claim that different keys produce disjoint streams. `keyOffset` is computed
once per generated invocation, never once per [element](../../../../glossary.md#term-element).

The exact independent test vectors are:

| Key | Counter | Word | Top 53 bits | `Double.toHexString(uniform53)` |
|---|---|---|---|---|
| `0000000000000000` | `0000000000000000` | `48218226ff3cd4bf` | `09043044dfe79a` | `0x1.2086089bfcf34p-2` |
| `0000000000000000` | `0000000000000001` | `ea8568d2e45fd6cb` | `1d50ad1a5c8bfa` | `0x1.d50ad1a5c8bfap-1` |
| `0000000000000001` | `0000000000000000` | `dce423fc82c0d5b8` | `1b9c847f90581a` | `0x1.b9c847f90581ap-1` |
| `ffffffffffffffff` | `ffffffffffffffff` | `e8ba9f99ca933538` | `1d1753f3395266` | `0x1.d1753f3395266p-1` |
| `0000000000001234` | `0000000000000007` | `3e4cf5a0c9489779` | `07c99eb4192912` | `0x1.f267ad064a448p-3` |

Tests must derive these values independently rather than calling production generator helpers.
They also lock threshold behavior at exact representable boundaries and counter wrap.

This selection conforms to the [Model](../../../../glossary.md#term-model) contract because Model fixes only raw key/counter state,
one abstract draw per logical element, modulo advancement, and replay within one conforming
prepared implementation/configuration. Model 0019B/0019B1 explicitly decline to select a [PRNG](../../../../glossary.md#graph-rng-state--graphrngstate) or
promise cross-backend/cross-route/cross-version bitstreams. The algorithm name, version, [constants](../../../../glossary.md#term-constant),
mapping, and conversion therefore remain CPU-private realization facts. Any later change creates
a new CPU generator configuration and generated-[artifact](../../../../glossary.md#term-artifact) [schema](../../../../glossary.md#term-schema); it must not silently reuse V1
[identity](../../../../glossary.md#term-identity-meanings).

### [Dropout](../../../../glossary.md#dropout--inverted-dropout) keep, value, [mask](../../../../glossary.md#term-boolean-mask), and state semantics

- Logical [element](../../../../glossary.md#term-element) `i` is the zero-based row-major ordinal in the value Shape, independent of
  physical layout, carrier, range, chunk, worker, or vectorization.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [layout glossary](../../../../glossary.md#layout); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Exactly one word and one `uniform53` draw are associated with every logical element, regardless
  of probability, input value, output/mask layout, or keep result. Probability positive or
  negative zero still consumes all <span data-glossary-navigation="true"> ([negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>`N` draws.
- Keep exactly when `uniform53(word(key, counter, i)) >= probability`. No FLOAT32 narrowing of
  either the draw or probability occurs before this binary64 comparison.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [narrowing glossary](../../../../glossary.md#term-narrowing))</span>
- Store mask `true` as [canonical](../../../../glossary.md#term-canonical-form) byte `1` and `false` as canonical byte `0`. BOOL remains the
  current one-byte <span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype))</span>`byte[]`/native [`MemorySegment`](../../../../glossary.md#term-memorysegment) carrier with no bit packing or noncanonical
  truth values.
- A dropped output is the represented positive zero of its value type, even when the input is
  negative zero, NaN, or infinity.<span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity))</span>
- For a kept element, compute `denominator = 1.0d - probability` once per invocation using one
  binary64 subtraction. Then:
  - [FLOAT64](../../../../glossary.md#term-float64) performs one [binary64](../../../../glossary.md#floating-point-precision-and-rounding) division `input / denominator` and stores that result;
  - FLOAT32 widens the exact binary32 input to binary64, performs one binary64 division by the
    same denominator, and narrows once to binary32 for storage.
  No reciprocal multiplication, fused operation, probability narrowing, alternate evaluation
  order, or vector implementation is permitted in this task.<span data-glossary-navigation="true"> ([binary32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [fused glossary](../../../../glossary.md#specialization-and-fusion); [operation glossary](../../../../glossary.md#operation); [evaluation glossary](../../../../glossary.md#term-evaluation); [vector glossary](../../../../glossary.md#term-vector))</span>
- This finite-precision order fixes scalar/parallel parity and the same-configuration replay
  boundary. It does not claim exact-real arithmetic, NaN payload preservation after kept
  arithmetic, or a cross-backend numerical order.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [precision glossary](../../../../glossary.md#term-precision); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [backend glossary](../../../../glossary.md#backend))</span>
- [Scalar](../../../../glossary.md#term-scalar) [Shape](../../../../glossary.md#shape) consumes one draw. Any zero [extent](../../../../glossary.md#term-extent) gives `N == 0`: output and mask are empty, no
  element call or worker submission occurs, and next state contains the unchanged key/counter.
  Counter wrap uses ordinary two's-complement <span data-glossary-navigation="true"> ([two's-complement glossary](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks))</span>`long` addition and never fails.

### Generated execution and safe parallel strategy

- Add focused [CPU](../../../../glossary.md#term-cpu)-private `CpuRandomIr`, `CpuRandomLowering`, and `CpuRandomEmitter` owners.
  The IR distinguishes INITIAL_STATE from DROPOUT and retains the generator configuration/version,
  exact raw initializer words or probability bits, value type, ordered boundary roles/access
  plans, uniform conversion, comparison, finite-precision scaling, canonical BOOL representation,
  and state-advancement policy.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [finite glossary](../../../../glossary.md#term-finite); [precision glossary](../../../../glossary.md#term-precision); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Lower exactly one occurrence into one computation unit and one generated artifact. INITIAL_STATE
  boundaries are <span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [artifact glossary](../../../../glossary.md#term-artifact))</span>`[stateOutput]`. [DROPOUT](../../../../glossary.md#dropout--inverted-dropout) boundaries are `[value, state, output, keepMask,
  nextState]`; all five [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) values are distinct and declared in that order.
- Generated entries retain the current direct [carrier](../../../../glossary.md#term-storage-carrier) [arguments](../../../../glossary.md#term-argument), cold `long[]` geometry, and
  primitive `start`/`end`. The executable performs one generated prologue call before element
  work: INITIAL_STATE writes its two output words; DROPOUT writes next state exactly once. The
  dropout prologue uses the empty sentinel range `[0, 0)` and performs no draw or value/mask
  access. Non-empty element calls never write next state.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- [Scalar](../../../../glossary.md#term-scalar) execution invokes non-empty [element](../../../../glossary.md#term-element) range `[0, N)` after the prologue. Parallel-scalar
  uses deterministic disjoint row-major ordinal ranges after the same single prologue; every
  worker computes <span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major))</span>`word(key, counter, i)` directly from its global ordinal. No worker depends on
  a predecessor's generator state, and scheduling/chunk changes cannot affect values.
- INITIAL_STATE always uses scalar/single-thread execution. DROPOUT may select scalar or
  parallel-scalar under the existing worker/range thresholds. Vector and parallel-vector random
  execution are out of scope.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [scope glossary](../../../../glossary.md#term-scope))</span>
- No workspace, scratch, materialization, replay buffer, random-word buffer, mask staging,
  per-thread generator, or persistent mutable prepared state is permitted. Bounded resources are
  exactly one unit, one realized artifact, one output buffer for INITIAL_STATE or five distinct
  boundary buffers for DROPOUT, zero workspaces, and at most the existing four complete
  specialization candidates. Per-invocation setup uses only primitive locals; the element loop
  allocates nothing.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [buffer glossary](../../../../glossary.md#term-buffer); [mutable glossary](../../../../glossary.md#term-mutable); [specialization glossary](../../../../glossary.md#term-specialization))</span>

### Overlap, [binding](../../../../glossary.md#term-binding), and failure order

- Capability and lowering reject unresolved, negative, inconsistent, non-static, or non-injective
  writable geometry before resource declaration. Checked element counts, byte sizes, spans,
  address arithmetic, and counter advancement facts fail preparation on overflow except the
  intentional modulo-<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [injective glossary](../../../../glossary.md#term-injective); [element glossary](../../../../glossary.md#term-element); [overflow glossary](../../../../glossary.md#term-overflow))</span>`2^64` state addition.
- Cold binding validates all carriers, byte sizes, alignments, read-only status, layout spans,
  workspace absence, and complete physical overlap before returning a bound invocation.<span data-glossary-navigation="true"> ([alignments glossary](../../../../glossary.md#term-alignment))</span><span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation))</span>
- INITIAL_STATE validates its complete writable output binding; it has no input/output or
  output/output pair. Zero inputs must be supported by specialization, preparation, finalization,
  generated signature, and executable code rather than represented by a dummy input.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [signature glossary](../../../../glossary.md#term-signature))</span>
- For DROPOUT, reject each of the three complete output spans against each of the two complete
  input spans, then reject all three output/output pairs. Use complete bindings, not selected
  range ordinals. Reject before the state prologue, any output/mask mutation, or worker submission.<span data-glossary-navigation="true"> ([DROPOUT glossary](../../../../glossary.md#dropout--inverted-dropout); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>
- Physical input/input overlap is permitted: both inputs are read-only during element work and no
  semantic alias is inferred. Logical boundary <span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) values remain distinct.
- Binding and prepared access declarations identify the first two dropout boundaries as reads and
  the last three as writes. One bound invocation supplies direct typed references for all five
  boundaries and performs the prologue plus zero or more element calls. Runtime receives no
  Operation, CompiledNode, graph, RNG service, or mutable generator.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Operation glossary](../../../../glossary.md#operation); [graph glossary](../../../../glossary.md#graph); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Invalid or unsupported work fails before mutation. A thrown generated or worker failure may
  leave output buffers partially written under the existing executable failure contract, but
  hidden RNG state is impossible: replay starts solely from the unchanged explicit input state.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>

### Compatibility and documentation

- Advance generated compatibility from [schema](../../../../glossary.md#term-schema) 18 to schema 19 with no migration reader.
- Stable structural identity and compatibility bytes include the random family, exact generator
  configuration name/version/constants and mapping, uniform conversion and threshold rule,
  probability raw bits for DROPOUT, initializer key/counter raw bits for INITIAL_STATE, value and
  boundary types, ordered output roles/count, access-plan structure, scaling order, canonical BOOL
  policy, prologue/state policy, execution compute mode, carrier pattern, and scratch absence.
  Concrete offsets, stride magnitudes, extents, slots, carrier objects, addresses, workers, run
  identity, and chunk count remain cold when they do not change emitted bytes.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [constants glossary](../../../../glossary.md#term-constant); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [DROPOUT glossary](../../../../glossary.md#dropout--inverted-dropout); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [carrier glossary](../../../../glossary.md#term-storage-carrier); [stride glossary](../../../../glossary.md#term-stride); [run glossary](../../../../glossary.md#term-run))</span>
- Extend capability, lowering dispatch, IR permits/encoding, route plan, preparation/finalization,
  generator validation/dispatch, specialization, executable binding, scalar reference, schema,
  package contracts, and focused tests only as this family requires.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [dispatch glossary](../../../../glossary.md#term-dispatch); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [validation glossary](../../../../glossary.md#term-validation); [specialization glossary](../../../../glossary.md#term-specialization); [binding glossary](../../../../glossary.md#term-binding); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- After executable Java stabilizes, hand the uncommitted diff and CPU-test evidence to a distinct
  clean documentation-focused context. It finalizes affected Javadocs/package summaries, the CPU
  backend guide, glossary, this task evidence, CPU master plan, and roadmap. Tensor, Compile,
  Runtime, and Training API guides are expected no-change because the public Model/compiler/
  runtime APIs do not change; the pass records the reasoned conclusion.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Model glossary](../../../../glossary.md#term-model); [compiler glossary](../../../../glossary.md#term-compiler))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- BFLOAT16 dropout arithmetic, FLOAT16, integral/BOOL dropout values, other random distributions,
  Bernoulli APIs, attention dropout, key derivation, split/fold-in/jump, or a public generator API<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [FLOAT16 glossary](../../../../glossary.md#term-float16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [distributions glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [Bernoulli glossary](../../../../glossary.md#term-bernoulli-distribution); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- a Model/cross-backend bitstream, public algorithm identifier, serialized RNG format, stable
  cross-version replay, cryptographic or statistical certification, or distinct-key disjointness<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [backend glossary](../../../../glossary.md#backend); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- more than one node, INITIAL_STATE-to-DROPOUT fusion, dropout fusion with pointwise work, general
  partition DAG decomposition, materialized splits, or saved-mask lifetime policy<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [fusion glossary](../../../../glossary.md#term-fusion); [pointwise glossary](../../../../glossary.md#term-pointwise); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [decomposition glossary](../../../../glossary.md#term-decomposition); [mask glossary](../../../../glossary.md#term-boolean-mask); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- compiler capture/inference/autograd changes, gradient formulas, CSE/DCE changes, or hidden-mask
  policy; completed Compiler behavior is only inspected to confirm input/output ordering<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [autograd glossary](../../../../glossary.md#autograd); [gradient glossary](../../../../glossary.md#term-gradient); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce))</span>
- Model, shared Runtime/Prepare, Config, Trace, backend-contract, architecture, architecture-test,
  Gradle/dependency, native-route, tuning, benchmark, persistence-policy, backend-conformance,
  integration, Engine, NN, or training changes<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Trace glossary](../../../../glossary.md#trace); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [persistence glossary](../../../../glossary.md#term-persistence); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- reflection, string [dispatch](../../../../glossary.md#term-dispatch), `java.util.Random`, `RandomGenerator`, `TensorRandoms`, a global,
  default, thread-local, or mutable prepared RNG, and per-element allocation<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [element glossary](../../../../glossary.md#term-element))</span>

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU 0005A partition-kernel reset](0005a-atomic-partition-kernel-architecture-reset.md)<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [kernel glossary](../../../../glossary.md#kernel))</span>
- [CPU 0005D evidence gate](0005d-materialization-specialization-and-persistence-evidence-gate.md)
- [CPU 0006 static affine views](0006-portable-static-affine-views-and-boundary-materialization.md)<span data-glossary-navigation="true"> ([views glossary](../../../../glossary.md#term-view-meanings))</span>
- [CPU 0006A movement](0006a-portable-pad-tile-and-tensor-composition-movement.md)
- [CPU 0006A2 indexing](0006a2-portable-gather-and-one-hot-indexing.md)
- [CPU 0006B1 scatter](0006b1-portable-functional-scatter.md)
- [CPU 0006B2 fold](0006b2-portable-overlap-fold.md)
- [CPU 0006C ordering](0006c-portable-stable-ordering-and-selection.md)
- [Model 0019B explicit graph RNG state](../../../modules/model/tasks/0019b-explicit-graph-rng-state-foundation.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [graph RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- [Model 0019B1 explicit graph dropout](../../../modules/model/tasks/0019b1-explicit-graph-dropout-construction.md)<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)

## Architecture constraints

- Model owns operation identity, raw state meaning, ordered roles, probability domain, logical draw
  count, keep rule, ideal inverted-dropout meaning, and portable state advancement. CPU owns only
  its private generator, finite-precision realization, IR, route, artifact, resources, and
  invocation.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [operation glossary](../../../../glossary.md#operation); [identity glossary](../../../../glossary.md#term-identity-meanings); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout); [CPU glossary](../../../../glossary.md#term-cpu); [finite glossary](../../../../glossary.md#term-finite); [precision glossary](../../../../glossary.md#term-precision); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- CPU analysis deterministically lowers and selects its supported route from projected facts,
  declares exact buffers and zero workspace before CPU-blind shared assignment, and finalizes one
  artifact only after assignments exist.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer))</span>
- Prepared recipes remain immutable and reusable. Every draw is a pure function of explicit state
  and logical index, so concurrent runs share no mutable RNG state and distinct <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [logical index glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [mutable glossary](../../../../glossary.md#term-mutable); [RNG state glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>[`RunState`](../../../../glossary.md#run-state--runstate) values
  retain ordinary isolation.
- Runtime hot execution receives direct bound carrier references and primitive geometry only. It
  performs no graph inspection, Operation/CompiledNode access, map lookup, reflection, string
  dispatch, service lookup, route selection, cache work, or generator mutation.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [carrier glossary](../../../../glossary.md#term-storage-carrier); [graph glossary](../../../../glossary.md#graph); [lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch); [cache glossary](../../../../glossary.md#term-cache))</span>
- Work remains within `backends/cpu`, adds no dependency or package, and preserves the existing
  generated portable route and optional artifact-store lifecycle.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Capability must be no broader than complete lowering, exact declarations, assignment,
  generation, binding, overlap validation, scalar/parallel execution, and reference evidence.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [binding glossary](../../../../glossary.md#term-binding); [validation glossary](../../../../glossary.md#term-validation); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Stop if exact execution requires a Model/compiler/shared Runtime or Prepare change, if a
  zero-input or three-output occurrence cannot fit the current backend-owned seams without such a
  change, or if the finite-precision rules conflict with current Model semantics.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [occurrence glossary](../../../../glossary.md#term-occurrence); [backend glossary](../../../../glossary.md#backend))</span>

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — sole supported capability provider.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — [CPU](../../../../glossary.md#term-cpu)-private structural random [IR](../../../../glossary.md#term-intermediate-representation-ir).
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — one-node semantic revalidation and
  cold geometry.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct Class-File emission.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare`, `.route.portable`, `.cache`,
  `.executable`, `.memory`, and `.reference` — existing staged realization, carriers, execution,
  and independent reference boundaries.

Packages added or changed:

- No package is added, removed, moved, or exported. Existing CPU-internal packages gain only the
  focused random-family types and direct integration required above.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr` — immutable family, generator,
  numeric, BOOL, state, boundary-role, and cache identity.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [BOOL glossary](../../../../glossary.md#data-type--datatype); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRandomLowering` — exact one-node
  INITIAL_STATE/DROPOUT revalidation, declarations, logical-index geometry, layouts, and zero-
  workspace plan.<span data-glossary-navigation="true"> ([DROPOUT glossary](../../../../glossary.md#dropout--inverted-dropout); [layouts glossary](../../../../glossary.md#layout))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuRandomEmitter` — allocation-free
  generated initializer, dropout prologue, counter-to-word mapping, mask, scaling, and stores.<span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask))</span>

Tests mirror these production packages. No generic random utility, generator service, [registry](../../../../glossary.md#term-registry),
manager, [facade](../../../../glossary.md#term-facade), or new public CPU type is permitted.

## Affected files

Expected production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuRandomIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRandomLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuRandomEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`

Expected [CPU](../../../../glossary.md#term-cpu) tests:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuRandomIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRandomLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuRandomGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuBufferBindingTest.java`

Expected documentation/[planning](../../../../glossary.md#term-planning) paths during implementation:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No [Model](../../../../glossary.md#term-model)/[compiler](../../../../glossary.md#term-compiler)/shared [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[API](../../../../glossary.md#term-application-programming-interface-api)/architecture/[Gradle](../../../../glossary.md#term-gradle)/[conformance](../../../../glossary.md#term-conformance)/integration path is
expected to change.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 43 paths: 24 production/package paths, 14 [CPU](../../../../glossary.md#term-cpu) test paths,
and 5 documentation/[planning](../../../../glossary.md#term-planning) paths. Stop before a 44th path, a new package, or any path outside
the listed CPU [module](../../../../glossary.md#term-module) and documentation/planning set. A stale existing CPU inventory test may
replace another listed test path but must not increase the ceiling.

If implementation requires a shared contract, public [API](../../../../glossary.md#term-application-programming-interface-api), [compiler](../../../../glossary.md#term-compiler) change, [BFLOAT16](../../../../glossary.md#term-bfloat16) arithmetic,
or more paths, stop and propose a focused follow-up or architecture decision. Do not hide
incomplete INITIAL_STATE or [DROPOUT](../../../../glossary.md#dropout--inverted-dropout) acceptance behind a follow-up.

## Acceptance criteria

- Capability and lowering admit exactly the specified one-node INITIAL_STATE and FLOAT64/FLOAT32
  DROPOUT matrix under complete static/resolved/injective guards; BFLOAT16 and every unsupported
  row remain truthfully false/fail-closed.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [node glossary](../../../../glossary.md#node); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [DROPOUT glossary](../../../../glossary.md#dropout--inverted-dropout); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [injective glossary](../../../../glossary.md#term-injective); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- INITIAL_STATE has exactly one output boundary, writes key then counter raw bits, performs no
  draw, and works for every word pair including signed extrema.<span data-glossary-navigation="true"> ([raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum))</span>
- DROPOUT boundaries, access roles, buffer declarations, assignments, generated parameters,
  direct bound fields, and stores are exactly <span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [parameters glossary](../../../../glossary.md#term-parameter))</span>`[value, state, output, keepMask, nextState]` with
  three writable outputs.
- Generator constants, V1 mapping, uniform conversion, threshold comparison, probability-zero
  behavior, test vectors, logical draw count, modulo wrap, and FLOAT64/FLOAT32 finite-precision
  scaling order match this specification exactly.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [vectors glossary](../../../../glossary.md#term-vector); [finite glossary](../../../../glossary.md#term-finite); [precision glossary](../../../../glossary.md#term-precision))</span>
- Scalar and parallel-scalar executions are bitwise identical for output, canonical byte mask,
  and next-state words across different valid range counts and schedules. Random value <span data-glossary-navigation="true"> ([Scalar glossary](../../../../glossary.md#term-scalar); [canonical glossary](../../../../glossary.md#term-canonical-form); [mask glossary](../../../../glossary.md#term-boolean-mask))</span>`i`
  depends only on input key/counter and global logical ordinal `i`.
- Dense, offset, positive-strided, and zero-strided value reads; distinct-lane injective state
  reads; injective non-dense writes; heap/native/mixed carriers; scalar, empty, singleton,
  probability boundaries, signed zero, finite values, NaN, infinities, and counter wrap agree with
  an independent scalar reference.<span data-glossary-navigation="true"> ([lane glossary](../../../../glossary.md#term-lane); [heap glossary](../../../../glossary.md#term-heap); [singleton glossary](../../../../glossary.md#term-singleton); [signed zero glossary](../../../../glossary.md#term-signed-zero); [NaN glossary](../../../../glossary.md#term-nan); [infinities glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
- One generated prologue writes state once before any worker submission; empty dropout still
  writes next state and submits no element work. INITIAL_STATE uses the same generated family
  boundary without a dummy input.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- All required input/output and output/output overlaps fail during cold binding before prologue,
  mutation, or submission; permitted input/input overlap is tested.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Zero workspace/materialization/replay scratch, bounded candidates, one realized artifact,
  immutable prepared state, and allocation-free element loops are enforced by construction and
  tests.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [artifact glossary](../../../../glossary.md#term-artifact); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Schema 19 rejects schema-18 artifacts. Random semantics/configuration and code-shaping facts are
  present in deterministic structural/fingerprint/compatibility identity; cold instance facts
  remain excluded as specified.<span data-glossary-navigation="true"> ([compatibility identity glossary](../../../../glossary.md#identity-meanings-and-algebraic-identities))</span><span data-glossary-navigation="true"> ([Schema glossary](../../../../glossary.md#term-schema); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Runtime hot execution contains no Operation/CompiledNode, graph inspection, map lookup,
  reflection, string dispatch, RNG service, mutable generator state, or per-element allocation.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Operation glossary](../../../../glossary.md#operation); [graph glossary](../../../../glossary.md#graph); [lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- Existing pointwise, movement, indexing, scatter, fold, ordering, worker, cache, and persistence
  behavior remains unchanged.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- A separate documentation-focused context finalizes affected Javadocs, package summaries, CPU
  guide, glossary, planning evidence, and explicit API/architecture/conformance no-change
  conclusions before this task becomes Complete.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, [run](../../../../glossary.md#term-run) focused tests for random [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/generated execution, capability,
preparation/finalization, [binding](../../../../glossary.md#term-binding)/overlap, [scalar](../../../../glossary.md#term-scalar)/parallel execution, [cache](../../../../glossary.md#term-cache)/[schema](../../../../glossary.md#term-schema), carriers, and
independent reference [vectors](../../../../glossary.md#term-vector). After executable Java stabilizes, run one final [module](../../../../glossary.md#term-module) suite:

```bash
./gradlew :backends:cpu:test
```

The separate clean documentation pass reuses that evidence unless it changes executable Java and
runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates affected Markdown links and anchors, balanced fences, one terminal newline,
trailing whitespace, generated Javadocs, exact changed-path membership and ceiling, package/type
placement, schema 19, test-vector transcription, and task/master/roadmap status coherence.

Repository-wide tests, architecture tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), and integration tests are deferred
to [CPU](../../../../glossary.md#term-cpu) 0009 or [CI](../../../../glossary.md#term-continuous-integration-ci). Run them here only if implementation unexpectedly changes a repository-wide,
[dependency](../../../../glossary.md#term-dependency), architecture, or reusable cross-backend contract; such a change is outside [scope](../../../../glossary.md#term-scope) and
normally requires stopping first.

## Dependencies

- CPU 0005A–0006C: Complete, including typed carriers, zero-/multi-input movement foundations,
  workers, multi-output ordering, cold overlap validation, and schema 18.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [validation glossary](../../../../glossary.md#term-validation); [schema glossary](../../../../glossary.md#term-schema))</span>
- [Model](../../../../glossary.md#term-model) 0019B and 0019B1: Complete raw state, [dropout](../../../../glossary.md#dropout--inverted-dropout) probability/role/draw/advancement semantics.
- Compiler capture/inference/optimization and stochastic-gradient work: Complete and inspected only
  to confirm that INITIAL_STATE remains zero-input/one-output and DROPOUT remains ordered
  two-input/three-output with the auxiliary mask retained.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [capture glossary](../../../../glossary.md#term-capture-meanings); [inference glossary](../../../../glossary.md#term-inference); [gradient glossary](../../../../glossary.md#term-gradient); [auxiliary mask glossary](../../../../glossary.md#auxiliary-mask))</span>
- Current staged [Prepare](../../../../glossary.md#prepare) and [Runtime](../../../../glossary.md#term-runtime) cold-[binding](../../../../glossary.md#term-normative) contracts: Complete and unchanged.

## Follow-up tasks

- CPU 0007 remains the next Draft frontier and owns portable reduction, scan, statistics,
  softmax/log-softmax, and normalization coverage. Do not create its detailed specification here.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [scan glossary](../../../../glossary.md#term-cumulative-scan); [softmax glossary](../../../../glossary.md#softmax--log-softmax); [log-softmax glossary](../../../../glossary.md#softmax--log-softmax); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- CPU 0009 remains responsible for portable generated-coverage/conformance closure, including the
  intentional BFLOAT16 dropout gap unless an earlier separately planned CPU task establishes a
  conforming direct BFLOAT16 scaling rule.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- A later explicit cross-backend RNG decision may standardize a portable bitstream. It would be a
  Model/architecture-level decision and must not reinterpret this CPU-private V1 identity.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate); [Model glossary](../../../../glossary.md#term-model); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>

## Architecture impact

Expected impact: None.

This task uses the existing concrete-[backend ownership](../../../../glossary.md#backend-ownership) of [lowering](../../../../glossary.md#lowering), route selection, generated
[artifacts](../../../../glossary.md#term-artifact), storage, and execution. It changes no [module](../../../../glossary.md#term-module) responsibility or [dependency direction](../../../../glossary.md#java-platform-and-build-tools).
If implementation requires architecture, another module, or a shared contract change, stop and
report the exact conflict instead of editing around it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Implement Synaptik CPU task 0006D exactly from its Ready specification. Do not use GSD. Read
AGENTS.md, ARCHITECTURE.md, the current architecture plan and focused Runtime/Prepare/Backend and
performance pages, documentation rules, planning guide, roadmap, CPU master plan, task 0006D,
completed CPU tasks 0005A/0005D/0006/0006A/0006A2/0006B1/0006B2/0006C, Model 0019B/0019B1, and
every affected source/test seam in full before editing.

Deliver one-node static resolved INITIAL_STATE and FLOAT64/FLOAT32 explicit-state DROPOUT through
the generated portable route, exact CPU-private V1 vectors, uniform/threshold/scaling rules,
canonical BOOL mask, modulo state, deterministic scalar/parallel parity, zero workspace, five-
boundary three-output binding, complete pre-mutation overlap checks, and schema 19. Keep BFLOAT16
dropout fail-closed. Preserve all exclusions and the 43-path ceiling. Stop on architecture,
shared-contract, cross-module, semantic, numerical, or scope conflict.

Run focused tests and one final ./gradlew :backends:cpu:test after executable Java stabilizes. Do
not commit or push. Then hand the uncommitted diff and exact Java evidence to a distinct clean
documentation-focused context following docs/developer-guide/documentation-rules.md. That pass
must finalize affected Javadocs/package summaries, CPU guide, glossary, task/master/roadmap,
no-change conclusions, Javadoc, Markdown, scope, and whitespace validation without repeating the
successful Java suite unless executable behavior changes. Do not mark Complete until both passes
and all acceptance criteria succeed. Return exact evidence, changed paths, unresolved issues,
context ID if available, and Status: Complete or Status: Incomplete with required follow-up.
```

## Local decisions

- `SYNAPTIK_CPU_SPLITMIX64_COUNTER_V1` is the exact CPU-private realization. Its full constants,
  mapping, conversion, and version enter artifact identity; it creates no Model promise.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [constants glossary](../../../../glossary.md#term-constant); [artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings); [Model glossary](../../../../glossary.md#term-model))</span>
- Probability and initializer word bits are baked semantic/code-shaping facts for this first
  bounded generated family. They enter IR/fingerprint/cache identity; dynamic dropout key/counter
  values remain explicit input data.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [cache glossary](../../../../glossary.md#term-cache); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- State output uses one generated prologue call before element ranges. This produces exactly one
  next-state write for scalar and parallel execution and also handles empty dropout without
  workspace or mutable orchestration state.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [scalar glossary](../../../../glossary.md#term-scalar); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- FLOAT64 and FLOAT32 use the exact binary64 denominator/division order above. BFLOAT16 remains
  fail-closed rather than adopting an unproved double-rounding path.<span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- [BOOL](../../../../glossary.md#data-type--datatype) [mask](../../../../glossary.md#term-boolean-mask) storage is [canonical](../../../../glossary.md#term-canonical-form) one-byte `0`/`1`, consistent with current CPU carriers and Model
  BOOL semantics.
- Input/input physical overlap is harmless and permitted; every input/output and output/output
  overlap is rejected from complete spans before mutation.

## Known limitations

- Coverage is one fully static, resolved-layout occurrence. Dynamic Shapes/layouts, negative
  physical offsets/strides, multi-node fusion, vector random execution, native routes, and
  BFLOAT16 dropout are unsupported.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [occurrence glossary](../../../../glossary.md#term-occurrence); [Shapes glossary](../../../../glossary.md#shape); [strides glossary](../../../../glossary.md#term-stride); [node glossary](../../../../glossary.md#node); [fusion glossary](../../../../glossary.md#term-fusion); [vector glossary](../../../../glossary.md#term-vector); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [dropout glossary](../../../../glossary.md#dropout--inverted-dropout))</span>
- The selected generator is not cryptographic and has no distinct-key stream-disjointness claim.
  Replay is bitwise only for this exact CPU V1 configuration and its fixed finite-precision rules,
  not across backends or future versions.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [finite glossary](../../../../glossary.md#term-finite); [precision glossary](../../../../glossary.md#term-precision); [backends glossary](../../../../glossary.md#backend))</span>
- Static element counts and resource sizes must fit the current checked host/preparation contracts;
  the Model's abstract modulo advancement for larger symbolic products remains outside this fully
  static CPU slice.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [host glossary](../../../../glossary.md#term-host); [Model glossary](../../../../glossary.md#term-model); [slice glossary](../../../../glossary.md#slice))</span>
- `testing/backend-conformance` has no current executable cross-backend RNG harness. CPU 0009 owns
  the later closure rather than this task inventing one.<span data-glossary-navigation="true"> ([RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation and clean-context documentation review are complete. Focused [CPU](../../../../glossary.md#term-cpu) tests cover
independent V1 [vectors](../../../../glossary.md#term-vector), capability, [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering), generated initializer/[dropout](../../../../glossary.md#dropout--inverted-dropout) execution,
preparation/finalization, [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache), [carrier](../../../../glossary.md#term-storage-carrier) [binding](../../../../glossary.md#term-binding), complete overlap rejection,
[scalar](../../../../glossary.md#term-scalar)/parallel replay, empty behavior, and an independent scalar reference. The authoritative
final CPU suite ran after executable Java first stabilized. A subsequent clean-context
implementation review corrected narrow acceptance gaps in key-offset setup, state-input
[injectivity](../../../../glossary.md#storage-mapping-contiguity-and-injectivity), structural generator [identity](../../../../glossary.md#term-identity-meanings), independent vectors, [layout](../../../../glossary.md#layout)/carrier coverage,
threshold boundaries, and permitted input overlap. Per the one-authoritative-suite limit, those
corrections were validated with focused tests only. No executable Java changed during the final
documentation pass, so it did not rerun CPU tests.

- Focused affected-seam [run](../../../../glossary.md#term-run): `BUILD SUCCESSFUL` in 1s; 122 tests, zero failures.
- Post-review capability/IR/lowering/generated/reference/executable run: `BUILD SUCCESSFUL` in 2s;
  71 tests, 0 failures, 0 errors, 0 skipped.
- Final random identity/generated run: `BUILD SUCCESSFUL` in 1s; 9 tests, 0 failures, 0 errors,
  0 skipped.
- Post-review cache/schema/generator/preparation/finalization/binding run: `BUILD SUCCESSFUL` in 1s;
  46 tests, 0 failures, 0 errors, 0 skipped.
- One authoritative final `./gradlew :backends:cpu:test`: `BUILD SUCCESSFUL` in 2s; 21 actionable
  tasks, 2 executed and 19 up-to-date; 47 suites, 275 tests, 0 failures, 0 errors, 1 skipped.
- Final implementation-pass `git diff --check` passed. The CPU 0006D scope is exactly 41 paths:
  24 production/package paths, 14 CPU test paths, and the three already-authorized planning paths.
  The two remaining authorized documentation paths are reserved for the mandatory documentation
  pass, keeping the overall ceiling at 43.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [planning glossary](../../../../glossary.md#term-planning))</span>
- Clean documentation context `019ffcab-2c42-7d62-be4a-4b2815654c89` independently reviewed the
  complete 24-path production/Javadoc and 14-path focused-test diff, rendered Javadocs, Model
  0019B/0019B1 boundary, CPU guide, glossary, task, master plan, and CPU-specific roadmap passages.
  It finalized Javadoc lifecycle, ownership, failure, overlap, and parameter contracts; added the
  current CPU-private RNG/dropout guide and bounded-replay glossary clarification; and changed no
  executable behavior or test.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Model glossary](../../../../glossary.md#term-model); [lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [parameter glossary](../../../../glossary.md#term-parameter); [RNG glossary](../../../../glossary.md#graph-rng-state--graphrngstate))</span>
- Final `./gradlew :backends:cpu:javadoc`: `BUILD SUCCESSFUL` in 2s; 11 actionable tasks, 2
  executed and 9 up-to-date. The generated pages for `CpuRandomIr`, `CpuRandomLowering`,
  `CpuRandomEmitter`, `CpuPreparedExecutable`, and affected package summaries were inspected.
  Javadoc emitted only the two expected incubating-Vector-module warnings and no documentation
  warnings.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- Local Markdown validation passed for all five authorized files: every local target and heading
  anchor resolved, fences were balanced, each file had exactly one terminal newline, and no file
  had trailing whitespace.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
- Exact transcription and coherence checks passed for all five V1 vectors, schema 19, BFLOAT16
  fail-closed wording, 0006D Complete status, CPU 0007 Draft status with no detailed task file,
  package/type placement, and the exact 43-path CPU task scope.<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Final combined-worktree `git diff --check` passed. Concurrent NN/training, settings, and
  architecture-test paths were excluded from the CPU count and preserved without modification by
  this documentation context.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Reasoned no-change review: Tensor, Compile, Runtime, and Training API guides remain accurate
  because this task changes only CPU-private realization, not shared or public APIs. The
  architecture contract, focused architecture pages, ADRs, and architecture tests remain accurate
  because ownership, lifecycle, and dependency direction do not change. Model, Compiler, shared
  Runtime/Prepare/Config/Trace/backend-contract, Gradle/dependencies, backend conformance,
  integration, native routes, fusion, tuning, Engine, NN, and training require no task change
  because none of their contracts or executable behavior is modified. CPU 0009 or CI retains the
  deferred cross-backend/repository checkpoint.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [architecture contract glossary](../../../../glossary.md#architecture-contract); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Trace glossary](../../../../glossary.md#trace); [backend glossary](../../../../glossary.md#backend); [Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [fusion glossary](../../../../glossary.md#term-fusion); [Engine glossary](../../../../glossary.md#term-engine); [CI glossary](../../../../glossary.md#term-continuous-integration-ci); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Implementation notes

Implementation adds [CPU](../../../../glossary.md#term-cpu)-private `CpuRandomIr`, `CpuRandomLowering`, and `CpuRandomEmitter` with
direct integrations through capability, [schema](../../../../glossary.md#term-schema) 19, portable generation, preparation,
finalization, execution, and [scalar](../../../../glossary.md#term-scalar) reference seams. INITIAL_STATE uses one zero-range generated
prologue; [DROPOUT](../../../../glossary.md#dropout--inverted-dropout) uses that prologue for next state followed by deterministic global-ordinal
[element](../../../../glossary.md#term-element) ranges and zero workspace. [BFLOAT16](../../../../glossary.md#term-bfloat16) remains [fail-closed](../../../../glossary.md#term-fail-closed).

## Completion summary

- Completed changes: added one-node static resolved INITIAL_STATE and FLOAT64/FLOAT32
  explicit-state DROPOUT through the portable generated route with CPU-private V1 replay,
  canonical BOOL mask, modulo state, scalar/parallel parity, zero workspace, complete overlap
  rejection, and schema 19; BFLOAT16 remains fail-closed.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [DROPOUT glossary](../../../../glossary.md#dropout--inverted-dropout); [CPU glossary](../../../../glossary.md#term-cpu); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [mask glossary](../../../../glossary.md#term-boolean-mask); [scalar glossary](../../../../glossary.md#term-scalar); [schema glossary](../../../../glossary.md#term-schema); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Files changed or created: 24 CPU production/package paths, 14 CPU tests, and exactly five
  documentation/planning paths, for 43 authorized CPU-task paths.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Tests and validation: reused the authoritative 47-suite/275-test CPU result and focused
  122-/71-/9-/46-test results; final CPU Javadoc, rendered-page inspection, Markdown, scope,
  vector, schema, status, package placement, and whitespace checks passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [vector glossary](../../../../glossary.md#term-vector))</span>
- Documentation-agent review: clean context `019ffcab-2c42-7d62-be4a-4b2815654c89` completed the
  independent finalization without changing executable behavior or rerunning CPU tests.
- Documentation impact: CPU guide, glossary, task, CPU master plan, and CPU roadmap status are
  synchronized to the implemented boundary.
- Javadoc review: affected types, constructors, methods, and package summaries are accurate and
  rendered without documentation warnings.
- Glossary impact: bounded replay now identifies the CPU V1 boundary without exposing it as public
  configuration or portable Model semantics.<span data-glossary-navigation="true"> ([bounded replay glossary](../../../../glossary.md#bounded-replay); [Model glossary](../../../../glossary.md#term-model))</span>
- Unresolved issues: None within task scope. The documented BFLOAT16 and cross-backend
  conformance limitations remain intentional follow-up boundaries.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Follow-up required: None for CPU 0006D. CPU 0007 is the next Draft frontier and has no detailed
  specification.

Status: Complete
