# Task 0006D: Portable Explicit-State RNG and Dropout Coverage

## Status

Complete

## Goal

Add the next executable CPU frontier after completed CPU 0006C: exactly one fully static,
resolved-layout current Model `INITIAL_STATE` or `DROPOUT` occurrence through the portable
generated-kernel route.

The task materializes the opaque two-word initial state without drawing randomness and executes
FLOAT64/FLOAT32 training dropout with one explicitly versioned CPU-private counter-based
generator. Dropout consumes one logical draw per row-major logical input element, writes the
dropped output, canonical BOOL keep mask, and exact next state, and replays independently of
scalar/parallel chunking. It adds no hidden mutable RNG state and makes no Model or cross-backend
bitstream promise.

This is one cohesive implementation session. Both occurrences share the same explicit state
representation, random-family lowering, generated emitter, multi-output binding, replay tests,
and schema transition. Splitting initialization from dropout would duplicate the state execution
boundary without reducing architectural risk; the bounded one-node scope and path ceiling below
remain comparable to completed CPU 0006C.

## Scope

### Exact occurrence and type matrix

- Admit exactly one CPU-owned node whose kind is either:
  - `GraphRngKind.INITIAL_STATE` with zero inputs and one INT64 `Shape.of(2)` output; or
  - `DropoutKind.DROPOUT` with ordered inputs `[value, state]` and ordered outputs
    `[output, keepMask, nextState]`.
- Require fully static Shapes and resolved layouts with non-negative storage offsets and strides.
  Every writable output layout is statically proved injective. The dropout state input is also
  injective so logical lane zero and lane one denote distinct key and counter words.
- Support dropout value/output only for FLOAT64 and FLOAT32. BFLOAT16 remains fail-closed because
  the current CPU carrier and represented-addition support does not establish a direct,
  correctly-rounded BFLOAT16 division/conversion contract for inverted scaling. INT32, INT64,
  and BOOL dropout values remain ineligible under the Model floating-only contract.
- `INITIAL_STATE` writes logical state lane zero as the exact key bits and lane one as the exact
  counter bits from `GraphRngStateAttrs`, with no signed interpretation, normalization, hashing,
  random draw, or counter advancement.
- `DROPOUT` loads state lane zero as key and lane one as counter, retains key exactly, and writes
  `counter + N` modulo `2^64`, where `N` is the checked static logical value element count.

### CPU-private generator configuration and test vectors

Select the immutable configuration named `SYNAPTIK_CPU_SPLITMIX64_COUNTER_V1`. It is a CPU-private
counter-to-word permutation, not a public algorithm identifier or backend-neutral serialized
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

`mix64` is a bijection on 64-bit words, so for a fixed key each distinct counter position maps to
one distinct 64-bit word across the full period. The key selects a translated, mixed CPU stream;
this task does not claim that different keys produce disjoint streams. `keyOffset` is computed
once per generated invocation, never once per element.

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

This selection conforms to the Model contract because Model fixes only raw key/counter state,
one abstract draw per logical element, modulo advancement, and replay within one conforming
prepared implementation/configuration. Model 0019B/0019B1 explicitly decline to select a PRNG or
promise cross-backend/cross-route/cross-version bitstreams. The algorithm name, version, constants,
mapping, and conversion therefore remain CPU-private realization facts. Any later change creates
a new CPU generator configuration and generated-artifact schema; it must not silently reuse V1
identity.

### Dropout keep, value, mask, and state semantics

- Logical element `i` is the zero-based row-major ordinal in the value Shape, independent of
  physical layout, carrier, range, chunk, worker, or vectorization.
- Exactly one word and one `uniform53` draw are associated with every logical element, regardless
  of probability, input value, output/mask layout, or keep result. Probability positive or
  negative zero still consumes all `N` draws.
- Keep exactly when `uniform53(word(key, counter, i)) >= probability`. No FLOAT32 narrowing of
  either the draw or probability occurs before this binary64 comparison.
- Store mask `true` as canonical byte `1` and `false` as canonical byte `0`. BOOL remains the
  current one-byte `byte[]`/native `MemorySegment` carrier with no bit packing or noncanonical
  truth values.
- A dropped output is the represented positive zero of its value type, even when the input is
  negative zero, NaN, or infinity.
- For a kept element, compute `denominator = 1.0d - probability` once per invocation using one
  binary64 subtraction. Then:
  - FLOAT64 performs one binary64 division `input / denominator` and stores that result;
  - FLOAT32 widens the exact binary32 input to binary64, performs one binary64 division by the
    same denominator, and narrows once to binary32 for storage.
  No reciprocal multiplication, fused operation, probability narrowing, alternate evaluation
  order, or vector implementation is permitted in this task.
- This finite-precision order fixes scalar/parallel parity and the same-configuration replay
  boundary. It does not claim exact-real arithmetic, NaN payload preservation after kept
  arithmetic, or a cross-backend numerical order.
- Scalar Shape consumes one draw. Any zero extent gives `N == 0`: output and mask are empty, no
  element call or worker submission occurs, and next state contains the unchanged key/counter.
  Counter wrap uses ordinary two's-complement `long` addition and never fails.

### Generated execution and safe parallel strategy

- Add focused CPU-private `CpuRandomIr`, `CpuRandomLowering`, and `CpuRandomEmitter` owners.
  The IR distinguishes INITIAL_STATE from DROPOUT and retains the generator configuration/version,
  exact raw initializer words or probability bits, value type, ordered boundary roles/access
  plans, uniform conversion, comparison, finite-precision scaling, canonical BOOL representation,
  and state-advancement policy.
- Lower exactly one occurrence into one computation unit and one generated artifact. INITIAL_STATE
  boundaries are `[stateOutput]`. DROPOUT boundaries are `[value, state, output, keepMask,
  nextState]`; all five `ValueId` values are distinct and declared in that order.
- Generated entries retain the current direct carrier arguments, cold `long[]` geometry, and
  primitive `start`/`end`. The executable performs one generated prologue call before element
  work: INITIAL_STATE writes its two output words; DROPOUT writes next state exactly once. The
  dropout prologue uses the empty sentinel range `[0, 0)` and performs no draw or value/mask
  access. Non-empty element calls never write next state.
- Scalar execution invokes non-empty element range `[0, N)` after the prologue. Parallel-scalar
  uses deterministic disjoint row-major ordinal ranges after the same single prologue; every
  worker computes `word(key, counter, i)` directly from its global ordinal. No worker depends on
  a predecessor's generator state, and scheduling/chunk changes cannot affect values.
- INITIAL_STATE always uses scalar/single-thread execution. DROPOUT may select scalar or
  parallel-scalar under the existing worker/range thresholds. Vector and parallel-vector random
  execution are out of scope.
- No workspace, scratch, materialization, replay buffer, random-word buffer, mask staging,
  per-thread generator, or persistent mutable prepared state is permitted. Bounded resources are
  exactly one unit, one realized artifact, one output buffer for INITIAL_STATE or five distinct
  boundary buffers for DROPOUT, zero workspaces, and at most the existing four complete
  specialization candidates. Per-invocation setup uses only primitive locals; the element loop
  allocates nothing.

### Overlap, binding, and failure order

- Capability and lowering reject unresolved, negative, inconsistent, non-static, or non-injective
  writable geometry before resource declaration. Checked element counts, byte sizes, spans,
  address arithmetic, and counter advancement facts fail preparation on overflow except the
  intentional modulo-`2^64` state addition.
- Cold binding validates all carriers, byte sizes, alignments, read-only status, layout spans,
  workspace absence, and complete physical overlap before returning a bound invocation.
- INITIAL_STATE validates its complete writable output binding; it has no input/output or
  output/output pair. Zero inputs must be supported by specialization, preparation, finalization,
  generated signature, and executable code rather than represented by a dummy input.
- For DROPOUT, reject each of the three complete output spans against each of the two complete
  input spans, then reject all three output/output pairs. Use complete bindings, not selected
  range ordinals. Reject before the state prologue, any output/mask mutation, or worker submission.
- Physical input/input overlap is permitted: both inputs are read-only during element work and no
  semantic alias is inferred. Logical boundary `ValueId` values remain distinct.
- Binding and prepared access declarations identify the first two dropout boundaries as reads and
  the last three as writes. One bound invocation supplies direct typed references for all five
  boundaries and performs the prologue plus zero or more element calls. Runtime receives no
  Operation, CompiledNode, graph, RNG service, or mutable generator.
- Invalid or unsupported work fails before mutation. A thrown generated or worker failure may
  leave output buffers partially written under the existing executable failure contract, but
  hidden RNG state is impossible: replay starts solely from the unchanged explicit input state.

### Compatibility and documentation

- Advance generated compatibility from schema 18 to schema 19 with no migration reader.
- Stable structural identity and compatibility bytes include the random family, exact generator
  configuration name/version/constants and mapping, uniform conversion and threshold rule,
  probability raw bits for DROPOUT, initializer key/counter raw bits for INITIAL_STATE, value and
  boundary types, ordered output roles/count, access-plan structure, scaling order, canonical BOOL
  policy, prologue/state policy, execution compute mode, carrier pattern, and scratch absence.
  Concrete offsets, stride magnitudes, extents, slots, carrier objects, addresses, workers, run
  identity, and chunk count remain cold when they do not change emitted bytes.
- Extend capability, lowering dispatch, IR permits/encoding, route plan, preparation/finalization,
  generator validation/dispatch, specialization, executable binding, scalar reference, schema,
  package contracts, and focused tests only as this family requires.
- After executable Java stabilizes, hand the uncommitted diff and CPU-test evidence to a distinct
  clean documentation-focused context. It finalizes affected Javadocs/package summaries, the CPU
  backend guide, glossary, this task evidence, CPU master plan, and roadmap. Tensor, Compile,
  Runtime, and Training API guides are expected no-change because the public Model/compiler/
  runtime APIs do not change; the pass records the reasoned conclusion.

## Out of scope

- BFLOAT16 dropout arithmetic, FLOAT16, integral/BOOL dropout values, other random distributions,
  Bernoulli APIs, attention dropout, key derivation, split/fold-in/jump, or a public generator API
- a Model/cross-backend bitstream, public algorithm identifier, serialized RNG format, stable
  cross-version replay, cryptographic or statistical certification, or distinct-key disjointness
- more than one node, INITIAL_STATE-to-DROPOUT fusion, dropout fusion with pointwise work, general
  partition DAG decomposition, materialized splits, or saved-mask lifetime policy
- compiler capture/inference/autograd changes, gradient formulas, CSE/DCE changes, or hidden-mask
  policy; completed Compiler behavior is only inspected to confirm input/output ordering
- Model, shared Runtime/Prepare, Config, Trace, backend-contract, architecture, architecture-test,
  Gradle/dependency, native-route, tuning, benchmark, persistence-policy, backend-conformance,
  integration, Engine, NN, or training changes
- reflection, string dispatch, `java.util.Random`, `RandomGenerator`, `TensorRandoms`, a global,
  default, thread-local, or mutable prepared RNG, and per-element allocation

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [CPU 0005A partition-kernel reset](0005a-atomic-partition-kernel-architecture-reset.md)
- [CPU 0005D evidence gate](0005d-materialization-specialization-and-persistence-evidence-gate.md)
- [CPU 0006 static affine views](0006-portable-static-affine-views-and-boundary-materialization.md)
- [CPU 0006A movement](0006a-portable-pad-tile-and-tensor-composition-movement.md)
- [CPU 0006A2 indexing](0006a2-portable-gather-and-one-hot-indexing.md)
- [CPU 0006B1 scatter](0006b1-portable-functional-scatter.md)
- [CPU 0006B2 fold](0006b2-portable-overlap-fold.md)
- [CPU 0006C ordering](0006c-portable-stable-ordering-and-selection.md)
- [Model 0019B explicit graph RNG state](../../../modules/model/tasks/0019b-explicit-graph-rng-state-foundation.md)
- [Model 0019B1 explicit graph dropout](../../../modules/model/tasks/0019b1-explicit-graph-dropout-construction.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)

## Architecture constraints

- Model owns operation identity, raw state meaning, ordered roles, probability domain, logical draw
  count, keep rule, ideal inverted-dropout meaning, and portable state advancement. CPU owns only
  its private generator, finite-precision realization, IR, route, artifact, resources, and
  invocation.
- CPU analysis deterministically lowers and selects its supported route from projected facts,
  declares exact buffers and zero workspace before CPU-blind shared assignment, and finalizes one
  artifact only after assignments exist.
- Prepared recipes remain immutable and reusable. Every draw is a pure function of explicit state
  and logical index, so concurrent runs share no mutable RNG state and distinct `RunState` values
  retain ordinary isolation.
- Runtime hot execution receives direct bound carrier references and primitive geometry only. It
  performs no graph inspection, Operation/CompiledNode access, map lookup, reflection, string
  dispatch, service lookup, route selection, cache work, or generator mutation.
- Work remains within `backends/cpu`, adds no dependency or package, and preserves the existing
  generated portable route and optional artifact-store lifecycle.
- Capability must be no broader than complete lowering, exact declarations, assignment,
  generation, binding, overlap validation, scalar/parallel execution, and reference evidence.
- Stop if exact execution requires a Model/compiler/shared Runtime or Prepare change, if a
  zero-input or three-output occurrence cannot fit the current backend-owned seams without such a
  change, or if the finite-precision rules conflict with current Model semantics.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — sole supported capability provider.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — CPU-private structural random IR.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — one-node semantic revalidation and
  cold geometry.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct Class-File emission.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare`, `.route.portable`, `.cache`,
  `.executable`, `.memory`, and `.reference` — existing staged realization, carriers, execution,
  and independent reference boundaries.

Packages added or changed:

- No package is added, removed, moved, or exported. Existing CPU-internal packages gain only the
  focused random-family types and direct integration required above.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr` — immutable family, generator,
  numeric, BOOL, state, boundary-role, and cache identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRandomLowering` — exact one-node
  INITIAL_STATE/DROPOUT revalidation, declarations, logical-index geometry, layouts, and zero-
  workspace plan.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuRandomEmitter` — allocation-free
  generated initializer, dropout prologue, counter-to-word mapping, mask, scaling, and stores.

Tests mirror these production packages. No generic random utility, generator service, registry,
manager, facade, or new public CPU type is permitted.

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

Expected CPU tests:

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

Expected documentation/planning paths during implementation:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No Model/compiler/shared Runtime/Prepare/API/architecture/Gradle/conformance/integration path is
expected to change.

## Maximum scope

This task may create or modify at most 43 paths: 24 production/package paths, 14 CPU test paths,
and 5 documentation/planning paths. Stop before a 44th path, a new package, or any path outside
the listed CPU module and documentation/planning set. A stale existing CPU inventory test may
replace another listed test path but must not increase the ceiling.

If implementation requires a shared contract, public API, compiler change, BFLOAT16 arithmetic,
or more paths, stop and propose a focused follow-up or architecture decision. Do not hide
incomplete INITIAL_STATE or DROPOUT acceptance behind a follow-up.

## Acceptance criteria

- Capability and lowering admit exactly the specified one-node INITIAL_STATE and FLOAT64/FLOAT32
  DROPOUT matrix under complete static/resolved/injective guards; BFLOAT16 and every unsupported
  row remain truthfully false/fail-closed.
- INITIAL_STATE has exactly one output boundary, writes key then counter raw bits, performs no
  draw, and works for every word pair including signed extrema.
- DROPOUT boundaries, access roles, buffer declarations, assignments, generated parameters,
  direct bound fields, and stores are exactly `[value, state, output, keepMask, nextState]` with
  three writable outputs.
- Generator constants, V1 mapping, uniform conversion, threshold comparison, probability-zero
  behavior, test vectors, logical draw count, modulo wrap, and FLOAT64/FLOAT32 finite-precision
  scaling order match this specification exactly.
- Scalar and parallel-scalar executions are bitwise identical for output, canonical byte mask,
  and next-state words across different valid range counts and schedules. Random value `i`
  depends only on input key/counter and global logical ordinal `i`.
- Dense, offset, positive-strided, and zero-strided value reads; distinct-lane injective state
  reads; injective non-dense writes; heap/native/mixed carriers; scalar, empty, singleton,
  probability boundaries, signed zero, finite values, NaN, infinities, and counter wrap agree with
  an independent scalar reference.
- One generated prologue writes state once before any worker submission; empty dropout still
  writes next state and submits no element work. INITIAL_STATE uses the same generated family
  boundary without a dummy input.
- All required input/output and output/output overlaps fail during cold binding before prologue,
  mutation, or submission; permitted input/input overlap is tested.
- Zero workspace/materialization/replay scratch, bounded candidates, one realized artifact,
  immutable prepared state, and allocation-free element loops are enforced by construction and
  tests.
- Schema 19 rejects schema-18 artifacts. Random semantics/configuration and code-shaping facts are
  present in deterministic structural/fingerprint/compatibility identity; cold instance facts
  remain excluded as specified.
- Runtime hot execution contains no Operation/CompiledNode, graph inspection, map lookup,
  reflection, string dispatch, RNG service, mutable generator state, or per-element allocation.
- Existing pointwise, movement, indexing, scatter, fold, ordering, worker, cache, and persistence
  behavior remains unchanged.
- A separate documentation-focused context finalizes affected Javadocs, package summaries, CPU
  guide, glossary, planning evidence, and explicit API/architecture/conformance no-change
  conclusions before this task becomes Complete.

## Tests / validation

During implementation, run focused tests for random IR/lowering/generated execution, capability,
preparation/finalization, binding/overlap, scalar/parallel execution, cache/schema, carriers, and
independent reference vectors. After executable Java stabilizes, run one final module suite:

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

Repository-wide tests, architecture tests, backend conformance, and integration tests are deferred
to CPU 0009 or CI. Run them here only if implementation unexpectedly changes a repository-wide,
dependency, architecture, or reusable cross-backend contract; such a change is outside scope and
normally requires stopping first.

## Dependencies

- CPU 0005A–0006C: Complete, including typed carriers, zero-/multi-input movement foundations,
  workers, multi-output ordering, cold overlap validation, and schema 18.
- Model 0019B and 0019B1: Complete raw state, dropout probability/role/draw/advancement semantics.
- Compiler capture/inference/optimization and stochastic-gradient work: Complete and inspected only
  to confirm that INITIAL_STATE remains zero-input/one-output and DROPOUT remains ordered
  two-input/three-output with the auxiliary mask retained.
- Current staged Prepare and Runtime cold-binding contracts: Complete and unchanged.

## Follow-up tasks

- CPU 0007 remains the next Draft frontier and owns portable reduction, scan, statistics,
  softmax/log-softmax, and normalization coverage. Do not create its detailed specification here.
- CPU 0009 remains responsible for portable generated-coverage/conformance closure, including the
  intentional BFLOAT16 dropout gap unless an earlier separately planned CPU task establishes a
  conforming direct BFLOAT16 scaling rule.
- A later explicit cross-backend RNG decision may standardize a portable bitstream. It would be a
  Model/architecture-level decision and must not reinterpret this CPU-private V1 identity.

## Architecture impact

Expected impact: None.

This task uses the existing concrete-backend ownership of lowering, route selection, generated
artifacts, storage, and execution. It changes no module responsibility or dependency direction.
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
  mapping, conversion, and version enter artifact identity; it creates no Model promise.
- Probability and initializer word bits are baked semantic/code-shaping facts for this first
  bounded generated family. They enter IR/fingerprint/cache identity; dynamic dropout key/counter
  values remain explicit input data.
- State output uses one generated prologue call before element ranges. This produces exactly one
  next-state write for scalar and parallel execution and also handles empty dropout without
  workspace or mutable orchestration state.
- FLOAT64 and FLOAT32 use the exact binary64 denominator/division order above. BFLOAT16 remains
  fail-closed rather than adopting an unproved double-rounding path.
- BOOL mask storage is canonical one-byte `0`/`1`, consistent with current CPU carriers and Model
  BOOL semantics.
- Input/input physical overlap is harmless and permitted; every input/output and output/output
  overlap is rejected from complete spans before mutation.

## Known limitations

- Coverage is one fully static, resolved-layout occurrence. Dynamic Shapes/layouts, negative
  physical offsets/strides, multi-node fusion, vector random execution, native routes, and
  BFLOAT16 dropout are unsupported.
- The selected generator is not cryptographic and has no distinct-key stream-disjointness claim.
  Replay is bitwise only for this exact CPU V1 configuration and its fixed finite-precision rules,
  not across backends or future versions.
- Static element counts and resource sizes must fit the current checked host/preparation contracts;
  the Model's abstract modulo advancement for larger symbolic products remains outside this fully
  static CPU slice.
- `testing/backend-conformance` has no current executable cross-backend RNG harness. CPU 0009 owns
  the later closure rather than this task inventing one.

## Validation evidence

Implementation and clean-context documentation review are complete. Focused CPU tests cover
independent V1 vectors, capability, IR/lowering, generated initializer/dropout execution,
preparation/finalization, schema/cache, carrier binding, complete overlap rejection,
scalar/parallel replay, empty behavior, and an independent scalar reference. The authoritative
final CPU suite ran after executable Java first stabilized. A subsequent clean-context
implementation review corrected narrow acceptance gaps in key-offset setup, state-input
injectivity, structural generator identity, independent vectors, layout/carrier coverage,
threshold boundaries, and permitted input overlap. Per the one-authoritative-suite limit, those
corrections were validated with focused tests only. No executable Java changed during the final
documentation pass, so it did not rerun CPU tests.

- Focused affected-seam run: `BUILD SUCCESSFUL` in 1s; 122 tests, zero failures.
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
  pass, keeping the overall ceiling at 43.
- Clean documentation context `019ffcab-2c42-7d62-be4a-4b2815654c89` independently reviewed the
  complete 24-path production/Javadoc and 14-path focused-test diff, rendered Javadocs, Model
  0019B/0019B1 boundary, CPU guide, glossary, task, master plan, and CPU-specific roadmap passages.
  It finalized Javadoc lifecycle, ownership, failure, overlap, and parameter contracts; added the
  current CPU-private RNG/dropout guide and bounded-replay glossary clarification; and changed no
  executable behavior or test.
- Final `./gradlew :backends:cpu:javadoc`: `BUILD SUCCESSFUL` in 2s; 11 actionable tasks, 2
  executed and 9 up-to-date. The generated pages for `CpuRandomIr`, `CpuRandomLowering`,
  `CpuRandomEmitter`, `CpuPreparedExecutable`, and affected package summaries were inspected.
  Javadoc emitted only the two expected incubating-Vector-module warnings and no documentation
  warnings.
- Local Markdown validation passed for all five authorized files: every local target and heading
  anchor resolved, fences were balanced, each file had exactly one terminal newline, and no file
  had trailing whitespace.
- Exact transcription and coherence checks passed for all five V1 vectors, schema 19, BFLOAT16
  fail-closed wording, 0006D Complete status, CPU 0007 Draft status with no detailed task file,
  package/type placement, and the exact 43-path CPU task scope.
- Final combined-worktree `git diff --check` passed. Concurrent NN/training, settings, and
  architecture-test paths were excluded from the CPU count and preserved without modification by
  this documentation context.
- Reasoned no-change review: Tensor, Compile, Runtime, and Training API guides remain accurate
  because this task changes only CPU-private realization, not shared or public APIs. The
  architecture contract, focused architecture pages, ADRs, and architecture tests remain accurate
  because ownership, lifecycle, and dependency direction do not change. Model, Compiler, shared
  Runtime/Prepare/Config/Trace/backend-contract, Gradle/dependencies, backend conformance,
  integration, native routes, fusion, tuning, Engine, NN, and training require no task change
  because none of their contracts or executable behavior is modified. CPU 0009 or CI retains the
  deferred cross-backend/repository checkpoint.

## Implementation notes

Implementation adds CPU-private `CpuRandomIr`, `CpuRandomLowering`, and `CpuRandomEmitter` with
direct integrations through capability, schema 19, portable generation, preparation,
finalization, execution, and scalar reference seams. INITIAL_STATE uses one zero-range generated
prologue; DROPOUT uses that prologue for next state followed by deterministic global-ordinal
element ranges and zero workspace. BFLOAT16 remains fail-closed.

## Completion summary

- Completed changes: added one-node static resolved INITIAL_STATE and FLOAT64/FLOAT32
  explicit-state DROPOUT through the portable generated route with CPU-private V1 replay,
  canonical BOOL mask, modulo state, scalar/parallel parity, zero workspace, complete overlap
  rejection, and schema 19; BFLOAT16 remains fail-closed.
- Files changed or created: 24 CPU production/package paths, 14 CPU tests, and exactly five
  documentation/planning paths, for 43 authorized CPU-task paths.
- Tests and validation: reused the authoritative 47-suite/275-test CPU result and focused
  122-/71-/9-/46-test results; final CPU Javadoc, rendered-page inspection, Markdown, scope,
  vector, schema, status, package placement, and whitespace checks passed.
- Documentation-agent review: clean context `019ffcab-2c42-7d62-be4a-4b2815654c89` completed the
  independent finalization without changing executable behavior or rerunning CPU tests.
- Documentation impact: CPU guide, glossary, task, CPU master plan, and CPU roadmap status are
  synchronized to the implemented boundary.
- Javadoc review: affected types, constructors, methods, and package summaries are accurate and
  rendered without documentation warnings.
- Glossary impact: bounded replay now identifies the CPU V1 boundary without exposing it as public
  configuration or portable Model semantics.
- Unresolved issues: None within task scope. The documented BFLOAT16 and cross-backend
  conformance limitations remain intentional follow-up boundaries.
- Follow-up required: None for CPU 0006D. CPU 0007 is the next Draft frontier and has no detailed
  specification.

Status: Complete
