# Task 0005J: Bounded Pointwise Coverage and Parity Hardening

## Status

Complete

## Goal

Close the highest-value inconsistencies inside the already implemented CPU pointwise family before
portable layout/indexing work begins. The task adds only vector realizations whose current Model
semantics and Java 26 Vector API behavior are sufficient, makes semantic support and execution-mode
eligibility exact and data-driven, and preserves scalar fallback for every admitted occurrence that
cannot use vector compute.

The bounded result is:

```text
implemented Model pointwise occurrence
  -> exact occurrence capability by kind, attributes, DataType, Shape, and layout
  -> existing typed CPU opcode and immediate validation
  -> exact vector eligibility by opcode, value role, access regime, carrier, and topology
  -> scalar | vector x single-thread | parallel
  -> one existing portable generated artifact and partition executable
```

This is a parity-hardening step, not total Cartesian coverage. It must not hold reductions, matrix
multiplication, or other operation families behind pointwise cases that need a new semantic policy
or a larger representation decision.

## Scope

### Exact support and eligibility ownership

- Keep `CpuCapabilityProvider` as the sole supported public CPU capability surface. It continues to
  answer occurrence-local semantic support from exact Model kind, attributes, input/output
  descriptors, data types, Shapes, and resolved layouts. It does not expose route, vector species,
  carrier, access-regime, fusion, or execution-mode data.
- Keep `CpuPointwiseOpcode` as the closed CPU-private semantic vocabulary. Replace its coarse
  opcode-only vector flag with the smallest typed internal metadata needed to distinguish direct
  value-vector operations, mask producers, mask combiners, mask consumers, and scalar-only rows.
  Data type, immediate shape, power realization, topology, and boundary facts remain additional
  gates rather than being collapsed into one public registry.
- Keep complete execution eligibility in CPU analysis. It combines the validated opcode/IR facts
  with exact `DataType`, internal-versus-boundary BOOL role, all five `CpuAccessPlan` regimes,
  ordered primitive-array/`MemorySegment` carriers, element count, preferred species, and the
  requested compute/orchestration axes before selecting one of the existing four modes.
- Add no public support matrix, public route registry, operation registry, generic feature bag,
  reflection, annotation discovery, string dispatch, or `Map<String, Object>`.
- Make the conformance matrix data-driven in tests. Every row records the Model kind and exact
  attributes, CPU opcode, admitted data types, scalar support, vector support form, boundary or
  virtual result role where relevant, access regime, carrier pattern, and all four execution
  modes. Production code remains typed and closed; tests are the exhaustive evidence table.

### Added floating value-vector parity

For homogeneous FLOAT32 or FLOAT64 IR, add preferred-species vector and parallel-vector
realization for these already supported opcodes:

| Model semantics | CPU opcodes | Vector realization |
|---|---|---|
| Model `MINIMUM` / `MAXIMUM` | `MIN`, `MAX` | lane-wise Java `min`/`max`, preserving Model NaN propagation and directional signed zero |
| scalar minimum/maximum | `SCALAR_MIN`, `SCALAR_MAX` | lane-wise scalar `min`/`max` with exact typed immediate bits |
| first-class clamp | `SCALAR_CLAMP` | ordered lane-wise `MAX(input, lower)` then `MIN(result, upper)`; one opcode and one IR instruction |
| rectified linear unit | `RELU` | lane-wise `MAX(input, +0)` with the completed NaN and positive-zero behavior |
| sign | `SIGN` | comparisons and blends that preserve both zero signs, map finite/infinite nonzero values to exact `-1` or `+1`, and classify NaN as NaN |
| same-type cast | `CAST` | represented-value identity for homogeneous FLOAT32/FLOAT64 chains |

The direct Java vector extrema operations are eligible only after focused special-value tests prove
the current `Math.min`/`Math.max`-equivalent NaN and signed-zero rules for both precisions. Clamp
retains lower-then-upper order. Sign must not use a conversion whose zero, infinity, or NaN behavior
differs from the completed scalar contract.

Preserve all twenty-one CPU 0005I vector opcodes and their existing numerical bounds. Do not alter
scalar/reference realization, ERF/GELU coefficients, or exact/default numerical mode.

### Added INT32 and INT64 value-vector parity

For homogeneous INT32 or INT64 IR, add preferred-species `IntVector` or `LongVector` realization
for exactly:

- `ADD`, `SUB`, and `MUL` with the current fixed-width modular result;
- `MIN` and `MAX` with exact signed order;
- `SCALAR_ADD`, `SCALAR_SUB`, `SCALAR_MUL`, `SCALAR_MIN`, and `SCALAR_MAX` with exact same-typed
  immediate bits; and
- same-type `CAST` as represented-value identity.

Integral DIV, POW, CLAMP, unary floating operations, promotion, widening, saturation, and mixed
numeric chains remain unsupported or scalar according to their existing semantic matrix. Species,
lane count, carrier descriptor, specialization identity, generated validation, and scalar tail
must use the exact integral type.

### Canonical BOOL vector parity

- Add preferred-species `ByteVector` realization for homogeneous canonical-BOOL `LOGICAL_AND`,
  `LOGICAL_OR`, `LOGICAL_NOT`, and same-type `CAST` chains.
- Preserve the cold boundary rule that externally supplied BOOL bytes are exactly `0` or `1`.
  Bitwise/vector realization therefore consumes and produces canonical bytes without numeric
  truthiness or repeated hot validation.
- BOOL vector chains may have BOOL array, BOOL `MemorySegment`, or representative mixed boundary
  carriers under the existing access rules. They retain scalar fallback for a general odometer or
  any other vector-ineligible access.
- Add no BOOL arithmetic, public bitwise family, alternate BOOL representation, or changed storage
  contract.

### Fused internal mask and WHERE boundary

- Introduce a generated-body-only `VectorMask` representation for a BOOL IR value only when that
  value is virtual inside the existing one-through-eight straight-line unit.
- Floating `GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`, `LESS_OR_EQUAL`, `EQUAL`, and
  `NOT_EQUAL` may produce an internal mask from homogeneous FLOAT32 or FLOAT64 vectors.
- `IS_FINITE`, `IS_NAN`, and `IS_INF` may produce the same matching typed internal mask.
- `LOGICAL_AND`, `LOGICAL_OR`, and `LOGICAL_NOT` may combine internal masks without converting
  them to byte vectors.
- Floating `WHERE` may consume an internal matching mask and blend same-typed FLOAT32 or FLOAT64
  branch vectors. A scalar/all-zero canonical-BOOL boundary condition may also be broadcast to a
  matching mask when the generated implementation remains direct and allocation-free.
- Every internal mask must be single-unit virtual state: it has no Runtime slot, boundary
  declaration, publication obligation, cross-partition use, or materialized store. Mask type and
  lane count exactly match the numeric species that produces or consumes it.
- A materialized comparison/classification result, a non-scalar external BOOL condition for
  floating `WHERE`, and a mask crossing a unit or partition remain scalar. The task does not add a
  general byte-vector-to-arbitrary-numeric-mask conversion or per-lane extraction/reinsertion.
- Mixed eligible and ineligible mask/value instructions cause deterministic whole-unit scalar
  fallback. Supported capability must not fail merely because vector realization is unavailable.

### Access, carrier, and execution-mode matrix

The final conformance matrix covers every selected semantic row across the applicable logical
types and these structural dimensions:

| Dimension | Required rows |
|---|---|
| Storage carrier | exact primitive array for the logical type, native-order `MemorySegment`, and representative mixed ordered boundary patterns |
| Access regime | `DENSE_LINEAR`, read-only `SCALAR_ALL_ZERO`, `LAST_AXIS_BIAS`, `BLOCK_OUTER`, and `GENERAL_ODOMETER` |
| Compute/orchestration | scalar single-thread, vector single-thread, scalar parallel, and vector parallel |
| Fallback | too-short range, general odometer, unsupported opcode/type, direct scalar power, unsafe mask boundary, and mixed vector types select scalar compute without changing semantic support |
| Range | zero elements, arbitrary nonzero `start`/`end`, full vectors, block boundaries, worker boundaries, and scalar tails |

Vector access keeps the completed rules: dense direct runs; scalar/all-zero read broadcast;
last-axis or block/outer complete contiguous runs; and scalar fallback for general odometers. No
gather, scatter, masked tail, or general odometer vectorization is added. Parallel orchestration
remains orthogonal: the same selected scalar or vector generated artifact runs either on the
invoking thread or over deterministic disjoint CPU-private worker ranges.

### Compatibility and lifecycle

- Advance `CpuGeneratorSchema.CURRENT_VERSION` from 9 to 10 because opcode eligibility metadata,
  lane types, carrier descriptors, mask locals, and emitted bodies change.
- Schema 10 rejects schema 9 and older private artifacts without a migration reader.
- Extend existing specialization/verification facts only as required to distinguish FLOAT32,
  FLOAT64, INT32, INT64, and BOOL vector bodies, exact preferred species, ordered carriers, and
  internal mask topology. Do not duplicate facts already present in boundary types, IR identity,
  carrier patterns, or species bits.
- Preserve deterministic class bytes, post-assignment finalization, optional persistence disabled
  by default, one-through-eight straight-line fusion, virtual values, one final materialized store,
  arbitrary ranges, scalar tails, four complete candidates, one realized artifact, zero fixed-
  shape variants, zero unrolled variants, cold binding, and concurrent-run isolation.
- Preserve CPU 0005D's FLOAT64-only optional one-input materialization. This task adds no FLOAT32,
  integral, BOOL, mask, output, or broad materialization candidate.

## Out of scope

- vector `FLOOR` or `CEIL`; Java 26 exposes no selected direct floating lane operator, and integer
  conversion would require a new proof for out-of-range values, infinities, NaNs, signed zero, and
  exact floating results
- vector `SIGMOID`, `GELU_TANH_APPROXIMATION`, or `SILU`; they remain scalar until a later bounded
  numerical/vector task fixes and proves an exact typed formula matrix
- general Tensor/Tensor or direct scalar `POW` vectorization, per-lane scalar power calls, multiply
  chains, exponentiation by squaring, or another power policy
- materialized comparison/classification masks, general external BOOL-mask loading for floating
  `WHERE`, mask publication, mask transfer, mask storage redesign, or a public mask type
- cross-type `CAST`; the live Model permits expression construction but deliberately defines no
  conversion, rounding, saturation, overflow, NaN, infinity, signed-zero, BFLOAT16, or BOOL policy
- BFLOAT16 or FLOAT16 execution, mixed precision, mixed-type vector IR, CPU-side promotion, or a
  new data type
- vector gather/scatter, masked tails, general-odometer vectorization, broad materialization,
  fixed-shape specialization, unrolling, new fusion forms, or general partition-DAG decomposition
- layout/indexing/random, reductions/scans/softmax/statistics/normalization, matrix multiplication,
  convolution/pooling/attention/loss, native/vendor routes, tuning, benchmarks, or later families
- Model, Compiler, Config, Planning, shared Prepare, Runtime, Backend Contract, Trace, Engine,
  OpenBLAS provider, Metal, CUDA, public API, dependency, Gradle, architecture, backend-conformance,
  or integration changes unless implementation proves an existing required contract is violated;
  if so, stop and report rather than widening this task
- a detailed specification for CPU 0006 or any later task, commits, or pushes

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md)
- [Runtime / Prepare / Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Planning guide](../../../planning-guide.md)
- [Implementation roadmap](../../../roadmap.md)
- [CPU backend master plan](../master-plan.md)
- [CPU 0005E pointwise family expansion](0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md)
- [CPU 0005G extrema, clamp, power, and logical coverage](0005g-extrema-clamp-tensor-power-and-logical-coverage.md)
- [CPU 0005H unary closure](0005h-portable-unary-transcendental-and-activation-closure.md)
- [CPU 0005I FLOAT32 vector parity](0005i-float32-vector-parity-and-vector-emission-boundary.md)
- [Model cast semantic contract](../../../modules/model/tasks/0015g-cast-semantic-kind-and-attributes.md)
- [Model cast expression contract](../../../modules/model/tasks/0015h-cast-tensor-expression.md)

## Architecture constraints

- Model remains the only owner of operation meaning. CPU eligibility may only realize the current
  exact contracts; it must not infer missing cast or numerical semantics from Java primitives.
- Planning selects CPU ownership. CPU analysis owns opcode lowering, vector/mask eligibility,
  route and execution-mode selection, specialization, and exact declarations.
- Shared Prepare sees one opaque selected CPU plan and exact requirements. Runtime receives one
  prepared executable and performs no operation, type, mask, carrier, vector, or route selection.
- Generated hot code sees no Model `Operation`, `CompiledNode`, graph, registry, reflection,
  string dispatch, generic carrier array, or policy lookup.
- Scalar and Vector API remain implementation routes inside the one CPU backend.
- No supported public CPU type, module dependency, lifecycle owner, or architecture rule changes.
  Stop and report if implementation requires one.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — occurrence-local semantic capability and its exhaustive
  test matrix; no new public type.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — typed opcode execution-form metadata,
  value/immediate validation, and virtual-mask topology facts.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — unchanged Model-to-opcode ownership
  plus exact virtual/boundary classification needed for mask eligibility.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — complete access/carrier/type/topology
  eligibility and selection of the four existing execution modes.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — typed float/double/int/long/byte
  vector carrier emission, internal masks, comparisons, logic, selection, scalar tails, and class
  generation.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema and specialization compatibility.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — review/test of parallel execution;
  no new production responsibility.

Packages added, removed, or moved:

- None.

Type placement:

- Extend the existing `CpuPointwiseOpcode` metadata and existing prepare selection rather than
  adding a public registry or one class per operation.
- Generated `VectorMask` locals remain a code-generation representation of virtual BOOL IR values;
  they are not a new Model, shared, Runtime, or stored CPU type.
- No new production type is planned. Stop and revise this task if a correct implementation needs
  another production abstraction or package.

## Affected files

Expected CPU production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorInstructionEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorMath.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcode.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`

Expected CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuFusedGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorMathTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcodeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`

Expected explanatory documentation and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- this task specification
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review-only paths include `CpuCapabilityProvider`, scalar/reference emitters and tests, current
access/materialization/route/memory/finalization contracts, Model operation/DataType/cast
contracts, public API guides, Config/Planning/Compiler/Prepare/Runtime contracts, Gradle,
architecture/ADR/tests, backend conformance, integration, native providers, and later CPU work.

## Maximum scope

This task may create or modify at most 32 paths:

| Category | Maximum | Accounting |
|---|---:|---|
| CPU production/package | 15 | Exact fifteen listed paths; no new production type |
| CPU tests | 12 | Exact twelve listed paths |
| Explanatory documentation | 2 | CPU backend guide and glossary |
| Planning/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **32** | **15 + 12 + 2 + 3** |

The task exceeds the ordinary guardrail for one technical reason: the same code-shaping lane type
and mask role must remain consistent across IR validation, lowering topology, prepare eligibility,
carrier descriptors, emitted locals/instructions/stores, specialization compatibility, parallel
execution, and one exhaustive conformance matrix. Splitting those seams would temporarily permit
an advertised vector mode that finalization cannot realize. No public or shared type is added.

Unused authorized paths may remain unchanged. Stop and revise planning if another production,
test, documentation, build, or module path is required.

## Acceptance criteria

- Public capability remains exact for all forty-eight current opcodes' Model occurrences and every
  adjacent rejected kind/type/attribute/Shape/layout row; vector eligibility neither broadens nor
  narrows semantic support.
- Internal eligibility is exact by opcode, DataType, immediate/power facts, virtual or boundary
  BOOL role, access regime, carrier, element count, preferred species, and requested execution
  mode. No broad public registry or generic parameter map exists.
- FLOAT32/FLOAT64 MIN, MAX, scalar extrema, ordered CLAMP, RELU, SIGN, and same-type CAST execute
  under vector and parallel-vector modes with exact special-value and signed-zero behavior.
- INT32/INT64 binary and scalar ADD/SUB/MUL/MIN/MAX plus same-type CAST execute under vector and
  parallel-vector modes with exact modular arithmetic and signed ordering.
- Canonical BOOL AND/OR/NOT plus same-type CAST execute under vector and parallel-vector modes over
  exact byte arrays and segments without creating noncanonical results.
- Floating comparisons/classifications may form internal typed masks; internal logical operations
  preserve those masks; and floating WHERE consumes a matching internal or scalar-broadcast mask.
  No internal mask is declared, stored, published, or allowed to cross a unit/partition.
- Materialized mask results, non-scalar external WHERE conditions, FLOOR, CEIL, complex activation
  vector rows, and direct/general POW select scalar compute without losing parallel orchestration.
- The data-driven tests cover heap arrays and `MemorySegment`, representative mixed carriers, all
  five access regimes, all four scalar/vector and single/parallel combinations, arbitrary ranges,
  zero work, scalar tails, fused lengths one and eight, and every specified fallback reason.
- `GENERAL_ODOMETER` remains a correct scalar fallback. No gather, masked tail, per-lane scalar
  call/reinsertion, or broad materialization appears.
- Schema 10 rejects older artifacts. Structural identity changes for lane type/species, opcode and
  mask topology, ordered carrier/access form, strategy, and existing code-shaping facts, while
  retaining all current exclusions for extents and runtime identities.
- One-through-eight fusion, one final store, one-copy/four-candidate/one-artifact/zero-fixed-shape/
  zero-unroll budgets, optional persistence policy, cold binding, and concurrent isolation remain
  unchanged.
- Cross-type CAST and BFLOAT16/FLOAT16 execution remain fail-closed. No Model, Compiler, Runtime,
  shared Prepare, Config, Planning, dependency, build, architecture, conformance, integration,
  native, tuning, or later-family change appears.
- A separate clean documentation-focused pass finalizes affected Javadocs/package summaries, CPU
  guide, glossary, task evidence, master plan, and roadmap; records reasoned no-change conclusions;
  and reuses stabilized Java evidence unless executable behavior changes.
- CPU 0005A–0005I remain `Complete`; CPU 0005J remains `Ready` until implementation, validation,
  documentation review, and final synchronization pass; CPU 0006–0017 remain `Draft` without
  detailed specifications.

## Tests / validation

Run the focused implementation matrix while executable Java changes:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcodeTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPointwisePartitionLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuClassFileKernelGeneratorTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuFusedGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuVectorMathTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecializationTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest
```

After executable Java stabilizes, run exactly one final affected-module suite:

```bash
./gradlew :backends:cpu:test
```

Do not rerun CPU 0005D's opt-in persistence timing experiment. Schema compatibility changes, but
the default `KEEP_DISABLED` policy and its performance question do not.

The separate documentation-focused pass runs after final Javadocs and Markdown:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates local Markdown links and anchors, balanced fences, final newlines, exact 32-path
scope, the 48-opcode semantic inventory, schema 10, all selected type/opcode/vector/mask rows, all
five access regimes and four execution modes, fallback/exclusion rows, unchanged budgets, status
synchronization, and the absence of Java/test changes outside `backends/cpu` or any Gradle,
architecture, shared-module, conformance, integration, native, or later-task path.

Repository-wide validation is deferred to CPU 0009 or continuous integration. This task changes
one concrete backend's private realization and no dependency or shared contract. Current
CPU-internal generated/reference tests remain the proportionate conformance evidence until the
composed Engine/backend harness exists.

## Dependencies

- Complete CPU 0005A–0005I partition, access, strategy, materialization, pointwise, numerical,
  vector-emission, artifact, and lifecycle contracts.
- Current exact Model arithmetic, extrema/clamp, unary, comparison, classification, BOOL logical,
  WHERE, same-type CAST, `DataType`, `ScalarValue`, Shape, and layout contracts.
- Java 26 Class-File API and incubating Byte/Int/Long/Float/Double Vector API contracts.
- Current staged Prepare, Runtime cold binding/execution, and backend-contract ownership.

## Follow-up tasks

- CPU 0006 remains Draft for portable layout, indexing, ordering, and explicit-state random family
  coverage and follows this bounded closure.
- CPU 0007 remains Draft for reductions, scans, softmax/log-softmax, statistics, and normalization;
  it must not wait for the deferred pointwise cases below.
- The CPU master plan records a named non-blocking remaining-vector follow-up for FLOOR/CEIL,
  complex activations, materialized/external mask handling, and any later evidence-backed
  pointwise vector rows. No detailed specification is created here.
- The CPU master plan records a separate named cross-type CAST follow-up gated by a future Model
  numerical conversion contract. No backend may infer that policy from Java casts or Vector API
  conversions, and no detailed specification is created here.
- CPU 0009 remains the portable generated-coverage checkpoint and explicitly includes CPU 0005J.

## Architecture impact

Expected impact: None.

The task changes only CPU-owned eligibility, specialization, and portable generated code. It does
not move ownership, add a dependency, change lifecycle staging, add a backend identity, or alter a
public/shared contract. Stop and report if implementation proves otherwise.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, completed
CPU tasks 0005A–0005I, and
docs/planning/backends/cpu/tasks/0005j-bounded-pointwise-coverage-and-parity-hardening.md in full.
Inspect every affected and review-only contract named by task 0005J before editing.

Implement CPU 0005J exactly inside its exact 32-path map. Add only the specified floating extrema/
clamp/ReLU/sign, signed-integral arithmetic/extrema, canonical-BOOL, same-type CAST, and narrowly
virtual floating-mask/WHERE vector parity. Make support and fallback evidence data-driven across
types, attributes, access regimes, carriers, and all four execution modes. Preserve scalar
fallback and every completed fusion/materialization/artifact/lifecycle budget. Stop on any Model
semantic, mask-storage, architecture, dependency, affected-file, or maximum-scope conflict.

Do not implement FLOOR/CEIL vector algorithms, general POW, cross-type CAST, BFLOAT16/FLOAT16,
general external/materialized masks, gather/odometer vectorization, broad materialization, later
families, native/tuning work, shared/build/architecture changes, commits, or pushes.

After focused tests and exactly one final CPU suite pass, hand the actual diff and exact Java
evidence to a separate clean-context documentation-focused agent/thread. That pass follows
documentation-rules.md, independently finalizes affected Javadocs, CPU guide, glossary, task/
master/roadmap evidence and documentation/scope checks, and reuses successful Java evidence unless
executable behavior changes or a concrete stale-evidence risk is recorded.

Do not mark 0005J Complete until both passes and every acceptance criterion succeed. Leave CPU
0006 and every later CPU task Draft without detailed specifications.
```

## Local decisions

- Insert 0005J after completed 0005I and before the existing Draft CPU 0006 family row. Reusing or
  renaming 0006 would erase the established layout/indexing owner; 0005J is the next ordered task.
- Include floating extrema/clamp/ReLU/sign because Java 26 supplies direct vector primitives or a
  small exact mask/blend realization that can be proved against current Model semantics.
- Exclude FLOOR/CEIL because Java 26 supplies no selected direct lane operator and conversion-based
  emulation needs a separate exceptional-value and range proof.
- Include INT32/INT64 arithmetic/extrema and canonical BOOL logic because their modular, signed-
  order, and canonical-byte contracts require no new numeric policy.
- Permit `VectorMask` only for virtual in-unit BOOL values. This safely enables comparison/
  classification/logical-mask-to-WHERE chains without inventing a stored mask representation.
- Keep materialized masks and general external WHERE conditions scalar. General mask storage and
  lane-count conversion are larger carrier/representation decisions and are not required for this
  high-value fused path.
- Keep direct/general POW scalar. A Vector API token or per-lane call is not by itself a proof of
  the completed exact/default power contract or a useful portable vector route.
- Keep cross-type CAST fail-closed because Model intentionally leaves conversion results undefined.
  This is a missing semantic prerequisite, not a CPU implementation omission.
- Preserve FLOAT64-only materialization and bump the private generator schema once to 10.

## Known limitations

- Pointwise execution remains fully static, resolved-layout, and limited to one connected
  one-through-eight straight-line unit with one final materialized output.
- General odometers and too-short runs remain scalar. No gather or masked tail is available.
- Materialized comparison/classification results and non-scalar external BOOL conditions for
  floating WHERE remain scalar; only virtual internal masks and scalar BOOL broadcasts vectorize.
- FLOOR, CEIL, complex activation vector formulas, direct/general POW, cross-type CAST, BFLOAT16,
  and FLOAT16 remain deferred under the explicit follow-up gates.
- Optional generated-class persistence remains disabled by default. Native routes, tuning, broader
  fusion, reductions, heavy families, and Engine composition remain later work.

## Validation evidence

Planning evidence before promotion to `Ready`:

- Read `AGENTS.md`, the complete architecture contract and current architecture index, planning
  guide and roadmap, documentation rules plus General/Planning profiles, CPU master plan, completed
  CPU 0005E–0005I task records and the directly relevant 0005A–0005D access/strategy history,
  current CPU capability/opcode/IR/lowering/prepare/vector/carrier/reference/tests, and current
  Model DataType, BOOL, comparison, WHERE, and CAST contracts.
- Confirmed CPU 0005I is Complete and existing CPU 0006 is the next Draft family owner. Therefore
  the correct inserted task number is 0005J; CPU 0006 is not repurposed.
- Confirmed Java 26 exposes direct vector MIN/MAX, comparisons, floating classifications, mask
  operations, and signed-integral/byte vector arithmetic, but no selected direct FLOOR/CEIL lane
  operator. The task requires focused semantic proof rather than assuming operator presence alone.
- Confirmed Model CAST permits all source/target expression pairs but deliberately defines no
  numerical conversion policy. Cross-type CPU CAST remains fail-closed and receives an explicit
  prerequisite-gated follow-up.
- Confirmed the bounded virtual-mask design does not add a stored representation or shared/runtime
  contract. Materialized and general external masks retain scalar fallback.
- Confirmed the expected implementation map fits the existing package structure with no new
  production type or package and a 32-path ceiling.
- `javap --module jdk.incubator.vector` confirmed the installed Java 26.0.1 Byte/Int/Long vector
  species and the selected arithmetic, extrema, comparison, blend, and mask operations. Inspection
  of the installed Java 26 Vector sources confirmed floating MIN/MAX use `Math.min`/`Math.max`
  lane semantics and that the selected direct FLOOR/CEIL operators are absent.
- Planning validation passed for exactly this new task, the CPU master plan, and the roadmap. A
  local-link check resolved 401 links; balanced-fence, final-newline, and trailing-whitespace
  checks passed; the canonical task headings are present; exactly CPU 0005J is `Ready`; CPU 0006
  has no detailed task file and remains `Draft`; and the 15 + 12 + 2 + 3 path accounting equals
  the 32-path ceiling.
- `git diff --check` passed for tracked edits. The separate no-index whitespace check for this new
  task produced no diagnostic and returned the expected difference status. Final status contained
  only this task plus the CPU master plan and roadmap; no Java, test, Gradle, architecture,
  conformance, integration, API, guide, glossary, or unrelated worktree path changed.
- No Java or Gradle test task was run because this is a planning-only change.

Implementation evidence:

- `./gradlew :backends:cpu:compileJava :backends:cpu:compileTestJava` passed.
- The exact required twelve-class focused command passed 12 suites and 76 tests with zero
  failures, errors, or skips. Additional targeted generated-kernel, vector-math, specialization,
  and preparation runs passed.
- Exactly one final `./gradlew :backends:cpu:test` passed 26 suites and 125 tests with zero
  failures or errors and one skipped opt-in persistence-evidence test. CPU 0005D's timing
  experiment was not repeated.
- The implementation changed the exact 15 authorized CPU production/package paths and 6 of the
  12 authorized CPU test paths. Together with the three pre-existing planning paths, handoff had
  24 changed paths, no new production type, and no unauthorized path. Implementation handoff
  `git diff --check` passed.
- Focused development evidence exposed that `ByteVector.not()` produces byte `0xFF` for logical
  true. The emitted BOOL NOT path was corrected to mask the complement with byte `1`, so every
  materialized BOOL vector result remains canonical `0` or `1`; all final validation then passed.

Documentation-focused evidence:

- Mandatory clean documentation context `/root` read the repository instructions, architecture
  contract and focused dependency/runtime/prepare/backend documents, documentation rules and the
  General, API/Javadoc, Backend Guide, Planning, and glossary guidance, planning guide, roadmap,
  CPU master plan, completed CPU 0005A–0005I records, this task, the actual diff, affected final
  production/test sources, generated CPU Javadocs, CPU guide, glossary, and necessary review-only
  Model and CPU contracts.
- The pass finalized affected Javadocs/package summaries, the CPU guide and glossary, this task,
  CPU master plan, and roadmap. It documented the exact floating, integral, canonical-BOOL, and
  virtual-mask vector rows; exact preferred species and schema-10 identity; and deterministic
  scalar/parallel-scalar fallback boundaries without broadening semantics.
- The stabilized Java evidence above was reused because the documentation pass changed no
  executable Java behavior and identified no stale-evidence risk. Final CPU Javadoc passed with
  only the two expected incubating-Vector warnings. Repository-local validation checked 290
  Markdown files, 5,453 local links, and 353 heading-anchor references; links, anchors, fences,
  final newlines, and trailing whitespace all passed. Exact authorized-path/32-path ceiling,
  semantic/schema/fallback/status, unchanged-layer, and `git diff --check` validations passed.
- Final scope is 26 changed paths: 15 authorized CPU production/package paths, 6 authorized CPU
  test paths, 2 explanatory documentation paths, and 3 planning/status paths. No production type
  or package was added.
- Public Tensor, Compile, Runtime, and Training API pages remain accurate because CPU 0005J only
  realizes existing Model semantics inside a private backend route. `ARCHITECTURE.md`, focused
  architecture documents and ADRs, Gradle, other modules, native providers, architecture tests,
  backend-conformance, integration tests, and later detailed task specifications remain unchanged
  because no dependency, ownership, public API, shared contract, or composed execution boundary
  changed.

## Implementation notes

The implementation retained all forty-eight semantic opcodes and replaced the coarse vector flag
with a closed internal vector-form role. Preparation derives one exact FLOAT32, FLOAT64, INT32,
INT64, or BOOL lane type, verifies the matching preferred species, and admits virtual masks only
for internal floating BOOL values or a scalar/all-zero external WHERE condition. Schema 10 makes
the new typed bodies and mask topology incompatible with every older stored artifact.

Floating extrema preserve NaN and signed-zero behavior, clamp evaluates lower maximum before upper
minimum, ReLU uses positive zero, and sign preserves signed zero and NaN. Integral arithmetic is
fixed-width modular and extrema use signed order. Canonical BOOL uses byte vectors and explicitly
masks NOT back to `0` or `1`. General odometers, too-short ranges, unsupported opcode/type pairs,
direct scalar power, materialized masks, non-scalar external WHERE conditions, and mixed unsafe
topologies select scalar or parallel-scalar compute without changing semantic support.

## Completion summary

Completed the bounded CPU pointwise parity hardening without changing the public capability
surface, semantic inventory, architecture, dependencies, or shared lifecycle contracts. Added the
specified preferred-species floating, signed-integral, canonical-BOOL, and virtual-mask vector
realizations; advanced private generated compatibility to schema 10; retained deterministic
scalar fallback, scalar tails, one-through-eight straight-line fusion, one final store,
FLOAT64-only one-input materialization, four candidates, one artifact, and zero fixed-shape or
unrolled variants. Final validation and documentation/status synchronization passed. CPU 0005A–
0005J are Complete; CPU 0006 and every later task remain Draft without a detailed specification.

No unresolved issue or follow-up is required for CPU 0005J. Deferred vector-policy cases and
cross-type cast remain owned by the named future gates; general partition-DAG decomposition and
bounded vertical/horizontal fusion remain Draft CPU 0008A work.

Status: Complete
