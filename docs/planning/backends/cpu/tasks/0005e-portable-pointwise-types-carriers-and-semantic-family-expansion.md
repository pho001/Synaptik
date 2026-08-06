# Task 0005E: Portable Pointwise Types, Carriers, and Semantic-Family Expansion

## Status

Complete

## Goal

Deliver the first bounded expansion from the completed FLOAT64 `ADD -> exact GELU -> MUL`
proving topology to a reusable portable pointwise family. One CPU-owned partition may contain a
single supported pointwise occurrence or one bounded straight-line fused chain. Common CPU
analysis lowers the chain once into the existing route-independent kernel intermediate
representation (IR), normalizes every external value through the completed access plans, selects
the completed portable route and scalar/vector plus single/parallel strategy, applies the
completed optional one-input materialization decision, and realizes one generated artifact and
one partition executable.

This is deliberately a first coverage increment, not "all pointwise everything." It establishes
five exact logical types and their direct heap/`MemorySegment` carriers, one family-oriented CPU
opcode vocabulary, and a useful core semantic matrix:

- same-type binary and exact-scalar `ADD`, `SUB`, and `MUL` for FLOAT64, FLOAT32, INT32, and INT64;
- unary `NEG` for FLOAT64 and FLOAT32, while preserving the existing FLOAT64 exact `GELU`;
- `IS_FINITE`, `IS_NAN`, and `IS_INF` for FLOAT64 and FLOAT32;
- all six current comparisons for same-type FLOAT64, FLOAT32, INT32, and INT64;
- `WHERE` for a BOOL condition and same-type FLOAT64 or FLOAT32 branches; and
- same-type `CAST` for FLOAT64, FLOAT32, INT32, INT64, and BOOL.

The execution boundary is:

```text
supported static pointwise partition
  -> bounded straight-line unit and family-oriented CPU opcodes
  -> existing normalized access plans and optional one-input materialization
  -> existing portable route, strategy selection, specialization, and artifact lifecycle
  -> one partition-level cold-bound executable
```

Every other operation, type combination, topology, conversion, or numerical mode remains
fail-closed and unadvertised.

## Scope

### Exact semantic and data-type matrix

- Accept parameterless `BinaryArithmeticKind.ADD`, `SUB`, and `MUL` only when both inputs and the
  result have the same exact data type, selected from FLOAT64, FLOAT32, INT32, and INT64.
- Accept `ScalarElementwiseKind.ADD`, `SUB`, and `MUL` only with `ScalarValueAttrs` whose exact
  `ScalarValue.dataType()` equals the input and output type, selected from FLOAT64, FLOAT32,
  INT32, and INT64. Retain exact scalar bits in the CPU IR and specialization identity; do not
  turn the scalar into another graph input or read Tensor storage.
- Accept `UnaryElementwiseKind.NEG` for same-type FLOAT64 and FLOAT32 input/output. Preserve the
  completed FLOAT64 exact `GELU` opcode and its existing oracle, special-value, and operation-order
  contract.
- Accept all `FloatingClassificationKind` values—`IS_FINITE`, `IS_NAN`, and `IS_INF`—for FLOAT64
  and FLOAT32 input and exact BOOL output of the same Shape.
- Accept all `BinaryComparisonKind` values—`GREATER_THAN`, `GREATER_OR_EQUAL`, `LESS_THAN`,
  `LESS_OR_EQUAL`, `EQUAL`, and `NOT_EQUAL`—for same-type FLOAT64, FLOAT32, INT32, and INT64
  inputs and exact BOOL broadcast result.
- Accept `WhereSelectionKind.WHERE` only for one BOOL condition, two same-type FLOAT64 or FLOAT32
  branches, and an output of that exact branch type. Derive the branch broadcast first and then
  broadcast the condition against it, matching the current Model construction contract.
- Accept `CastKind.CAST` only when `CastAttrs.targetDataType()` equals both the input and output
  type and that type is FLOAT64, FLOAT32, INT32, INT64, or BOOL. This realizes the represented
  value identity of a same-type explicit cast. Cross-type conversion remains unsupported because
  the current Model contract intentionally does not define its rounding, overflow, saturation,
  NaN, or BOOL conversion result.
- Require exact family-owned attribute classes: `NoOperationAttrs.INSTANCE` for parameterless
  kinds, `ScalarValueAttrs` for the selected scalar arithmetic kinds, and `CastAttrs` for CAST.
  Reject every custom or mismatched kind/attribute pair before lowering or artifact access.
- Require every descriptor Shape to be fully static and every layout resolved. Reuse the completed
  right-aligned `ShapeBroadcast` and `LayoutDescriptor` normalization without adding a CPU shape,
  broadcasting, or layout contract.
- Continue to fail closed for BFLOAT16. Its `short[]` and two-byte segment representation remains
  observable storage plumbing, not executable arithmetic permission. FLOAT16 remains absent and
  blocked on future Model task 0026.

### Exact/default numerical behavior

- FLOAT64 and FLOAT32 ADD, SUB, MUL, and NEG use Java's strict floating evaluation in the exact
  result type and preserved graph order. Do not reassociate, contract, widen an intermediate,
  substitute an approximation, or promise a NaN payload or bitwise result beyond the Model
  contract. Preserve IEEE classifications, infinities, underflow/overflow behavior, and signed
  zero implied by the ordered primitive operation.
- INT32 and INT64 ADD, SUB, and MUL use the current fixed-width two's-complement modular Model
  meaning, exactly matching Java `int` and `long` wraparound.
- Floating comparisons use represented numeric values: ordered relations are false for NaN;
  negative and positive zero compare equal; `EQUAL` is numeric equality; and `NOT_EQUAL` is its
  logical complement. Integral comparisons use signed order. Comparison and classification
  results store canonical BOOL byte `0` or `1`.
- `IS_FINITE`, `IS_NAN`, and `IS_INF` use the current classifications for both signed zeros,
  subnormals, infinities, and every NaN. No NaN payload is inspected beyond classification.
- WHERE treats canonical BOOL byte `1` as true and `0` as false and copies the selected represented
  branch value. Cold binding rejects a non-canonical BOOL condition representation rather than
  inventing truthiness. Selection does not authorize branch-value conversion.
- Same-type CAST preserves the represented value, including signed zero and classification. It
  makes no stronger NaN-payload promise than the source/target type contracts and is not removed
  from the compiled graph by CPU lowering.
- Existing exact GELU remains the only transcendental/activation computation in this task. The
  task adds no relaxed or fast math and no new accuracy policy.

### Family-oriented lowering and emission

- Add one CPU-private `CpuPointwiseOpcode` enum in `internal.ir`. It contains the exact selected
  arithmetic, unary, classification, comparison, selection, cast, and existing exact-GELU opcodes.
  Each opcode declares its family, exact arity, result category, whether it carries an exact scalar
  immediate, and whether the current Vector emitter may realize it. Do not add one planner,
  lowerer, IR instruction class, emitter class, executable class, or registry entry per Model
  operation.
- Generalize `CpuKernelIr.Instruction` to retain one `CpuPointwiseOpcode`, ordered topology-local
  input ordinals, one output ordinal, and an optional nested `CpuKernelIr.ScalarImmediate` holding
  exact `DataType` plus primitive bits copied from `ScalarValue` during lowering. Validation is
  family-oriented and rejects an immediate on parameterless operations or its absence/type
  mismatch on selected scalar arithmetic.
- Keep Model `Operation`, `OperationKind`, `OperationAttrs`, `CompiledNode`, graph identities, and
  strings out of the generated and Runtime hot paths. Lowering maps each admitted Model occurrence
  to the CPU opcode once during analysis.
- Extend the existing `CpuScalarEmitter` and `CpuVectorEmitter` through family-grouped emission
  methods. Shared load/store and loop emission remains in `CpuCarrierEmitter` and
  `CpuLoopEmitter`. The generator consumes the already-typed opcode sequence; it does not contain
  a second Model-operation planner or string dispatch.
- Extend `CpuScalarReferenceKernel` with the same family grouping for differential conformance.
  It remains test/fail-closed reference code and never interprets Model operations or CPU IR in a
  Runtime invocation.

### Partition, fusion, access, and materialization boundaries

- Accept one non-empty CPU partition of at most eight supported pointwise occurrences. It may be
  one occurrence or one straight-line chain in stored partition order: each non-final occurrence
  has one output, that output is consumed exactly once by a later occurrence in the same unit,
  has no publication or cross-partition obligation, and is not consumed outside the unit. Side
  inputs are materialized boundaries. The final result is the unit's sole materialized output.
- Require the internal dataflow to be connected and acyclic by the existing partition order. No
  independent subchains, multi-output node, internal fan-out, multiple materialized outputs, or
  unit sequencing is admitted in this increment.
- Allow a later supported pointwise occurrence to expand an earlier virtual result by ordinary
  right-aligned broadcasting. Derive every node result with the current family-specific Model
  Shape rule, require the declared descriptor to match, and normalize all external boundaries
  against the final iteration Shape. Virtual values remain typed IR values without buffer
  declarations or Runtime slots.
- Fusion is legal only when the complete chain satisfies the exact type/Shape/attribute matrix,
  preserves stored node order and numerical order, has no internal publication/fan-out/
  cross-partition use, and passes the completed output injectivity and boundary alias checks.
  Profitability is bounded here: every legal chain of at most eight instructions is selected as
  one unit; a longer or otherwise incompatible partition fails closed rather than silently
  generating multiple per-operation kernels.
- Reuse `CpuAccessPlan` unchanged as the sole external-value access system, including its five
  regimes, exact spans, injective writes, arbitrary half-open ranges, heap/segment/mixed carriers,
  scalar fallback, and no per-element cursor/division/modulo rules.
- Reuse the completed direct-versus-one-input materialization analysis. Enumerate direct plus at
  most the first three eligible read boundaries in deterministic boundary order, under the
  existing four-complete-candidate ceiling. Direct remains correct and wins ties. A selected copy
  remains eligible only for a FLOAT64 read boundary and uses the completed FLOAT64 contiguous
  workspace/copy path. FLOAT32, integral, and BOOL boundaries remain direct in this increment. No
  second copy, output copy, graph value, or Runtime transfer recipe is introduced.

### Carriers, route, vector, parallel, artifact, and executable reuse

- Generalize the structural carrier vocabulary to these exact heap forms plus exact-segment form:

  | Data type | Heap carrier | `MemorySegment` element representation |
  |---|---|---|
  | FLOAT64 | `double[]` | native-order IEEE binary64, 8-byte element width |
  | FLOAT32 | `float[]` | native-order IEEE binary32, 4-byte element width |
  | INT32 | `int[]` | native-order signed 32-bit two's-complement, 4-byte width |
  | INT64 | `long[]` | native-order signed 64-bit two's-complement, 8-byte width |
  | BOOL | `byte[]` | canonical byte `0` or `1`, 1-byte width |

- Each generated class still has one direct static entry for its actual ordered boundary carrier
  pattern. Boundary count is now derived from the selected chain rather than fixed at four. The
  exact ordered data-type/carrier pattern participates in entry descriptor, specialization,
  artifact metadata, verification, and compatibility. Concrete carriers, extents, offsets,
  strides, assignments, and addresses remain cold facts.
- `CpuPartitionAnalysisInputs.DEFAULT` remains persistence-free, manifest-disabled, scalar,
  single-thread, and direct-only, but its old four-segment carrier list cannot remain a general
  topology contract. Replace the fixed list with a default carrier policy of exact-segment for
  each derived boundary; explicit analysis input may supply an immutable ordered carrier pattern
  whose count must equal the lowered boundary count.
- Reuse the completed portable route, Class-File generation, optional verified persistence,
  process-local compatible interning, post-assignment finalization, strong artifact ownership,
  direct cold binding, and one partition-level executable. No second route or cache is added.
- Scalar and parallel-scalar realize every opcode/type combination in this task. Vector and
  parallel-vector remain eligible only for chains whose values are all FLOAT64 and whose opcodes
  are drawn from ADD, SUB, MUL, NEG, and existing exact GELU, subject to the completed access-run
  rules. BOOL-producing, WHERE, CAST, FLOAT32, and integral chains select scalar compute; this is a
  cold supported fallback, not a failure or a vector-coverage claim.
- Preserve deterministic disjoint parallel chunks, scalar tails for vector code, worker borrowing,
  joined failure/interruption behavior, and zero-work handling. Do not add gather, masked tails,
  another species search, or worker lifecycle ownership.
- Bump the generator/schema compatibility version because opcode structure, entry descriptors,
  carrier vocabulary, and boundary cardinality change. Old entries are safe incompatible misses;
  there is still no migration reader.
- Preserve the completed `KEEP_DISABLED` default persistence verdict. This task does not rerun the
  CPU 0005D timing evidence because it materially changes semantic and carrier fixtures but does
  not propose enabling persistence; a future deliberate persistence-policy evaluation owns a new
  representative evidence matrix.

### Tests and documentation

- Add table-driven capability and lowering tests for every exact supported kind/type/attribute
  row and representative rejected adjacent row, including BFLOAT16, mixed-type operands,
  cross-type CAST, unsupported unary/transcendental/extrema/division/power, bad BOOL, unresolved or
  dynamic layouts, and over-budget or non-linear partitions.
- Add generated-versus-reference differential coverage across scalar/zero/non-zero shapes, all
  five access regimes, offsets/positive and zero strides, arbitrary ranges, heap/segment/mixed
  carriers, single operations, fused chains, every supported data type, and every supported opcode.
- Include boundary vectors for floating signed zero, subnormals, finite extrema, infinities, and
  NaNs; integral min/max values and wraparound; every comparison result; classification truth
  tables; WHERE choices; canonical BOOL output; scalar immediate exact bits; and same-type CAST.
- Preserve exact-GELU oracle tests and the existing vector/parallel/materialization/artifact/
  executable suites. Add representative all-heap, all-segment, and mixed patterns per data type
  and boundary order; do not enumerate an exponential carrier-pattern Cartesian product for every
  eight-node chain.
- Finalize affected CPU Javadocs, package summaries, the CPU backend guide, glossary, and planning
  records in the required separate clean documentation-focused context. Review the Tensor and
  Compile API guides and record a reasoned no-change conclusion: this task adds backend coverage,
  not a public API or Model semantic.

## Out of scope

- Binary or scalar DIV, MIN, MAX, POW, scalar CLAMP, and every mixed-type promoted arithmetic or
  comparison. Exact scalar POW strength reduction remains solely CPU 0005F.
- Unary ABS, RECIPROCAL, LOG, LOG1P, EXP, EXPM1, ERF, SQRT, RSQRT, FLOOR, CEIL, SIGN, RELU,
  SIGMOID, TANH, GELU_TANH_APPROXIMATION, and SILU. Existing exact FLOAT64 GELU remains supported;
  no new transcendental implementation is added.
- BFLOAT16 execution, FLOAT16, mixed precision, cross-type CAST, BOOL arithmetic/logical
  operations, integral WHERE, or a backend-defined conversion/rounding/saturation policy.
- Reductions, scans, statistics, softmax/normalization, layout/view/indexing/scatter/ordering,
  random/dropout, matrix multiplication, convolution, pooling, attention, loss, or any operation
  assigned to CPU 0006–0008.
- Multiple execution units, disconnected subchains, internal fan-out, multiple materialized
  outputs, more than eight instructions, more than one materialization, or cross-partition fusion.
- Relaxed/fast math, numerical-mode configuration, native/vendor routes, OpenBLAS integration,
  route tuning, tuning-cache work, benchmarks, persistence-policy enablement, fixed shapes,
  unrolling, vector gather, masked tails, or additional Vector species.
- Model, Compiler, Planning, shared Prepare, Runtime, Config, Backend Contract, Trace, Engine,
  OpenBLAS provider, another backend, public API, Gradle, dependency, architecture contract,
  focused architecture explanation, ADR, architecture test, backend-conformance module, or
  integration-test changes.
- Creating a detailed CPU 0005F or later task specification, committing, or pushing.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Model,
  Planning, Prepare, Runtime, concrete backend ownership, CPU backend routes, and dependency rules.
- [Current architecture documentation](../../../../architecture/current-architecture-plan.md).
- [Planning Guide](../../../planning-guide.md).
- [CPU backend master plan](../master-plan.md).
- [Completed CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md).
- [Completed CPU 0005B](0005b-universal-access-plans-and-right-aligned-broadcasting.md).
- [Completed CPU 0005C](0005c-vector-and-parallel-portable-strategies.md).
- [Completed CPU 0005D](0005d-materialization-specialization-and-persistence-evidence-gate.md).
- [CPU backend guide](../../../../backend-guide/cpu-backend.md#atomic-partition-kernel-reset).
- [Glossary: CPU kernel intermediate representation](../../../../glossary.md#cpu-kernel-intermediate-representation).
- [Glossary: CPU access plan](../../../../glossary.md#cpu-access-plan).

## Architecture constraints

- `Operation` and the Model operation families remain the sole semantic source. CPU maps supported
  occurrences into private opcodes but does not change or replace Model meaning. The deliberate
  same-type CAST limit avoids inventing the currently undefined cross-type conversion policy.
- Planning continues to select only `BackendId("cpu")` ownership. CPU analysis owns pointwise
  lowering, bounded fusion, access normalization, materialization comparison, strategy and route
  selection, specialization, and exact declarations.
- Shared Prepare sees only the opaque selected CPU plan and exact buffer/workspace requirements.
  It learns no opcode, carrier, fusion, access, numerical, candidate, or artifact policy.
- CPU finalization occurs after shared slot assignment and cannot change the selected unit,
  opcodes, route, strategy, carrier pattern, materialization, or resource set.
- Runtime receives one immutable prepared executable and cold-bound invocation. Its hot path sees
  no `Operation`, `CompiledNode`, graph, CPU IR, route selection, artifact/cache access, storage
  discovery, or semantic dispatch.
- The task changes no public supported CPU API beyond truthful answers from the existing
  `CpuCapabilityProvider`, no module ownership, no dependency direction, and no lifecycle stage.
- If implementation discovers that any selected row requires a Model semantic decision, shared
  contract, another module, dependency/build change, or architecture change, stop and report the
  exact prerequisite instead of broadening CPU-local policy.

The architecture audit conclusion for planning is `None`: the authoritative contract already
assigns concrete lowering, fusion, route choice, generated kernels, storage, finalization, and
execution to the CPU backend. No `ARCHITECTURE.md` change is required for this bounded expansion.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — occurrence-local truthful capability reporting.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — canonical typed values, family-oriented
  opcodes/instructions, access plans, loop, and stores.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — complete-partition validation,
  straight-line unit formation, fusion, boundary discovery, and optional materialization.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct family-grouped scalar and
  bounded vector emission over existing carrier/loop primitives.
- `io.github.pho001.synaptik.backend.cpu.internal.memory` — exact typed heap/segment arguments.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — typed inputs, candidate selection,
  declarations, opaque plan retention, and post-assignment finalization.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — unchanged sole route leaf.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — specialization/schema/artifact
  compatibility and existing budgets.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — one cold-bound partition recipe and
  existing worker orchestration.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — differential scalar conformance.

Packages added or changed:

- No package is added and no responsibility moves between packages.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode` — one new enum because the
  CPU IR needs a stable Model-free typed semantic vocabulary shared across lowering, identity,
  scalar/vector emission, and reference conformance.
- `CpuKernelIr.Instruction` — generalized in place for the opcode, typed immediate, arity, and
  result validation; its nested `ScalarImmediate` owns the CPU-private exact typed bits, and no
  per-operation instruction subtype is added.
- `CpuScalarEmitter` and `CpuVectorEmitter` — extended in place with family-grouped methods; no
  per-operation emitter class, registry, manager, or generic string dispatch is added.

## Affected files

Expected CPU production paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcode.java` (new)
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMaterializationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuBufferArgument.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/memory/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStore.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuCarrierEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneratedKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuLoopEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuScalarEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`

Expected CPU test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPointwiseOpcodeTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuKernelIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPointwisePartitionLoweringTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMaterializationPlanTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseGeneratedKernelTest.java` (new)
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelPersistenceEvidenceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuBufferBindingTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`

Expected explanatory and planning paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- `docs/planning/backends/cpu/tasks/0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No other path is authorized by default.

## Maximum scope

This task may create or modify at most 51 repository paths:

| Category | Maximum | Path accounting |
|---|---:|---|
| CPU production | 31 | Thirty existing production/package paths plus one new opcode type |
| CPU tests | 15 | Twelve existing tests, including the evolved-IR persistence fixture, plus three focused new tests |
| Explanatory documentation | 2 | CPU backend guide and glossary |
| Planning/status | 3 | This task, CPU master plan, and roadmap |
| **Total** | **51** | **31 + 15 + 2 + 3** |

The ceiling is larger than an ordinary task because this cohesive change replaces the fixed
four-boundary/three-node proving assumptions across the already-completed vertical pipeline. It
does not authorize unrelated cleanup. If implementation needs another production abstraction,
test path, package, module, shared contract, or more than 51 paths, stop and revise planning or
propose a bounded follow-up.

## Acceptance criteria

- Capability reporting exactly matches the semantic/type/attribute/Shape/layout matrix in Scope
  and remains occurrence-local; complete-partition lowering applies the stricter topology,
  fusion, virtuality, alias, publication, boundary, and budget checks.
- One through eight supported occurrences lower through one family-oriented opcode sequence, one
  canonical IR, one unit, one artifact, one partition executable, and one bound invocation.
  Internal single-use results remain graph/logical-memory values without declarations or slots.
- Disconnected, fan-out, multi-output, multiple-final-output, over-eight, unsupported, or partially
  supported partitions fail before declarations or artifact access.
- The exact five-type carrier table works for heap, exact segment, and representative mixed
  patterns. Boundary count is derived; default exact-segment policy and explicit pattern count/
  order are validated without a fixed four-entry assumption.
- Scalar and parallel-scalar generated/reference results agree for every admitted opcode/type.
  FLOAT64 eligible numeric-only chains also agree under vector and parallel-vector execution;
  other admitted chains deterministically fall back to scalar compute.
- FLOAT32 operations stay in FLOAT32; INT32/INT64 arithmetic is modular; comparisons,
  classifications, WHERE, canonical BOOL results/conditions, same-type CAST, exact scalar bits,
  and existing exact GELU satisfy the precise rules in Scope.
- Structural identity changes for opcode sequence, types, exact scalar bits, topology, access
  form, ordered carrier pattern, vector compute/species, materialized source, numerical mode, or
  generator schema. It excludes compatible extents, concrete strides/offsets, graph/value/slot
  identities, resources, workers, roots, runs, and cost evidence.
- Existing access-plan regimes, span/write-injectivity/alias checks, arbitrary ranges, zero-work
  behavior, one-copy materialization, four-candidate/one-artifact/zero-shape/zero-unroll budgets,
  worker behavior, optional persistence verification, and prepared-executable ownership remain
  enforced.
- Cross-type CAST, mixed promotion, BFLOAT16, unselected arithmetic/unary semantics, POW,
  reductions/layout/indexing/random work, relaxed math, and native/vendor routes are rejected and
  unadvertised.
- Production contains no per-operation planner/emitter/executable class, Model-operation or graph
  interpretation in Runtime, string dispatch, registry, service locator, reflection, hot carrier/
  route/opcode switch, cursor allocation, or per-element division/modulo.
- No Java, test, build, architecture, dependency, or shared-module path changes outside
  `backends/cpu`; no backend-conformance or integration claim is made before the planned portable
  coverage checkpoint.
- A separate clean documentation-focused agent finalizes affected Javadocs, package summaries,
  CPU guide, glossary, task evidence, master plan, and roadmap in the same overall change. It
  records reasoned no-change conclusions for public API guides, architecture, other modules,
  Gradle, architecture tests, backend-conformance, and integration tests.
- CPU 0005E remains `Ready` until implementation, final CPU tests, documentation review, and all
  validation gates pass. CPU 0005A–0005D remain `Complete`; CPU 0005F and every later CPU task
  remain `Draft` without a detailed specification.

## Tests / validation

Implementation development may use focused classes. Before the single final CPU module run, run
the focused semantic/differential matrix:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcodeTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPointwisePartitionLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuPointwiseGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest
```

After executable Java stabilizes, run exactly one final affected-module suite:

```bash
./gradlew :backends:cpu:test
```

The ordinary suite must continue to skip the opt-in CPU 0005D persistence timing method. Do not
rerun that timing evidence because persistence remains disabled and this task makes no enablement
claim.

Documentation-focused pass, after final Javadocs and Markdown:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

Also validate:

- local Markdown targets and heading anchors in this task, CPU master plan, roadmap, CPU guide,
  and glossary;
- fences, final newlines, trailing whitespace, exact 51-path ceiling, and no unauthorized paths;
- exact supported/rejected semantic and data-type inventory, all five carrier mappings, family
  opcode/type inventory, public/internal type inventory, and absence of per-operation classes;
- CPU 0005A–0005D `Complete`, CPU 0005E `Ready` until implementation completion, CPU 0005F and
  tasks 0006–0017 `Draft`, and absence of later detailed task specifications; and
- unchanged `ARCHITECTURE.md`, focused architecture/ADR files, other modules, Gradle files,
  architecture tests, `testing/backend-conformance`, and `testing/integration-tests`.

Repository-wide Java validation is deferred to CPU 0009, the portable generated-coverage closure
checkpoint, or continuous integration. The current backend-conformance project is only a
placeholder and adding a CPU dependency would violate this task's cross-module/build exclusions;
the CPU-internal generated/reference differential matrix is the proportionate task-level proof.

The documentation pass reuses the implementation pass's successful Java evidence and must not
repeat it unless executable Java changes afterward or it records a concrete stale-evidence risk.

## Dependencies

- Complete CPU 0005A partition-kernel reset and CPU-private unit/IR/route/executable architecture.
- Complete CPU 0005B static right-aligned access plans, layouts, carriers, spans, and broadcasting.
- Complete CPU 0005C scalar/vector and single/parallel strategy implementation.
- Complete CPU 0005D one-copy materialization, specialization budgets, and optional-persistence
  evidence gate.
- Current Model operation, `DataType`, `ScalarValue`, `ShapeBroadcast`, and `LayoutDescriptor`
  contracts; Compiler and Prepare validation already preserve the exact semantic occurrences.
- Current staged Prepare, Runtime cold-binding/execution, and backend-contract ownership contracts.

## Follow-up tasks

- CPU 0005F remains Draft and separately owns exact/default scalar-POW strength reduction. It must
  consume semantic `POW` and cannot broaden the 0005E numerical mode.
- CPU 0006–0008 remain Draft for layout/indexing/random, reduction/scan/normalization, and larger
  computation families.
- CPU 0009 remains the portable generated-coverage closure checkpoint. Remaining pointwise rows
  excluded here—including extrema, division, cross-type promotion/conversion, BFLOAT16, and the
  other unary functions—must receive a later bounded implementation task and Model semantic
  prerequisites where needed before that checkpoint can claim complete selected coverage. This
  task does not silently assign them to 0005F.
- CPU 0010 and later native/tuning/numerical-policy work remains Draft and cannot replace the
  portable fallback or grant relaxed permission.

## Architecture impact

Expected impact: None.

The current architecture explicitly assigns concrete lowering, backend-private IR, fusion,
route/strategy selection, generated artifacts, physical CPU representations, finalization, and
execution to the CPU backend. The task adds no module edge, lifecycle stage, backend identity,
public API, or architecture rule. If implementation requires one, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md.
Read the directly referenced completed CPU tasks and current Model/Compiler/Prepare/Runtime/backend
contracts needed to verify each semantic row.

Implement CPU 0005E exactly as specified. Preserve the completed common partition lowering,
CPU-private IR/access/materialization/route/strategy/artifact/executable architecture and the exact
51-path ceiling. Do not invent cross-type CAST semantics, implement excluded pointwise or later
families, change another module/build/architecture contract, create per-operation classes, commit,
or push. Stop and report any semantic, architecture, dependency, or scope conflict.

After executable Java stabilizes and the focused plus one final CPU suite pass, hand the exact diff
and recorded test evidence to a separate clean documentation-focused agent/thread. That pass must
follow docs/developer-guide/documentation-rules.md; independently finalize affected Javadocs, CPU
guide, glossary, planning evidence, links, scope and status checks; record reasoned no-change
conclusions for public APIs and excluded layers; and not repeat successful Java tests unless
executable behavior changes or a concrete risk requires it.

Update this task's decisions, limitations, validation evidence, implementation notes, completion
summary, and synchronized final status. Do not mark Complete until every acceptance criterion and
the documentation pass succeed.
```

## Local decisions

- The first increment chooses basic ordered primitive arithmetic plus predicates and selection.
  Floating extrema, division, broad unary/transcendental functions, BFLOAT16, promotion, and
  cross-type conversion are deferred because they add distinct exceptional-value, accuracy,
  conversion, or representation decisions.
- Cross-type CAST is not a CPU-local decision: current Model documentation explicitly leaves
  numerical conversion undefined. Same-type CAST is included because represented-value identity
  is sufficient and useful for fused compiler-generated chains.
- One family-oriented opcode enum and the existing scalar/vector emitter owners replace the fixed
  three-semantic proving vocabulary. Per-operation implementation classes are prohibited.
- The bounded topology is one connected straight-line chain with at most eight instructions and
  one materialized result. This removes the fixed three-node/four-boundary proof assumption without
  prematurely implementing arbitrary DAG unit scheduling.
- Scalar/parallel-scalar is the complete fallback for the selected matrix. Vector coverage remains
  intentionally limited to compatible FLOAT64 numeric-only chains and is never inferred from
  carrier availability alone.
- The existing four-candidate/one-artifact/zero-shape/zero-unroll budget remains authoritative.
  Materialization considers at most three eligible boundaries in stable order rather than growing
  candidates with chain width.
- The actual evolved-IR fixture is
  `CpuGeneratedKernelPersistenceEvidenceTest`; it replaces the planned but unchanged
  `CpuClassFileKernelGeneratorTest` allocation in the fifteen-test path map. This preserves the
  exact 51-path ceiling and records the mechanical schema-5 constructor update truthfully.

## Known limitations

- Supported partitions are fully static, resolved-layout, connected straight-line pointwise
  chains of at most eight occurrences with one final materialized output.
- This task does not close the Model pointwise inventory. The excluded exact rows remain
  fail-closed and require later bounded planning before CPU 0009 closure.
- BFLOAT16 storage can still bind as `short[]` or `MemorySegment` at the representation layer, but
  no 0005E operation advertises or executes BFLOAT16. FLOAT16 does not exist yet.
- General cross-type CAST execution remains blocked on a Model-owned numerical conversion
  contract. Same-type CAST does not imply redundant-cast compiler optimization.
- Newly supported FLOAT32, integral, predicate, WHERE, and CAST chains use scalar compute, with
  existing optional parallel orchestration. This task makes no vector or performance claim for
  them.
- Backend conformance and end-to-end integration remain deferred because their current project
  surfaces do not yet consume the CPU implementation without prohibited build/dependency work.
- Optional class-byte persistence remains disabled by default under the CPU 0005D
  `KEEP_DISABLED` verdict; this task records no new timing evidence.

## Validation evidence

The implementation context completed executable validation before this independent documentation
pass. No executable Java or test changed afterward, so the documentation context reused the exact
evidence and did not rerun `:backends:cpu:test`:

- The required focused semantic/differential command from this task passed.
- A broader focused CPU regression batch passed after remediation.
- The sole final `./gradlew :backends:cpu:test` passed. Preserved JUnit XML was independently
  recounted as 24 suites, 73 tests, zero failures, zero errors, and one skipped test. The skip is
  the opt-in CPU 0005D persistence timing measurement; CPU 0005E retains the `KEEP_DISABLED`
  default and did not rerun that evidence benchmark.

Clean documentation context `/root/cpu_0005e_docs` independently reviewed the final diff, source,
tests, completed CPU 0005A–0005D contracts, public Model/Planning/Prepare/Runtime boundaries,
Tensor/Compile/Runtime API documentation, CPU guide, glossary, planning records, and the General,
API/Javadoc, Backend Guide, Planning, Developer Guide, and Example profiles. It changed only
Javadocs/package comments and the five authorized Markdown records. Final documentation commands
and audit results were:

- `./gradlew :backends:cpu:javadoc` passed after final Javadoc edits with only the two expected
  incubating-Vector-module warnings and two warnings for intentionally implicit constructors.
- Local Markdown target/anchor, balanced-fence, final-newline, and trailing-whitespace checks
  passed for the CPU guide, glossary, this task, CPU master plan, and roadmap: 696 local links and
  293 explicit anchors were validated.
- The exact scope audit found 24 changed CPU production/package paths, nine changed CPU test paths,
  two explanatory-documentation paths, and three planning/status paths: 38 total, all authorized
  and below the exact 51-path ceiling. The test allocation truthfully includes the persistence
  evidence fixture and leaves six permitted test paths unchanged.
- The semantic audit confirmed one nineteen-opcode vocabulary; exact carrier types FLOAT64,
  FLOAT32, INT32, INT64, and BOOL; schema 5; same-type CAST only; dynamic derived boundary count;
  one-to-eight connected straight-line chains; virtual intermediates and one final store; scalar
  generated coverage for every admitted row; FLOAT64 numeric-only vector eligibility; all-heap,
  all-segment, and mixed binding; and cold canonical-BOOL validation.
- Package/type and hot-boundary checks found no per-operation implementation classes, later route
  leaf, registry, service locator, Model operation or graph interpretation in Runtime, hot opcode/
  carrier/route dispatch, or new public supported CPU type.
- Status checks confirmed CPU 0005A–0005E `Complete`; CPU 0005F and tasks 0006–0017 remain `Draft`,
  and no later detailed CPU task specification exists.
- `git diff --check` passed after final documentation and status edits.

## Implementation notes

CPU capability and lowering now map the exact admitted occurrence matrix into one nineteen-opcode
CPU-private vocabulary. Lowering accepts one through eight connected straight-line occurrences,
derives external reads deterministically, retains internal single-use values as virtual IR values,
and emits exactly one final materialized store. Boundary cardinality is derived rather than fixed;
default analysis selects one exact segment carrier per boundary, while explicit patterns support
the exact typed primitive-array forms and mixed bindings.

Generated and reference scalar execution cover FLOAT64, FLOAT32, INT32, INT64, and BOOL results
for every admitted opcode row. Vector and parallel-vector remain restricted to FLOAT64 numeric-
only opcode sequences that satisfy the existing access-run rules; all other admitted rows use
scalar or parallel-scalar compute. Cold binding validates canonical BOOL bytes. The existing
access regimes, optional one-FLOAT64-input materialization, strategy/orchestration, artifact,
persistence, direct invocation, and ownership boundaries remain intact; the generator schema is 5.

No public Tensor, Compile, Runtime, Prepare, Planning, Backend Contract, Config, Trace, Engine,
OpenBLAS provider, architecture contract, focused architecture explanation, ADR, module dependency,
Gradle configuration, architecture test, backend-conformance test, or integration test changed.
Tensor and Compile API guides remain accurate because Model semantics and compilation behavior did
not change; the Runtime API remains accurate because CPU 0005E implements the existing cold-binding
and prepared-executable contracts privately. Backend-conformance and integration remain deferred
because their current projects do not consume CPU without out-of-scope build/dependency changes.

## Completion summary

- Completed changes: implemented and documented the bounded five-type portable pointwise matrix,
  family-oriented nineteen-opcode IR, dynamic boundary cardinality, typed heap/segment carriers,
  scalar execution for all admitted rows, and FLOAT64 numeric-only vector eligibility.
- Files changed or created: 24 CPU production/package paths, nine CPU tests, the CPU guide,
  glossary, this task, CPU master plan, and roadmap; 38 total under the 51-path ceiling.
- Tests and validation: reused the successful required focused matrix, broader focused regression,
  and sole final 24-suite/73-test CPU pass with zero failures/errors and one evidence skip; final
  CPU Javadoc, Markdown, scope, semantic, status, and whitespace gates passed.
- Documentation-agent review: `/root/cpu_0005e_docs` independently finalized affected Javadocs,
  package summaries, CPU guide, glossary, and synchronized planning/status evidence.
- Documentation impact: current pointwise coverage, topology, carrier, vector fallback, BOOL,
  materialization, artifact, and excluded-family boundaries are explicit without new authority.
- Javadoc review: affected CPU contracts now describe the derived-boundary, typed opcode/carrier,
  scalar-immediate, BOOL validation, specialization, and cold/hot lifecycle contracts.
- Glossary impact: CPU portable route, generated kernel, specialization, artifact, plan,
  executable, IR, access, and materialization entries now reflect completed CPU 0005E.
- Architecture/build/dependency impact: None.
- Unresolved issues: None.
- Follow-up required: None. CPU 0005F and later work remain Draft.

Status: Complete
