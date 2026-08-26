# Task 0008: Portable Grouped NCHW Conv2d Execution Foundation

## Status

Complete

## Goal

Add truthful portable generated execution for the current first-class grouped NCHW `CONV2D`
operation, with or without its intrinsic rank-one output-channel bias, over fully static resolved
layouts. Establish the smallest family-owned Conv2d-led epilogue and deterministic split boundary
needed before dimensional-convolution closure, without introducing general partition-DAG fusion,
recognition, profitability, or representation planning.

## Motivation and mental model

The previous Draft CPU 0008 row combined Conv2d, MATMUL, two pooling families, attention, three
loss families, and bounded epilogues. Those families have different semantic algorithms,
multi-output and reduction behavior, Shape rules, numerical policies, resources, generated-loop
shapes, and evidence matrices. Implementing them together would violate the planning guide's
cohesive-task rule and make failures difficult to isolate.

This task therefore establishes one complete capability:

```text
one CPU-owned partition containing one grouped NCHW Conv2d occurrence
  -> validate exact static descriptors, layouts, types, geometry, aliases, and publication
  -> lower one family-owned convolution unit
  -> optionally retain one narrowly legal Conv2d-led pointwise epilogue
  -> otherwise form a deterministic family-local materialized split
  -> generate or safely reuse exact compatible class bytes
  -> cold-bind direct typed carriers and optional worker state
  -> execute the prepared recipe without graph or route decisions
```

MATMUL, pooling, attention, and loss execution remain explicitly named Draft follow-ups. CPU
0008A remains the immediate next task because Conv1d visibly composes through Conv2d and Conv3d
needs the rank-two route as its implementation oracle.

## Scope

### Exact admitted Conv2d occurrence

- Add truthful `CpuCapabilityProvider` admission for exactly `Conv2dKind.CONV2D` with
  `Conv2dAttrs` and exactly ordered `[input, weight]` or `[input, weight, bias]` inputs plus one
  output.
- Admit only fully static Shapes and present resolved `LayoutDescriptor` values accepted by the
  existing `PrepareContext` boundary.
- Require input Shape `[N, C_in, H, W]`, weight Shape
  `[C_out, C_in/groups, K_h, K_w]`, and output Shape `[N, C_out, H_out, W_out]` with the exact
  Model/Compiler formula and constraints.
- Require positive static kernel extents, positive stride/dilation/groups, non-negative symmetric
  padding, exact checked channel divisibility, exact
  `weightChannelsPerGroup * groups == C_in`, and exact optional bias Shape `[C_out]`.
- Preserve the exact ordered floating promotion across input, weight, then optional intrinsic
  bias. Admit FLOAT64, FLOAT32, and BFLOAT16 occurrences only where the result descriptor matches
  that promotion.
- Preserve all six current Model data types as the closed repository type contract: Conv2d rejects
  BOOL, INT32, and INT64 because the Model operation is floating-only. `FLOAT16` is not a current
  `DataType`; if that future type appears before its Model semantic task and an explicit CPU route,
  capability and lowering must fail closed rather than infer support from a two-byte carrier.
- Support zero batch, input height/width, output-channel, input-channel, or output-spatial extents
  only where the current Model/Compiler geometry makes the descriptor legal. Empty output domains
  perform no read, write, generated invocation, or worker submission. An empty channel
  contraction begins from positive zero before optional intrinsic bias.

### Semantic and numerical contract

- Implement grouped NCHW cross-correlation, not kernel-reversing mathematical convolution.
- For each output coordinate, traverse the selected contiguous input-channel group, kernel height,
  and kernel width in increasing logical order. Out-of-range padded coordinates are conceptual
  positive zero and participate in ordinary IEEE-754 multiplication, including multiplication by
  infinity.
- FLOAT64 output uses FLOAT64 accumulation. FLOAT32 and BFLOAT16 output use FLOAT32 accumulation;
  BFLOAT16 narrows once at the final semantic output boundary.
- Optional intrinsic bias participates exactly once in the selected accumulation domain before the
  final output conversion.
- Preserve the Model permission for reassociation and fused multiply-add without claiming
  cross-backend bitwise identity. Within each selected generated specialization, however, the
  optimal clean Java oracle and generated code must use the identical semantic algorithm,
  traversal, hot-loop/dataflow shape, rounding boundaries, and avoidable-overhead profile.
- Define and test NaN, infinity, signed-zero, conceptual-padding, empty-contraction, mixed-floating,
  group, stride, dilation, and symmetric-padding behavior directly from the current Model contract.

### Layout, access, and aliasing

- Consume existing resolved layouts and existing typed heap-array, native-order `MemorySegment`,
  and mixed-carrier cold bindings. Do not add another shared representation or storage API.
- Derive checked base offsets, element strides, referenced spans, group geometry, kernel geometry,
  and output-coordinate traversal during analysis/finalization or cold binding, never per output
  element through a graph object or generic layout lookup.
- Support arbitrary legal non-negative resolved strides and offsets through a correct typed scalar
  fallback. Add dense or otherwise specialized forms only when cold proof and retained evidence
  justify them.
- Validate complete carrier compatibility, liveness, byte order/alignment, referenced spans, and
  output/input overlap before the first output mutation or worker submission.
- Reject every output/input overlap for this first capability. Input/input aliasing, including a
  repeated exact carrier, is allowed only when each logical read remains valid and the complete
  spans pass. Output workspace overlap is rejected if a family-local split declares workspace.
- Do not introduce in-place convolution, hidden copying, provider packing, layout reorder,
  representation reuse policy, or automatic alias analysis.

### Publication and resource contract

- The final published Conv2d or epilogue result remains a normal graph output backed by its
  declared buffer. No backend-private value may replace a requested publication.
- A same-unit legal intermediate remains a graph/logical value without a Runtime slot only when it
  is single-use, unpublished, and all semantic and resource conditions below hold.
- Every deterministic split intermediate is declared exactly once as a normal buffer resource with
  checked byte size and alignment before shared slot assignment. It is not an analysis-local
  workspace and must participate in ordinary Runtime validity transitions.
- The direct Conv2d computation requires zero scratch workspace. Any implementation proposal that
  needs im2col, packing, a partial-sum buffer, a combine buffer, or another scratch resource must
  stop and replan; this foundation selects direct grouped traversal.
- No persistent prepared resource, native provider resource, transfer recipe, or new shared
  resource kind is added.

### Bounded Conv2d-led epilogue and safe split

- A partition may contain one Conv2d occurrence followed in direct single-use order by:
  - an optional compatible ordinary `BinaryArithmeticKind.ADD` bias occurrence; and
  - at most one already CPU-supported exact unary activation or `ScalarElementwiseKind.CLAMP`.
- The optional external ADD is bias-like only when one operand is the Conv2d result, the other is
  an external input whose exact resolved right-aligned broadcast produces the Conv2d output Shape,
  and both operands and output have the same FLOAT64 or FLOAT32 type. Intrinsic Conv2d rank-one bias
  remains a separate first-class Conv2d input and may coexist only if the graph explicitly contains
  both operations.
- The optional activation is limited to an existing exact FLOAT64/FLOAT32 pointwise opcode whose
  current capability, numerical algorithm, and carrier topology are already implemented. It must
  consume the immediately preceding result exactly once and produce the same Shape and type.
- Fusion is legal only for a connected straight-line chain with one Conv2d, single-use unpublished
  intermediates, one final publication, compatible layouts/accesses, no alias, state, random,
  multi-output, exceptional validation, or numerical-order conflict, and no new workspace.
- Fused code must preserve every operation boundary's represented result. It must not fold external
  ADD into the convolution accumulator, reassociate activation with convolution or ADD, erase an
  intermediate narrowing, or treat intrinsic and external bias as one semantic role.
- BFLOAT16 Conv2d is supported only as a direct first-class occurrence, including its intrinsic
  bias. External BFLOAT16 pointwise epilogues remain fail-closed because current CPU pointwise
  arithmetic does not support BFLOAT16.
- When a CPU-owned partition contains the admitted Conv2d-led bounded topology but fusion is
  illegal, lower a deterministic family-local split: direct Conv2d materializes its semantic
  output, followed by the existing exact pointwise unit where that unit is itself supported.
- This split is not general partition-DAG decomposition. Reject a partition with a predecessor of
  Conv2d, fan-out inside the proposed chain, more than the bounded nodes, another specialized
  family, horizontal fusion, multiple Conv2d occurrences, or an unsupported pointwise suffix.
  CPU 0008B owns the general solution; capability tests must keep unsupported complete partition
  shapes fail-closed.

### Narrow two-unit materialized-split contract

The family-local split is one explicit exception to the current CPU plan's one-unit invariant. It
is permitted only when all of the following are true:

- the partition has the exact bounded Conv2d-led straight-line topology admitted above;
- the direct Conv2d lead is independently supported;
- the complete remaining non-empty suffix, consisting of the optional ADD, the optional
  activation/clamp, or both in admitted order, is independently realizable as one existing exact
  pointwise unit;
- fusion of the complete chain is illegal under the gates above; and
- exactly one materialized Conv2d-result buffer is sufficient to connect the two units while
  preserving every requested graph publication.

The selected plan is then tagged as the CPU-private Conv2d materialized-suffix form and contains
exactly two units in producer-to-consumer order. Every existing family, direct Conv2d, and legal
fused Conv2d epilogue retains exactly one unit. A two-unit plan with another tag or family, a third
unit, a second materialized edge, fan-out to another computation, a predecessor unit, another
specialized family, or a suffix that is not independently supported fails preparation. This tag
is not a general DAG, fusion-candidate, recognition, profitability, or representation-planning
model and must not be reused by CPU 0008B–0008E as their general contract.

Each of the two `ExecutionUnitPlan` entries owns its exact portable route plan, ordered boundary
`ValueId` values, access bindings, requested and generated carrier patterns, logical range
extents and count, scalar/parallel-scalar strategy, selected range count, minimum work per worker,
zero vector-species bits, unit-local geometry, and diagnostic lowering reason. Existing one-unit
plans may retain their current partition-level mirrors for compatibility, but the split finalizer
and executable must consume the per-unit facts and must not reuse Conv2d ranges, bindings,
carriers, or geometry for the pointwise suffix. Neither split unit may select materialization or
workspace.

Analysis declares all distinct graph-value buffers before returning
`BackendPartitionAnalysis`, including the Conv2d result exactly once in deterministic producer
order before the final suffix result. Its byte size is the checked referenced element span times
the descriptor byte width and its alignment is that byte width, exactly like another CPU graph
buffer. The value is never a `Workspace` requirement. Repeated operand roles reuse the same
declaration and later the same assigned buffer position. Requested Conv2d or suffix-intermediate
publications remain ordinary publications; the two-unit ceiling applies only when the suffix can
still be represented by one exact pointwise unit without dropping any of them.

After shared assignment, `CpuPartitionFinalizer` validates the complete declaration/assignment
set and both units before artifact realization. It resolves each unit's selections by exact
`ValueId`, verifies the borrowed worker group independently against that unit's range count, and
loads or generates exactly two artifacts in unit order. It then returns one CPU-private composite
`PreparedExecutable`, because the unchanged shared `PreparedPartition` and schedule validation
require one executable occurrence per planned partition. The dedicated
`CpuPreparedExecutableSequence` owner is permitted solely for this two-unit form; it is not a
general unit list, scheduler, interpreter, or fusion planner.

The composite declares every external input as `READ_ONLY` and the materialized Conv2d result and
final suffix result as `WRITE_ONLY`, deduplicated by exact buffer selection. Cold binding resolves
and type-checks all selected representations, constructs both child bound invocations, and
validates complete per-unit and cross-unit span, liveness, alignment, writability, and overlap
preconditions before either child is called or any worker work is submitted. The second unit's
read of the intermediate is an internal ordered dependency of the already-bound composite; it is
not exposed as a pre-existing Runtime read, because a newly created intermediate starts invalid.

Runtime therefore applies its ordinary atomic executable transition: it validates all external
reads and invalidates every copy of both declared outputs before the composite call, then marks
the exact intermediate and final written copies valid only after both child invocations return
successfully. The composite performs no direct `RunState` validity mutation. If either unit
throws, Runtime marks neither output valid, publishes nothing, and performs its existing failure
cleanup; bytes written by the first unit remain logically invalid. This preserves failure-before-
mutation for every validation that can be completed cold and failure-before-publication for an
execution failure without adding a backend-owned coherence rule.

Execution is strictly `Conv2d artifact -> completion/join -> pointwise artifact`. Each unit owns
disjoint complete ranges under its own prepared strategy. A shared borrowed `CpuWorkerGroup` may
serve both units, but their submissions never overlap; the second starts only after the first has
completed successfully. Empty ranges submit no work and do not call their generated entry.

The final publication suffix remains Runtime-owned and unchanged. It may publish any requested
declared result only after the single composite occurrence succeeds; the CPU composite neither
publishes nor transfers ownership. Each artifact retains its own existing specialization and
generated-artifact cache identity. The Conv2d and pointwise keys exclude plan tags, `ValueId`
values, slots, representations, addresses, and run state. There is no cached or persisted
composite artifact and no family-wide two-unit cache key; the lowering manifest may record the
ordered split and the two artifact keys for diagnostics only.

Generated-code evidence must inspect both realized classes and prove that each contains its own
direct typed loop. Split semantic/performance evidence compares the ordered generated pair with
two optimal clean Java loops over the same ordinary intermediate carrier and identical operation
boundaries. Existing pointwise evidence may support the unchanged suffix emitter, but this task
must still retain the exact split suffix class inspection, ordered-invocation integration check,
intermediate validity/publication checks, and complete generated-pair performance result.

### Portable route, generated code, and execution

- Add one focused CPU-private Conv2d IR/geometry owner, one lowering owner, one direct Class-File
  emitter, and one independent direct Java reference/oracle owner. Reuse existing carrier,
  specialization, artifact, worker, prepared-executable, and staged-finalization contracts.
- The family IR identity includes kind, intrinsic-bias presence, ordered boundary data types,
  accumulation/output type, grouping and kernel/stride/padding/dilation semantics, admitted
  epilogue topology, access/carrier code-shaping facts, and selected execution strategy. Runtime
  identities, `ValueId` values, exact carrier objects, addresses, slots, and run state stay out.
- Advance `CpuGeneratorSchema` exactly once. Older schema envelopes are incompatible safe misses;
  no migration reader is added.
- Generate direct typed scalar bodies for `double[]`, `float[]`, raw-BFLOAT16 `short[]`,
  `MemorySegment`, and the admitted mixed patterns. Generated entries are field-free and
  constructor-free and expose one typed static entry matching the selected boundary pattern.
- Use scalar and bounded parallel-scalar execution over deterministic disjoint complete output-cell
  ranges. The generated method retains primitive range inputs. Each output cell is owned by one
  range; no partial/combine accumulation or synchronization is permitted.
- Vector API convolution, im2col, Winograd, FFT, native BLAS/DNN providers, packing, autotuning,
  dynamic route choice, and shape-baked specialization are excluded.
- The optimal clean Java oracle must be independently readable and must not delegate its hot loop
  to generated code or to the same emitter helper. Generated hot loops must not be bridge-only and
  must not call a Synaptik-owned per-output or per-contribution helper.
- Runtime execution uses already prepared direct fields. No operation/graph inspection, route
  decision, cache lookup, reflection, string dispatch, boxing, collection/map lookup, allocation,
  type discovery, layout construction, or repeated compatibility cast is allowed in generated or
  bound hot work.
- If specialization cannot be proved, choose the direct typed general scalar generated fallback.
  Never fall back to a Runtime interpreter or advertise a form that preparation cannot realize.

### Documentation and Javadoc

- Add or update meaningful Javadoc for every affected production type and method, including
  semantic roles, geometry, numerical boundaries, ownership, resources, concurrency, nullability,
  return semantics, and failure conditions.
- Update `docs/backend-guide/cpu-backend.md` with the current admitted Conv2d subset, direct
  traversal, epilogue/split boundary, carrier/layout/alias/resource behavior, generated-code
  evidence, and limitations.
- Update `docs/glossary.md` only if implementation introduces a reusable new term. Otherwise record
  a reasoned no-change conclusion.
- The separate clean documentation-focused pass must review the final diff, Model/Compiler Conv2d
  contracts, affected Javadocs/package summaries, CPU guide, glossary, task, master plan, and
  roadmap. It reuses successful Java/performance evidence unless it changes executable behavior or
  records a concrete stale-evidence risk.

## Out of scope

- MATMUL and linear execution; max/average pooling; scaled dot-product attention; mean-squared,
  dense categorical, or index categorical loss execution. Draft CPU 0008F–0008I own them.
- Conv1d end-to-end composition validation or direct Conv3d execution; CPU 0008A owns them.
- General partition-DAG decomposition, horizontal/vertical fusion, typed specialized-subgraph
  recognition, general legality/profitability facts, or bounded representation/materialization
  variants; CPU 0008B–0008E own them. The exact tagged two-unit Conv2d materialized-suffix form is
  the sole exception and must not become reusable general DAG machinery.
- A public or backend-generic `ConvNd`, another Model/Compiler operation, inference or gradient
  change, Compiler 0006C, or any public API expansion.
- Dynamic or unresolved Shapes/layouts, asymmetric implicit padding, transposed/deformable/causal
  specialized convolution, depthwise-special naming, sparse/quantized convolution, dynamic rank,
  or arbitrary-rank convolution.
- FLOAT16, mixed-precision accumulation policy beyond current ordered promotion, BFLOAT16 external
  pointwise epilogues, relaxed math, or a new determinism/numerical mode.
- Vector/native convolution, OpenBLAS, Accelerate, oneMKL, oneDNN, AOCL, ZenDNN, im2col, Winograd,
  FFT, packing/reorder, tuning, tuning-cache use, measurement-driven preparation, tracing payloads,
  or benchmark-setting mutation.
- New shared Prepare, Runtime, Planning, Backend Contract, Config, Trace, Engine, architecture,
  dependency, Gradle, toolchain, or module-boundary contracts.
- General backend conformance or end-to-end Engine integration infrastructure. CPU 0009 owns the
  portable conformance checkpoint; Engine 0004 owns the typed end-to-end convolution lifecycle.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially core invariants, Planning,
  Prepare, Runtime, concrete backend, generated CPU kernel, performance-evidence, and testing rules
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Performance evidence and model autotuning](../../../../architecture/performance-evidence-and-tuning.md)
- [ADR 0002: Backend-owned lowering](../../../../design/decisions/0002-backend-owned-lowering.md)
- [ADR 0008: Performance evidence and tuning boundaries](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)
- [ADR 0010: Staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [CPU kernel strategy](../../../../design/notes/cpu-kernel-strategy.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [Kernel routes](../../../../backend-guide/kernel-routes.md)
- [Planning guide](../../../planning-guide.md)

## Architecture constraints

- Planning continues to select only CPU ownership. CPU analysis owns Conv2d lowering, bounded
  family-local epilogue legality/split, route choice, and exact resource declarations.
- CPU analysis remains deterministic from `PrepareContext` and explicit CPU inputs. Finalization
  generates or reuses the already selected artifact only after shared slot assignment and adds no
  undeclared resource.
- Shared Prepare still receives one analysis and one finalized executable for the planned
  partition. The tagged split is internal to that CPU executable; it requires no change to
  `PreparedPartition`, `PreparedSchedule`, `PreparedExecutionRunner`, or any shared validity API.
- Runtime owns the composite occurrence's normal access transition. The CPU sequence neither
  changes `RunState` validity nor publishes an intermediate from inside backend work.
- Runtime receives only immutable prepared recipes and direct cold-bound invocations. Its hot path
  sees no `Operation`, `CompiledNode`, route, layout, or graph fact.
- Providers do not reinterpret graphs. This task adds no provider route.
- Generated code follows an optimal clean Java implementation of the identical specialized case as
  its design, review, and performance oracle.
- Benchmark/performance evidence is observational and cannot mutate production settings. No search
  or cache mutation occurs in Runtime.
- Any required public/shared/module/dependency/architecture change is a stop condition, not an
  implementation detail.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — truthful public capability admission only.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — immutable family code-shaping identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — family geometry, bounded epilogue,
  and split construction.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct Class-File body emission.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent direct Java oracle.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — already selected route facts.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — exact declarations/finalization state.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — cold binding and direct execution.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema and compatibility identity.

Packages added or changed:

- No package is added. Only the existing CPU-private responsibility packages above may change.

Expected new type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv2dIr` — immutable Conv2d and admitted
  epilogue code-shaping identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLowering` — exact static
  geometry, boundary, fusion, and family-local split owner.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuConv2dEmitter` — direct generated
  scalar computation body.
- `io.github.pho001.synaptik.backend.cpu.internal.reference.CpuConv2dReferenceKernel` — independent
  clean Java semantic/performance oracle.
- `io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableSequence` — the
  sole two-unit Conv2d materialized-suffix composite, with complete cold validation and ordered
  direct child invocation; it is not a general scheduler.

Tests mirror their production packages. A test-only performance harness remains under
`internal.codegen.emit`; it is not production tuning infrastructure.

## Affected files

Expected production and Javadoc paths:

- `CpuCapabilityProvider.java` and CPU package Javadoc;
- `internal/ir/CpuPortableKernelIr.java`, new `CpuConv2dIr.java`, and IR package Javadoc;
- `internal/lowering/CpuPartitionLowering.java`, new `CpuConv2dLowering.java`, and lowering package
  Javadoc;
- `internal/codegen/emit/CpuClassFileKernelGenerator.java`, new `CpuConv2dEmitter.java`, and emitter
  package Javadoc;
- `internal/reference/CpuConv2dReferenceKernel.java` and reference package Javadoc;
- `internal/cache/CpuGeneratorSchema.java` and compatibility owners only where Conv2d identity
  requires them;
- route, prepare, finalizer, and executable owners plus package Javadocs only where the new IR,
  family-local split, declarations, or direct bound fields require integration; and
- new `internal/executable/CpuPreparedExecutableSequence.java` only for the exact tagged two-unit
  split and its atomic Runtime-facing access declaration.

Expected tests:

- focused capability, IR, lowering, preparation/finalization, executable/binding, reference,
  generated-kernel, schema/cache-compatibility, package-inventory, evidence, and performance tests;
- explicit negative matrices for non-floating types, the exact six-value `DataType` inventory and
  absence of FLOAT16 admission, unresolved Shapes or layouts, descriptor mismatch, illegal
  geometry, unsupported partitions, illegal epilogues,
  overlap, span/alignment/carrier mismatch, and mutation-before-validation;
- unchanged-family controls chosen from current pointwise and batch-normalization generated routes.

Expected documentation/planning paths:

- `docs/backend-guide/cpu-backend.md`;
- `docs/glossary.md` only if terminology changes;
- this task;
- `docs/planning/backends/cpu/master-plan.md`; and
- `docs/planning/roadmap.md`.

## Maximum scope

This task may create or modify at most:

- 19 production/Javadoc paths, including exactly the four planned family production types and at
  most the one additional `CpuPreparedExecutableSequence` private owner named above;
- 14 CPU test/resource paths;
- the five named documentation/planning paths, with the glossary optional; and
- 38 total paths.

No other new production owner and no shared module, architecture, Gradle, backend-conformance,
integration, benchmark-tool, later task-specification, or vendor-provider path is permitted. If
the bounded family-local split or epilogue cannot fit this ceiling, or if another production
owner/new type is necessary, stop and replan rather than folding general 0008B–0008E work into
this task.

## Acceptance criteria

- CPU capability returns true exactly for the admitted static resolved-layout grouped NCHW Conv2d
  matrix and false for every excluded kind/type/signature/descriptor/geometry case.
- Direct unbiased and intrinsic-biased Conv2d execute correctly for groups 1, intermediate groups,
  and depthwise geometry; unit/non-unit stride and dilation; zero/non-zero symmetric padding;
  zero/empty legal domains; FLOAT64, FLOAT32, BFLOAT16, and ordered mixed-floating inputs; heap,
  segment, and selected mixed carriers; contiguous and arbitrary legal resolved layouts.
- Results match the current Model/Compiler Shape, grouping, promotion, conceptual-padding,
  accumulation, bias, special-value, and final-conversion contracts.
- The bounded FLOAT64/FLOAT32 external ADD plus at-most-one activation/clamp topology is fused only
  when every stated legality gate holds and preserves each semantic operation boundary.
- Every other admitted bounded topology uses a deterministic family-local materialized split with
  exact declarations, or fails capability/preparation closed if the suffix is not independently
  supported. No requested result or split intermediate is silently dropped.
- The only multi-unit plan is the exact tagged Conv2d materialized-suffix form with exactly two
  complete per-unit fact sets. Every old family, direct Conv2d, and legal fused Conv2d plan still
  rejects `units().size() != 1`; malformed tags, unit counts, boundaries, or roles fail before
  finalization.
- Analysis declares the Conv2d intermediate exactly once as an ordinary buffer before shared
  assignment, never as workspace. Finalization resolves exact assigned selections and realizes
  both ordered cache-compatible artifacts without adding a requirement.
- Cold binding validates both units and every cross-unit alias/access condition before the first
  write or worker submission. One composite execution runs Conv2d to completion before the
  pointwise suffix; Runtime invalidates and validates the intermediate and final output through
  the composite's ordinary declared accesses and publishes only after complete success.
- Scalar and parallel-scalar strategies, ranges, and worker-capacity checks are unit-local. The
  two units never overlap worker submissions, and an empty unit submits no work.
- BFLOAT16 direct Conv2d works with intrinsic bias where otherwise legal; BFLOAT16 external
  pointwise epilogues, all non-floating types, and future FLOAT16 fail closed.
- All validation and overlap checks complete before output mutation or worker submission. Failure
  exposes no partially written published result.
- Direct Conv2d uses zero workspace and scalar or parallel-scalar complete-output-cell ranges.
  Parallel and scalar results follow the same per-cell algorithm and do not share partial state.
- Generated classes are deterministic, final, field-free, constructor-free, and direct. Retained
  `javap -c -v`/Class-File inspection proves the expected typed entry signatures and absence of
  method handles, `invokedynamic`, dynamic constants, bootstrap methods, reflection, boxing,
  collection/map dispatch, allocation, route lookup, graph/operation references, and Synaptik-owned
  per-output/per-contribution helper calls.
- Split evidence inspects both exact generated classes, proves ordered direct invocation with no
  bridge/interpreter body, and compares the generated pair against two optimal clean Java loops
  using the same ordinary intermediate representation.
- Every performance target compares generated execution with an optimal clean Java implementation
  of the identical specialized case. Each target and unchanged-family control passes five accepted
  isolated fixed-heap forks and the aggregate median at generated/direct `<= 1.15x`.
- Evidence includes at minimum representative dense/grouped/depthwise, padded/dilated/strided,
  general-layout/mixed-carrier, BFLOAT16, fused-epilogue, and parallel forms. One whole fork may be
  rejected only before measurement for a recorded environment/classpath/control failure; no ratio
  sample is discarded.
- Schema advances exactly once and older artifacts are safe misses. Deterministic regeneration
  produces identical class bytes for the same specialization.
- Current pointwise, affine/movement, indexing/scatter/fold/order/random, scan/reduction,
  softmax/normalization, batch-normalization, materialization, persistence, worker, preparation,
  and execution tests remain green.
- No Model, Compiler, Planning, Prepare, Runtime, Engine, Backend Contract, Config, Trace, provider,
  architecture, dependency, build, conformance, integration, or public API contract changes.
- Production and test types match the package map and maximum-scope ceiling.
- A separate clean documentation-focused agent pass finalizes affected Javadocs, CPU guide,
  glossary impact, and planning evidence in the same overall change.
- CPU 0008 becomes `Complete` only after all gates pass. CPU 0008A is then the sole Ready CPU task;
  CPU 0008B–0008I and 0009 onward remain Draft and have no premature detailed task files.

## Tests / validation

During implementation, run focused tests for the affected owners, for example:

```bash
./gradlew :backends:cpu:test \
  --tests '*CpuConv2d*' \
  --tests '*CpuCapabilityProviderTest' \
  --tests '*CpuPartitionPreparerTest' \
  --tests '*CpuPartitionFinalizerTest' \
  --tests '*CpuPreparedExecutableTest' \
  --tests '*CpuPreparedExecutableSequenceTest' \
  --tests '*CpuGeneratedKernelShapeTest' \
  --tests '*CpuInternalPackageInventoryTest'
```

After executable code stabilizes, run exactly one authoritative uncached CPU suite:

```bash
./gradlew :backends:cpu:test --rerun-tasks
```

Retain machine-readable test counts and the deterministic Class-File/evidence bundle. Run the
task-owned evidence test or harness in five fresh isolated Java 26 fixed-heap forks using its exact
documented command. Each fork must use deterministic inputs, randomized generated/direct order,
at least five warmup batches, nine measured rounds, adaptive batches of at least 25 ms, semantic
checks before timing, and explicit environment/control acceptance. Record per-case ratios,
aggregate medians, accepted/rejected forks, generated-class checksums, complete `javap -c -v`
member reports, and the evidence manifest digest.

The final documentation-focused pass receives the stabilized diff, final CPU XML, retained
performance/Class-File bundle, and exact implementation commands. It does not rerun successful
Java/performance work unless executable behavior changed or a concrete stale-evidence risk is
recorded. After final Javadocs it runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

It also validates local Markdown links and anchors, headings, balanced fences, final newlines,
terminology, generated Javadoc pages, exact path/type ceilings, schema/status/order, absence of
later task specifications, and empty staging.

Repository-wide and architecture suites remain deferred to CPU 0009 or continuous integration
because this task changes no dependency or shared boundary. Backend-conformance tests are not
added in this task because the current `testing/backend-conformance` module has only its foundation
marker and this task promises one CPU-private admitted subset, not a cross-backend contract; CPU
0009 is the named conformance checkpoint. Integration tests are not required because Engine does
not yet expose the public lifecycle and this task makes no end-to-end promise; Engine 0004 owns the
three-rank convolution lifecycle. Any implementation change to those facts is a stop condition and
requires replanning plus proportionate validation.

## Dependencies

- Complete CPU 0005A–0007F2, including current whole-partition lowering, access plans, generated
  artifact/schema, materialization, worker/resource, direct generated-family, corrective parity,
  preparation/finalization, and bound-execution contracts.
- Complete Model 0020 owns grouped NCHW Conv2d semantics, attributes, public expressions, promotion,
  Shape, numerical meaning, and provenance.
- Complete Model 0025G owns visible Conv1d composition but is not implemented by this task; its
  existence fixes CPU 0008A's immediate validation dependency.
- Complete Model 0025H and Compiler 0006B fix the later Conv3d forward contract and ordering but do
  not expand this task to rank three.
- Complete Compiler 0001–0006B own capture, inference/validation, gradients for Conv2d, publication,
  planning artifacts, and Conv3d forward adoption. Draft Compiler 0006C is not a prerequisite for
  forward CPU execution.
- Current shared Planning, Prepare, Runtime, Backend Contract, Config, and Trace contracts are
  sufficient and unchanged.

## Follow-up tasks

- CPU 0008A: validate NCW Conv1d through visible composition and add direct grouped NCDHW Conv3d.
- CPU 0008B: general partition-DAG computation-unit decomposition and bounded fusion.
- CPU 0008C: typed specialized-subgraph and epilogue recognition.
- CPU 0008D: bounded fusion profitability and typed cold decision facts.
- CPU 0008E: bounded multi-input materialization and representation reuse.
- CPU 0008F: portable MATMUL execution and bounded linear epilogues.
- CPU 0008G: portable max/average Pool2d execution.
- CPU 0008H: portable scaled-dot-product attention execution.
- CPU 0008I: portable loss-family execution.
- CPU 0009: portable generated-coverage and backend-conformance closure checkpoint.
- Engine 0004: typed end-to-end Conv1d/Conv2d/Conv3d lifecycle validation.
- Compiler 0006C: separately prove and close Conv3d gradients; it does not block forward execution.

## Architecture impact

Expected impact: None.

This task consumes the existing concrete-backend analysis, lowering, exact-resource declaration,
post-assignment finalization, generated-kernel, cold-binding, worker, and prepared-execution
boundaries. It changes no owner, dependency direction, shared contract, or public API. If the
implementation needs a shared contract, public surface, or architecture change, stop and report
the exact conflict before editing that boundary.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the isolated implementation agent for Synaptik CPU task 0008.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0008-portable-grouped-nchw-conv2d-execution-foundation.md.
Read the task's directly referenced Model, Compiler, CPU source/test, architecture, generated-code,
performance-evidence, and documentation contracts. Implement exactly the Ready specification and
do not implement later CPU tasks. Stop and report any architecture, shared-contract, or maximum-
scope conflict rather than inventing a boundary.

After implementation and recorded CPU/Class-File/performance validation, hand the stabilized diff
and evidence to a separate clean documentation-focused agent. That pass must follow
docs/developer-guide/documentation-rules.md, finalize Javadocs, CPU guide, glossary impact, and
planning evidence in the same overall change, and must not repeat successful Java/performance
work unless executable behavior changes or it records a concrete reason.

Do not commit or push unless the coordinating user explicitly authorizes it. Update this task's
local decisions, known limitations, validation evidence, implementation notes, completion summary,
and final status only from actual results.
```

## Local decisions

- The bounded illegal-but-independently-supported Conv2d suffix stays in CPU 0008 because the
  materialized split is part of this task's accepted Conv2d capability and its implementation is
  already in progress. Inserting or reordering another task would separate a required fallback
  from the capability it makes truthful.
- Shared Prepare and Runtime remain unchanged. One CPU-private composite executable represents
  the two ordered artifacts to the existing one-executable-per-partition handoff, while its outer
  access declaration lets Runtime perform the ordinary atomic validity transition.
- `CpuPartitionPreparationPlan` preserves the one-unit invariant by default and admits exactly
  two units only under the closed Conv2d materialized-suffix tag. The exception is deliberately
  unsuitable as CPU 0008B's future general DAG model.
- One additional private executable-sequence owner is preferable to adding general scheduling to
  `CpuPreparedExecutable` or changing shared schedule contracts. No other new production owner is
  permitted by this task.

## Known limitations

- The first route is direct scalar/parallel-scalar grouped NCHW traversal; vector/native,
  packing/im2col/Winograd/FFT, tuning, and shape-baked specialization remain deferred.
- BFLOAT16 supports direct Conv2d and intrinsic bias only; external BFLOAT16 pointwise epilogues
  remain unsupported.
- Family-local epilogue/split handling does not solve general predecessor, fan-out, multi-Conv2d,
  horizontal, or mixed-specialized-family partitions; CPU 0008B owns that closure.
- Backend-conformance and public end-to-end Engine validation remain at CPU 0009 and Engine 0004.

## Validation evidence

- The one authoritative uncached CPU validation completed successfully on 2026-08-26 with
  `./gradlew :backends:cpu:test --rerun-tasks`: 485 tests, 3 skipped, 0 failures, and 0 errors;
  all 22 requested Gradle tasks executed.
- Focused implementation runs covered capability, lowering, generated bodies, deterministic
  evidence, preparation/finalization, composite binding/execution, package inventory, and retained
  existing-family preparation behavior before the authoritative run.
- Five fresh isolated OpenJDK 26.0.1 fixed-heap forks used `--enable-preview`,
  `--add-modules jdk.incubator.vector`, `-Xms1g`, and `-Xmx1g`. Each fork passed all eight
  generated/direct targets and controls with five randomized warmup rounds, nine measured rounds,
  and adaptive batches of at least 25 ms. Every per-fork ratio was at most 1.15. Aggregate medians
  were: dense FLOAT32 0.994777342, grouped FLOAT64 1.057987982, depthwise BFLOAT16 0.942743986,
  general-layout mixed carriers 0.966266916, fused ADD+RELU 0.994248242, parallel FLOAT32
  0.979604459, materialized split pair 0.993886270, and unchanged FLOAT32 ADD control 0.994390269.
  Accepted forks: 5; rejected forks: 0.
- Retained evidence is under
  `/private/tmp/synaptik-cpu-0008-retained-evidence-20260826`. It contains the five raw fork CSVs,
  aggregate summary, eight deterministic generated classes, compatibility bytes, specializations,
  descriptors, lowering manifests, member reports, SHA-256 checksums, and complete retained
  `javap -c -v` reports (2,629 lines). The manifest-file SHA-256 is
  `c501650e90412ce5f7664fbfa1f6fd4c42f695ed171f05ffcd838ee871d9ba8b`.
- Class-File inspection found one direct static `invoke` method per final field-free class and no
  constructor, allocation opcode, invokedynamic, dynamic constant, bootstrap method, reflection,
  boxing, collection/map dispatch, or Synaptik-owned hot helper call. The only calls in retained
  bodies are exact primitive `Math.max`, BFLOAT16 bit conversion, and typed `MemorySegment`
  access where selected.
- The mandatory separate documentation-focused review finalized every affected Conv2d and
  composite-executable Javadoc, the directly relevant CPU package summaries, the CPU backend
  guide, this task, the CPU master plan, the roadmap, and the glossary's existing Conv2d/current-
  schema entries. It found no architecture-contract, shared API, Gradle, conformance, integration,
  or additional terminology owner requiring change.
- Final CPU Javadoc generation, local Markdown link/anchor/fence validation, terminology/status
  checks, retained-evidence digest verification, generated-document inspection, exact scope and
  staging checks, and `git diff --check` passed. The successful CPU Java and performance suites
  were not repeated because the documentation pass changed no executable behavior.
- Documentation context `01a03e0b-d36d-7063-9adc-6f614b8ccc87` ran
  `./gradlew :backends:cpu:javadoc`: all 11 tasks completed successfully (2 executed, 9
  up-to-date), with only the two expected incubating Vector API warnings and no Javadoc content
  warning. The targeted local checker reported `Markdown validation passed for 5 files`; all
  manifest entries reported `OK`; final scope was exactly 37 paths; staged path count was zero;
  and tracked plus untracked whitespace checks produced no diagnostic.
- Glossary revision was necessary because its existing Conv2d entry still described concrete CPU
  execution as planned and its generated-artifact entries stopped at schema 50. The review
  corrected those existing definitions without introducing a new reusable term or glossary owner.

## Implementation notes

- Added the four planned family owners (`CpuConv2dIr`, `CpuConv2dLowering`, `CpuConv2dEmitter`, and
  `CpuConv2dReferenceKernel`) plus the sole permitted `CpuPreparedExecutableSequence` owner.
- Direct generated grouped NCHW traversal supports the exact admitted FLOAT64, FLOAT32, BFLOAT16,
  ordered-promotion, intrinsic-bias, static-layout, grouping, padding, stride, dilation, empty,
  carrier, and scalar/parallel-scalar matrix. FLOAT32/BFLOAT16 hot work remains in primitive
  FLOAT32 locals, and selected segment accesses use frozen typed layouts.
- Legal external FLOAT64/FLOAT32 ADD and ADD+RELU chains retain represented operation boundaries
  in one direct Conv2d artifact. Other admitted activation/clamp forms, and requested intermediate
  publications that make fusion illegal, use the exact tagged two-unit materialized suffix when
  the complete pointwise suffix is independently supported; unsupported suffixes fail closed.
- Split analysis declares one ordinary intermediate before shared assignment, retains complete
  unit-local route/boundary/access/carrier/range/worker/geometry facts, and finalizes exactly two
  ordered cache-compatible artifacts. The composite validates child and cross-unit spans and
  aliases before either child can execute, declares external reads plus intermediate/final writes
  once, and performs no direct `RunState` validity or publication mutation.
- Generator schema advanced exactly once to version 51. Existing artifacts are safe misses and
  deterministic regeneration produces identical bytes.
- Final scope is 19 production/Javadoc paths, including exactly five new production types; 13 CPU
  test paths; 5 documentation/planning paths; and 37 total paths. No shared module, architecture, Gradle,
  conformance, integration, later-task specification, or public API path changed.

## Completion summary

CPU 0008 is complete. It delivers direct generated grouped NCHW Conv2d for FLOAT64, FLOAT32, and
BFLOAT16 with intrinsic optional bias, groups/depthwise geometry, explicit padding/stride/dilation,
typed arrays/segments/mixed carriers, scalar or parallel-scalar complete-output-cell ranges, legal
ADD and ADD-plus-RELU epilogues, and the sole tagged two-unit materialized suffix. The composite
uses one ordinary intermediate buffer and presents one atomic `PreparedExecutable` access boundary
to Runtime while retaining strict child order and no direct validity mutation. Generator schema is
51. Implementation, retained structural/performance evidence, Javadoc, backend-guide, glossary,
planning, and final documentation validation gates passed without widening shared contracts.

CPU 0008A is now the sole Ready CPU task. CPU 0008B through CPU 0008E remain Draft in their
established order.

Status: Complete
