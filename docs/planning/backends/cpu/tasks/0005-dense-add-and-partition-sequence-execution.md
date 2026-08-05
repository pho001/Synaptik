# Task 0005: Dense Add and Partition-Sequence Execution

## Status

Ready

## Goal

Deliver the first truthful executable Model-operation coverage through the completed CPU
representation, generator, artifact, Prepare, Runtime, and worker foundations:

```text
maximal consecutive CPU-owned partition
  -> validate every occurrence against the exact CPU-0005 ADD matrix
  -> derive one ordered generated kernel recipe per graph node
  -> share exact graph-value buffer requirements across those recipes
  -> shared Prepare assigns logical buffers once
  -> finalize/load one exact generated artifact per node recipe
  -> cold-bind direct typed per-node invocations
  -> one partition BoundInvocation executes them in node order
```

The supported semantic matrix is deliberately narrow: parameterless
`BinaryArithmeticKind.ADD`, exact equal input/output shapes, fully static shapes, exact equal
input/output data type, and `FLOAT64`, `FLOAT32`, `INT32`, or `INT64`. The first route uses
canonical native `MemorySegment` buffers and scalar single-thread generated code. This task also
extends the CPU-0004 single-kernel recipe into an ordered partition recipe so truthful per-node
capability reporting remains executable when Planning groups any number of consecutive supported
occurrences into one maximal same-owner partition.

The task does not claim the rest of the elementwise family. CPU 0005A and later Draft work extend
the exact matrix, carriers, and modes after this vertical slice is complete.

## Scope

- Change `CpuCapabilityProvider` from globally fail-closed to returning `true` only for the exact
  CPU-0005 ADD occurrence matrix and `false` for every other valid query.
- Accept one ADD occurrence only when it has two inputs and one output, all three descriptors have
  the same supported data type and exact equal fully static shape, and every resolved layout is
  canonical contiguous with zero storage offset. An unresolved layout is valid because the CPU
  route materializes the logical value into its selected canonical representation. A resolved
  view, offset, broadcast, permuted, or otherwise non-canonical layout is rejected.
- Preserve null validation and use structural operation kind/attribute contracts; only
  `NoOperationAttrs.INSTANCE` with `BinaryArithmeticKind.ADD` is admitted.
- Add a package-private immutable pointwise ADD lowering recipe for one graph-node occurrence.
  It retains exact ordered input/output `ValueId` references, exact data type, exact static element
  count, and a deterministic lowering fingerprint containing every bytecode-relevant semantic
  fact but no graph-local identity.
- Add a package-private family candidate source that validates every node in the supplied
  `PrepareContext` against the same public capability predicate and builds one ordered kernel
  recipe per partition node. It must never accept a partially supported partition.
- Extend the CPU-0004 candidate/plan/finalization path from exactly one kernel to one non-empty
  ordered partition candidate containing one or more complete kernel candidates. Preserve the
  completed validation, target filtering, assignment-before-artifact, and cold-binding rules for
  each kernel.
- Deduplicate graph-value buffer declarations once per partition by `ValueId`, retain their first
  deterministic encounter order, and let every kernel use the exact shared declaration reference.
  Input/output/read-write access is the union of all node uses of that value in the partition.
- Declare exact byte size as checked static element count multiplied by the data-type width and
  exact alignment as the data-type width. Zero-element values request zero bytes.
- Generate one scalar single-thread ADD kernel per node. Its exact method signature contains two
  readable and one writable native `MemorySegment`, plus the baked or direct element count already
  required by the specialization. It loops in logical flat order and implements ordinary JVM
  primitive addition: IEEE binary addition for FLOAT64/FLOAT32 and Java two's-complement modular
  addition for INT32/INT64.
- Add only the loop/emitter primitive needed to express this exact generated scalar ADD body.
  Operation semantics remain in the pointwise family emitter rather than the shared generator.
- Replace the single-kernel `CpuPortablePreparedExecutable` recipe with an immutable ordered
  kernel-recipe list. Finalization resolves all shared assignments before the first artifact-store
  access, then loads/generates artifacts in node order.
- Bind each node recipe once from the already validated partition-level direct buffer arguments.
  The resulting partition invocation performs one `RunState` guard and invokes direct typed
  per-node calls in node order. It performs no graph, slot, storage, route, cache, or operation
  dispatch in the hot path.
- Keep task-0004 synthetic tests valid after the ordered-recipe generalization and add focused
  capability, multi-node partition, numerical, storage, ordering, failure, and bytecode tests.
- Keep all execution implementation types package-private in the existing flat
  `io.github.pho001.synaptik.backend.cpu.execution` package. This concrete family needs direct
  collaboration with the completed package-private carrier, generator, artifact, worker,
  candidate, and executable contracts; no cross-package public internal SPI is justified yet.
- Finalize affected Javadocs, CPU backend guidance, glossary impact, and synchronized planning
  evidence through a separate clean documentation-focused pass.

## Out of scope

- SUB, MUL, DIV, MIN, MAX, POW, scalar arithmetic, clamp, comparison, logical, selection, cast,
  classification, unary, activation, reduction, layout, indexing, random, linear algebra,
  convolution, attention, normalization, or loss coverage;
- BFLOAT16, BOOL arithmetic, FLOAT16, mixed data types, promotion, broadcasting, dynamic shapes,
  resolved non-canonical views, strided traversal, or in-place aliasing;
- heap-array or mixed-carrier ADD routes, Vector API execution, parallel execution, unrolling,
  fusion, recomputation, workspace use, tuning, benchmarks, or native vendor libraries;
- changing ADD semantics, Model descriptors, Planning partitioning, compiler graph structure,
  Prepare projection, Runtime schedule/resource contracts, or capability-query shape;
- a public CPU backend/composition facade, registry, service locator, generic operation manager,
  `.internal` package hierarchy, JPMS module descriptor, or public implementation-only SPI;
- changing generated artifact compatibility except for exact new lowering fingerprints produced
  through the existing schema; changing CPU-0003 storage format, publication, validation, or
  cache lifetime;
- execution fallback after CPU ownership, partition splitting during Prepare, or silently skipping
  an unsupported node inside an owned partition;
- other modules, dependencies, Gradle, architecture contracts, ADRs, conformance/integration
  tests, later detailed task specifications, commits, or pushes.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [CPU task 0001](0001-cpu-capability-representation-binding-and-parallel-foundation.md)
- [CPU task 0002](0002-portable-class-file-api-generator-foundation.md)
- [CPU task 0003](0003-bounded-generated-artifact-cache-and-cold-finalization.md)
- [CPU task 0004](0004-typed-portable-analysis-specialization-and-finalization.md)

## Architecture constraints

- Planning owns per-occurrence backend eligibility and maximal consecutive same-owner
  partitioning. CPU preparation must therefore accept the complete supported partition, not
  assume that a CPU-owned partition contains one node.
- Capability truth and candidate acceptance must use one exact semantic predicate. Any occurrence
  reported as supported must be lowerable when it occurs in any ordered CPU-0005-only partition.
- The partition recipe may sequence node kernels without fusing them. This is a backend-private
  implementation choice and does not alter graph or partition semantics.
- `PrepareContext` remains the complete CPU analysis projection. The CPU route may derive exact
  resource needs but may not reacquire Tensor, Compiler, Planning, or storage history.
- Shared Prepare assigns each partition graph-value buffer once. CPU finalization consumes those
  assignments and cannot invent, resize, or reclassify a resource afterward.
- Artifact work remains finalization-only. Bound Runtime execution sees only immutable generated
  recipes and direct cold-bound fields.
- One partition-level `BoundInvocation` owns the sole Runtime state guard. Package-private child
  kernel calls are not separate Runtime schedule entries and add no lifecycle state.
- All new family and sequence machinery remains CPU-private. No dependency or architecture
  boundary changes.

## Package impact

No Java package is added. The root CPU package changes only its existing public capability
provider. The concrete lowering, candidate, generated body, ordered recipe, binder, and invocation
remain package-private in `io.github.pho001.synaptik.backend.cpu.execution`.

This placement is intentional. Java subpackages are separate access domains, while the first real
family needs direct access to the completed package-private execution contracts. A new
`execution.internal.*` hierarchy would therefore require technically public bridge types before a
second concrete consumer proves that boundary useful. CPU 0005 records the concrete dependencies;
later family work may split the package only when a small stable collaboration can be named from
actual use.

## Exact semantic and representation matrix

| Fact | Supported in CPU 0005 |
|---|---|
| Kind/attributes | `BinaryArithmeticKind.ADD` + `NoOperationAttrs.INSTANCE` |
| Inputs/outputs | exactly two inputs and one output |
| Data type | all equal; FLOAT64, FLOAT32, INT32, or INT64 |
| Shape | exact equal fully static Shape; scalar and zero extents valid |
| Resolved layout | canonical contiguous, zero offset, not a view |
| Unresolved layout | accepted and materialized as canonical flat representation |
| Carrier | native `MemorySegment` for all three arguments |
| Execution mode | `SCALAR_SINGLE_THREAD` |
| Numerical mode | `EXACT_DEFAULT` |
| Tail/combine | `Tail.NONE`, fixed combine order |
| Output | fresh selected logical buffer; no input/output alias |

No other row is inferred from a related type or operation.

## Ordered partition contract

- The candidate source visits `context.nodes()` exactly once in stored partition order.
- Each node must have one output and the exact CPU-0005 ADD semantics. All input/output IDs must
  resolve to the exact projected `GraphValue` references already validated by `PrepareContext`.
- Buffer declarations are interned by equal `ValueId` in first encounter order while visiting each
  node's inputs then output. Their geometry must match on every later use.
- A value read and written by different node recipes receives partition-level `READ_WRITE` access;
  input-only values are `READ_ONLY`, and output-only values are `WRITE_ONLY`.
- Every node kernel receives its arguments in `[left, right, output]` order and references the
  exact shared requirement objects. Repeated input IDs remain repeated argument positions.
- Finalization first resolves every shared declaration and every node use to prepared plan
  positions. Only then may it consult the artifact store, in node order.
- The prepared executable retains immutable node recipes in partition order. Cold binding creates
  a direct typed node invocation for each recipe, then one immutable partition invocation.
- Hot execution invokes node calls in order. Intermediate and boundary values use the assigned
  shared buffers; no graph lookup or data-dependent dispatch occurs.

## Validation and failure order

Automated tests lock this order:

1. capability query nullity, operation kind/attributes, occurrence count, data types, shapes,
   static extent, then layouts;
2. pointwise source context, exact CPU owner inherited from CPU 0004, non-empty ordered nodes,
   per-node semantic validation, projected value resolution, checked geometry, and declaration
   interning;
3. ordered partition candidate structure, shared declaration identity, per-kernel specialization,
   emitter fingerprint, uses, and binder;
4. target/mode validation and deterministic selection through the generalized CPU-0004 preparer;
5. complete shared assignment mapping before artifact access, then node-order artifact finalization;
6. partition cold binding, per-node direct carrier extraction, then immutable partition invocation;
7. one partition state guard followed by exact ordered node execution.

Unsupported valid queries return `false`. A CPU-owned partition containing a node outside the
exact matrix fails closed before any resource declaration, artifact access, generation, or
execution.

## Affected files

Expected production paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableCandidateSource.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableKernelCandidate.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortableInvocationBinder.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionCandidate.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddCandidateSource.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddKernelEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddInvocation.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/execution/package-info.java`

Expected test paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderPublicShapeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPortablePreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddCandidateSourceTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/execution/CpuPointwiseAddExecutionTest.java`

Expected explanatory documentation:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`

Expected planning paths:

- this task;
- `docs/planning/backends/cpu/master-plan.md`;
- `docs/planning/roadmap.md`.

Review only unless a concrete contradiction stops the task: architecture/ADRs/tests; Model,
Planning, Prepare, Runtime, Compiler, Config, Trace, Engine, other backends, Gradle, conformance and
integration paths; completed CPU 0001–0004 task history.

## Maximum scope

At most 26 paths: 14 production, 7 tests, 2 explanatory documents, and 3 planning documents. The
implementation may use fewer paths or co-locate a small package-private role where that improves
cohesion without changing the exact behavior. Any need for another module, public execution type,
generic invocation adapter, partitioning change, build/dependency change, or operation beyond the
exact matrix is a stop condition rather than implicit scope expansion.

## Acceptance criteria

- [x] `CpuCapabilityProvider` advertises exactly the CPU-0005 matrix and remains false otherwise.
- [x] Any non-empty maximal partition consisting only of advertised occurrences is accepted and
      represented as an ordered node-kernel sequence; unsupported mixed partitions fail closed.
- [x] Shared buffer declarations are unique by value ID, deterministic, geometrically exact, and
      reused by exact reference across node recipes.
- [x] Generated scalar single-thread ADD is exact for FLOAT64, FLOAT32, INT32, and INT64, including
      scalar, zero-element, repeated-input, fan-out, chain, independent-node, and boundary cases.
- [x] Floating behavior follows JVM IEEE addition; integral overflow is exact modular addition.
- [x] Finalization resolves all assignments before artifact access and loads/generates artifacts
      in node order through the unchanged CPU-0003 store.
- [x] One partition bound invocation performs one state guard and direct ordered typed node calls
      without graph/slot/storage/route/cache/operation lookup in the hot path.
- [x] CPU 0001–0004 contracts remain compatible; their focused tests pass after the ordered-recipe
      generalization.
- [x] No public CPU type, package, registry, facade, service locator, generic manager, or JPMS
      boundary is added.
- [x] No unadvertised operation, data type, layout, carrier, mode, fusion, or vendor route is
      implemented or claimed.
- [x] Focused tests and exactly one final `:backends:cpu:test` after Java stability pass.
- [x] A separate clean documentation pass finalizes Javadocs, CPU guide, glossary, planning
      evidence, links, scope, status, and whitespace without repeating successful Java tests unless
      it changes executable Java.

## Tests / validation

Focused development commands may target the seven affected tests. After executable Java is stable,
run exactly once:

```bash
./gradlew :backends:cpu:test
git diff --check
```

The separate documentation pass runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates generated Javadocs, Markdown links/anchors/fences/newlines, exact scope, public
surface, capability matrix, task/master/roadmap status, preserved completed history, absence of a
detailed CPU 0005A specification, and unchanged excluded paths. Repository-wide validation remains
deferred to CPU 0009 because this task changes one backend and no architecture/dependency boundary.

## Dependencies

- Complete CPU 0001 representation, direct arguments, worker, and Runtime binding contracts.
- Complete CPU 0002 specialization, emitter, generated class, and exact direct handle contracts.
- Complete CPU 0003 durable artifact store and cold loading.
- Complete CPU 0004 typed analysis, resource declaration, assignment, finalization, and executable
  foundation.
- Current Model ADD, static Shape, descriptor/layout, compiled graph, Planning partition/memory,
  Prepare projection, and Runtime resource contracts.

## Follow-up tasks

- CPU 0005A remains Draft without a detailed specification. It extends the pointwise matrix with
  remaining exact arithmetic plus heap/mixed carriers and Vector/parallel routes based on the
  concrete CPU-0005 sequence contract.
- Later CPU 0005-family Draft slices add comparison/logical/selection/cast/classification and
  exact unary/activation semantics. CPU 0006–0009 retain their existing family/closure ownership.
- A later Engine/CPU composition task owns public construction, artifact-root and worker lifetime,
  representation materialization, and end-to-end scheduling.

## Architecture impact

Expected impact: None. The task extends only CPU-private implementation recipes behind existing
Planning, Prepare, Runtime, Model, and generated-kernel contracts. Stop if implementation requires
any shared or authoritative architecture change.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, documentation/planning rules, the CPU master plan, completed CPU
tasks 0001–0004, and task 0005 in full. Inspect every affected and review-only source/test named by
the task before editing.

Implement docs/planning/backends/cpu/tasks/0005-dense-add-and-partition-sequence-execution.md
exactly within its 26-path maximum. Add only the exact static canonical dense FLOAT64/FLOAT32/
INT32/INT64 parameterless ADD matrix, scalar single-thread native-segment generated route, and the
CPU-private ordered node-kernel recipe needed to execute any maximal partition made only of those
advertised occurrences. Preserve completed CPU 0001–0004, shared Planning/Prepare/Runtime/Model
contracts, artifact semantics, and architecture boundaries. Add no other operation, carrier,
mode, fusion, public execution surface, package hierarchy, dependency, build, architecture, or
later detailed task. Stop on architecture, partition-truth, invocation, lifecycle, numerical,
affected-file, or scope conflict.

Run focused tests while developing and exactly one final :backends:cpu:test after executable Java
stabilizes. Then hand the actual diff and exact Java evidence to a separate clean-context
documentation-focused agent in the same overall change. That agent must independently inspect the
final contracts, finalize affected Javadocs, CPU guide, glossary, task/master/roadmap evidence,
run CPU Javadoc and documentation/scope/status/whitespace checks, and reuse successful Java
evidence unless executable behavior changes or it records a concrete stale-evidence reason.

Do not mark CPU 0005 Complete until both passes and every acceptance criterion succeed. Leave CPU
0005A and all later work Draft without detailed specifications.
```

## Local decisions

- Start with one exact operation but solve maximal-partition sequencing now. Per-occurrence
  capability would otherwise be untruthful as soon as Planning grouped two adjacent ADD nodes.
- Sequence one generated kernel per node instead of adding premature fusion. This establishes a
  complete correctness route and leaves later fusion as an alternative candidate, not a semantic
  requirement.
- Use only scalar single-thread native segments in the first vertical slice. It exercises the
  canonical internal representation and exact generator/artifact/runtime path without combining
  operation expansion with carrier and mode expansion.
- Keep the flat execution package. The concrete family depends on many completed package-private
  roles, so moving it now would force public internal bridges without a second proven consumer.
- Restrict resolved layouts to canonical zero-offset non-view geometry. Unresolved ordinary
  elementwise outputs are admitted because the selected CPU representation is a canonical
  materialization, not an alias of unresolved physical geometry.

## Known limitations

- CPU 0005 supports only ADD and only one portable route. It is a correctness and lifecycle
  vertical slice, not pointwise-family closure or a performance milestone.
- No public Engine composition exists yet, so focused CPU tests prove the backend-private complete
  prepare/finalize/bind/execute path without claiming public end-to-end execution.
- Sequenced nodes materialize intermediates and dispatch separate generated kernels. Later exact
  fusion candidates may remove those boundaries without changing semantics.
- Dynamic shapes, non-canonical layouts, heap/mixed carriers, Vector/parallel routes, BFLOAT16,
  BOOL, and FLOAT16 remain unsupported.

## Validation evidence

Implementation context `019fd19a-4262-7580-9674-226595356fbc` ran the sole final
`./gradlew :backends:cpu:test` after executable Java stabilized: 31 tests passed with no failures,
errors, or skips and Gradle reported `BUILD SUCCESSFUL`. Clean documentation context
`019fd1af-84ce-7aa1-b2d9-d07424ac37ec` (`/root/cpu_0005_docs`) independently inspected the final
implementation, tests, generated Javadoc,
architecture boundary, documentation profiles, and planning contracts. It changed comments and
documentation only, so it reused the successful Java evidence rather than repeating the suite.

The documentation pass ran `./gradlew :backends:cpu:javadoc` successfully. Its targeted Markdown
checker validated five changed documents, 677 local links, 287 anchors, balanced fences, final
newlines, and trailing whitespace. Generated-page inspection found the exact ADD route, ordered
node-kernel recipe, sole partition guard, and fail-closed descriptions. `javap -public` confirmed
that `CpuCapabilityProvider` remains the only changed public CPU surface and retains only its
existing public constant, constructor, `backendId()`, and `supports(...)` members. Source/import
inspection confirmed the exact operation/type/layout matrix and architecture-approved inward
dependencies. Exact inventory validation found 24 paths: 13 production, 6 tests, 2 explanatory
documents, and 3 planning documents, within the 26-path ceiling. `git diff --check` passed.

No architecture rule, ADR, module boundary, dependency, Gradle configuration, public package,
backend-conformance contract, integration contract, or other module changed, so architecture,
architecture-test, conformance, integration, Gradle, and other-module updates were correctly not
required. The task directory contains no detailed CPU 0005A specification; master-plan and roadmap
status checks retain CPU 0005A and later work as Draft.

## Implementation notes

- The capability provider now reports only exact parameterless ADD with two inputs and one output,
  equal fully static shapes, one common supported data type, and unresolved or canonical resolved
  layouts.
- The production candidate source constructs deterministic partition-wide buffer declarations and
  one node recipe per advertised occurrence, rejecting empty, mixed, unsupported, or aliased
  partitions before artifact work.
- Finalization resolves every shared assignment before loading artifacts in planned node order.
  Cold binding builds direct typed child calls, and the bound partition owns the sole state guard.
- The generated emitter is deliberately scalar, single-thread, and native-segment only. FLOAT32
  and FLOAT64 preserve JVM IEEE addition, while INT32 and INT64 preserve Java modular overflow.
- The clean documentation pass finalized affected Javadocs, package documentation, the CPU guide,
  glossary status, task evidence, master plan, and roadmap without widening capability claims.

## Completion summary

Completed changes: implemented and documented the exact dense ADD capability, ordered maximal-
partition candidate/finalization/binding/execution route, deterministic shared-resource reuse, and
four exact scalar native-segment kernels while preserving all exclusions.

Files changed or created: 13 CPU production files, 6 CPU tests, `docs/backend-guide/cpu-backend.md`,
`docs/glossary.md`, this task, `docs/planning/backends/cpu/master-plan.md`, and
`docs/planning/roadmap.md`; no file outside the task's permitted paths changed.

Tests and validation: sole final CPU module suite passed 31 tests; CPU Javadoc built successfully;
generated docs, Markdown links/anchors/fences/newlines/whitespace, public shape, imports, exact
scope, statuses, Draft-task absence, and diff whitespace passed inspection.

Unresolved issues: none within CPU 0005. CPU 0005A and later operation, carrier, execution-mode,
fusion, compiler, public composition, conformance, and integration work remain intentionally Draft.

Required follow-up: none for CPU 0005.

Status: Complete
