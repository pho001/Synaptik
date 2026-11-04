# Task 0005: Dense Add and [Partition](../../../../glossary.md#partition)-Sequence Execution

## Reader context

This was the first real arithmetic slice, but its unit of generated work was still one graph
node. Planning could group consecutive supported ADD occurrences into one CPU-owned partition;
CPU preparation therefore had to share intermediate buffer declarations and retain child calls
in producer-before-consumer order. Runtime invoked one partition-level bound object, which
called those already-bound children rather than rediscovering graph dependencies.

The Tensor/graph descriptions supplied shapes and types; native segments held the actual left,
right, and result values. Each ADD wrote a distinct result buffer instead of modifying either
input. The later atomic reset replaced this per-node implementation with computation-oriented
units and virtual intermediates. The superseded status protects that distinction: the narrow
matrix and historical test receipts are not a current support table.

Current reading alongside this historical record:

- [CPU partition and unit lowering](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternallowering)
  — explains virtual intermediates, boundary stores, and family-specific mappings.
- [CPU binding, validation, and ranges](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalexecutable)
  — identifies current-value validation, direct invocation state, and worker/resource lifetimes.

## Status

Superseded

## Goal

Deliver the first truthful executable [Model](../../../../glossary.md#term-model)-[operation](../../../../glossary.md#operation) coverage through the completed [CPU](../../../../glossary.md#term-cpu)
representation, generator, [artifact](../../../../glossary.md#term-artifact), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), and worker foundations:

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

The supported semantic [matrix](../../../../glossary.md#term-matrix-meanings) is deliberately narrow: parameterless
`BinaryArithmeticKind.ADD`, exact equal input/output [shapes](../../../../glossary.md#shape), fully static shapes, exact equal
input/output [data type](../../../../glossary.md#data-type--datatype), and [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), [`INT32`](../../../../glossary.md#data-type--datatype), or [`INT64`](../../../../glossary.md#data-type--datatype). The first route uses
[canonical](../../../../glossary.md#term-canonical-form) native [`MemorySegment`](../../../../glossary.md#term-memorysegment) [buffers](../../../../glossary.md#term-buffer) and [scalar](../../../../glossary.md#term-scalar) single-thread generated code. This task also
extends the CPU-0004 single-[kernel](../../../../glossary.md#kernel) recipe into an ordered [partition](../../../../glossary.md#partition) recipe so truthful per-[node](../../../../glossary.md#node)
capability reporting remains executable when [Planning](../../../../glossary.md#term-planning) groups any number of consecutive supported
occurrences into one maximal same-[owner](../../../../glossary.md#term-owner) partition.

The task does not claim the rest of the [elementwise](../../../../glossary.md#term-elementwise) family. CPU 0005A and later Draft work extend
the exact matrix, carriers, and modes after this vertical [slice](../../../../glossary.md#slice) is complete.

## [Scope](../../../../glossary.md#term-scope)

- Change `CpuCapabilityProvider` from globally [fail-closed](../../../../glossary.md#term-fail-closed) to returning `true` only for the exact
  CPU-0005 ADD occurrence matrix and <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [occurrence glossary](../../../../glossary.md#term-occurrence); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>`false` for every other valid query.
- Accept one ADD occurrence only when it has two inputs and one output, all three descriptors have
  the same supported data type and exact equal fully static shape, and every resolved layout is
  canonical contiguous with zero storage offset. An unresolved layout is valid because the CPU
  route materializes the logical value into its selected canonical representation. A resolved
  view, offset, broadcast, permuted, or otherwise non-canonical layout is rejected.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [data type glossary](../../../../glossary.md#data-type--datatype); [shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [canonical glossary](../../../../glossary.md#term-canonical-form); [contiguous glossary](../../../../glossary.md#term-contiguous); [storage offset glossary](../../../../glossary.md#term-storage-offset); [view glossary](../../../../glossary.md#term-view-meanings))</span>
- Preserve null [validation](../../../../glossary.md#term-validation) and use structural [operation](../../../../glossary.md#operation) kind/attribute contracts; only
  `NoOperationAttrs.INSTANCE` with `BinaryArithmeticKind.ADD` is admitted.
- Add a package-private immutable pointwise ADD lowering recipe for one graph-node occurrence.
  It retains exact ordered input/output <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [pointwise glossary](../../../../glossary.md#term-pointwise); [lowering glossary](../../../../glossary.md#lowering))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) references, exact data type, exact static element
  count, and a deterministic lowering fingerprint containing every bytecode-relevant semantic
  fact but no graph-local identity.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [fingerprint glossary](../../../../glossary.md#term-fingerprint); [bytecode glossary](../../../../glossary.md#term-bytecode); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Add a package-private family candidate source that validates every [node](../../../../glossary.md#node) in the supplied
  [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) against the same public capability predicate and builds one ordered kernel
  recipe per partition node. It must never accept a partially supported partition.
- Extend the CPU-0004 candidate/plan/finalization path from exactly one kernel to one non-empty
  ordered partition candidate containing one or more complete kernel candidates. Preserve the
  completed validation, target filtering, assignment-before-artifact, and cold-binding rules for
  each kernel.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-normative))</span><span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Deduplicate [graph](../../../../glossary.md#graph)-value [buffer](../../../../glossary.md#term-buffer) declarations once per [partition](../../../../glossary.md#partition) by `ValueId`, retain their first
  deterministic encounter order, and let every kernel use the exact shared declaration reference.
  Input/output/read-write access is the union of all node uses of that value in the partition.
- Declare exact byte size as checked static element count multiplied by the data-type width and
  exact alignment as the data-type width. Zero-element values request zero bytes.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
- Generate one scalar single-thread ADD kernel per node. Its exact method signature contains two
  readable and one writable native <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [signature glossary](../../../../glossary.md#term-signature))</span>[`MemorySegment`](../../../../glossary.md#term-memorysegment), plus the baked or direct element count already
  required by the specialization. It loops in logical flat order and implements ordinary JVM
  primitive addition: IEEE binary addition for FLOAT64/FLOAT32 and Java two's-complement modular
  addition for INT32/INT64.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [IEEE glossary](../../../../glossary.md#term-ieee); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [two's-complement glossary](../../../../glossary.md#arithmetic-rearrangement-and-finite-difference-checks); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>
- Add only the loop/emitter primitive needed to express this exact generated scalar ADD body.
  Operation semantics remain in the pointwise family emitter rather than the shared generator.
- Replace the single-[kernel](../../../../glossary.md#kernel) `CpuPortablePreparedExecutable` recipe with an immutable ordered
  kernel-recipe list. Finalization resolves all shared assignments before the first artifact-store
  access, then loads/generates artifacts in node order.
- Bind each node recipe once from the already validated partition-level direct buffer arguments.
  The resulting partition invocation performs one <span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument))</span>[`RunState`](../../../../glossary.md#run-state--runstate) guard and invokes direct typed
  per-node calls in node order. It performs no graph, slot, storage, route, cache, or operation
  dispatch in the hot path.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard); [cache glossary](../../../../glossary.md#term-cache); [dispatch glossary](../../../../glossary.md#term-dispatch); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- Keep task-0004 synthetic tests valid after the ordered-recipe generalization and add focused
  capability, multi-node partition, numerical, storage, ordering, failure, and bytecode tests.<span data-glossary-navigation="true"> ([generalization glossary](../../../../glossary.md#term-generalization))</span>
- Keep all execution implementation types package-private in the existing flat
  `io.github.pho001.synaptik.backend.cpu.execution` package. This concrete family needs direct
  collaboration with the completed package-private carrier, generator, artifact, worker,
  candidate, and executable contracts; no cross-package public internal SPI is justified yet.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi))</span>
- Finalize affected Javadocs, CPU backend guidance, glossary impact, and synchronized planning
  evidence through a separate clean documentation-focused pass.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- SUB, MUL, DIV, MIN, MAX, POW, scalar arithmetic, clamp, comparison, logical, selection, cast,
  classification, unary, activation, reduction, layout, indexing, random, linear algebra,
  convolution, attention, normalization, or loss coverage;<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [unary glossary](../../../../glossary.md#term-unary); [activation glossary](../../../../glossary.md#term-activation); [layout glossary](../../../../glossary.md#layout); [linear glossary](../../../../glossary.md#term-linear-meanings); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [normalization glossary](../../../../glossary.md#term-normalization); [loss glossary](../../../../glossary.md#loss))</span>
- BFLOAT16, BOOL arithmetic, FLOAT16, mixed data types, promotion, broadcasting, dynamic shapes,
  resolved non-canonical views, strided traversal, or in-place aliasing;<span data-glossary-navigation="true"> ([BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [FLOAT16 glossary](../../../../glossary.md#term-float16); [broadcasting glossary](../../../../glossary.md#broadcasting); [shapes glossary](../../../../glossary.md#shape); [canonical glossary](../../../../glossary.md#term-canonical-form); [views glossary](../../../../glossary.md#term-view-meanings); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
- heap-array or mixed-carrier ADD routes, Vector API execution, parallel execution, unrolling,
  fusion, recomputation, workspace use, tuning, benchmarks, or native vendor libraries;<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier); [Vector API glossary](../../../../glossary.md#term-vector-api); [unrolling glossary](../../../../glossary.md#term-unrolling); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- changing ADD semantics, Model descriptors, Planning partitioning, compiler graph structure,
  Prepare projection, Runtime schedule/resource contracts, or capability-query shape;<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Planning glossary](../../../../glossary.md#term-planning); [compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- a public [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend)/composition [facade](../../../../glossary.md#term-facade), [registry](../../../../glossary.md#term-registry), [service locator](../../../../glossary.md#term-service-locator), generic [operation](../../../../glossary.md#operation) manager,
  `.internal` package hierarchy, [JPMS](../../../../glossary.md#term-java-platform-module-system-jpms) [module](../../../../glossary.md#term-module) [descriptor](../../../../glossary.md#term-descriptor), or public implementation-only [SPI](../../../../glossary.md#term-service-provider-interface-spi);
- changing generated artifact compatibility except for exact new lowering fingerprints produced
  through the existing schema; changing CPU-0003 storage format, publication, validation, or
  cache lifetime;<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [lowering glossary](../../../../glossary.md#lowering); [schema glossary](../../../../glossary.md#term-schema); [publication glossary](../../../../glossary.md#term-publication); [validation glossary](../../../../glossary.md#term-validation); [cache glossary](../../../../glossary.md#term-cache); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- execution fallback after CPU ownership, partition splitting during Prepare, or silently skipping
  an unsupported node inside an owned partition;<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [partition glossary](../../../../glossary.md#partition); [node glossary](../../../../glossary.md#node))</span>
- other modules, dependencies, Gradle, architecture contracts, ADRs, conformance/integration
  tests, later detailed task specifications, commits, or pushes.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [conformance glossary](../../../../glossary.md#term-conformance); [commits glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime, Prepare, and Backend Boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU task 0001](0001-cpu-capability-representation-binding-and-parallel-foundation.md)
- [CPU task 0002](0002-portable-class-file-api-generator-foundation.md)
- [CPU task 0003](0003-bounded-generated-artifact-cache-and-cold-finalization.md)
- [CPU task 0004](0004-typed-portable-analysis-specialization-and-finalization.md)

## Architecture constraints

- Planning owns per-occurrence backend eligibility and maximal consecutive same-owner
  partitioning. CPU preparation must therefore accept the complete supported partition, not
  assume that a CPU-owned partition contains one node.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [occurrence glossary](../../../../glossary.md#term-occurrence); [backend glossary](../../../../glossary.md#backend); [owner glossary](../../../../glossary.md#term-owner); [CPU glossary](../../../../glossary.md#term-cpu); [node glossary](../../../../glossary.md#node))</span>
- Capability truth and candidate acceptance must use one exact semantic predicate. Any occurrence
  reported as supported must be lowerable when it occurs in any ordered CPU-0005-only partition.
- The partition recipe may sequence node kernels without fusing them. This is a backend-private
  implementation choice and does not alter graph or partition semantics.<span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel); [graph glossary](../../../../glossary.md#graph))</span>
- [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) remains the complete CPU analysis projection. The CPU route may derive exact
  resource needs but may not reacquire Tensor, Compiler, Planning, or storage history.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Shared Prepare assigns each partition graph-value buffer once. CPU finalization consumes those
  assignments and cannot invent, resize, or reclassify a resource afterward.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- Artifact work remains finalization-only. Bound Runtime execution sees only immutable generated
  recipes and direct cold-bound fields.<span data-glossary-navigation="true"> ([Artifact glossary](../../../../glossary.md#term-artifact); [Runtime glossary](../../../../glossary.md#term-runtime); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- One [partition](../../../../glossary.md#partition)-level [`BoundInvocation`](../../../../glossary.md#bound-invocation--boundinvocation) owns the sole Runtime state guard. Package-private child
  kernel calls are not separate Runtime schedule entries and add no lifecycle state.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard); [kernel glossary](../../../../glossary.md#kernel); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- All new family and sequence machinery remains CPU-private. No dependency or architecture
  boundary changes.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

No Java package is added. The root [CPU](../../../../glossary.md#term-cpu) package changes only its existing public capability
provider. The concrete [lowering](../../../../glossary.md#lowering), candidate, generated body, ordered recipe, binder, and invocation
remain package-private in `io.github.pho001.synaptik.backend.cpu.execution`.

This placement is intentional. Java subpackages are separate access domains, while the first real
family needs direct access to the completed package-private execution contracts. A new
`execution.internal.*` hierarchy would therefore require technically public bridge types before a
second concrete consumer proves that boundary useful. CPU 0005 records the concrete dependencies;
later family work may split the package only when a small stable collaboration can be named from
actual use.

## Exact semantic and representation [matrix](../../../../glossary.md#term-matrix-meanings)

| Fact | Supported in [CPU](../../../../glossary.md#term-cpu) 0005 |
|---|---|
| Kind/attributes | `BinaryArithmeticKind.ADD` + `NoOperationAttrs.INSTANCE` |
| Inputs/outputs | exactly two inputs and one output |
| [Data type](../../../../glossary.md#data-type--datatype) | all equal; [FLOAT64](../../../../glossary.md#term-float64), [FLOAT32](../../../../glossary.md#term-float32), [INT32](../../../../glossary.md#data-type--datatype), or [INT64](../../../../glossary.md#data-type--datatype) |
| [Shape](../../../../glossary.md#shape) | exact equal fully static Shape; [scalar](../../../../glossary.md#term-scalar) and zero extents valid |
| Resolved [layout](../../../../glossary.md#layout) | [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous), zero offset, not a [view](../../../../glossary.md#term-view-meanings) |
| Unresolved layout | accepted and materialized as canonical flat representation |
| [Carrier](../../../../glossary.md#term-storage-carrier) | native [`MemorySegment`](../../../../glossary.md#term-memorysegment) for all three [arguments](../../../../glossary.md#term-argument) |
| Execution mode | `SCALAR_SINGLE_THREAD` |
| Numerical mode | `EXACT_DEFAULT` |
| Tail/combine | `Tail.NONE`, fixed combine order |
| Output | fresh selected logical [buffer](../../../../glossary.md#term-buffer); no input/output [alias](../../../../glossary.md#aliasing-and-arena) |

No other row is inferred from a related type or [operation](../../../../glossary.md#operation).

## Ordered [partition](../../../../glossary.md#partition) contract

- The candidate source visits `context.nodes()` exactly once in stored partition order.
- Each node must have one output and the exact CPU-0005 ADD semantics. All input/output IDs must
  resolve to the exact projected <span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>`GraphValue` references already validated by [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext).
- [Buffer](../../../../glossary.md#term-buffer) declarations are interned by equal [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) in first encounter order while visiting each
  node's inputs then output. Their geometry must match on every later use.
- A value read and written by different [node](../../../../glossary.md#node) recipes receives partition-level `READ_WRITE` access;
  input-only values are `READ_ONLY`, and output-only values are `WRITE_ONLY`.
- Every node [kernel](../../../../glossary.md#kernel) receives its [arguments](../../../../glossary.md#term-argument) in `[left, right, output]` order and references the
  exact shared requirement objects. Repeated input IDs remain repeated argument positions.
- Finalization first resolves every shared declaration and every node use to prepared plan
  positions. Only then may it consult the artifact store, in node order.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- The prepared executable retains immutable node recipes in partition order. Cold binding creates
  a direct typed node invocation for each recipe, then one immutable partition invocation.<span data-glossary-navigation="true"> ([prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [immutable glossary](../../../../glossary.md#term-immutable); [binding glossary](../../../../glossary.md#term-binding))</span>
- Hot execution invokes node calls in order. Intermediate and boundary values use the assigned
  shared buffers; no graph lookup or data-dependent dispatch occurs.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>

## [Validation](../../../../glossary.md#term-validation) and failure order

Automated tests lock this order:

1. capability query nullity, operation kind/attributes, occurrence count, data types, shapes,
   static extent, then layouts;<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [occurrence glossary](../../../../glossary.md#term-occurrence); [shapes glossary](../../../../glossary.md#shape); [extent glossary](../../../../glossary.md#term-extent); [layouts glossary](../../../../glossary.md#layout))</span>
2. pointwise source context, exact CPU owner inherited from CPU 0004, non-empty ordered nodes,
   per-node semantic validation, projected value resolution, checked geometry, and declaration
   interning;<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [owner glossary](../../../../glossary.md#term-owner))</span>
3. ordered partition candidate structure, shared declaration identity, per-kernel specialization,
   emitter fingerprint, uses, and binder;<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [kernel glossary](../../../../glossary.md#kernel); [specialization glossary](../../../../glossary.md#term-specialization); [fingerprint glossary](../../../../glossary.md#term-fingerprint))</span>
4. [target](../../../../glossary.md#term-target)/mode validation and deterministic selection through the generalized [CPU](../../../../glossary.md#term-cpu)-0004 preparer;
5. complete shared assignment mapping before [artifact](../../../../glossary.md#term-artifact) access, then [node](../../../../glossary.md#node)-order artifact finalization;
6. [partition](../../../../glossary.md#partition) cold [binding](../../../../glossary.md#term-binding), per-node direct [carrier](../../../../glossary.md#term-storage-carrier) extraction, then [immutable](../../../../glossary.md#term-immutable) partition invocation;
7. one partition state [guard](../../../../glossary.md#term-guard) followed by exact ordered node execution.

Unsupported valid queries return `false`. A CPU-owned partition containing a node outside the
exact [matrix](../../../../glossary.md#term-matrix-meanings) fails closed before any resource declaration, artifact access, generation, or
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

Expected [planning](../../../../glossary.md#term-planning) paths:

- this task;
- `docs/planning/backends/cpu/master-plan.md`;
- `docs/planning/roadmap.md`.

Review only unless a concrete contradiction stops the task: architecture/ADRs/tests; [Model](../../../../glossary.md#term-model),
Planning, [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Compiler](../../../../glossary.md#term-compiler), Config, [Trace](../../../../glossary.md#trace), [Engine](../../../../glossary.md#term-engine), other [backends](../../../../glossary.md#backend), [Gradle](../../../../glossary.md#term-gradle), [conformance](../../../../glossary.md#term-conformance) and
integration paths; completed [CPU](../../../../glossary.md#term-cpu) 0001–0004 task history.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 26 paths: 14 production, 7 tests, 2 explanatory documents, and 3 [planning](../../../../glossary.md#term-planning) documents. The
implementation may use fewer paths or co-locate a small package-private role where that improves
cohesion without changing the exact behavior. Any need for another [module](../../../../glossary.md#term-module), public execution type,
generic invocation adapter, partitioning change, build/[dependency](../../../../glossary.md#term-dependency) change, or [operation](../../../../glossary.md#operation) beyond the
exact [matrix](../../../../glossary.md#term-matrix-meanings) is a stop condition rather than implicit scope expansion.

## Acceptance criteria

- [x] `CpuCapabilityProvider` advertises exactly the [CPU](../../../../glossary.md#term-cpu)-0005 [matrix](../../../../glossary.md#term-matrix-meanings) and remains false otherwise.
- [x] Any non-empty maximal partition consisting only of advertised occurrences is accepted and
      represented as an ordered node-kernel sequence; unsupported mixed partitions fail closed.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [node glossary](../../../../glossary.md#node); [kernel glossary](../../../../glossary.md#kernel); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- [x] Shared buffer declarations are unique by value ID, deterministic, geometrically exact, and
      reused by exact reference across node recipes.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer))</span>
- [x] Generated scalar single-thread ADD is exact for FLOAT64, FLOAT32, INT32, and INT64, including
      scalar, zero-element, repeated-input, fan-out, chain, independent-node, and boundary cases.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [FLOAT64 glossary](../../../../glossary.md#term-float64); [FLOAT32 glossary](../../../../glossary.md#term-float32); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [element glossary](../../../../glossary.md#term-element); [fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- [x] Floating behavior follows [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) [IEEE](../../../../glossary.md#term-ieee) addition; integral [overflow](../../../../glossary.md#term-overflow) is exact modular addition.
- [x] Finalization resolves all assignments before artifact access and loads/generates artifacts
      in node order through the unchanged CPU-0003 store.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>
- [x] One partition bound invocation performs one state guard and direct ordered typed node calls
      without graph/slot/storage/route/cache/operation lookup in the hot path.<span data-glossary-navigation="true"> ([bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [guard glossary](../../../../glossary.md#term-guard); [graph glossary](../../../../glossary.md#graph); [cache glossary](../../../../glossary.md#term-cache); [operation glossary](../../../../glossary.md#operation); [lookup glossary](../../../../glossary.md#term-lookup); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
- [x] CPU 0001–0004 contracts remain compatible; their focused tests pass after the ordered-recipe
      generalization.<span data-glossary-navigation="true"> ([generalization glossary](../../../../glossary.md#term-generalization))</span>
- [x] No public CPU type, package, registry, facade, service locator, generic manager, or JPMS
      boundary is added.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [facade glossary](../../../../glossary.md#term-facade); [service locator glossary](../../../../glossary.md#term-service-locator); [JPMS glossary](../../../../glossary.md#term-java-platform-module-system-jpms))</span>
- [x] No unadvertised operation, data type, layout, carrier, mode, fusion, or vendor route is
      implemented or claimed.<span data-glossary-navigation="true"> ([data type glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [carrier glossary](../../../../glossary.md#term-storage-carrier); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- [x] Focused tests and exactly one final `:backends:cpu:test` after Java stability pass.
- [x] A separate clean documentation pass finalizes Javadocs, CPU guide, glossary, planning
      evidence, links, scope, status, and whitespace without repeating successful Java tests unless
      it changes executable Java.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [scope glossary](../../../../glossary.md#term-scope))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

Focused development commands may [target](../../../../glossary.md#term-target) the seven affected tests. After executable Java is stable,
[run](../../../../glossary.md#term-run) exactly once:

```bash
./gradlew :backends:cpu:test
git diff --check
```

The separate documentation pass runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates generated Javadocs, Markdown links/anchors/fences/newlines, exact [scope](../../../../glossary.md#term-scope), public
surface, capability [matrix](../../../../glossary.md#term-review-matrix), task/master/roadmap status, preserved completed history, absence of a
detailed [CPU](../../../../glossary.md#term-cpu) 0005A specification, and unchanged excluded paths. Repository-wide validation remains
deferred to CPU 0009 because this task changes one [backend](../../../../glossary.md#backend) and no architecture/[dependency](../../../../glossary.md#term-dependency) boundary.

## Dependencies

- Complete [CPU](../../../../glossary.md#term-cpu) 0001 representation, direct [arguments](../../../../glossary.md#term-argument), worker, and [Runtime](../../../../glossary.md#term-runtime) [binding](../../../../glossary.md#term-normative) contracts.
- Complete CPU 0002 [specialization](../../../../glossary.md#term-specialization), emitter, generated class, and exact direct handle contracts.
- Complete CPU 0003 durable [artifact](../../../../glossary.md#term-artifact) store and cold loading.
- Complete CPU 0004 typed analysis, resource declaration, assignment, finalization, and executable
  foundation.
- Current Model ADD, static Shape, descriptor/layout, compiled graph, Planning partition/memory,
  Prepare projection, and Runtime resource contracts.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Shape glossary](../../../../glossary.md#shape); [descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [compiled graph glossary](../../../../glossary.md#compiled-graph--compiledgraphmodel); [Planning glossary](../../../../glossary.md#term-planning); [partition glossary](../../../../glossary.md#partition); [Prepare glossary](../../../../glossary.md#prepare))</span>

## Follow-up tasks

- CPU 0005A remains Draft without a detailed specification. It extends the pointwise matrix with
  remaining exact arithmetic plus heap/mixed carriers and Vector/parallel routes based on the
  concrete CPU-0005 sequence contract.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [pointwise glossary](../../../../glossary.md#term-pointwise); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [heap glossary](../../../../glossary.md#term-heap); [Vector glossary](../../../../glossary.md#term-vector))</span>
- Later CPU 0005-family Draft slices add comparison/logical/selection/cast/classification and
  exact unary/activation semantics. CPU 0006–0009 retain their existing family/closure ownership.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [activation glossary](../../../../glossary.md#term-activation); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- A later Engine/CPU composition task owns public construction, artifact-root and worker lifetime,
  representation materialization, and end-to-end scheduling.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [artifact glossary](../../../../glossary.md#term-artifact); [lifetime glossary](../../../../glossary.md#term-lifetime); [materialization glossary](../../../../glossary.md#materialization))</span>

## Architecture impact

Expected impact: None. The task extends only [CPU](../../../../glossary.md#term-cpu)-private implementation recipes behind existing
[Planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Model](../../../../glossary.md#term-model), and generated-[kernel](../../../../glossary.md#kernel) contracts. Stop if implementation requires
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
  capability would otherwise be untruthful as soon as Planning grouped two adjacent ADD nodes.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [partition glossary](../../../../glossary.md#partition); [occurrence glossary](../../../../glossary.md#term-occurrence); [Planning glossary](../../../../glossary.md#term-planning); [nodes glossary](../../../../glossary.md#node))</span>
- Sequence one generated kernel per node instead of adding premature fusion. This establishes a
  complete correctness route and leaves later fusion as an alternative candidate, not a semantic
  requirement.<span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Use only scalar single-thread native segments in the first vertical slice. It exercises the
  canonical internal representation and exact generator/artifact/runtime path without combining
  operation expansion with carrier and mode expansion.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [slice glossary](../../../../glossary.md#slice); [canonical glossary](../../../../glossary.md#term-canonical-form); [artifact glossary](../../../../glossary.md#term-artifact); [runtime glossary](../../../../glossary.md#term-runtime); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Keep the flat execution package. The concrete family depends on many completed package-private
  roles, so moving it now would force public internal bridges without a second proven consumer.
- Restrict resolved layouts to canonical zero-offset non-view geometry. Unresolved ordinary
  elementwise outputs are admitted because the selected CPU representation is a canonical
  materialization, not an alias of unresolved physical geometry.<span data-glossary-navigation="true"> ([layouts glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [elementwise glossary](../../../../glossary.md#term-elementwise); [CPU glossary](../../../../glossary.md#term-cpu); [materialization glossary](../../../../glossary.md#materialization); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>

## Known limitations

- CPU 0005 supports only ADD and only one portable route. It is a correctness and lifecycle
  vertical slice, not pointwise-family closure or a performance milestone.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lifecycle glossary](../../../../glossary.md#lifecycle); [slice glossary](../../../../glossary.md#slice); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- No public Engine composition exists yet, so focused CPU tests prove the backend-private complete
  prepare/finalize/bind/execute path without claiming public end-to-end execution.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [prepare glossary](../../../../glossary.md#prepare))</span>
- Sequenced nodes materialize intermediates and dispatch separate generated kernels. Later exact
  fusion candidates may remove those boundaries without changing semantics.<span data-glossary-navigation="true"> ([nodes glossary](../../../../glossary.md#node); [dispatch glossary](../../../../glossary.md#term-dispatch); [kernels glossary](../../../../glossary.md#kernel); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Dynamic shapes, non-canonical layouts, heap/mixed carriers, Vector/parallel routes, BFLOAT16,
  BOOL, and FLOAT16 remain unsupported.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [canonical glossary](../../../../glossary.md#term-canonical-form); [layouts glossary](../../../../glossary.md#layout); [heap glossary](../../../../glossary.md#term-heap); [Vector glossary](../../../../glossary.md#term-vector); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype); [FLOAT16 glossary](../../../../glossary.md#term-float16))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

Implementation context `019fd19a-4262-7580-9674-226595356fbc` ran the sole final
`./gradlew :backends:cpu:test` after executable Java stabilized: 31 tests passed with no failures,
errors, or skips and [Gradle](../../../../glossary.md#term-gradle) reported `BUILD SUCCESSFUL`. Clean documentation context
`019fd1af-84ce-7aa1-b2d9-d07424ac37ec` (`/root/cpu_0005_docs`) independently inspected the final
implementation, tests, generated [Javadoc](../../../../glossary.md#term-javadoc),
architecture boundary, documentation profiles, and [planning](../../../../glossary.md#term-planning) contracts. It changed comments and
documentation only, so it reused the successful Java evidence rather than repeating the suite.

The documentation pass ran `./gradlew :backends:cpu:javadoc` successfully. Its targeted Markdown
checker validated five changed documents, 677 local links, 287 anchors, balanced fences, final
newlines, and trailing whitespace. Generated-page inspection found the exact ADD route, ordered
[node](../../../../glossary.md#node)-[kernel](../../../../glossary.md#kernel) recipe, sole [partition](../../../../glossary.md#partition) [guard](../../../../glossary.md#term-guard), and [fail-closed](../../../../glossary.md#term-fail-closed) descriptions. `javap -public` confirmed
that `CpuCapabilityProvider` remains the only changed public [CPU](../../../../glossary.md#term-cpu) surface and retains only its
existing public [constant](../../../../glossary.md#term-constant), constructor, `backendId()`, and `supports(...)` members. Source/import
inspection confirmed the exact [operation](../../../../glossary.md#operation)/type/[layout](../../../../glossary.md#layout) [matrix](../../../../glossary.md#term-matrix-meanings) and architecture-approved inward
dependencies. Exact inventory validation found 24 paths: 13 production, 6 tests, 2 explanatory
documents, and 3 planning documents, within the 26-path ceiling. `git diff --check` passed.

No architecture rule, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [module](../../../../glossary.md#term-module) boundary, [dependency](../../../../glossary.md#term-dependency), Gradle configuration, public package,
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) contract, integration contract, or other module changed, so architecture,
architecture-test, conformance, integration, Gradle, and other-module updates were correctly not
required. The task directory contains no detailed CPU 0005A specification; master-plan and roadmap
status checks retain CPU 0005A and later work as Draft.

## Implementation notes

- The capability provider now reports only exact parameterless ADD with two inputs and one output,
  equal fully static shapes, one common supported data type, and unresolved or canonical resolved
  layouts.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [data type glossary](../../../../glossary.md#data-type--datatype); [canonical glossary](../../../../glossary.md#term-canonical-form); [layouts glossary](../../../../glossary.md#layout))</span>
- The production candidate source constructs deterministic partition-wide buffer declarations and
  one node recipe per advertised occurrence, rejecting empty, mixed, unsupported, or aliased
  partitions before artifact work.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [buffer glossary](../../../../glossary.md#term-buffer); [node glossary](../../../../glossary.md#node); [occurrence glossary](../../../../glossary.md#term-occurrence); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Finalization resolves every shared assignment before loading artifacts in planned node order.
  Cold binding builds direct typed child calls, and the bound partition owns the sole state guard.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [guard glossary](../../../../glossary.md#term-guard))</span>
- The generated emitter is deliberately scalar, single-thread, and native-segment only. FLOAT32
  and FLOAT64 preserve JVM IEEE addition, while INT32 and INT64 preserve Java modular overflow.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [IEEE glossary](../../../../glossary.md#term-ieee); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- The clean documentation pass finalized affected Javadocs, package documentation, the CPU guide,
  glossary status, task evidence, master plan, and roadmap without widening capability claims.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [widening glossary](../../../../glossary.md#term-widening))</span>

## Completion summary

Completed changes: implemented and documented the exact dense ADD capability, ordered maximal-
[partition](../../../../glossary.md#partition) candidate/finalization/[binding](../../../../glossary.md#term-binding)/execution route, deterministic shared-resource reuse, and
four exact [scalar](../../../../glossary.md#term-scalar) native-segment [kernels](../../../../glossary.md#kernel) while preserving all exclusions.

Files changed or created: 13 [CPU](../../../../glossary.md#term-cpu) production files, 6 CPU tests, `docs/backend-guide/cpu-backend.md`,
`docs/glossary.md`, this task, `docs/planning/backends/cpu/master-plan.md`, and
`docs/planning/roadmap.md`; no file outside the task's permitted paths changed.

Tests and [validation](../../../../glossary.md#term-validation): sole final CPU [module](../../../../glossary.md#term-module) suite passed 31 tests; CPU [Javadoc](../../../../glossary.md#term-javadoc) built successfully;
generated docs, Markdown links/anchors/fences/newlines/whitespace, public [shape](../../../../glossary.md#shape), imports, exact
[scope](../../../../glossary.md#term-scope), statuses, Draft-task absence, and diff whitespace passed inspection.

Unresolved issues: none within CPU 0005. CPU 0005A and later [operation](../../../../glossary.md#operation), [carrier](../../../../glossary.md#term-storage-carrier), execution-mode,
[fusion](../../../../glossary.md#term-fusion), [compiler](../../../../glossary.md#term-compiler), public composition, [conformance](../../../../glossary.md#term-conformance), and integration work remain intentionally Draft.

Required follow-up: none for CPU 0005.

Status: Complete
