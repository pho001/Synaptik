# Task 0010H: Source-Only Published-[Constant](../../../../glossary.md#term-constant) [CPU](../../../../glossary.md#term-cpu) [Materialization](../../../../glossary.md#materialization)

## Reader context — historical integration boundary

A source-only published constant is a graph input with a known splat value and a requested
publication, but no producing or consuming operation. It is not a missing CPU kernel. In this
milestone CPU contributes byte geometry and an initializer recipe; Prepare assigns its buffer;
Runtime's creation phase invokes that recipe once for each fresh `RunState`. ‘Once’ is per run, not
per reusable preparation or per publication alias. The graph/descriptor supplies meaning, while each
created CPU buffer holds actual values and is cleaned up by run/result ownership. Host snapshot
export later reads that initialized publication; it does not initialize it or take over the buffer.

Current reading: [Supported lifecycle integration](../../../../backend-guide/cpu-backend.md#supported-lifecycle-integration); [Canonical host snapshot ownership](../../../../backend-guide/cpu-backend.md#canonical-caller-owned-host-snapshots).

## Status

Complete

## Goal

Complete the current [CPU](../../../../glossary.md#term-cpu) composition for fully static source-only published [compile](../../../../glossary.md#compile)-time splat
[constants](../../../../glossary.md#term-constant) that accompany the sole non-empty maximal CPU [partition](../../../../glossary.md#partition). CPU derives each constant's
physical byte size and [alignment](../../../../glossary.md#term-alignment) from its [canonical](../../../../glossary.md#term-canonical-form) final [descriptor](../../../../glossary.md#term-descriptor), contributes the exact
`ProducerlessPublishedConstantResource` values to shared [Prepare](../../../../glossary.md#prepare), and lets the existing CPU
assembler build one [immutable](../../../../glossary.md#term-immutable) initialized-[buffer](../../../../glossary.md#term-buffer) creator recipe for every assigned constant
representation.

The reusable [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) remains an immutable recipe and owns no physical, persistent, or
closeable resource. On every execution, existing [Runtime](../../../../glossary.md#term-runtime) `RunStateCreation` invokes each
initialized-buffer creator exactly once, producing and initializing one fresh [run](../../../../glossary.md#term-run)-owned CPU
representation before cold [binding](../../../../glossary.md#term-binding), reads, execution, or [publication](../../../../glossary.md#term-publication). In this task,
"materialized once" always means once per fresh [`RunState`](../../../../glossary.md#run-state--runstate), never once per reusable
`PreparedExecution`.

## Current defect and selected flow

[Compiler](../../../../glossary.md#term-compiler) 0006B5 closes the [canonical](../../../../glossary.md#term-canonical-form) logical [descriptor](../../../../glossary.md#term-descriptor) for the exact source-only [publication](../../../../glossary.md#term-publication)
role. [Prepare](../../../../glossary.md#prepare) 0005 accepts an exact [graph](../../../../glossary.md#graph)-value/logical-requirement pair plus [backend](../../../../glossary.md#backend)-supplied
physical geometry, appends its [buffer](../../../../glossary.md#term-buffer) after ordinary [partition](../../../../glossary.md#partition) declarations, and exposes the
assignment through [`PreparedScheduleContext`](../../../../glossary.md#prepared-schedule-context--preparedschedulecontext). The current [Engine](../../../../glossary.md#term-engine) [CPU](../../../../glossary.md#term-cpu) adapter still calls the
three-[argument](../../../../glossary.md#term-argument) `GraphPreparation.prepare(...)`, so no CPU physical contribution reaches Prepare
and preparation fails closed before the assembler can reuse its existing [constant](../../../../glossary.md#term-constant) branch.

The repaired flow is:

```text
Engine CPU composition
  -> CpuBackendIntegration.prepare(CompileArtifacts)
  -> CPU validates the sole non-empty maximal CPU partition
  -> CPU derives exact producerless published-constant resources in graph-value order
  -> GraphPreparation.prepare(artifacts, preparations, resources, assembler)
  -> Prepare appends physical buffers and assignments after ordinary declarations
  -> CpuPreparedScheduleAssembler creates one InitializedBuffer recipe per assignment
  -> PreparedExecution retains recipes only

each PreparedExecutionRunner.run(...)
  -> RunStateCreation invokes each recipe once
  -> fresh CPU buffer is allocated and splat-initialized
  -> initialized representation becomes valid
  -> cold binding, execution, publication, leases, and eventual run-owned cleanup
```

There is no second initialization within a [run](../../../../glossary.md#term-run), separate [materialization](../../../../glossary.md#materialization) schedule step,
synthetic executable, [node](../../../../glossary.md#node), or partition [owner](../../../../glossary.md#term-owner), persistent prepared buffer, or hidden global
[cache](../../../../glossary.md#term-cache). No [Runtime](../../../../glossary.md#term-runtime) task or Runtime source change is part of this work.

## [Scope](../../../../glossary.md#term-scope)

- Add exactly one supported [Engine](../../../../glossary.md#term-engine)-facing [operation](../../../../glossary.md#operation) to `CpuBackendIntegration`:

  ```java
  public PreparedExecution prepare(CompileArtifacts artifacts)
  ```

  It delegates to the owned CPU-private composition and returns the immutable reusable recipe.
  Existing <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`preparations(CompileArtifacts)` and `scheduleAssembler()` operations remain
  source-compatible; ordinary users still consume CPU through Engine.
- Add one matching [CPU](../../../../glossary.md#term-cpu)-private `CpuBackendComposition.prepare(CompileArtifacts)` composition
  entry. It must preserve the existing closed-owner and sole-partition validation precedence,
  obtain the current positional CPU preparation, derive the complete ordered producerless
  resource list, and call the four-argument <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [partition glossary](../../../../glossary.md#partition); [validation glossary](../../../../glossary.md#term-validation); [argument glossary](../../../../glossary.md#term-argument))</span>`GraphPreparation.prepare(...)` with the retained
  assembler.
- Change `CpuEngineBackendComposition.prepare(...)` to delegate to
  `CpuBackendIntegration.prepare(...)`. Engine must not inspect graph roles, logical memory,
  descriptors, scalar constants, byte geometry, or <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [constants glossary](../../../../glossary.md#term-constant))</span>`ProducerlessPublishedConstantResource`.
- Derive contributions only for graph values having the exact complete role already established
  by Compiler, Planning, and Prepare:<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare))</span>

  1. the value is a final [graph](../../../../glossary.md#graph) input;
  2. it has an exact `CompileConstantPlan.ConstantSource` splat classification;
  3. it is a requested graph output/[publication](../../../../glossary.md#term-publication);
  4. no [node](../../../../glossary.md#node) consumes it and no node produces it;
  5. its exact `LogicalMemoryRequirement` is producerless, consumerless, and graph-output
     required;
  6. its [Shape](../../../../glossary.md#shape) is fully static; and
  7. its resolved [layout](../../../../glossary.md#layout) equals `LayoutDescriptor.contiguous(shape)`.

  Iterate final `CompiledGraphModel.values()` order. Retain the exact graph-value and logical-
  requirement references in each contribution. Reject missing, duplicate, contradictory, or
  non-canonical facts rather than guessing or repairing them. Consumed or unpublished constants
  remain ordinary partition-connected values and receive no producerless contribution.
- Preserve the current executable domain. Artifacts must still contain exactly one non-empty
  maximal partition owned by <span data-glossary-navigation="true"> ([Artifacts glossary](../../../../glossary.md#term-artifact))</span>`CpuCapabilityProvider.CPU_BACKEND_ID`. Zero-node/zero-partition and
  every mixed- or multi-partition composition remain rejected before resource derivation,
  backend preparation, or schedule assembly.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- Derive physical geometry from the [canonical](../../../../glossary.md#term-canonical-form) resolved [descriptor](../../../../glossary.md#term-descriptor):

  - `elementSpan = descriptor.layout().orElseThrow().referencedElementSpan()`;
  - `byteSize = Math.multiplyExact(elementSpan, descriptor.dataType().byteWidth())`; and
  - `byteAlignment = descriptor.dataType().byteWidth()`.

  Canonical contiguous geometry has zero offset, is not a view, and has the exact row-major
  strides for the Shape. Do not derive size from a node, kernel, carrier instance, publication,
  or Java array ceiling. Preserve checked <span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [view glossary](../../../../glossary.md#term-view-meanings); [row-major glossary](../../../../glossary.md#term-row-major); [strides glossary](../../../../glossary.md#term-stride); [kernel glossary](../../../../glossary.md#kernel); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`long` [overflow](../../../../glossary.md#term-overflow).
- Support all six current [`DataType`](../../../../glossary.md#data-type--datatype) values through the existing exhaustive CPU initializer:
  [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), [`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`INT64`](../../../../glossary.md#data-type--datatype), [`INT32`](../../../../glossary.md#data-type--datatype), and [`BOOL`](../../../../glossary.md#data-type--datatype). Require the exact scalar type to
  equal the descriptor type. Do not add conversion, widening, rounding, BOOL normalization, or a
  default enum branch.<span data-glossary-navigation="true"> ([widening glossary](../../../../glossary.md#term-widening); [rounding glossary](../../../../glossary.md#term-rounding); [normalization glossary](../../../../glossary.md#term-normalization))</span>
- A rank-zero descriptor has canonical span one, so its buffer size is one element width and its
  scalar is written once per run. A fully static Shape with any zero extent has canonical span and
  byte size zero; its fresh zero-byte representation is still created once per run, initialization
  performs zero stores, and the representation becomes initially valid. Preserve the Model's
  zero-before-product behavior and reject dynamic, unresolved, non-canonical, negative, or
  overflowing geometry.<span data-glossary-navigation="true"> ([rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [buffer glossary](../../../../glossary.md#term-buffer); [element glossary](../../../../glossary.md#term-element); [run glossary](../../../../glossary.md#term-run); [extent glossary](../../../../glossary.md#term-extent); [Model glossary](../../../../glossary.md#term-model))</span>
- Reuse `CpuPreparedScheduleAssembler` without adding an executable materialization occurrence.
  Prepare's appended assignment enters the assembler's existing constant-source branch, which
  emits exactly one <span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`PreparedRepresentationPlan.InitializedBuffer` at representation index zero.
  The recipe delegates to `CpuRepresentationRecipes.initializedBuffer(...)`; assembly itself must
  allocate, initialize, bind, execute, or publish nothing.
- Preserve existing schedule order: one representation-creation prefix, the sole partition
  execution occurrence, then forward publications followed by gradient publications. A
  source-only constant is not an executable selection and gains no fake read or write. Its
  initial validity comes solely from existing <span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>`RunStateCreation` behavior before all later cold
  binding and action traversal.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Add focused CPU and Engine tests, update affected Javadocs and explanatory API/CPU documentation,
  and complete the mandatory separate clean documentation-focused pass after executable Java
  stabilizes.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

### Deterministic [validation](../../../../glossary.md#term-validation) and failure order

The new [CPU](../../../../glossary.md#term-cpu) composition entry must preserve this order for deterministic, non-racy checks:

1. reject a closed CPU integration through the existing [owner](../../../../glossary.md#term-owner) [guard](../../../../glossary.md#term-guard);
2. reject null [`artifacts`](../../../../glossary.md#term-artifact);
3. reject anything other than the existing sole non-empty CPU [partition](../../../../glossary.md#partition) domain;
4. construct the existing positional preparation, retaining its current CPU boundary-analysis
   validation order;
5. scan final graph values in encounter order and correlate exact constant-source, graph-input,
   graph-output/publication, node-use, and logical-requirement facts;<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [constant glossary](../../../../glossary.md#term-constant); [publication glossary](../../../../glossary.md#term-publication); [node glossary](../../../../glossary.md#node))</span>
6. for each required value, reject a dynamic [Shape](../../../../glossary.md#shape), unresolved [layout](../../../../glossary.md#layout), or layout unequal to
   `LayoutDescriptor.contiguous(shape)`, in that order;
7. reject scalar/descriptor type disagreement, then derive checked byte size and element-width
   alignment; and<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [descriptor glossary](../../../../glossary.md#term-descriptor); [element glossary](../../../../glossary.md#term-element); [alignment glossary](../../../../glossary.md#term-alignment))</span>
8. invoke shared [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation), whose existing top-level, complete-coverage, assignment,
   finalization, assembler, and schedule validation order remains authoritative.

Null failures remain `NullPointerException`; unsupported role, domain, type, or geometry failures
are `IllegalArgumentException`; checked Shape/layout/byte arithmetic retains
`ArithmeticException`; closed [ownership](../../../../glossary.md#term-ownership-meanings) remains `IllegalStateException`. Add stable, value-
identifying messages for new deliberate CPU failures, but do not translate established lower-
layer failures. If a [race](../../../../glossary.md#term-race) with adapter closure begins after the open check, existing [lifecycle](../../../../glossary.md#lifecycle)
coordination rules remain authoritative.

### [Ownership](../../../../glossary.md#term-ownership-meanings), validity, cleanup, and [concurrency](../../../../glossary.md#term-concurrency)

- [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), [`PreparedRepresentationPlan`](../../../../glossary.md#prepared-representation-plan--preparedrepresentationplan), and every buffer creator are immutable
  reusable recipes. They retain scalar and geometry facts but no <span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [immutable glossary](../../../../glossary.md#term-immutable); [scalar glossary](../../../../glossary.md#term-scalar))</span>[`Arena`](../../../../glossary.md#term-arena), [`MemorySegment`](../../../../glossary.md#term-memorysegment), CPU
  buffer, lease, validity bit, or close action.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lease glossary](../../../../glossary.md#term-lease))</span>
- Each `PreparedExecutionRunner.run(...)` creates one fresh [`RunState`](../../../../glossary.md#run-state--runstate). `RunStateCreation` walks
  dense plan order and invokes the one creator for each non-caller representation exactly once.
  `CpuRepresentationRecipes.initializedBuffer(...)` allocates one fresh aligned
  `CpuNativeBuffer`, fills every complete [element](../../../../glossary.md#term-element), and returns it.
- Initialization completes inside creator invocation. After all representations and workspaces
  are created, `RunStateCreation` constructs the state and marks every initialized-buffer
  position valid. The runner then cold-binds every execution, transfer, and publication
  occurrence before executing any action. No read or publication can observe an uninitialized or
  invalid source-only constant.<span data-glossary-navigation="true"> ([publication occurrence glossary](../../../../glossary.md#publication-occurrence); [publication glossary](../../../../glossary.md#term-publication); [constant glossary](../../../../glossary.md#term-constant))</span>
- If allocation or initialization fails, the initializer closes its newly allocated buffer and
  preserves a distinct cleanup failure as suppressed. `RunStateCreation` then releases all
  earlier successfully created run resources in reverse creation order and returns no state. If
  later binding, execution, or publication fails, the runner closes the created state once and
  preserves existing suppression behavior.<span data-glossary-navigation="true"> ([releases glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [run glossary](../../../../glossary.md#term-run); [binding glossary](../../../../glossary.md#term-binding))</span>
- Successful results lease the exact run-owned state. Repeated publication aliases or duplicate
  requested result occurrences refer to the assigned buffer/representation without copying or
  reinitializing it. Closing the final result ownership releases it through existing Runtime
  behavior. Host export reads the exact valid leased representation and returns detached bytes;
  it neither initializes nor changes ownership or validity.<span data-glossary-navigation="true"> ([aliases glossary](../../../../glossary.md#aliasing-and-arena); [Runtime glossary](../../../../glossary.md#term-runtime); [Host glossary](../../../../glossary.md#term-host); [detached glossary](../../../../glossary.md#detached-model-results))</span>
- Two executions of the same prepared recipe, including concurrent executions, receive distinct
  initialized CPU buffers and independent validity/cleanup state. Within one run there is exactly
  one initialization per planned representation, irrespective of the number of publications,
  aliases, leases, or host-export calls. No shared mutable initializer state or cache is added.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [cache glossary](../../../../glossary.md#term-cache))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Making a source-only [constant](../../../../glossary.md#term-constant) the only [graph value](../../../../glossary.md#graph-value) in a runnable zero-[node](../../../../glossary.md#node) schedule
- Mixed-[backend](../../../../glossary.md#backend) or multi-[partition](../../../../glossary.md#partition) schedule composition
- A Runtime task or any Runtime production, test, API, schedule, validity, lease, alias,
  publication, or cleanup modification<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [lease glossary](../../../../glossary.md#term-lease); [alias glossary](../../../../glossary.md#aliasing-and-arena); [publication glossary](../../../../glossary.md#term-publication))</span>
- A separate materialization step, transfer, executable, node, partition, finalizer assignment,
  owner, representation index, persistent prepared allocation, pool, memoization, or global cache<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [owner glossary](../../../../glossary.md#term-owner); [memoization glossary](../../../../glossary.md#term-memoization); [cache glossary](../../../../glossary.md#term-cache))</span>
- Compiler, Planning, Prepare, Model, Config, Training, NN, Trace, provider, OpenBLAS, tuning,
  generated-code, kernel, or hot-loop changes<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Model glossary](../../../../glossary.md#term-model); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Trace glossary](../../../../glossary.md#trace); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [kernel glossary](../../../../glossary.md#kernel))</span>
- Dynamic Shape binding, unresolved or non-canonical layout repair, general views, broadcast
  storage, scalar conversion, dense payload constants, or non-splat constants<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [binding glossary](../../../../glossary.md#term-binding); [canonical glossary](../../../../glossary.md#term-canonical-form); [layout glossary](../../../../glossary.md#layout); [views glossary](../../../../glossary.md#term-view-meanings); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Removal or [signature](../../../../glossary.md#term-signature) changes of the existing supported [CPU](../../../../glossary.md#term-cpu) preparation/assembler [SPI](../../../../glossary.md#term-service-provider-interface-spi)
- [Engine](../../../../glossary.md#term-engine) 0006 implementation, backward [API](../../../../glossary.md#term-application-programming-interface-api) work, or its final [scalar](../../../../glossary.md#term-scalar)-[objective](../../../../glossary.md#term-objective) [fixture](../../../../glossary.md#term-fixture)
- Architecture-contract, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [dependency](../../../../glossary.md#term-dependency), build, backend-[conformance](../../../../glossary.md#term-conformance), or benchmark changes

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Backend glossary](../../../../glossary.md#backend))</span>
- [ADR 0010: staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0011: per-run Runtime resource ownership](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership))</span>
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span> and [roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- [CPU 0010F lifecycle integration](0010f-supported-cpu-lifecycle-integration-adapter.md)
- [CPU 0010G host snapshot export](0010g-canonical-caller-owned-host-snapshot-export.md)<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- [Compiler 0006B5 descriptor closure](../../../modules/compiler/tasks/0006b5-published-compile-time-constant-descriptor-closure.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- [Prepare 0005 resource handoff](../../../modules/prepare/tasks/0005-producerless-published-constant-resource-handoff-and-shared-slot-assignment.md)<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- [Engine 0006 scalar-objective backward convenience](../../../modules/engine/tasks/0006-one-shot-scalar-objective-backward-convenience.md)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [scalar glossary](../../../../glossary.md#term-scalar); [objective glossary](../../../../glossary.md#term-objective))</span>
- [Runtime](../../../../glossary.md#term-runtime) [0007 representation plan](../../../modules/runtime/tasks/0007-representation-creation-and-residency-foundation.md),
  [0009 publication](../../../modules/runtime/tasks/0009-publication-and-result-schedule-steps.md)<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>,
  [0010 runner](../../../modules/runtime/tasks/0010-prepared-runner-and-dynamic-execution.md), and
  [0015 leased access](../../../modules/runtime/tasks/0015-leased-publication-representation-access.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)

## Architecture constraints

- Compiler owns final graph identity, source classification, publication order, and canonical
  logical descriptors. CPU consumes those facts without rewriting them.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [graph glossary](../../../../glossary.md#graph); [identity glossary](../../../../glossary.md#term-identity-meanings); [publication glossary](../../../../glossary.md#term-publication); [canonical glossary](../../../../glossary.md#term-canonical-form); [descriptors glossary](../../../../glossary.md#term-descriptor); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Planning owns producer/consumer/output obligations. CPU retains the exact logical requirement
  and invents no partition relationship.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [producer glossary](../../../../glossary.md#term-producer); [partition glossary](../../../../glossary.md#partition))</span>
- Concrete CPU owns byte size, alignment, representation type, and initialization recipe.
  Prepare validates/transports declarations and assigns slots; Runtime invokes creators and owns
  mutable per-run state.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [mutable glossary](../../../../glossary.md#term-mutable); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- Engine is the explicit composition root but must not derive CPU facts. It delegates preparation
  through supported CPU SPI and exposes no CPU-private or Prepare contribution type in ordinary
  user APIs.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [composition root glossary](../../../../glossary.md#term-composition-root); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi))</span>
- CPU may use existing Compiler, Model, Planning, Prepare, and Runtime contracts through its
  already declared execution-side dependencies. CPU must not depend on Engine.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>
- The sole non-empty CPU partition contract and existing complete schedule assembler remain
  unchanged. Stop if implementation requires a second assembler, a general registry/facade, a
  synthetic owner, or a dependency/architecture change.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [facade glossary](../../../../glossary.md#term-facade); [owner glossary](../../../../glossary.md#term-owner); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Package impact

Changed supported package:

- `io.github.pho001.synaptik.backend.cpu` — add only the supported
  `prepare(CompileArtifacts)` integration [operation](../../../../glossary.md#operation); expose no internal [CPU](../../../../glossary.md#term-cpu) type or
  `ProducerlessPublishedConstantResource`.

Changed CPU-private package:

- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas` — add the exact
  composition entry and focused private resource derivation.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — executable behavior remains unchanged;
  finalize assembler Javadoc so source-only assignments and recipe-only ownership are explicit.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

Changed [Engine](../../../../glossary.md#term-engine)-private package:

- `io.github.pho001.synaptik.engine` — the existing CPU adapter delegates preparation to the new
  supported CPU operation. No Engine public/protected API changes.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

No package or [module](../../../../glossary.md#term-module) is added.

## Affected files

Expected production/[Javadoc](../../../../glossary.md#term-javadoc) files:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuBackendIntegration.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuBackendComposition.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPreparedScheduleAssembler.java`
  (Javadoc clarification only unless implementation proves a narrowly required correction)
- `modules/engine/src/main/java/io/github/pho001/synaptik/engine/CpuEngineBackendComposition.java`

Expected test files:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/spi/CpuBackendIntegrationAndCpuPreparedScheduleAssemblerPublicTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/CpuBackendCompositionTest.java`
- `modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineStandardCompositionTest.java`

Expected explanatory documentation:

- `docs/api/public-api.md`
- `docs/backend-guide/cpu-backend.md`

[Planning](../../../../glossary.md#term-planning)/status files:

- `docs/planning/backends/cpu/tasks/0010h-source-only-published-constant-cpu-materialization.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`
- directly stale prerequisite/status sentences in [Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare), and [Engine](../../../../glossary.md#term-engine) master plans only

No other path is authorized by default.

## Maximum [scope](../../../../glossary.md#term-scope)

This cross-boundary [lifecycle](../../../../glossary.md#lifecycle) completion may modify at most fifteen paths: four production/[Javadoc](../../../../glossary.md#term-javadoc)
paths, three existing test paths, two explanatory documents, and six [planning](../../../../glossary.md#term-planning)/status paths. It
adds no production or test file. If executable behavior requires another source/test path, a
[Runtime](../../../../glossary.md#term-runtime) change, a build/[dependency](../../../../glossary.md#term-dependency) edit, or an architecture document, stop and revise the plan
before implementation. An unchanged assembler or documentation path is unused capacity, not
permission for unrelated work.

## Acceptance criteria

- `CpuBackendIntegration` remains public, final, privately constructed, and `AutoCloseable`. Its
  public declarations are exactly its existing eight operations plus
  `prepare(CompileArtifacts)`. No supported [signature](../../../../glossary.md#term-signature) exposes `.internal`, Engine, a physical
  carrier, or <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>`ProducerlessPublishedConstantResource`.
- The new integration operation rejects closure before null and delegates all composition work to
  its exact owned <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>`CpuBackendComposition`.
- CPU accepts exactly one non-empty CPU partition and rejects zero-node, zero-partition,
  mixed-owner, and multi-partition artifacts before derivation or backend preparation. Tests lock
  these fail-closed boundaries.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [partition glossary](../../../../glossary.md#partition); [node glossary](../../../../glossary.md#node); [owner glossary](../../../../glossary.md#term-owner); [artifacts glossary](../../../../glossary.md#term-artifact); [backend glossary](../../../../glossary.md#backend); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Required contributions are complete, unique, and in final graph-value order. Each retains the
  exact artifact <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>`GraphValue` and `LogicalMemoryRequirement`; consumed, produced, unpublished,
  bindable, dynamic, unresolved, or non-canonical values do not silently enter the list.
- [Canonical](../../../../glossary.md#term-canonical-form) span times [element](../../../../glossary.md#term-element) width, with checked `long` arithmetic, is the exact physical byte
  size. Element width is the exact power-of-two alignment. Rank-zero is one element; any zero
  extent is zero bytes even if other unused extents would otherwise overflow.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [Rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [extent glossary](../../../../glossary.md#term-extent); [overflow glossary](../../../../glossary.md#term-overflow))</span>
- All six current data types preserve the exact scalar representation through the existing CPU
  initializer, including raw floating/BFLOAT16 values, signed integers, and canonical BOOL.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Shared Prepare appends each source-only assignment after all ordinary buffers without changing
  their slots, plan indices, finalizer assignments, workspaces, or partition preparation.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [buffers glossary](../../../../glossary.md#term-buffer))</span>
- The CPU assembler emits one representation-creation prefix, one existing execution occurrence,
  and the existing ordered publication suffix. Each source-only assigned constant has exactly one<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [publication glossary](../../../../glossary.md#term-publication); [constant glossary](../../../../glossary.md#term-constant))</span>
  `InitializedBuffer` creator at representation index zero and no executable selection,
  transfer, or extra schedule step.
- Assembly/preparation performs no allocation or initialization. Every run invokes each
  initialized creator once, produces a distinct run-owned representation, initializes it once,
  and marks it valid before binding or action traversal. Repeated publications, aliases, leases,
  and host exports do not trigger initialization.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [binding glossary](../../../../glossary.md#term-binding); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [leases glossary](../../../../glossary.md#term-lease); [host glossary](../../../../glossary.md#term-host))</span>
- Two sequential and two concurrent executions of one [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) prove distinct
  run-owned representations and identical initialized content. Closing either result cannot
  affect the other run or the reusable recipe.
- Zero-byte initialized representations publish successfully without a store. Scalar and all-six-
  type cases export the expected canonical host bytes while the result lease is open.
- Allocation/initialization failure cleanup and later runner-failure cleanup remain the existing
  Runtime/CPU behavior; focused source inspection and existing Runtime tests are cited rather than
  modified or redundantly recreated.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime))</span>
- `CpuEngineBackendComposition` delegates to `integration.prepare(artifacts)` and contains no
  producerless-role or geometry derivation. Engine 0006 remains Draft and unimplemented.
- No new ordinary Engine API, CPU facade/type, Runtime mechanism, inventory entry, backend-
  conformance contract, integration-test fixture, architecture rule, ADR, module edge, benchmark,
  generated code, or provider/tuning behavior is added.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [facade glossary](../../../../glossary.md#term-facade); [conformance glossary](../../../../glossary.md#term-conformance); [test fixture glossary](../../../../glossary.md#adr-ci-and-qualification); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>
- A distinct clean documentation-focused pass finalizes all changed Javadocs, public API and CPU
  guide text, link/term impact, no-change decisions, and planning evidence before implementation
  may be marked Complete.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

The implementation context runs focused tests while developing, then one final affected-[module](../../../../glossary.md#term-module)
validation after executable Java stabilizes:

```bash
./gradlew :backends:cpu:test :modules:engine:test
```

Focused tests must cover supported public shape; direct integration preparation; exact source-only
eligibility and [graph](../../../../glossary.md#graph)-value order; exact reference [handoff](../../../../glossary.md#term-handoff); sole-[partition](../../../../glossary.md#partition) rejection; [canonical](../../../../glossary.md#term-canonical-form),
dynamic, unresolved, non-canonical, [scalar](../../../../glossary.md#term-scalar), zero-[element](../../../../glossary.md#term-element), and [overflow](../../../../glossary.md#term-overflow) geometry; all six data
types; prepared recipe shape; initial validity; sequential and concurrent per-[run](../../../../glossary.md#term-run) representation
[identity](../../../../glossary.md#term-identity-meanings)/content/cleanup; [publication](../../../../glossary.md#term-publication) [aliases](../../../../glossary.md#aliasing-and-arena); and [Engine](../../../../glossary.md#term-engine)'s [CPU](../../../../glossary.md#term-cpu) delegation.

The clean documentation-focused context reuses successful Java evidence unless it changes
executable Java, then runs:

```bash
./gradlew :backends:cpu:javadoc :modules:engine:javadoc
git diff --check
```

It also records:

- `javap -public` or the existing reflection assertion for the exact nine-operation supported CPU
  surface and absence of <span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>`.internal`, Engine, [carrier](../../../../glossary.md#term-storage-carrier), and producerless contribution types;
- [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) of the distinct-package CPU [SPI](../../../../glossary.md#term-service-provider-interface-spi) test against declared dependencies;
- Markdown local-link/anchor, heading-order, balanced-fence, trailing-whitespace, and final-newline
  checks for changed documentation;
- the exact at-most-fifteen-path allowlist and preservation of unrelated dirty-worktree changes;
- source/import scans proving CPU has no Engine dependency, Engine derives no CPU facts, Runtime is
  unchanged, and no synthetic schedule occurrence/cache was introduced; and<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [cache glossary](../../../../glossary.md#term-cache))</span>
- synchronized task/master/roadmap statuses with CPU 0010H Complete only after implementation and
  documentation validation, while Engine 0006 remains Draft.

No benchmark is required: initialization is cold once-per-run setup, not a per-element executable
hot loop. [Backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) is unchanged because no shared backend contract or behavior is
added. `testing/integration-tests` is unchanged because the focused CPU SPI run crosses actual
[Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) behavior and the Engine-private delegation test locks the only Engine
wiring change; Engine 0006 owns the later ordinary end-to-end scalar-[objective](../../../../glossary.md#term-objective) [fixture](../../../../glossary.md#term-fixture). CPU source
inventory is unchanged because no production file is added. Architecture tests and repository-
wide validation are not required because no [dependency](../../../../glossary.md#term-dependency), module boundary, shared build, or
architecture rule changes; [CI](../../../../glossary.md#term-continuous-integration-ci) remains the broader independent gate.

## Dependencies

- [Compiler 0006B5](../../../modules/compiler/tasks/0006b5-published-compile-time-constant-descriptor-closure.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span> —
  Complete; supplies exact canonical logical descriptor closure for the eligible role.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- [Prepare 0005](../../../modules/prepare/tasks/0005-producerless-published-constant-resource-handoff-and-shared-slot-assignment.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span> —
  Complete; supplies exact resource handoff, validation, deterministic append-only assignment,
  and the four-argument orchestration entry.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff); [validation glossary](../../../../glossary.md#term-validation); [argument glossary](../../../../glossary.md#term-argument))</span>
- [CPU 0010F](0010f-supported-cpu-lifecycle-integration-adapter.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> — Complete; supplies the
  supported owner, sole-partition preparation, representation recipes, and schedule assembler.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [partition glossary](../../../../glossary.md#partition))</span>
- [CPU 0010G](0010g-canonical-caller-owned-host-snapshot-export.md) — Complete; supplies detached
  canonical host export for validation and later Engine consumption.<span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results); [host glossary](../../../../glossary.md#term-host); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Existing [Runtime](../../../../glossary.md#term-runtime) 0007/0009/0010/0015 contracts — Complete and sufficient without modification.

All prerequisites are complete and the task is actionable.

## Follow-up tasks

- [Engine 0006](../../../modules/engine/tasks/0006-one-shot-scalar-objective-backward-convenience.md)<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
  remains Draft and is the next downstream task after CPU 0010H implementation, documentation,
  and validation are Complete.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [validation glossary](../../../../glossary.md#term-validation))</span>
- Zero-node/pass-through and mixed-backend composition remain future architecture/planning work
  only when a concrete consumer justifies them.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [backend composition glossary](../../../../glossary.md#backend-composition); [planning glossary](../../../../glossary.md#term-planning))</span>

## Documentation-focused clean-context pass

After executable Java and focused tests stabilize, hand the exact diff, baseline dirty status,
test evidence, source-only eligibility/geometry decisions, public-shape result, and [lifecycle](../../../../glossary.md#lifecycle)
[ownership](../../../../glossary.md#term-ownership-meanings) analysis to a distinct clean documentation-focused context. It must read the General,
[API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), and [Planning](../../../../glossary.md#term-planning) profiles and finalize affected Javadocs, `docs/api/public-api.md`, the
[CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) guide, and planning evidence.

The pass must explicitly state that [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) owns recipes only and that [materialization](../../../../glossary.md#materialization)
occurs once per fresh [`RunState`](../../../../glossary.md#run-state--runstate). Review [Runtime](../../../../glossary.md#term-runtime) API/lifecycle documentation, glossary, [Compile](../../../../glossary.md#compile)
API, Training API, architecture pages/tests, backend [conformance](../../../../glossary.md#term-conformance), integration tests, CPU inventory,
and build files. Change them only if a concrete stale statement or executable impact exists;
otherwise record a reasoned no-change conclusion. Do not repeat successful Java tests unless the
pass changes executable Java or identifies a concrete risk.

## Architecture impact

Expected impact: None.

This task fills the concrete [CPU](../../../../glossary.md#term-cpu) side of existing [Compiler](../../../../glossary.md#term-compiler) -> [Planning](../../../../glossary.md#term-planning) -> [Prepare](../../../../glossary.md#prepare) -> [Runtime](../../../../glossary.md#term-runtime)
contracts. It changes neither [ownership](../../../../glossary.md#term-ownership-meanings) nor [dependency direction](../../../../glossary.md#java-platform-and-build-tools). The new supported CPU [operation](../../../../glossary.md#operation)
keeps physical fact derivation in CPU and lets [Engine](../../../../glossary.md#term-engine) remain composition-only. If implementation
requires persistent prepared storage, another Runtime concept, a synthetic [graph](../../../../glossary.md#graph)/schedule [owner](../../../../glossary.md#term-owner),
a build edge, or an architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) edit, stop and report the conflict rather than [widening](../../../../glossary.md#term-widening).

## Implementation prompt

```text
Work in /Users/phujka/IdeaProjects/Synaptik on CPU task 0010H. Do not use GSD, commit, or push.
Use a separate clean implementation context. Read AGENTS.md, ARCHITECTURE.md, the planning guide,
documentation rules, CPU master plan, and this task in full. Read every directly referenced
Compiler 0006B5, Prepare 0005, CPU 0010F/0010G, Engine 0006, and Runtime contract and inspect the
listed source/tests before editing.

Implement exactly task 0010H within its fifteen-path ceiling. Add only the supported
CpuBackendIntegration.prepare(CompileArtifacts) operation, CPU-private exact resource derivation
and four-argument GraphPreparation composition, Engine-private delegation, focused tests, and
authorized documentation. Preserve the sole non-empty CPU partition boundary. PreparedExecution
must retain only immutable initialized-buffer creator recipes; every run must allocate and
initialize its own representation exactly once through existing RunStateCreation. Add no Runtime
task/change, materialization schedule step, synthetic executable/node/partition, persistent
prepared buffer, or cache.

After executable Java and focused tests stabilize, use the mandatory distinct clean
documentation-focused context described by the task. Do not mark implementation Complete until
all acceptance, documentation, status, and validation gates pass. Stop and report any required
architecture, dependency, Runtime, or out-of-scope change.
```

## Stop conditions

Stop and report instead of implementing or [widening](../../../../glossary.md#term-widening) if:

- exact eligibility cannot be derived from existing immutable Compile artifacts and logical
  requirements;<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [Compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>
- accepted input from [Compiler](../../../../glossary.md#term-compiler) 0006B5 is not [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous) or requires [CPU](../../../../glossary.md#term-cpu) to repair it;
- a source-only value must be assigned to a [partition](../../../../glossary.md#partition)/finalizer or selected by an executable;
- preparation or assembly must allocate/initialize a physical resource;
- correctness needs initialization more than once per representation per [`RunState`](../../../../glossary.md#run-state--runstate);
- success requires a zero-node schedule, mixed/multi-partition composition, transfer, Runtime
  change/task, persistent resource/cache, new dependency, architecture/ADR edit, benchmark, new
  production/test file, or a sixteenth path; or<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [Runtime glossary](../../../../glossary.md#term-runtime); [cache glossary](../../../../glossary.md#term-cache); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Engine](../../../../glossary.md#term-engine) must inspect or construct `ProducerlessPublishedConstantResource` or any CPU fact.

## Local decisions

- One supported [CPU](../../../../glossary.md#term-cpu) [`prepare`](../../../../glossary.md#prepare) operation is narrower than exposing resource contributions to
  Engine. Existing preparation and assembler operations remain for compatibility, but the built-in
  Engine path uses the complete CPU-owned composition entry.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Final graph-value order is the deterministic resource order because Compile and Prepare already
  use that order; numeric IDs, constant-list order, map order, and publication duplicates are not
  new ordering policies.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [Compile glossary](../../../../glossary.md#compile); [constant glossary](../../../../glossary.md#term-constant); [publication glossary](../../../../glossary.md#term-publication))</span>
- Canonical referenced span is the physical element capacity for this exact non-view source. CPU's
  current native carrier uses element width for alignment, including one-byte BOOL.<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [referenced span glossary](../../../../glossary.md#referenced-element-span); [element glossary](../../../../glossary.md#term-element); [view glossary](../../../../glossary.md#term-view-meanings); [carrier glossary](../../../../glossary.md#term-storage-carrier); [alignment glossary](../../../../glossary.md#term-alignment); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
- Prepare's append-only assignment and the assembler's existing constant branch already separate
  physical declaration, immutable initializer recipe, and per-run representation ownership. No
  second materialization mechanism is justified.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [materialization glossary](../../../../glossary.md#materialization))</span>
- Runtime already proves creator invocation, initial validity, cold binding, result leases, and
  cleanup. CPU tests exercise the composed behavior; Runtime tests and source are not changed.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [binding glossary](../../../../glossary.md#term-binding); [leases glossary](../../../../glossary.md#term-lease))</span>

## Known limitations

This task supports only fully static [canonical](../../../../glossary.md#term-canonical-form) splat [constants](../../../../glossary.md#term-constant) accompanying current non-empty
[CPU](../../../../glossary.md#term-cpu) work. It does not make pure constant [graphs](../../../../glossary.md#graph) runnable, support dynamic or [view](../../../../glossary.md#term-view-meanings) geometry,
introduce dense non-splat payloads, or compose multiple [backend](../../../../glossary.md#backend) schedules. [Engine](../../../../glossary.md#term-engine) 0006 remains
Draft until this implementation is complete.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context `01a0a5c5-c93a-7010-9067-2ae5582e2786` created this detailed [Ready](../../../../glossary.md#term-ready) specification
after full architecture/planning/documentation reading and focused inspection of the completed
[Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare), [CPU](../../../../glossary.md#term-cpu), [Engine](../../../../glossary.md#term-engine), and [Runtime](../../../../glossary.md#term-runtime) contracts. It performed no Java change or Java test.

Planning validation:

- Markdown heading, fence, link-[target](../../../../glossary.md#term-target), trailing-whitespace, and final-newline checks;
- task ordering and Ready/[dependency](../../../../glossary.md#term-dependency) [synchronization](../../../../glossary.md#term-synchronization) across directly affected planning indexes;
- exact intended-path inspection against the pre-existing dirty worktree; and
- `git diff --check`.

Executable implementation context `01a0b0fe-cbc1-7171-a219-b02db3c6954d` completed the bounded
Java and test change at repository `HEAD` `d446e3ccfb9ee15a7710ac620e5bb591f7c770d4`.

Executable validation reused by the documentation pass:

- `./gradlew :backends:cpu:test :modules:engine:test` passed in 2 minutes 41 seconds: CPU reported
  198 suites and 1,012 tests, Engine reported 5 suites and 36 tests, and the combined result was
  203 suites and 1,048 tests with zero failures, zero errors, and 28 existing CPU skips.
- `javap -public` confirmed that `CpuBackendIntegration` has exactly nine public operations,
  including `prepare(CompileArtifacts)`, with no `.internal`, Engine, physical-carrier, or
  producerless-resource type in its supported signatures.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [signatures glossary](../../../../glossary.md#term-signature))</span>
- The implementation context's exact seven-path [scope](../../../../glossary.md#term-scope), prohibited-area scans, and
  `git diff --check` passed after executable Java stabilized.
- One early [compile](../../../../glossary.md#compile) attempt failed only because four new `java.util` imports were accidentally
  placed after the class declaration. Moving those imports to the import section corrected the
  source; subsequent compilation and the final tests passed. This was an import-placement error,
  not a design or behavior defect.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>

Clean documentation-focused context `/root` independently inspected the implementation and tests,
then finalized the [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [Backend](../../../../glossary.md#backend) Guide, Planning, and Example-profile concerns. It changed
Javadoc only in three production files, changed no executable Java [token](../../../../glossary.md#term-token), and did not edit the
three test files. It finalized `docs/api/public-api.md`, `docs/backend-guide/cpu-backend.md`, and
the six planning/status files. The pass made these boundaries explicit:

- [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) owns [immutable](../../../../glossary.md#term-immutable) recipes only; each fresh [`RunState`](../../../../glossary.md#run-state--runstate) invokes each initialized
  recipe once and owns a distinct physical representation;
- source-only published constants gain no executable schedule step, and pure zero-node constant
  graphs remain unsupported;<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [node glossary](../../../../glossary.md#node); [graphs glossary](../../../../glossary.md#graph))</span>
- Engine delegates complete preparation without interpreting CPU roles or geometry; and
- CPU 0010H is Complete while Engine 0006 remains Draft and is the next task to reassess and plan.

Documentation validation:

- `./gradlew :backends:cpu:javadoc :modules:engine:javadoc` passed. The output contained the two
  existing incubating-module warnings and 92 pre-existing missing-<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>`@param` warnings in unrelated
  CPU internal records; no warning names an affected task-0010H declaration.
- `javap -classpath <declared module outputs> -public
  io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration` again showed the exact
  nine-operation supported CPU surface and no internal-type leakage.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- `javac --release 26 -cp <declared module outputs> -d
  /tmp/synaptik-cpu-0010h-spi /tmp/CpuIntegrationSurfaceFixture.java` compiled a distinct-package
  fixture importing <span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture))</span>`CpuBackendIntegration`, `CompileArtifacts`, and `PreparedExecution`.
- `ruby /tmp/check_synaptik_0010h_docs.rb` passed local Markdown link/anchor, heading-order,
  balanced-fence, trailing-whitespace, LF, and final-newline checks for all eight changed Markdown
  files. `ruby /tmp/check_synaptik_0010h_scope.rb` passed the exact-path/status, public-delegation,
  source/import, Runtime, build, architecture, conformance, and integration exclusions.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance))</span>
- `git diff --check` passed for the final combined change.
- The final worktree contains exactly the fifteen authorized paths. Runtime, [Model](../../../../glossary.md#term-model),
  `modules/planning`, build/dependency files, architecture pages/tests, backend conformance, integration tests,
  generated code, tuning, provider behavior, and every non-allowlisted plan are unchanged.

No-change review conclusions:

- Runtime API and lifecycle documentation already defines initialized creators, initial validity,
  per-run ownership, rollback, leases, aliases, and cleanup; this task composes those contracts
  without changing them.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [rollback glossary](../../../../glossary.md#term-rollback); [leases glossary](../../../../glossary.md#term-lease); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Compile API documentation already defines the canonical source-only constant role, and Compiler
  0006B5 remains the completed logical-descriptor owner; no Compiler API change is needed.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [descriptor glossary](../../../../glossary.md#term-descriptor); [owner glossary](../../../../glossary.md#term-owner))</span>
- Training API is unaffected because this task adds neither gradient policy nor training
  orchestration.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- The glossary already defines the reused project terms and no new reusable term was introduced;
  adding an entry solely for this implementation would duplicate existing definitions.
- Architecture pages and tests remain accurate because ownership and dependency direction did not
  change. Backend-conformance and integration suites remain unchanged because the focused CPU SPI
  test crosses the existing Compiler/Prepare/Runtime composition and Engine 0006 still owns the
  later ordinary scalar-objective fixture.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi); [scalar glossary](../../../../glossary.md#term-scalar); [objective glossary](../../../../glossary.md#term-objective))</span>
- CPU inventory and build files remain unchanged because no production file, operation, route,
  dependency, or generated artifact was added.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>

## Completion summary

- Completed changes: added CPU-owned complete preparation, exact source-only resource derivation,
  existing initialized-buffer recipe composition, Engine-only delegation, focused coverage, and
  finalized lifecycle documentation/status.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [buffer glossary](../../../../glossary.md#term-buffer); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Files changed or created: exactly the fifteen authorized existing paths; no file was added.
- Tests and validation: reused the final 1,048-test implementation run; documentation Javadoc,
  public-surface, distinct-package compilation, Markdown, scope, forbidden-change, status, and
  whitespace gates passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run); [Javadoc glossary](../../../../glossary.md#term-javadoc); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Documentation-agent review: clean context `/root` finalized the affected Javadocs, public API,
  CPU backend guide, and planning records without changing executable Java behavior.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation impact: the supported source-only constant lifecycle and limitations are now
  current in both explanatory documents.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Javadoc review: all four changed production files were reviewed; three received final wording,
  while `CpuPreparedScheduleAssembler` already accurately documented recipe-only ownership and
  per-run creation and required no further documentation-pass edit.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Glossary impact: no change; all terminology is existing and already defined where reusable.
- Unresolved issues: None within task [scope](../../../../glossary.md#term-scope).
- Follow-up required: Reassess and plan Draft Engine 0006; pure zero-node and mixed-backend
  composition remain explicitly deferred.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [backend composition glossary](../../../../glossary.md#backend-composition))</span>

Status: Complete
