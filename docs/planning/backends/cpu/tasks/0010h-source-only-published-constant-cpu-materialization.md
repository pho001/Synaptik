# Task 0010H: Source-Only Published-Constant CPU Materialization

## Status

Complete

## Goal

Complete the current CPU composition for fully static source-only published compile-time splat
constants that accompany the sole non-empty maximal CPU partition. CPU derives each constant's
physical byte size and alignment from its canonical final descriptor, contributes the exact
`ProducerlessPublishedConstantResource` values to shared Prepare, and lets the existing CPU
assembler build one immutable initialized-buffer creator recipe for every assigned constant
representation.

The reusable `PreparedExecution` remains an immutable recipe and owns no physical, persistent, or
closeable resource. On every execution, existing Runtime `RunStateCreation` invokes each
initialized-buffer creator exactly once, producing and initializing one fresh run-owned CPU
representation before cold binding, reads, execution, or publication. In this task,
"materialized once" always means once per fresh `RunState`, never once per reusable
`PreparedExecution`.

## Current defect and selected flow

Compiler 0006B5 closes the canonical logical descriptor for the exact source-only publication
role. Prepare 0005 accepts an exact graph-value/logical-requirement pair plus backend-supplied
physical geometry, appends its buffer after ordinary partition declarations, and exposes the
assignment through `PreparedScheduleContext`. The current Engine CPU adapter still calls the
three-argument `GraphPreparation.prepare(...)`, so no CPU physical contribution reaches Prepare
and preparation fails closed before the assembler can reuse its existing constant branch.

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

There is no second initialization within a run, separate materialization schedule step,
synthetic executable, node, or partition owner, persistent prepared buffer, or hidden global
cache. No Runtime task or Runtime source change is part of this work.

## Scope

- Add exactly one supported Engine-facing operation to `CpuBackendIntegration`:

  ```java
  public PreparedExecution prepare(CompileArtifacts artifacts)
  ```

  It delegates to the owned CPU-private composition and returns the immutable reusable recipe.
  Existing `preparations(CompileArtifacts)` and `scheduleAssembler()` operations remain
  source-compatible; ordinary users still consume CPU through Engine.
- Add one matching CPU-private `CpuBackendComposition.prepare(CompileArtifacts)` composition
  entry. It must preserve the existing closed-owner and sole-partition validation precedence,
  obtain the current positional CPU preparation, derive the complete ordered producerless
  resource list, and call the four-argument `GraphPreparation.prepare(...)` with the retained
  assembler.
- Change `CpuEngineBackendComposition.prepare(...)` to delegate to
  `CpuBackendIntegration.prepare(...)`. Engine must not inspect graph roles, logical memory,
  descriptors, scalar constants, byte geometry, or `ProducerlessPublishedConstantResource`.
- Derive contributions only for graph values having the exact complete role already established
  by Compiler, Planning, and Prepare:

  1. the value is a final graph input;
  2. it has an exact `CompileConstantPlan.ConstantSource` splat classification;
  3. it is a requested graph output/publication;
  4. no node consumes it and no node produces it;
  5. its exact `LogicalMemoryRequirement` is producerless, consumerless, and graph-output
     required;
  6. its Shape is fully static; and
  7. its resolved layout equals `LayoutDescriptor.contiguous(shape)`.

  Iterate final `CompiledGraphModel.values()` order. Retain the exact graph-value and logical-
  requirement references in each contribution. Reject missing, duplicate, contradictory, or
  non-canonical facts rather than guessing or repairing them. Consumed or unpublished constants
  remain ordinary partition-connected values and receive no producerless contribution.
- Preserve the current executable domain. Artifacts must still contain exactly one non-empty
  maximal partition owned by `CpuCapabilityProvider.CPU_BACKEND_ID`. Zero-node/zero-partition and
  every mixed- or multi-partition composition remain rejected before resource derivation,
  backend preparation, or schedule assembly.
- Derive physical geometry from the canonical resolved descriptor:

  - `elementSpan = descriptor.layout().orElseThrow().referencedElementSpan()`;
  - `byteSize = Math.multiplyExact(elementSpan, descriptor.dataType().byteWidth())`; and
  - `byteAlignment = descriptor.dataType().byteWidth()`.

  Canonical contiguous geometry has zero offset, is not a view, and has the exact row-major
  strides for the Shape. Do not derive size from a node, kernel, carrier instance, publication,
  or Java array ceiling. Preserve checked `long` overflow.
- Support all six current `DataType` values through the existing exhaustive CPU initializer:
  `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT64`, `INT32`, and `BOOL`. Require the exact scalar type to
  equal the descriptor type. Do not add conversion, widening, rounding, BOOL normalization, or a
  default enum branch.
- A rank-zero descriptor has canonical span one, so its buffer size is one element width and its
  scalar is written once per run. A fully static Shape with any zero extent has canonical span and
  byte size zero; its fresh zero-byte representation is still created once per run, initialization
  performs zero stores, and the representation becomes initially valid. Preserve the Model's
  zero-before-product behavior and reject dynamic, unresolved, non-canonical, negative, or
  overflowing geometry.
- Reuse `CpuPreparedScheduleAssembler` without adding an executable materialization occurrence.
  Prepare's appended assignment enters the assembler's existing constant-source branch, which
  emits exactly one `PreparedRepresentationPlan.InitializedBuffer` at representation index zero.
  The recipe delegates to `CpuRepresentationRecipes.initializedBuffer(...)`; assembly itself must
  allocate, initialize, bind, execute, or publish nothing.
- Preserve existing schedule order: one representation-creation prefix, the sole partition
  execution occurrence, then forward publications followed by gradient publications. A
  source-only constant is not an executable selection and gains no fake read or write. Its
  initial validity comes solely from existing `RunStateCreation` behavior before all later cold
  binding and action traversal.
- Add focused CPU and Engine tests, update affected Javadocs and explanatory API/CPU documentation,
  and complete the mandatory separate clean documentation-focused pass after executable Java
  stabilizes.

### Deterministic validation and failure order

The new CPU composition entry must preserve this order for deterministic, non-racy checks:

1. reject a closed CPU integration through the existing owner guard;
2. reject null `artifacts`;
3. reject anything other than the existing sole non-empty CPU partition domain;
4. construct the existing positional preparation, retaining its current CPU boundary-analysis
   validation order;
5. scan final graph values in encounter order and correlate exact constant-source, graph-input,
   graph-output/publication, node-use, and logical-requirement facts;
6. for each required value, reject a dynamic Shape, unresolved layout, or layout unequal to
   `LayoutDescriptor.contiguous(shape)`, in that order;
7. reject scalar/descriptor type disagreement, then derive checked byte size and element-width
   alignment; and
8. invoke shared `GraphPreparation`, whose existing top-level, complete-coverage, assignment,
   finalization, assembler, and schedule validation order remains authoritative.

Null failures remain `NullPointerException`; unsupported role, domain, type, or geometry failures
are `IllegalArgumentException`; checked Shape/layout/byte arithmetic retains
`ArithmeticException`; closed ownership remains `IllegalStateException`. Add stable, value-
identifying messages for new deliberate CPU failures, but do not translate established lower-
layer failures. If a race with adapter closure begins after the open check, existing lifecycle
coordination rules remain authoritative.

### Ownership, validity, cleanup, and concurrency

- `PreparedExecution`, `PreparedRepresentationPlan`, and every buffer creator are immutable
  reusable recipes. They retain scalar and geometry facts but no `Arena`, `MemorySegment`, CPU
  buffer, lease, validity bit, or close action.
- Each `PreparedExecutionRunner.run(...)` creates one fresh `RunState`. `RunStateCreation` walks
  dense plan order and invokes the one creator for each non-caller representation exactly once.
  `CpuRepresentationRecipes.initializedBuffer(...)` allocates one fresh aligned
  `CpuNativeBuffer`, fills every complete element, and returns it.
- Initialization completes inside creator invocation. After all representations and workspaces
  are created, `RunStateCreation` constructs the state and marks every initialized-buffer
  position valid. The runner then cold-binds every execution, transfer, and publication
  occurrence before executing any action. No read or publication can observe an uninitialized or
  invalid source-only constant.
- If allocation or initialization fails, the initializer closes its newly allocated buffer and
  preserves a distinct cleanup failure as suppressed. `RunStateCreation` then releases all
  earlier successfully created run resources in reverse creation order and returns no state. If
  later binding, execution, or publication fails, the runner closes the created state once and
  preserves existing suppression behavior.
- Successful results lease the exact run-owned state. Repeated publication aliases or duplicate
  requested result occurrences refer to the assigned buffer/representation without copying or
  reinitializing it. Closing the final result ownership releases it through existing Runtime
  behavior. Host export reads the exact valid leased representation and returns detached bytes;
  it neither initializes nor changes ownership or validity.
- Two executions of the same prepared recipe, including concurrent executions, receive distinct
  initialized CPU buffers and independent validity/cleanup state. Within one run there is exactly
  one initialization per planned representation, irrespective of the number of publications,
  aliases, leases, or host-export calls. No shared mutable initializer state or cache is added.

## Out of scope

- Making a source-only constant the only graph value in a runnable zero-node schedule
- Mixed-backend or multi-partition schedule composition
- A Runtime task or any Runtime production, test, API, schedule, validity, lease, alias,
  publication, or cleanup modification
- A separate materialization step, transfer, executable, node, partition, finalizer assignment,
  owner, representation index, persistent prepared allocation, pool, memoization, or global cache
- Compiler, Planning, Prepare, Model, Config, Training, NN, Trace, provider, OpenBLAS, tuning,
  generated-code, kernel, or hot-loop changes
- Dynamic Shape binding, unresolved or non-canonical layout repair, general views, broadcast
  storage, scalar conversion, dense payload constants, or non-splat constants
- Removal or signature changes of the existing supported CPU preparation/assembler SPI
- Engine 0006 implementation, backward API work, or its final scalar-objective fixture
- Architecture-contract, ADR, dependency, build, backend-conformance, or benchmark changes

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Runtime, Prepare, and Backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0010: staged backend preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: per-run Runtime resource ownership](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Planning guide](../../../planning-guide.md) and [roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [CPU 0010F lifecycle integration](0010f-supported-cpu-lifecycle-integration-adapter.md)
- [CPU 0010G host snapshot export](0010g-canonical-caller-owned-host-snapshot-export.md)
- [Compiler 0006B5 descriptor closure](../../../modules/compiler/tasks/0006b5-published-compile-time-constant-descriptor-closure.md)
- [Prepare 0005 resource handoff](../../../modules/prepare/tasks/0005-producerless-published-constant-resource-handoff-and-shared-slot-assignment.md)
- [Engine 0006 scalar-objective backward convenience](../../../modules/engine/tasks/0006-one-shot-scalar-objective-backward-convenience.md)
- Runtime [0007 representation plan](../../../modules/runtime/tasks/0007-representation-creation-and-residency-foundation.md),
  [0009 publication](../../../modules/runtime/tasks/0009-publication-and-result-schedule-steps.md),
  [0010 runner](../../../modules/runtime/tasks/0010-prepared-runner-and-dynamic-execution.md), and
  [0015 leased access](../../../modules/runtime/tasks/0015-leased-publication-representation-access.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)

## Architecture constraints

- Compiler owns final graph identity, source classification, publication order, and canonical
  logical descriptors. CPU consumes those facts without rewriting them.
- Planning owns producer/consumer/output obligations. CPU retains the exact logical requirement
  and invents no partition relationship.
- Concrete CPU owns byte size, alignment, representation type, and initialization recipe.
  Prepare validates/transports declarations and assigns slots; Runtime invokes creators and owns
  mutable per-run state.
- Engine is the explicit composition root but must not derive CPU facts. It delegates preparation
  through supported CPU SPI and exposes no CPU-private or Prepare contribution type in ordinary
  user APIs.
- CPU may use existing Compiler, Model, Planning, Prepare, and Runtime contracts through its
  already declared execution-side dependencies. CPU must not depend on Engine.
- The sole non-empty CPU partition contract and existing complete schedule assembler remain
  unchanged. Stop if implementation requires a second assembler, a general registry/facade, a
  synthetic owner, or a dependency/architecture change.

## Package impact

Changed supported package:

- `io.github.pho001.synaptik.backend.cpu` — add only the supported
  `prepare(CompileArtifacts)` integration operation; expose no internal CPU type or
  `ProducerlessPublishedConstantResource`.

Changed CPU-private package:

- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas` — add the exact
  composition entry and focused private resource derivation.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — executable behavior remains unchanged;
  finalize assembler Javadoc so source-only assignments and recipe-only ownership are explicit.

Changed Engine-private package:

- `io.github.pho001.synaptik.engine` — the existing CPU adapter delegates preparation to the new
  supported CPU operation. No Engine public/protected API changes.

No package or module is added.

## Affected files

Expected production/Javadoc files:

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

Planning/status files:

- `docs/planning/backends/cpu/tasks/0010h-source-only-published-constant-cpu-materialization.md`
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`
- directly stale prerequisite/status sentences in Compiler, Prepare, and Engine master plans only

No other path is authorized by default.

## Maximum scope

This cross-boundary lifecycle completion may modify at most fifteen paths: four production/Javadoc
paths, three existing test paths, two explanatory documents, and six planning/status paths. It
adds no production or test file. If executable behavior requires another source/test path, a
Runtime change, a build/dependency edit, or an architecture document, stop and revise the plan
before implementation. An unchanged assembler or documentation path is unused capacity, not
permission for unrelated work.

## Acceptance criteria

- `CpuBackendIntegration` remains public, final, privately constructed, and `AutoCloseable`. Its
  public declarations are exactly its existing eight operations plus
  `prepare(CompileArtifacts)`. No supported signature exposes `.internal`, Engine, a physical
  carrier, or `ProducerlessPublishedConstantResource`.
- The new integration operation rejects closure before null and delegates all composition work to
  its exact owned `CpuBackendComposition`.
- CPU accepts exactly one non-empty CPU partition and rejects zero-node, zero-partition,
  mixed-owner, and multi-partition artifacts before derivation or backend preparation. Tests lock
  these fail-closed boundaries.
- Required contributions are complete, unique, and in final graph-value order. Each retains the
  exact artifact `GraphValue` and `LogicalMemoryRequirement`; consumed, produced, unpublished,
  bindable, dynamic, unresolved, or non-canonical values do not silently enter the list.
- Canonical span times element width, with checked `long` arithmetic, is the exact physical byte
  size. Element width is the exact power-of-two alignment. Rank-zero is one element; any zero
  extent is zero bytes even if other unused extents would otherwise overflow.
- All six current data types preserve the exact scalar representation through the existing CPU
  initializer, including raw floating/BFLOAT16 values, signed integers, and canonical BOOL.
- Shared Prepare appends each source-only assignment after all ordinary buffers without changing
  their slots, plan indices, finalizer assignments, workspaces, or partition preparation.
- The CPU assembler emits one representation-creation prefix, one existing execution occurrence,
  and the existing ordered publication suffix. Each source-only assigned constant has exactly one
  `InitializedBuffer` creator at representation index zero and no executable selection,
  transfer, or extra schedule step.
- Assembly/preparation performs no allocation or initialization. Every run invokes each
  initialized creator once, produces a distinct run-owned representation, initializes it once,
  and marks it valid before binding or action traversal. Repeated publications, aliases, leases,
  and host exports do not trigger initialization.
- Two sequential and two concurrent executions of one `PreparedExecution` prove distinct
  run-owned representations and identical initialized content. Closing either result cannot
  affect the other run or the reusable recipe.
- Zero-byte initialized representations publish successfully without a store. Scalar and all-six-
  type cases export the expected canonical host bytes while the result lease is open.
- Allocation/initialization failure cleanup and later runner-failure cleanup remain the existing
  Runtime/CPU behavior; focused source inspection and existing Runtime tests are cited rather than
  modified or redundantly recreated.
- `CpuEngineBackendComposition` delegates to `integration.prepare(artifacts)` and contains no
  producerless-role or geometry derivation. Engine 0006 remains Draft and unimplemented.
- No new ordinary Engine API, CPU facade/type, Runtime mechanism, inventory entry, backend-
  conformance contract, integration-test fixture, architecture rule, ADR, module edge, benchmark,
  generated code, or provider/tuning behavior is added.
- A distinct clean documentation-focused pass finalizes all changed Javadocs, public API and CPU
  guide text, link/term impact, no-change decisions, and planning evidence before implementation
  may be marked Complete.

## Tests / validation

The implementation context runs focused tests while developing, then one final affected-module
validation after executable Java stabilizes:

```bash
./gradlew :backends:cpu:test :modules:engine:test
```

Focused tests must cover supported public shape; direct integration preparation; exact source-only
eligibility and graph-value order; exact reference handoff; sole-partition rejection; canonical,
dynamic, unresolved, non-canonical, scalar, zero-element, and overflow geometry; all six data
types; prepared recipe shape; initial validity; sequential and concurrent per-run representation
identity/content/cleanup; publication aliases; and Engine's CPU delegation.

The clean documentation-focused context reuses successful Java evidence unless it changes
executable Java, then runs:

```bash
./gradlew :backends:cpu:javadoc :modules:engine:javadoc
git diff --check
```

It also records:

- `javap -public` or the existing reflection assertion for the exact nine-operation supported CPU
  surface and absence of `.internal`, Engine, carrier, and producerless contribution types;
- compilation of the distinct-package CPU SPI test against declared dependencies;
- Markdown local-link/anchor, heading-order, balanced-fence, trailing-whitespace, and final-newline
  checks for changed documentation;
- the exact at-most-fifteen-path allowlist and preservation of unrelated dirty-worktree changes;
- source/import scans proving CPU has no Engine dependency, Engine derives no CPU facts, Runtime is
  unchanged, and no synthetic schedule occurrence/cache was introduced; and
- synchronized task/master/roadmap statuses with CPU 0010H Complete only after implementation and
  documentation validation, while Engine 0006 remains Draft.

No benchmark is required: initialization is cold once-per-run setup, not a per-element executable
hot loop. Backend conformance is unchanged because no shared backend contract or behavior is
added. `testing/integration-tests` is unchanged because the focused CPU SPI run crosses actual
Compiler/Prepare/Runtime behavior and the Engine-private delegation test locks the only Engine
wiring change; Engine 0006 owns the later ordinary end-to-end scalar-objective fixture. CPU source
inventory is unchanged because no production file is added. Architecture tests and repository-
wide validation are not required because no dependency, module boundary, shared build, or
architecture rule changes; CI remains the broader independent gate.

## Dependencies

- [Compiler 0006B5](../../../modules/compiler/tasks/0006b5-published-compile-time-constant-descriptor-closure.md) —
  Complete; supplies exact canonical logical descriptor closure for the eligible role.
- [Prepare 0005](../../../modules/prepare/tasks/0005-producerless-published-constant-resource-handoff-and-shared-slot-assignment.md) —
  Complete; supplies exact resource handoff, validation, deterministic append-only assignment,
  and the four-argument orchestration entry.
- [CPU 0010F](0010f-supported-cpu-lifecycle-integration-adapter.md) — Complete; supplies the
  supported owner, sole-partition preparation, representation recipes, and schedule assembler.
- [CPU 0010G](0010g-canonical-caller-owned-host-snapshot-export.md) — Complete; supplies detached
  canonical host export for validation and later Engine consumption.
- Existing Runtime 0007/0009/0010/0015 contracts — Complete and sufficient without modification.

All prerequisites are complete and the task is actionable.

## Follow-up tasks

- [Engine 0006](../../../modules/engine/tasks/0006-one-shot-scalar-objective-backward-convenience.md)
  remains Draft and is the next downstream task after CPU 0010H implementation, documentation,
  and validation are Complete.
- Zero-node/pass-through and mixed-backend composition remain future architecture/planning work
  only when a concrete consumer justifies them.

## Documentation-focused clean-context pass

After executable Java and focused tests stabilize, hand the exact diff, baseline dirty status,
test evidence, source-only eligibility/geometry decisions, public-shape result, and lifecycle
ownership analysis to a distinct clean documentation-focused context. It must read the General,
API/Javadoc, and Planning profiles and finalize affected Javadocs, `docs/api/public-api.md`, the
CPU backend guide, and planning evidence.

The pass must explicitly state that `PreparedExecution` owns recipes only and that materialization
occurs once per fresh `RunState`. Review Runtime API/lifecycle documentation, glossary, Compile
API, Training API, architecture pages/tests, backend conformance, integration tests, CPU inventory,
and build files. Change them only if a concrete stale statement or executable impact exists;
otherwise record a reasoned no-change conclusion. Do not repeat successful Java tests unless the
pass changes executable Java or identifies a concrete risk.

## Architecture impact

Expected impact: None.

This task fills the concrete CPU side of existing Compiler -> Planning -> Prepare -> Runtime
contracts. It changes neither ownership nor dependency direction. The new supported CPU operation
keeps physical fact derivation in CPU and lets Engine remain composition-only. If implementation
requires persistent prepared storage, another Runtime concept, a synthetic graph/schedule owner,
a build edge, or an architecture/ADR edit, stop and report the conflict rather than widening.

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

Stop and report instead of implementing or widening if:

- exact eligibility cannot be derived from existing immutable Compile artifacts and logical
  requirements;
- accepted input from Compiler 0006B5 is not canonical contiguous or requires CPU to repair it;
- a source-only value must be assigned to a partition/finalizer or selected by an executable;
- preparation or assembly must allocate/initialize a physical resource;
- correctness needs initialization more than once per representation per `RunState`;
- success requires a zero-node schedule, mixed/multi-partition composition, transfer, Runtime
  change/task, persistent resource/cache, new dependency, architecture/ADR edit, benchmark, new
  production/test file, or a sixteenth path; or
- Engine must inspect or construct `ProducerlessPublishedConstantResource` or any CPU fact.

## Local decisions

- One supported CPU `prepare` operation is narrower than exposing resource contributions to
  Engine. Existing preparation and assembler operations remain for compatibility, but the built-in
  Engine path uses the complete CPU-owned composition entry.
- Final graph-value order is the deterministic resource order because Compile and Prepare already
  use that order; numeric IDs, constant-list order, map order, and publication duplicates are not
  new ordering policies.
- Canonical referenced span is the physical element capacity for this exact non-view source. CPU's
  current native carrier uses element width for alignment, including one-byte BOOL.
- Prepare's append-only assignment and the assembler's existing constant branch already separate
  physical declaration, immutable initializer recipe, and per-run representation ownership. No
  second materialization mechanism is justified.
- Runtime already proves creator invocation, initial validity, cold binding, result leases, and
  cleanup. CPU tests exercise the composed behavior; Runtime tests and source are not changed.

## Known limitations

This task supports only fully static canonical splat constants accompanying current non-empty
CPU work. It does not make pure constant graphs runnable, support dynamic or view geometry,
introduce dense non-splat payloads, or compose multiple backend schedules. Engine 0006 remains
Draft until this implementation is complete.

## Validation evidence

Planning context `01a0a5c5-c93a-7010-9067-2ae5582e2786` created this detailed Ready specification
after full architecture/planning/documentation reading and focused inspection of the completed
Compiler, Prepare, CPU, Engine, and Runtime contracts. It performed no Java change or Java test.

Planning validation:

- Markdown heading, fence, link-target, trailing-whitespace, and final-newline checks;
- task ordering and Ready/dependency synchronization across directly affected planning indexes;
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
  producerless-resource type in its supported signatures.
- The implementation context's exact seven-path scope, prohibited-area scans, and
  `git diff --check` passed after executable Java stabilized.
- One early compile attempt failed only because four new `java.util` imports were accidentally
  placed after the class declaration. Moving those imports to the import section corrected the
  source; subsequent compilation and the final tests passed. This was an import-placement error,
  not a design or behavior defect.

Clean documentation-focused context `/root` independently inspected the implementation and tests,
then finalized the API/Javadoc, Backend Guide, Planning, and Example-profile concerns. It changed
Javadoc only in three production files, changed no executable Java token, and did not edit the
three test files. It finalized `docs/api/public-api.md`, `docs/backend-guide/cpu-backend.md`, and
the six planning/status files. The pass made these boundaries explicit:

- `PreparedExecution` owns immutable recipes only; each fresh `RunState` invokes each initialized
  recipe once and owns a distinct physical representation;
- source-only published constants gain no executable schedule step, and pure zero-node constant
  graphs remain unsupported;
- Engine delegates complete preparation without interpreting CPU roles or geometry; and
- CPU 0010H is Complete while Engine 0006 remains Draft and is the next task to reassess and plan.

Documentation validation:

- `./gradlew :backends:cpu:javadoc :modules:engine:javadoc` passed. The output contained the two
  existing incubating-module warnings and 92 pre-existing missing-`@param` warnings in unrelated
  CPU internal records; no warning names an affected task-0010H declaration.
- `javap -classpath <declared module outputs> -public
  io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration` again showed the exact
  nine-operation supported CPU surface and no internal-type leakage.
- `javac --release 26 -cp <declared module outputs> -d
  /tmp/synaptik-cpu-0010h-spi /tmp/CpuIntegrationSurfaceFixture.java` compiled a distinct-package
  fixture importing `CpuBackendIntegration`, `CompileArtifacts`, and `PreparedExecution`.
- `ruby /tmp/check_synaptik_0010h_docs.rb` passed local Markdown link/anchor, heading-order,
  balanced-fence, trailing-whitespace, LF, and final-newline checks for all eight changed Markdown
  files. `ruby /tmp/check_synaptik_0010h_scope.rb` passed the exact-path/status, public-delegation,
  source/import, Runtime, build, architecture, conformance, and integration exclusions.
- `git diff --check` passed for the final combined change.
- The final worktree contains exactly the fifteen authorized paths. Runtime, Model,
  `modules/planning`, build/dependency files, architecture pages/tests, backend conformance, integration tests,
  generated code, tuning, provider behavior, and every non-allowlisted plan are unchanged.

No-change review conclusions:

- Runtime API and lifecycle documentation already defines initialized creators, initial validity,
  per-run ownership, rollback, leases, aliases, and cleanup; this task composes those contracts
  without changing them.
- Compile API documentation already defines the canonical source-only constant role, and Compiler
  0006B5 remains the completed logical-descriptor owner; no Compiler API change is needed.
- Training API is unaffected because this task adds neither gradient policy nor training
  orchestration.
- The glossary already defines the reused project terms and no new reusable term was introduced;
  adding an entry solely for this implementation would duplicate existing definitions.
- Architecture pages and tests remain accurate because ownership and dependency direction did not
  change. Backend-conformance and integration suites remain unchanged because the focused CPU SPI
  test crosses the existing Compiler/Prepare/Runtime composition and Engine 0006 still owns the
  later ordinary scalar-objective fixture.
- CPU inventory and build files remain unchanged because no production file, operation, route,
  dependency, or generated artifact was added.

## Completion summary

- Completed changes: added CPU-owned complete preparation, exact source-only resource derivation,
  existing initialized-buffer recipe composition, Engine-only delegation, focused coverage, and
  finalized lifecycle documentation/status.
- Files changed or created: exactly the fifteen authorized existing paths; no file was added.
- Tests and validation: reused the final 1,048-test implementation run; documentation Javadoc,
  public-surface, distinct-package compilation, Markdown, scope, forbidden-change, status, and
  whitespace gates passed.
- Documentation-agent review: clean context `/root` finalized the affected Javadocs, public API,
  CPU backend guide, and planning records without changing executable Java behavior.
- Documentation impact: the supported source-only constant lifecycle and limitations are now
  current in both explanatory documents.
- Javadoc review: all four changed production files were reviewed; three received final wording,
  while `CpuPreparedScheduleAssembler` already accurately documented recipe-only ownership and
  per-run creation and required no further documentation-pass edit.
- Glossary impact: no change; all terminology is existing and already defined where reusable.
- Unresolved issues: None within task scope.
- Follow-up required: Reassess and plan Draft Engine 0006; pure zero-node and mixed-backend
  composition remain explicitly deferred.

Status: Complete
