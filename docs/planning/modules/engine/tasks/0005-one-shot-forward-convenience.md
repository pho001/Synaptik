# Task 0005: One-Shot Forward Convenience

## Status

Complete

## Goal

Add the smallest ordinary Engine-owned convenience that executes one forward request from one
Tensor output or an ordered non-empty output list plus explicit logical input Tensors and returns
detached immutable host values. One synchronous call owns the complete temporary
`compile -> prepare -> run -> materialize every exact forward publication -> close result`
lifecycle while preserving the reusable `CompiledGraph`, `PreparedExecution`, `RunResult`, and
per-publication materialization APIs for callers that need repeated runs or selective output
access.

The convenience uses one caller-supplied non-negative aggregate byte limit for all returned
canonical payloads. It neither caches nor reuses compilation or preparation, and it does not add
execution, backward, gradient, or Runtime state to Model `Tensor`.

## Required reading

Read these files and contracts in full in the clean implementation context rather than relying
only on this specification:

- root `AGENTS.md`, [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), and the
  [current architecture plan](../../../../architecture/current-architecture-plan.md);
- [Lifecycle](../../../../architecture/lifecycle.md),
  [Module boundaries](../../../../architecture/module-boundaries.md),
  [Dependency rules](../../../../architecture/dependency-rules.md),
  [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md),
  and [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md);
- [Planning guide](../../../planning-guide.md), [roadmap](../../../roadmap.md), this task, the
  [Engine master plan](../master-plan.md), and completed Engine tasks
  [0001](0001-advanced-composition-and-representation-level-lifecycle-foundation.md),
  [0002](0002-standard-built-in-composition.md),
  [0003](0003-typed-logical-input-binding-and-published-result-access.md), and
  [0004](0004-explicit-host-materialization-boundary.md);
- completed Runtime task
  [0015](../../runtime/tasks/0015-leased-publication-representation-access.md) and CPU task
  [0010G](../../../backends/cpu/tasks/0010g-canonical-caller-owned-host-snapshot-export.md),
  including their final source, tests, Javadocs, and completion evidence;
- every current Engine production and test file, all three ordinary Engine integration tests,
  and the focused Engine architecture test;
- current Model `Tensor`, `TensorId`, `TensorDescriptor`, `DataType`, `Shape`,
  `LayoutDescriptor`, `HostTensorStorage`, and `MemorySegmentStorage` contracts;
- the current public, compile, runtime, training, Tensor, and glossary sections that describe
  Engine execution, input binding, publication, materialization, ownership, and planned one-shot
  behavior; and
- [Documentation rules](../../../../developer-guide/documentation-rules.md) plus the General,
  API/Javadoc, Planning, and Example profiles under
  `docs/developer-guide/documentation/`.

`ARCHITECTURE.md` is authoritative. Stop before editing if the implementation would require an
architecture change, a new dependency direction, or an inward public-contract change.

## Scope

- Add two ordinary `Engine.forward(...)` overloads: one for exactly one output and one for an
  ordered non-empty list of outputs.
- Require explicit logical input Tensors on both overloads. Use the same final Compiler-owned
  `TensorId` binding inventory, descriptor equality, association snapshot, storage validation,
  borrow order, and caller ownership rules as the reusable task-0003 run path.
- Interpret one `long maximumTotalBytes` as a bound on the checked sum of all returned canonical
  payload byte counts. Validate the complete result set before the first physical copy.
- Keep one Engine lifecycle admission open across validation, compilation, preparation, run,
  aggregate-size preflight, ordered materialization, result cleanup, and return readiness.
- Compile once with the existing ordinary `FORWARD_ONLY` fixed settings, prepare once, run once,
  materialize each exact forward publication once in publication order, and close the temporary
  result once before returning.
- Return only existing detached immutable `HostTensorValue` instances. The multi-output overload
  returns an immutable ordered `List<HostTensorValue>`; the single-output overload returns its one
  element directly.
- Share private/package-private ungated lifecycle helpers with the reusable lower-level path so
  logical binding, publication construction, materialization, and cleanup semantics do not drift.
- Add focused public-shape, semantic, failure, ownership, cleanup, concurrency, and real CPU
  integration coverage, including the existing advanced-surface inventory test whose exact
  ordinary `Engine` method-name list must admit the two new overloads.
- Finalize affected Javadocs and explanatory documentation through a distinct clean
  documentation-focused pass in the same overall change.

## Exact public API

Add exactly these declarations to the existing public final `Engine`:

```java
public HostTensorValue forward(
        Tensor output,
        List<Tensor> inputs,
        long maximumTotalBytes);

public List<HostTensorValue> forward(
        List<Tensor> outputs,
        List<Tensor> inputs,
        long maximumTotalBytes);
```

No new public or protected type, constructor, field, nested type, enum, record, request object,
options object, result carrier, or overload is authorized. Every other public Engine and advanced
Engine declaration remains source-compatible and unchanged.

The single-output overload is justified because the dominant one-output case should not require
constructing and unwrapping two lists. It is semantically the exact singleton specialization of
the ordered overload, not a different lifecycle or policy. The ordered overload is required
because Compiler publication order and current multi-output expression boundaries are already
first-class and executing outputs separately could repeat shared graph work and lose occurrence
order. `List<HostTensorValue>` is sufficient because every returned value is already immutable,
detached, and self-describing; another carrier or request abstraction would add no ownership,
role, or lifecycle fact.

`maximumTotalBytes` is a primitive `long` rather than a new limit type because this task has one
quantity, one unit, and no optional/default policy. The name and Javadoc must state that the limit
is aggregate across output payloads, not per output, input storage, intermediate storage, Runtime
allocation, or peak memory.

Ordinary public/protected signatures continue to expose only Engine, Model, and JDK types. They
must not expose Compiler, Prepare, Runtime, Planning, backend-contract, CPU, another backend,
advanced handles, `.internal`, representations, storage objects, arenas, memory segments, slots,
`ValueId`, or backend identity.

## Forward request semantics

### Arguments and snapshots

After Engine admission, the single-output overload validates in this order:

1. reject null `output` with `NullPointerException("output")`;
2. reject null `inputs`, then a null input element in increasing index order, using the existing
   `inputs` and `inputs[index]` messages;
3. reject `maximumTotalBytes < 0` with
   `IllegalArgumentException("maximumTotalBytes must be non-negative: " + maximumTotalBytes)`.

After Engine admission, the ordered overload validates in this order:

1. reject null `outputs` with `NullPointerException("outputs")`;
2. snapshot its elements in encounter order and reject the first null as `outputs[index]`;
3. reject an empty output list with `IllegalArgumentException("outputs must not be empty")`;
4. reject repeated exact Tensor object identity with the existing indexed duplicate form used by
   ordinary compilation, replacing the list name with `outputs`;
5. reject null `inputs`, then the first null input element in increasing index order;
6. reject a negative aggregate limit with the exact single-output message above.

Both overloads snapshot caller list structure before inward work, retain no caller collection,
and never mutate it. The single overload constructs one private singleton snapshot and enters the
same ungated implementation as the multi-output form; it must not invoke the public multi-output
overload or acquire a second admission. Inputs remain caller-supplied ordinary Tensors and may be
listed in any order. Empty inputs are locally valid and reach the established exact binding
validation; Engine does not invent an input from output provenance or storage.

### Exact internal sequence

For each admitted call, execute this sequence exactly once and in this order:

1. validate and snapshot the public arguments above;
2. compile the ordered output snapshot through the existing ordinary forward-only Compiler path
   with `GraphOptimizationConfig.standard()`, `BackendIntent.unconstrained()`, and
   `PartitionScoringConfig.neutral()`;
3. construct the same owner-bound compile metadata used by `Engine.compile(outputs)`;
4. prepare that exact compile result once through the current owned CPU composition;
5. run that exact prepared result once with the explicit Tensor input snapshot, using the
   established task-0003 logical binding, storage snapshot, borrowing, Runtime execution,
   publication-count defense, and wrapper ownership rules;
6. before any physical copy, validate the complete ordered forward-publication set and its
   aggregate canonical byte count as specified below;
7. materialize every exact publication object once in increasing publication index order through
   the existing task-0004 descriptor/representation/CPU-copy path;
8. build the immutable ordered detached-value list, or select its sole value for the single form;
9. close the temporary Runtime result lease and Engine-created borrow wrappers once; and
10. only after successful cleanup, release Engine admission and return the detached value or
    immutable list.

The one-shot path must not call public `compile`, `prepare`, `run`, or `materialize` methods from
inside its admission because those methods acquire their own admissions and, for results, the
ordinary reusable registration lifecycle. Refactor narrowly into shared ungated operations or an
equivalent package-private implementation that preserves the reusable methods' exact observable
semantics. Do not introduce a callback executor, generic request/facade/manager, service locator,
second lifecycle gate, or second result registry.

The temporary compile and prepared handles own no closeable resource. The temporary result is
owned solely by the admitted one-shot call and must never be returned or left registered for later
Engine cleanup. Existing reusable handles and results keep their current owner-bound reuse and
explicit-close behavior unchanged.

## Publication and byte-limit semantics

The compiled request is forward-only, so the successful result must contain exactly one forward
publication occurrence for each requested output and no gradient occurrence. Before copying,
validate all of these invariants in increasing index order:

- result count and publication-list size equal `outputs.size()`;
- publication `index()` equals its dense list index;
- publication `role()` is `FORWARD`;
- publication `tensorId()` equals the corresponding requested output's `TensorId`;
- `derivativeOrder()` and `targetIndex()` are empty; and
- the final publication descriptor is the exact descriptor used for host materialization.

An impossible mismatch is `IllegalStateException` with an index/count-specific diagnostic and
causes ordinary cleanup. Do not repair, reorder, search by Tensor identity, merge aliases, or
substitute a different occurrence. The returned list position is therefore exactly the requested
output position and Runtime publication position.

For every publication in order, perform the same descriptor-derived preflight used by explicit
materialization:

- require a fully static Shape;
- require a resolved layout;
- compute logical element count with the Model zero-before-product rule;
- compute canonical byte count with checked `long` multiplication; and
- require that individual byte count is at most `Integer.MAX_VALUE`.

Accumulate individual canonical byte counts in publication order with `Math.addExact`. A numeric
overflow remains `ArithmeticException`. Zero-element outputs contribute zero. Only after all
descriptors and the complete sum have passed, compare the sum with `maximumTotalBytes`. If it is
larger, throw:

```text
total canonical byte count exceeds maximumTotalBytes: required=<sum>, maximum=<limit>
```

This complete preflight happens after successful run construction but before the first physical
copy. Therefore a limit, static-Shape, resolved-layout, JVM-array-ceiling, or aggregate-overflow
failure performs zero CPU host-copy calls and then closes the temporary result. It does not avoid
the compile, prepare, or run work already required to obtain and validate the exact publication
set.

After preflight, call the established exact-occurrence materialization implementation once per
publication in order. Supply that occurrence's already checked individual canonical byte count as
its per-copy `maximumBytes`; this is the tight bound required for that payload and cannot weaken
the aggregate caller limit. Each returned `HostTensorValue` must match the preflighted type,
Shape, element count, and byte count. Repeated outputs are already rejected by exact Tensor
identity, but distinct output occurrences may alias one inward representation; they still produce
distinct uncached snapshots and distinct `HostTensorValue` objects.

The aggregate limit covers only the sum of returned canonical payload lengths. It does not claim
to bound compilation data, prepared recipes, input storage, Runtime buffers/workspaces, temporary
CPU access, list/object overhead, defensive cloning, peak heap use, or memory held before cleanup.
Every `HostTensorValue` retains task 0004's `Integer.MAX_VALUE` byte-array ceiling and available-
heap limitation.

## Input ownership and storage lifetime

- Logical input validation is exactly the reusable task-0003 contract: every final bindable
  `TensorId` appears once, unexpected/duplicate/missing identities fail, descriptors compare by
  complete value equality, and a descriptor-equal different identity is not a replacement.
- Only after complete logical validation, read each required Tensor's synchronized
  `hostStorage()` association once in final Compiler binding order and retain that exact storage
  for the call.
- Validate type, resolved-layout span where present, liveness, and current-thread accessibility
  before borrowing. CPU cold binding remains an independent inward validation.
- Distinct logical inputs sharing one storage object receive distinct fresh borrow wrappers.
  Engine does not deduplicate wrappers, copy input bytes, infer layout, convert values, or write
  back.
- Caller storage and its arena remain caller-owned. They must stay live, accessible where used,
  and free from conflicting mutation through synchronous method completion, including result and
  wrapper cleanup. Because the temporary result is closed before return, task 0003's longer
  caller-storage-through-result-close obligation ends when this method returns or throws after
  cleanup.
- Replacing or clearing a Tensor association after its per-Tensor snapshot cannot redirect the
  active call. The snapshots are not atomic across inputs and provide no payload memory fence,
  pinning, or protection from caller races.

## Failure, cleanup, and lifecycle semantics

### Failure precedence

1. Engine admission is first. Once closure has begun, preserve
   `IllegalStateException("advanced engine is closed")` before every argument, limit, compile,
   prepare, input, descriptor, publication, Runtime, CPU, or cleanup precondition.
2. Public argument validation follows the exact overload-specific order above.
3. Compiler failures precede Prepare failures; Prepare failures precede logical-input and run
   failures; run/publication failures precede aggregate materialization preflight; preflight
   failures precede CPU copying.
4. Publication preflight is complete before copying. During copying, the first failure in
   publication order stops further materialization.
5. The temporary result is then closed exactly once. If a primary failure already exists, a
   distinct cleanup `RuntimeException` or `Error` is suppressed on that exact primary object. A
   cleanup failure equal by reference to the primary is not self-suppressed. If no earlier
   failure exists, the first cleanup failure becomes primary and no value/list is returned.

Nulls remain `NullPointerException`; local list/limit/identity/descriptor problems remain
`IllegalArgumentException`; lifecycle and impossible publication/result inconsistency remain
`IllegalStateException`; checked count arithmetic remains `ArithmeticException`; inward
unchecked failures and `OutOfMemoryError` retain their original type and object identity. Do not
introduce an Engine exception hierarchy or translate Compiler, Prepare, Runtime, CPU, or JDK
failures.

Before Runtime result construction, preserve existing task-0003 rollback: partial borrows close
in reverse order; run failure closes run-owned state through Runtime and then wrappers; caller
storage is never closed. After result construction, the one-shot call owns that exact result and
all wrappers through its established close protocol. A failed call publishes no partial return,
even if earlier output snapshots were already constructed.

### Engine close and concurrency

- Both overloads are synchronous and thread-safe through the existing Engine lifecycle gate.
  Separate one-shot calls may run concurrently and receive isolated Runtime states.
- One admission spans the whole operation. Engine close that begins after admission waits for
  compile, prepare, run, every copy, and result cleanup to finish before closing the CPU
  composition.
- Because only detached values escape, an admitted successful call may return its completed value
  or list after another thread has begun Engine closure. This matches task 0004's admitted-copy
  rule and does not create a new usable Engine handle or open lease after closure.
- A call that loses admission fails with the stable closed-Engine error and performs no argument
  inspection or inward work.
- No temporary one-shot result is added to the Engine's open-result registry. Existing reusable
  results remain registered and continue to close in reverse successful-run order during Engine
  shutdown.
- The implementation adds no cancellation, asynchronous API, worker, shutdown hook, process-
  global state, automatic retry, fallback, or synchronization of caller payload mutation.

## Out of scope

- Any method on `Tensor`, including `execute`, `forward`, `run`, `backward`, or a gradient/value
  accessor; any Model-to-Runtime, Model-to-Engine, or Model-to-backend dependency.
- Engine task 0006 scalar-objective backward convenience, backward-capable one-shot overloads,
  implicit scalar seeds, implicit targets, target discovery, `withBackward`, mutable Tensor
  gradient state, optimizer/session work, or training orchestration.
- A request/options/configuration/result wrapper, default byte limit, unlimited sentinel,
  per-output limits, streaming/chunked/mapped output, caller destination, typed arrays, scalar
  decoding, Tensor reconstruction, persistence format, or output cache.
- Reuse or caching of compiled graphs, prepared executions, result bytes, backend decisions, or
  hidden process-global lifecycle state. Callers needing reuse continue to use the lower-level
  ordinary compile/prepare/run/materialize API.
- Model autotuning, Config 0006A mapping, tools/tuning integration, profiling-based selection, or
  any Engine 0007 behavior.
- Cross-backend host transfer or materialization abstraction, another concrete backend, mixed-
  backend schedule composition, backend selection controls, registration/discovery API,
  reflection, `ServiceLoader`, service location, or fallback from CPU.
- Successful zero-node/pass-through preparation, dynamic or binding-dependent output Shapes,
  unresolved output layouts, layout inference, CPU capability expansion, operation-specific
  Engine switches, or any Engine 0006-or-later implementation.
- Changes to Model, Compiler, Planning, Prepare, Runtime, CPU, Config, Trace, Backend Contract,
  build files/dependencies, architecture contracts/docs, ADRs, architecture tests,
  backend-conformance tests, NN, Training, tuning, generated code, or unrelated integration
  behavior.
- Creating an Engine 0006-or-later detailed task specification or changing Engine 0006–0008 from
  `Draft`.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core lifecycle, Core invariants,
  Engine, Runtime, Prepare, concrete backends, dependency rules, and compile/prepare/run lifecycle.
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)

## Architecture constraints

- Engine remains the outer composition and public lifecycle owner. Compiler owns compilation,
  Prepare owns staged preparation, Runtime owns each isolated run and lease cleanup, and CPU owns
  concrete execution, representation access, and canonical copying.
- The convenience must be orchestration over the exact existing lower-level contracts, not a
  parallel compiler, binder, runner, publication mapper, or materializer.
- `Tensor` remains Model state without execution, Runtime, backward, gradient-lifecycle, backend,
  or Engine responsibility.
- Prepared execution remains immutable and reusable even though this convenience intentionally
  creates a fresh prepared recipe per call. The reusable ordinary API remains available and
  unchanged.
- Runtime performs only already-prepared work and does not discover a backend, compile, lower,
  select a route, interpret outputs, or copy host values.
- CPU remains the sole supported standard lifecycle adapter and the sole current concrete host-
  copy implementation. No cross-backend abstraction is inferred from this convenience.
- No dependency direction, module boundary, authoritative architecture rule, shared contract, or
  build structure changes. If implementation requires one, stop and report the exact conflict.

## Package impact

Existing public package changed:

- `io.github.pho001.synaptik.engine` remains the deliberate small ordinary facade and adds only
  the two `Engine.forward(...)` methods.

Existing package-private machinery changed:

- `AdvancedEngine` gains only narrow ordinary one-shot orchestration and/or shared ungated helpers
  needed to hold one lifecycle admission across the complete sequence. This adds no advanced
  public method.

Packages added or moved: none.

Type placement:

- `io.github.pho001.synaptik.engine.Engine` owns both overloads because Engine already owns
  standard composition, ordinary lifecycle entry, and failure/lifetime coordination.
- `io.github.pho001.synaptik.engine.AdvancedEngine` remains the private lifecycle owner used by
  ordinary Engine; it is the only existing place permitted to coordinate the single admission.
- Existing `io.github.pho001.synaptik.engine.HostTensorValue` is the result element because it
  already owns detached immutable canonical host bytes and no additional carrier fact is needed.

## Affected files

Exact implementation allowlist:

Engine production and Javadoc (three paths):

1. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java`.
2. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java`.
3. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/package-info.java`.

Focused tests (four paths):

4. `modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineTypedLifecycleTest.java`.
5. `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/EngineTypedPublicShapeTest.java`.
6. `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/AdvancedEnginePublicShapeTest.java`.
7. `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineTypedLifecycleIntegrationTest.java`.

Explanatory documentation (four paths):

8. `docs/api/public-api.md`.
9. `docs/api/compile-api.md`.
10. `docs/api/runtime-api.md`.
11. `docs/glossary.md`.

Planning/status (three paths):

12. This task.
13. `docs/planning/modules/engine/master-plan.md`.
14. `docs/planning/roadmap.md`.

Review without modification: `RunResult`, `HostTensorValue`, `CompiledGraph`,
`PreparedExecution`, the package-private backend composition seam, advanced public types, existing
standard/advanced lifecycle tests and integrations other than the four allowlisted test paths,
Training and Tensor API references, all architecture/ADR files, Engine and integration build
files, the focused architecture test, every inward module, backend conformance, and later task
rows/specifications. The documentation pass must record reasoned no-change conclusions.

## Maximum scope

Implementation may create or modify at most the exact fourteen paths above: three Engine
production/Javadoc, four focused tests, four explanatory documents, and three planning/status
paths. Unused capacity in one category is not permission to edit another path. No production type,
test helper file, module, package, dependency, build file, architecture file, or fifteenth path is
authorized. If another path is required, stop and replan before editing.

## Acceptance criteria

- `Engine` exposes exactly the two specified `forward` overloads in addition to its current public
  surface; all generic signatures, return types, modifiers, constructor visibility, and ordinary
  import boundaries are enforced by the distinct-package public-shape test.
- `AdvancedEnginePublicShapeTest` changes only its exact ordinary `Engine` method-name inventory
  by adding the two `forward` entries; all advanced-surface assertions, including every
  advanced-type, constructor, field, modifier, lifecycle, and forbidden-signature assertion,
  remain unchanged.
- No new public type exists. The single overload returns one existing `HostTensorValue`; the
  ordered overload returns one immutable output-ordered list of existing values.
- Both overloads use one admission and exactly one compile, prepare, run, and result cleanup. The
  multi-output overload performs exactly one materialization per forward occurrence in increasing
  order; the single overload uses the same internal implementation without nested public calls.
- Caller output/input collections are snapshotted and not retained or mutated. Output identity,
  input `TensorId` binding, descriptor checks, association snapshots, borrow order, shared-storage
  handling, caller ownership, and CPU validation remain exactly as specified.
- Complete publication invariants and the aggregate byte count are validated before copying.
  Static/resolved, zero-element, individual JVM ceiling, checked sum overflow, exact-limit
  success, total-limit failure, and no-copy-on-preflight-failure cases are covered.
- Returned list/value ordering follows the requested output list and exact publication order.
  Distinct occurrences that alias inwardly are copied independently and produce distinct values;
  no cache, deduplication, search, reorder, or reuse occurs.
- The temporary result and wrappers always close before normal return. Compile, prepare, binding,
  run, publication-defense, size-preflight, mid-copy, value-construction, and cleanup failures
  preserve the exact primary/suppressed rules and publish no partial return.
- Bounded-latch tests prove Engine close waits for an admitted one-shot call, a successful admitted
  call may return detached values after closure begins, a losing call observes closed precedence,
  and concurrent calls have isolated runs without deadlock or leaked registered results.
- The reusable ordinary compile/prepare/run/materialize lifecycle, result registration and close
  order, advanced API, current CPU-only limitations, and existing public behavior remain passing.
- Real `Engine.standard()` integration executes one and two supported resolved static CONTIGUOUS
  outputs from explicit caller-backed FLOAT32 inputs, verifies output order and canonical bytes,
  proves the aggregate exact-limit/failure boundary, and confirms caller arenas remain caller-
  owned. It makes no unsupported ADD, zero-node, cross-backend, dynamic, or layout claim.
- Meaningful Javadocs document purpose, arguments, nullability, aggregate byte units and scope,
  return immutability/order/ownership, lifecycle, concurrency, failures, CPU limits, and the reuse
  tradeoff. Package Javadoc distinguishes one-shot from reusable lifecycle use.
- The documentation-focused pass updates Public, Compile, Runtime, and glossary wording from
  planned to current, includes a complete one-shot example, and does not imply backward,
  autotuning, transfer, cross-backend, Tensor execution, caching, or prepared reuse.
- No non-allowlisted path changes. Engine 0005 is `Complete` in this task, master plan, and
  roadmap. Engine 0006–0008 remain `Draft`, and no 0006-or-later Engine task specification
  exists.

## Tests / validation

This is task-tier Engine validation. During implementation run focused tests as needed, then after
executable Java stabilizes record one final pass of:

```bash
./gradlew :modules:engine:test
./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.EngineTypedLifecycleIntegrationTest
./gradlew :testing:architecture-tests:test --tests io.github.pho001.synaptik.testing.architecture.EngineCompositionContractTest
```

The implementation context must report exact suite/test counts and hand that evidence to the
documentation context. Do not run Runtime or CPU suites: their completed 0015/0010G behavior is
unchanged. Do not run backend conformance because no shared backend contract changes. Repository-
wide tests remain deferred to Engine 0008/CI because no dependency, architecture, shared build,
or second-production-module contract changes.

The distinct clean documentation-focused pass reuses successful Java evidence unless it changes
executable Java behavior, then runs:

```bash
./gradlew :modules:engine:javadoc
git diff --check
```

Also validate:

- generated Javadoc for the two overloads and package lifecycle description;
- both exact Engine public-surface inventories, with `AdvancedEnginePublicShapeTest` changed only
  to add the two `forward` method names, plus a distinct-package fixture using only Engine, Model,
  and JDK types;
- changed Markdown local targets and effective anchors, heading order, balanced fences, trailing
  whitespace, LF line endings, and final newlines;
- current-versus-planned claims and glossary terminology;
- the exact fourteen-path implementation allowlist and absence of build/dependency, architecture,
  inward-module, later-specification, or unrelated changes;
- task/master/roadmap status agreement, 0005 Complete, 0006–0008 Draft, and no Engine 0006-or-
  later detailed task; and
- preservation hashes for every dirty path that predated implementation.

Do not run Java tests during this planning task.

## Documentation handoff

Primary profiles are Planning for this task/master/roadmap, API/Javadoc for the changed Engine
methods and API references, General for glossary terminology, and Example for the ordinary
one-shot walkthrough.

After executable code and tests stabilize, hand a distinct clean documentation-focused context:

- this task and the exact goal/public API;
- the final Engine and test diff;
- exact implementation test commands, counts, and outcomes;
- the dirty-worktree baseline and preservation hashes;
- the one-admission lifecycle, explicit-input ownership, aggregate-limit, publication-order,
  cleanup/suppression, no-cache/no-reuse, and CPU-only decisions;
- completed Engine 0003–0004, Runtime 0015, and CPU 0010G contracts and unchanged evidence; and
- the four authorized explanatory paths plus the applicable documentation profiles.

The pass must independently inspect final behavior and tests. It must explain **one-shot forward
execution** at first use as fresh per-call compilation, preparation, execution, complete forward-
publication materialization, and cleanup. Add or revise a glossary entry only for that reusable
distinction; do not create synonyms or imply Tensor ownership.

The complete example must use `Engine.standard()`, one or more supported static resolved
CONTIGUOUS outputs, explicit caller-backed input Tensors, an explicit aggregate byte limit, and a
try-with-resources Engine. It must show that no `RunResult` or prepared handle escapes, returned
values are ordered/detached, and caller storage remains owned by the caller. It must state that
each call compiles and prepares afresh and recommend the lower-level lifecycle for repeated runs
or selective materialization.

Training API and Tensor API require reasoned no-change conclusions: this forward-only capability
does not change gradient/training behavior or place execution on Tensor. Architecture pages and
ADRs require no edit because the implementation realizes the existing Engine composition role.
The documentation context must not rerun stable Java tests unless it changes executable behavior
or records a concrete risk. It records all reviewed paths, changes, commands, results, and no-
change conclusions before this task may become `Complete`.

## Dependencies

- Engine 0001 advanced lifecycle foundation — Complete.
- Engine 0002 standard CPU-only composition — Complete.
- Engine 0003 ordinary typed logical input binding, publication occurrence mapping, and reusable
  result lifecycle — Complete.
- Engine 0004 exact bounded detached host materialization — Complete.
- Runtime 0015 exact borrowed publication-representation access under the result lease — Complete.
- CPU 0010G canonical caller-owned host snapshot export — Complete.
- Current Model descriptor/storage contracts and the current Compiler/Prepare/Runtime/CPU ordinary
  path — implemented and sufficient.

All prerequisites are complete. No architecture decision or material user choice remains for this
bounded design.

## Follow-up tasks

- Engine 0006 remains Draft: add only the separately specified scalar-objective backward
  convenience with explicit targets after 0005 is Complete. Preserve the current explicit-seed
  compile overload and advanced full-request path.
- Engine 0007 remains Draft: optional Config 0006A and tools/tuning 0001 composition before
  preparation. One-shot forward in this task is always the current deterministic untuned path.
- Engine 0008 remains Draft: lifecycle capability checkpoint after 0001–0006.
- Cross-backend materialization, mixed-backend schedule composition, output streaming, and
  compile/prepare caching require separately demonstrated owner contracts; no detailed task is
  created here.

## Architecture impact

Expected impact: None.

The task adds ordinary Engine orchestration over existing compiler, preparation, Runtime lease,
and CPU copy contracts. Engine already owns public lifecycle composition and already depends on
all named inward contracts. It adds no dependency, changes no owner, and exposes no inward type.
`Tensor` remains execution-free, Runtime remains concrete-backend-independent, and CPU remains
Engine-independent. `ARCHITECTURE.md`, focused architecture pages, ADRs, dependency rules, and
architecture tests require no semantic update. Stop and report if implementation evidence
contradicts this conclusion.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in /Users/phujka/IdeaProjects/Synaptik on Engine task 0005. Do not use GSD,
commit, or push.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, the Engine master plan, and
docs/planning/modules/engine/tasks/0005-one-shot-forward-convenience.md in full. Read the final
Engine 0001–0004, Runtime 0015, and CPU 0010G contracts and inspect all current Engine source,
tests, Javadocs, API documentation, and the dirty worktree before editing.

Implement task 0005 exactly within its fourteen-path allowlist. Add only the two ordinary
Engine.forward overloads and one-admission internal orchestration over the existing forward-only
compile, typed binding, exact ordered publication materialization, aggregate byte bound, and
result cleanup contracts. Preserve the reusable lifecycle and every pre-existing dirty change.
Do not add Tensor execution/backward, implicit seeds/targets, task 0006 behavior, a request/result
type, caching/reuse, tuning, cross-backend abstraction, transfer/fallback, build/dependency or
architecture changes, or later task specifications. Stop on architecture or scope conflict.

Use apply_patch. Run the task-tier Java validation once after executable code stabilizes. Then
hand the same diff and exact evidence to a distinct clean documentation-focused context, which
must follow the documentation rules and finalize the authorized Javadocs, explanatory docs,
glossary impact, example, and validation without repeating successful Java suites unless
executable behavior changes. Mark Complete only after every implementation and documentation
gate passes.
```

## Local decisions

- Name the operation `forward` on ordinary `Engine`. The name states the graph scope without
  implying prepared-handle reuse or placing execution on Tensor.
- Provide both one-output and ordered-list overloads. They share one internal implementation;
  neither introduces a policy or request abstraction.
- Return the existing `HostTensorValue` and immutable `List<HostTensorValue>`. A new carrier would
  duplicate already self-describing detached values and add no lifecycle authority.
- Use one aggregate `long maximumTotalBytes`. Aggregate accounting prevents several individually
  permitted outputs from bypassing the caller's total returned-payload budget, while the existing
  per-value JVM ceiling remains intact.
- Complete aggregate descriptor/count validation before any CPU copy, then pass each exact
  individual byte count to the existing materializer as its tight per-copy limit.
- Keep the entire lifecycle under one existing Engine admission and keep the temporary result
  local and unregistered. Detached values may safely return after concurrent Engine closure has
  begun, while Engine close waits for cleanup before closing CPU.
- Create no cache. Repeated workloads should use the explicit reusable lower-level lifecycle,
  whose API and ownership remain unchanged.

## Known limitations

- Current execution and materialization are CPU-only and accept exactly the operation, static-
  Shape, resolved-layout, and one-non-empty-maximal-partition domain supported by the current CPU
  composition. Compile success remains no guarantee of prepare or run success.
- Every invocation compiles and prepares from scratch. It is convenience for one use, not an
  efficient repeated-execution API.
- Every output is materialized eagerly before return. There is no selective, lazy, streamed,
  chunked, mapped, caller-destination, typed-array, or Tensor result form.
- The aggregate limit bounds canonical returned payload bytes only, not peak or total memory, and
  each payload remains subject to `Integer.MAX_VALUE` and available heap.
- Caller input associations are snapshotted per Tensor, not atomically across inputs, and caller
  payload mutation/closure is not synchronized by Engine.
- No backward, implicit target/seed, autotuning, cross-backend transfer, mixed-backend composition,
  zero-node preparation, or output cache is included.

## Validation evidence

- Planning context `01a0a527-b8d0-78e1-9c56-f39f73a91e26` read the required architecture,
  planning, completed Engine 0001–0004, Runtime 0015, CPU 0010G, Engine source/test/integration,
  Model descriptor/storage, API, glossary, and documentation-profile contracts. No architecture,
  dependency, inward-contract, or material user-choice blocker was found.
- Recorded SHA-256 preservation hashes for every pre-existing dirty path before planning edits.
- Java tests were not run, as required for this planning-only task.
- Implementation context `01a0a533-6b58-7d70-9cb6-3bcc9b07ee6c` implemented the one-admission
  lifecycle in the original six paths and corrected its initial cleanup-test failures before
  reporting stable executable behavior.
- Scope-correction context `01a0a53c-4ea1-72d2-88d4-79c03a92a585` changed only this specification
  from thirteen to fourteen authorized paths so the stale
  `AdvancedEnginePublicShapeTest` ordinary-method inventory could be corrected.
- Completion context `01a0a540-aad0-7551-856b-b10109f56a68` changed exactly the two expected
  `forward` names in that inventory. Its final `./gradlew :modules:engine:test` passed five suites
  and 33 tests; the focused integration run passed one suite and five tests; the focused
  architecture run passed one suite and one test; and `git diff --check` passed.
- Clean documentation context `01a0a545-5c95-7cc0-84b8-ba3dd02914a2` independently inspected the
  final source, public signatures, focused tests, generated behavior, architecture/lifecycle
  contracts, API references, glossary, and planning state. It changed Javadocs only in the three
  authorized Engine production paths and changed only the four authorized explanatory documents
  plus this task, the Engine master plan, and the roadmap.
- That documentation context ran `./gradlew :modules:engine:javadoc` successfully and inspected
  generated `Engine`, `AdvancedEngine`, and package pages for both overloads, lifecycle admission,
  aggregate preflight, cleanup, and reusable-path guidance. Exact `javap -public` inspection
  confirmed both required signatures and no advanced-surface addition. A Java 26 distinct-package
  in-memory-compiled fixture, using only Engine, Model, and JDK public types and launched with
  `--add-modules jdk.incubator.vector`, reflected both generic signatures and ran the singleton
  and ordered overloads through `Engine.standard()`; it verified output order, immutable list
  behavior, and detached post-close access.
- Before documentation edits, comment-stripped executable Java hashes were
  `78b5924d1bf2a13ff49a3acf5d70e1ae70693d2341b0b06bc72f370f326bd0f7` for `Engine.java` and
  `32a96e18deb0cf3e2d727ddddb7a410f7d1080dee51b5e00145fe4390d5e6623` for
  `AdvancedEngine.java`. The combined SHA-256 preservation digest for every dirty path outside
  the ten documentation-finalization paths was
  `4c999c62ba96d91ac97cab8c245b6b7dd87f4bb9b84fafe1151055b690fb87c9`.
- Final rechecking reproduced both executable hashes and the non-finalization dirty-path digest
  exactly. The seven changed Markdown paths passed local-target/anchor, effective-heading,
  balanced-fence, trailing-whitespace, LF, and final-newline validation. All fourteen task paths
  were present and no fifteenth task path was attributed; task/master/roadmap status agreed on
  0005 Complete, 0006–0008 remained Draft, no later Engine task specification existed, and final
  `git diff --check` passed.
- Java test suites were not repeated in the documentation context because executable Java tokens
  did not change and independent inspection identified no stale-evidence risk.

## Implementation notes

`Engine` now exposes exactly the requested singleton and ordered `forward(...)` overloads.
Package-private ordinary orchestration in `AdvancedEngine` shares the established compile,
prepare, logical-binding, publication, materialization, and cleanup operations while keeping one
admission across the whole one-shot call. The four focused tests cover the exact public surface,
ordered/static CPU success, aggregate bounds, failure/cleanup behavior, and lifecycle concurrency.

Documentation now defines one-shot forward execution at first use, records explicit inputs,
immutable detached ordered results, aggregate canonical-payload scope, CPU/static/resolved limits,
cleanup and concurrency behavior, and the absence of caching, reuse, backward, Tensor execution,
tuning, and cross-backend transfer. The Public API includes a complete supported
`Engine.standard()` example and directs repeated or selective use to the lower-level lifecycle.

Independent no-change review concluded that `RunResult` and `HostTensorValue` already state the
exact lease, occurrence authentication, detached-byte, and post-close contracts; `CompiledGraph`
and `PreparedExecution` already state owner binding and reuse; and `AdvancedCompiledGraph`,
`AdvancedPreparedExecution`, `AdvancedRunResult`, and the package-private composition seam already
state the unchanged advanced boundary. The four focused tests plus standard and advanced Engine
tests required no documentation-pass edits because their final assertions match the implemented
surface and handed-off evidence.

Training API requires no change because the task is forward-only and adds no gradient,
optimizer, parameter-update, or session contract. Tensor API requires no change because execution
remains on Engine and Tensor remains expression/storage metadata. Architecture, ADR 0011,
lifecycle/module/dependency/runtime-prepare-backend pages, inward Runtime/CPU contracts,
build/dependencies, architecture tests, backend conformance, and later Engine tasks require no
change because no owner, dependency, shared contract, backend behavior, or planned later scope
changed.

## Completion summary

Completed changes: added the two exact ordinary one-shot forward overloads and their private
one-admission orchestration; completed focused semantic, public-shape, integration, cleanup, and
concurrency coverage; finalized all affected Javadocs, Public/Compile/Runtime API explanations,
glossary terminology, the complete supported example, and synchronized planning status.

Files changed or created: exactly the fourteen paths listed in **Affected files**. No executable
Java or test file changed during the final documentation pass, and every pre-existing dirty
non-finalization path was preserved.

Tests and validation: reused the final 33 Engine tests, five focused integration tests, and one
focused architecture test from completion context `01a0a540-aad0-7551-856b-b10109f56a68`; the
documentation pass completed Engine Javadoc, generated-page inspection, `javap`, reflection, the
distinct-package Java 26 fixture, Markdown/link/anchor/fence/format validation, executable-token
hashes, fourteen-path scope and status checks, dirty-path preservation, and whitespace validation.

Unresolved issues: none for task 0005.

Required follow-up: none. Draft Engine 0006–0008 remain separately planned work.

Status: Complete
