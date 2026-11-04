# Task 0005: One-Shot Forward Convenience

## Status

Complete

## Goal

Add the smallest ordinary [Engine](../../../../glossary.md#term-engine)-owned convenience that executes one forward request from one
[Tensor](../../../../glossary.md#tensor) output or an ordered non-empty output list plus explicit logical input Tensors and returns
[detached](../../../../glossary.md#detached-model-results) [immutable](../../../../glossary.md#term-immutable) [host](../../../../glossary.md#term-host) values. One [synchronous](../../../../glossary.md#term-synchronous) call owns the complete temporary
`compile -> prepare -> run -> materialize every exact forward publication -> close result`
[lifecycle](../../../../glossary.md#lifecycle) while preserving the reusable `CompiledGraph`, [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), [`RunResult`](../../../../glossary.md#run-result--runresult), and
per-[publication](../../../../glossary.md#term-publication) [materialization](../../../../glossary.md#materialization) APIs for callers that need repeated runs or selective output
access.

The convenience uses one caller-supplied non-negative aggregate byte limit for all returned
[canonical](../../../../glossary.md#term-canonical-form) payloads. It neither caches nor reuses [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) or preparation, and it does not add
execution, backward, [gradient](../../../../glossary.md#term-gradient), or [Runtime](../../../../glossary.md#term-runtime) state to [Model](../../../../glossary.md#term-model) `Tensor`.

## Required reading

Read these files and contracts in full in the clean implementation context rather than relying
only on this specification:

- root `AGENTS.md`, [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), and the
  [current architecture plan](../../../../architecture/current-architecture-plan.md);
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>,
  [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>,
  [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>,
  and [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>;
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>, [roadmap](../../../roadmap.md), this task, the
  [Engine master plan](../master-plan.md), and completed [Engine](../../../../glossary.md#term-engine) tasks
  [0001](0001-advanced-composition-and-representation-level-lifecycle-foundation.md),
  [0002](0002-standard-built-in-composition.md),
  [0003](0003-typed-logical-input-binding-and-published-result-access.md), and
  [0004](0004-explicit-host-materialization-boundary.md);
- completed [Runtime](../../../../glossary.md#term-runtime) task
  [0015](../../runtime/tasks/0015-leased-publication-representation-access.md) and [CPU](../../../../glossary.md#term-cpu) task
  [0010G](../../../backends/cpu/tasks/0010g-canonical-caller-owned-host-snapshot-export.md),
  including their final source, tests, Javadocs, and completion evidence;
- every current Engine production and test file, all three ordinary Engine integration tests,
  and the focused Engine architecture test;<span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- current [Model](../../../../glossary.md#term-model) [`Tensor`](../../../../glossary.md#tensor), [`TensorId`](../../../../glossary.md#tensorid), `TensorDescriptor`, [`DataType`](../../../../glossary.md#data-type--datatype), [`Shape`](../../../../glossary.md#shape),
  `LayoutDescriptor`, `HostTensorStorage`, and `MemorySegmentStorage` contracts;
- the current public, compile, runtime, training, Tensor, and glossary sections that describe
  Engine execution, input binding, publication, materialization, ownership, and planned one-shot
  behavior; and<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [binding glossary](../../../../glossary.md#term-binding); [publication glossary](../../../../glossary.md#term-publication); [materialization glossary](../../../../glossary.md#materialization); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md) plus the General,
  API/Javadoc, Planning, and Example profiles under<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
  `docs/developer-guide/documentation/`.

`ARCHITECTURE.md` is authoritative. Stop before editing if the implementation would require an
architecture change, a new [dependency direction](../../../../glossary.md#java-platform-and-build-tools), or an inward public-contract change.

## [Scope](../../../../glossary.md#term-scope)

- Add two ordinary `Engine.forward(...)` overloads: one for exactly one output and one for an
  ordered non-empty list of outputs.
- Require explicit logical input [Tensors](../../../../glossary.md#tensor) on both overloads. Use the same final [Compiler](../../../../glossary.md#term-compiler)-owned
  [`TensorId`](../../../../glossary.md#tensorid) binding inventory, descriptor equality, association snapshot, storage validation,
  borrow order, and caller ownership rules as the reusable task-0003 run path.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [descriptor glossary](../../../../glossary.md#term-descriptor); [snapshot glossary](../../../../glossary.md#term-snapshot); [validation glossary](../../../../glossary.md#term-validation); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [run glossary](../../../../glossary.md#term-run))</span>
- Interpret one `long maximumTotalBytes` as a bound on the checked sum of all returned canonical
  payload byte counts. Validate the complete result set before the first physical copy.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>
- Keep one Engine lifecycle admission open across validation, compilation, preparation, run,
  aggregate-size preflight, ordered materialization, result cleanup, and return readiness.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [admission glossary](../../../../glossary.md#term-admission); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [preflight glossary](../../../../glossary.md#term-preflight); [materialization glossary](../../../../glossary.md#materialization))</span>
- [Compile](../../../../glossary.md#compile) once with the existing ordinary `FORWARD_ONLY` fixed settings, prepare once, run once,
  materialize each exact forward publication once in publication order, and close the temporary
  result once before returning.<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare); [publication glossary](../../../../glossary.md#term-publication))</span>
- Return only existing [detached](../../../../glossary.md#detached-model-results) [immutable](../../../../glossary.md#term-immutable) [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue) instances. The multi-output overload
  returns an immutable ordered `List<HostTensorValue>`; the single-output overload returns its one
  element directly.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- Share private/package-private ungated lifecycle helpers with the reusable lower-level path so
  logical binding, publication construction, materialization, and cleanup semantics do not drift.
- Add focused public-shape, semantic, failure, ownership, cleanup, concurrency, and real CPU
  integration coverage, including the existing advanced-surface inventory test whose exact
  ordinary <span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [concurrency glossary](../../../../glossary.md#term-concurrency); [CPU glossary](../../../../glossary.md#term-cpu))</span>[`Engine`](../../../../glossary.md#term-engine) method-name list must admit the two new overloads.
- Finalize affected Javadocs and explanatory documentation through a distinct clean
  documentation-focused pass in the same overall change.

## Exact public [API](../../../../glossary.md#term-application-programming-interface-api)

Add exactly these declarations to the existing public final [`Engine`](../../../../glossary.md#term-engine):

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
options object, result [carrier](../../../../glossary.md#term-storage-carrier), or overload is authorized. Every other public Engine and advanced
Engine declaration remains source-compatible and unchanged.

The single-output overload is justified because the dominant one-output case should not require
constructing and unwrapping two lists. It is semantically the exact [singleton](../../../../glossary.md#term-singleton) [specialization](../../../../glossary.md#term-specialization) of
the ordered overload, not a different [lifecycle](../../../../glossary.md#lifecycle) or policy. The ordered overload is required
because [Compiler](../../../../glossary.md#term-compiler) [publication](../../../../glossary.md#term-publication) order and current multi-output expression boundaries are already
first-class and executing outputs separately could repeat shared [graph](../../../../glossary.md#graph) work and lose [occurrence](../../../../glossary.md#term-occurrence)
order. `List<HostTensorValue>` is sufficient because every returned value is already [immutable](../../../../glossary.md#term-immutable),
[detached](../../../../glossary.md#detached-model-results), and self-describing; another carrier or request abstraction would add no [ownership](../../../../glossary.md#term-ownership-meanings),
role, or lifecycle fact.

`maximumTotalBytes` is a primitive `long` rather than a new limit type because this task has one
quantity, one unit, and no optional/default policy. The name and [Javadoc](../../../../glossary.md#term-javadoc) must state that the limit
is aggregate across output payloads, not per output, input storage, intermediate storage, [Runtime](../../../../glossary.md#term-runtime)
allocation, or peak memory.

Ordinary public/protected [signatures](../../../../glossary.md#term-signature) continue to expose only Engine, [Model](../../../../glossary.md#term-model), and [JDK](../../../../glossary.md#term-java-development-kit-jdk) types. They
must not expose Compiler, [Prepare](../../../../glossary.md#prepare), Runtime, [Planning](../../../../glossary.md#term-planning), [backend](../../../../glossary.md#backend)-contract, [CPU](../../../../glossary.md#term-cpu), another backend,
advanced handles, `.internal`, representations, storage objects, [arenas](../../../../glossary.md#term-arena), memory segments, slots,
[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), or [backend identity](../../../../glossary.md#backend-identity--backendid).

## Forward request semantics

### [Arguments](../../../../glossary.md#term-argument) and snapshots

After [Engine](../../../../glossary.md#term-engine) [admission](../../../../glossary.md#term-admission), the single-output overload validates in this order:

1. reject null `output` with `NullPointerException("output")`;
2. reject null `inputs`, then a null input [element](../../../../glossary.md#term-element) in increasing index order, using the existing
   `inputs` and `inputs[index]` messages;
3. reject `maximumTotalBytes < 0` with
   `IllegalArgumentException("maximumTotalBytes must be non-negative: " + maximumTotalBytes)`.

After Engine admission, the ordered overload validates in this order:

1. reject null `outputs` with `NullPointerException("outputs")`;
2. [snapshot](../../../../glossary.md#term-snapshot) its elements in encounter order and reject the first null as `outputs[index]`;
3. reject an empty output list with `IllegalArgumentException("outputs must not be empty")`;
4. reject repeated exact Tensor object identity with the existing indexed duplicate form used by
   ordinary compilation, replacing the list name with <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>`outputs`;
5. reject null `inputs`, then the first null input element in increasing index order;
6. reject a negative aggregate limit with the exact single-output message above.

Both overloads snapshot caller list structure before inward work, retain no caller collection,
and never mutate it. The single overload constructs one private [singleton](../../../../glossary.md#term-singleton) snapshot and enters the
same ungated implementation as the multi-output form; it must not invoke the public multi-output
overload or acquire a second admission. Inputs remain caller-supplied ordinary [Tensors](../../../../glossary.md#tensor) and may be
listed in any order. Empty inputs are locally valid and reach the established exact [binding](../../../../glossary.md#term-binding)
[validation](../../../../glossary.md#term-validation); Engine does not invent an input from output [provenance](../../../../glossary.md#provenance) or storage.

### Exact internal sequence

For each admitted call, execute this sequence exactly once and in this order:

1. validate and [snapshot](../../../../glossary.md#term-snapshot) the public [arguments](../../../../glossary.md#term-argument) above;
2. compile the ordered output snapshot through the existing ordinary forward-only Compiler path
   with <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>`GraphOptimizationConfig.standard()`, `BackendIntent.unconstrained()`, and
   `PartitionScoringConfig.neutral()`;
3. construct the same [owner](../../../../glossary.md#term-owner)-bound [compile](../../../../glossary.md#compile) [metadata](../../../../glossary.md#term-metadata) used by `Engine.compile(outputs)`;
4. [prepare](../../../../glossary.md#prepare) that exact compile result once through the current owned [CPU](../../../../glossary.md#term-cpu) composition;
5. run that exact prepared result once with the explicit Tensor input snapshot, using the
   established task-0003 logical binding, storage snapshot, borrowing, Runtime execution,
   publication-count defense, and wrapper ownership rules;<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [binding glossary](../../../../glossary.md#term-binding); [borrowing glossary](../../../../glossary.md#term-borrowing); [publication glossary](../../../../glossary.md#term-publication); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
6. before any physical copy, validate the complete ordered forward-publication set and its
   aggregate canonical byte count as specified below;<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
7. materialize every exact publication object once in increasing publication index order through
   the existing task-0004 descriptor/representation/CPU-copy path;<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
8. build the [immutable](../../../../glossary.md#term-immutable) ordered [detached](../../../../glossary.md#detached-model-results)-value list, or select its sole value for the single form;
9. close the temporary [Runtime](../../../../glossary.md#term-runtime) result [lease](../../../../glossary.md#term-lease) and [Engine](../../../../glossary.md#term-engine)-created borrow wrappers once; and
10. only after successful cleanup, release Engine admission and return the detached value or
    immutable list.<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release); [detached value glossary](../../../../glossary.md#detached-model-results))</span>

The one-shot path must not call public `compile`, `prepare`, [`run`](../../../../glossary.md#term-run), or `materialize` methods from
inside its [admission](../../../../glossary.md#term-admission) because those methods acquire their own admissions and, for results, the
ordinary reusable registration [lifecycle](../../../../glossary.md#lifecycle). Refactor narrowly into shared ungated operations or an
equivalent package-private implementation that preserves the reusable methods' exact observable
semantics. Do not introduce a [callback](../../../../glossary.md#term-callback) executor, generic request/[facade](../../../../glossary.md#term-facade)/manager, [service locator](../../../../glossary.md#term-service-locator),
second lifecycle gate, or second result [registry](../../../../glossary.md#term-registry).

The temporary compile and prepared handles own no closeable resource. The temporary result is
owned solely by the admitted one-shot call and must never be returned or left registered for later
Engine cleanup. Existing reusable handles and results keep their current owner-bound reuse and
explicit-close behavior unchanged.

## [Publication](../../../../glossary.md#term-publication) and byte-limit semantics

The compiled request is forward-only, so the successful result must contain exactly one forward
[publication occurrence](../../../../glossary.md#publication-occurrence) for each requested output and no [gradient](../../../../glossary.md#term-gradient) [occurrence](../../../../glossary.md#term-occurrence). Before copying,
validate all of these invariants in increasing index order:

- result count and publication-list size equal `outputs.size()`;
- publication `index()` equals its dense list index;
- publication `role()` is `FORWARD`;
- publication `tensorId()` equals the corresponding requested output's [`TensorId`](../../../../glossary.md#tensorid);
- `derivativeOrder()` and `targetIndex()` are empty; and
- the final publication [descriptor](../../../../glossary.md#term-descriptor) is the exact descriptor used for [host](../../../../glossary.md#term-host) [materialization](../../../../glossary.md#materialization).

An impossible mismatch is `IllegalStateException` with an index/count-specific diagnostic and
causes ordinary cleanup. Do not repair, reorder, search by [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings), merge [aliases](../../../../glossary.md#aliasing-and-arena), or
substitute a different occurrence. The returned list position is therefore exactly the requested
output position and [Runtime](../../../../glossary.md#term-runtime) publication position.

For every publication in order, perform the same descriptor-derived [preflight](../../../../glossary.md#term-preflight) used by explicit
materialization:

- require a fully static [Shape](../../../../glossary.md#shape);
- require a resolved [layout](../../../../glossary.md#layout);
- compute logical [element](../../../../glossary.md#term-element) count with the [Model](../../../../glossary.md#term-model) zero-before-product rule;
- compute [canonical](../../../../glossary.md#term-canonical-form) byte count with checked `long` multiplication; and
- require that individual byte count is at most `Integer.MAX_VALUE`.

Accumulate individual canonical byte counts in publication order with `Math.addExact`. A numeric
[overflow](../../../../glossary.md#term-overflow) remains `ArithmeticException`. Zero-element outputs contribute zero. Only after all
descriptors and the complete sum have passed, compare the sum with `maximumTotalBytes`. If it is
larger, throw:

```text
total canonical byte count exceeds maximumTotalBytes: required=<sum>, maximum=<limit>
```

This complete preflight happens after successful [run](../../../../glossary.md#term-run) construction but before the first physical
copy. Therefore a limit, static-Shape, resolved-layout, [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)-array-ceiling, or aggregate-overflow
failure performs zero [CPU](../../../../glossary.md#term-cpu) host-copy calls and then closes the temporary result. It does not avoid
the [compile](../../../../glossary.md#compile), [prepare](../../../../glossary.md#prepare), or run work already required to obtain and validate the exact publication
set.

After preflight, call the established exact-occurrence materialization implementation once per
publication in order. Supply that occurrence's already checked individual canonical byte count as
its per-copy `maximumBytes`; this is the tight bound required for that [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) and cannot weaken
the aggregate caller limit. Each returned [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue) must match the preflighted type,
Shape, element count, and byte count. Repeated outputs are already rejected by exact Tensor
[identity](../../../../glossary.md#term-reference-identity), but distinct output occurrences may alias one inward representation; they still produce
distinct uncached snapshots and distinct `HostTensorValue` objects.

The aggregate limit covers only the sum of returned canonical payload lengths. It does not claim
to bound [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) data, prepared recipes, input storage, Runtime [buffers](../../../../glossary.md#term-buffer)/workspaces, temporary
CPU access, list/object overhead, defensive cloning, peak [heap](../../../../glossary.md#term-heap) use, or memory held before cleanup.
Every `HostTensorValue` retains task 0004's `Integer.MAX_VALUE` byte-array ceiling and available-
heap limitation.

## Input [ownership](../../../../glossary.md#term-ownership-meanings) and storage [lifetime](../../../../glossary.md#term-lifetime)

- Logical input [validation](../../../../glossary.md#term-validation) is exactly the reusable task-0003 contract: every final bindable
  [`TensorId`](../../../../glossary.md#tensorid) appears once, unexpected/duplicate/missing identities fail, descriptors compare by
  complete value equality, and a descriptor-equal different identity is not a replacement.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Only after complete logical validation, read each required [Tensor](../../../../glossary.md#tensor)'s synchronized
  `hostStorage()` association once in final Compiler binding order and retain that exact storage
  for the call.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [binding glossary](../../../../glossary.md#term-binding))</span>
- Validate type, resolved-layout span where present, liveness, and current-thread accessibility
  before borrowing. CPU cold binding remains an independent inward validation.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [liveness glossary](../../../../glossary.md#term-liveness); [borrowing glossary](../../../../glossary.md#term-borrowing); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Distinct logical inputs sharing one storage object receive distinct fresh borrow wrappers.
  Engine does not deduplicate wrappers, copy input bytes, infer layout, convert values, or write
  back.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- Caller storage and its arena remain caller-owned. They must stay live, accessible where used,
  and free from conflicting mutation through synchronous method completion, including result and
  wrapper cleanup. Because the temporary result is closed before return, task 0003's longer
  caller-storage-through-result-close obligation ends when this method returns or throws after
  cleanup.<span data-glossary-navigation="true"> ([arena glossary](../../../../glossary.md#term-arena); [synchronous glossary](../../../../glossary.md#term-synchronous))</span>
- Replacing or clearing a Tensor association after its per-Tensor snapshot cannot redirect the
  active call. The snapshots are not atomic across inputs and provide no payload memory fence,
  pinning, or protection from caller races.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>

## Failure, cleanup, and [lifecycle](../../../../glossary.md#lifecycle) semantics

### Failure precedence

1. [Engine](../../../../glossary.md#term-engine) [admission](../../../../glossary.md#term-admission) is first. Once closure has begun, preserve
   `IllegalStateException("advanced engine is closed")` before every argument, limit, compile,
   prepare, input, descriptor, publication, Runtime, CPU, or cleanup precondition.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
2. Public [argument](../../../../glossary.md#term-argument) [validation](../../../../glossary.md#term-validation) follows the exact overload-specific order above.
3. Compiler failures precede Prepare failures; Prepare failures precede logical-input and run
   failures; run/publication failures precede aggregate materialization preflight; preflight
   failures precede CPU copying.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [preflight glossary](../../../../glossary.md#term-preflight))</span>
4. Publication preflight is complete before copying. During copying, the first failure in
   publication order stops further materialization.
5. The temporary result is then closed exactly once. If a primary failure already exists, a
   distinct cleanup `RuntimeException` or `Error` is suppressed on that exact primary object. A
   cleanup failure equal by reference to the primary is not self-suppressed. If no earlier
   failure exists, the first cleanup failure becomes primary and no value/list is returned.

Nulls remain `NullPointerException`; local list/limit/[identity](../../../../glossary.md#term-identity-meanings)/[descriptor](../../../../glossary.md#term-descriptor) problems remain
`IllegalArgumentException`; [lifecycle](../../../../glossary.md#lifecycle) and impossible [publication](../../../../glossary.md#term-publication)/result inconsistency remain
`IllegalStateException`; checked count arithmetic remains `ArithmeticException`; inward
unchecked failures and [`OutOfMemoryError`](../../../../glossary.md#memory-management-atomic-counters-and-complexity) retain their original type and object [identity](../../../../glossary.md#term-reference-identity). Do not
introduce an Engine exception hierarchy or translate [Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [CPU](../../../../glossary.md#term-cpu), or [JDK](../../../../glossary.md#term-java-development-kit-jdk)
failures.

Before Runtime result construction, preserve existing task-0003 [rollback](../../../../glossary.md#term-rollback): partial borrows close
in reverse order; [run](../../../../glossary.md#term-run) failure closes run-owned state through Runtime and then wrappers; caller
storage is never closed. After result construction, the one-shot call owns that exact result and
all wrappers through its established close protocol. A failed call publishes no partial return,
even if earlier output snapshots were already constructed.

### [Engine](../../../../glossary.md#term-engine) close and [concurrency](../../../../glossary.md#term-concurrency)

- Both overloads are synchronous and thread-safe through the existing Engine lifecycle gate.
  Separate one-shot calls may run concurrently and receive isolated Runtime states.<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [thread-safe glossary](../../../../glossary.md#term-thread-safe); [lifecycle glossary](../../../../glossary.md#lifecycle); [run glossary](../../../../glossary.md#term-run); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- One admission spans the whole operation. Engine close that begins after admission waits for
  compile, prepare, run, every copy, and result cleanup to finish before closing the CPU
  composition.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [operation glossary](../../../../glossary.md#operation); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Because only detached values escape, an admitted successful call may return its completed value
  or list after another thread has begun Engine closure. This matches task 0004's admitted-copy
  rule and does not create a new usable Engine handle or open lease after closure.<span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results); [lease glossary](../../../../glossary.md#term-lease))</span>
- A call that loses admission fails with the stable closed-Engine error and performs no argument
  inspection or inward work.<span data-glossary-navigation="true"> ([argument glossary](../../../../glossary.md#term-argument))</span>
- No temporary one-shot result is added to the Engine's open-result registry. Existing reusable
  results remain registered and continue to close in reverse successful-run order during Engine
  shutdown.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry))</span>
- The implementation adds no cancellation, asynchronous API, worker, shutdown hook, process-
  global state, automatic retry, fallback, or synchronization of caller payload mutation.<span data-glossary-navigation="true"> ([cancellation glossary](../../../../glossary.md#term-cancellation); [asynchronous glossary](../../../../glossary.md#term-asynchronous); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [synchronization glossary](../../../../glossary.md#term-synchronization); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Any method on [`Tensor`](../../../../glossary.md#tensor), including `execute`, `forward`, [`run`](../../../../glossary.md#term-run), `backward`, or a gradient/value
  accessor; any Model-to-Runtime, Model-to-Engine, or Model-to-backend dependency.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [Model glossary](../../../../glossary.md#term-model); [Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Engine task 0006 scalar-objective backward convenience, backward-capable one-shot overloads,
  implicit scalar seeds, implicit targets, target discovery, <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [objective glossary](../../../../glossary.md#term-objective); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [targets glossary](../../../../glossary.md#term-target))</span>`withBackward`, mutable Tensor
  gradient state, optimizer/session work, or training orchestration.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- A request/options/configuration/result wrapper, default byte limit, unlimited sentinel,
  per-output limits, streaming/chunked/mapped output, caller destination, typed arrays, scalar
  decoding, Tensor reconstruction, persistence format, or output cache.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [cache glossary](../../../../glossary.md#term-cache))</span>
- Reuse or caching of compiled graphs, prepared executions, result bytes, backend decisions, or
  hidden process-global lifecycle state. Callers needing reuse continue to use the lower-level
  ordinary compile/prepare/run/materialize API.<span data-glossary-navigation="true"> ([graphs glossary](../../../../glossary.md#graph); [lifecycle glossary](../../../../glossary.md#lifecycle); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Model autotuning, Config 0006A mapping, tools/tuning integration, profiling-based selection, or
  any Engine 0007 behavior.<span data-glossary-navigation="true"> ([Model autotuning glossary](../../../../glossary.md#autotuning--model-autotuning))</span>
- Cross-backend host transfer or materialization abstraction, another concrete backend, mixed-
  backend schedule composition, backend selection controls, registration/discovery API,
  reflection, <span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization); [concrete backend glossary](../../../../glossary.md#concrete-backend))</span>`ServiceLoader`, service location, or fallback from [CPU](../../../../glossary.md#term-cpu).
- Successful zero-node/pass-through preparation, dynamic or binding-dependent output Shapes,
  unresolved output layouts, layout inference, CPU capability expansion, operation-specific
  Engine switches, or any Engine 0006-or-later implementation.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [binding glossary](../../../../glossary.md#term-binding); [Shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [inference glossary](../../../../glossary.md#term-inference); [operation glossary](../../../../glossary.md#operation))</span>
- Changes to Model, Compiler, Planning, Prepare, Runtime, CPU, Config, Trace, Backend Contract,
  build files/dependencies, architecture contracts/docs, ADRs, architecture tests,
  backend-conformance tests, NN, Training, tuning, generated code, or unrelated integration
  behavior.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Trace glossary](../../../../glossary.md#trace); [conformance glossary](../../../../glossary.md#term-conformance); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Creating an [Engine](../../../../glossary.md#term-engine) 0006-or-later detailed task specification or changing Engine 0006–0008 from
  `Draft`.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Core lifecycle, Core invariants,
  Engine, Runtime, Prepare, concrete backends, dependency rules, and compile/prepare/run lifecycle.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [Engine glossary](../../../../glossary.md#term-engine); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [backends glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [compile glossary](../../../../glossary.md#compile); [run glossary](../../../../glossary.md#term-run))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## Architecture constraints

- Engine remains the outer composition and public lifecycle owner. Compiler owns compilation,
  Prepare owns staged preparation, Runtime owns each isolated run and lease cleanup, and CPU owns
  concrete execution, representation access, and canonical copying.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [owner glossary](../../../../glossary.md#term-owner); [Compiler glossary](../../../../glossary.md#term-compiler); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [run glossary](../../../../glossary.md#term-run); [lease glossary](../../../../glossary.md#term-lease); [CPU glossary](../../../../glossary.md#term-cpu); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- The convenience must be orchestration over the exact existing lower-level contracts, not a
  parallel compiler, binder, runner, publication mapper, or materializer.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- [`Tensor`](../../../../glossary.md#tensor) remains Model state without execution, Runtime, backward, gradient-lifecycle, backend,
  or Engine responsibility.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend))</span>
- Prepared execution remains immutable and reusable even though this convenience intentionally
  creates a fresh prepared recipe per call. The reusable ordinary API remains available and
  unchanged.<span data-glossary-navigation="true"> ([Prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [immutable glossary](../../../../glossary.md#term-immutable); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Runtime performs only already-prepared work and does not discover a backend, compile, lower,
  select a route, interpret outputs, or copy host values.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [host glossary](../../../../glossary.md#term-host))</span>
- CPU remains the sole supported standard lifecycle adapter and the sole current concrete host-
  copy implementation. No cross-backend abstraction is inferred from this convenience.
- No dependency direction, module boundary, authoritative architecture rule, shared contract, or
  build structure changes. If implementation requires one, stop and report the exact conflict.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing public package changed:

- `io.github.pho001.synaptik.engine` remains the deliberate small ordinary facade and adds only
  the two <span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade))</span>`Engine.forward(...)` methods.

Existing package-private machinery changed:

- `AdvancedEngine` gains only narrow ordinary one-shot orchestration and/or shared ungated helpers
  needed to hold one lifecycle admission across the complete sequence. This adds no advanced
  public method.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [admission glossary](../../../../glossary.md#term-admission))</span>

Packages added or moved: none.

Type placement:

- `io.github.pho001.synaptik.engine.Engine` owns both overloads because Engine already owns
  standard composition, ordinary lifecycle entry, and failure/lifetime coordination.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- `io.github.pho001.synaptik.engine.AdvancedEngine` remains the private lifecycle owner used by
  ordinary Engine; it is the only existing place permitted to coordinate the single admission.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Existing `io.github.pho001.synaptik.engine.HostTensorValue` is the result element because it
  already owns detached immutable canonical host bytes and no additional carrier fact is needed.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [detached glossary](../../../../glossary.md#detached-model-results); [immutable glossary](../../../../glossary.md#term-immutable); [canonical glossary](../../../../glossary.md#term-canonical-form); [host glossary](../../../../glossary.md#term-host); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>

## Affected files

Exact implementation allowlist:

[Engine](../../../../glossary.md#term-engine) production and [Javadoc](../../../../glossary.md#term-javadoc) (three paths):

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

[Planning](../../../../glossary.md#term-planning)/status (three paths):

12. This task.
13. `docs/planning/modules/engine/master-plan.md`.
14. `docs/planning/roadmap.md`.

Review without modification: [`RunResult`](../../../../glossary.md#run-result--runresult), [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue), `CompiledGraph`,
[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), the package-private [backend composition](../../../../glossary.md#backend-composition) seam, advanced public types, existing
standard/advanced [lifecycle](../../../../glossary.md#lifecycle) tests and integrations other than the four allowlisted test paths,
Training and [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) references, all architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) files, Engine and integration build
files, the focused [architecture test](../../../../glossary.md#term-architecture-test), every inward [module](../../../../glossary.md#term-module), [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), and later task
rows/specifications. The documentation pass must record reasoned no-change conclusions.

## Maximum [scope](../../../../glossary.md#term-scope)

Implementation may create or modify at most the exact fourteen paths above: three [Engine](../../../../glossary.md#term-engine)
production/[Javadoc](../../../../glossary.md#term-javadoc), four focused tests, four explanatory documents, and three [planning](../../../../glossary.md#term-planning)/status
paths. Unused capacity in one category is not permission to edit another path. No production type,
test helper file, [module](../../../../glossary.md#term-module), package, [dependency](../../../../glossary.md#term-dependency), build file, architecture file, or fifteenth path is
authorized. If another path is required, stop and replan before editing.

## Acceptance criteria

- [`Engine`](../../../../glossary.md#term-engine) exposes exactly the two specified `forward` overloads in addition to its current public
  surface; all generic signatures, return types, modifiers, constructor visibility, and ordinary
  import boundaries are enforced by the distinct-package public-shape test.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [shape glossary](../../../../glossary.md#shape))</span>
- `AdvancedEnginePublicShapeTest` changes only its exact ordinary `Engine` method-name inventory
  by adding the two `forward` entries; all advanced-surface assertions, including every
  advanced-type, constructor, field, modifier, lifecycle, and forbidden-signature assertion,
  remain unchanged.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- No new public type exists. The single overload returns one existing [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue); the
  ordered overload returns one immutable output-ordered list of existing values.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Both overloads use one admission and exactly one compile, prepare, run, and result cleanup. The
  multi-output overload performs exactly one materialization per forward occurrence in increasing
  order; the single overload uses the same internal implementation without nested public calls.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [materialization glossary](../../../../glossary.md#materialization); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Caller output/input collections are snapshotted and not retained or mutated. Output identity,
  input <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>[`TensorId`](../../../../glossary.md#tensorid) binding, descriptor checks, association snapshots, borrow order, shared-storage
  handling, caller ownership, and CPU validation remain exactly as specified.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [descriptor glossary](../../../../glossary.md#term-descriptor); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [validation glossary](../../../../glossary.md#term-validation))</span>
- Complete publication invariants and the aggregate byte count are validated before copying.
  Static/resolved, zero-element, individual JVM ceiling, checked sum overflow, exact-limit
  success, total-limit failure, and no-copy-on-preflight-failure cases are covered.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [element glossary](../../../../glossary.md#term-element); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [overflow glossary](../../../../glossary.md#term-overflow); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- Returned list/value ordering follows the requested output list and exact publication order.
  Distinct occurrences that alias inwardly are copied independently and produce distinct values;
  no cache, deduplication, search, reorder, or reuse occurs.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena); [cache glossary](../../../../glossary.md#term-cache))</span>
- The temporary result and wrappers always close before normal return. Compile, prepare, binding,
  run, publication-defense, size-preflight, mid-copy, value-construction, and cleanup failures
  preserve the exact primary/suppressed rules and publish no partial return.
- Bounded-latch tests prove Engine close waits for an admitted one-shot call, a successful admitted
  call may return detached values after closure begins, a losing call observes closed precedence,
  and concurrent calls have isolated runs without deadlock or leaked registered results.<span data-glossary-navigation="true"> ([latch glossary](../../../../glossary.md#control-flow-graph); [detached glossary](../../../../glossary.md#detached-model-results); [deadlock glossary](../../../../glossary.md#term-deadlock))</span>
- The reusable ordinary compile/prepare/run/materialize lifecycle, result registration and close
  order, advanced API, current CPU-only limitations, and existing public behavior remain passing.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Real `Engine.standard()` integration executes one and two supported resolved static CONTIGUOUS
  outputs from explicit caller-backed FLOAT32 inputs, verifies output order and canonical bytes,
  proves the aggregate exact-limit/failure boundary, and confirms caller arenas remain caller-
  owned. It makes no unsupported ADD, zero-node, cross-backend, dynamic, or layout claim.<span data-glossary-navigation="true"> ([CONTIGUOUS glossary](../../../../glossary.md#term-contiguous); [FLOAT32 glossary](../../../../glossary.md#term-float32); [canonical glossary](../../../../glossary.md#term-canonical-form); [arenas glossary](../../../../glossary.md#term-arena); [node glossary](../../../../glossary.md#node); [backend glossary](../../../../glossary.md#backend); [layout glossary](../../../../glossary.md#layout))</span>
- Meaningful Javadocs document purpose, arguments, nullability, aggregate byte units and scope,
  return immutability/order/ownership, lifecycle, concurrency, failures, CPU limits, and the reuse
  tradeoff. Package Javadoc distinguishes one-shot from reusable lifecycle use.<span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [nullability glossary](../../../../glossary.md#term-nullability); [scope glossary](../../../../glossary.md#term-scope); [concurrency glossary](../../../../glossary.md#term-concurrency); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- The documentation-focused pass updates Public, Compile, Runtime, and glossary wording from
  planned to current, includes a complete one-shot example, and does not imply backward,
  autotuning, transfer, cross-backend, Tensor execution, caching, or prepared reuse.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [Tensor glossary](../../../../glossary.md#tensor))</span>
- No non-allowlisted path changes. Engine 0005 is `Complete` in this task, master plan, and
  roadmap. Engine 0006–0008 remain `Draft`, and no 0006-or-later Engine task specification
  exists.

## Tests / [validation](../../../../glossary.md#term-validation)

This is task-tier [Engine](../../../../glossary.md#term-engine) validation. During implementation [run](../../../../glossary.md#term-run) focused tests as needed, then after
executable Java stabilizes record one final pass of:

```bash
./gradlew :modules:engine:test
./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.EngineTypedLifecycleIntegrationTest
./gradlew :testing:architecture-tests:test --tests io.github.pho001.synaptik.testing.architecture.EngineCompositionContractTest
```

The implementation context must report exact suite/test counts and hand that evidence to the
documentation context. Do not run [Runtime](../../../../glossary.md#term-runtime) or [CPU](../../../../glossary.md#term-cpu) suites: their completed 0015/0010G behavior is
unchanged. Do not run [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance) because no shared backend contract changes. Repository-
wide tests remain deferred to Engine 0008/[CI](../../../../glossary.md#term-continuous-integration-ci) because no [dependency](../../../../glossary.md#term-dependency), architecture, shared build,
or second-production-[module](../../../../glossary.md#term-module) contract changes.

The distinct clean documentation-focused pass reuses successful Java evidence unless it changes
executable Java behavior, then runs:

```bash
./gradlew :modules:engine:javadoc
git diff --check
```

Also validate:

- generated [Javadoc](../../../../glossary.md#term-javadoc) for the two overloads and package [lifecycle](../../../../glossary.md#lifecycle) description;
- both exact Engine public-surface inventories, with `AdvancedEnginePublicShapeTest` changed only
  to add the two `forward` method names, plus a distinct-package fixture using only Engine, Model,
  and JDK types;<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [Model glossary](../../../../glossary.md#term-model); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>
- changed Markdown local targets and effective anchors, heading order, balanced fences, trailing
  whitespace, LF line endings, and final newlines;<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- current-versus-planned claims and glossary terminology;
- the exact fourteen-path implementation allowlist and absence of build/dependency, architecture,
  inward-module, later-specification, or unrelated changes;
- task/master/roadmap status agreement, 0005 Complete, 0006–0008 Draft, and no Engine 0006-or-
  later detailed task; and
- preservation [hashes](../../../../glossary.md#term-hash) for every dirty path that predated implementation.

Do not run Java tests during this [planning](../../../../glossary.md#term-planning) task.

## Documentation [handoff](../../../../glossary.md#term-handoff)

Primary profiles are [Planning](../../../../glossary.md#term-planning) for this task/master/roadmap, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc) for the changed [Engine](../../../../glossary.md#term-engine)
methods and API references, General for glossary terminology, and Example for the ordinary
one-shot walkthrough.

After executable code and tests stabilize, hand a distinct clean documentation-focused context:

- this task and the exact goal/public API;
- the final Engine and test diff;
- exact implementation test commands, counts, and outcomes;
- the dirty-worktree baseline and preservation [hashes](../../../../glossary.md#term-hash);
- the one-admission lifecycle, explicit-input ownership, aggregate-limit, publication-order,
  cleanup/suppression, no-cache/no-reuse, and CPU-only decisions;<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [cache glossary](../../../../glossary.md#term-cache))</span>
- completed Engine 0003–0004, [Runtime](../../../../glossary.md#term-runtime) 0015, and [CPU](../../../../glossary.md#term-cpu) 0010G contracts and unchanged evidence; and
- the four authorized explanatory paths plus the applicable documentation profiles.

The pass must independently inspect final behavior and tests. It must explain **[one-shot forward
execution](../../../../glossary.md#one-shot-forward-execution)** at first use as fresh per-call [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), preparation, execution, complete forward-
[publication](../../../../glossary.md#term-publication) [materialization](../../../../glossary.md#materialization), and cleanup. Add or revise a glossary entry only for that reusable
distinction; do not create synonyms or imply [Tensor](../../../../glossary.md#tensor) [ownership](../../../../glossary.md#term-ownership-meanings).

The complete example must use `Engine.standard()`, one or more supported static resolved
[CONTIGUOUS](../../../../glossary.md#term-contiguous) outputs, explicit caller-backed input Tensors, an explicit aggregate byte limit, and a
try-with-resources Engine. It must show that no [`RunResult`](../../../../glossary.md#run-result--runresult) or prepared handle escapes, returned
values are ordered/[detached](../../../../glossary.md#detached-model-results), and caller storage remains owned by the caller. It must state that
each call compiles and prepares afresh and recommend the lower-level [lifecycle](../../../../glossary.md#lifecycle) for repeated runs
or selective materialization.

Training API and Tensor API require reasoned no-change conclusions: this forward-only capability
does not change [gradient](../../../../glossary.md#term-gradient)/training behavior or place execution on Tensor. Architecture pages and
ADRs require no edit because the implementation realizes the existing Engine composition role.
The documentation context must not rerun stable Java tests unless it changes executable behavior
or records a concrete risk. It records all reviewed paths, changes, commands, results, and no-
change conclusions before this task may become `Complete`.

## Dependencies

- [Engine](../../../../glossary.md#term-engine) 0001 advanced [lifecycle](../../../../glossary.md#lifecycle) foundation — Complete.
- Engine 0002 standard [CPU](../../../../glossary.md#term-cpu)-only composition — Complete.
- Engine 0003 ordinary typed logical input binding, publication occurrence mapping, and reusable
  result lifecycle — Complete.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [publication occurrence glossary](../../../../glossary.md#publication-occurrence))</span>
- Engine 0004 exact bounded [detached](../../../../glossary.md#detached-model-results) [host](../../../../glossary.md#term-host) [materialization](../../../../glossary.md#materialization) — Complete.
- [Runtime](../../../../glossary.md#term-runtime) 0015 exact [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [publication](../../../../glossary.md#term-publication)-representation access under the result [lease](../../../../glossary.md#term-lease) — Complete.
- CPU 0010G [canonical](../../../../glossary.md#term-canonical-form) caller-owned host [snapshot](../../../../glossary.md#term-snapshot) export — Complete.
- Current Model descriptor/storage contracts and the current Compiler/Prepare/Runtime/CPU ordinary
  path — implemented and sufficient.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [descriptor glossary](../../../../glossary.md#term-descriptor); [Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare))</span>

All prerequisites are complete. No architecture decision or material user choice remains for this
bounded design.

## Follow-up tasks

- Engine 0006 remains Draft: add only the separately specified scalar-objective backward
  convenience with explicit targets after 0005 is Complete. Preserve the current explicit-seed
  compile overload and advanced full-request path.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [objective glossary](../../../../glossary.md#term-objective); [targets glossary](../../../../glossary.md#term-target); [seed glossary](../../../../glossary.md#term-seed); [compile glossary](../../../../glossary.md#compile))</span>
- Engine 0007 remains Draft: optional Config 0006A and tools/tuning 0001 composition before
  preparation. One-shot forward in this task is always the current deterministic untuned path.
- [Engine](../../../../glossary.md#term-engine) 0008 remains Draft: [lifecycle](../../../../glossary.md#lifecycle) capability [checkpoint](../../../../glossary.md#term-checkpoint) after 0001–0006.
- Cross-backend materialization, mixed-backend schedule composition, output streaming, and
  compile/prepare caching require separately demonstrated owner contracts; no detailed task is
  created here.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [materialization glossary](../../../../glossary.md#materialization); [prepare glossary](../../../../glossary.md#prepare); [owner glossary](../../../../glossary.md#term-owner))</span>

## Architecture impact

Expected impact: None.

The task adds ordinary [Engine](../../../../glossary.md#term-engine) orchestration over existing [compiler](../../../../glossary.md#term-compiler), preparation, [Runtime](../../../../glossary.md#term-runtime) [lease](../../../../glossary.md#term-lease),
and [CPU](../../../../glossary.md#term-cpu) copy contracts. Engine already owns public [lifecycle](../../../../glossary.md#lifecycle) composition and already depends on
all named inward contracts. It adds no [dependency](../../../../glossary.md#term-dependency), changes no [owner](../../../../glossary.md#term-owner), and exposes no inward type.
[`Tensor`](../../../../glossary.md#tensor) remains execution-free, Runtime remains concrete-[backend](../../../../glossary.md#backend)-independent, and CPU remains
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

- Name the [operation](../../../../glossary.md#operation) `forward` on ordinary [`Engine`](../../../../glossary.md#term-engine). The name states the graph scope without
  implying prepared-handle reuse or placing execution on Tensor.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [scope glossary](../../../../glossary.md#term-scope); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Provide both one-output and ordered-list overloads. They share one internal implementation;
  neither introduces a policy or request abstraction.
- Return the existing [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue) and [immutable](../../../../glossary.md#term-immutable) `List<HostTensorValue>`. A new carrier would
  duplicate already self-describing detached values and add no lifecycle authority.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [detached glossary](../../../../glossary.md#detached-model-results); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Use one aggregate `long maximumTotalBytes`. Aggregate accounting prevents several individually
  permitted outputs from bypassing the caller's total returned-payload budget, while the existing
  per-value JVM ceiling remains intact.<span data-glossary-navigation="true"> ([payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- Complete aggregate descriptor/count validation before any CPU copy, then pass each exact
  individual byte count to the existing materializer as its tight per-copy limit.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [validation glossary](../../../../glossary.md#term-validation); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Keep the entire lifecycle under one existing Engine admission and keep the temporary result
  local and unregistered. Detached values may safely return after concurrent Engine closure has
  begun, while Engine close waits for cleanup before closing CPU.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission))</span>
- Create no cache. Repeated workloads should use the explicit reusable lower-level lifecycle,
  whose API and ownership remain unchanged.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

## Known limitations

- Current execution and materialization are CPU-only and accept exactly the operation, static-
  Shape, resolved-layout, and one-non-empty-maximal-partition domain supported by the current CPU
  composition. Compile success remains no guarantee of prepare or run success.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [CPU glossary](../../../../glossary.md#term-cpu); [operation glossary](../../../../glossary.md#operation); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [partition glossary](../../../../glossary.md#partition); [Compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run))</span>
- Every invocation compiles and prepares from scratch. It is convenience for one use, not an
  efficient repeated-execution API.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Every output is materialized eagerly before return. There is no selective, lazy, streamed,
  chunked, mapped, caller-destination, typed-array, or Tensor result form.<span data-glossary-navigation="true"> ([lazy glossary](../../../../glossary.md#term-lazy-evaluation); [Tensor glossary](../../../../glossary.md#tensor))</span>
- The aggregate limit bounds canonical returned payload bytes only, not peak or total memory, and
  each payload remains subject to <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication))</span>`Integer.MAX_VALUE` and available [heap](../../../../glossary.md#term-heap).
- Caller input associations are snapshotted per Tensor, not atomically across inputs, and caller
  payload mutation/closure is not synchronized by Engine.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- No backward, implicit target/seed, autotuning, cross-backend transfer, mixed-backend composition,
  zero-node preparation, or output cache is included.<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [seed glossary](../../../../glossary.md#term-seed); [autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [backend glossary](../../../../glossary.md#backend); [backend composition glossary](../../../../glossary.md#backend-composition); [node glossary](../../../../glossary.md#node); [cache glossary](../../../../glossary.md#term-cache))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- [Planning](../../../../glossary.md#term-planning) context `01a0a527-b8d0-78e1-9c56-f39f73a91e26` read the required architecture,
  planning, completed Engine 0001–0004, Runtime 0015, CPU 0010G, Engine source/test/integration,
  Model descriptor/storage, API, glossary, and documentation-profile contracts. No architecture,
  dependency, inward-contract, or material user-choice blocker was found.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [CPU glossary](../../../../glossary.md#term-cpu); [Model glossary](../../../../glossary.md#term-model); [descriptor glossary](../../../../glossary.md#term-descriptor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Recorded [SHA-256](../../../../glossary.md#term-sha-256) preservation [hashes](../../../../glossary.md#term-hash) for every pre-existing dirty path before planning edits.
- Java tests were not [run](../../../../glossary.md#term-run), as required for this planning-only task.
- Implementation context `01a0a533-6b58-7d70-9cb6-3bcc9b07ee6c` implemented the one-admission
  lifecycle in the original six paths and corrected its initial cleanup-test failures before
  reporting stable executable behavior.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Scope](../../../../glossary.md#term-scope)-correction context `01a0a53c-4ea1-72d2-88d4-79c03a92a585` changed only this specification
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
  plus this task, the Engine master plan, and the roadmap.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature))</span>
- That documentation context ran `./gradlew :modules:engine:javadoc` successfully and inspected
  generated [`Engine`](../../../../glossary.md#term-engine), `AdvancedEngine`, and package pages for both overloads, lifecycle admission,
  aggregate preflight, cleanup, and reusable-path guidance. Exact <span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>`javap -public` inspection
  confirmed both required signatures and no advanced-surface addition. A Java 26 distinct-package
  in-memory-compiled fixture, using only Engine, Model, and JDK public types and launched with<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk))</span>
  `--add-modules jdk.incubator.vector`, reflected both generic signatures and ran the singleton
  and ordered overloads through <span data-glossary-navigation="true"> ([singleton glossary](../../../../glossary.md#term-singleton))</span>`Engine.standard()`; it verified output order, immutable list
  behavior, and detached post-close access.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [detached glossary](../../../../glossary.md#detached-model-results))</span>
- Before documentation edits, comment-stripped executable Java hashes were
  `78b5924d1bf2a13ff49a3acf5d70e1ae70693d2341b0b06bc72f370f326bd0f7` for `Engine.java` and
  `32a96e18deb0cf3e2d727ddddb7a410f7d1080dee51b5e00145fe4390d5e6623` for
  `AdvancedEngine.java`. The combined SHA-256 preservation digest for every dirty path outside
  the ten documentation-finalization paths was<span data-glossary-navigation="true"> ([digest glossary](../../../../glossary.md#term-digest))</span>
  `4c999c62ba96d91ac97cab8c245b6b7dd87f4bb9b84fafe1151055b690fb87c9`.
- Final rechecking reproduced both executable hashes and the non-finalization dirty-path digest
  exactly. The seven changed Markdown paths passed local-target/anchor, effective-heading,
  balanced-fence, trailing-whitespace, LF, and final-newline validation. All fourteen task paths
  were present and no fifteenth task path was attributed; task/master/roadmap status agreed on
  0005 Complete, 0006–0008 remained Draft, no later Engine task specification existed, and final<span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target))</span>
  `git diff --check` passed.
- Java test suites were not repeated in the documentation context because executable Java tokens
  did not change and independent inspection identified no stale-evidence risk.<span data-glossary-navigation="true"> ([tokens glossary](../../../../glossary.md#term-token))</span>

## Implementation notes

[`Engine`](../../../../glossary.md#term-engine) now exposes exactly the requested [singleton](../../../../glossary.md#term-singleton) and ordered `forward(...)` overloads.
Package-private ordinary orchestration in `AdvancedEngine` shares the established [compile](../../../../glossary.md#compile),
[prepare](../../../../glossary.md#prepare), logical-[binding](../../../../glossary.md#term-binding), [publication](../../../../glossary.md#term-publication), [materialization](../../../../glossary.md#materialization), and cleanup operations while keeping one
[admission](../../../../glossary.md#term-admission) across the whole one-shot call. The four focused tests cover the exact public surface,
ordered/static [CPU](../../../../glossary.md#term-cpu) success, aggregate bounds, failure/cleanup behavior, and [lifecycle](../../../../glossary.md#lifecycle) [concurrency](../../../../glossary.md#term-concurrency).

Documentation now defines [one-shot forward execution](../../../../glossary.md#one-shot-forward-execution) at first use, records explicit inputs,
[immutable](../../../../glossary.md#term-immutable) [detached](../../../../glossary.md#detached-model-results) ordered results, aggregate [canonical](../../../../glossary.md#term-canonical-form)-[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) [scope](../../../../glossary.md#term-scope), CPU/static/resolved limits,
cleanup and concurrency behavior, and the absence of caching, reuse, backward, [Tensor](../../../../glossary.md#tensor) execution,
tuning, and cross-[backend](../../../../glossary.md#backend) transfer. The Public [API](../../../../glossary.md#term-application-programming-interface-api) includes a complete supported
`Engine.standard()` example and directs repeated or selective use to the lower-level lifecycle.

Independent no-change review concluded that [`RunResult`](../../../../glossary.md#run-result--runresult) and [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue) already state the
exact [lease](../../../../glossary.md#term-lease), [occurrence](../../../../glossary.md#term-occurrence) [authentication](../../../../glossary.md#term-authentication), detached-byte, and post-close contracts; `CompiledGraph`
and [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) already state [owner](../../../../glossary.md#term-owner) binding and reuse; and `AdvancedCompiledGraph`,
`AdvancedPreparedExecution`, `AdvancedRunResult`, and the package-private composition seam already
state the unchanged advanced boundary. The four focused tests plus standard and advanced Engine
tests required no documentation-pass edits because their final assertions match the implemented
surface and handed-off evidence.

Training API requires no change because the task is forward-only and adds no [gradient](../../../../glossary.md#term-gradient),
[optimizer](../../../../glossary.md#term-optimizer), [parameter](../../../../glossary.md#term-parameter)-update, or session contract. Tensor API requires no change because execution
remains on Engine and Tensor remains expression/storage [metadata](../../../../glossary.md#term-metadata). Architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0011,
lifecycle/[module](../../../../glossary.md#term-module)/[dependency](../../../../glossary.md#term-dependency)/[runtime](../../../../glossary.md#term-runtime)-prepare-backend pages, inward Runtime/CPU contracts,
build/dependencies, architecture tests, backend [conformance](../../../../glossary.md#term-conformance), and later Engine tasks require no
change because no owner, dependency, shared contract, backend behavior, or planned later scope
changed.

## Completion summary

Completed changes: added the two exact ordinary one-shot forward overloads and their private
one-[admission](../../../../glossary.md#term-admission) orchestration; completed focused semantic, public-[shape](../../../../glossary.md#shape), integration, cleanup, and
[concurrency](../../../../glossary.md#term-concurrency) coverage; finalized all affected Javadocs, Public/[Compile](../../../../glossary.md#compile)/[Runtime](../../../../glossary.md#term-runtime) [API](../../../../glossary.md#term-application-programming-interface-api) explanations,
glossary terminology, the complete supported example, and synchronized [planning](../../../../glossary.md#term-planning) status.

Files changed or created: exactly the fourteen paths listed in **Affected files**. No executable
Java or test file changed during the final documentation pass, and every pre-existing dirty
non-finalization path was preserved.

Tests and [validation](../../../../glossary.md#term-validation): reused the final 33 [Engine](../../../../glossary.md#term-engine) tests, five focused integration tests, and one
focused [architecture test](../../../../glossary.md#term-architecture-test) from completion context `01a0a540-aad0-7551-856b-b10109f56a68`; the
documentation pass completed Engine [Javadoc](../../../../glossary.md#term-javadoc), generated-page inspection, `javap`, reflection, the
distinct-package Java 26 [fixture](../../../../glossary.md#term-fixture), Markdown/link/anchor/fence/format validation, executable-[token](../../../../glossary.md#term-token)
[hashes](../../../../glossary.md#term-hash), fourteen-path [scope](../../../../glossary.md#term-scope) and status checks, dirty-path preservation, and whitespace validation.

Unresolved issues: none for task 0005.

Required follow-up: none. Draft Engine 0006–0008 remain separately planned work.

Status: Complete
