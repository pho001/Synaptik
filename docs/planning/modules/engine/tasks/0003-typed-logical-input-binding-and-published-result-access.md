# Task 0003: Typed Logical Input [Binding](../../../../glossary.md#term-binding) and Published-Result Access

## Status

Complete

## Goal

Deliver the ordinary [Engine](../../../../glossary.md#term-engine)-owned `compile -> prepare -> run` [lifecycle](../../../../glossary.md#lifecycle) for the current [CPU](../../../../glossary.md#term-cpu)
composition. Bind caller [Tensors](../../../../glossary.md#tensor) by logical [`TensorId`](../../../../glossary.md#tensorid), using their current [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) [host](../../../../glossary.md#term-host)-storage
associations, and expose every ordered forward and first-order [gradient](../../../../glossary.md#term-gradient) [publication](../../../../glossary.md#term-publication) as typed
logical [metadata](../../../../glossary.md#term-metadata) with a result [lease](../../../../glossary.md#term-lease). Ordinary callers must not construct a CPU adapter or supply
[Compiler](../../../../glossary.md#term-compiler) [identities](../../../../glossary.md#term-identity-meanings), [Runtime](../../../../glossary.md#term-runtime) positions, or representations.

This task exposes publication [identity](../../../../glossary.md#term-identity-meanings) and lifecycle, not numerical values. Task 0004 owns host
[materialization](../../../../glossary.md#materialization) and value access; tasks 0005–0006 own one-shot forward and [scalar](../../../../glossary.md#term-scalar)-[objective](../../../../glossary.md#term-objective)
backward convenience.

## Dependencies and readiness

- [Engine](../../../../glossary.md#term-engine) [0001](0001-advanced-composition-and-representation-level-lifecycle-foundation.md) and
  [0002](0002-standard-built-in-composition.md) are Complete.
- [Compiler 0006B4](../../compiler/tasks/0006b4-stable-caller-input-tensor-identity-bindings.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
  is Complete. Its implemented `CompileConstantPlan.bindableInputBindings()` supplies immutable
  ordered <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`BindableInput(TensorId tensorId, ValueId valueId)` entries; `bindableInputs()` remains
  the exact ordered compatibility projection. The eight-component `CompileArtifacts` is unchanged.
- Existing `PublicationPlan`, [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation), [`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner), [Runtime](../../../../glossary.md#term-runtime)
  [`RunResult`](../../../../glossary.md#run-result--runresult), and `CpuBackendIntegration` supply the remaining required public collaborations.
  Prepare already verifies caller-input order and forward-then-gradient publication occurrences.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication))</span>
- The mandatory clean replan is complete in context
  `01a0a465-0a0a-7813-a035-669dd8423d56`. That context independently audited the interrupted
  draft against the current source and contracts. The former identity blocker is resolved; no
  architecture or inward API change is needed for this bounded scope.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [scope glossary](../../../../glossary.md#term-scope))</span>
- Implementation context `01a0a46c-8a9e-7862-bd81-0d3495092894` proved that Engine production
  compiles, all 22 Engine tests pass, and the real zero-node integration reaches the expected
  preparation rejection. It also exposed a planning-only fixture error: both required ADD
  integrations fail during compile-time hard eligibility, before Engine preparation, because a
  public <span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node); [planning glossary](../../../../glossary.md#term-planning); [fixture glossary](../../../../glossary.md#term-fixture); [compile glossary](../../../../glossary.md#compile))</span>`Tensor.add` result has unresolved layout and CPU rejects any occurrence with unresolved
  input or output layout. Appending <span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [CPU glossary](../../../../glossary.md#term-cpu); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>`contiguous()` resolves only that new CONTIGUOUS result; it
  does not retroactively resolve the ADD occurrence. The corrected integration fixture below uses
  only supported CONTIGUOUS occurrences over resolved FLOAT32 leaves and requires no production,
  architecture, or allowlist expansion.<span data-glossary-navigation="true"> ([CONTIGUOUS glossary](../../../../glossary.md#term-contiguous); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>

## Required reading

Read these contracts in the clean implementation context, not merely this task's summaries:

- Root `AGENTS.md`, [architecture contract](../../../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract))</span>, and
  [current architecture plan](../../../../architecture/current-architecture-plan.md).
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>,
  [dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>,
  [lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>, and
  [Runtime/Prepare/backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>.
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>, [roadmap](../../../roadmap.md),
  [Engine master plan](../master-plan.md), the dependencies above, and their actual production
  source, tests, and affected Javadoc.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- [Model](../../../../glossary.md#term-model) [`Tensor`](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings)/[descriptor](../../../../glossary.md#term-descriptor)/storage methods, [`TensorId`](../../../../glossary.md#tensorid), `TensorDescriptor`,
  `LayoutDescriptor`, `HostTensorStorage`, and `MemorySegmentStorage`; [Compiler](../../../../glossary.md#term-compiler)
  `GraphCompilationPort`, `CompileArtifacts`, `CompileConstantPlan`, `PublicationPlan`,
  `FunctionalGradientRequest`, and Model forward/[gradient](../../../../glossary.md#term-gradient) [publication](../../../../glossary.md#term-publication) [bindings](../../../../glossary.md#term-binding).
- All [Engine](../../../../glossary.md#term-engine) production files and existing Engine unit/integration tests; public
  [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation), [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution), [`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner), [`RunResult`](../../../../glossary.md#run-result--runresult),
  [`PreparedPublication`](../../../../glossary.md#prepared-publication--preparedpublication), [`BoundPublication`](../../../../glossary.md#bound-publication--boundpublication), run-state ownership/cleanup contracts, and supported
  CPU integration. CPU internals may be read as evidence but never imported by Engine.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Current Config [compile](../../../../glossary.md#compile) leaves, [Public API](../../../../api/public-api.md)<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>,
  [Compile API](../../../../api/compile-api.md), [Runtime API](../../../../api/runtime-api.md),
  relevant [Training API](../../../../api/training-api.md) status, and
  [glossary](../../../../glossary.md) terms.
- [Documentation rules](../../../../developer-guide/documentation-rules.md) and the General,
  Planning, API/Javadoc, and Example profiles under `docs/developer-guide/documentation/`.

## [Scope](../../../../glossary.md#term-scope)

One cohesive [Engine](../../../../glossary.md#term-engine) capability contains the ordinary methods, [immutable](../../../../glossary.md#term-immutable) [compile](../../../../glossary.md#compile)/prepared
handles, logical input [metadata](../../../../glossary.md#term-metadata), [identity](../../../../glossary.md#term-identity-meanings)-checked per-[run](../../../../glossary.md#term-run) storage snapshots, [publication
occurrence](../../../../glossary.md#publication-occurrence) handles, one shared [admission](../../../../glossary.md#term-admission)/cleanup protocol, focused tests, and documentation.
No new package or public request/configuration abstraction is needed.

### Exact public [API](../../../../glossary.md#term-application-programming-interface-api) and package placement

All new production types live in the deliberate small public [facade](../../../../glossary.md#term-facade) package
`io.github.pho001.synaptik.engine`. These are complete declared public method inventories;
inherited Object methods and generated record/enum methods are not additional facade APIs.
No type has a public or protected constructor except the explicitly listed input record.

| Type | Exact new or complete declared public surface |
|---|---|
| Existing final `Engine implements AutoCloseable` | Preserve `static Engine standard()`, `boolean isClosed()`, `void close()`; add `CompiledGraph compile(List<Tensor> forwardOutputs)`, `CompiledGraph compile(List<Tensor> forwardOutputs, List<Tensor> cotangentSeeds, List<Tensor> targets)`, `PreparedExecution prepare(CompiledGraph compiledGraph)`, `RunResult run(PreparedExecution preparedExecution, List<Tensor> inputs)` |
| New final `CompiledGraph` | `List<CompiledGraph.Input> inputs()` |
| New nested public record `CompiledGraph.Input` | Components and [canonical](../../../../glossary.md#term-canonical-form) constructor exactly `Input(TensorId tensorId, TensorDescriptor descriptor)`; both components non-null |
| New final [`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution) | `CompiledGraph compiledGraph()` |
| New final `RunResult implements AutoCloseable` | `int resultCount()`, `List<RunResult.Publication> publications()`, `boolean isClosed()`, `void close()` |
| New nested public static final `RunResult.Publication` | `int index()`, `TensorId tensorId()`, `TensorDescriptor descriptor()`, `RunResult.Role role()`, `OptionalInt derivativeOrder()`, `OptionalInt targetIndex()`, `boolean isClosed()` |
| New nested public enum `RunResult.Role` | Exactly `FORWARD`, [`GRADIENT`](../../../../glossary.md#term-gradient), in that order; no custom public members |

[`Tensor`](../../../../glossary.md#tensor), `TensorDescriptor`, and [`TensorId`](../../../../glossary.md#tensorid) are [Model](../../../../glossary.md#term-model) types, and collections/optionals are [JDK](../../../../glossary.md#term-java-development-kit-jdk)
types. Ordinary public/protected [signatures](../../../../glossary.md#term-signature), generic [arguments](../../../../glossary.md#term-argument), constructors, fields, nested
types, and supertypes must expose no Advanced handle, [Compiler](../../../../glossary.md#term-compiler), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Planning](../../../../glossary.md#term-planning),
[backend](../../../../glossary.md#backend)-contract, [CPU](../../../../glossary.md#term-cpu), other backend, or `.internal` type. In particular, no [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), slot,
representation, delegate accessor, raw Object/map [binding](../../../../glossary.md#term-binding), or inward conversion is allowed.
There is no public `InputBinding` or gradient-request type in this task.

`CompiledGraph` and `PreparedExecution` have exact [reference identity](../../../../glossary.md#term-reference-identity) and are owned by the exact
ordinary [Engine](../../../../glossary.md#term-engine) instance that created them. They are not records and do not implement value
equality or `AutoCloseable`. `PreparedExecution.compiledGraph()` returns the exact originating
ordinary compiled handle. [Metadata](../../../../glossary.md#term-metadata) remains readable after Engine closure; handles cannot then be
used for new work. The input record is descriptive data, not an executable binding [token](../../../../glossary.md#term-token); making
another equal record grants no [ownership](../../../../glossary.md#term-ownership-meanings) or execution authority.

### [Compile](../../../../glossary.md#compile) requests and [metadata](../../../../glossary.md#term-metadata)

Both overloads [snapshot](../../../../glossary.md#term-snapshot) non-null caller lists without retaining or mutating their containers.
Validate top-level references in declaration order, then snapshot lists in that order and reject
null [elements](../../../../glossary.md#term-element). Forward outputs must be non-empty and [identity](../../../../glossary.md#term-identity-meanings)-unique. The three-list overload
also requires a non-empty identity-unique [target](../../../../glossary.md#term-target) list and exactly one explicit non-null [seed](../../../../glossary.md#term-seed)
per forward output. [Seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) may repeat and may be expressions.

The one-list overload calls the existing compile port once with `FORWARD_ONLY` and no [derivative](../../../../glossary.md#term-derivative)
request. The three-list overload calls it once with `FORWARD_AND_BACKWARD` and exactly one
`FunctionalGradientRequest.Stage`: each forward output becomes a `ForwardTensorReference`,
each aligned seed becomes `Optional.of(seed)`, targets keep their request order,
`createGraph=false`, and `DisconnectedPolicy.ERROR`. [Compiler](../../../../glossary.md#term-compiler) owns all semantic [validation](../../../../glossary.md#term-validation),
[autograd](../../../../glossary.md#autograd), [graph](../../../../glossary.md#graph) inventory, transformations, and target/seed eligibility. In particular it checks
floating types, exact output/seed [Shape](../../../../glossary.md#shape) and type, seed `requiresGrad=false`, and valid targets.
[Engine](../../../../glossary.md#term-engine) must not inspect [producers](../../../../glossary.md#term-producer) or implement derivative rules.

Both paths use the existing `GraphOptimizationConfig.standard()`,
`BackendIntent.unconstrained()`, and `PartitionScoringConfig.neutral()` with the standard [owner](../../../../glossary.md#term-owner)'s
existing capability and availability collaborators. No Config aggregate, options overload,
inferred target, implicit [scalar](../../../../glossary.md#term-scalar) seed, [second derivative](../../../../glossary.md#term-second-derivative) stage, TRAINING_STEP convenience,
`withBackward`, or [Tensor](../../../../glossary.md#tensor) execution/backward method is added. The existing advanced request
surface retains its full current one/two-stage and seed/policy choices unchanged.

Build logical input metadata only from the final [artifacts](../../../../glossary.md#term-artifact):
iterate `constants().bindableInputBindings()` in stored order, resolve each entry's `valueId()`
to its final `GraphValue.descriptor()`, and retain only the logical ID and exact [immutable](../../../../glossary.md#term-immutable)
[descriptor](../../../../glossary.md#term-descriptor) in the public input record. Do not reverse-map by descriptor equality, label, Tensor
allocation order, caller list position, or [capture](../../../../glossary.md#term-capture-meanings) traversal. Compiled handles may privately
retain the existing artifacts but must retain no caller Tensor, request list, [provenance](../../../../glossary.md#provenance), or
[host](../../../../glossary.md#term-host)-storage reference. Compile and [prepare](../../../../glossary.md#prepare) never call `Tensor.hostStorage()`.

A reused [leaf](../../../../glossary.md#term-leaf) has one input; equal-descriptor leaves remain separate. Compiler-created [constants](../../../../glossary.md#term-constant)
are omitted. Surviving explicit seed leaves are ordinary bindable inputs, not automatically
constants. Inputs retained by Compiler's [dead-code elimination](../../../../glossary.md#term-dead-code-elimination-dce) remain required even if unused;
Engine does not filter or reorder that contract. `inputs()` is an immutable ordered snapshot.

### [Prepare](../../../../glossary.md#prepare) and typed input [binding](../../../../glossary.md#term-binding)

Prepare validates the compiled handle's exact ordinary [Engine](../../../../glossary.md#term-engine) [owner](../../../../glossary.md#term-owner) before delegating once to
the existing [CPU](../../../../glossary.md#term-cpu) composition. It adds no new source/slot mapping, schedule assembly, allocation,
or preparation fallback. Each success returns a fresh [immutable](../../../../glossary.md#term-immutable) ordinary prepared handle.

[Run](../../../../glossary.md#term-run) accepts the required logical [Tensors](../../../../glossary.md#tensor) in any list order. [Validation](../../../../glossary.md#term-validation) order after [admission](../../../../glossary.md#term-admission) is:

1. Require non-null `preparedExecution`, then non-null `inputs`; validate the prepared handle's
   exact ordinary owner; snapshot the list and reject null elements.
2. Match by `Tensor.id()` value equality against the compiled input inventory. Reject duplicate
   supplied IDs, unexpected IDs, and missing IDs; a descriptor-equal different TensorId is not a
   replacement. Compare supplied descriptors with the compiled descriptor by complete value
   equality, including type, Shape, layout state/geometry, and <span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>`requiresGrad`.
3. After the complete logical validation passes, visit required inputs in compiled binding order
   and call each Tensor's synchronized `hostStorage()` exactly once. Retain those exact storage
   references for this run. Missing storage is an error; neither an earlier compile-time
   association nor another Tensor may supply it.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile))</span>
4. Validate all snapshots before borrowing: exact data type, sufficient capacity for a present
   layout's <span data-glossary-navigation="true"> ([borrowing glossary](../../../../glossary.md#term-borrowing); [data type glossary](../../../../glossary.md#data-type--datatype))</span>`referencedElementSpan()` (including zero), live scope, and current-thread segment
   accessibility. An absent layout does not imply contiguous geometry and does not license
   Engine to infer or resolve one. Prepared CPU compatibility remains an independent inward check.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>
5. Borrow one fresh wrapper per logical input through the existing CPU composition, in final
   binding order, then call the Runtime runner once with that ordered list. Distinct TensorIds
   may share one storage object; never deduplicate their wrappers because Runtime forbids repeated
   representation object identity. Read-only inputs are accepted where the prepared access allows.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span>

Wrong owner, duplicate/unexpected/missing ID, [descriptor](../../../../glossary.md#term-descriptor) mismatch, and logical type/span mismatch
throw `IllegalArgumentException`. Absent, dead, or currently inaccessible storage throws
`IllegalStateException`. Null references/[elements](../../../../glossary.md#term-element) throw `NullPointerException`. Logical binding
diagnostics identify the relevant [TensorId](../../../../glossary.md#tensorid) where one exists; count errors identify the expected
and actual counts. Do not establish a new public exception family or rewrite inward exceptions.

The [synchronization](../../../../glossary.md#term-synchronization) is a per-Tensor association [snapshot](../../../../glossary.md#term-snapshot), not an [atomic](../../../../glossary.md#resource-transaction-and-rollback) snapshot across the
entire input set, a memory fence for [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) writes, or [arena](../../../../glossary.md#term-arena) pinning. Replacing/clearing a Tensor
association after its snapshot cannot redirect that run. The exact [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) storage and its
bytes must remain valid, accessible where used, and free from conflicting caller mutation until
the result closes, including the interval after [synchronous](../../../../glossary.md#term-synchronous) run return. The caller owns every
[host storage](../../../../glossary.md#host-storage)/arena and Engine never closes, slices, reinterprets, copies, converts, or writes back
into that association. Engine owns only the fresh non-owning wrapper objects it creates.

Prepared handles can be shared by concurrent calls; each call snapshots independently and has one
isolated [Runtime](../../../../glossary.md#term-runtime) state. Shared stable host storage is allowed. To change associations for
different runs, callers must coordinate replacement and snapshots; this [API](../../../../glossary.md#term-application-programming-interface-api) promises no
independent `TensorId -> HostTensorStorage` per-run override.

### [Publication occurrence](../../../../glossary.md#publication-occurrence) and [alias](../../../../glossary.md#aliasing-and-arena) semantics

Privately construct [immutable](../../../../glossary.md#term-immutable) [publication](../../../../glossary.md#term-publication) specifications from `PublicationPlan`, resolving final
[descriptors](../../../../glossary.md#term-descriptor) through each [binding](../../../../glossary.md#term-binding)'s [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot). The order is all forward bindings followed by all
[gradient](../../../../glossary.md#term-gradient) bindings in their existing [derivative](../../../../glossary.md#term-derivative)-order/[target](../../../../glossary.md#term-target) order, never [graph](../../../../glossary.md#graph)-output-set order.
For each successful [run](../../../../glossary.md#term-run), allocate one fresh `RunResult.Publication` per [occurrence](../../../../glossary.md#term-occurrence):

| Property | FORWARD | GRADIENT |
|---|---|---|
| `index()` | Dense zero-based position in the complete result list | Dense zero-based position after the forward prefix |
| `tensorId()` | Original requested forward output ID | Requested [differentiation target](../../../../glossary.md#functional-gradient-request) ID, not a generated gradient [Tensor](../../../../glossary.md#tensor) ID |
| `descriptor()` | Final published forward value descriptor | Final gradient value descriptor, not blindly the target descriptor |
| `derivativeOrder()` | Empty | Present and exactly 1 for this ordinary overload |
| `targetIndex()` | Empty | Present zero-based position in the explicit target list |

Compare the completed [Runtime](../../../../glossary.md#term-runtime) result count with
`forwardBindings().size() + gradientBindings().size()` before returning any ordinary result.
Mismatch is an `IllegalStateException` and triggers full [rollback](../../../../glossary.md#term-rollback). Never compare with
`graph().outputs().size()`: the graph boundary deduplicates gradient values, while publication
roles deliberately retain all occurrences.

`publications()` returns one immutable list, and repeated queries return the same occurrence
objects in the same order. Every run has a different result and different occurrence objects.
Publications use [reference identity](../../../../glossary.md#term-reference-identity), not value equality, even when all descriptive fields match.
They retain their originating result [lease](../../../../glossary.md#term-lease) privately, have no independent close [operation](../../../../glossary.md#operation),
and expose no parent/delegate accessor. Closing one result invalidates only its own lease;
[Engine](../../../../glossary.md#term-engine) closure closes every remaining lease. Publication `isClosed()` delegates to its result
and becomes true when closure starts. All immutable [metadata](../../../../glossary.md#term-metadata), including `publications()`,
`resultCount()`, and publication accessors, remains readable after closure.

Two roles may internally select the same exact Runtime representation, including forward/gradient
or repeated-gradient aliases. Preserve every role without copying, merging, or deduplication.
Logical [identity](../../../../glossary.md#term-identity-meanings), list position, and Java occurrence identity do not assert physical [aliasing](../../../../glossary.md#term-aliasing) or
independent storage. No alias-query or value/[host](../../../../glossary.md#term-host)-storage accessor is added, and no result is
attached to a [Model](../../../../glossary.md#term-model) Tensor or a [mutable](../../../../glossary.md#term-mutable) Tensor gradient field.

### [Admission](../../../../glossary.md#term-admission), [ownership](../../../../glossary.md#term-ownership-meanings), and cleanup

Reuse AdvancedEngine's single [lifecycle](../../../../glossary.md#lifecycle) lock, active-[operation](../../../../glossary.md#operation) accounting, open-result [registry](../../../../glossary.md#term-registry),
reverse result-close order, and retained close failure. Ordinary [Engine](../../../../glossary.md#term-engine) keeps its one private
final advanced delegate. Add narrow package-private ordinary entry points on that delegate;
perform their entire ordinary [validation](../../../../glossary.md#term-validation), conversion, storage [snapshot](../../../../glossary.md#term-snapshot)/[borrowing](../../../../glossary.md#term-borrowing), inward work,
count checking, and outward handle construction inside one admission. Factor private ungated work
where useful; do not nest public advanced calls or introduce a second gate, second registry,
[preflight](../../../../glossary.md#term-preflight) `isClosed()` [race](../../../../glossary.md#term-race), generic [callback](../../../../glossary.md#term-callback) service, or [service locator](../../../../glossary.md#term-service-locator).

Preserve the exact existing closed-[owner](../../../../glossary.md#term-owner) failure,
`IllegalStateException("advanced engine is closed")`, for both surfaces. Once closure begins,
this failure precedes null, membership, storage, and inward validation. A previously admitted
operation may finish its work, but cannot return a newly usable handle/result if closure won
before final [publication](../../../../glossary.md#term-publication). Register a complete result and [release](../../../../glossary.md#term-release) admission atomically under the
same existing lock; failed construction, mapping, or registration [releases](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) admission once.

Extend private `AdvancedRunResult` construction to own an [immutable](../../../../glossary.md#term-immutable) list of ordinary-created
borrow wrappers; the existing advanced [run](../../../../glossary.md#term-run) path supplies an empty list and its public [signatures](../../../../glossary.md#term-signature)
and caller-ownership semantics remain unchanged. Its one close protocol first attempts to close
the exact [Runtime](../../../../glossary.md#term-runtime) result, then attempts every owned wrapper once in reverse borrow order, and
unregisters even on failure. Ordinary `RunResult.close()` delegates to that same registered owner.
Engine auto-close therefore performs the same cleanup as explicit ordinary result-close.

Before ownership transfers to the registered result, Engine owns the acquired wrappers. A failed
borrow closes only earlier wrappers in reverse order. A failed Runtime run relies on the
runner's established state [rollback](../../../../glossary.md#term-rollback), then closes the wrappers. Failure after successful Runtime
return (count check, wrapper/[metadata](../../../../glossary.md#term-metadata) construction, or closing-owner rejection) closes that exact
Runtime result before the wrappers. Retain the original unchecked failure object and suppress
distinct cleanup failures; cleanup attempts all resources and never self-suppresses. No failed
run returns a partial result. Avoid double close when ownership has already moved to the result.

Concurrent/repeated result and Engine closes keep the existing wait-uninterruptibly-and-restore-
interrupt behavior and return/rethrow the same retained cleanup outcome. Metadata access is safe
concurrently with closure; do not expose the non-[thread-safe](../../../../glossary.md#term-thread-safe) inward result. Separate runs share
no [mutable](../../../../glossary.md#term-mutable) [run state](../../../../glossary.md#run-state--runstate). No [cancellation](../../../../glossary.md#term-cancellation), [asynchronous](../../../../glossary.md#term-asynchronous) execution, or new persistent [prepared
resource](../../../../glossary.md#prepared-resource--preparedresource) lifecycle is introduced.

### Current limits that must remain explicit

The supported [CPU](../../../../glossary.md#term-cpu) composition accepts exactly one non-empty maximal CPU [partition](../../../../glossary.md#partition); zero-[node](../../../../glossary.md#node)/
pass-through, mixed-[owner](../../../../glossary.md#term-owner), and multi-partition preparation remain rejected. Unsupported or
dynamic work still fails at the existing responsible boundary, without [Engine](../../../../glossary.md#term-engine) [operation](../../../../glossary.md#operation) switches.
A successful [compile](../../../../glossary.md#compile) is not a guarantee of successful CPU preparation or execution.

CPU may reject a retained unused caller input without a matching [buffer](../../../../glossary.md#term-buffer) declaration. CPU cold
[binding](../../../../glossary.md#term-binding) also currently requires exact prepared byte size and compatible [carrier](../../../../glossary.md#term-storage-carrier)/[alignment](../../../../glossary.md#term-alignment);
[Model](../../../../glossary.md#term-model)'s sufficient-span rule alone does not guarantee that oversized storage executes.
Do not drop inputs or [slice](../../../../glossary.md#slice) storage to make these cases pass. No inward production modification
is authorized to broaden them. Future [host](../../../../glossary.md#term-host)/value access requires an explicit supported
[Runtime](../../../../glossary.md#term-runtime)/[backend](../../../../glossary.md#backend) bridge in task 0004, not reflection into leased representations here.

The integration [fixture](../../../../glossary.md#term-fixture) must not use public ADD merely because CPU has an ADD [kernel](../../../../glossary.md#kernel): public ADD
construction leaves the ADD output [layout](../../../../glossary.md#layout) unresolved, so the complete [occurrence](../../../../glossary.md#term-occurrence) is not currently
hard-eligible. A following [CONTIGUOUS request](../../../../glossary.md#contiguous-request) does not rewrite that [descriptor](../../../../glossary.md#term-descriptor). A [provenance](../../../../glossary.md#provenance)-free
explicit [seed](../../../../glossary.md#term-seed) [leaf](../../../../glossary.md#term-leaf) also cannot serve directly as the repeated published [gradient](../../../../glossary.md#term-gradient) in this CPU
composition because schedule assembly has no [prepared buffer assignment](../../../../glossary.md#prepared-buffer-assignment--preparedbufferassignment) for that pass-through
[publication](../../../../glossary.md#term-publication). The corrected fixture makes the shared seed an independently produced [CONTIGUOUS](../../../../glossary.md#term-contiguous)
result over a resolved leaf, so every published value has a prepared assignment without adding
pass-through support.

## Out of [scope](../../../../glossary.md#term-scope)

- Any Java change outside Engine, new Gradle dependency, architecture rule, ADR, shared contract,
  CPU route, preparation mapping, Runtime publication/access, or backend conformance behavior.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [CPU glossary](../../../../glossary.md#term-cpu); [Runtime glossary](../../../../glossary.md#term-runtime); [publication glossary](../../../../glossary.md#term-publication); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Host values, arrays, segments, scalar/element access, copies/transfers, detached results,
  Tensor result attachment, publication policy, or persistence/NN/Training adapters.<span data-glossary-navigation="true"> ([Host glossary](../../../../glossary.md#term-host); [scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element); [detached glossary](../../../../glossary.md#detached-model-results); [Tensor glossary](../../../../glossary.md#tensor); [persistence glossary](../../../../glossary.md#term-persistence); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context))</span>
- Independent storage-binding overrides, symbolic binding/inference, graph traversal,
  kernel selection, backend discovery, mixed-backend assembly, tuning, caches, or tracing payloads.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [inference glossary](../../../../glossary.md#term-inference); [graph glossary](../../../../glossary.md#graph); [kernel glossary](../../../../glossary.md#kernel))</span>
- New Config aggregates, ordinary compile policy tuning, higher-order request ergonomics,
  scalar default seeds, target inference, <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [target glossary](../../../../glossary.md#term-target))</span>`output.execute()`, `withBackward()`, or one-shot work.
- Creating task 0004 or another future detailed specification during this implementation.

## Architecture constraints and no-change conclusions

[Engine](../../../../glossary.md#term-engine) is the outer composition and logical-[binding](../../../../glossary.md#term-binding) layer. [Compiler](../../../../glossary.md#term-compiler) owns [identity](../../../../glossary.md#term-identity-meanings) [provenance](../../../../glossary.md#provenance),
[graph](../../../../glossary.md#graph) traversal/[autograd](../../../../glossary.md#autograd), and final [publication](../../../../glossary.md#term-publication) facts; [Prepare](../../../../glossary.md#prepare) owns graph-value-to-slot mapping;
[Runtime](../../../../glossary.md#term-runtime) owns isolated execution state and logical cleanup; [CPU](../../../../glossary.md#term-cpu) owns physical compatibility,
storage mechanics, and [lowering](../../../../glossary.md#lowering). Only documented public integration collaborators may cross
these boundaries. Cold logical ID [lookup](../../../../glossary.md#term-lookup) is permitted; no new per-[element](../../../../glossary.md#term-element) or hot-path lookup,
reflection, boxing, graph [dispatch](../../../../glossary.md#term-dispatch), storage copy, or executable work is introduced.

The implementation audit must explicitly reaffirm: no changes to `ARCHITECTURE.md`, focused
architecture decisions/rules, [Model](../../../../glossary.md#term-model), Compiler, Config, [Planning](../../../../glossary.md#term-planning), shared Prepare, Runtime, CPU or
other [backends](../../../../glossary.md#backend), [Gradle](../../../../glossary.md#term-gradle), backend [conformance](../../../../glossary.md#term-conformance), [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), Training, [persistence](../../../../glossary.md#term-persistence), or tuning production.
Architecture-test source remains unchanged because no [dependency](../../../../glossary.md#term-dependency) rule changes; its focused
existing Engine inventory test is still [run](../../../../glossary.md#term-run). Integration-test source changes because ordinary
end-to-end orchestration is new. [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api) [Javadoc](../../../../glossary.md#term-javadoc) remains accurate because Tensor behavior is
unchanged. Training API status needs a narrow documentation correction for ordinary [gradient](../../../../glossary.md#term-gradient)
publication [metadata](../../../../glossary.md#term-metadata), not an [optimizer](../../../../glossary.md#term-optimizer) or materialized-gradient claim. Other user guides and
architecture explanations remain conceptual; record targeted no-change conclusions unless a
task-specific contradiction makes the fixed documentation [scope](../../../../glossary.md#term-scope) insufficient, in which case
stop and request a scope correction rather than silently expanding it.

## Exact affected files and hard ceiling

Paths below are relative to repository root. These 19 paths are the entire implementation
allowlist, not permission to overwrite pre-existing changes.

Production (seven paths), all in `modules/engine/src/main/java/io/github/pho001/synaptik/engine/`:

1. `Engine.java` — four ordinary method additions, preserving standard construction and delegate.
2. `AdvancedEngine.java` — bounded package-private ordinary [admission](../../../../glossary.md#term-admission) paths and [ownership](../../../../glossary.md#term-ownership-meanings) [handoff](../../../../glossary.md#term-handoff).
3. `AdvancedRunResult.java` — private ownership/cleanup of ordinary-created borrow wrappers.
4. New `CompiledGraph.java` — [owner](../../../../glossary.md#term-owner)-bound recipe and nested logical input [metadata](../../../../glossary.md#term-metadata).
5. New `PreparedExecution.java` — owner-bound prepared recipe and originating [compile](../../../../glossary.md#compile) handle.
6. New `RunResult.java` — typed result [lease](../../../../glossary.md#term-lease), nested [publication](../../../../glossary.md#term-publication) occurrences and roles.
7. `package-info.java` — current ordinary/advanced and planned [materialization](../../../../glossary.md#materialization) boundaries.

Tests (four paths):

8. New `modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineTypedLifecycleTest.java`.
9. New `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/EngineTypedPublicShapeTest.java`.
10. Existing `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/AdvancedEnginePublicShapeTest.java`
    — update only the former construction-only Engine method assertion; retain advanced assertions.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
11. New `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineTypedLifecycleIntegrationTest.java`.

Documentation and [planning](../../../../glossary.md#term-planning) (eight paths):

12. `docs/api/public-api.md`.
13. `docs/api/compile-api.md`.
14. `docs/api/runtime-api.md`.
15. `docs/api/training-api.md`.
16. `docs/glossary.md`.
17. This task.
18. `docs/planning/modules/engine/master-plan.md`.
19. `docs/planning/roadmap.md`.

The hard ceiling is 19 changed/created paths attributable to this task: seven production,
four tests, five explanatory docs, and three planning docs. This small exception to the usual
12–18-path guardrail is [atomic](../../../../glossary.md#resource-transaction-and-rollback): the same [lifecycle](../../../../glossary.md#lifecycle) decision needs its compile/prepared/result
types, shared admission/cleanup, ordinary and advanced [regression](../../../../glossary.md#term-regression) assertions, one end-to-end
[fixture](../../../../glossary.md#term-fixture), and synchronized public/compile/[runtime](../../../../glossary.md#term-runtime)/training/glossary status. Splitting those into
mechanical tasks would leave the single public capability undocumented or unvalidated.
No second production [module](../../../../glossary.md#term-module) is implemented and no unrelated refactor is permitted. If an
additional helper file, [dependency](../../../../glossary.md#term-dependency), or inward contract is genuinely required, stop and replan
before editing it. Keep handle metadata logic in the handle types and admission in the lifecycle
owner rather than creating a catch-all [facade](../../../../glossary.md#term-facade).

## Acceptance criteria

- The exact ordinary API compiles from a distinct package using only Engine, Model, and JDK
  imports. It contains no inward/advanced type escape, public handle construction, extra overload,
  config aggregate, value accessor, or scalar/one-shot backward convenience.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Engine glossary](../../../../glossary.md#term-engine); [Model glossary](../../../../glossary.md#term-model); [JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- One-list forward and explicitly seeded three-list first-order requests delegate once with the
  fixed settings; semantic failures retain inward ownership and identities.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Final bindable identities/descriptors/order come only from Compiler 0006B4, without Tensor
  retention or storage reads during compile/prepare. Arbitrary supplied input order is accepted;
  every required ID is supplied exactly once and descriptors/storage are validated as specified.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [Compiler glossary](../../../../glossary.md#term-compiler); [Tensor glossary](../../../../glossary.md#tensor); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare))</span>
- Each run snapshots associations once, borrows fresh wrappers in final order, preserves shared
  underlying storage, and creates exactly one isolated Runtime state. Caller storage survives all
  success/failure/close paths.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Result count and metadata match every forward-then-gradient occurrence, including repeated
  gradient values and forward/gradient aliases. Identity, derivative role, target index,
  descriptor, immutability, per-run identity, and post-close inspection match the specified API.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [gradient glossary](../../../../glossary.md#term-gradient); [occurrence glossary](../../../../glossary.md#term-occurrence); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [Identity glossary](../../../../glossary.md#term-identity-meanings); [derivative glossary](../../../../glossary.md#term-derivative); [target glossary](../../../../glossary.md#term-target))</span>
- The real public CPU integration uses only static resolved FLOAT32 leaves and CONTIGUOUS
  occurrences. Its seeded case publishes two identity-distinct gradient targets backed by one
  repeated produced seed value; broader forward/gradient alias cases remain covered by the
  package-local Engine publication fixtures and are not presented as CPU support that the current
  public expression path cannot execute.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [FLOAT32 glossary](../../../../glossary.md#term-float32); [CONTIGUOUS glossary](../../../../glossary.md#term-contiguous); [seed glossary](../../../../glossary.md#term-seed); [publication glossary](../../../../glossary.md#term-publication))</span>
- Exact ordinary owner checks and shared admission prevent shutdown races, partial publication,
  leaked Runtime results/wrappers, double cleanup, and failure replacement. Advanced public
  behavior and standard construction remain intact.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [admission glossary](../../../../glossary.md#term-admission))</span>
- Current CPU limitations are tested/documented without inward fixes. Final tests, Javadoc,
  documentation review, scope checks, and whitespace validation pass; all no-change conclusions
  and evidence are recorded before marking Complete.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope); [validation glossary](../../../../glossary.md#term-validation))</span>

## Focused test design

Use package-local composition fakes for controlled ordering, resource counters, failures, and
latches, following the existing [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle) tests. Use valid public [compile artifacts](../../../../glossary.md#compile-artifacts) and
prepared [Runtime](../../../../glossary.md#term-runtime) fixtures; no production reflection, new service interface, or fake [CPU](../../../../glossary.md#term-cpu) success
claim. Package-private handle constructors may supply a deliberately wrong-[descriptor](../../../../glossary.md#term-descriptor) input
[metadata](../../../../glossary.md#term-metadata) [fixture](../../../../glossary.md#term-fixture) to test defense against malformed mappings without forging [Model](../../../../glossary.md#term-model) [Tensor](../../../../glossary.md#tensor) IDs.

- Compile/input metadata: forward request order; two equal-descriptor distinct leaves; repeated
  use of one leaf; immutable lists and input-record null checks; no host association required to
  compile/prepare; explicit seed leaves and seed expressions; compiler-generated constants omitted;
  exact gradient target order; duplicate/null/empty requests and invalid seed/target rejection.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [host glossary](../../../../glossary.md#term-host); [prepare glossary](../../../../glossary.md#prepare); [seed glossary](../../../../glossary.md#term-seed); [compiler glossary](../../../../glossary.md#term-compiler); [constants glossary](../../../../glossary.md#term-constant))</span>
- Binding: supplied input order reversed; missing/extra/duplicate IDs; wrong ordinary owner even
  if a package-local test shares one advanced delegate; descriptor mismatch; absent, dead,
  inaccessible, wrong-carrier, insufficient/incompatible storage; no borrow before all logical
  and snapshot checks finish. Model already rejects some malformed associations at attachment:
  test those at that owning boundary or an Engine-owned metadata fixture, not by mutating Model
  internals to invent a reachable ordinary input.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [carrier glossary](../../../../glossary.md#term-storage-carrier); [snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Snapshots/lifetime: a latch-controlled first borrow permits association replacement after all
  snapshots; prove the exact old references were used and later runs take fresh snapshots.
  Distinct inputs sharing storage get distinct wrappers. Read-only input succeeds for a read-only
  prepared role. Caller arenas remain live after result/Engine close; no copy or reassociation.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [latch glossary](../../../../glossary.md#control-flow-graph); [arenas glossary](../../../../glossary.md#term-arena); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- Publications: ordinary forward success and explicitly seeded first-order success; repeated
  gradient-value roles and forward/gradient alias roles remain separate; target ID differs from
  generated gradient identity; immutable metadata, stable repeated accessor identity, fresh objects
  per run, and readable metadata with closed lifecycle after result/Engine closure.<span data-glossary-navigation="true"> ([alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Failure/cleanup: partial borrow; runner creation/cold-bind/execute/publication failure; an
  intentionally wrong publication count returned by the fake prepared schedule; failure after
  Runtime success; cleanup failure suppression; reverse wrapper close; reverse result close before
  backend close; no partial result. Counters prove exact once and original exception/error identity.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [prepared schedule glossary](../../../../glossary.md#prepared-schedule--preparedschedule); [backend glossary](../../../../glossary.md#backend))</span>
- Concurrency: independent repeated and concurrent runs, close racing admitted compile/prepare/run
  and logical mapping, closed-before-invalid-input precedence, repeated/concurrent result and
  Engine close, interruption restoration, and retained cleanup failure. Use bounded latch waits,
  not timing sleeps or nondeterministic stress.<span data-glossary-navigation="true"> ([Concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- Public shape: automate exact methods, generic signatures, no public/protected constructors
  except the input record, nested enum/record/final types, ordinary forbidden type closure, and
  absence of value/materialization/convenience methods. Extend existing advanced shape assertion
  only for Engine's four additions; no repeated manual javap/reflection checklist.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [signatures glossary](../../../../glossary.md#term-signature); [materialization glossary](../../../../glossary.md#materialization))</span>
- Integration: use only ordinary Engine, Model, and [JDK](../../../../glossary.md#term-java-development-kit-jdk) imports through real `Engine.standard()`.
  Use exact-size, suitably aligned `MemorySegmentStorage` for static resolved FLOAT32 leaves so the
  test exercises the current CPU carrier contract rather than depending on an unselected heap-array
  route. The forward-only case uses two identity-distinct leaves and outputs<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [heap glossary](../../../../glossary.md#term-heap))</span>
  `[left.contiguous(), right.contiguous()]`; [compile](../../../../glossary.md#compile) must report inputs `[left, right]`, run must
  accept `[right, left]`, and the two metadata-only publications must retain forward order.

  The explicitly seeded case is exact: create [identity](../../../../glossary.md#term-identity-meanings)-distinct [gradient](../../../../glossary.md#term-gradient)-eligible `left` and
  `right` leaves plus a non-gradient `seedLeaf`, all with the same static resolved FLOAT32 Shape
  and layout geometry and with gradient eligibility as just specified; create<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
  `leftOutput = left.contiguous()`,
  `rightOutput = right.contiguous()`, and `seed = seedLeaf.contiguous()`; then call
  `compile([leftOutput, rightOutput], [seed, seed], [left, right])`. Compile must report bindable
  inputs `[left, right, seedLeaf]`, and [run](../../../../glossary.md#term-run) must accept the reversed [binding](../../../../glossary.md#term-binding) order
  `[seedLeaf, right, left]`. The result must contain exactly four metadata-only occurrences in
  order: forward `leftOutput`, forward `rightOutput`, first-order [target](../../../../glossary.md#term-target) `left` at target index
  zero, and first-order target `right` at target index one. The two gradient roles must remain
  separate identity-distinct publication objects while Compiler/Prepare/Runtime retain their one
  repeated produced seed value. Exercise a second run as needed to prove per-run occurrence
  identity and close behavior. Assert the current zero-node prepare rejection separately. Do not
  inspect numerical results, physical aliasing, backend storage, or inward result access.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [node glossary](../../../../glossary.md#node); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>

## [Validation](../../../../glossary.md#term-validation) plan

This is task-tier validation: [Engine](../../../../glossary.md#term-engine)-only production behavior plus one focused end-to-end [fixture](../../../../glossary.md#term-fixture),
with unchanged dependencies and shared contracts. [Run](../../../../glossary.md#term-run) focused tests during development as needed,
then record one final pass after executable Java stabilizes:

```bash
./gradlew :modules:engine:test
./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.EngineTypedLifecycleIntegrationTest
./gradlew :testing:architecture-tests:test --tests io.github.pho001.synaptik.testing.architecture.EngineCompositionContractTest
```

The failed ADD integration attempts from implementation context
`01a0a46c-8a9e-7862-bd81-0d3495092894` are corrective [planning](../../../../glossary.md#term-planning) evidence, not accepted final test
failures and not a reason to rerun or weaken the 22 passing Engine tests. Replace both ADD fixtures
with the exact [CONTIGUOUS](../../../../glossary.md#term-contiguous) fixtures above, then rerun the focused integration class once. Record its
final test count and confirm that the seeded case reaches real [CPU](../../../../glossary.md#term-cpu) [prepare](../../../../glossary.md#prepare) and run, not merely
[compile](../../../../glossary.md#compile) or [metadata](../../../../glossary.md#term-metadata) construction. Do not authorize or test an inward ADD/[layout](../../../../glossary.md#layout) workaround.

The independent documentation pass reuses those successful results and runs final:

```bash
./gradlew :modules:engine:javadoc
git diff --check
```

Validate changed Markdown links/anchors, headings, fences, whitespace, final newlines, current vs
planned claims, exact 19-path ceiling relative to the starting dirty-worktree [snapshot](../../../../glossary.md#term-snapshot), and
0003/master/roadmap status agreement. Verify 0004–0008 stay Draft with no new task 0004 spec.
The public-[shape](../../../../glossary.md#shape) test owns recurring reflection/import boundary assertions; inspect generated
[Javadoc](../../../../glossary.md#term-javadoc) for all new types/members and revised [lifecycle](../../../../glossary.md#lifecycle) contracts once after finalization.
No repeated successful Java suites in the documentation or coordinator context. Root `test`
remains Engine 0008/[CI](../../../../glossary.md#term-continuous-integration-ci) [checkpoint](../../../../glossary.md#term-checkpoint) work unless new evidence proves shared-contract, [dependency](../../../../glossary.md#term-dependency),
or second-production-[module](../../../../glossary.md#term-module) impact; such impact requires replan before implementation.

## Documentation [handoff](../../../../glossary.md#term-handoff)

Document type: [Planning](../../../../glossary.md#term-planning) for this spec/master/roadmap; [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc) for Java contracts and API
references; General for glossary prose; Example for [lifecycle](../../../../glossary.md#lifecycle) snippets. The clean implementation
agent may draft Javadoc, but a distinct clean documentation-focused context must independently
finalize all seven affected production Javadocs, five explanatory documents, and the three
planning records in the same overall change.

The handoff must include final source/tests, exact public API, successful test commands/results,
dirty-worktree baseline, this task, directly relevant [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/[CPU](../../../../glossary.md#term-cpu)
contracts, and documentation rules/profiles. Explain [bindable input](../../../../glossary.md#bindable-input), association [snapshot](../../../../glossary.md#term-snapshot),
[cotangent](../../../../glossary.md#cotangent), [publication occurrence](../../../../glossary.md#publication-occurrence), [target](../../../../glossary.md#term-target) [identity](../../../../glossary.md#term-identity-meanings), and result [lease](../../../../glossary.md#term-lease) at first use. Glossary entries
must distinguish ordinary [Engine](../../../../glossary.md#term-engine) `CompiledGraph`/[`PreparedExecution`](../../../../glossary.md#prepared-execution--preparedexecution)/[`RunResult`](../../../../glossary.md#run-result--runresult) from Model
[graph](../../../../glossary.md#graph) and Runtime recipe/lease types. Correct the affected API pages' stale construction-only,
[gradient](../../../../glossary.md#term-gradient)-[publication](../../../../glossary.md#term-publication), and [borrowed](../../../../glossary.md#ownership-borrowing-and-lease)-input [lifetime](../../../../glossary.md#term-lifetime) wording. Do not promise [host](../../../../glossary.md#term-host) values or training.

Include a complete ordinary forward example and a seeded-gradient example with imports,
resolved supported [descriptors](../../../../glossary.md#term-descriptor), storage setup, explicit [ownership](../../../../glossary.md#term-ownership-meanings), arbitrary [binding](../../../../glossary.md#term-binding) order,
[metadata](../../../../glossary.md#term-metadata) inspection, and try-with-resources cleanup. Exercise their essential call paths in the
public [integration test](../../../../glossary.md#term-integration-test). Explain that task 0004 adds values and task 0006 later adds [scalar](../../../../glossary.md#term-scalar)
convenience with explicit targets, rather than presenting this lower-level [seed](../../../../glossary.md#term-seed) overload as the
final one-shot user experience.

The docs agent does not rerun successful Java suites unless it changes executable Java or names
a concrete risk. It runs final Javadoc and documentation checks, records reviewed unchanged
contracts, and returns its own completion status. If any required item is incomplete, this task
cannot be marked Complete.

## Implementation prompt

> Implement Engine task 0003 exactly as specified here in a new clean implementation context.
> Read the required contracts and inspect/preserve the existing dirty worktree. No GSD, commit,
> push, architecture invention, or out-of-scope edit. Keep the exact ordinary API, TensorId
> mapping, single admission/cleanup protocol, metadata-only publication semantics, CPU limits,
> corrected CONTIGUOUS-only public integration fixtures, and 19-path ceiling. Do not retain the
> failed ADD assumption or add pass-through publication support. Use apply_patch. Run task-tier
> validation once after code stabilizes,
> then hand the same change and evidence to a distinct clean documentation-focused agent.
> Stop and report any architecture or scope gap. Do not mark Complete before code, tests,
> independent documentation, Javadoc, and final checks pass; leave 0004–0008 Draft.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [commit glossary](../../../../glossary.md#term-commit); [scope glossary](../../../../glossary.md#term-scope); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [TensorId glossary](../../../../glossary.md#tensorid); [admission glossary](../../../../glossary.md#term-admission); [metadata glossary](../../../../glossary.md#term-metadata); [publication glossary](../../../../glossary.md#term-publication); [CPU glossary](../../../../glossary.md#term-cpu); [CONTIGUOUS glossary](../../../../glossary.md#term-contiguous); [Run glossary](../../../../glossary.md#term-run); [validation glossary](../../../../glossary.md#term-validation); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Local decisions and follow-up

- The historical blocker was the missing originating Tensor identity; completed Compiler 0006B4
  closes it without changing artifacts' component count or Prepare/Runtime/CPU production.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [identity glossary](../../../../glossary.md#term-identity-meanings); [Compiler glossary](../../../../glossary.md#term-compiler); [artifacts glossary](../../../../glossary.md#term-artifact); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Choose association-only `List<Tensor>` [binding](../../../../glossary.md#term-binding) and [immutable](../../../../glossary.md#term-immutable) `CompiledGraph.Input` metadata.
  A storage override carrier is not needed for this capability.<span data-glossary-navigation="true"> ([metadata glossary](../../../../glossary.md#term-metadata); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Choose an explicit one-stage three-list compile overload, not a new request/config abstraction.
  Full Compiler request choices remain advanced; scalar and one-shot ergonomics remain later.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Choose fresh identity-based publication handles and metadata-readable-after-close semantics.
  Aliases remain occurrences, not inferred physical ownership or materialized Tensor values.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [Aliases glossary](../../../../glossary.md#aliasing-and-arena); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Choose the exact CONTIGUOUS integration fixture recorded above. The shared explicit seed is a
  CONTIGUOUS expression rather than a leaf so its repeated gradient value has a prepared CPU buffer
  assignment. This proves repeated gradient-publication occurrences for two identity-distinct
  targets. Keep the more general forward/gradient alias proof in package-local Engine tests; do
  not invent public ADD eligibility or CPU pass-through publication support for integration.<span data-glossary-navigation="true"> ([CONTIGUOUS glossary](../../../../glossary.md#term-contiguous); [fixture glossary](../../../../glossary.md#term-fixture); [seed glossary](../../../../glossary.md#term-seed); [leaf glossary](../../../../glossary.md#term-leaf); [gradient glossary](../../../../glossary.md#term-gradient); [buffer glossary](../../../../glossary.md#term-buffer); [targets glossary](../../../../glossary.md#term-target); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Extend the existing private admission/result owner rather than wrapping separately admitted
  public calls. Borrow wrappers are Engine-owned; storage remains caller-owned.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [owner glossary](../../../../glossary.md#term-owner))</span>
- No architecture decision changes. The 19-path exception is justified above; no tasks run out
  of order, and the planning seam audit changes only the three synchronized planning files.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [planning glossary](../../../../glossary.md#term-planning))</span>
- Task 0004 must decide the supported representation-to-host access bridge, selected publication
  addressing, payload bounds, copy/lease ownership, and cleanup before implementation. It must
  not treat the metadata-only lease as an existing value API.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [payload glossary](../../../../glossary.md#artifact-envelope-and-atomic-publication); [lease glossary](../../../../glossary.md#term-lease); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Tasks 0005–0006 build over 0003–0004; no exact future convenience signatures are fixed here.
  Task 0007 remains optional tuning, and 0008 remains the capability checkpoint.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Completion summary

- Completed changes: Added the ordinary [owner](../../../../glossary.md#term-owner)-bound [compile](../../../../glossary.md#compile) and [prepare](../../../../glossary.md#prepare) handles, arbitrary-order
  [`TensorId`](../../../../glossary.md#tensorid) input matching, per-[run](../../../../glossary.md#term-run) caller-owned `HostTensorStorage` snapshots, synchronous run,
  and metadata-only forward/gradient publication occurrences. Publication metadata preserves
  requested output or target identity, target position, derivative order, occurrence multiplicity,
  and inward aliasing without exposing physical alias identity or values. Corrected real CPU
  coverage uses only CONTIGUOUS over resolved FLOAT32 leaves and<span data-glossary-navigation="true"> ([synchronous glossary](../../../../glossary.md#term-synchronous); [metadata glossary](../../../../glossary.md#term-metadata); [gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [target glossary](../../../../glossary.md#term-target); [identity glossary](../../../../glossary.md#term-identity-meanings); [derivative glossary](../../../../glossary.md#term-derivative); [occurrence glossary](../../../../glossary.md#term-occurrence); [aliasing glossary](../../../../glossary.md#term-aliasing); [alias glossary](../../../../glossary.md#aliasing-and-arena); [CONTIGUOUS glossary](../../../../glossary.md#term-contiguous); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>
  `seedLeaf.contiguous()` as the shared explicit [seed](../../../../glossary.md#term-seed).
- Files changed or created: exactly the seven production, four test, five explanatory, and three
  planning paths listed in <span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>[Exact affected files and hard ceiling](#exact-affected-files-and-hard-ceiling).
  The shared worktree's Compiler 0006B4 files and its additions in <span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>`public-api.md`,
  `compile-api.md`, `glossary.md`, and `roadmap.md` were preserved.
- Implementation [validation](../../../../glossary.md#term-validation): Reused stable evidence from implementation context
  `01a0a46c-8a9e-7862-bd81-0d3495092894`: [Engine](../../../../glossary.md#term-engine) production compiled;
  `./gradlew :modules:engine:test` passed 22 tests; corrected
  `EngineTypedLifecycleIntegrationTest` passed 3 tests through the real [CPU](../../../../glossary.md#term-cpu) composition; focused
  `EngineCompositionContractTest` passed 1 test; exact 11-Java-path, trailing-whitespace, and
  `git diff --check` checks passed. No executable Java changed afterward.
- Documentation-agent review: Clean documentation context
  `01a0a488-b8dd-7bf1-a8fc-947f531f420f` independently reviewed and finalized all seven affected
  production Javadocs plus the eight documentation/planning paths. It applied the General,
  API/Javadoc, Planning, and Example profiles and changed no executable Java token.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [token glossary](../../../../glossary.md#term-token))</span>
- Documentation validation: `./gradlew :modules:engine:javadoc` passed after final Javadoc edits
  with no warnings. Generated pages for the package and all six affected types were inspected.
  Exact `javap -public` inspection confirmed the ordinary and advanced public shapes. The complete
  ordinary forward and seeded-gradient examples compiled with <span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape))</span>`javac` against current module
  classes. A targeted checker passed local links and anchors, heading syntax, balanced fences,
  trailing whitespace, and final newlines for all eight Markdown paths. Current/planned wording,
  0003/master/roadmap status, all five Draft 0004–0008 rows, absence of an Engine 0004 task spec,
  exact cumulative 19-path task scope, and final <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed.
- Documentation impact: Public, Compile, Runtime, and Training API pages now describe the current
  ordinary surface, precise caller-storage/result lifetimes, corrected CONTIGUOUS examples, and
  metadata-only gradient boundary. The glossary now distinguishes bindable identity metadata,
  caller-input association snapshots, Engine facade handles/results, and publication occurrences
  from Model graph and Runtime recipe/lease contracts.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [facade glossary](../../../../glossary.md#term-facade); [Model glossary](../../../../glossary.md#term-model); [graph glossary](../../../../glossary.md#graph); [lease glossary](../../../../glossary.md#term-lease))</span>
- Javadoc review: All seven affected production paths now state identity, ownership, lifecycle,
  closure, alias, and current CPU-preparation boundaries. Parameter, return, and expected failure
  documentation remains complete; no executable behavior was altered.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [lifecycle glossary](../../../../glossary.md#lifecycle); [Parameter glossary](../../../../glossary.md#term-parameter))</span>
- No-change conclusions: `ARCHITECTURE.md` and focused architecture pages require no change because
  task 0003 realizes the existing Engine composition and typed-binding ownership without changing
  a module boundary. Build files and dependencies are unchanged. Model Tensor/storage, Compiler
  0006B4, Prepare, Runtime, and CPU production and documentation outside the allowlist remain
  accurate because their contracts and behavior did not change. Training behavior remains absent;
  only its API status now acknowledges metadata-only Engine gradient occurrences. Other modules,
  architecture tests, backend conformance, and additional integration coverage require no change.
  Root tests were not run because this is a task-tier Engine capability with unchanged executable
  behavior, dependencies, and shared contracts; Engine task 0008/CI retains the checkpoint.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [Tensor glossary](../../../../glossary.md#tensor); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [CI glossary](../../../../glossary.md#term-continuous-integration-ci); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Unresolved issues: None within task 0003. Host values remain task 0004; one-shot forward and
  scalar-objective backward convenience remain tasks 0005–0006. Tasks 0004–0008 remain Draft,
  and no task 0004 specification was created.<span data-glossary-navigation="true"> ([Host glossary](../../../../glossary.md#term-host); [scalar glossary](../../../../glossary.md#term-scalar); [objective glossary](../../../../glossary.md#term-objective))</span>
- Follow-up required: None for task 0003.

Status: Complete
