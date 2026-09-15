# Task 0003: Typed Logical Input Binding and Published-Result Access

## Status

Complete

## Goal

Deliver the ordinary Engine-owned `compile -> prepare -> run` lifecycle for the current CPU
composition. Bind caller Tensors by logical `TensorId`, using their current borrowed host-storage
associations, and expose every ordered forward and first-order gradient publication as typed
logical metadata with a result lease. Ordinary callers must not construct a CPU adapter or supply
Compiler identities, Runtime positions, or representations.

This task exposes publication identity and lifecycle, not numerical values. Task 0004 owns host
materialization and value access; tasks 0005–0006 own one-shot forward and scalar-objective
backward convenience.

## Dependencies and readiness

- Engine [0001](0001-advanced-composition-and-representation-level-lifecycle-foundation.md) and
  [0002](0002-standard-built-in-composition.md) are Complete.
- [Compiler 0006B4](../../compiler/tasks/0006b4-stable-caller-input-tensor-identity-bindings.md)
  is Complete. Its implemented `CompileConstantPlan.bindableInputBindings()` supplies immutable
  ordered `BindableInput(TensorId tensorId, ValueId valueId)` entries; `bindableInputs()` remains
  the exact ordered compatibility projection. The eight-component `CompileArtifacts` is unchanged.
- Existing `PublicationPlan`, `GraphPreparation`, `PreparedExecutionRunner`, Runtime
  `RunResult`, and `CpuBackendIntegration` supply the remaining required public collaborations.
  Prepare already verifies caller-input order and forward-then-gradient publication occurrences.
- The mandatory clean replan is complete in context
  `01a0a465-0a0a-7813-a035-669dd8423d56`. That context independently audited the interrupted
  draft against the current source and contracts. The former identity blocker is resolved; no
  architecture or inward API change is needed for this bounded scope.
- Implementation context `01a0a46c-8a9e-7862-bd81-0d3495092894` proved that Engine production
  compiles, all 22 Engine tests pass, and the real zero-node integration reaches the expected
  preparation rejection. It also exposed a planning-only fixture error: both required ADD
  integrations fail during compile-time hard eligibility, before Engine preparation, because a
  public `Tensor.add` result has unresolved layout and CPU rejects any occurrence with unresolved
  input or output layout. Appending `contiguous()` resolves only that new CONTIGUOUS result; it
  does not retroactively resolve the ADD occurrence. The corrected integration fixture below uses
  only supported CONTIGUOUS occurrences over resolved FLOAT32 leaves and requires no production,
  architecture, or allowlist expansion.

## Required reading

Read these contracts in the clean implementation context, not merely this task's summaries:

- Root `AGENTS.md`, [architecture contract](../../../../../ARCHITECTURE.md), and
  [current architecture plan](../../../../architecture/current-architecture-plan.md).
- [Module boundaries](../../../../architecture/module-boundaries.md),
  [dependency rules](../../../../architecture/dependency-rules.md),
  [lifecycle](../../../../architecture/lifecycle.md), and
  [Runtime/Prepare/backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md).
- [Planning guide](../../../planning-guide.md), [roadmap](../../../roadmap.md),
  [Engine master plan](../master-plan.md), the dependencies above, and their actual production
  source, tests, and affected Javadoc.
- Model `Tensor` identity/descriptor/storage methods, `TensorId`, `TensorDescriptor`,
  `LayoutDescriptor`, `HostTensorStorage`, and `MemorySegmentStorage`; Compiler
  `GraphCompilationPort`, `CompileArtifacts`, `CompileConstantPlan`, `PublicationPlan`,
  `FunctionalGradientRequest`, and Model forward/gradient publication bindings.
- All Engine production files and existing Engine unit/integration tests; public
  `GraphPreparation`, `PreparedExecution`, `PreparedExecutionRunner`, `RunResult`,
  `PreparedPublication`, `BoundPublication`, run-state ownership/cleanup contracts, and supported
  CPU integration. CPU internals may be read as evidence but never imported by Engine.
- Current Config compile leaves, [Public API](../../../../api/public-api.md),
  [Compile API](../../../../api/compile-api.md), [Runtime API](../../../../api/runtime-api.md),
  relevant [Training API](../../../../api/training-api.md) status, and
  [glossary](../../../../glossary.md) terms.
- [Documentation rules](../../../../developer-guide/documentation-rules.md) and the General,
  Planning, API/Javadoc, and Example profiles under `docs/developer-guide/documentation/`.

## Scope

One cohesive Engine capability contains the ordinary methods, immutable compile/prepared
handles, logical input metadata, identity-checked per-run storage snapshots, publication
occurrence handles, one shared admission/cleanup protocol, focused tests, and documentation.
No new package or public request/configuration abstraction is needed.

### Exact public API and package placement

All new production types live in the deliberate small public facade package
`io.github.pho001.synaptik.engine`. These are complete declared public method inventories;
inherited Object methods and generated record/enum methods are not additional facade APIs.
No type has a public or protected constructor except the explicitly listed input record.

| Type | Exact new or complete declared public surface |
|---|---|
| Existing final `Engine implements AutoCloseable` | Preserve `static Engine standard()`, `boolean isClosed()`, `void close()`; add `CompiledGraph compile(List<Tensor> forwardOutputs)`, `CompiledGraph compile(List<Tensor> forwardOutputs, List<Tensor> cotangentSeeds, List<Tensor> targets)`, `PreparedExecution prepare(CompiledGraph compiledGraph)`, `RunResult run(PreparedExecution preparedExecution, List<Tensor> inputs)` |
| New final `CompiledGraph` | `List<CompiledGraph.Input> inputs()` |
| New nested public record `CompiledGraph.Input` | Components and canonical constructor exactly `Input(TensorId tensorId, TensorDescriptor descriptor)`; both components non-null |
| New final `PreparedExecution` | `CompiledGraph compiledGraph()` |
| New final `RunResult implements AutoCloseable` | `int resultCount()`, `List<RunResult.Publication> publications()`, `boolean isClosed()`, `void close()` |
| New nested public static final `RunResult.Publication` | `int index()`, `TensorId tensorId()`, `TensorDescriptor descriptor()`, `RunResult.Role role()`, `OptionalInt derivativeOrder()`, `OptionalInt targetIndex()`, `boolean isClosed()` |
| New nested public enum `RunResult.Role` | Exactly `FORWARD`, `GRADIENT`, in that order; no custom public members |

`Tensor`, `TensorDescriptor`, and `TensorId` are Model types, and collections/optionals are JDK
types. Ordinary public/protected signatures, generic arguments, constructors, fields, nested
types, and supertypes must expose no Advanced handle, Compiler, Prepare, Runtime, Planning,
backend-contract, CPU, other backend, or `.internal` type. In particular, no `ValueId`, slot,
representation, delegate accessor, raw Object/map binding, or inward conversion is allowed.
There is no public `InputBinding` or gradient-request type in this task.

`CompiledGraph` and `PreparedExecution` have exact reference identity and are owned by the exact
ordinary Engine instance that created them. They are not records and do not implement value
equality or `AutoCloseable`. `PreparedExecution.compiledGraph()` returns the exact originating
ordinary compiled handle. Metadata remains readable after Engine closure; handles cannot then be
used for new work. The input record is descriptive data, not an executable binding token; making
another equal record grants no ownership or execution authority.

### Compile requests and metadata

Both overloads snapshot non-null caller lists without retaining or mutating their containers.
Validate top-level references in declaration order, then snapshot lists in that order and reject
null elements. Forward outputs must be non-empty and identity-unique. The three-list overload
also requires a non-empty identity-unique target list and exactly one explicit non-null seed
per forward output. Seeds may repeat and may be expressions.

The one-list overload calls the existing compile port once with `FORWARD_ONLY` and no derivative
request. The three-list overload calls it once with `FORWARD_AND_BACKWARD` and exactly one
`FunctionalGradientRequest.Stage`: each forward output becomes a `ForwardTensorReference`,
each aligned seed becomes `Optional.of(seed)`, targets keep their request order,
`createGraph=false`, and `DisconnectedPolicy.ERROR`. Compiler owns all semantic validation,
autograd, graph inventory, transformations, and target/seed eligibility. In particular it checks
floating types, exact output/seed Shape and type, seed `requiresGrad=false`, and valid targets.
Engine must not inspect producers or implement derivative rules.

Both paths use the existing `GraphOptimizationConfig.standard()`,
`BackendIntent.unconstrained()`, and `PartitionScoringConfig.neutral()` with the standard owner's
existing capability and availability collaborators. No Config aggregate, options overload,
inferred target, implicit scalar seed, second derivative stage, TRAINING_STEP convenience,
`withBackward`, or Tensor execution/backward method is added. The existing advanced request
surface retains its full current one/two-stage and seed/policy choices unchanged.

Build logical input metadata only from the final artifacts:
iterate `constants().bindableInputBindings()` in stored order, resolve each entry's `valueId()`
to its final `GraphValue.descriptor()`, and retain only the logical ID and exact immutable
descriptor in the public input record. Do not reverse-map by descriptor equality, label, Tensor
allocation order, caller list position, or capture traversal. Compiled handles may privately
retain the existing artifacts but must retain no caller Tensor, request list, provenance, or
host-storage reference. Compile and prepare never call `Tensor.hostStorage()`.

A reused leaf has one input; equal-descriptor leaves remain separate. Compiler-created constants
are omitted. Surviving explicit seed leaves are ordinary bindable inputs, not automatically
constants. Inputs retained by Compiler's dead-code elimination remain required even if unused;
Engine does not filter or reorder that contract. `inputs()` is an immutable ordered snapshot.

### Prepare and typed input binding

Prepare validates the compiled handle's exact ordinary Engine owner before delegating once to
the existing CPU composition. It adds no new source/slot mapping, schedule assembly, allocation,
or preparation fallback. Each success returns a fresh immutable ordinary prepared handle.

Run accepts the required logical Tensors in any list order. Validation order after admission is:

1. Require non-null `preparedExecution`, then non-null `inputs`; validate the prepared handle's
   exact ordinary owner; snapshot the list and reject null elements.
2. Match by `Tensor.id()` value equality against the compiled input inventory. Reject duplicate
   supplied IDs, unexpected IDs, and missing IDs; a descriptor-equal different TensorId is not a
   replacement. Compare supplied descriptors with the compiled descriptor by complete value
   equality, including type, Shape, layout state/geometry, and `requiresGrad`.
3. After the complete logical validation passes, visit required inputs in compiled binding order
   and call each Tensor's synchronized `hostStorage()` exactly once. Retain those exact storage
   references for this run. Missing storage is an error; neither an earlier compile-time
   association nor another Tensor may supply it.
4. Validate all snapshots before borrowing: exact data type, sufficient capacity for a present
   layout's `referencedElementSpan()` (including zero), live scope, and current-thread segment
   accessibility. An absent layout does not imply contiguous geometry and does not license
   Engine to infer or resolve one. Prepared CPU compatibility remains an independent inward check.
5. Borrow one fresh wrapper per logical input through the existing CPU composition, in final
   binding order, then call the Runtime runner once with that ordered list. Distinct TensorIds
   may share one storage object; never deduplicate their wrappers because Runtime forbids repeated
   representation object identity. Read-only inputs are accepted where the prepared access allows.

Wrong owner, duplicate/unexpected/missing ID, descriptor mismatch, and logical type/span mismatch
throw `IllegalArgumentException`. Absent, dead, or currently inaccessible storage throws
`IllegalStateException`. Null references/elements throw `NullPointerException`. Logical binding
diagnostics identify the relevant TensorId where one exists; count errors identify the expected
and actual counts. Do not establish a new public exception family or rewrite inward exceptions.

The synchronization is a per-Tensor association snapshot, not an atomic snapshot across the
entire input set, a memory fence for payload writes, or arena pinning. Replacing/clearing a Tensor
association after its snapshot cannot redirect that run. The exact borrowed storage and its
bytes must remain valid, accessible where used, and free from conflicting caller mutation until
the result closes, including the interval after synchronous run return. The caller owns every
host storage/arena and Engine never closes, slices, reinterprets, copies, converts, or writes back
into that association. Engine owns only the fresh non-owning wrapper objects it creates.

Prepared handles can be shared by concurrent calls; each call snapshots independently and has one
isolated Runtime state. Shared stable host storage is allowed. To change associations for
different runs, callers must coordinate replacement and snapshots; this API promises no
independent `TensorId -> HostTensorStorage` per-run override.

### Publication occurrence and alias semantics

Privately construct immutable publication specifications from `PublicationPlan`, resolving final
descriptors through each binding's `ValueId`. The order is all forward bindings followed by all
gradient bindings in their existing derivative-order/target order, never graph-output-set order.
For each successful run, allocate one fresh `RunResult.Publication` per occurrence:

| Property | FORWARD | GRADIENT |
|---|---|---|
| `index()` | Dense zero-based position in the complete result list | Dense zero-based position after the forward prefix |
| `tensorId()` | Original requested forward output ID | Requested differentiation target ID, not a generated gradient Tensor ID |
| `descriptor()` | Final published forward value descriptor | Final gradient value descriptor, not blindly the target descriptor |
| `derivativeOrder()` | Empty | Present and exactly 1 for this ordinary overload |
| `targetIndex()` | Empty | Present zero-based position in the explicit target list |

Compare the completed Runtime result count with
`forwardBindings().size() + gradientBindings().size()` before returning any ordinary result.
Mismatch is an `IllegalStateException` and triggers full rollback. Never compare with
`graph().outputs().size()`: the graph boundary deduplicates gradient values, while publication
roles deliberately retain all occurrences.

`publications()` returns one immutable list, and repeated queries return the same occurrence
objects in the same order. Every run has a different result and different occurrence objects.
Publications use reference identity, not value equality, even when all descriptive fields match.
They retain their originating result lease privately, have no independent close operation,
and expose no parent/delegate accessor. Closing one result invalidates only its own lease;
Engine closure closes every remaining lease. Publication `isClosed()` delegates to its result
and becomes true when closure starts. All immutable metadata, including `publications()`,
`resultCount()`, and publication accessors, remains readable after closure.

Two roles may internally select the same exact Runtime representation, including forward/gradient
or repeated-gradient aliases. Preserve every role without copying, merging, or deduplication.
Logical identity, list position, and Java occurrence identity do not assert physical aliasing or
independent storage. No alias-query or value/host-storage accessor is added, and no result is
attached to a Model Tensor or a mutable Tensor gradient field.

### Admission, ownership, and cleanup

Reuse AdvancedEngine's single lifecycle lock, active-operation accounting, open-result registry,
reverse result-close order, and retained close failure. Ordinary Engine keeps its one private
final advanced delegate. Add narrow package-private ordinary entry points on that delegate;
perform their entire ordinary validation, conversion, storage snapshot/borrowing, inward work,
count checking, and outward handle construction inside one admission. Factor private ungated work
where useful; do not nest public advanced calls or introduce a second gate, second registry,
preflight `isClosed()` race, generic callback service, or service locator.

Preserve the exact existing closed-owner failure,
`IllegalStateException("advanced engine is closed")`, for both surfaces. Once closure begins,
this failure precedes null, membership, storage, and inward validation. A previously admitted
operation may finish its work, but cannot return a newly usable handle/result if closure won
before final publication. Register a complete result and release admission atomically under the
same existing lock; failed construction, mapping, or registration releases admission once.

Extend private `AdvancedRunResult` construction to own an immutable list of ordinary-created
borrow wrappers; the existing advanced run path supplies an empty list and its public signatures
and caller-ownership semantics remain unchanged. Its one close protocol first attempts to close
the exact Runtime result, then attempts every owned wrapper once in reverse borrow order, and
unregisters even on failure. Ordinary `RunResult.close()` delegates to that same registered owner.
Engine auto-close therefore performs the same cleanup as explicit ordinary result-close.

Before ownership transfers to the registered result, Engine owns the acquired wrappers. A failed
borrow closes only earlier wrappers in reverse order. A failed Runtime run relies on the
runner's established state rollback, then closes the wrappers. Failure after successful Runtime
return (count check, wrapper/metadata construction, or closing-owner rejection) closes that exact
Runtime result before the wrappers. Retain the original unchecked failure object and suppress
distinct cleanup failures; cleanup attempts all resources and never self-suppresses. No failed
run returns a partial result. Avoid double close when ownership has already moved to the result.

Concurrent/repeated result and Engine closes keep the existing wait-uninterruptibly-and-restore-
interrupt behavior and return/rethrow the same retained cleanup outcome. Metadata access is safe
concurrently with closure; do not expose the non-thread-safe inward result. Separate runs share
no mutable run state. No cancellation, asynchronous execution, or new persistent prepared
resource lifecycle is introduced.

### Current limits that must remain explicit

The supported CPU composition accepts exactly one non-empty maximal CPU partition; zero-node/
pass-through, mixed-owner, and multi-partition preparation remain rejected. Unsupported or
dynamic work still fails at the existing responsible boundary, without Engine operation switches.
A successful compile is not a guarantee of successful CPU preparation or execution.

CPU may reject a retained unused caller input without a matching buffer declaration. CPU cold
binding also currently requires exact prepared byte size and compatible carrier/alignment;
Model's sufficient-span rule alone does not guarantee that oversized storage executes.
Do not drop inputs or slice storage to make these cases pass. No inward production modification
is authorized to broaden them. Future host/value access requires an explicit supported
Runtime/backend bridge in task 0004, not reflection into leased representations here.

The integration fixture must not use public ADD merely because CPU has an ADD kernel: public ADD
construction leaves the ADD output layout unresolved, so the complete occurrence is not currently
hard-eligible. A following CONTIGUOUS request does not rewrite that descriptor. A provenance-free
explicit seed leaf also cannot serve directly as the repeated published gradient in this CPU
composition because schedule assembly has no prepared buffer assignment for that pass-through
publication. The corrected fixture makes the shared seed an independently produced CONTIGUOUS
result over a resolved leaf, so every published value has a prepared assignment without adding
pass-through support.

## Out of scope

- Any Java change outside Engine, new Gradle dependency, architecture rule, ADR, shared contract,
  CPU route, preparation mapping, Runtime publication/access, or backend conformance behavior.
- Host values, arrays, segments, scalar/element access, copies/transfers, detached results,
  Tensor result attachment, publication policy, or persistence/NN/Training adapters.
- Independent storage-binding overrides, symbolic binding/inference, graph traversal,
  kernel selection, backend discovery, mixed-backend assembly, tuning, caches, or tracing payloads.
- New Config aggregates, ordinary compile policy tuning, higher-order request ergonomics,
  scalar default seeds, target inference, `output.execute()`, `withBackward()`, or one-shot work.
- Creating task 0004 or another future detailed specification during this implementation.

## Architecture constraints and no-change conclusions

Engine is the outer composition and logical-binding layer. Compiler owns identity provenance,
graph traversal/autograd, and final publication facts; Prepare owns graph-value-to-slot mapping;
Runtime owns isolated execution state and logical cleanup; CPU owns physical compatibility,
storage mechanics, and lowering. Only documented public integration collaborators may cross
these boundaries. Cold logical ID lookup is permitted; no new per-element or hot-path lookup,
reflection, boxing, graph dispatch, storage copy, or executable work is introduced.

The implementation audit must explicitly reaffirm: no changes to `ARCHITECTURE.md`, focused
architecture decisions/rules, Model, Compiler, Config, Planning, shared Prepare, Runtime, CPU or
other backends, Gradle, backend conformance, NN, Training, persistence, or tuning production.
Architecture-test source remains unchanged because no dependency rule changes; its focused
existing Engine inventory test is still run. Integration-test source changes because ordinary
end-to-end orchestration is new. Tensor API Javadoc remains accurate because Tensor behavior is
unchanged. Training API status needs a narrow documentation correction for ordinary gradient
publication metadata, not an optimizer or materialized-gradient claim. Other user guides and
architecture explanations remain conceptual; record targeted no-change conclusions unless a
task-specific contradiction makes the fixed documentation scope insufficient, in which case
stop and request a scope correction rather than silently expanding it.

## Exact affected files and hard ceiling

Paths below are relative to repository root. These 19 paths are the entire implementation
allowlist, not permission to overwrite pre-existing changes.

Production (seven paths), all in `modules/engine/src/main/java/io/github/pho001/synaptik/engine/`:

1. `Engine.java` — four ordinary method additions, preserving standard construction and delegate.
2. `AdvancedEngine.java` — bounded package-private ordinary admission paths and ownership handoff.
3. `AdvancedRunResult.java` — private ownership/cleanup of ordinary-created borrow wrappers.
4. New `CompiledGraph.java` — owner-bound recipe and nested logical input metadata.
5. New `PreparedExecution.java` — owner-bound prepared recipe and originating compile handle.
6. New `RunResult.java` — typed result lease, nested publication occurrences and roles.
7. `package-info.java` — current ordinary/advanced and planned materialization boundaries.

Tests (four paths):

8. New `modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineTypedLifecycleTest.java`.
9. New `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/EngineTypedPublicShapeTest.java`.
10. Existing `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/AdvancedEnginePublicShapeTest.java`
    — update only the former construction-only Engine method assertion; retain advanced assertions.
11. New `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineTypedLifecycleIntegrationTest.java`.

Documentation and planning (eight paths):

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
12–18-path guardrail is atomic: the same lifecycle decision needs its compile/prepared/result
types, shared admission/cleanup, ordinary and advanced regression assertions, one end-to-end
fixture, and synchronized public/compile/runtime/training/glossary status. Splitting those into
mechanical tasks would leave the single public capability undocumented or unvalidated.
No second production module is implemented and no unrelated refactor is permitted. If an
additional helper file, dependency, or inward contract is genuinely required, stop and replan
before editing it. Keep handle metadata logic in the handle types and admission in the lifecycle
owner rather than creating a catch-all facade.

## Acceptance criteria

- The exact ordinary API compiles from a distinct package using only Engine, Model, and JDK
  imports. It contains no inward/advanced type escape, public handle construction, extra overload,
  config aggregate, value accessor, or scalar/one-shot backward convenience.
- One-list forward and explicitly seeded three-list first-order requests delegate once with the
  fixed settings; semantic failures retain inward ownership and identities.
- Final bindable identities/descriptors/order come only from Compiler 0006B4, without Tensor
  retention or storage reads during compile/prepare. Arbitrary supplied input order is accepted;
  every required ID is supplied exactly once and descriptors/storage are validated as specified.
- Each run snapshots associations once, borrows fresh wrappers in final order, preserves shared
  underlying storage, and creates exactly one isolated Runtime state. Caller storage survives all
  success/failure/close paths.
- Result count and metadata match every forward-then-gradient occurrence, including repeated
  gradient values and forward/gradient aliases. Identity, derivative role, target index,
  descriptor, immutability, per-run identity, and post-close inspection match the specified API.
- The real public CPU integration uses only static resolved FLOAT32 leaves and CONTIGUOUS
  occurrences. Its seeded case publishes two identity-distinct gradient targets backed by one
  repeated produced seed value; broader forward/gradient alias cases remain covered by the
  package-local Engine publication fixtures and are not presented as CPU support that the current
  public expression path cannot execute.
- Exact ordinary owner checks and shared admission prevent shutdown races, partial publication,
  leaked Runtime results/wrappers, double cleanup, and failure replacement. Advanced public
  behavior and standard construction remain intact.
- Current CPU limitations are tested/documented without inward fixes. Final tests, Javadoc,
  documentation review, scope checks, and whitespace validation pass; all no-change conclusions
  and evidence are recorded before marking Complete.

## Focused test design

Use package-local composition fakes for controlled ordering, resource counters, failures, and
latches, following the existing Engine lifecycle tests. Use valid public compile artifacts and
prepared Runtime fixtures; no production reflection, new service interface, or fake CPU success
claim. Package-private handle constructors may supply a deliberately wrong-descriptor input
metadata fixture to test defense against malformed mappings without forging Model Tensor IDs.

- Compile/input metadata: forward request order; two equal-descriptor distinct leaves; repeated
  use of one leaf; immutable lists and input-record null checks; no host association required to
  compile/prepare; explicit seed leaves and seed expressions; compiler-generated constants omitted;
  exact gradient target order; duplicate/null/empty requests and invalid seed/target rejection.
- Binding: supplied input order reversed; missing/extra/duplicate IDs; wrong ordinary owner even
  if a package-local test shares one advanced delegate; descriptor mismatch; absent, dead,
  inaccessible, wrong-carrier, insufficient/incompatible storage; no borrow before all logical
  and snapshot checks finish. Model already rejects some malformed associations at attachment:
  test those at that owning boundary or an Engine-owned metadata fixture, not by mutating Model
  internals to invent a reachable ordinary input.
- Snapshots/lifetime: a latch-controlled first borrow permits association replacement after all
  snapshots; prove the exact old references were used and later runs take fresh snapshots.
  Distinct inputs sharing storage get distinct wrappers. Read-only input succeeds for a read-only
  prepared role. Caller arenas remain live after result/Engine close; no copy or reassociation.
- Publications: ordinary forward success and explicitly seeded first-order success; repeated
  gradient-value roles and forward/gradient alias roles remain separate; target ID differs from
  generated gradient identity; immutable metadata, stable repeated accessor identity, fresh objects
  per run, and readable metadata with closed lifecycle after result/Engine closure.
- Failure/cleanup: partial borrow; runner creation/cold-bind/execute/publication failure; an
  intentionally wrong publication count returned by the fake prepared schedule; failure after
  Runtime success; cleanup failure suppression; reverse wrapper close; reverse result close before
  backend close; no partial result. Counters prove exact once and original exception/error identity.
- Concurrency: independent repeated and concurrent runs, close racing admitted compile/prepare/run
  and logical mapping, closed-before-invalid-input precedence, repeated/concurrent result and
  Engine close, interruption restoration, and retained cleanup failure. Use bounded latch waits,
  not timing sleeps or nondeterministic stress.
- Public shape: automate exact methods, generic signatures, no public/protected constructors
  except the input record, nested enum/record/final types, ordinary forbidden type closure, and
  absence of value/materialization/convenience methods. Extend existing advanced shape assertion
  only for Engine's four additions; no repeated manual javap/reflection checklist.
- Integration: use only ordinary Engine, Model, and JDK imports through real `Engine.standard()`.
  Use exact-size, suitably aligned `MemorySegmentStorage` for static resolved FLOAT32 leaves so the
  test exercises the current CPU carrier contract rather than depending on an unselected heap-array
  route. The forward-only case uses two identity-distinct leaves and outputs
  `[left.contiguous(), right.contiguous()]`; compile must report inputs `[left, right]`, run must
  accept `[right, left]`, and the two metadata-only publications must retain forward order.

  The explicitly seeded case is exact: create identity-distinct gradient-eligible `left` and
  `right` leaves plus a non-gradient `seedLeaf`, all with the same static resolved FLOAT32 Shape
  and layout geometry and with gradient eligibility as just specified; create
  `leftOutput = left.contiguous()`,
  `rightOutput = right.contiguous()`, and `seed = seedLeaf.contiguous()`; then call
  `compile([leftOutput, rightOutput], [seed, seed], [left, right])`. Compile must report bindable
  inputs `[left, right, seedLeaf]`, and run must accept the reversed binding order
  `[seedLeaf, right, left]`. The result must contain exactly four metadata-only occurrences in
  order: forward `leftOutput`, forward `rightOutput`, first-order target `left` at target index
  zero, and first-order target `right` at target index one. The two gradient roles must remain
  separate identity-distinct publication objects while Compiler/Prepare/Runtime retain their one
  repeated produced seed value. Exercise a second run as needed to prove per-run occurrence
  identity and close behavior. Assert the current zero-node prepare rejection separately. Do not
  inspect numerical results, physical aliasing, backend storage, or inward result access.

## Validation plan

This is task-tier validation: Engine-only production behavior plus one focused end-to-end fixture,
with unchanged dependencies and shared contracts. Run focused tests during development as needed,
then record one final pass after executable Java stabilizes:

```bash
./gradlew :modules:engine:test
./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.EngineTypedLifecycleIntegrationTest
./gradlew :testing:architecture-tests:test --tests io.github.pho001.synaptik.testing.architecture.EngineCompositionContractTest
```

The failed ADD integration attempts from implementation context
`01a0a46c-8a9e-7862-bd81-0d3495092894` are corrective planning evidence, not accepted final test
failures and not a reason to rerun or weaken the 22 passing Engine tests. Replace both ADD fixtures
with the exact CONTIGUOUS fixtures above, then rerun the focused integration class once. Record its
final test count and confirm that the seeded case reaches real CPU prepare and run, not merely
compile or metadata construction. Do not authorize or test an inward ADD/layout workaround.

The independent documentation pass reuses those successful results and runs final:

```bash
./gradlew :modules:engine:javadoc
git diff --check
```

Validate changed Markdown links/anchors, headings, fences, whitespace, final newlines, current vs
planned claims, exact 19-path ceiling relative to the starting dirty-worktree snapshot, and
0003/master/roadmap status agreement. Verify 0004–0008 stay Draft with no new task 0004 spec.
The public-shape test owns recurring reflection/import boundary assertions; inspect generated
Javadoc for all new types/members and revised lifecycle contracts once after finalization.
No repeated successful Java suites in the documentation or coordinator context. Root `test`
remains Engine 0008/CI checkpoint work unless new evidence proves shared-contract, dependency,
or second-production-module impact; such impact requires replan before implementation.

## Documentation handoff

Document type: Planning for this spec/master/roadmap; API/Javadoc for Java contracts and API
references; General for glossary prose; Example for lifecycle snippets. The clean implementation
agent may draft Javadoc, but a distinct clean documentation-focused context must independently
finalize all seven affected production Javadocs, five explanatory documents, and the three
planning records in the same overall change.

The handoff must include final source/tests, exact public API, successful test commands/results,
dirty-worktree baseline, this task, directly relevant Model/Compiler/Prepare/Runtime/CPU
contracts, and documentation rules/profiles. Explain bindable input, association snapshot,
cotangent, publication occurrence, target identity, and result lease at first use. Glossary entries
must distinguish ordinary Engine `CompiledGraph`/`PreparedExecution`/`RunResult` from Model
graph and Runtime recipe/lease types. Correct the affected API pages' stale construction-only,
gradient-publication, and borrowed-input lifetime wording. Do not promise host values or training.

Include a complete ordinary forward example and a seeded-gradient example with imports,
resolved supported descriptors, storage setup, explicit ownership, arbitrary binding order,
metadata inspection, and try-with-resources cleanup. Exercise their essential call paths in the
public integration test. Explain that task 0004 adds values and task 0006 later adds scalar
convenience with explicit targets, rather than presenting this lower-level seed overload as the
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
> independent documentation, Javadoc, and final checks pass; leave 0004–0008 Draft.

## Local decisions and follow-up

- The historical blocker was the missing originating Tensor identity; completed Compiler 0006B4
  closes it without changing artifacts' component count or Prepare/Runtime/CPU production.
- Choose association-only `List<Tensor>` binding and immutable `CompiledGraph.Input` metadata.
  A storage override carrier is not needed for this capability.
- Choose an explicit one-stage three-list compile overload, not a new request/config abstraction.
  Full Compiler request choices remain advanced; scalar and one-shot ergonomics remain later.
- Choose fresh identity-based publication handles and metadata-readable-after-close semantics.
  Aliases remain occurrences, not inferred physical ownership or materialized Tensor values.
- Choose the exact CONTIGUOUS integration fixture recorded above. The shared explicit seed is a
  CONTIGUOUS expression rather than a leaf so its repeated gradient value has a prepared CPU buffer
  assignment. This proves repeated gradient-publication occurrences for two identity-distinct
  targets. Keep the more general forward/gradient alias proof in package-local Engine tests; do
  not invent public ADD eligibility or CPU pass-through publication support for integration.
- Extend the existing private admission/result owner rather than wrapping separately admitted
  public calls. Borrow wrappers are Engine-owned; storage remains caller-owned.
- No architecture decision changes. The 19-path exception is justified above; no tasks run out
  of order, and the planning seam audit changes only the three synchronized planning files.
- Task 0004 must decide the supported representation-to-host access bridge, selected publication
  addressing, payload bounds, copy/lease ownership, and cleanup before implementation. It must
  not treat the metadata-only lease as an existing value API.
- Tasks 0005–0006 build over 0003–0004; no exact future convenience signatures are fixed here.
  Task 0007 remains optional tuning, and 0008 remains the capability checkpoint.

## Completion summary

- Completed changes: Added the ordinary owner-bound compile and prepare handles, arbitrary-order
  `TensorId` input matching, per-run caller-owned `HostTensorStorage` snapshots, synchronous run,
  and metadata-only forward/gradient publication occurrences. Publication metadata preserves
  requested output or target identity, target position, derivative order, occurrence multiplicity,
  and inward aliasing without exposing physical alias identity or values. Corrected real CPU
  coverage uses only CONTIGUOUS over resolved FLOAT32 leaves and
  `seedLeaf.contiguous()` as the shared explicit seed.
- Files changed or created: exactly the seven production, four test, five explanatory, and three
  planning paths listed in [Exact affected files and hard ceiling](#exact-affected-files-and-hard-ceiling).
  The shared worktree's Compiler 0006B4 files and its additions in `public-api.md`,
  `compile-api.md`, `glossary.md`, and `roadmap.md` were preserved.
- Implementation validation: Reused stable evidence from implementation context
  `01a0a46c-8a9e-7862-bd81-0d3495092894`: Engine production compiled;
  `./gradlew :modules:engine:test` passed 22 tests; corrected
  `EngineTypedLifecycleIntegrationTest` passed 3 tests through the real CPU composition; focused
  `EngineCompositionContractTest` passed 1 test; exact 11-Java-path, trailing-whitespace, and
  `git diff --check` checks passed. No executable Java changed afterward.
- Documentation-agent review: Clean documentation context
  `01a0a488-b8dd-7bf1-a8fc-947f531f420f` independently reviewed and finalized all seven affected
  production Javadocs plus the eight documentation/planning paths. It applied the General,
  API/Javadoc, Planning, and Example profiles and changed no executable Java token.
- Documentation validation: `./gradlew :modules:engine:javadoc` passed after final Javadoc edits
  with no warnings. Generated pages for the package and all six affected types were inspected.
  Exact `javap -public` inspection confirmed the ordinary and advanced public shapes. The complete
  ordinary forward and seeded-gradient examples compiled with `javac` against current module
  classes. A targeted checker passed local links and anchors, heading syntax, balanced fences,
  trailing whitespace, and final newlines for all eight Markdown paths. Current/planned wording,
  0003/master/roadmap status, all five Draft 0004–0008 rows, absence of an Engine 0004 task spec,
  exact cumulative 19-path task scope, and final `git diff --check` passed.
- Documentation impact: Public, Compile, Runtime, and Training API pages now describe the current
  ordinary surface, precise caller-storage/result lifetimes, corrected CONTIGUOUS examples, and
  metadata-only gradient boundary. The glossary now distinguishes bindable identity metadata,
  caller-input association snapshots, Engine facade handles/results, and publication occurrences
  from Model graph and Runtime recipe/lease contracts.
- Javadoc review: All seven affected production paths now state identity, ownership, lifecycle,
  closure, alias, and current CPU-preparation boundaries. Parameter, return, and expected failure
  documentation remains complete; no executable behavior was altered.
- No-change conclusions: `ARCHITECTURE.md` and focused architecture pages require no change because
  task 0003 realizes the existing Engine composition and typed-binding ownership without changing
  a module boundary. Build files and dependencies are unchanged. Model Tensor/storage, Compiler
  0006B4, Prepare, Runtime, and CPU production and documentation outside the allowlist remain
  accurate because their contracts and behavior did not change. Training behavior remains absent;
  only its API status now acknowledges metadata-only Engine gradient occurrences. Other modules,
  architecture tests, backend conformance, and additional integration coverage require no change.
  Root tests were not run because this is a task-tier Engine capability with unchanged executable
  behavior, dependencies, and shared contracts; Engine task 0008/CI retains the checkpoint.
- Unresolved issues: None within task 0003. Host values remain task 0004; one-shot forward and
  scalar-objective backward convenience remain tasks 0005–0006. Tasks 0004–0008 remain Draft,
  and no task 0004 specification was created.
- Follow-up required: None for task 0003.

Status: Complete
