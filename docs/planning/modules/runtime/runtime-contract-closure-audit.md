# Runtime contract closure audit

## Executive conclusion and closure verdict

Verdict: `BLOCKING_GAP`.

Runtime 0001-0010 form a cohesive prepared-execution and per-run lifecycle boundary, but the
selected Runtime milestone cannot close. Three blocking findings remain: `RunState.close()` can
abort cleanup when distinct resources throw the same exception object; three general architecture
pages and the architecture-test guide contain stale implementation-status claims outside this
task's permitted correction set; and the architecture-test project has no Runtime dependency or
hot-path enforcement despite the contract's named Runtime checks. The audit itself is complete.
No Java, Javadoc source, architecture document, ADR, Gradle file, or test is repaired here.

## Authority, scope, and method

[`ARCHITECTURE.md`](../../../../ARCHITECTURE.md) is authoritative. This artifact is
non-authoritative planning evidence for Runtime 0011. Clean documentation context
`019fbee0-00ac-7530-8308-982306a7a9f8` read the
task specification, governing architecture and accepted decisions, documentation and planning
profiles, Runtime history, current source/tests/rendered Javadocs, API and backend guidance,
glossary, adjacent owner plans, build declarations, and architecture/dependency tests. Source,
tests, compiled/rendered surfaces, and exact inspection commands are primary evidence; completed
task prose is corroborating history only.

The initial working tree contained exactly the Ready 0011 specification and its Runtime
master-plan and roadmap synchronization. This audit preserves Runtime 0001-0010 and changes only
the four always-authorized planning paths. It does not create Runtime 0012 or a Prepare 0003
specification.

## Architecture and module-boundary assessment

The implementation follows the authoritative split: Runtime owns immutable recipes, one mutable
whole-run state, validity, cold binding, ordered traversal, publication, leases, and cleanup
orchestration; concrete backends own physical representations and mechanics; Prepare owns staged
translation/finalization; Engine owns composition and public value access. Runtime source imports
only Runtime and JDK types, while its declared but currently unused Gradle dependencies are Config,
Backend Contract, and Trace. No Compiler, Planning, Prepare, Engine, Model, or concrete-backend
edge or hot-path graph type was found.

No architecture decision is required. The cleanup defect contradicts an existing Runtime
contract rather than revealing uncertain ownership. The stale explanatory status and missing
enforcement also require no new architecture rule.

## Production, test, public-surface, package, and generated-Javadoc inventory

Reproducible inventory commands found 25 production files, 17 test files/suites, five packages,
34 generated public type pages, five package summaries, `allclasses-index.html`, and
`overview-tree.html`.

| Package | Production files | Public declarations represented in generated Javadoc | Test suites |
|---|---:|---|---:|
| `runtime.memory` | 4 | `BufferSlot`, `WorkspaceSlot`, `PreparedMemoryPlan`, `BufferEntry`, `WorkspaceEntry` | 3 |
| `runtime.resource` | 4 | `BufferRepresentation`, `WorkspaceRepresentation`, `PreparedRepresentationPlan`, `BufferPreparation`, `CallerInput`, `CreatedBuffer`, `BufferCreator`, `WorkspaceCreator` | 1 |
| `runtime.execution` | 6 | `PreparedExecution`, `PreparedExecutable`, `BufferAccess`, `BufferSelection`, `WorkspaceSelection`, `BoundInvocation`, `PreparedBufferTransfer`, `BoundBufferTransfer` | 5 |
| `runtime.schedule` | 2 | `PreparedSchedule`, `Step`, `RepresentationCreationStep`, `ExecutionStep`, `BufferTransferStep`, `PublicationStep` | 1 |
| `runtime.run` | 9 | `RunResourceOwnership`, `BufferRepresentationBinding`, `RunState`, `PreparedPublication`, `BoundPublication`, `RunResult`, `PreparedExecutionRunner` | 7 |

`RunStateCreation` is the sole package-private top-level declaration. Package-private association
accessors on bound actions/publications and private runner step implementations remain internal.
The generated public inventory excludes all of them. Source and rendered indexes agree on all 34
public type pages.

The 17 suites are `BufferSlotTest`, `WorkspaceSlotTest`, `PreparedMemoryPlanTest`,
`RepresentationContractTest`, `PreparedExecutionTest`, `PreparedExecutableTest`,
`BoundInvocationTest`, `PreparedBufferTransferTest`, `BoundBufferTransferTest`,
`PreparedScheduleTest`, `BufferRepresentationBindingTest`, `RunStateTest`,
`RunStateCreationTest`, `PreparedPublicationTest`, `BoundPublicationTest`, `RunResultTest`, and
`PreparedExecutionRunnerTest`.

## Prepared artifact and exact-plan contract assessment

Prepared slot, geometry, creation, executable, transfer, publication, schedule, and aggregate
values are immutable recipes. Constructors snapshot caller-owned list structure and retain the
documented exact immutable element references. `PreparedExecution`, schedule steps, executable,
transfer, publication, representation plan, and run state consistently require the same
`PreparedMemoryPlan` object by reference identity rather than structural equality.

The nominal domains remain separate: Model/Planning identities do not enter Runtime; buffer and
workspace slots are different records; prepared coordinates are dense list positions; physical
representations are backend-owned identities; bound actions retain one exact per-run state and
direct resolved representations. No implicit numeric conversion connects these domains.

## Run-state creation, ownership, and cold-binding assessment

`RunStateCreation.create` validates the dense caller count, null elements, and caller identity
uniqueness before callbacks. It creates buffers in buffer/representation order, then workspaces,
and constructs one state only after all results exist. Partial creation rolls back created results
in reverse order, excludes borrowed inputs, retains the primary unchecked failure, and checks
against self-suppression.

Caller inputs are borrowed for the complete state/result lifetime. Created buffers and all
workspaces are run-owned after successful construction. Executable, transfer, and publication
binding validates the exact plan/state association and compatibility before traversal and retains
direct per-run references. Reusable recipes retain no mutable run state.

## Representation, validity, transfer, and aliasing assessment

Every buffer position owns one or more structurally resident representations and an independent
boolean per copy; workspaces are scratch outside logical validity. Borrowed inputs begin valid and
created buffers begin invalid. Zero, one, or multiple copies may be valid.

A bound transfer treats a valid destination as a no-op, otherwise requires a valid source, calls
backend work once, and marks only the destination valid after success. Failure changes no
validity. Materialization is the same explicit transfer. There is no fallback, implicit copy,
conversion, coherence, invalidation, route search, or lazy representation creation. Distinct
publication/result positions may intentionally alias one exact valid representation.

## Executable traversal, publication, result lease, and cleanup assessment

The runner cold-binds every non-creation occurrence before the first action. Executable reads are
checked before all copies of each output buffer are invalidated; only exact declared writes become
valid after success. Read/write overlap therefore consumes the old value before conservative
invalidation. Publication is a one-shot local transition on one already-resolved valid copy and
performs no physical work. Successful complete publication constructs `RunResult`, which privately
snapshots direct aliases and leases cleanup of the complete open state without exposing values.

Normal cleanup is closed-first, idempotent, workspace-reverse then buffer/representation-reverse,
and skips borrowed buffers. Finding `RUNTIME-CLEANUP-001` prevents closure because its shared-
throwable case violates the promised attempt-all and primary/suppressed behavior.

## Empty, repeated, concurrent, and failure-path assessment

Empty geometry, creation-only and empty schedules, zero publications, empty results, repeated
schedule occurrences, repeated executable selections, and result aliases are explicitly tested.
Repeated and concurrent runner calls share only immutable recipes; each call creates a distinct
state, validity arrays, bound actions, resources, result lease, and cleanup lifecycle. A single
state or bound action is deliberately not thread-safe.

Construction, binding, invalid-read, backend-action, transfer, publication, result-construction,
and cleanup failures have focused coverage. The untested shared-throwable cleanup case is the one
blocking failure-path gap.

## Hot-path performance-boundary assessment

Cold work may allocate arrays and bound objects, validate compatibility, and resolve resources.
Bound traversal uses a direct `BoundStep[]`, direct invocation/transfer/publication fields, and
precomputed primitive coordinate arrays. Existing source/bytecode-oriented tests reject `Map`,
reflection, `ServiceLoader`, graph/backend lookup, and hot compatibility casts.

No hot traversal code inspects `Operation` or `CompiledNode`, discovers a backend, searches a
route/kernel, uses reflection, performs map or boxing lookup, synchronizes, or allocates per bound
occurrence. The runner's linear primitive-array deduplication and cold array copies occur before
the traversal loop.

## Public and package-private API cohesion assessment

Every public declaration participates in a current cross-package Runtime lifecycle, represents a
backend extension seam, or is the public result/runner boundary documented by Runtime 0001-0010.
No public root facade, manager, registry, service locator, generic resource map, unchecked generic
access, or raw `Object` carrier exists. The five packages separate geometry, nominal physical
roles/creation recipes, executable actions, ordered schedule recipes, and mutable run lifecycle.

`RunStateCreation` and package-private bound association accessors are implementation details used
only to validate exact ownership and occurrence association. Publishing them would widen the API
without a current consumer. The public surface is necessary and cohesive; it is not sufficient
for result-value access or public Engine composition, which remain downstream-owned.

## Package and dependency assessment

`modules/runtime/build.gradle.kts` declares only Config, Backend Contract, and Trace. Production
imports contain only Runtime packages and `java.util` classes. No code currently consumes Config,
Backend Contract, or Trace, and no forbidden module type enters Runtime. The absence of a current
consumer does not authorize speculative removal during this audit; it is recorded as a minimality
observation for future build hygiene.

Runtime does not absorb Model semantics, Compiler graphs/publication identities, Planning
ownership/logical memory, Prepare orchestration, Engine composition, backend discovery/lowering,
physical storage, Trace payload design, Config policy, or tuning. Finding
`ARCHITECTURE-ENFORCEMENT-001` records that these currently compliant facts lack the contract-
named Runtime architecture tests.

## Documentation, examples, glossary, and completed-history consistency

Runtime API, Public API, backend guide, glossary, focused Runtime/Prepare/backend explanation, and
generated Runtime Javadocs agree on the current lifecycle and its planned boundaries. Examples are
explicitly current or conceptual and expose no concrete backend or public result-value access.
Generated `RunState` documentation accurately states the intended cleanup contract, but source
does not satisfy it in the blocking shared-throwable case.

Completed Runtime 0001-0010 remain unchanged. Their decisions, limitations, validation counts,
no-change conclusions, and follow-ups reconcile with source/tests except that earlier cleanup
coverage did not exercise identical throwable identity. Finding `DOCUMENTATION-STATUS-001` records
stale implementation-status prose in `current-architecture-plan.md`, `lifecycle.md`,
`module-boundaries.md`, and `developer-guide/architecture-tests.md`; those paths are review-only
in 0011 and are not silently expanded into this change.

## Validation coverage and checkpoint applicability

Runtime 0010's final 17-suite/143-test result is valid prior focused evidence. Runtime 0011 is the
capability checkpoint, so repository-wide tests, architecture tests, and Runtime Javadoc are
triggered once by the combined command. Backend conformance is not triggered because no concrete
backend implements this path. Integration is not triggered because Engine exposes no end-to-end
execution consumer.

The architecture suite checks current Config, Planning, and NN/Training edges, but contains no
Runtime dependency or `Operation`/`CompiledNode` hot-path test. Manual Gradle/import/source/
bytecode inspection supplies audit evidence but does not replace the contract's durable expected
enforcement.

## Deferred and downstream-owned work

| Work | Owner | Closure disposition |
|---|---|---|
| Public result-value access and lifecycle composition | Engine/result API owner | `NON_BLOCKING_DEFERRED`; Runtime's private lease is coherent without exposing values. |
| Compile-to-prepared translation, coverage, schedule construction, and final assembly | Prepare 0003 | `NON_BLOCKING_DEFERRED`; remains Draft without a specification. |
| Physical representations, transfers, invocations, lowering, routes, kernels, and execution | Concrete backends | `NON_BLOCKING_DEFERRED`; current Runtime contracts are backend-neutral. |
| Typed run payloads and emission | Trace 0006 plus Runtime/Engine producer | `NON_BLOCKING_DEFERRED`; no stable payload family exists. |
| Run/publication policy | Config 0007 and its future consumer | `NON_BLOCKING_DEFERRED`; current runner has no policy decision. |
| Measurement and tuning | Backend prepare and tuning tooling | `NON_BLOCKING_DEFERRED`; Runtime executes selected recipes and only permits passive observation. |

## Findings, severity, and disposition

| Finding | Finding label | Evidence and impact | Owner and disposition |
|---|---|---|---|
| `RUNTIME-CLEANUP-001` | `BLOCKING` | `RunState.close()` assigns the first cleanup throwable to `firstFailure`, then unconditionally calls `firstFailure.addSuppressed(failure)` for later failures. If a distinct later resource throws that same object, Java rejects self-suppression and aborts remaining cleanup, replacing the promised attempt-all/primary-failure behavior. Existing tests use distinct throwable objects. | Runtime. Plan a separate bounded Java/test/Javadoc repair after 0011; do not create Runtime 0012 here. |
| `DOCUMENTATION-STATUS-001` | `BLOCKING` | Three review-only architecture pages still say only initial Model foundations or placeholder modules exist, while the review-only architecture-test guide says focused implementations are absent. Current source, roadmap, and existing architecture suites disprove those statements. | Documentation/architecture explanation owner. Correct in separate authorized documentation planning; no architecture rule changes. |
| `ARCHITECTURE-ENFORCEMENT-001` | `BLOCKING` | `testing/architecture-tests` has Config, Planning, and NN/Training suites but no Runtime dependency/hot-path suite, while `ARCHITECTURE.md` names Runtime concrete-backend/Engine independence and `Operation`/`CompiledNode` exclusion as expected enforcement. | Architecture-test owner. Add focused falsifiable enforcement in a separate task; do not modify tests in 0011. |
| `DOWNSTREAM-OWNERSHIP-001` | `NON_BLOCKING_DEFERRED` | Public values, Prepare orchestration, Engine composition, concrete execution, Trace payloads, Config policy, and tuning have explicit non-Runtime owners. | Keep each current owner/task Draft; no specification created here. |
| `RUNTIME-BOUNDARY-001` | `NO_CHANGE_CONFIRMED` | Exact plan identity, nominal separation, cold binding, validity, transfer, publication, result lease, isolation, package cohesion, imports, and hot traversal otherwise agree across source, tests, rendered Javadocs, APIs, guide, glossary, and architecture. | No change in 0011. |

The required evidence matrix follows. Every row uses only the task's finding vocabulary.

| Area | Current contract | Primary evidence | Required invariant or question | Finding label | Disposition |
|---|---|---|---|---|---|
| Production inventory | 25 files in five packages | `find modules/runtime/src/main/java -type f \| sort` | Every production file inventoried | `NO_CHANGE_CONFIRMED` | Inventory above |
| Test inventory | 17 suites, prior 143 tests | `find modules/runtime/src/test/java -type f ! -name .gitkeep \| sort`; XML aggregation | Every suite/file inventoried | `NO_CHANGE_CONFIRMED` | Inventory above |
| Package inventory | memory/resource/execution/schedule/run | Source paths and package summaries | Five cohesive owners | `NO_CHANGE_CONFIRMED` | Retain |
| Public-surface inventory | 34 rendered public type pages | source declaration scan; Javadoc page list/allclasses index | Necessary, minimal, cohesive surface | `NO_CHANGE_CONFIRMED` | Retain |
| Package-private inventory | `RunStateCreation`, association accessors, runner internals | source declaration/modifier scan; generated-page absence | Internal mechanisms stay internal | `NO_CHANGE_CONFIRMED` | Retain |
| Generated-Javadoc inventory | 34 type pages, five package summaries and indexes | `find modules/runtime/build/docs/javadoc ...`; rendered index inspection | Visibility equals source | `NO_CHANGE_CONFIRMED` | Retain |
| Task-history inventory | Runtime 0001-0010 Complete and unchanged | task files plus `git diff --exit-code -- ...0001...0010...` | History corroborates primary evidence | `NO_CHANGE_CONFIRMED` | Preserve |
| Prepared immutability and exact plan | Immutable snapshots; exact plan reference across recipes/state | constructors/tests for plan, execution, schedule, binding | Reference identity survives each boundary | `NO_CHANGE_CONFIRMED` | Retain |
| Nominal identities | Model/planning IDs, slots, representations, and run bindings remain distinct | Runtime imports/types and API/glossary | No numeric/structural substitution | `NO_CHANGE_CONFIRMED` | Retain |
| Public necessity/minimality | Current lifecycle/backend seams only | production use scan and public API | Every public declaration has a current role | `NO_CHANGE_CONFIRMED` | Retain |
| Validation/snapshots/equality/retention | Ordered validation, immutable list snapshots, ordinary record equality, exact documented references | constructors and focused tests | Fail closed without hidden normalization | `NO_CHANGE_CONFIRMED` | Retain |
| Dense caller input and all-or-cleaned creation | Complete prevalidation, deterministic creation, rollback | `RunStateCreation`; `RunStateCreationTest` | No callback before caller validation; no leaked creation | `NO_CHANGE_CONFIRMED` | Retain |
| Borrowed/run-owned lifetime | Inputs borrowed; created buffers/workspaces run-owned through result close | `RunState`, `RunResult`, tests | Cleanup never closes inputs | `NO_CHANGE_CONFIRMED` | Retain |
| Multi-copy validity/workspaces | Independent buffer bits; workspace has no logical validity | `RunState`; `RunStateTest` | Zero/one/many valid copies permitted | `NO_CHANGE_CONFIRMED` | Retain |
| Cold binding | Executable, transfer, publication bind before traversal | binder sources; runner ordering tests | Compatibility/resolution completed cold | `NO_CHANGE_CONFIRMED` | Retain |
| First-only creation and rollback | Optional first occurrence; reverse rollback | schedule/creation source and tests | One prefix; created resources cleaned | `NO_CHANGE_CONFIRMED` | Retain |
| Transfer/materialization | Destination-valid no-op; source required; success-only destination validity | transfer source/tests | No fallback/coherence/invalidation | `NO_CHANGE_CONFIRMED` | Retain |
| Executable overlap/transitions | Reads before all-output invalidation; exact successful writes valid | runner source/tests | In-place reads old value; failure stays invalid | `NO_CHANGE_CONFIRMED` | Retain |
| Publication suffix/aliasing | Dense suffix, exact valid copy, aliases allowed, no fallback | schedule/publication/result source/tests | Ordered complete publications only | `NO_CHANGE_CONFIRMED` | Retain |
| Whole-state result lease | Private alias snapshot; count/lifecycle only | `RunResult`; tests; rendered Javadoc | No value exposure; cleanup transfers after validation | `NO_CHANGE_CONFIRMED` | Retain |
| Cleanup | Closed-first, reverse, idempotent, attempt-all, primary/suppressed | `RunState.close()` and `RunStateTest` | Same throwable identity must not abort cleanup | `BLOCKING` | `RUNTIME-CLEANUP-001` |
| Empty/repeated/concurrent runs | Empty/repeated valid; concurrent calls isolated | schedule/state/runner tests | No mutable state/resource sharing | `NO_CHANGE_CONFIRMED` | Retain |
| Direct-reference hot path | Bound fields and primitive arrays; no prohibited mechanisms | runner/bound sources; bytecode tests | No graph/discovery/search/reflection/map/boxing/sync/hot allocation | `NO_CHANGE_CONFIRMED` | Retain |
| Package/dependency surface | Three declared inward dependencies; only Runtime/JDK imports | Runtime Gradle and import scan | No forbidden edge or ownership leak | `NO_CHANGE_CONFIRMED` | Retain |
| Adjacent module boundaries | Model/Config/Backend Contract/Trace/Compiler/Planning/Prepare/Engine/backend owners remain separate | architecture, ADRs, adjacent master plans/source boundaries | Missing capabilities have explicit owners | `NO_CHANGE_CONFIRMED` | Retain |
| Architecture enforcement | No Runtime-focused architecture suite exists | architecture-test file inventory and guide | Contract-named Runtime rules durably enforced | `BLOCKING` | `ARCHITECTURE-ENFORCEMENT-001` |
| Runtime/Public API and examples | Current lifecycle and planned values/composition are distinguished | Runtime/Public API review and examples | No current/planned confusion | `NO_CHANGE_CONFIRMED` | Retain |
| Backend guide and glossary | Current extension patterns/terms match source | guide/glossary review | No concrete backend or hidden discovery claim | `NO_CHANGE_CONFIRMED` | Retain |
| General architecture status | Three architecture pages and the architecture-test guide claim obsolete early repository state | exact `rg` matches in four review-only files | Implementation status must be current | `DOCUMENTATION_DRIFT` | `DOCUMENTATION-STATUS-001`; blocking because correction is outside scope |
| Repository checkpoint | Repository, architecture, Runtime Javadoc are required once | combined Gradle command | Capability checkpoint passes | `NO_CHANGE_CONFIRMED` | Record final evidence below |
| Conformance/integration | No concrete backend or Engine path | backend/engine plans and source inventories | Applicability correctly deferred | `NON_BLOCKING_DEFERRED` | Not triggered |
| Public result values/Engine | Not implemented in Runtime | `RunResult`, Engine plan, public API | Explicit non-Runtime owner | `NON_BLOCKING_DEFERRED` | Engine/result owner |
| Prepare 0003 | Translation/orchestration remains Draft | Prepare master plan/task inventory | No Runtime semantic gap | `NON_BLOCKING_DEFERRED` | Leave Draft without specification |
| Concrete backend execution | Backend plans remain Draft | backend plans/source inventory | Physical work remains backend-owned | `NON_BLOCKING_DEFERRED` | No conformance trigger |
| Trace/Config/tuning | Run payload, run policy, tuning remain downstream | Trace/Config/tuning plans and boundaries | No speculative Runtime consumption | `NON_BLOCKING_DEFERRED` | Leave current owners |
| Status/history | 0011 audit complete but Runtime milestone open | task/master plan/roadmap checks | Verdict-synchronized status, no 0012 | `NO_CHANGE_CONFIRMED` | Task Complete; milestone remains open |

## Checkpoint evidence and Runtime-milestone decision

The one final combined checkpoint passed:

```bash
./gradlew test :testing:architecture-tests:test :modules:runtime:javadoc
```

Gradle reported `BUILD SUCCESSFUL` with 53 actionable tasks, 8 executed and 45 up-to-date.
Current XML evidence contains 205 suites and 1,530 tests with zero failures, errors, or skips:
Runtime 17/143, architecture 3/3, Model 127/1,031, Compiler 31/208, Planning 9/68, Prepare 7/22,
Backend Contract 4/22, Config 4/17, and Trace 3/16. Generated inspection found all five Runtime
package summaries, 34 public type pages, `allclasses-index.html`, and `overview-tree.html`, with
no package-private Runtime declaration exposed.

Final Markdown, working-tree, exact-scope, status, order, and history checks passed as recorded in
the completed task. The final changed-path set is exactly this artifact, task 0011, the Runtime
master plan, and the roadmap. Runtime 0001-0010 are unchanged; no Ready task remains; no Runtime
0012 or Prepare 0003 specification exists; and `git diff --check` reports no error.

Runtime 0011 is `Complete` because the audit is complete, while the Runtime milestone and roadmap
area remain in progress under `BLOCKING_GAP`. Prepare 0003 and every later task remain Draft
without a detailed specification. No Runtime 0012 row or specification is created.
