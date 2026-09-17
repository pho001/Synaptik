# Task 0006: One-Shot Scalar-Objective Backward Convenience

## Status

Complete

## Goal

Add the smallest ordinary Engine-owned one-shot backward convenience for one scalar objective and
an explicit ordered non-empty list of differentiation targets. One synchronous call lowers to
Compiler's existing absent-unit-seed functional-gradient contract, freshly compiles, prepares,
runs, materializes the objective and every first-order gradient under one aggregate byte bound,
closes temporary state, and returns only detached immutable host values.

This complements the reusable ordinary explicit-seed compile path and the advanced full-request
path. It adds no Tensor execution, mutable autograd state, target inference, or Compiler semantics.

## Motivation and current boundary

Engine 0003 provides reusable explicitly seeded first-order compilation and ordered gradient
publications. Engine 0004 provides bounded detached host materialization, and Engine 0005 provides
one-shot forward orchestration. Compiler 0006 already defines the missing scalar seed: an absent
cotangent is legal only for an exact scalar floating gradient-eligible output and denotes an exact
positive-one logical splat of that output type. The selected outward Engine mapping remains
valid. Compiler 0006B5, Prepare 0005, and CPU 0010H now close the complete logical,
shared-resource, and physical-materialization chain for that source-only publication. Complete
Engine 0005A already
supplies the transient expression-leaf inventory and authoritative compiled-input selection seam
reused by this later convenience; Engine 0006 adds no input-classification or liveness rule.

```text
scalar objective + explicit targets
  -> transient reachable provenance-free Tensor-leaf inventory
  -> one absent-seed, ERROR-policy Compiler stage
  -> authoritative final compiled-input selection
  -> fresh compile -> prepare -> run
  -> one forward objective then target-ordered gradients
  -> aggregate preflight -> independent copies -> cleanup
  -> detached objective + immutable target-aligned gradients
```

### Prerequisite diagnosis

When the absent seed is itself the requested gradient, Compiler captures it as a compile-time
constant graph input and graph output with no producer partition and no consumer partition.
Planning correctly retains that producerless/consumerless graph-output obligation. The former
blocker had three parts: the final descriptor had unresolved layout, Prepare accepted no complete
resource contribution for a value outside partition-node projections, and CPU therefore had no
assigned representation to initialize.

The completed owner-ordered repair closes each part without changing Runtime. Compiler 0006B5
assigns the canonical contiguous logical layout only to eligible fully static source-only
published constant splats. CPU 0010H derives the exact physical geometry and contributes the
resource through Prepare 0005's complete validated producerless-resource handoff; Prepare assigns
its deterministic shared slot. CPU then emits an initialized-buffer recipe, and existing Runtime
cold setup creates one fresh run-owned, initially valid representation per `RunState`. The
architecture-correct order is:

```text
Compiler 0006B5 logical descriptor closure (Complete)
  -> Prepare 0005 source-only publication resource handoff and assignment (Complete)
  -> CPU 0010H source-only constant physical declaration/materialization (Complete)
  -> Engine 0006 outward convenience

Engine 0005A automatic input discovery and compute convenience (Complete; seam available)
  -> Engine 0006 outward convenience
```

Runtime needs no new task. Its initialized representation, initial-validity, ordered
publication/alias, result-lease, and cleanup contracts cover the required lifecycle. Compiler
0006B5 supplies the resolved canonical descriptor, Prepare 0005 validates and assigns the complete
source-only resource set, and CPU 0010H creates a fresh initialized, initially valid physical
representation for each run state.

## Exact proposed public API

Add exactly:

```java
public ScalarObjectiveBackwardResult backward(
        Tensor objective,
        List<Tensor> targets,
        long maximumTotalBytes);

public final class ScalarObjectiveBackwardResult {
    public HostTensorValue objective();
    public List<HostTensorValue> gradients();
}
```

The result has one package-private constructor
`ScalarObjectiveBackwardResult(HostTensorValue objective, List<HostTensorValue> gradients)`.
It rejects nulls in component and list order, snapshots list membership, and retains the exact
already-detached values. It has no other public member, public/protected constructor, field,
nested type, close method, equality override, Tensor/target metadata, or inward reference.

`objective()` is the scalar forward occurrence. `gradients().get(i)` is the first derivative for
exact requested `targets.get(i)`. The list and values are immutable, resource-free, and readable
after Engine and caller storage close. A distinct carrier is required because a combined list
would give position zero a different role; a richer request/publication hierarchy adds no current
need. Ordinary signatures continue to expose only Engine, Model, and JDK types.

## Detailed semantics and invariants

### Validation and Compiler lowering

Engine admission precedes argument inspection. Then validate: non-null `objective`; non-null,
non-empty, exact-object-identity-unique `targets` with indexed null/duplicate diagnostics;
and non-negative `maximumTotalBytes` with `maximumTotalBytes must be non-negative: <value>`.
Snapshot the target list before inward work.

Before or around compilation, reuse Engine 0005A's identity-safe iterative traversal of the
objective Tensor expression closure to inventory exact provenance-free leaf Tensors. After
compilation, select only leaves matching final `CompiledGraph.inputs()` entries and preserve that
authoritative order. Parameters and buffers participate as ordinary reachable leaves. Do not
accept an explicit input list, traverse Compiler IR, infer liveness, read storage for an
unselected leaf, or retain Tensor/provenance state after the synchronous call.

Compiler, not Engine, validates that the objective has exact `Shape.scalar()`, floating type, and
`requiresGrad == true`; that every target is floating, gradient-eligible, identity-present in the
complete objective-rooted forward inventory, and differentiably connected; and that every
operation/attribute route is supported. Targets may be leaves, intermediates, or the objective.

Lower exactly once with `FORWARD_AND_BACKWARD`, singleton forward output, standard optimization,
unconstrained intent, neutral scoring, and:

```java
new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
        List.of(new FunctionalGradientRequest.ForwardTensorReference(objective)),
        List.of(Optional.empty()),
        targets,
        false,
        FunctionalGradientRequest.DisconnectedPolicy.ERROR)))
```

The absent seed is Compiler's exact positive-one scalar logical splat, never a caller input or
Engine Tensor. ERROR is fixed. The existing ordinary
`compile(forwardOutputs, cotangentSeeds, targets)` and advanced one/two-stage request path remain
unchanged; explicit seeds, ZERO, multiple outputs, and higher order use those paths.

### Execution, result order, identity, and aliases

Under one admission: validate; compile once; prepare once; bind/run once through Engine 0003;
validate publications; preflight all returned bytes; copy objective then gradients; construct the
carrier; close the temporary result/wrappers; then return.

The result has exactly `1 + targets.size()` occurrences. Index zero must be FORWARD with the
objective `TensorId` and empty derivative/target metadata. Index `i + 1` must be GRADIENT with
`targets.get(i).id()`, derivative order one, and target index `i`. Every descriptor is non-null.
Any mismatch is an index/count-specific `IllegalStateException`; Engine neither searches, repairs,
reorders, nor merges roles.

Targets are identity-unique; equal descriptors do not merge identities. Automatically discovered
leaves match final Compiler `BindableInput` identities exactly once and are selected in final
binding order. Repeated exact leaves produce one binding; distinct equal-descriptor leaves remain
distinct. Shared caller storage gets distinct borrow wrappers. Compiler may map several target
roles—or a forward and gradient role—to one final value/representation. Each occurrence is
nevertheless copied independently into a distinct uncached `HostTensorValue`; no physical-alias
or equality API is added.

### Aggregate-byte accounting, ownership, and limits

Before the first copy, inspect objective then all gradient descriptors: require fully static Shape
and resolved layout; compute checked zero-before-product element count and checked canonical byte
count; reject any value above `Integer.MAX_VALUE`; and add counts with `Math.addExact`. Compare the
complete sum with the limit and fail with:

```text
total canonical byte count exceeds maximumTotalBytes: required=<sum>, maximum=<limit>
```

The bound includes objective and all gradient payloads only—not inputs, recipes, Runtime
buffers/workspaces, object overhead, defensive copies, or peak memory. Pass each exact count as
the tight per-copy limit. Validate returned value metadata/counts against preflight. No partial
return, cache, deduplication, streaming, mapping, typed-array conversion, caller destination, or
Tensor reconstruction is allowed.

Automatically selected inputs use the existing TensorId/descriptor checks and per-Tensor
synchronized `hostStorage()` snapshot in final binding order. Storage and arenas remain
caller-owned and must stay live, accessible, and free from conflicting mutation through
synchronous cleanup. Snapshots are not atomic across selected leaves, do not pin memory, and do
not synchronize payload writes.

### Failure, cleanup, close, and concurrency

Failure precedence is closed-Engine admission; structural arguments; Compiler; Prepare; logical
binding/run; publication and aggregate preflight; then copies in result order. Preserve original
unchecked failure/Error identity. Before result construction, use Engine 0003 reverse rollback.
Afterward, close the temporary result and wrappers exactly once. A distinct cleanup failure is
suppressed on the primary in encounter order; never self-suppress. Without an earlier failure,
cleanup failure becomes primary. No partial carrier is returned.

Reuse the existing lifecycle gate and ungated helpers; do not nest public lifecycle calls or add a
gate, registry, callback executor, or service. The temporary result is never registered or
returned. Engine close waits for an admitted call through copies and cleanup. A losing call fails
before arguments; a winning call may return detached values after closure begins. Concurrent
calls have isolated Runtime states. No async, cancellation, retry, or fallback is added.

The intended support remains CPU-only: exactly one non-empty maximal CPU partition, supported
kinds and types, fully static Shapes, and resolved compatible layouts. Compile success does not
guarantee prepare/run success. The Engine-owned real fixture will use one resolved scalar FLOAT32
gradient-eligible leaf and its supported
`contiguous()` objective, discover the leaf automatically, and verify the scalar value and typed
positive-one gradient without claiming broader CPU gradient coverage. Current contracts now make
that non-empty CPU graph plus source-only seed publication preparable and materializable; the
fixture remains Engine-owned implementation evidence rather than prerequisite-task evidence.

## Scope

- Add the exact method and result type, fixed Compiler mapping, one-admission orchestration,
  focused tests, Javadocs, API/glossary updates, and synchronized planning evidence.
- Treat forward objective and target-aligned gradients as one cohesive detached result.

## Out of scope

- Tensor execution/backward/gradient state; inferred/default-all targets; no-argument
  `withBackward()`; non-scalar objectives; explicit-seed or multi-output convenience; ZERO;
  vector-Jacobian/Jacobian materialization; second order; `createGraph`; `TRAINING_STEP`;
  optimizer/session behavior.
- New request/options/builders, default/unlimited/per-value limits, caches/reuse, tuning,
  persistence, typed arrays, cross-backend transfer, mixed schedules, or another backend.
- Changes to Model, Compiler, Config, Planning, Prepare, Runtime, CPU, NN, Training, build files,
  dependencies, architecture contracts/pages, ADRs, architecture-test source, conformance tests,
  generated code, or Engine 0007–0008 specifications.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Runtime/Prepare/backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Planning guide](../../../planning-guide.md), [roadmap](../../../roadmap.md), and
  [Engine master plan](../master-plan.md)
- Engine [0003](0003-typed-logical-input-binding-and-published-result-access.md),
  [0004](0004-explicit-host-materialization-boundary.md), and
  [0005](0005-one-shot-forward-convenience.md)
- Compiler [0006](../../compiler/tasks/0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md),
  [0006B4](../../compiler/tasks/0006b4-stable-caller-input-tensor-identity-bindings.md), and
  [0006B5](../../compiler/tasks/0006b5-published-compile-time-constant-descriptor-closure.md)
- Runtime [0015](../../runtime/tasks/0015-leased-publication-representation-access.md) and CPU
  [0010G](../../../backends/cpu/tasks/0010g-canonical-caller-owned-host-snapshot-export.md)
- Prepare [0005](../../prepare/tasks/0005-producerless-published-constant-resource-handoff-and-shared-slot-assignment.md)
  and CPU
  [0010H](../../../backends/cpu/tasks/0010h-source-only-published-constant-cpu-materialization.md)

## Architecture constraints

Engine owns outward orchestration; Compiler exclusively owns gradient semantics; Model owns
immutable Tensor identity/provenance; Runtime owns isolated run state; CPU owns physical copying.
No owner, dependency, inward public contract, or architecture rule changes. Engine imports no CPU
internal type. It may reuse 0005A's transient traversal of immutable Tensor expression provenance,
but it does not traverse Compiler IR, infer graph liveness, or implement derivative, seed, or
disconnected policy. Stop and replan if an inward API, another policy/output mode, dependency,
architecture test, or second production module must change.

## Package impact

The existing `io.github.pho001.synaptik.engine` facade gains the result beside
`HostTensorValue`; `Engine` owns the public call and `AdvancedEngine` remains the private ordinary
orchestration/gate owner with no advanced public addition. No package/module is added.

## Affected files

Exact future allowlist (sixteen paths):

1. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/ScalarObjectiveBackwardResult.java` (new)
2. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java`
3. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java`
4. `modules/engine/src/main/java/io/github/pho001/synaptik/engine/package-info.java`
5. `modules/engine/src/test/java/io/github/pho001/synaptik/engine/EngineTypedLifecycleTest.java`
6. `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/EngineTypedPublicShapeTest.java`
7. `modules/engine/src/test/java/io/github/pho001/synaptik/engine/api/AdvancedEnginePublicShapeTest.java`
8. `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineTypedLifecycleIntegrationTest.java`
9. `docs/api/public-api.md`
10. `docs/api/compile-api.md`
11. `docs/api/runtime-api.md`
12. `docs/api/training-api.md`
13. `docs/glossary.md`
14. This task
15. `docs/planning/modules/engine/master-plan.md`
16. `docs/planning/roadmap.md`

Review without editing `RunResult`, `HostTensorValue`, ordinary handles, all advanced public
types, Model/Compiler/Prepare/Runtime/CPU contracts, Tensor API, architecture/ADRs, builds,
architecture tests, conformance tests, other modules, and Engine 0007–0008; record reasoned
no-change conclusions.

## Maximum scope

At most those sixteen paths. Integration changes are justified by new end-to-end behavior.
Architecture/conformance source is excluded because no dependency/shared-backend boundary changes;
run the existing focused architecture test unchanged. Stop and replan before any seventeenth path,
inward contract, dependency/build edit, or other-module executable change.

## Acceptance criteria

- Exact public shape/import boundary, constructor visibility, immutability, and target-aligned
  result semantics are automated from a distinct package.
- Validation order and fixed one-stage absent-seed/ERROR mapping match this specification;
  Compiler alone owns semantic preflight.
- Explicit-seed ordinary and full advanced paths remain unchanged and passing.
- Automatic leaf inventory and authoritative compiled-input selection exactly reuse 0005A;
  selected-input storage/borrow/run behavior exactly reuses 0003.
- Publications are exactly objective then derivative-order-one targets with defended identities,
  indices, descriptors, and alias multiplicity.
- Complete static/resolved/individual/aggregate preflight precedes copying; exact-limit,
  overflow/JVM ceiling, alias, and zero-copy-on-preflight-failure cases are covered.
- Failures return no partial carrier and preserve cleanup/suppression identity.
- Bounded-latch tests cover close/admission/concurrent isolated runs without sleeps.
- With Compiler 0006B5, Prepare 0005, CPU 0010H, and Engine 0005A Complete, real supported scalar
  CPU integration verifies
  objective, positive-one gradient, byte bound, automatically selected input identity,
  caller ownership, and detached post-close access. Engine owns this final end-to-end test; the
  prerequisite tasks own their focused descriptor, handoff/assignment, and CPU materialization
  tests respectively.
- Javadocs and Public/Compile/Runtime/Training/glossary docs explain exact semantics, limitations,
  lower-level alternatives, and no Tensor mutation/training implication.
- Only allowlisted paths change; 0005 remains Complete; task 0006 reaches Complete only through
  the normal implementation statuses; 0007–0008 remain Draft and unspecified.

## Tests / validation

Implementation task-tier commands:

```bash
./gradlew :modules:engine:test
./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.EngineTypedLifecycleIntegrationTest
./gradlew :testing:architecture-tests:test --tests io.github.pho001.synaptik.testing.architecture.EngineCompositionContractTest
```

Do not rerun unchanged Compiler/Runtime/Prepare/CPU or conformance suites. Root tests remain Engine
0008/CI. The clean documentation pass reuses Java evidence unless executable behavior changes and
runs:

```bash
./gradlew :modules:engine:javadoc
git diff --check
```

It also checks generated pages, exact public inventories, a distinct-package fixture, Markdown
targets/anchors/unique headings/fences/trailing whitespace/LF/exact final newline, terminology,
the sixteen-path scope, forbidden changes, preservation hashes, and status/link agreement. Do not
run Java tests for this planning-only change.

## Dependencies

Engine 0001–0005A, Compiler 0006/0006B4–0006B5, Prepare 0005, Runtime 0015, and CPU 0010G–0010H are
Complete. Engine 0005A already supplies automatic reachable-leaf discovery plus selection in
`CompiledGraph.inputs()` order. Compiler 0006B5 supplies the resolved canonical descriptor,
Prepare 0005 supplies complete source-only resource handoff and shared assignment, and CPU 0010H
supplies physical declaration plus fresh per-run initialized materialization. Current Model
scalar/type/Tensor/storage contracts and Runtime lifecycle contracts are sufficient; no
Runtime task, architecture decision, or material outward API choice remains.

## Documentation-focused clean-context pass

After Java/tests stabilize, provide a distinct clean documentation context this task, final diff,
test evidence, dirty baseline/hashes, fixed request mapping, order/alias/byte/ownership/cleanup/
concurrency decisions, relevant completed contracts, and five authorized explanatory paths. Apply
Planning, API/Javadoc, General, and Example profiles. The complete example uses
`Engine.standard()`, a resolved scalar FLOAT32 leaf and gradient-eligible `contiguous()` objective,
an explicit target list, aggregate limit, and caller arena; it verifies objective and
positive-one gradient bytes and post-close access. Explain fresh compilation/preparation and route
explicit seeds/reuse/full policies to existing lower-level APIs. Record reasoned no-change
conclusions for Tensor API, architecture/ADRs, inward docs, builds/tests, and other modules.

## Follow-up tasks

- Engine 0007 and 0008 remain Draft without specifications.
- Non-scalar, ZERO, higher-order, optimizer/training, reuse, and cross-backend conveniences require
  separately demonstrated needs; create none here.

## Architecture impact

Expected impact: None. This maps an Engine call to established owner contracts. Stop on contrary
implementation evidence rather than changing architecture in this task.

## Implementation prompt

```text
Work in /Users/phujka/IdeaProjects/Synaptik on Engine task 0006. Do not use GSD, commit, or push.
Read AGENTS.md, ARCHITECTURE.md, the architecture index, planning guide, Engine master plan, this
task, and final Engine 0003–0005A, Compiler 0006/0006B4–0006B5, Prepare 0005,
Runtime 0015, and CPU 0010G/0010H contracts in full. Verify Compiler 0006B5, Prepare 0005,
CPU 0010H, and Engine 0005A are Complete before editing. Inspect affected source/tests/docs and
the dirty worktree.

Implement exactly the sixteen-path allowlist: Engine.backward, ScalarObjectiveBackwardResult,
the one-admission absent-scalar-seed/explicit-target/ERROR mapping, 0005A automatic input
discovery/final-binding selection, publication/aggregate-copy/cleanup semantics, tests, and
authorized docs. Preserve reusable/advanced APIs and dirty work. Do not add an explicit input
list.
Add no Tensor state, inferred target, other policy/stage, request abstraction, cache/reuse, tuning,
transfer, dependency, architecture, inward-contract, or later-task change. Stop on conflict. Use
apply_patch and run final task-tier validation once. Then hand the diff/evidence to a distinct
clean documentation-focused context; mark Complete only after all implementation/documentation
gates pass.
```

## Local decisions

- Use `backward` with explicit objective/targets; this matches established terminology without a
  mutable or no-argument meaning.
- Return objective plus gradients in a role-separating carrier; avoiding a second forward call is
  useful, and all bytes share the aggregate bound.
- Fix Compiler's absent unit seed and ERROR policy; other choices already have lower-level APIs.
- Validate structure in Engine and semantics in Compiler; preflight all results before copying.
- Reuse the current gate and temporary result cleanup; add no resource owner or registry.

## Known limitations

Exact scalar floating objective, explicit unique connected targets, fixed ERROR, CPU-only
static/resolved supported execution, fresh compile/prepare, eager complete materialization,
`Integer.MAX_VALUE` per value, non-atomic selected-input association snapshots, and no training/
optimizer/checkpoint/transfer meaning. Engine 0005A's automatic reachable-leaf discovery plus
selection in `CompiledGraph.inputs()` order and the complete source-only constant chain are
already available.

## Validation evidence

Planning context: `01a0a557-092d-7cb0-baa4-d6ebb3527336`.

- Read the required architecture/planning/task lineage, applicable API/glossary/docs profiles,
  current source/tests, and starting dirty diff/status.
- Confirmed Compiler's exact absent-seed, target, ERROR, publication ordering/alias contracts and
  Engine 0003–0005's sufficient outward binding, lifecycle, aggregate-copy, and cleanup seams.
- Follow-up diagnosis context `01a0a570-06d4-7012-814b-ca71af8676e7` disproved inward readiness:
  the absent seed is a source-only published constant with unresolved layout, Prepare contributes
  and assigns only partition-analysis resources, CPU receives no assignment, and Engine preflight
  would reject unresolved layout even with a fabricated assignment.
- Reassessment at main commit `f2dfecbd653766eef19fa76306744067d4cdbd08` confirmed Complete
  Compiler 0006B5 closes the canonical logical descriptor, Complete Prepare 0005 validates and
  assigns the source-only resource, and Complete CPU 0010H contributes its geometry and builds a
  fresh initialized, initially valid representation for each Runtime run state.
- Confirmed current Engine 0005A leaf discovery/final-binding selection, publication metadata,
  host-copy preflight, and cleanup helpers support the proposed Engine-only implementation without
  a new inward API. Runtime's existing initialized-representation, ordered-publication, lease,
  isolated-state, and cleanup contracts require no new task.
- This clean planning/documentation reassessment changed exactly this task, the Engine master plan,
  and the roadmap. `git diff --check`, exact three-path scope, Markdown relative-link/anchor,
  heading/fence/whitespace/LF/final-newline, stale CPU-0010H wording, and cross-file
  status/order/dependency checks all passed.
- Preserved baseline copies/diffs for the pre-edited Engine master plan and roadmap.
- Java tests/Javadoc were not run because this assignment is planning-only.

Implementation context: `01a0b11d-cc89-7590-8836-0555b05e7001`.

- Implemented the exact ordinary `Engine.backward(objective, targets, maximumTotalBytes)` surface,
  package-private fixed request mapping, detached `ScalarObjectiveBackwardResult`, publication and
  byte preflight, alias-copy, cleanup, close, and concurrency behavior within the eight Java/test
  paths.
- `./gradlew :modules:engine:test` passed with 41 tests and zero failures, errors, or skips.
- The focused `EngineTypedLifecycleIntegrationTest` passed all 6 tests with zero failures, errors,
  or skips. The focused `EngineCompositionContractTest` completed successfully; its report records
  1 passing test.
- `git diff --check`, `javap`, public-shape and source scans, exact eleven-path implementation
  scope, and preservation of the initial three-path planning diff passed. No validation command
  failed; a pre-final-review test-only latch threshold was corrected before the final tier.

Documentation context: `01a0b126-f60b-7a11-8d7c-621e532952f0`.

- Independently reviewed the final implementation and all four changed tests against every
  acceptance criterion. The implementation fixes one absent positive-one scalar seed and
  `DisconnectedPolicy.ERROR`, reuses automatic leaf discovery plus final compiled-input
  selection, validates objective-first then target-ordered publications, preflights the complete
  static/resolved per-value and aggregate byte set before copying, copies aliases independently,
  and returns detached values after one fresh lifecycle and cleanup.
- Finalized Public, Compile, Runtime, and Training API explanations, the existing one-shot
  glossary family, status/evidence in this task, the Engine master plan, and the roadmap. The
  complete public example uses `Engine.standard()`, caller-owned resolved scalar `FLOAT32`
  storage, `contiguous()`, an explicit target list and aggregate bound, verifies objective and
  positive-one gradient bytes, and reads both after Engine/storage closure.
- The four production Javadocs were already accurate and were left byte-for-byte unchanged.
  Their SHA-256 hashes remain the implementation hashes recorded in the handoff, so no executable
  Java token comparison was necessary. All four test hashes likewise remain unchanged.
- `./gradlew :modules:engine:javadoc`, generated-page inspection, `javap -public`, the
  package-private result-constructor inspection, and a distinct-package public fixture all passed.
  Markdown target/anchor, unique-heading, balanced-fence, trailing-whitespace, LF, final-newline,
  exact sixteen-path, forbidden-change, status-agreement, stale-wording, signature-leak, and
  `git diff --check` validations passed.
- Reused the implementation Java-test evidence as required; no Java test suite was rerun because
  executable Java and tests were unchanged during documentation finalization.

## Implementation notes

- `Engine.backward(...)` admits once, snapshots and validates structural arguments, inventories
  reachable provenance-free leaves, constructs one singleton absent-seed/ERROR Compiler request,
  and binds only final compiled inputs in their authoritative order.
- The result boundary requires exactly one objective publication followed by one derivative-order-
  one publication for every target position. Complete byte preflight precedes objective-first and
  target-ordered independent copies; returned values own no inward resource.
- Existing ordinary explicit-seed compilation and advanced full-request compilation were not
  widened or redirected. Pure zero-node and mixed/multi-partition compositions remain rejected.
- No inward module, dependency, build, architecture, backend-conformance, Tensor/Model source, or
  Engine 0007–0008 specification changed.
- Tensor API and Model source required no change because Engine consumes existing immutable Tensor
  identity, descriptor, provenance, and storage contracts without changing them.
- Architecture, ADRs, and architecture-test source required no change because ownership,
  dependency direction, and module boundaries are unchanged; the existing focused architecture
  test passed unchanged.
- Compiler, Prepare, Runtime, and CPU contracts and documentation beyond the allowed API pages
  required no change because the completed 0006B5 -> Prepare 0005 -> CPU 0010H chain already
  supplies every inward descriptor, assignment, initialized representation, publication, and copy
  contract used here.
- Build files and dependencies required no change because all named types are already available to
  Engine and the integration test. Backend conformance required no change because this task adds
  Engine orchestration, not backend behavior.
- Other modules and plans required no change because the capability is confined to Engine; Engine
  0007–0008 remain Draft and unspecified as required.

## Completion summary

Completed changes:
- Added the exact one-shot scalar-objective backward API and detached objective/gradient carrier.
- Implemented automatic input selection, fixed Compiler request lowering, publication defense,
  complete aggregate preflight, independent copying, lifecycle cleanup, and concurrency behavior.

Files changed or created:
- Four Engine production paths, three Engine test paths, and one integration-test path listed in
  the allowlist.
- `docs/api/public-api.md`, `docs/api/compile-api.md`, `docs/api/runtime-api.md`,
  `docs/api/training-api.md`, `docs/glossary.md`, this task, the Engine master plan, and the
  roadmap.

Tests and validation performed:
- Reused 41 passing Engine tests, 6 passing focused integration tests, and 1 passing focused
  architecture test from the implementation context.
- The documentation context passed Engine Javadoc, generated-page and public-shape inspection,
  fixture compilation, Markdown/format/scope/status scans, and `git diff --check`.

Documentation impact:
- Existing production Javadocs were reviewed and remained accurate without edits. Public,
  Compile, Runtime, and Training API pages now document the current call and its boundaries; the
  glossary adds the smallest matching one-shot backward entry; planning status and evidence are
  synchronized.

Unresolved issues:
- None within Engine 0006.

Required follow-up:
- None for this task. Engine 0007–0008 remain Draft and unspecified.

Status: Complete
