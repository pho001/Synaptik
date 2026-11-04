# Task 0006: One-Shot [Scalar](../../../../glossary.md#term-scalar)-[Objective](../../../../glossary.md#term-objective) Backward Convenience

## Status

Complete

## Goal

Add the smallest ordinary [Engine](../../../../glossary.md#term-engine)-owned one-shot backward convenience for one [scalar objective](../../../../glossary.md#loss) and
an explicit ordered non-empty list of differentiation [targets](../../../../glossary.md#term-target). One [synchronous](../../../../glossary.md#term-synchronous) call lowers to
[Compiler](../../../../glossary.md#term-compiler)'s existing absent-unit-[seed](../../../../glossary.md#term-seed) functional-[gradient](../../../../glossary.md#term-gradient) contract, freshly compiles, prepares,
runs, materializes the [objective](../../../../glossary.md#term-objective) and every first-order gradient under one aggregate byte bound,
closes temporary state, and returns only [detached](../../../../glossary.md#detached-model-results) [immutable](../../../../glossary.md#term-immutable) [host](../../../../glossary.md#term-host) values.

This complements the reusable ordinary explicit-seed [compile](../../../../glossary.md#compile) path and the advanced full-request
path. It adds no [Tensor](../../../../glossary.md#tensor) execution, [mutable](../../../../glossary.md#term-mutable) [autograd](../../../../glossary.md#autograd) state, target [inference](../../../../glossary.md#term-inference), or Compiler semantics.

## Motivation and current boundary

[Engine](../../../../glossary.md#term-engine) 0003 provides reusable explicitly seeded first-order [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) and ordered [gradient](../../../../glossary.md#term-gradient)
publications. Engine 0004 provides bounded [detached](../../../../glossary.md#detached-model-results) [host](../../../../glossary.md#term-host) [materialization](../../../../glossary.md#materialization), and Engine 0005 provides
one-shot forward orchestration. [Compiler](../../../../glossary.md#term-compiler) 0006 already defines the missing [scalar](../../../../glossary.md#term-scalar) [seed](../../../../glossary.md#term-seed): an absent
[cotangent](../../../../glossary.md#cotangent) is legal only for an exact scalar floating gradient-eligible output and denotes an exact
positive-one [logical splat](../../../../glossary.md#logical-splat) of that output type. The selected outward Engine mapping remains
valid. Compiler 0006B5, [Prepare](../../../../glossary.md#prepare) 0005, and [CPU](../../../../glossary.md#term-cpu) 0010H now close the complete logical,
shared-resource, and physical-materialization chain for that source-only [publication](../../../../glossary.md#term-publication). Complete
Engine 0005A already
supplies the transient expression-[leaf](../../../../glossary.md#term-leaf) inventory and authoritative compiled-input selection seam
reused by this later convenience; Engine 0006 adds no input-classification or [liveness](../../../../glossary.md#term-liveness) rule.

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

When the absent [seed](../../../../glossary.md#term-seed) is itself the requested [gradient](../../../../glossary.md#term-gradient), [Compiler](../../../../glossary.md#term-compiler) [captures](../../../../glossary.md#term-capture-meanings) it as a [compile](../../../../glossary.md#compile)-time
[constant](../../../../glossary.md#term-constant) [graph](../../../../glossary.md#graph) input and graph output with no [producer](../../../../glossary.md#term-producer) [partition](../../../../glossary.md#partition) and no consumer partition.
[Planning](../../../../glossary.md#term-planning) correctly retains that producerless/consumerless graph-output obligation. The former
blocker had three parts: the final [descriptor](../../../../glossary.md#term-descriptor) had unresolved [layout](../../../../glossary.md#layout), [Prepare](../../../../glossary.md#prepare) accepted no complete
resource contribution for a value outside partition-[node](../../../../glossary.md#node) projections, and [CPU](../../../../glossary.md#term-cpu) therefore had no
assigned representation to initialize.

The completed [owner](../../../../glossary.md#term-owner)-ordered repair closes each part without changing [Runtime](../../../../glossary.md#term-runtime). Compiler 0006B5
assigns the [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous) logical layout only to eligible fully static source-only
published constant splats. CPU 0010H derives the exact physical geometry and contributes the
resource through Prepare 0005's complete validated producerless-resource [handoff](../../../../glossary.md#term-handoff); Prepare assigns
its deterministic shared slot. CPU then emits an initialized-[buffer](../../../../glossary.md#term-buffer) recipe, and existing Runtime
cold setup creates one fresh [run](../../../../glossary.md#term-run)-owned, initially valid representation per [`RunState`](../../../../glossary.md#run-state--runstate). The
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
[publication](../../../../glossary.md#term-publication)/[alias](../../../../glossary.md#aliasing-and-arena), result-[lease](../../../../glossary.md#term-lease), and cleanup contracts cover the required [lifecycle](../../../../glossary.md#lifecycle). Compiler
0006B5 supplies the resolved canonical descriptor, Prepare 0005 validates and assigns the complete
source-only resource set, and CPU 0010H creates a fresh initialized, initially valid physical
representation for each [run state](../../../../glossary.md#run-state--runstate).

## Exact proposed public [API](../../../../glossary.md#term-application-programming-interface-api)

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
already-[detached](../../../../glossary.md#detached-model-results) values. It has no other public member, public/protected constructor, field,
nested type, close method, equality override, [Tensor](../../../../glossary.md#tensor)/[target](../../../../glossary.md#term-target) [metadata](../../../../glossary.md#term-metadata), or inward reference.

`objective()` is the [scalar](../../../../glossary.md#term-scalar) forward [occurrence](../../../../glossary.md#term-occurrence). `gradients().get(i)` is the first [derivative](../../../../glossary.md#term-derivative) for
exact requested `targets.get(i)`. The list and values are [immutable](../../../../glossary.md#term-immutable), resource-free, and readable
after [Engine](../../../../glossary.md#term-engine) and caller storage close. A distinct [carrier](../../../../glossary.md#term-storage-carrier) is required because a combined list
would give position zero a different role; a richer request/[publication](../../../../glossary.md#term-publication) hierarchy adds no current
need. Ordinary [signatures](../../../../glossary.md#term-signature) continue to expose only Engine, [Model](../../../../glossary.md#term-model), and [JDK](../../../../glossary.md#term-java-development-kit-jdk) types.

## Detailed semantics and invariants

### [Validation](../../../../glossary.md#term-validation) and [Compiler](../../../../glossary.md#term-compiler) [lowering](../../../../glossary.md#lowering)

[Engine](../../../../glossary.md#term-engine) [admission](../../../../glossary.md#term-admission) precedes [argument](../../../../glossary.md#term-argument) inspection. Then validate: non-null [`objective`](../../../../glossary.md#term-objective); non-null,
non-empty, exact-object-[identity](../../../../glossary.md#term-identity-meanings)-unique [`targets`](../../../../glossary.md#term-target) with indexed null/duplicate diagnostics;
and non-negative `maximumTotalBytes` with `maximumTotalBytes must be non-negative: <value>`.
[Snapshot](../../../../glossary.md#term-snapshot) the target list before inward work.

Before or around [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), reuse Engine 0005A's identity-safe iterative traversal of the
objective [Tensor](../../../../glossary.md#tensor) expression closure to inventory exact [provenance](../../../../glossary.md#provenance)-free [leaf](../../../../glossary.md#term-leaf) Tensors. After
compilation, select only leaves matching final `CompiledGraph.inputs()` entries and preserve that
authoritative order. [Parameters](../../../../glossary.md#term-parameter) and [buffers](../../../../glossary.md#term-buffer) participate as ordinary reachable leaves. Do not
accept an explicit input list, traverse Compiler [IR](../../../../glossary.md#term-intermediate-representation-ir), infer [liveness](../../../../glossary.md#term-liveness), read storage for an
unselected leaf, or retain Tensor/provenance state after the [synchronous](../../../../glossary.md#term-synchronous) call.

Compiler, not Engine, validates that the objective has exact `Shape.scalar()`, floating type, and
`requiresGrad == true`; that every target is floating, [gradient](../../../../glossary.md#term-gradient)-eligible, identity-present in the
complete objective-rooted forward inventory, and differentiably connected; and that every
[operation](../../../../glossary.md#operation)/attribute route is supported. Targets may be leaves, intermediates, or the objective.

Lower exactly once with `FORWARD_AND_BACKWARD`, [singleton](../../../../glossary.md#term-singleton) forward output, standard optimization,
unconstrained intent, neutral scoring, and:

```java
new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
        List.of(new FunctionalGradientRequest.ForwardTensorReference(objective)),
        List.of(Optional.empty()),
        targets,
        false,
        FunctionalGradientRequest.DisconnectedPolicy.ERROR)))
```

The absent [seed](../../../../glossary.md#term-seed) is Compiler's exact positive-one [scalar](../../../../glossary.md#term-scalar) [logical splat](../../../../glossary.md#logical-splat), never a caller input or
Engine Tensor. ERROR is fixed. The existing ordinary
`compile(forwardOutputs, cotangentSeeds, targets)` and advanced one/two-stage request path remain
unchanged; explicit [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), ZERO, multiple outputs, and higher order use those paths.

### Execution, result order, [identity](../../../../glossary.md#term-identity-meanings), and [aliases](../../../../glossary.md#aliasing-and-arena)

Under one [admission](../../../../glossary.md#term-admission): validate; [compile](../../../../glossary.md#compile) once; [prepare](../../../../glossary.md#prepare) once; bind/[run](../../../../glossary.md#term-run) once through [Engine](../../../../glossary.md#term-engine) 0003;
validate publications; [preflight](../../../../glossary.md#term-preflight) all returned bytes; copy [objective](../../../../glossary.md#term-objective) then [gradients](../../../../glossary.md#term-gradient); construct the
[carrier](../../../../glossary.md#term-storage-carrier); close the temporary result/wrappers; then return.

The result has exactly `1 + targets.size()` occurrences. Index zero must be FORWARD with the
objective [`TensorId`](../../../../glossary.md#tensorid) and empty [derivative](../../../../glossary.md#term-derivative)/[target](../../../../glossary.md#term-target) [metadata](../../../../glossary.md#term-metadata). Index `i + 1` must be GRADIENT with
`targets.get(i).id()`, derivative order one, and target index `i`. Every [descriptor](../../../../glossary.md#term-descriptor) is non-null.
Any mismatch is an index/count-specific `IllegalStateException`; Engine neither searches, repairs,
reorders, nor merges roles.

Targets are identity-unique; equal descriptors do not merge identities. Automatically discovered
leaves match final [Compiler](../../../../glossary.md#term-compiler) `BindableInput` identities exactly once and are selected in final
[binding](../../../../glossary.md#term-binding) order. Repeated exact [leaves](../../../../glossary.md#term-leaf) produce one binding; distinct equal-descriptor leaves remain
distinct. Shared caller storage gets distinct borrow wrappers. Compiler may map several target
roles—or a forward and gradient role—to one final value/representation. Each [occurrence](../../../../glossary.md#term-occurrence) is
nevertheless copied independently into a distinct uncached [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue); no physical-alias
or equality [API](../../../../glossary.md#term-application-programming-interface-api) is added.

### Aggregate-byte accounting, [ownership](../../../../glossary.md#term-ownership-meanings), and limits

Before the first copy, inspect [objective](../../../../glossary.md#term-objective) then all [gradient](../../../../glossary.md#term-gradient) [descriptors](../../../../glossary.md#term-descriptor): require fully static [Shape](../../../../glossary.md#shape)
and resolved [layout](../../../../glossary.md#layout); compute checked zero-before-product [element](../../../../glossary.md#term-element) count and checked [canonical](../../../../glossary.md#term-canonical-form) byte
count; reject any value above `Integer.MAX_VALUE`; and add counts with `Math.addExact`. Compare the
complete sum with the limit and fail with:

```text
total canonical byte count exceeds maximumTotalBytes: required=<sum>, maximum=<limit>
```

The bound includes objective and all gradient payloads only—not inputs, recipes, [Runtime](../../../../glossary.md#term-runtime)
[buffers](../../../../glossary.md#term-buffer)/workspaces, object overhead, defensive copies, or peak memory. Pass each exact count as
the tight per-copy limit. Validate returned value [metadata](../../../../glossary.md#term-metadata)/counts against [preflight](../../../../glossary.md#term-preflight). No partial
return, [cache](../../../../glossary.md#term-cache), deduplication, streaming, mapping, typed-array conversion, caller destination, or
[Tensor](../../../../glossary.md#tensor) reconstruction is allowed.

Automatically selected inputs use the existing [TensorId](../../../../glossary.md#tensorid)/descriptor checks and per-Tensor
synchronized `hostStorage()` [snapshot](../../../../glossary.md#term-snapshot) in final [binding](../../../../glossary.md#term-binding) order. Storage and [arenas](../../../../glossary.md#term-arena) remain
caller-owned and must stay live, accessible, and free from conflicting mutation through
[synchronous](../../../../glossary.md#term-synchronous) cleanup. Snapshots are not [atomic](../../../../glossary.md#resource-transaction-and-rollback) across selected [leaves](../../../../glossary.md#term-leaf), do not pin memory, and do
not synchronize [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) writes.

### Failure, cleanup, close, and [concurrency](../../../../glossary.md#term-concurrency)

Failure precedence is closed-[Engine](../../../../glossary.md#term-engine) [admission](../../../../glossary.md#term-admission); structural [arguments](../../../../glossary.md#term-argument); [Compiler](../../../../glossary.md#term-compiler); [Prepare](../../../../glossary.md#prepare); logical
[binding](../../../../glossary.md#term-binding)/[run](../../../../glossary.md#term-run); [publication](../../../../glossary.md#term-publication) and aggregate [preflight](../../../../glossary.md#term-preflight); then copies in result order. Preserve original
unchecked failure/Error [identity](../../../../glossary.md#term-identity-meanings). Before result construction, use Engine 0003 reverse [rollback](../../../../glossary.md#term-rollback).
Afterward, close the temporary result and wrappers exactly once. A distinct cleanup failure is
suppressed on the primary in encounter order; never self-suppress. Without an earlier failure,
cleanup failure becomes primary. No partial [carrier](../../../../glossary.md#term-storage-carrier) is returned.

Reuse the existing [lifecycle](../../../../glossary.md#lifecycle) gate and ungated helpers; do not nest public lifecycle calls or add a
gate, [registry](../../../../glossary.md#term-registry), [callback](../../../../glossary.md#term-callback) executor, or service. The temporary result is never registered or
returned. Engine close waits for an admitted call through copies and cleanup. A losing call fails
before arguments; a winning call may return [detached](../../../../glossary.md#detached-model-results) values after closure begins. Concurrent
calls have isolated [Runtime](../../../../glossary.md#term-runtime) states. No async, [cancellation](../../../../glossary.md#term-cancellation), retry, or fallback is added.

The intended support remains [CPU](../../../../glossary.md#term-cpu)-only: exactly one non-empty maximal CPU [partition](../../../../glossary.md#partition), supported
kinds and types, fully static [Shapes](../../../../glossary.md#shape), and resolved compatible [layouts](../../../../glossary.md#layout). [Compile](../../../../glossary.md#compile) success does not
guarantee prepare/run success. The Engine-owned real [fixture](../../../../glossary.md#term-fixture) will use one resolved [scalar](../../../../glossary.md#term-scalar) [FLOAT32](../../../../glossary.md#term-float32)
[gradient](../../../../glossary.md#term-gradient)-eligible [leaf](../../../../glossary.md#term-leaf) and its supported
`contiguous()` [objective](../../../../glossary.md#term-objective), discover the leaf automatically, and verify the scalar value and typed
positive-one gradient without claiming broader CPU gradient coverage. Current contracts now make
that non-empty CPU [graph](../../../../glossary.md#graph) plus source-only [seed](../../../../glossary.md#term-seed) publication preparable and materializable; the
fixture remains Engine-owned implementation evidence rather than prerequisite-task evidence.

## [Scope](../../../../glossary.md#term-scope)

- Add the exact method and result type, fixed Compiler mapping, one-admission orchestration,
  focused tests, Javadocs, API/glossary updates, and synchronized planning evidence.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [admission glossary](../../../../glossary.md#term-admission); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Treat forward [objective](../../../../glossary.md#term-objective) and [target](../../../../glossary.md#term-target)-aligned [gradients](../../../../glossary.md#term-gradient) as one cohesive [detached](../../../../glossary.md#detached-model-results) result.

## Out of [scope](../../../../glossary.md#term-scope)

- [Tensor](../../../../glossary.md#tensor) execution/backward/[gradient](../../../../glossary.md#term-gradient) state; inferred/default-all [targets](../../../../glossary.md#term-target); no-[argument](../../../../glossary.md#term-argument)
  `withBackward()`; non-scalar objectives; explicit-seed or multi-output convenience; ZERO;
  vector-Jacobian/Jacobian materialization; second order; <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [seed glossary](../../../../glossary.md#term-seed); [vector glossary](../../../../glossary.md#term-vector); [Jacobian glossary](../../../../glossary.md#term-jacobian); [materialization glossary](../../../../glossary.md#materialization))</span>`createGraph`; `TRAINING_STEP`;
  optimizer/session behavior.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- New request/options/builders, default/unlimited/per-value limits, caches/reuse, tuning,
  persistence, typed arrays, cross-backend transfer, mixed schedules, or another backend.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [backend glossary](../../../../glossary.md#backend))</span>
- Changes to Model, Compiler, Config, Planning, Prepare, Runtime, CPU, NN, Training, build files,
  dependencies, architecture contracts/pages, ADRs, architecture-test source, conformance tests,
  generated code, or Engine 0007–0008 specifications.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [CPU glossary](../../../../glossary.md#term-cpu); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Runtime/Prepare/backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0011](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>, [roadmap](../../../roadmap.md), and
  [Engine master plan](../master-plan.md)
- [Engine](../../../../glossary.md#term-engine) [0003](0003-typed-logical-input-binding-and-published-result-access.md),
  [0004](0004-explicit-host-materialization-boundary.md), and
  [0005](0005-one-shot-forward-convenience.md)
- [Compiler](../../../../glossary.md#term-compiler) [0006](../../compiler/tasks/0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md),
  [0006B4](../../compiler/tasks/0006b4-stable-caller-input-tensor-identity-bindings.md), and
  [0006B5](../../compiler/tasks/0006b5-published-compile-time-constant-descriptor-closure.md)
- [Runtime](../../../../glossary.md#term-runtime) [0015](../../runtime/tasks/0015-leased-publication-representation-access.md) and [CPU](../../../../glossary.md#term-cpu)
  [0010G](../../../backends/cpu/tasks/0010g-canonical-caller-owned-host-snapshot-export.md)
- [Prepare](../../../../glossary.md#prepare) [0005](../../prepare/tasks/0005-producerless-published-constant-resource-handoff-and-shared-slot-assignment.md)
  and CPU
  [0010H](../../../backends/cpu/tasks/0010h-source-only-published-constant-cpu-materialization.md)

## Architecture constraints

[Engine](../../../../glossary.md#term-engine) owns outward orchestration; [Compiler](../../../../glossary.md#term-compiler) exclusively owns [gradient](../../../../glossary.md#term-gradient) semantics; [Model](../../../../glossary.md#term-model) owns
[immutable](../../../../glossary.md#term-immutable) [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-identity-meanings)/[provenance](../../../../glossary.md#provenance); [Runtime](../../../../glossary.md#term-runtime) owns isolated [run state](../../../../glossary.md#run-state--runstate); [CPU](../../../../glossary.md#term-cpu) owns physical copying.
No [owner](../../../../glossary.md#term-owner), [dependency](../../../../glossary.md#term-dependency), inward public contract, or architecture rule changes. Engine imports no CPU
internal type. It may reuse 0005A's transient traversal of immutable Tensor expression provenance,
but it does not traverse Compiler [IR](../../../../glossary.md#term-intermediate-representation-ir), infer [graph](../../../../glossary.md#graph) [liveness](../../../../glossary.md#term-liveness), or implement [derivative](../../../../glossary.md#term-derivative), [seed](../../../../glossary.md#term-seed), or
disconnected policy. Stop and replan if an inward [API](../../../../glossary.md#term-application-programming-interface-api), another policy/output mode, dependency,
[architecture test](../../../../glossary.md#term-architecture-test), or second production [module](../../../../glossary.md#term-module) must change.

## Package impact

The existing `io.github.pho001.synaptik.engine` [facade](../../../../glossary.md#term-facade) gains the result beside
[`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue); [`Engine`](../../../../glossary.md#term-engine) owns the public call and `AdvancedEngine` remains the private ordinary
orchestration/gate [owner](../../../../glossary.md#term-owner) with no advanced public addition. No package/[module](../../../../glossary.md#term-module) is added.

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

Review without editing [`RunResult`](../../../../glossary.md#run-result--runresult), [`HostTensorValue`](../../../../glossary.md#host-tensor-value--hosttensorvalue), ordinary handles, all advanced public
types, [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/[CPU](../../../../glossary.md#term-cpu) contracts, [Tensor](../../../../glossary.md#tensor) [API](../../../../glossary.md#term-application-programming-interface-api), architecture/ADRs, builds,
architecture tests, [conformance](../../../../glossary.md#term-conformance) tests, other [modules](../../../../glossary.md#term-module), and [Engine](../../../../glossary.md#term-engine) 0007–0008; record reasoned
no-change conclusions.

## Maximum [scope](../../../../glossary.md#term-scope)

At most those sixteen paths. Integration changes are justified by new end-to-end behavior.
Architecture/[conformance](../../../../glossary.md#term-conformance) source is excluded because no [dependency](../../../../glossary.md#term-dependency)/shared-[backend](../../../../glossary.md#backend) boundary changes;
[run](../../../../glossary.md#term-run) the existing focused [architecture test](../../../../glossary.md#term-architecture-test) unchanged. Stop and replan before any seventeenth path,
inward contract, dependency/build edit, or other-[module](../../../../glossary.md#term-module) executable change.

## Acceptance criteria

- Exact public shape/import boundary, constructor visibility, immutability, and target-aligned
  result semantics are automated from a distinct package.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [target glossary](../../../../glossary.md#term-target))</span>
- Validation order and fixed one-stage absent-seed/ERROR mapping match this specification;
  Compiler alone owns semantic preflight.<span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [Compiler glossary](../../../../glossary.md#term-compiler); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- Explicit-[seed](../../../../glossary.md#term-seed) ordinary and full advanced paths remain unchanged and passing.
- Automatic leaf inventory and authoritative compiled-input selection exactly reuse 0005A;
  selected-input storage/borrow/run behavior exactly reuses 0003.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf); [run glossary](../../../../glossary.md#term-run))</span>
- Publications are exactly objective then derivative-order-one targets with defended identities,
  indices, descriptors, and alias multiplicity.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [derivative glossary](../../../../glossary.md#term-derivative); [descriptors glossary](../../../../glossary.md#term-descriptor); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Complete static/resolved/individual/aggregate preflight precedes copying; exact-limit,
  overflow/JVM ceiling, alias, and zero-copy-on-preflight-failure cases are covered.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
- Failures return no partial [carrier](../../../../glossary.md#term-storage-carrier) and preserve cleanup/suppression [identity](../../../../glossary.md#term-identity-meanings).
- Bounded-[latch](../../../../glossary.md#control-flow-graph) tests cover close/[admission](../../../../glossary.md#term-admission)/concurrent isolated runs without sleeps.
- With Compiler 0006B5, Prepare 0005, CPU 0010H, and Engine 0005A Complete, real supported scalar
  CPU integration verifies
  objective, positive-one gradient, byte bound, automatically selected input identity,
  caller ownership, and detached post-close access. Engine owns this final end-to-end test; the
  prerequisite tasks own their focused descriptor, handoff/assignment, and CPU materialization
  tests respectively.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [scalar glossary](../../../../glossary.md#term-scalar); [gradient glossary](../../../../glossary.md#term-gradient); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [detached glossary](../../../../glossary.md#detached-model-results); [handoff glossary](../../../../glossary.md#term-handoff); [materialization glossary](../../../../glossary.md#materialization))</span>
- Javadocs and Public/Compile/Runtime/Training/glossary docs explain exact semantics, limitations,
  lower-level alternatives, and no Tensor mutation/training implication.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Only allowlisted paths change; 0005 remains Complete; task 0006 reaches Complete only through
  the normal implementation statuses; 0007–0008 remain Draft and unspecified.

## Tests / [validation](../../../../glossary.md#term-validation)

Implementation task-tier commands:

```bash
./gradlew :modules:engine:test
./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.EngineTypedLifecycleIntegrationTest
./gradlew :testing:architecture-tests:test --tests io.github.pho001.synaptik.testing.architecture.EngineCompositionContractTest
```

Do not rerun unchanged [Compiler](../../../../glossary.md#term-compiler)/[Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[CPU](../../../../glossary.md#term-cpu) or [conformance](../../../../glossary.md#term-conformance) suites. Root tests remain [Engine](../../../../glossary.md#term-engine)
0008/[CI](../../../../glossary.md#term-continuous-integration-ci). The clean documentation pass reuses Java evidence unless executable behavior changes and
runs:

```bash
./gradlew :modules:engine:javadoc
git diff --check
```

It also checks generated pages, exact public inventories, a distinct-package [fixture](../../../../glossary.md#term-fixture), Markdown
[targets](../../../../glossary.md#term-target)/anchors/unique headings/fences/trailing whitespace/LF/exact final newline, terminology,
the sixteen-path [scope](../../../../glossary.md#term-scope), forbidden changes, preservation [hashes](../../../../glossary.md#term-hash), and status/link agreement. Do not
[run](../../../../glossary.md#term-run) Java tests for this [planning](../../../../glossary.md#term-planning)-only change.

## Dependencies

[Engine](../../../../glossary.md#term-engine) 0001–0005A, [Compiler](../../../../glossary.md#term-compiler) 0006/0006B4–0006B5, [Prepare](../../../../glossary.md#prepare) 0005, [Runtime](../../../../glossary.md#term-runtime) 0015, and [CPU](../../../../glossary.md#term-cpu) 0010G–0010H are
Complete. Engine 0005A already supplies automatic reachable-[leaf](../../../../glossary.md#term-leaf) discovery plus selection in
`CompiledGraph.inputs()` order. Compiler 0006B5 supplies the resolved [canonical](../../../../glossary.md#term-canonical-form) [descriptor](../../../../glossary.md#term-descriptor),
Prepare 0005 supplies complete source-only resource [handoff](../../../../glossary.md#term-handoff) and shared assignment, and CPU 0010H
supplies physical declaration plus fresh per-[run](../../../../glossary.md#term-run) initialized [materialization](../../../../glossary.md#materialization). Current [Model](../../../../glossary.md#term-model)
[scalar](../../../../glossary.md#term-scalar)/type/[Tensor](../../../../glossary.md#tensor)/storage contracts and Runtime [lifecycle](../../../../glossary.md#lifecycle) contracts are sufficient; no
Runtime task, architecture decision, or material outward [API](../../../../glossary.md#term-application-programming-interface-api) choice remains.

## Documentation-focused clean-context pass

After Java/tests stabilize, provide a distinct clean documentation context this task, final diff,
test evidence, dirty baseline/[hashes](../../../../glossary.md#term-hash), fixed request mapping, order/[alias](../../../../glossary.md#aliasing-and-arena)/byte/[ownership](../../../../glossary.md#term-ownership-meanings)/cleanup/
[concurrency](../../../../glossary.md#term-concurrency) decisions, relevant completed contracts, and five authorized explanatory paths. Apply
[Planning](../../../../glossary.md#term-planning), [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), General, and Example profiles. The complete example uses
`Engine.standard()`, a resolved [scalar](../../../../glossary.md#term-scalar) [FLOAT32](../../../../glossary.md#term-float32) [leaf](../../../../glossary.md#term-leaf) and [gradient](../../../../glossary.md#term-gradient)-eligible `contiguous()` [objective](../../../../glossary.md#term-objective),
an explicit [target](../../../../glossary.md#term-target) list, aggregate limit, and caller [arena](../../../../glossary.md#term-arena); it verifies objective and
positive-one gradient bytes and post-close access. Explain fresh [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names)/preparation and route
explicit [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings)/reuse/full policies to existing lower-level APIs. Record reasoned no-change
conclusions for [Tensor](../../../../glossary.md#tensor) API, architecture/ADRs, inward docs, builds/tests, and other [modules](../../../../glossary.md#term-module).

## Follow-up tasks

- [Engine](../../../../glossary.md#term-engine) 0007 and 0008 remain Draft without specifications.
- Non-scalar, ZERO, higher-order, optimizer/training, reuse, and cross-backend conveniences require
  separately demonstrated needs; create none here.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [optimizer glossary](../../../../glossary.md#term-optimizer); [backend glossary](../../../../glossary.md#backend))</span>

## Architecture impact

Expected impact: None. This maps an [Engine](../../../../glossary.md#term-engine) call to established [owner](../../../../glossary.md#term-owner) contracts. Stop on contrary
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
  mutable or no-argument meaning.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [targets glossary](../../../../glossary.md#term-target); [mutable glossary](../../../../glossary.md#term-mutable); [argument glossary](../../../../glossary.md#term-argument))</span>
- Return objective plus gradients in a role-separating carrier; avoiding a second forward call is
  useful, and all bytes share the aggregate bound.<span data-glossary-navigation="true"> ([gradients glossary](../../../../glossary.md#term-gradient); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Fix [Compiler](../../../../glossary.md#term-compiler)'s absent unit [seed](../../../../glossary.md#term-seed) and ERROR policy; other choices already have lower-level APIs.
- Validate structure in [Engine](../../../../glossary.md#term-engine) and semantics in Compiler; [preflight](../../../../glossary.md#term-preflight) all results before copying.
- Reuse the current gate and temporary result cleanup; add no [resource owner](../../../../glossary.md#ownership-borrowing-and-lease) or [registry](../../../../glossary.md#term-registry).

## Known limitations

Exact [scalar](../../../../glossary.md#term-scalar) floating [objective](../../../../glossary.md#term-objective), explicit unique connected [targets](../../../../glossary.md#term-target), fixed ERROR, [CPU](../../../../glossary.md#term-cpu)-only
static/resolved supported execution, fresh [compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare), [eager](../../../../glossary.md#term-eager-evaluation) complete [materialization](../../../../glossary.md#materialization),
`Integer.MAX_VALUE` per value, non-[atomic](../../../../glossary.md#resource-transaction-and-rollback) selected-input association snapshots, and no training/
[optimizer](../../../../glossary.md#term-optimizer)/[checkpoint](../../../../glossary.md#term-checkpoint)/transfer meaning. [Engine](../../../../glossary.md#term-engine) 0005A's automatic reachable-[leaf](../../../../glossary.md#term-leaf) discovery plus
selection in `CompiledGraph.inputs()` order and the complete source-only [constant](../../../../glossary.md#term-constant) chain are
already available.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) context: `01a0a557-092d-7cb0-baa4-d6ebb3527336`.

- Read the required architecture/planning/task lineage, applicable API/glossary/docs profiles,
  current source/tests, and starting dirty diff/status.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Confirmed Compiler's exact absent-seed, target, ERROR, publication ordering/alias contracts and
  Engine 0003–0005's sufficient outward binding, lifecycle, aggregate-copy, and cleanup seams.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [seed glossary](../../../../glossary.md#term-seed); [target glossary](../../../../glossary.md#term-target); [publication glossary](../../../../glossary.md#term-publication); [alias glossary](../../../../glossary.md#aliasing-and-arena); [binding glossary](../../../../glossary.md#term-binding); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Follow-up diagnosis context `01a0a570-06d4-7012-814b-ca71af8676e7` disproved inward readiness:
  the absent seed is a source-only published constant with unresolved layout, Prepare contributes
  and assigns only partition-analysis resources, CPU receives no assignment, and Engine preflight
  would reject unresolved layout even with a fabricated assignment.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [layout glossary](../../../../glossary.md#layout); [Prepare glossary](../../../../glossary.md#prepare); [partition glossary](../../../../glossary.md#partition); [CPU glossary](../../../../glossary.md#term-cpu); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- Reassessment at main [commit](../../../../glossary.md#term-commit) `f2dfecbd653766eef19fa76306744067d4cdbd08` confirmed Complete
  Compiler 0006B5 closes the canonical logical descriptor, Complete Prepare 0005 validates and
  assigns the source-only resource, and Complete CPU 0010H contributes its geometry and builds a
  fresh initialized, initially valid representation for each Runtime run state.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [descriptor glossary](../../../../glossary.md#term-descriptor); [Runtime glossary](../../../../glossary.md#term-runtime); [run state glossary](../../../../glossary.md#run-state--runstate))</span>
- Confirmed current Engine 0005A leaf discovery/final-binding selection, publication metadata,
  host-copy preflight, and cleanup helpers support the proposed Engine-only implementation without
  a new inward API. Runtime's existing initialized-representation, ordered-publication, lease,
  isolated-state, and cleanup contracts require no new task.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf); [metadata glossary](../../../../glossary.md#term-metadata); [host glossary](../../../../glossary.md#term-host); [lease glossary](../../../../glossary.md#term-lease))</span>
- This clean planning/documentation reassessment changed exactly this task, the Engine master plan,
  and the roadmap. `git diff --check`, exact three-path scope, Markdown relative-link/anchor,
  heading/fence/whitespace/LF/final-newline, stale CPU-0010H wording, and cross-file
  status/order/dependency checks all passed.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Preserved baseline copies/diffs for the pre-edited [Engine](../../../../glossary.md#term-engine) master plan and roadmap.
- Java tests/[Javadoc](../../../../glossary.md#term-javadoc) were not [run](../../../../glossary.md#term-run) because this assignment is planning-only.

Implementation context: `01a0b11d-cc89-7590-8836-0555b05e7001`.

- Implemented the exact ordinary `Engine.backward(objective, targets, maximumTotalBytes)` surface,
  package-private fixed request mapping, detached <span data-glossary-navigation="true"> ([detached glossary](../../../../glossary.md#detached-model-results))</span>`ScalarObjectiveBackwardResult`, publication and
  byte preflight, alias-copy, cleanup, close, and concurrency behavior within the eight Java/test
  paths.<span data-glossary-navigation="true"> ([concurrency glossary](../../../../glossary.md#term-concurrency))</span>
- `./gradlew :modules:engine:test` passed with 41 tests and zero failures, errors, or skips.
- The focused `EngineTypedLifecycleIntegrationTest` passed all 6 tests with zero failures, errors,
  or skips. The focused `EngineCompositionContractTest` completed successfully; its report records
  1 passing test.
- `git diff --check`, `javap`, public-shape and source scans, exact eleven-path implementation
  scope, and preservation of the initial three-path planning diff passed. No validation command
  failed; a pre-final-review test-only latch threshold was corrected before the final tier.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [latch glossary](../../../../glossary.md#control-flow-graph))</span>

Documentation context: `01a0b126-f60b-7a11-8d7c-621e532952f0`.

- Independently reviewed the final implementation and all four changed tests against every
  acceptance criterion. The implementation fixes one absent positive-one scalar seed and
  `DisconnectedPolicy.ERROR`, reuses automatic leaf discovery plus final compiled-input
  selection, validates objective-first then target-ordered publications, preflights the complete
  static/resolved per-value and aggregate byte set before copying, copies aliases independently,
  and returns detached values after one fresh lifecycle and cleanup.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective))</span>
- Finalized Public, Compile, Runtime, and Training API explanations, the existing one-shot
  glossary family, status/evidence in this task, the Engine master plan, and the roadmap. The
  complete public example uses <span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>`Engine.standard()`, caller-owned resolved [scalar](../../../../glossary.md#term-scalar) [`FLOAT32`](../../../../glossary.md#term-float32)
  storage, `contiguous()`, an explicit target list and aggregate bound, verifies objective and
  positive-one gradient bytes, and reads both after Engine/storage closure.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- The four production Javadocs were already accurate and were left byte-for-byte unchanged.
  Their SHA-256 hashes remain the implementation hashes recorded in the handoff, so no executable
  Java token comparison was necessary. All four test hashes likewise remain unchanged.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([SHA-256 glossary](../../../../glossary.md#term-sha-256); [handoff glossary](../../../../glossary.md#term-handoff); [token glossary](../../../../glossary.md#term-token))</span>
- `./gradlew :modules:engine:javadoc`, generated-page inspection, `javap -public`, the
  package-private result-constructor inspection, and a distinct-package public fixture all passed.
  Markdown target/anchor, unique-heading, balanced-fence, trailing-whitespace, LF, final-newline,
  exact sixteen-path, forbidden-change, status-agreement, stale-wording, signature-leak, and<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [signature glossary](../../../../glossary.md#term-signature))</span>
  `git diff --check` validations passed.
- Reused the implementation Java-test evidence as required; no Java test suite was rerun because
  executable Java and tests were unchanged during documentation finalization.

## Implementation notes

- `Engine.backward(...)` admits once, snapshots and validates structural arguments, inventories
  reachable provenance-free leaves, constructs one singleton absent-seed/ERROR Compiler request,
  and binds only final compiled inputs in their authoritative order.<span data-glossary-navigation="true"> ([leaves glossary](../../../../glossary.md#term-leaf))</span><span data-glossary-navigation="true"> ([arguments glossary](../../../../glossary.md#term-argument); [provenance glossary](../../../../glossary.md#provenance); [singleton glossary](../../../../glossary.md#term-singleton); [seed glossary](../../../../glossary.md#term-seed); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- The result boundary requires exactly one objective publication followed by one derivative-order-
  one publication for every target position. Complete byte preflight precedes objective-first and
  target-ordered independent copies; returned values own no inward resource.<span data-glossary-navigation="true"> ([objective glossary](../../../../glossary.md#term-objective); [publication glossary](../../../../glossary.md#term-publication); [derivative glossary](../../../../glossary.md#term-derivative); [target glossary](../../../../glossary.md#term-target); [preflight glossary](../../../../glossary.md#term-preflight))</span>
- Existing ordinary explicit-seed compilation and advanced full-request compilation were not
  widened or redirected. Pure zero-node and mixed/multi-partition compositions remain rejected.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [node glossary](../../../../glossary.md#node); [partition glossary](../../../../glossary.md#partition))</span>
- No inward module, dependency, build, architecture, backend-conformance, Tensor/Model source, or
  Engine 0007–0008 specification changed.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Tensor glossary](../../../../glossary.md#tensor); [Model glossary](../../../../glossary.md#term-model); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Tensor API and Model source required no change because Engine consumes existing immutable Tensor
  identity, descriptor, provenance, and storage contracts without changing them.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [immutable glossary](../../../../glossary.md#term-immutable); [identity glossary](../../../../glossary.md#term-identity-meanings); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Architecture, ADRs, and architecture-test source required no change because ownership,
  dependency direction, and module boundaries are unchanged; the existing focused architecture
  test passed unchanged.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- Compiler, Prepare, Runtime, and CPU contracts and documentation beyond the allowed API pages
  required no change because the completed 0006B5 -> Prepare 0005 -> CPU 0010H chain already
  supplies every inward descriptor, assignment, initialized representation, publication, and copy
  contract used here.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Build files and dependencies required no change because all named types are already available to
  Engine and the integration test. Backend conformance required no change because this task adds
  Engine orchestration, not backend behavior.<span data-glossary-navigation="true"> ([integration test glossary](../../../../glossary.md#term-integration-test))</span>
- Other modules and plans required no change because the capability is confined to Engine; Engine
  0007–0008 remain Draft and unspecified as required.

## Completion summary

Completed changes:
- Added the exact one-shot [scalar](../../../../glossary.md#term-scalar)-[objective](../../../../glossary.md#term-objective) backward [API](../../../../glossary.md#term-application-programming-interface-api) and [detached](../../../../glossary.md#detached-model-results) objective/[gradient](../../../../glossary.md#term-gradient) [carrier](../../../../glossary.md#term-storage-carrier).
- Implemented automatic input selection, fixed Compiler request lowering, publication defense,
  complete aggregate preflight, independent copying, lifecycle cleanup, and concurrency behavior.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [lowering glossary](../../../../glossary.md#lowering); [publication glossary](../../../../glossary.md#term-publication); [preflight glossary](../../../../glossary.md#term-preflight); [lifecycle glossary](../../../../glossary.md#lifecycle); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>

Files changed or created:
- Four Engine production paths, three Engine test paths, and one integration-test path listed in
  the allowlist.
- `docs/api/public-api.md`, `docs/api/compile-api.md`, `docs/api/runtime-api.md`,
  `docs/api/training-api.md`, `docs/glossary.md`, this task, the Engine master plan, and the
  roadmap.

Tests and [validation](../../../../glossary.md#term-validation) performed:
- Reused 41 passing Engine tests, 6 passing focused integration tests, and 1 passing focused
  architecture test from the implementation context.<span data-glossary-navigation="true"> ([architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- The documentation context passed Engine Javadoc, generated-page and public-shape inspection,
  fixture compilation, Markdown/format/scope/status scans, and <span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [shape glossary](../../../../glossary.md#shape); [fixture glossary](../../../../glossary.md#term-fixture); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check`.

Documentation impact:
- Existing production Javadocs were reviewed and remained accurate without edits. Public,
  Compile, Runtime, and Training API pages now document the current call and its boundaries; the
  glossary adds the smallest matching one-shot backward entry; planning status and evidence are
  synchronized.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [Runtime glossary](../../../../glossary.md#term-runtime); [planning glossary](../../../../glossary.md#term-planning))</span>

Unresolved issues:
- None within [Engine](../../../../glossary.md#term-engine) 0006.

Required follow-up:
- None for this task. Engine 0007–0008 remain Draft and unspecified.

Status: Complete
