# Task 0006B4: Stable Caller-Input Tensor Identity Bindings

## Status

Complete

## Goal

Expose one immutable ordered Compiler association from every final caller-bindable graph
`ValueId` to the originating logical caller `TensorId`. This lets Engine validate and order
Tensor/`HostTensorStorage` bindings from final compile artifacts without retraversing Tensor
expressions, guessing from descriptors, using string labels, or treating an undocumented caller
position as identity.

The association extends the existing compile-input source classification:

```text
captured provenance-free Tensor leaf
  -> compile-local TensorId / ValueId source fact
  -> canonicalization and exact optimization remapping
  -> final CompileConstantPlan.BindableInput(TensorId, ValueId)
  -> Engine-owned typed binding after this task is Complete
```

This task changes compile-time metadata only. It retains no live `Tensor`, `TensorProducer`,
`TensorProvenance`, storage, prepared state, Runtime coordinate, backend representation, or
execution state and adds no per-element or Runtime hot-path work.

## Scope

### Public bindable-input entry

Enrich the existing public final `io.github.pho001.synaptik.compiler.CompileConstantPlan` rather
than adding a second graph-input artifact or another `CompileArtifacts` record component. Add
exactly this nested public record and public accessor:

```java
public record BindableInput(TensorId tensorId, ValueId valueId) {}

public List<BindableInput> bindableInputBindings()
```

`BindableInput` validates `tensorId` and then `valueId` with `Objects.requireNonNull`, retains the
exact immutable identity references, and uses ordinary record equality and hashing. It identifies
one final caller-bindable graph input only when carried by the owning `CompileConstantPlan` inside
one validated `CompileArtifacts`; the standalone record does not prove graph membership.

`bindableInputBindings()` returns an immutable membership snapshot in exact final
`graph.inputs()` order after constant entries are omitted. Each binding has a unique `TensorId`
and a unique `ValueId`. Its `ValueId` is the exact corresponding final graph-input identity. The
final descriptor is intentionally not copied: consumers obtain it from that exact value in
`CompileArtifacts.graph().values()`, and `CompileArtifacts` validates the association against the
same final graph.

Preserve the existing public compatibility accessor unchanged in name and erased/generic shape:

```java
public List<ValueId> bindableInputs()
```

It remains an immutable ordered projection of `bindableInputBindings().valueId()` and must equal
the current result exactly. Existing Prepare, CPU, and advanced Engine code continues to consume
that accessor unchanged. `constantSources()` and `ConstantSource` remain unchanged. The
package-private `CompileConstantPlan` constructor changes to accept the typed bindable entries and
constant sources; no public constructor exists or is added.

The exact package-private constructor shape is:

```java
CompileConstantPlan(
        List<CompileConstantPlan.BindableInput> bindableInputBindings,
        List<CompileConstantPlan.ConstantSource> constantSources)
```

It validates top-level arguments in declaration order, then bindable entries and constant sources
in encounter order. Failure paths use `bindableInputBindings[index]` and
`constantSources[index]`; they distinguish a repeated `TensorId`, repeated `ValueId`, and a
constant-source `ValueId` overlap. The existing `ConstantSource` validation remains unchanged.

This is one cohesive source-role plan: bindable inputs and fixed constant sources are the two
disjoint classifications of the final graph-input boundary. A separate artifact would duplicate
graph ownership/order validation or add a ninth `CompileArtifacts` component and source/binary
record-shape incompatibility without supplying another responsibility.

### Capture and transformation sidecar

Extend package-private `CompileTimeConstantGraph` to this exact record shape:

```java
record CompileTimeConstantGraph(
        CompiledGraphModel graph,
        Map<ValueId, CompileTimeConstantGraph.Splat> constants,
        Map<ValueId, TensorId> bindableTensorIds)
```

Retain this exact package-private compatibility constructor for existing graph-only inference and
focused pass fixtures:

```java
CompileTimeConstantGraph(
        CompiledGraphModel graph,
        Map<ValueId, CompileTimeConstantGraph.Splat> constants) {
    this(graph, constants, Map.of());
}
```

The map may be partial for existing graph-only
same-package construction and focused pass tests, but its keys must always be unique graph inputs
without constant facts and its values must be non-null and unique. Every successful complete
`GraphCompiler.compile(...)` path requires the map to cover every final non-constant graph input
exactly; incomplete test-only/internal sidecars must fail before `CompileArtifacts` construction
rather than publish a partial association.

`GraphCapture` records `tensor.id()` exactly when it first assigns a `ValueId` to a reachable
provenance-free leaf that is not present in the merged constant ingress. Capture continues to use
exact Tensor object identity for occurrence coalescing and ingress matching. It projects only the
immutable `TensorId`; neither the leaf Tensor nor its provenance survives capture.

Preserve and remap the sidecar through every current graph rebuild:

- canonicalization, common-subexpression elimination, and exact arithmetic rewriting continue to
  use `CompileTimeConstantGraph.replaceGraphPreservingInputRoles(...)`, which remaps the identity
  sidecar together with constant facts by validated input position;
- constant folding remaps all original bindable input identities, assigns no caller identity to a
  synthetic folded constant input, and retains the current original-input-first order;
- dead-code elimination retains every caller-bindable graph input under its current contract and
  remaps its identity; it may prune only unused constant inputs, including compiler-generated
  constants, without creating or dropping a bindable identity; and
- forward-only and combined first-/second-order compilation use the same capture, sidecar,
  optimization, final-validation, and artifact construction path.

No production path may synthesize a `TensorId` from a `ValueId` or reconstruct identity from
descriptor equality, numeric IDs, operation structure, input position outside the sidecar's
validated remapping contract, labels, storage, or factory history.

### Final plan and artifact invariants

`GraphCompiler` derives `CompileConstantPlan` in one scan of final `graph.inputs()`:

- an input with a final constant fact becomes the unchanged `ConstantSource` and has no
  `BindableInput`;
- every other input must have exactly one final `TensorId` in the sidecar and becomes one
  `BindableInput`; and
- missing, extra, duplicated, constant-overlapping, or non-input identity metadata is an internal
  inconsistency and fails compilation without returning partial artifacts. A missing final
  identity fails with `IllegalStateException` and message
  `missing bindable Tensor identity for <ValueId>`; illegal extra, null, duplicate, or
  constant-overlapping sidecar state fails earlier with the package-private sidecar constructor's
  `NullPointerException` or `IllegalArgumentException`.

`CompileArtifacts` keeps its exact existing eight record components, order, public canonical
constructor, accessors, and record semantics. Its source validation additionally proves:

- `constants.bindableInputBindings().size() == constants.bindableInputs().size()`;
- the two bindable lists align entry-for-entry by exact final `ValueId` value;
- typed bindable entries and constant sources together classify every `graph.inputs()` entry once
  in graph-input order;
- every bindable entry names a graph input and has a unique `TensorId` and `ValueId`; and
- the descriptor used later by Engine is the `GraphValue.descriptor()` for that exact final
  `ValueId`, not a copied or guessed descriptor.

`GraphCompilationPort.compile(...)` keeps its exact class modifiers, private constructor, single
public static eight-parameter method, parameter order, and return type. Its constant-free contract
means every reachable provenance-free leaf, including a caller-provided provenance-free explicit
cotangent seed, is represented by a typed bindable entry. The port performs no new work beyond its
existing single delegation.

### Edge semantics

- Repeated occurrences of the same exact Tensor object resolve through capture's identity map to
  one graph input and one typed binding at the leaf's first encounter position.
- Distinct Tensor objects, including objects with equal descriptors, retain distinct `TensorId`
  and `ValueId` bindings in encounter/final graph-input order.
- A provenance-free explicit cotangent seed is a caller-bindable input and retains its own
  `TensorId`; repeated use of that exact seed still produces one binding. An explicit seed
  expression contributes only its reachable provenance-free caller leaves as bindable inputs.
- A default cotangent seed and every Compiler-generated derivative constant remain explicit
  logical constants, carry no public bindable entry, and require no caller binding.
- Explicit caller constant ingress in the package-private complete entry remains a
  `ConstantSource`, not a bindable entry. `GraphCompilationPort` continues to expose no such
  ingress.
- Current dead-code elimination preserves every captured caller-bindable input even when later
  work no longer consumes it; therefore it remains in final `graph.inputs()`, the source plan, and
  the typed association and still participates in the current Prepare caller-input count/order.
- Constant folding may introduce a synthetic constant graph input but never assigns it a caller
  Tensor identity.
- Forward-only, one-stage, and two-stage compilation preserve the same rules. A caller seed leaf
  introduced only by a derivative stage is still caller-bindable; formula intermediates and
  produced seed expressions are not graph inputs.
- Any failed capture, preflight, inference, transformation, planning, or final artifact validation
  returns no `CompileArtifacts` and therefore exposes no partial association. Existing opaque
  `TensorId` consumption on derivative-construction failures remains unchanged.

## Out of scope

- Runtime, Prepare, CPU, Engine, Config, Model, Planning, Trace, backend, NN, Training, or
  Checkpoint production changes
- production synthesis of a `TensorId` from a `ValueId`; deterministic test-only identities in
  the authorized Prepare compatibility fixture do not relax the capture-owned production rule
- Engine task 0003 implementation or replan, Engine task 0004 specification or implementation,
  host materialization, result access, storage snapshotting, or exception translation
- retaining `Tensor`, producer, provenance, labels, `HostTensorStorage`, or mutable request state
  in a graph, sidecar, source plan, or compile artifact
- a descriptor component in `BindableInput`; descriptor lookup remains through the exact final
  graph value
- string names, descriptor-based matching, positional-only Engine binding, graph retraversal, a
  reverse `ValueId`-to-Tensor conversion, or a public identity registry
- changing graph input order, constant classification, publication order/aliases, compile mode,
  functional request, seed semantics, graph phases, derivative order, optimization rules, pass
  order, partitioning, logical memory, diagnostics, or failure translation
- changing the eight-component `CompileArtifacts` record shape or the public
  `GraphCompilationPort` method shape
- generated code, backend lowering, physical allocation, preparation, execution, or performance
  claims
- Compiler 0006C Conv3d gradient closure, Compiler 0007 algebra, recurrent BPTT, or another
  derivative stage
- architecture, ADR, dependency, Gradle, architecture-test, backend-conformance, or integration-
  test source changes unless implementation proves a contradiction and stops

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md), especially Model Tensor identity,
  Compiler ownership, compile artifacts, compile lifecycle, Engine typed logical binding, and
  forbidden `ValueId`-to-Tensor conversion state
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Engine master plan](../../engine/master-plan.md)
- [Blocked Engine 0003](../../engine/tasks/0003-typed-logical-input-binding-and-published-result-access.md)
- [Compiler 0005 compile artifacts](0005-publication-planning-orchestration-and-compile-artifacts.md)
- [Compiler 0006 functional gradients](0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
- [Compiler 0006B3 integration port](0006b3-public-constant-free-complete-compile-entry.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [General documentation style](../../../../developer-guide/documentation/general-style.md)
- [API and Javadoc style](../../../../developer-guide/documentation/api-and-javadoc-style.md)
- [Planning documentation style](../../../../developer-guide/documentation/planning-style.md)

## Architecture constraints

- `ARCHITECTURE.md` is authoritative and already assigns Tensor-expression traversal,
  graph-local identity, transformations, and compile artifacts to Compiler.
- Engine owns typed logical binding and must consume this Compiler association rather than
  reconstruct it. Runtime and Prepare remain coordinate/representation-level.
- Tensor identity remains Model-owned and immutable. Compiler projects `TensorId` while it still
  has the exact Tensor reference and retains no live Tensor/provenance afterward.
- Compile artifacts remain immutable non-executable recipes without physical or mutable state.
- Compiler keeps its current allowed dependencies and gains no outward dependency.
- Publication contracts, repeated gradient bindings, forward/gradient aliases, and graph output
  de-duplication remain unchanged.
- The change is compile-time metadata and adds no work to a per-element or Runtime hot path.
- If exact implementation requires an architecture rule, Model change, outward dependency,
  ninth `CompileArtifacts` component, Prepare/Runtime/Engine change, or public Tensor retention,
  stop and report the conflict.

## Package impact

Existing package used and changed:

- `io.github.pho001.synaptik.compiler` remains the cohesive Compiler front-end and public
  compile-artifact package.

No package is added and no type moves.

Type placement:

- `io.github.pho001.synaptik.compiler.CompileConstantPlan.BindableInput` — public immutable nested
  entry because `CompileConstantPlan` already owns the complete disjoint bindable/constant source
  classification and final graph-input order.
- `io.github.pho001.synaptik.compiler.CompileTimeConstantGraph` — package-private owner of the
  transformation-preserved capture sidecar beside its existing constant-source sidecar.
- Existing `GraphCapture`, graph transforms, `GraphCompiler`, and `CompileArtifacts` remain the
  implementation and cross-validation owners described above.

A separate top-level input-plan artifact is rejected because it would split one source
classification, duplicate order/coverage invariants, and require a breaking public aggregate
shape without a distinct lifecycle or owner.

## Affected files

Expected Compiler production/Javadoc paths (8):

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileTimeConstantGraph.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCapture.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardConstantFolding.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardDeadCodeElimination.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileConstantPlan.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompiler.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompilationPort.java`

Expected Compiler tests (7):

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/CompileTimeConstantGraphTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCaptureTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardGraphOptimizationTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/CompileConstantPlanTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/CompileArtifactsTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/spi/GraphCompilationPortPublicShapeTest.java`

Expected unchanged-consumer Prepare compatibility fixture (1):

- `modules/prepare/src/test/java/io/github/pho001/synaptik/prepare/GraphPreparationTest.java`

This is the sole sixteenth implementation/test path.

Expected documentation and planning paths (6):

- `docs/api/compile-api.md`
- `docs/api/public-api.md`
- `docs/glossary.md`
- `docs/planning/modules/compiler/tasks/0006b4-stable-caller-input-tensor-identity-bindings.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Review unchanged and record a reasoned conclusion unless evidence requires stopping:
`ValidatedGraph`, `GraphCompilation`, `GraphCanonicalization`,
`ForwardCommonSubexpressionElimination`, `ForwardExactArithmeticRewriting`, publication types,
Model Tensor/TensorId/TensorDescriptor/graph contracts, Prepare source mapping, Runtime caller
input and publication contracts, CPU schedule assembly, Engine integration/lifecycle source,
Compile/Runtime/Training APIs, architecture/focused architecture/ADR documents, architecture and
integration tests, backend conformance, Gradle/dependencies, every other module, Compiler 0006C
and 0007, and Engine 0004.

## Maximum scope

Hard ceiling: 22 paths total — eight Compiler production/Javadoc files, seven Compiler tests, the
one Prepare compatibility test fixture, and six documentation/planning files listed above. The
first three groups are exactly sixteen implementation/test paths.

The ceiling exceeds the normal 12-18 path guardrail because one public source-plan association
must be carried atomically through capture, both input-boundary-changing transforms, shared
remapping used by the other rebuilds, final construction, public cross-validation, the existing
integration port, focused transformation tests, and its API documentation. Splitting any
transformation would permit a successful compile to publish a missing or wrong identity.
The sole extra path corrects an unchanged-consumer reflective fixture whose erased constructor
lookup still succeeds but whose legacy `List<ValueId>` argument no longer matches the typed
constructor contents; it does not expand Prepare production behavior.

No path outside the exact allowlist may change. If another production transformation, consumer,
test owner, build file, architecture document, or twenty-third path is required, stop and update
this task in a clean planning context before implementation continues.

## Acceptance criteria

- `CompileConstantPlan.BindableInput` and `bindableInputBindings()` have exactly the public shape,
  validation, immutability, identity, and order semantics specified above.
- Existing `bindableInputs()` and `constantSources()` public signatures and observable results
  remain compatible; `CompileArtifacts` retains exactly eight components in current order and
  `GraphCompilationPort` retains exactly one public compile method with its current signature.
- Complete compilation exposes exactly one typed binding for every final non-constant
  `graph.inputs()` value, in final graph-input order, and none for a constant source.
- Every typed binding has unique non-null `TensorId` and `ValueId`; its value names the exact final
  graph input from which the descriptor is obtained.
- Capture projects leaf identity once without retaining Tensor/provenance, and canonicalization,
  exact rewriting, constant folding, DCE, CSE, and cleanup preserve or remap it correctly.
- Repeated exact Tensor occurrences collapse to one binding; distinct equal-descriptor Tensors
  remain distinct bindings.
- Forward-only, combined first-order, and combined second-order tests cover ordinary leaves,
  provenance-free explicit cotangent seeds, default/generated constants, explicit internal
  constant ingress, synthetic folded constants, and unused/dead inputs with the exact semantics
  above.
- Failed compilation returns no artifact or partial public association and preserves existing
  failure types/order and Tensor-ID consumption rules.
- Prepare's existing positional caller-input contract remains compatible through unchanged
  `bindableInputs()` and requires no production consumer change. Each reflective
  `CompileConstantPlan` construction in `GraphPreparationTest` supplies one
  `CompileConstantPlan.BindableInput` with a deterministic test `TensorId` for each fixture
  bindable `ValueId`, preserving the existing `ValueId` order and all tested Prepare semantics.
- No production code synthesizes `TensorId` from `ValueId`; the fixture's deterministic test-only
  identities establish valid constructor inputs without modeling production capture.
- No Tensor/provenance/storage/descriptor copy, string identity, public registry, graph
  retraversal, publication change, optimization change, Runtime/Prepare/CPU/Engine production
  change, hot-path work, dependency, Gradle, architecture, or unrelated refactor is present.
- Production and affected constructor/method Javadocs document every parameter, return,
  nullability, ordering, identity, immutability, compatibility, and expected failure condition.
- A separate clean documentation-focused context finalizes Javadocs, Compile/Public API text,
  glossary impact, planning evidence, links, and no-change conclusions after Java stabilizes.
- Exact path scope, public/package-private shape, statuses/dependencies, links/anchors/headings,
  fences, whitespace, final newlines, and `git diff --check` pass before completion.

## Tests / validation

Implementation-focused Compiler tests:

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.CompileTimeConstantGraphTest \
  --tests io.github.pho001.synaptik.compiler.GraphCaptureTest \
  --tests io.github.pho001.synaptik.compiler.ForwardGraphOptimizationTest \
  --tests io.github.pho001.synaptik.compiler.CompileConstantPlanTest \
  --tests io.github.pho001.synaptik.compiler.CompileArtifactsTest \
  --tests io.github.pho001.synaptik.compiler.GraphCompilerTest \
  --tests io.github.pho001.synaptik.compiler.spi.GraphCompilationPortPublicShapeTest
```

After executable Java stabilizes, run the Compiler module once and the focused unchanged consumer
compatibility suites:

```bash
./gradlew :modules:compiler:test
./gradlew :modules:prepare:test \
  --tests io.github.pho001.synaptik.prepare.GraphPreparationTest \
  --tests io.github.pho001.synaptik.prepare.GraphPreparationPublicShapeTest
./gradlew :modules:engine:test \
  --tests io.github.pho001.synaptik.engine.AdvancedEngineLifecycleTest
```

Because this changes a public compile artifact consumed across the compile/prepare/Engine
boundary, close it with one repository/architecture checkpoint:

```bash
./gradlew test :testing:architecture-tests:test
```

After correcting `GraphPreparationTest`, rerun the focused Prepare compatibility command above
and then rerun the repository/architecture checkpoint, even though the earlier Compiler and
Engine results may be reused when no executable source in those modules changed afterward.

The documentation-focused pass reuses successful Java evidence unless it changes executable Java
behavior. After final Javadocs and documentation:

```bash
./gradlew :modules:model:javadoc :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/api/compile-api.md \
  docs/api/public-api.md \
  docs/glossary.md \
  docs/planning/modules/compiler/master-plan.md \
  docs/planning/modules/compiler/tasks/0006b4-stable-caller-input-tensor-identity-bindings.md \
  docs/planning/roadmap.md
git diff --check
```

If the Markdown validator is absent, create an equivalent temporary validator outside the
repository. It must validate local targets and heading anchors, unique headings, balanced fences,
trailing whitespace, and exactly one final newline.

Required manual/source evidence:

- `javap -public` and reflection confirm the nested `BindableInput` record, the added accessor,
  unchanged legacy accessors, exact eight-component `CompileArtifacts`, and unchanged one-method
  `GraphCompilationPort`;
- source/bytecode inspection confirms the port still supplies empty ingress and delegates once;
- source and tests prove no live Tensor/provenance/storage retention after capture and complete
  identity coverage through every final compilation mode and transform;
- current Prepare production consumes the unchanged `bindableInputs()` order; the authorized
  reflective fixture constructs deterministic test-only `BindableInput` entries in the same
  fixture `ValueId` order and preserves its Prepare assertions;
- no Runtime, Prepare production, CPU, Engine, Model, other-module, architecture, dependency, Gradle,
  conformance, integration-test source, generated-code, or unrelated planning change;
- exact 22-path allowlist and ceiling, comprising exactly sixteen implementation/test paths and
  six documentation/planning paths; Compiler 0006B4 `Complete` only after all gates, Engine 0003
  still `Blocked` pending its clean replan, Compiler 0006C/0007 and Engine 0004+ statuses unchanged;
  and
- final documentation profile, links, anchors, headings, fences, terminology, newlines,
  whitespace, and diff review.

## Dependencies

- Compiler 0005 immutable publication/source plans and complete artifact validation — Complete.
- Compiler 0006 functional first-/second-order requests, explicit/default seeds, combined capture,
  transformation remapping, and derivative metadata — Complete.
- Compiler 0006B3 public constant-free complete compile integration port — Complete.
- Model immutable `TensorId`, `TensorDescriptor`, canonical Tensor/provenance, and graph identity
  contracts — Complete.
- Current Prepare source mapping and Engine advanced integration contracts — complete and
  unchanged consumers.
- Blocked Engine 0003 demonstrates the immediate need for this association and remains blocked
  until this task is Complete and a later clean replan succeeds.

Compiler 0006C and 0007 are independent Draft gradient/algebra side branches. This Engine-
unblocking task is inserted after 0006B3 and before those rows, so it is the first unfinished
Compiler task and does not skip an earlier queue item. The roadmap explicitly returns the active
frontier to Compiler for 0006B4 while Engine 0003 is blocked.

## Follow-up tasks

- After this task is `Complete`, replan blocked Engine 0003 in a separate clean planning context
  against the implemented exact association. Engine 0003 must not become `Ready` or begin Java
  implementation merely because this Compiler prerequisite is Complete.
- Engine 0004 remains `Draft` without a detailed specification and must not be created or
  implemented here.
- Compiler 0006C and 0007 remain independent `Draft` side branches without detailed task
  specifications; this task changes neither gradient coverage nor algebra.

## Architecture impact

Expected impact: None.

The architecture already makes Compiler responsible for Tensor-expression traversal,
graph-local identity, transformations, and compile artifacts, and makes Engine responsible for
typed logical binding. Projecting `TensorId` while Compiler owns the exact Tensor reference and
carrying it as immutable compile metadata realizes that existing split. Runtime and Prepare stay
at representation/coordinate level, dependency directions do not change, and the existing
prohibition on a `ValueId`-to-Tensor conversion map is preserved because no Tensor reference or
reverse conversion is retained.

If implementation requires changing an architecture rule or retaining live Tensor/provenance,
stop and report the exact conflict rather than editing architecture files.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are working in /Users/phujka/IdeaProjects/Synaptik on Compiler task 0006B4. Do not commit or
push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/modules/compiler/master-plan.md, and
docs/planning/modules/compiler/tasks/0006b4-stable-caller-input-tensor-identity-bindings.md in
full. Implement the task exactly within its 22-path allowlist. Preserve the eight-component
CompileArtifacts record, GraphCompilationPort signature/delegation, publication contracts,
legacy bindableInputs() behavior, and all Runtime/Prepare/CPU/Engine production source. Update
only the authorized Prepare test fixture outside Compiler and documentation: each reflective
CompileConstantPlan construction must pass deterministic test TensorId/ValueId BindableInput
entries in the existing ValueId order without changing tested Prepare semantics. Never synthesize
TensorId from ValueId in production. Stop on an architecture, package, or scope conflict.

After executable implementation and recorded Java/checkpoint evidence, hand the actual diff and
evidence to a distinct clean documentation-focused agent/thread. That pass must follow
docs/developer-guide/documentation-rules.md, apply the General, API/Javadoc, and Planning
profiles, finalize affected Javadocs, Compile/Public API documentation, glossary impact,
planning evidence, links, and no-change conclusions in the same overall change, and not repeat
successful Java tests unless executable behavior changes or a concrete risk is recorded.

Update the task with local decisions, exact command results/counts, documentation context ID,
implementation notes, completion summary, and final status. After the fixture correction, rerun
the focused Prepare compatibility command and the repository/architecture checkpoint. Keep
Engine 0003 Blocked pending a new clean replan. Do not create Engine 0004, commit, or push.
```

## Local decisions

- `TensorId` is sufficient. Engine receives the caller's Tensor and can compare its immutable
  identity, while the final descriptor belongs to the exact final graph value. Retaining a live
  Tensor would preserve mutable host-storage association and provenance unnecessarily and would
  contradict the established post-capture artifact boundary.
- `CompileConstantPlan` is enriched rather than adding a separate artifact because it already
  owns bindable-versus-constant source roles and their final graph-input order. The old
  `bindableInputs()` projection remains for source and binary compatibility of existing consumers;
  the eight-component `CompileArtifacts` record shape is unchanged.
- The internal identity map is keyed by graph-local `ValueId` and contains only captured
  non-constant leaf identities. It is forward-remapped metadata, not a reverse conversion to a
  Tensor and not a second graph representation.
- Existing DCE deliberately retains unused bindable inputs. This task records and preserves that
  behavior rather than silently redefining the public input boundary.
- No production consumer implementation change is needed. The Prepare compatibility test uses
  reflective construction to isolate shared preparation and therefore must supply valid typed
  bindings itself. Deterministic fixture-only `TensorId` values are test data, not a production
  identity derivation rule; the fixture retains its existing bindable `ValueId` order and Prepare
  assertions.

## Known limitations

- `BindableInput` identifies a logical Tensor and final graph value but does not carry a
  descriptor, storage, label, user-facing name, Runtime coordinate, or backend representation.
- `TensorId` uniqueness follows the current Tensor factory identity policy; this artifact does not
  create a global registry or serialization identity.
- Explicit caller constant ingress remains package-private and unavailable through
  `GraphCompilationPort`.
- Engine 0003 remains blocked until its required clean replan fixes the ordinary Engine API. No
  ordinary typed binding or result access exists yet.
- Current Prepare continues to require a caller representation for every final bindable entry,
  including an unused bindable input preserved by DCE.
- `GraphPreparationTest` uses deterministic test-only Tensor identities because it deliberately
  constructs Compiler artifacts without production capture; those identities do not assert a
  numeric or derivable relationship between `TensorId` and `ValueId`.

## Validation evidence

Planning context: `01a0a42f-0a36-7301-8ec2-0c4bf7011c4b`.

Corrective planning context: `01a0a446-763d-71f1-8897-16f8173974c8`.

Planning-only evidence:

- Read the required architecture contract/index, planning guide/roadmap, Compiler and Engine
  master plans, blocked Engine 0003, Compiler 0005/0006/0006B1/0006B2/0006B3 and their directly
  relevant contracts, current capture/artifact/transformation source and focused tests, Model
  Tensor/TensorId/TensorDescriptor/provenance/graph identity contracts, Prepare source mapping,
  Engine integration contracts, and the General, API/Javadoc, and Planning profiles. No
  architecture conflict was found.
- The initial planning pass's `git status --short` recorded only the prior approved Engine
  master/roadmap edits and untracked Engine 0003 task. That pass's final combined
  tracked/untracked scan contained exactly those three preserved paths plus this task and the
  Compiler master plan: five planning paths and no Java, build, architecture, API, glossary,
  generated, or unrelated path.
- `python3 /tmp/validate_synaptik_markdown.py <the five changed planning paths>` passed with
  `validated 5 Markdown files`, covering local targets and anchors, unique headings, balanced
  fences, trailing whitespace, and final newlines.
- Explicit status/dependency scans confirmed Compiler 0006B4 is the `Ready` first-unfinished row,
  Compiler 0006C/0007 remain `Draft`, Engine 0003 remains `Blocked` on 0006B4 and a later clean
  replan, Engine 0001–0002 remain `Complete`, and Engine 0004–0008 remain `Draft`.
- `find docs/planning/modules/engine/tasks -maxdepth 1 -type f -name '0004-*.md'` returned no
  path. Exact five-path comparison and final-newline checks passed.
- `git diff --check` passed. Separate `git diff --no-index --check /dev/null <new-file>` checks
  produced no diagnostics for both untracked task files; exit status one was the expected
  no-index content-difference status.
- Java tests and Javadoc were not run in that initial clean planning pass because it changed
  planning Markdown only.

Implementation evidence:

- The implementation context reported the focused Compiler selection passed 69 tests, the full
  Compiler module passed 251 tests, and the focused advanced Engine compatibility selection
  passed 6 tests.
- The focused Prepare compatibility selection ran 12 tests and failed 8 with
  `ClassCastException`. Read-only inspection localized every failure to the legacy reflective
  `CompileConstantPlan` fixture construction in `GraphPreparationTest`: constructor erasure still
  accepts `List.class`, but the fixture supplies `ValueId` elements where the implemented
  constructor now requires `CompileConstantPlan.BindableInput` elements.
- After the authorized fixture correction, implementation context
  `01a0a43b-7a66-7551-bd64-a578003a89fd` reran the focused Prepare selection successfully: 12
  tests passed. It then ran `./gradlew test :testing:architecture-tests:test` successfully with
  3,078 tests, zero failures/errors, 28 skipped, and 9 architecture tests passed.
- That context's final `git diff --check` passed and its scope audit found exactly the sixteen
  authorized Java paths with no Prepare production change.

Documentation-focused review context: `01a0a450-02f6-78a3-a62d-7495417cb94a`.

Documentation-focused evidence:

- Applied the General, API/Javadoc, and Planning documentation profiles. Reviewed the final eight
  Compiler production files, seven Compiler tests, one Prepare compatibility fixture, Compile
  and public API references, glossary, task/master/roadmap state, and the directly relevant Model,
  Compiler, Prepare, Runtime, and Engine identity/source/publication contracts.
- No executable Java behavior or Java text changed in the documentation context. The eight
  affected production Javadocs already describe parameters, results, failures, ordering,
  identity, immutability, compatibility projection, and the absence of live Tensor/storage state;
  no Javadoc-only refinement was necessary.
- `./gradlew :modules:model:javadoc :modules:compiler:javadoc` passed. Model Javadoc was up to
  date; Compiler Javadoc executed successfully; the build completed with eight actionable tasks,
  one executed and seven up to date.
- `python3 /tmp/validate_synaptik_markdown.py docs/api/compile-api.md docs/api/public-api.md
  docs/glossary.md docs/planning/modules/compiler/master-plan.md
  docs/planning/modules/compiler/tasks/0006b4-stable-caller-input-tensor-identity-bindings.md
  docs/planning/roadmap.md` passed with `validated 6 Markdown files`, covering local targets and
  heading anchors, unique headings, balanced fences, trailing whitespace, and final newlines.
- `javap -public` confirmed public `BindableInput(TensorId, ValueId)`, both typed and compatibility
  accessors, the unchanged eight-component `CompileArtifacts` constructor/accessors, and the
  unchanged single public eight-parameter `GraphCompilationPort.compile(...)` method.
- Final scope inspection found exactly the task's sixteen authorized Java paths and six
  documentation/planning paths, plus only the two pre-existing Engine planning paths that this
  context was required to preserve. No Prepare production, Runtime, CPU, Engine Java, Model,
  Config, Planning, Trace, backend, NN, Training, Checkpoint, Gradle, architecture-test,
  conformance-test, or integration-test source changed.
- No architecture, focused architecture, or ADR edit was needed because the change realizes the
  existing Compiler/Engine ownership split without changing a dependency or lifecycle rule. The
  older illustrative five-component `CompileArtifacts` snippet in `ARCHITECTURE.md` predates the
  already-completed eight-component artifact contract and was left unchanged because this task
  neither changes that record shape nor authorizes an architecture revision.
- Prepare, Runtime, Training, and Engine API documentation required no change: Prepare continues
  consuming the legacy ordered `bindableInputs()` projection, Runtime and Training receive no new
  contract, and Engine 0003 must define its ordinary API only in the mandated later clean replan.
  Compiler 0006C/0007 and Engine 0004+ remain unchanged; no Engine 0004 specification exists.
- `git diff --check` passed. Compiler 0006B4 is `Complete` in its task, Compiler master, and
  roadmap; Engine 0003 remains `Blocked` in the preserved Engine records and roadmap.

## Implementation notes

- Added `CompileConstantPlan.BindableInput(TensorId, ValueId)` and the immutable ordered
  `bindableInputBindings()` view while retaining `bindableInputs()` as its exact ordered
  compatibility projection.
- Captured caller leaf identities without retaining live Tensor/provenance/storage and remapped
  the sidecar through canonicalization, exact rewriting, constant folding, DCE, and CSE.
- Preserved the exact eight-component `CompileArtifacts`, `GraphCompilationPort` shape and single
  delegation, publication semantics, and all Prepare/Runtime/Engine production behavior.
- Updated only the authorized Prepare compatibility fixture outside Compiler production/tests;
  its deterministic Tensor IDs remain test data and preserve the existing positional `ValueId`
  semantics.
- The separate documentation pass independently reviewed the complete Java diff and relevant
  contracts, found the eight production Javadocs accurate without refinement, and finalized the
  Compile API, public API, glossary, task, Compiler master plan, and roadmap.

## Completion summary

- Completed changes: implemented the stable ordered caller `TensorId`-to-final-`ValueId`
  association through capture, transformations, final source-plan construction, public artifact
  validation, and the constant-free integration port.
- Files changed or created: exactly eight Compiler production/Javadoc paths, seven Compiler test
  paths, one Prepare compatibility-test path, and six documentation/planning paths authorized by
  this task.
- Tests and validation: implementation context `01a0a43b-7a66-7551-bd64-a578003a89fd` recorded
  69 focused Compiler tests, 251 full Compiler tests, 6 focused Engine tests, and 12 corrected
  focused Prepare tests passing; its repository/architecture checkpoint recorded 3,078 tests,
  zero failures/errors, 28 skipped, and 9 architecture tests passing. The documentation context
  reused those stable executable results and ran the final documentation gates recorded above.
- Documentation-agent review: clean context `01a0a450-02f6-78a3-a62d-7495417cb94a`
  independently finalized the API, glossary, planning, and Javadoc review.
- Documentation impact: `compile-api.md` and `public-api.md` now explain typed binding order,
  compatibility projection, identity semantics, and the non-executable boundary.
- Javadoc review: all eight authorized production files were reviewed; the implementation
  Javadocs already document the changed parameters, results, failures, ordering, identity,
  immutability, and retention boundaries, so no Java refinement was needed.
- Glossary impact: added the reusable bindable-input-binding term and aligned the bindable input,
  compile artifacts, and compile constant plan entries.
- Unresolved issues: Engine 0003 remains `Blocked` pending its required separate clean replan;
  Engine 0004 was not created. No Compiler 0006B4 issue remains.
- Follow-up required: replan Engine 0003 in a new clean planning context.

Status: Complete
