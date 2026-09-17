# Task 0006B5: Published Compile-Time Constant Descriptor Closure

## Status

Complete

## Goal

Close one general Compiler-owned descriptor-contract gap for fully static, source-only,
published compile-time splat constants. After final graph optimization determines that such a
value remains a graph input and graph output but has no node consumer, give its unresolved
`TensorDescriptor` the canonical contiguous logical layout already implied by its fully static
`Shape`.

## Current defect

The current pipeline can produce a valid source-only publication obligation with no producer and
no consumer. This is particularly visible when functional differentiation creates an absent
scalar unit seed and that seed is itself the requested gradient: capture retains it as a constant
graph input and graph output, optimization retains the publication, and logical-memory planning
truthfully retains a producerless/consumerless graph-output requirement. The final descriptor,
however, still has unresolved layout, so later lifecycle layers cannot derive a complete
materialization contract. This task closes only that Compiler descriptor fact. It is not an
Engine-specific workaround and does not make the value physical or executable.

## Exact semantics and invariants

Run descriptor closure exactly once after `ForwardGraphOptimization.optimize(...)` returns its
final validated graph and before `GraphCompiler` derives final forward/gradient bindings,
publication, backend ownership, partitions, or logical-memory artifacts. This is the safest
placement because final DCE, CSE, and constant folding have already fixed the surviving graph
topology and source roles, while every downstream artifact will observe one consistent closed
descriptor.

A value is eligible if and only if all of these facts hold in the final optimized graph:

1. its `ValueId` occurs in `CompiledGraphModel.inputs()`;
2. the same ID has a `CompileTimeConstantGraph.Splat` fact;
3. the same ID occurs in `CompiledGraphModel.outputs()`;
4. no final `CompiledNode.inputs()` occurrence consumes the ID;
5. its `Shape` is fully static; and
6. its `TensorDescriptor.layout()` is empty.

For each eligible value, replace only its descriptor with a descriptor containing
`LayoutDescriptor.contiguous(shape)`. Preserve the exact `DataType`, exact `Shape`, and
`requiresGrad` value. Current constant validation already requires `requiresGrad == false`; this
task must neither weaken nor duplicate that rule.

Canonical layout treatment is exact for every current `DataType`: `FLOAT64`, `FLOAT32`,
`BFLOAT16`, `INT32`, `INT64`, and `BOOL`. Data type does not alter element-layout geometry.

- `Shape.scalar()` closes to rank zero, no strides, element offset zero, non-view contiguous
  layout, and referenced element span one.
- Any fully static Shape containing a zero-sized dimension closes through the same canonical
  factory. It retains the Shape's rank and canonical strides, uses element offset zero, is not a
  view, and has referenced element span zero.
- Other fully static Shapes receive the factory's canonical row-major element strides, element
  offset zero, non-view status, and checked referenced span.

If canonical stride or span arithmetic overflows for an otherwise eligible value, fail closed at
compile time with a deterministic `IllegalArgumentException` that identifies the `ValueId` and
retains the arithmetic failure as its cause. Do not leave that eligible descriptor unresolved or
silently skip it.

Preserve state unchanged for every non-eligible value, including:

- dynamic or partially dynamic source-only published constants;
- source constants that are consumed by any node, even when also published;
- constants that are not published;
- caller-bindable graph inputs;
- produced values and non-constant values; and
- eligible values whose layout is already resolved.

Rebuilding the immutable graph must preserve exact value encounter order, `ValueId`s, node list
and exact node references, node inputs/outputs, `NodeId`s, input/output order, graph phases,
deferred constraints, constant splat values, bindable Tensor identities, derivative-order values,
forward publication identities/order, gradient publication target/order metadata, aliases, and
publication roles. Derivative metadata must own the rebuilt exact graph reference without
changing its ordered node-to-order mapping.

The closure adds no allocation, byte size, alignment, slot, representation, residency, physical
layout, backend choice, caller binding, node, operation, synthetic topology, or new value. Logical
memory planning must continue to report the source-only publication as graph-output-required with
empty producer and consumer sets.

## Scope

- Add one package-private Compiler transformation for the exact eligibility predicate and
  descriptor replacement above.
- Invoke it in both forward-only and backward-capable compilation after final optimization and
  before final boundary/publication derivation.
- Cover scalar, zero-element, ordinary static, all-supported-type, preservation, overflow, and
  complete-compile integration cases with focused Compiler tests.
- Update affected Compiler Javadoc and Compile API/planning documentation through the required
  clean documentation pass.

## Out of scope

- Physical allocation, byte/alignment declarations, prepared slots, representations,
  initialization recipes, copying, execution, or publication materialization.
- Changes to graph topology, graph inputs/outputs, node construction, operation semantics,
  optimization eligibility/order, constant folding, derivative formulas, seed policy, caller
  binding, publication roles/order, partitioning, or logical-memory ownership.
- Backend selection or any Prepare, Runtime, CPU, Engine, Model, Config, build, architecture,
  ADR, API-facade, conformance, integration, or performance work.
- General default-layout inference for static descriptors. Only the closed final source-only
  published constant role establishes the invariant used here.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Runtime/Prepare/backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [Planning guide](../../../planning-guide.md), [roadmap](../../../roadmap.md), and
  [Compiler master plan](../master-plan.md)
- Compiler [0006](0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
  and [0006B4](0006b4-stable-caller-input-tensor-identity-bindings.md)
- Planning [0005](../../planning/tasks/0005-logical-materialization-and-memory-requirements.md)
  and [0006](../../planning/tasks/0006-planning-contract-closure-audit.md)
- Prepare [0003](../../prepare/tasks/0003-prepare-orchestration-and-validation.md)

## Architecture constraints

Compiler owns final immutable graph descriptors and compile artifacts but not physical buffers.
Planning continues to describe logical producer, consumer, and publication obligations without
inventing execution. Prepare continues to own shared resource handoff and deterministic slot
assignment, concrete backends own physical representation declarations/materialization, Runtime
owns initialized representation validity and publication lifetime, and Engine owns outward
orchestration. No dependency direction, public API, module boundary, or architecture rule changes.

Do not conceal the downstream Prepare/CPU gaps in Compiler. Stop and replan if implementation
requires allocation, backend knowledge, topology synthesis, a public contract, another module, or
an architecture/ADR change.

## Package impact

One package-private transformation is added beside the existing graph transformations in
`io.github.pho001.synaptik.compiler`. No public or exported package shape changes.

## Affected files

Exact future implementation allowlist (eight paths):

1. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/PublishedCompileTimeConstantDescriptorClosure.java` (new)
2. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompiler.java`
3. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/PublishedCompileTimeConstantDescriptorClosureTest.java` (new)
4. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`
5. `docs/api/compile-api.md`
6. This task
7. `docs/planning/modules/compiler/master-plan.md`
8. `docs/planning/roadmap.md`

Review without editing Model descriptor/layout contracts, Planning logical-memory source/tests,
Prepare, Runtime, CPU, Engine, Config, Tensor/Public/Runtime/Training API guides, glossary,
architecture/ADRs, architecture tests, build files, and other modules. If review demonstrates a
required edit outside this allowlist, stop and replan rather than widening the task implicitly.

## Maximum scope

At most those eight paths. The implementation is a small final-graph descriptor closure plus
focused Compiler evidence. It introduces no performance task, benchmark, repository-wide
checkpoint, dependency edit, or generated-code work.

## Acceptance criteria

- The transformation runs after final optimization and before final output, gradient,
  publication, partition, and logical-memory derivation in forward-only and backward-capable
  compilation.
- The eligibility predicate is exactly graph input + constant fact + graph output + no node
  consumer + fully static Shape + unresolved layout.
- Scalars close to canonical rank-zero span-one layout; zero-element static Shapes close to
  canonical span-zero layout; ordinary static Shapes close to canonical row-major layout.
- Each of the six current data types receives identical Shape-derived treatment and preserves its
  exact type and splat value.
- Canonical-layout arithmetic overflow fails closed with value context and retained cause.
- Dynamic, consumed, unpublished, bindable, produced, non-constant, and already-resolved values
  remain exactly unchanged.
- Graph/value/node IDs, orders, topology, phases, constraints, source roles, bindable Tensor
  identities, derivative orders, forward and gradient bindings, aliases, and publication order
  remain unchanged.
- Complete compilation exposes the resolved descriptor consistently in graph, publication,
  constants, diagnostics where applicable, and logical memory; producerless/consumerless
  graph-output truth remains intact and no physical fact appears.
- Public/package inventory is unchanged, only allowlisted paths change, and Compiler 0006B5
  remains Draft until implementation and documentation validation complete.
- The documentation pass reviews all affected Javadocs, Compile API wording, terminology, and
  glossary impact. It records reasoned no-change conclusions for the Tensor/Public/Runtime/
  Training guides, glossary, architecture/ADRs, Planning/Prepare/Runtime/CPU/Engine contracts,
  tests outside Compiler, and build structure.

## Tests and validation

Focused Compiler tests must prove:

- direct transformation of scalar, multiple zero-element Shapes, and ordinary static Shapes;
- all six current data types and exact constant-splat preservation;
- exact preservation of every non-eligible category and already-resolved descriptor identity;
- deterministic overflow failure and cause;
- exact graph topology, IDs, boundary order, phase and derivative-order preservation;
- absent-unit-seed backward compilation where the seed is a source-only gradient publication;
- forward-only source-only published constant compilation; and
- complete artifacts retaining empty producer/consumer sets plus graph-output obligation without
  adding a partition or physical declaration.

Future implementation task-tier validation:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.PublishedCompileTimeConstantDescriptorClosureTest --tests io.github.pho001.synaptik.compiler.GraphCompilerTest
./gradlew :modules:compiler:test
./gradlew :modules:compiler:javadoc
git diff --check
```

The clean documentation pass also validates Markdown links, anchors, heading uniqueness, fences,
trailing whitespace, LF/final newline, exact eight-path scope, package/public inventories, task/
master/roadmap status and ordering, and preservation of excluded files. Do not run unchanged
Planning, Prepare, Runtime, CPU, Engine, architecture, conformance, integration, or root suites;
their behavior belongs to later owner tasks or CI.

Validation tier: affected Compiler module plus affected documentation. No repository-wide or
performance validation is justified by this descriptor-only Compiler change.

## Dependencies

Compiler 0006 and 0006B4, Planning 0005–0006, and the current Model descriptor/layout contracts
are Complete. This task is the first unfinished Compiler task and precedes Draft Compiler 0006C
and 0007 in execution order.

## Handoff to Prepare 0005

Compiler 0006B5 hands Prepare a fully static source-only published constant whose final logical
descriptor is resolved while its Planning requirement correctly remains producerless,
consumerless, and graph-output-required. Draft Prepare 0005 then owns explicit contribution of
that publication resource to the shared handoff and deterministic slot assignment. Compiler does
not prescribe backend selection, physical geometry, representation, initialization, or schedule
assembly.

## Documentation-focused clean-context pass

After Java and focused Compiler tests stabilize, provide a distinct clean documentation context
with this task, the exact diff, test evidence, baseline hashes, source-only eligibility and
placement decisions, and downstream ownership boundary. Apply the General, API/Javadoc, and
Planning profiles. Finalize affected production Javadocs and `docs/api/compile-api.md`; review the
glossary and other API guides and record reasoned no-change conclusions unless an allowlist change
requires replanning. Do not repeat successful Java tests unless documentation changes executable
Java behavior or exposes a concrete failure.

## Follow-up tasks

- Prepare 0005: explicit producerless published-constant resource contribution/handoff and
  deterministic shared slot assignment without backend selection or physical geometry.
- CPU 0010H: materialize such a constant in the current sole non-empty CPU composition through
  existing initialized-buffer/publication machinery; keep zero-node and mixed/multi-partition
  compositions rejected.
- Engine 0006: implement the already selected outward scalar-objective backward convenience only
  after Compiler 0006B5, Prepare 0005, and CPU 0010H are Complete.

Runtime needs no new task: current initialized representations, initial validity, ordered
publications/aliases, leases, and cleanup already provide the required runtime lifecycle.

## Architecture impact

Expected impact: None. This closes a Compiler-owned logical descriptor fact within existing
immutable graph and artifact ownership. Stop rather than edit if implementation evidence
contradicts that conclusion.

## Implementation prompt

```text
Work in /Users/phujka/IdeaProjects/Synaptik on Compiler task 0006B5. Do not use GSD, commit, or
push. Use a separate clean implementation context. Read AGENTS.md, ARCHITECTURE.md, the relevant
architecture lifecycle/module/runtime-prepare-backend docs and ADRs, the planning guide and
roadmap, Compiler master plan and this task, Compiler 0006/0006B4, Planning 0005–0006, Prepare
0003, and directly relevant Model/Compiler source and tests. Inspect and preserve the dirty
worktree.

Implement exactly the eight-path allowlist. Add one package-private descriptor closure after
final optimization and before final bindings/publication/planning. Resolve only fully static,
source-only, published compile-time splat constants with unresolved layout through
LayoutDescriptor.contiguous(shape). Cover scalar, zero-element, ordinary static, every supported
data type, overflow, preservation, metadata, and complete-artifact cases. Preserve topology,
roles, IDs, phases, derivative metadata, aliases, publication order, consumed constants, caller
inputs, and generated derivative semantics. Add no allocation, physical fact, backend choice,
binding, synthetic graph element, performance work, dependency, public API, architecture, or
downstream implementation. Stop on conflict or required ninth path.

Run the focused and full Compiler tests once, then Compiler Javadoc and planning/documentation
gates. Hand the stable diff and evidence to a distinct clean documentation-focused context. Mark
the task Complete only after every code, test, documentation, scope, and status gate passes.
```

## Local decisions

- Final optimized topology, not pre-optimization capture, determines source-only eligibility.
- Publication is represented by final graph-output membership; no Engine-specific request fact
  enters the transformation.
- Existing Model canonical layout construction defines scalar, zero-element, and checked static
  geometry; Compiler does not reproduce that arithmetic.
- An eligible overflow is a malformed unclosable final contract and fails closed with context.
- Downstream materialization gaps remain visible and separately owned.

## Known limitations

Dynamic Shapes remain unresolved. Consumed and unpublished constants retain their current
descriptor behavior. This task alone does not make a source-only constant preparable or
executable; Prepare 0005 and CPU 0010H are required before Engine 0006.

## Validation evidence

Planning diagnosis context: `01a0a570-06d4-7012-814b-ca71af8676e7`.
Implementation context: `01a0a589-5750-7392-b5dc-26890c043a9a`.
Documentation context: `01a0a593-f2a4-7de1-889f-5fb01a65642d`.

- The planning context read the required architecture, planning, Compiler/Planning/Prepare/
  Runtime/CPU/Engine lineage, documentation profiles, directly relevant source/tests, and starting
  dirty diff/status. It verified the absent-seed source-only publication shape; selected the post-
  optimization, pre-boundary placement; preserved Planning's producerless/consumerless graph-
  output obligation; and confirmed that Prepare and CPU own the remaining handoff and physical
  materialization gaps while Runtime needs no new task.
- The implementation context added the package-private descriptor closure and invoked it after
  `ForwardGraphOptimization.optimize(...)` in both forward-only and backward-capable paths. The
  implementation changes exactly four Java paths: two production paths and two Compiler tests.
- The implementation context's focused command
  `./gradlew :modules:compiler:test --tests
  io.github.pho001.synaptik.compiler.PublishedCompileTimeConstantDescriptorClosureTest --tests
  io.github.pho001.synaptik.compiler.GraphCompilerTest` passed 32 tests with zero failures,
  errors, or skips after one test-import correction.
- The implementation context's final `./gradlew :modules:compiler:test` passed 38 suites and 258
  tests with zero failures, errors, or skips. It also passed
  `./gradlew :modules:compiler:javadoc` and `git diff --check`. No executable Java changed after
  that evidence, so the documentation context did not repeat Java tests.
- The documentation context applied the General, API/Javadoc, and Planning profiles. It
  independently reviewed the exact implementation and tests, the final Compiler graph/artifact
  flow, affected Javadocs, Compile API, glossary and adjacent API guides, architecture and ADR
  boundaries, owner plans, build structure, and the intentionally dirty baseline.
- Source inspection confirmed the exact graph-input + splat + graph-output + no-consumer + fully-
  static + unresolved-layout predicate; canonical scalar, zero-element, ordinary-static, and six-
  data-type treatment; deterministic overflow wrapping with `ValueId` and cause; exact no-change
  identity for an unchanged result; and rebuilt graph, constant, constraint, bindable-identity,
  phase, boundary, node-reference, and derivative-order preservation.
- Complete-compile inspection confirmed the closure precedes forward/gradient binding,
  publication, capability queries, partitioning, and logical-memory derivation. Tests retain the
  source-only value as a constant graph input/output with no node consumer or partition and one
  graph-output-required logical-memory row with empty producer and consumer sets.
- Documentation-context `./gradlew :modules:compiler:javadoc` passed after final Javadoc review
  with `BUILD SUCCESSFUL`; all seven actionable tasks were up to date. Generated
  `PublishedCompileTimeConstantDescriptorClosure.html` and `GraphCompiler.html` render the final
  contracts, and the closure method is identified as package-private.
- `python3 /tmp/validate_synaptik_markdown.py` passed for the four changed Markdown paths. It
  validated local links and anchors, balanced fences, LF/final newlines, and trailing whitespace;
  a separate heading scan found no duplicate heading text.
- Source, bytecode, and generated-page inventory checks confirmed one final package-private,
  field-free `PublishedCompileTimeConstantDescriptorClosure`, one private constructor, one
  package-private static `close(ValidatedGraph)` method, and no new public Compiler type or method.
  The two call sites are immediately outside `ForwardGraphOptimization.optimize(...)` and precede
  final boundary derivation in both compile branches.
- Scope checks found exactly the eight task paths in the overall Compiler 0006B5 change. This
  documentation pass edited only Compile API, this task, Compiler master plan, and roadmap; it
  reviewed the two allowlisted Java Javadocs without changing them and did not touch either test.
  Pre-existing dirty paths outside the allowlist remain excluded and preserved.
- Status/order checks found Compiler Complete through 0006B5 in both master plan and roadmap,
  followed by Draft Prepare 0005, Draft CPU 0010H, and Draft Engine 0006. Prepare 0005 remains a
  master-plan placeholder with no detailed task file. The Engine 0006 dependency chain is
  preserved.
- Final `git diff --check` passed. No Java tests, root suite, architecture tests, backend-
  conformance tests, or integration tests were rerun because documentation changed no executable
  Java token and the implementation context supplied fresh successful Compiler test evidence.

No-change conclusions:

- Tensor, Public, Runtime, and Training guides need no edit. The change adds no Tensor or public
  facade contract, caller-visible Runtime representation or lifecycle, optimizer/training
  behavior, or executable result; the Compile API is the focused owner of this internal compiler
  descriptor rule.
- The glossary needs no edit. `Compile-time constant source`, `Logical splat`, `Layout`,
  `TensorDescriptor`, graph input/output, and logical-memory terminology already define every
  reusable term; “source-only published compile-time constant” is a local conjunction of those
  existing facts rather than a new reusable domain object.
- `ARCHITECTURE.md`, focused architecture pages, and ADRs need no edit because module ownership,
  dependency direction, immutable graph meaning, compile/prepare/run ordering, and ADR 0009's
  explicit-splat rule are unchanged.
- Planning 0005–0006, Prepare 0003, and current Planning/Prepare/Runtime/CPU/Engine contracts need
  no edit. Compiler supplies one resolved logical descriptor while Planning preserves its
  existing graph-output obligation; Prepare 0005 and CPU 0010H remain the separately owned Draft
  resource and materialization prerequisites, and Engine 0006 remains dependent on that chain.
- Tests outside Compiler need no edit because no non-Compiler behavior or public cross-module
  shape changes. Architecture tests, backend conformance, and integration tests therefore have no
  new invariant to enforce in this task.
- Gradle and build structure need no edit because the new package-private type uses existing
  Compiler-to-Model dependencies and creates no package, module, dependency, generated-code, or
  repository-wide build change.

## Implementation notes

- Added `PublishedCompileTimeConstantDescriptorClosure`, a stateless package-private final-graph
  transformation. It returns the exact validated input when nothing qualifies and otherwise
  rebuilds only eligible `GraphValue` descriptors through `LayoutDescriptor.contiguous(shape)`.
- Routed both graph-stage compile branches through the closure exactly once after final
  optimization. Complete compilation consequently derives all publication, constant, diagnostic,
  ownership, partition, and memory artifacts from the closed graph.
- Focused tests cover package shape, null behavior, scalar/zero/ordinary geometry, all six data
  types and exact splats, every excluded category, resolved-descriptor identity, topology and
  metadata preservation, overflow, forward-only publication, absent-unit-seed gradients, and
  complete logical-memory truth.
- Finalized the Compile API's current logical descriptor rule and synchronized this task, the
  Compiler master plan, and the roadmap. Existing implementation Javadocs were already complete
  and accurate after independent review, so the documentation pass did not churn them.

## Completion summary

- Completed changes: implemented final source-only published compile-time constant descriptor
  closure with exact eligibility, placement, canonical layout, overflow, preservation, and
  complete-artifact behavior.
- Files changed or created: exactly the task's eight paths—two Compiler production paths, two
  Compiler tests, Compile API, this task, Compiler master plan, and roadmap. The documentation
  pass edited only the four allowlisted Markdown paths; both approved Java Javadocs were reviewed
  without change.
- Tests and validation: reused the implementation context's passing 32-test focused result and
  38-suite/258-test Compiler module result because no executable Java changed afterward; final
  Javadoc, Markdown, scope, inventory, status, preservation, and whitespace results are recorded
  in the validation evidence.
- Documentation-agent review: clean documentation context
  `01a0a593-f2a4-7de1-889f-5fb01a65642d` completed the independent pass.
- Documentation impact: finalized Compile API behavior and synchronized planning status/evidence;
  adjacent guides, glossary, architecture, ADRs, and downstream contracts required no change for
  the reasons recorded above.
- Javadoc review: both changed production types provide meaningful placement, semantics,
  preservation, input/result, failure, and non-physical-boundary documentation; no edit was
  needed.
- Glossary impact: no change; existing terms cover the capability without creating a new domain
  term.
- Unresolved issues: None within Compiler 0006B5.
- Follow-up required: Draft Prepare 0005, then Draft CPU 0010H, then Draft Engine 0006. Prepare
  0005 remains the next operational master-plan placeholder and is not detailed by this task.

Status: Complete
