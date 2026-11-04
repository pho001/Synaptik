# Task 0006B5: Published [Compile](../../../../glossary.md#compile)-Time [Constant](../../../../glossary.md#term-constant) [Descriptor](../../../../glossary.md#term-descriptor) Closure

## Status

Complete

## Goal

Close one general [Compiler](../../../../glossary.md#term-compiler)-owned [descriptor](../../../../glossary.md#term-descriptor)-contract gap for fully static, source-only,
published [compile](../../../../glossary.md#compile)-time splat [constants](../../../../glossary.md#term-constant). After final [graph optimization](../../../../glossary.md#forward-graph-optimization) determines that such a
value remains a [graph](../../../../glossary.md#graph) input and graph output but has no [node](../../../../glossary.md#node) consumer, give its unresolved
`TensorDescriptor` the [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous) logical [layout](../../../../glossary.md#layout) already implied by its fully static
[`Shape`](../../../../glossary.md#shape).

## Current defect

The current pipeline can produce a valid source-only [publication](../../../../glossary.md#term-publication) obligation with no [producer](../../../../glossary.md#term-producer) and
no consumer. This is particularly visible when functional differentiation creates an absent
[scalar](../../../../glossary.md#term-scalar) unit [seed](../../../../glossary.md#term-seed) and that seed is itself the requested [gradient](../../../../glossary.md#term-gradient): [capture](../../../../glossary.md#term-capture-meanings) retains it as a [constant](../../../../glossary.md#term-constant)
[graph](../../../../glossary.md#graph) input and graph output, optimization retains the publication, and logical-memory [planning](../../../../glossary.md#term-planning)
truthfully retains a producerless/consumerless graph-output requirement. The final [descriptor](../../../../glossary.md#term-descriptor),
however, still has unresolved [layout](../../../../glossary.md#layout), so later [lifecycle](../../../../glossary.md#lifecycle) layers cannot derive a complete
[materialization](../../../../glossary.md#materialization) contract. This task closes only that [Compiler](../../../../glossary.md#term-compiler) descriptor fact. It is not an
[Engine](../../../../glossary.md#term-engine)-specific workaround and does not make the value physical or executable.

## Exact semantics and invariants

[Run](../../../../glossary.md#term-run) [descriptor](../../../../glossary.md#term-descriptor) closure exactly once after `ForwardGraphOptimization.optimize(...)` returns its
final validated [graph](../../../../glossary.md#graph) and before `GraphCompiler` derives final forward/[gradient](../../../../glossary.md#term-gradient) [bindings](../../../../glossary.md#term-binding),
[publication](../../../../glossary.md#term-publication), [backend ownership](../../../../glossary.md#backend-ownership), partitions, or logical-memory [artifacts](../../../../glossary.md#term-artifact). This is the safest
placement because final [DCE](../../../../glossary.md#term-dead-code-elimination-dce), [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse), and [constant folding](../../../../glossary.md#term-constant-folding) have already fixed the surviving graph
[topology](../../../../glossary.md#term-topology) and source roles, while every downstream artifact will observe one consistent closed
descriptor.

A value is eligible if and only if all of these facts hold in the final optimized graph:

1. its [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) occurs in `CompiledGraphModel.inputs()`;
2. the same ID has a `CompileTimeConstantGraph.Splat` fact;
3. the same ID occurs in `CompiledGraphModel.outputs()`;
4. no final `CompiledNode.inputs()` [occurrence](../../../../glossary.md#term-occurrence) consumes the ID;
5. its [`Shape`](../../../../glossary.md#shape) is fully static; and
6. its `TensorDescriptor.layout()` is empty.

For each eligible value, replace only its descriptor with a descriptor containing
`LayoutDescriptor.contiguous(shape)`. Preserve the exact [`DataType`](../../../../glossary.md#data-type--datatype), exact `Shape`, and
`requiresGrad` value. Current [constant](../../../../glossary.md#term-constant) [validation](../../../../glossary.md#term-validation) already requires `requiresGrad == false`; this
task must neither weaken nor duplicate that rule.

[Canonical](../../../../glossary.md#term-canonical-form) [layout](../../../../glossary.md#layout) treatment is exact for every current `DataType`: [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32),
[`BFLOAT16`](../../../../glossary.md#term-bfloat16), [`INT32`](../../../../glossary.md#data-type--datatype), [`INT64`](../../../../glossary.md#data-type--datatype), and [`BOOL`](../../../../glossary.md#data-type--datatype). [Data type](../../../../glossary.md#data-type--datatype) does not alter [element](../../../../glossary.md#term-element)-layout geometry.

- `Shape.scalar()` closes to rank zero, no strides, element offset zero, non-view contiguous
  layout, and referenced element span one.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [view glossary](../../../../glossary.md#term-view-meanings); [contiguous glossary](../../../../glossary.md#term-contiguous); [referenced element span glossary](../../../../glossary.md#referenced-element-span))</span>
- Any fully static Shape containing a zero-sized dimension closes through the same canonical
  factory. It retains the Shape's rank and canonical strides, uses element offset zero, is not a
  view, and has referenced element span zero.<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension))</span>
- Other fully static Shapes receive the factory's canonical row-major element strides, element
  offset zero, non-view status, and checked referenced span.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [referenced span glossary](../../../../glossary.md#referenced-element-span))</span>

If canonical [stride](../../../../glossary.md#term-stride) or span arithmetic overflows for an otherwise eligible value, [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation) at
[compile](../../../../glossary.md#compile) time with a deterministic `IllegalArgumentException` that identifies the `ValueId` and
retains the arithmetic failure as its cause. Do not leave that eligible descriptor unresolved or
silently skip it.

Preserve state unchanged for every non-eligible value, including:

- dynamic or partially dynamic source-only published constants;
- source constants that are consumed by any [node](../../../../glossary.md#node), even when also published;
- constants that are not published;
- caller-bindable graph inputs;
- produced values and non-constant values; and
- eligible values whose layout is already resolved.

Rebuilding the [immutable](../../../../glossary.md#term-immutable) graph must preserve exact value encounter order, `ValueId`s, node list
and exact node references, node inputs/outputs, [`NodeId`](../../../../glossary.md#nodeid)s, input/output order, graph phases,
deferred constraints, constant splat values, bindable [Tensor](../../../../glossary.md#tensor) [identities](../../../../glossary.md#term-identity-meanings), [derivative](../../../../glossary.md#term-derivative)-order values,
forward publication identities/order, gradient publication [target](../../../../glossary.md#term-target)/order [metadata](../../../../glossary.md#term-metadata), [aliases](../../../../glossary.md#aliasing-and-arena), and
publication roles. Derivative metadata must own the rebuilt exact graph reference without
changing its ordered node-to-order mapping.

The closure adds no allocation, byte size, [alignment](../../../../glossary.md#term-alignment), slot, representation, [residency](../../../../glossary.md#residency), physical
layout, [backend](../../../../glossary.md#backend) choice, caller binding, node, [operation](../../../../glossary.md#operation), synthetic topology, or new value. Logical
memory [planning](../../../../glossary.md#term-planning) must continue to report the source-only publication as graph-output-required with
empty [producer](../../../../glossary.md#term-producer) and consumer sets.

## [Scope](../../../../glossary.md#term-scope)

- Add one package-private Compiler transformation for the exact eligibility predicate and
  descriptor replacement above.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Invoke it in both forward-only and backward-capable compilation after final optimization and
  before final boundary/publication derivation.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [publication glossary](../../../../glossary.md#term-publication))</span>
- Cover scalar, zero-element, ordinary static, all-supported-type, preservation, overflow, and
  complete-compile integration cases with focused Compiler tests.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element); [overflow glossary](../../../../glossary.md#term-overflow); [compile glossary](../../../../glossary.md#compile))</span>
- Update affected Compiler Javadoc and Compile API/planning documentation through the required
  clean documentation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Physical allocation, byte/alignment declarations, prepared slots, representations,
  initialization recipes, copying, execution, or publication materialization.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment); [publication glossary](../../../../glossary.md#term-publication); [materialization glossary](../../../../glossary.md#materialization))</span>
- Changes to graph topology, graph inputs/outputs, node construction, operation semantics,
  optimization eligibility/order, constant folding, derivative formulas, seed policy, caller
  binding, publication roles/order, partitioning, or logical-memory ownership.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [topology glossary](../../../../glossary.md#term-topology); [node glossary](../../../../glossary.md#node); [operation glossary](../../../../glossary.md#operation); [constant folding glossary](../../../../glossary.md#term-constant-folding); [derivative glossary](../../../../glossary.md#term-derivative); [seed glossary](../../../../glossary.md#term-seed); [binding glossary](../../../../glossary.md#term-binding); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Backend selection or any Prepare, Runtime, CPU, Engine, Model, Config, build, architecture,
  ADR, API-facade, conformance, integration, or performance work.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [Model glossary](../../../../glossary.md#term-model); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [facade glossary](../../../../glossary.md#term-facade); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- General default-layout inference for static descriptors. Only the closed final source-only
  published constant role establishes the invariant used here.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [inference glossary](../../../../glossary.md#term-inference); [descriptors glossary](../../../../glossary.md#term-descriptor); [constant glossary](../../../../glossary.md#term-constant); [invariant glossary](../../../../glossary.md#term-invariant))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture index](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Runtime/Prepare/backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend))</span>
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Planning guide](../../../planning-guide.md), [roadmap](../../../roadmap.md), and
  [Compiler master plan](../master-plan.md)
- [Compiler](../../../../glossary.md#term-compiler) [0006](0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md)
  and [0006B4](0006b4-stable-caller-input-tensor-identity-bindings.md)
- [Planning](../../../../glossary.md#term-planning) [0005](../../planning/tasks/0005-logical-materialization-and-memory-requirements.md)
  and [0006](../../planning/tasks/0006-planning-contract-closure-audit.md)
- [Prepare](../../../../glossary.md#prepare) [0003](../../prepare/tasks/0003-prepare-orchestration-and-validation.md)

## Architecture constraints

[Compiler](../../../../glossary.md#term-compiler) owns final [immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph) [descriptors](../../../../glossary.md#term-descriptor) and [compile artifacts](../../../../glossary.md#compile-artifacts) but not physical [buffers](../../../../glossary.md#term-buffer).
[Planning](../../../../glossary.md#term-planning) continues to describe logical [producer](../../../../glossary.md#term-producer), consumer, and [publication](../../../../glossary.md#term-publication) obligations without
inventing execution. [Prepare](../../../../glossary.md#prepare) continues to own shared resource [handoff](../../../../glossary.md#term-handoff) and deterministic slot
assignment, concrete [backends](../../../../glossary.md#backend) own physical representation declarations/[materialization](../../../../glossary.md#materialization), [Runtime](../../../../glossary.md#term-runtime)
owns initialized representation validity and publication [lifetime](../../../../glossary.md#term-lifetime), and [Engine](../../../../glossary.md#term-engine) owns outward
orchestration. No [dependency direction](../../../../glossary.md#java-platform-and-build-tools), public [API](../../../../glossary.md#term-application-programming-interface-api), [module](../../../../glossary.md#term-module) boundary, or architecture rule changes.

Do not conceal the downstream Prepare/[CPU](../../../../glossary.md#term-cpu) gaps in Compiler. Stop and replan if implementation
requires allocation, backend knowledge, [topology](../../../../glossary.md#term-topology) synthesis, a public contract, another module, or
an architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) change.

## Package impact

One package-private transformation is added beside the existing [graph](../../../../glossary.md#graph) transformations in
`io.github.pho001.synaptik.compiler`. No public or exported package [shape](../../../../glossary.md#shape) changes.

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

Review without editing [Model](../../../../glossary.md#term-model) [descriptor](../../../../glossary.md#term-descriptor)/[layout](../../../../glossary.md#layout) contracts, [Planning](../../../../glossary.md#term-planning) logical-memory source/tests,
[Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [CPU](../../../../glossary.md#term-cpu), [Engine](../../../../glossary.md#term-engine), Config, [Tensor](../../../../glossary.md#tensor)/Public/Runtime/Training [API](../../../../glossary.md#term-application-programming-interface-api) guides, glossary,
architecture/ADRs, architecture tests, build files, and other [modules](../../../../glossary.md#term-module). If review demonstrates a
required edit outside this allowlist, stop and replan rather than [widening](../../../../glossary.md#term-widening) the task implicitly.

## Maximum [scope](../../../../glossary.md#term-scope)

At most those eight paths. The implementation is a small final-[graph](../../../../glossary.md#graph) [descriptor](../../../../glossary.md#term-descriptor) closure plus
focused [Compiler](../../../../glossary.md#term-compiler) evidence. It introduces no performance task, benchmark, repository-wide
[checkpoint](../../../../glossary.md#term-checkpoint), [dependency](../../../../glossary.md#term-dependency) edit, or generated-code work.

## Acceptance criteria

- The transformation runs after final optimization and before final output, gradient,
  publication, partition, and logical-memory derivation in forward-only and backward-capable
  compilation.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [publication glossary](../../../../glossary.md#term-publication); [partition glossary](../../../../glossary.md#partition); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- The eligibility predicate is exactly graph input + constant fact + graph output + no node
  consumer + fully static Shape + unresolved layout.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [constant glossary](../../../../glossary.md#term-constant); [node glossary](../../../../glossary.md#node); [Shape glossary](../../../../glossary.md#shape))</span>
- Scalars close to canonical rank-zero span-one layout; zero-element static Shapes close to
  canonical span-zero layout; ordinary static Shapes close to canonical row-major layout.<span data-glossary-navigation="true"> ([Scalars glossary](../../../../glossary.md#term-scalar); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [element glossary](../../../../glossary.md#term-element); [row-major glossary](../../../../glossary.md#term-row-major))</span>
- Each of the six current data types receives identical Shape-derived treatment and preserves its
  exact type and splat value.
- [Canonical](../../../../glossary.md#term-canonical-form)-[layout](../../../../glossary.md#layout) arithmetic [overflow](../../../../glossary.md#term-overflow) fails closed with value context and retained cause.
- Dynamic, consumed, unpublished, bindable, produced, non-constant, and already-resolved values
  remain exactly unchanged.
- Graph/value/node IDs, orders, topology, phases, constraints, source roles, bindable Tensor
  identities, derivative orders, forward and gradient bindings, aliases, and publication order
  remain unchanged.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [Tensor glossary](../../../../glossary.md#tensor); [derivative glossary](../../../../glossary.md#term-derivative); [bindings glossary](../../../../glossary.md#term-binding); [aliases glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Complete compilation exposes the resolved descriptor consistently in graph, publication,
  constants, diagnostics where applicable, and logical memory; producerless/consumerless
  graph-output truth remains intact and no physical fact appears.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- Public/package inventory is unchanged, only allowlisted paths change, and Compiler 0006B5
  remains Draft until implementation and documentation validation complete.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [validation glossary](../../../../glossary.md#term-validation))</span>
- The documentation pass reviews all affected Javadocs, Compile API wording, terminology, and
  glossary impact. It records reasoned no-change conclusions for the Tensor/Public/Runtime/
  Training guides, glossary, architecture/ADRs, Planning/Prepare/Runtime/CPU/Engine contracts,
  tests outside Compiler, and build structure.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Runtime glossary](../../../../glossary.md#term-runtime); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>

## Tests and [validation](../../../../glossary.md#term-validation)

Focused [Compiler](../../../../glossary.md#term-compiler) tests must prove:

- direct transformation of [scalar](../../../../glossary.md#term-scalar), multiple zero-[element](../../../../glossary.md#term-element) [Shapes](../../../../glossary.md#shape), and ordinary static Shapes;
- all six current data types and exact [constant](../../../../glossary.md#term-constant)-splat preservation;
- exact preservation of every non-eligible category and already-resolved [descriptor](../../../../glossary.md#term-descriptor) [identity](../../../../glossary.md#term-identity-meanings);
- deterministic [overflow](../../../../glossary.md#term-overflow) failure and cause;
- exact [graph](../../../../glossary.md#graph) [topology](../../../../glossary.md#term-topology), IDs, boundary order, phase and [derivative](../../../../glossary.md#term-derivative)-order preservation;
- absent-unit-[seed](../../../../glossary.md#term-seed) backward [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) where the seed is a source-only [gradient](../../../../glossary.md#term-gradient) [publication](../../../../glossary.md#term-publication);
- forward-only source-only published constant compilation; and
- complete artifacts retaining empty producer/consumer sets plus graph-output obligation without
  adding a partition or physical declaration.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [producer glossary](../../../../glossary.md#term-producer); [partition glossary](../../../../glossary.md#partition))</span>

Future implementation task-tier validation:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.PublishedCompileTimeConstantDescriptorClosureTest --tests io.github.pho001.synaptik.compiler.GraphCompilerTest
./gradlew :modules:compiler:test
./gradlew :modules:compiler:javadoc
git diff --check
```

The clean documentation pass also validates Markdown links, anchors, heading uniqueness, fences,
trailing whitespace, LF/final newline, exact eight-path [scope](../../../../glossary.md#term-scope), package/public inventories, task/
master/roadmap status and ordering, and preservation of excluded files. Do not [run](../../../../glossary.md#term-run) unchanged
[Planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [CPU](../../../../glossary.md#term-cpu), [Engine](../../../../glossary.md#term-engine), architecture, [conformance](../../../../glossary.md#term-conformance), integration, or root suites;
their behavior belongs to later [owner](../../../../glossary.md#term-owner) tasks or [CI](../../../../glossary.md#term-continuous-integration-ci).

Validation tier: affected Compiler [module](../../../../glossary.md#term-module) plus affected documentation. No repository-wide or
performance validation is justified by this descriptor-only Compiler change.

## Dependencies

[Compiler](../../../../glossary.md#term-compiler) 0006 and 0006B4, [Planning](../../../../glossary.md#term-planning) 0005–0006, and the current [Model](../../../../glossary.md#term-model) [descriptor](../../../../glossary.md#term-descriptor)/[layout](../../../../glossary.md#layout) contracts
are Complete. This task is the first unfinished Compiler task and precedes Draft Compiler 0006C
and 0007 in execution order.

## [Handoff](../../../../glossary.md#term-handoff) to [Prepare](../../../../glossary.md#prepare) 0005

[Compiler](../../../../glossary.md#term-compiler) 0006B5 hands Prepare a fully static source-only published [constant](../../../../glossary.md#term-constant) whose final logical
[descriptor](../../../../glossary.md#term-descriptor) is resolved while its [Planning](../../../../glossary.md#term-planning) requirement correctly remains producerless,
consumerless, and [graph](../../../../glossary.md#graph)-output-required. Draft Prepare 0005 then owns explicit contribution of
that [publication](../../../../glossary.md#term-publication) resource to the shared handoff and deterministic slot assignment. Compiler does
not prescribe [backend](../../../../glossary.md#backend) selection, physical geometry, representation, initialization, or schedule
assembly.

## Documentation-focused clean-context pass

After Java and focused [Compiler](../../../../glossary.md#term-compiler) tests stabilize, provide a distinct clean documentation context
with this task, the exact diff, test evidence, baseline [hashes](../../../../glossary.md#term-hash), source-only eligibility and
placement decisions, and downstream [ownership](../../../../glossary.md#term-ownership-meanings) boundary. Apply the General, [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), and
[Planning](../../../../glossary.md#term-planning) profiles. Finalize affected production Javadocs and `docs/api/compile-api.md`; review the
glossary and other API guides and record reasoned no-change conclusions unless an allowlist change
requires replanning. Do not repeat successful Java tests unless documentation changes executable
Java behavior or exposes a concrete failure.

## Follow-up tasks

- Prepare 0005: explicit producerless published-constant resource contribution/handoff and
  deterministic shared slot assignment without backend selection or physical geometry.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [constant glossary](../../../../glossary.md#term-constant); [handoff glossary](../../../../glossary.md#term-handoff); [backend glossary](../../../../glossary.md#backend))</span>
- CPU 0010H: materialize such a constant in the current sole non-empty CPU composition through
  existing initialized-buffer/publication machinery; keep zero-node and mixed/multi-partition
  compositions rejected.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [buffer glossary](../../../../glossary.md#term-buffer); [publication glossary](../../../../glossary.md#term-publication); [node glossary](../../../../glossary.md#node); [partition glossary](../../../../glossary.md#partition))</span>
- Engine 0006: implement the already selected outward scalar-objective backward convenience only
  after Compiler 0006B5, Prepare 0005, and CPU 0010H are Complete.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [scalar glossary](../../../../glossary.md#term-scalar); [objective glossary](../../../../glossary.md#term-objective); [Compiler glossary](../../../../glossary.md#term-compiler))</span>

[Runtime](../../../../glossary.md#term-runtime) needs no new task: current initialized representations, initial validity, ordered
publications/[aliases](../../../../glossary.md#aliasing-and-arena), [leases](../../../../glossary.md#term-lease), and cleanup already provide the required runtime [lifecycle](../../../../glossary.md#lifecycle).

## Architecture impact

Expected impact: None. This closes a [Compiler](../../../../glossary.md#term-compiler)-owned logical [descriptor](../../../../glossary.md#term-descriptor) fact within existing
[immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph) and [artifact](../../../../glossary.md#term-artifact) [ownership](../../../../glossary.md#term-ownership-meanings). Stop rather than edit if implementation evidence
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

- Final optimized [topology](../../../../glossary.md#term-topology), not pre-optimization [capture](../../../../glossary.md#term-capture-meanings), determines source-only eligibility.
- Publication is represented by final graph-output membership; no Engine-specific request fact
  enters the transformation.<span data-glossary-navigation="true"> ([Publication glossary](../../../../glossary.md#term-publication); [graph glossary](../../../../glossary.md#graph); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Existing Model canonical layout construction defines scalar, zero-element, and checked static
  geometry; Compiler does not reproduce that arithmetic.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [canonical glossary](../../../../glossary.md#term-canonical-form); [layout glossary](../../../../glossary.md#layout); [scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- An eligible [overflow](../../../../glossary.md#term-overflow) is a malformed unclosable final contract and fails closed with context.
- Downstream [materialization](../../../../glossary.md#materialization) gaps remain visible and separately owned.

## Known limitations

Dynamic [Shapes](../../../../glossary.md#shape) remain unresolved. Consumed and unpublished [constants](../../../../glossary.md#term-constant) retain their current
[descriptor](../../../../glossary.md#term-descriptor) behavior. This task alone does not make a source-only constant preparable or
executable; [Prepare](../../../../glossary.md#prepare) 0005 and [CPU](../../../../glossary.md#term-cpu) 0010H are required before [Engine](../../../../glossary.md#term-engine) 0006.

## [Validation](../../../../glossary.md#term-validation) evidence

[Planning](../../../../glossary.md#term-planning) diagnosis context: `01a0a570-06d4-7012-814b-ca71af8676e7`.
Implementation context: `01a0a589-5750-7392-b5dc-26890c043a9a`.
Documentation context: `01a0a593-f2a4-7de1-889f-5fb01a65642d`.

- The planning context read the required architecture, planning, Compiler/Planning/Prepare/
  Runtime/CPU/Engine lineage, documentation profiles, directly relevant source/tests, and starting
  dirty diff/status. It verified the absent-seed source-only publication shape; selected the post-
  optimization, pre-boundary placement; preserved Planning's producerless/consumerless graph-
  output obligation; and confirmed that Prepare and CPU own the remaining handoff and physical
  materialization gaps while Runtime needs no new task.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [seed glossary](../../../../glossary.md#term-seed); [publication glossary](../../../../glossary.md#term-publication); [shape glossary](../../../../glossary.md#shape); [graph glossary](../../../../glossary.md#graph); [handoff glossary](../../../../glossary.md#term-handoff); [materialization glossary](../../../../glossary.md#materialization))</span>
- The implementation context added the package-private [descriptor](../../../../glossary.md#term-descriptor) closure and invoked it after
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
  boundaries, owner plans, build structure, and the intentionally dirty baseline.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [artifact glossary](../../../../glossary.md#term-artifact); [Compile glossary](../../../../glossary.md#compile); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [owner glossary](../../../../glossary.md#term-owner))</span>
- Source inspection confirmed the exact graph-input + splat + graph-output + no-consumer + fully-
  static + unresolved-layout predicate; canonical scalar, zero-element, ordinary-static, and six-
  data-type treatment; deterministic overflow wrapping with <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [scalar glossary](../../../../glossary.md#term-scalar); [element glossary](../../../../glossary.md#term-element); [overflow glossary](../../../../glossary.md#term-overflow))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) and cause; exact no-change
  identity for an unchanged result; and rebuilt graph, constant, constraint, bindable-identity,
  phase, boundary, node-reference, and derivative-order preservation.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [constant glossary](../../../../glossary.md#term-constant); [node glossary](../../../../glossary.md#node); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- Complete-compile inspection confirmed the closure precedes forward/gradient binding,
  publication, capability queries, partitioning, and logical-memory derivation. Tests retain the
  source-only value as a constant graph input/output with no node consumer or partition and one
  graph-output-required logical-memory row with empty producer and consumer sets.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [binding glossary](../../../../glossary.md#term-binding); [partition glossary](../../../../glossary.md#partition); [producer glossary](../../../../glossary.md#term-producer))</span>
- Documentation-context `./gradlew :modules:compiler:javadoc` passed after final Javadoc review
  with `BUILD SUCCESSFUL`; all seven actionable tasks were up to date. Generated
  `PublishedCompileTimeConstantDescriptorClosure.html` and `GraphCompiler.html` render the final
  contracts, and the closure method is identified as package-private.
- `python3 /tmp/validate_synaptik_markdown.py` passed for the four changed Markdown paths. It
  validated local links and anchors, balanced fences, LF/final newlines, and trailing whitespace;
  a separate heading scan found no duplicate heading text.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
- Source, bytecode, and generated-page inventory checks confirmed one final package-private,
  field-free <span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode))</span>`PublishedCompileTimeConstantDescriptorClosure`, one private constructor, one
  package-private static `close(ValidatedGraph)` method, and no new public Compiler type or method.
  The two call sites are immediately outside `ForwardGraphOptimization.optimize(...)` and precede
  final boundary derivation in both compile branches.
- Scope checks found exactly the eight task paths in the overall Compiler 0006B5 change. This
  documentation pass edited only Compile API, this task, Compiler master plan, and roadmap; it
  reviewed the two allowlisted Java Javadocs without changing them and did not touch either test.
  Pre-existing dirty paths outside the allowlist remain excluded and preserved.<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope))</span>
- Status/order checks found Compiler Complete through 0006B5 in both master plan and roadmap,
  followed by Draft Prepare 0005, Draft CPU 0010H, and Draft Engine 0006. Prepare 0005 remains a
  master-plan placeholder with no detailed task file. The Engine 0006 dependency chain is
  preserved.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Final `git diff --check` passed. No Java tests, root suite, architecture tests, backend-
  conformance tests, or integration tests were rerun because documentation changed no executable
  Java token and the implementation context supplied fresh successful Compiler test evidence.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [token glossary](../../../../glossary.md#term-token))</span>

No-change conclusions:

- Tensor, Public, Runtime, and Training guides need no edit. The change adds no Tensor or public
  facade contract, caller-visible Runtime representation or lifecycle, optimizer/training
  behavior, or executable result; the Compile API is the focused owner of this internal compiler
  descriptor rule.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [facade glossary](../../../../glossary.md#term-facade); [lifecycle glossary](../../../../glossary.md#lifecycle); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- The glossary needs no edit. [`Compile-time constant source`](../../../../glossary.md#compile-time-constant-source), [`Logical splat`](../../../../glossary.md#logical-splat), [`Layout`](../../../../glossary.md#layout),
  `TensorDescriptor`, graph input/output, and logical-memory terminology already define every
  reusable term; “source-only published compile-time constant” is a local conjunction of those
  existing facts rather than a new reusable domain object.
- `ARCHITECTURE.md`, focused architecture pages, and ADRs need no edit because module ownership,
  dependency direction, immutable graph meaning, compile/prepare/run ordering, and ADR 0009's
  explicit-splat rule are unchanged.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [immutable glossary](../../../../glossary.md#term-immutable); [run glossary](../../../../glossary.md#term-run))</span>
- Planning 0005–0006, Prepare 0003, and current Planning/Prepare/Runtime/CPU/Engine contracts need
  no edit. Compiler supplies one resolved logical descriptor while Planning preserves its
  existing graph-output obligation; Prepare 0005 and CPU 0010H remain the separately owned Draft
  resource and materialization prerequisites, and Engine 0006 remains dependent on that chain.
- Tests outside Compiler need no edit because no non-Compiler behavior or public cross-module
  shape changes. Architecture tests, backend conformance, and integration tests therefore have no
  new invariant to enforce in this task.<span data-glossary-navigation="true"> ([invariant glossary](../../../../glossary.md#term-invariant))</span>
- Gradle and build structure need no edit because the new package-private type uses existing
  Compiler-to-Model dependencies and creates no package, module, dependency, generated-code, or
  repository-wide build change.<span data-glossary-navigation="true"> ([Gradle glossary](../../../../glossary.md#term-gradle); [Model glossary](../../../../glossary.md#term-model))</span>

## Implementation notes

- Added `PublishedCompileTimeConstantDescriptorClosure`, a stateless package-private final-graph
  transformation. It returns the exact validated input when nothing qualifies and otherwise
  rebuilds only eligible <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>`GraphValue` [descriptors](../../../../glossary.md#term-descriptor) through `LayoutDescriptor.contiguous(shape)`.
- Routed both graph-stage compile branches through the closure exactly once after final
  optimization. Complete compilation consequently derives all publication, constant, diagnostic,
  ownership, partition, and memory artifacts from the closed graph.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [publication glossary](../../../../glossary.md#term-publication); [constant glossary](../../../../glossary.md#term-constant); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [partition glossary](../../../../glossary.md#partition); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Focused tests cover package shape, null behavior, scalar/zero/ordinary geometry, all six data
  types and exact splats, every excluded category, resolved-descriptor identity, topology and
  metadata preservation, overflow, forward-only publication, absent-unit-seed gradients, and
  complete logical-memory truth.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [scalar glossary](../../../../glossary.md#term-scalar); [identity glossary](../../../../glossary.md#term-identity-meanings); [topology glossary](../../../../glossary.md#term-topology); [metadata glossary](../../../../glossary.md#term-metadata); [overflow glossary](../../../../glossary.md#term-overflow); [seed glossary](../../../../glossary.md#term-seed); [gradients glossary](../../../../glossary.md#term-gradient))</span>
- Finalized the Compile API's current logical descriptor rule and synchronized this task, the
  Compiler master plan, and the roadmap. Existing implementation Javadocs were already complete
  and accurate after independent review, so the documentation pass did not churn them.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Compiler glossary](../../../../glossary.md#term-compiler))</span>

## Completion summary

- Completed changes: implemented final source-only published compile-time constant descriptor
  closure with exact eligibility, placement, canonical layout, overflow, preservation, and
  complete-artifact behavior.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [constant glossary](../../../../glossary.md#term-constant); [descriptor glossary](../../../../glossary.md#term-descriptor); [canonical glossary](../../../../glossary.md#term-canonical-form); [layout glossary](../../../../glossary.md#layout); [overflow glossary](../../../../glossary.md#term-overflow); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Files changed or created: exactly the task's eight paths—two Compiler production paths, two
  Compiler tests, Compile API, this task, Compiler master plan, and roadmap. The documentation
  pass edited only the four allowlisted Markdown paths; both approved Java Javadocs were reviewed
  without change.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Tests and validation: reused the implementation context's passing 32-test focused result and
  38-suite/258-test Compiler module result because no executable Java changed afterward; final
  Javadoc, Markdown, scope, inventory, status, preservation, and whitespace results are recorded
  in the validation evidence.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: clean documentation context
  `01a0a593-f2a4-7de1-889f-5fb01a65642d` completed the independent pass.
- Documentation impact: finalized Compile API behavior and synchronized planning status/evidence;
  adjacent guides, glossary, architecture, ADRs, and downstream contracts required no change for
  the reasons recorded above.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Javadoc review: both changed production types provide meaningful placement, semantics,
  preservation, input/result, failure, and non-physical-boundary documentation; no edit was
  needed.
- Glossary impact: no change; existing terms cover the capability without creating a new domain
  term.
- Unresolved issues: None within [Compiler](../../../../glossary.md#term-compiler) 0006B5.
- Follow-up required: Draft Prepare 0005, then Draft CPU 0010H, then Draft Engine 0006. Prepare
  0005 remains the next operational master-plan placeholder and is not detailed by this task.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>

Status: Complete
