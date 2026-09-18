# Task 0006B6: Final Convolution Logical-Layout Closure

## Status

Ready

## Goal

Close the final Compiler-owned logical-layout gap that prevents Planning from asking a backend
about otherwise valid public NCW Conv1d, grouped NCHW Conv2d, and grouped NCDHW Conv3d forward
graphs. After final graph optimization and validation, assign canonical contiguous logical layout
to eligible static convolution results and propagate that newly closed layout through only the
direct singleton-height `SQUEEZE` used by the existing public Conv1d composition.

Keep Model construction unchanged: Conv2d and Conv3d expressions continue to begin with unresolved
result layouts, and Conv1d remains the visible `EXPAND_DIMS -> CONV2D -> SQUEEZE` composition.
Planning remains a backend-neutral consumer of final descriptors, CPU capability admission remains
strict, and Prepare and Runtime gain no layout inference.

## Current defect

`TensorConv2dExpressions` and `TensorConv3dExpressions` intentionally construct convolution result
descriptors with unresolved layout. `StructuredOperationInference` independently validates their
type and Shape but derives the same unresolved result. `GraphCompiler` therefore gives Planning an
`OperationCapabilityQuery` whose Conv2d or Conv3d output layout is absent. The CPU capability
provider correctly rejects every occurrence with unresolved input or output geometry.

A following `contiguous()` cannot repair the occurrence already queried by Planning. Conv1d has an
additional consequence: its Conv2d result is unresolved when ordinary `SQUEEZE` inference runs, so
the public rank-three result also remains unresolved even when the input and weight leaves have
static contiguous layouts. CPU already implements the numerical families; the missing fact is the
Compiler's final logical descriptor, not backend behavior.

## Selected mechanism

Add one package-private `ConvolutionLogicalLayoutClosure` final-graph rewrite in
`io.github.pho001.synaptik.compiler`. Invoke it exactly once in both graph-compilation branches,
after `ForwardGraphOptimization.optimize(...)` and
`PublishedCompileTimeConstantDescriptorClosure.close(...)`, and before final forward/gradient
bindings, publication, capability queries, partitioning, or logical-memory planning.

This is not a controlled inference mode. Current inference remains the independent validation
oracle for Model construction semantics, including unresolved convolution layouts, and every
optimization candidate continues to pass that oracle before closure. The new pass changes only
final logical descriptors after topology and semantics are fixed. It follows the existing
post-validation closure seam established by Compiler 0006B5 and avoids a second inference policy
that would conflate expression validation with final Compiler layout choice.

The pass processes final nodes in stored graph order and maintains the descriptor selected for
each `ValueId`. It may make exactly two kinds of replacement:

1. an eligible `CONV2D` or `CONV3D` output receives canonical contiguous logical layout; and
2. an eligible direct `SQUEEZE(axis=2)` output receives the exact view layout derived from a
   Conv2d output closed by this same pass.

It adds no node, edge, value, operation, attribute, phase, or synthetic Conv1d kind.

## Exact convolution closure rule

A final node output is eligible for convolution closure if and only if all of these facts hold:

1. the exact final operation kind is `Conv2dKind.CONV2D` or `Conv3dKind.CONV3D`;
2. the node has the already validated single output for that kind;
3. the output descriptor's `Shape` is fully static; and
4. the output descriptor's layout is unresolved.

For an eligible output, replace only its `TensorDescriptor` with one retaining the exact
`DataType`, exact `Shape`, and exact `requiresGrad` value and containing
`LayoutDescriptor.contiguous(shape)`. This is a backend-independent logical choice for the fresh
convolution result, not a physical representation or CPU route.

Static versus dynamic behavior is strict:

- a fully static eligible Shape closes to the Model's canonical row-major layout, including the
  established scalar/zero-extent behavior even though current convolution ranks are four/five;
- a dynamic or partially dynamic convolution result remains unresolved and unchanged;
- an already resolved result remains unchanged; and
- the pass does not infer bindings, specialize symbolic dimensions, or use input storage to make
  a Shape static.

Use `LayoutDescriptor.contiguous(shape)` as the sole source of stride and referenced-span
arithmetic. If canonical stride or span arithmetic overflows, fail compilation with a deterministic
`IllegalArgumentException` identifying the operation kind, `NodeId`, and output `ValueId`, and
retain the arithmetic failure as the cause. Do not skip the eligible output or leave it unresolved.

## Exact Conv1d propagation rule

Public Conv1d remains four ordinary producers: input `EXPAND_DIMS(axis=2)`, weight
`EXPAND_DIMS(axis=2)`, mapped `CONV2D`, and result `SQUEEZE(axis=2)`. The new pass does not identify
that sequence as a fake Conv1d operation and does not inspect ancestors to classify it.

Instead, after closing a Conv2d result, the pass applies the existing exact singleton-axis view
rule only when a final node satisfies all of these facts:

1. the operation kind is `AxisTransformKind.SQUEEZE` with `AxisTransformAttrs.axis() == 2`;
2. it has one input and one output;
3. its exact input `ValueId` is a Conv2d output newly closed by this pass;
4. the input Shape is fully static rank four and axis two has extent one;
5. the output Shape is the already validated rank-three Shape obtained by removing axis two; and
6. the output layout is unresolved.

Derive the result layout by dropping stride index two from the newly closed Conv2d layout,
preserving its storage offset, and constructing `LayoutDescriptor.of(outputShape, strides,
offset, true)`. Thus the visible Conv1d output becomes a resolved view without changing the two
upstream `EXPAND_DIMS` nodes, the Conv2d occurrence, or the squeeze occurrence. This rule also
remains correct if the Conv2d result has another consumer or publication.

If checked stride/span arithmetic overflows, fail with deterministic `IllegalArgumentException`
context naming the squeeze `NodeId`, input `ValueId`, and output `ValueId`, retaining the arithmetic
cause. If any eligibility fact is absent, preserve the squeeze descriptor unchanged. Do not
propagate recursively through `PERMUTE`, another `SQUEEZE`, `EXPAND_DIMS`, reshape, contiguous,
slice, publication, or any unrelated operation. In particular, do not introduce a general
post-inference layout propagation policy.

## Preservation invariants

If no descriptor is eligible, return the exact input `ValidatedGraph`. Otherwise rebuild only
changed `GraphValue` descriptors and preserve:

- value encounter order, every `ValueId`, and exact unchanged `GraphValue` references;
- node order, every `NodeId`, exact `CompiledNode` and `Operation` instances, inputs, outputs,
  attributes, and phases;
- graph input/output order and publication mapping;
- compile-time constant splats and their identities;
- caller-bindable `TensorId` to final-`ValueId` associations;
- deferred constraints and their order;
- derivative-order metadata and the exact mapping values, rebuilding only its graph reference as
  required by immutable graph ownership;
- forward and gradient publication identities, target metadata, occurrence order, and roles;
- CSE/DCE results, aliases, and final graph topology; and
- Planning query order, partition inputs, and logical-memory identity.

The pass must not rerun CSE, DCE, constant folding, inference, or graph validation after closure.
It must not allocate storage, select a backend, create physical geometry, or modify Model,
Planning, Prepare, Runtime, Engine, or CPU production behavior.

## Scope

- Add the dedicated package-private final graph closure described above.
- Route forward-only and backward-capable graph compilation through it after 0006B5 closure.
- Add focused Compiler coverage for positive, negative, dynamic, arithmetic-failure, metadata,
  topology, and Planning-query cases.
- Update the existing `Conv3dCompilerTest` Planning-query assertion so it compares the query
  output with the final compiled Conv3d node output descriptor rather than the intentionally
  unresolved Model producer descriptor.
- Add exact public Engine numerical integration fixtures for NCW Conv1d, grouped NCHW Conv2d,
  and grouped NCDHW Conv3d using only public Tensor and Engine APIs.
- Update the Compile API and synchronized planning documents through the required separate clean
  documentation-focused pass.

## Out of scope

- Changes to convolution construction, Shape semantics, attributes, gradients, operation kinds,
  Model Javadocs, or the unresolved descriptors visible before compilation.
- Conv3d gradients; Compiler 0006C remains Draft and forward-only for Conv3d remains explicit.
- General default layout, layout propagation, dynamic Shape binding, graph specialization, or an
  inference mode.
- Planning policy, capability relaxation, backend fallback, Engine behavior, CPU behavior,
  Prepare/Runtime inference, physical layout, storage, allocation, execution, or materialization.
- Forged `CompiledGraphModel`, `OperationCapabilityQuery`, prepared artifacts, CPU internals, or
  manually constructed inward descriptors as substitutes for public integration evidence.
- Architecture, ADR, dependency, Gradle, conformance, generated-code, performance, or unrelated
  documentation work.
- Reordering or implementing Compiler 0006C or 0007.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Runtime/Prepare/backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Planning guide](../../../planning-guide.md) and [roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- Compiler [0006B](0006b-conv3d-forward-adoption-and-explicit-gradient-boundary.md),
  [0006B3](0006b3-public-constant-free-complete-compile-entry.md),
  [0006B4](0006b4-stable-caller-input-tensor-identity-bindings.md), and
  [0006B5](0006b5-published-compile-time-constant-descriptor-closure.md)
- [Engine master plan](../../engine/master-plan.md) and
  [Engine 0008](../../engine/tasks/0008-engine-lifecycle-capability-checkpoint.md)

## Architecture constraints

Compiler owns inference, validation, final immutable logical descriptors, and construction of the
capability queries consumed by Planning. Planning remains backend-neutral and selects ownership
from exactly those descriptors. CPU continues to require static resolved injective convolution
descriptors. Prepare consumes planned facts without inferring layout, Runtime executes prepared
work without graph interpretation, and Engine remains the outer composition root.

No architecture rule changes. Stop if implementation needs a public Compiler contract, another
module's production change, a backend-specific predicate, runtime binding inference, an ADR, or an
`ARCHITECTURE.md` edit.

## Package impact

Add one package-private, field-free transformation beside current Compiler graph transformations
in `io.github.pho001.synaptik.compiler`. No public/exported package shape changes. Add one
integration test in `io.github.pho001.synaptik.testing.integration`; it validates existing public
cross-module behavior and adds no Engine or CPU production surface.

## Affected files

Exact future implementation allowlist (twelve paths):

1. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ConvolutionLogicalLayoutClosure.java` (new)
2. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompiler.java`
3. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ConvolutionLogicalLayoutClosureTest.java` (new)
4. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`
5. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/Conv3dCompilerTest.java`
6. `testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/EngineConvolutionIntegrationTest.java` (new)
7. `docs/api/compile-api.md`
8. this task
9. `docs/planning/modules/compiler/master-plan.md`
10. `docs/planning/modules/engine/master-plan.md`
11. `docs/planning/modules/engine/tasks/0008-engine-lifecycle-capability-checkpoint.md`
12. `docs/planning/roadmap.md`

Review without editing Model, Planning, Prepare, Runtime, Engine, CPU, Config, Backend Contract,
Gradle, architecture/ADRs, architecture tests, backend conformance, other API guides, or glossary.
If a required thirteenth path appears, stop and replan rather than widening scope implicitly.

## Maximum scope

At most the twelve paths above: two Compiler production paths, three Compiler test paths, one public
integration-test path, Compile API, and five synchronized task/master/roadmap paths. No production
module outside Compiler changes.

## Exact public numerical fixtures

`EngineConvolutionIntegrationTest` must construct ordinary public storage-backed FLOAT32 Tensor
leaves with canonical layouts, build expressions through public `Tensor.conv1d`, `conv2d`, and
`conv3d`, and execute through public `Engine.standard()` only. For each fixture, exercise the
ordinary reusable `compile -> prepare -> run` path with input Tensors supplied in non-Compiler
order to prove typed `TensorId` binding, materialize the public result, and also exercise the
one-shot `compute` path. Do not import Compiler, Planning, Prepare, Runtime, Backend Contract, or
CPU types.

Use these exact fixtures and row-major expected values:

- NCW Conv1d: input Shape `[1,1,4]` values `[1,2,3,4]`, weight Shape `[1,1,2]` values
  `[1,1]`, bias Shape `[1]` value `[0.5]`, default attributes, output Shape `[1,1,3]`, expected
  `[3.5,5.5,7.5]`.
- grouped NCHW Conv2d: input Shape `[1,2,2,2]` values `[1,2,3,4,5,6,7,8]`, weight Shape
  `[2,1,1,1]` values `[2,-1]`, bias Shape `[2]` values `[1,10]`, attributes
  `(strideHeight=1, strideWidth=1, paddingHeight=0, paddingWidth=0, dilationHeight=1,
  dilationWidth=1, groups=2)`, output Shape `[1,2,2,2]`, expected
  `[3,5,7,9,5,4,3,2]`.
- grouped NCDHW Conv3d: input Shape `[1,2,2,1,2]` values `[1,2,3,4,5,6,7,8]`, weight Shape
  `[2,1,1,1,1]` values `[2,-1]`, bias Shape `[2]` values `[1,10]`, attributes
  `(strideDepth=1, strideHeight=1, strideWidth=1, paddingDepth=0, paddingHeight=0,
  paddingWidth=0, dilationDepth=1, dilationHeight=1, dilationWidth=1, groups=2)`, output Shape
  `[1,2,2,1,2]`, expected `[3,5,7,9,5,4,3,2]`.

Assert output descriptor Shape and resolved layout as well as numerical values. Conv3d is forward
only; do not request gradients or imply derivative support.

## Acceptance criteria

- Model Conv2d/Conv3d construction and ordinary inference still produce unresolved layouts.
- The dedicated pass runs after final optimization and 0006B5 closure, and before every final
  binding, publication, capability, partition, and logical-memory derivation in both compile
  branches.
- Only fully static unresolved Conv2d/Conv3d outputs close to canonical contiguous layout.
- Dynamic/partially dynamic and already resolved descriptors remain unchanged.
- Only direct unresolved `SQUEEZE(axis=2)` outputs consuming a Conv2d result newly closed by the
  pass receive the exact view layout; no fake Conv1d kind or general propagation policy appears.
- Canonical and squeeze-view arithmetic uses checked Model factories and failures retain operation,
  node, value, and arithmetic-cause context.
- IDs, topology, operation instances, constants, bindable Tensor IDs, constraints, phases,
  derivative metadata, publication mapping, CSE/DCE results, and graph order are exact.
- Recording capability-provider tests prove Planning queries receive resolved Conv2d, Conv1d
  squeeze, and Conv3d result descriptors from the final graph; dynamic convolution queries remain
  unresolved and unsupported rather than being silently closed.
- Existing `Conv3dCompilerTest.passesExactPublicationDiagnosticsAndPlanningQueryWithoutProviderClaim`
  compares `query.outputs()` with the final compiled Conv3d node output descriptor, while the
  Model producer descriptor remains unresolved as required by the construction contract.
- The exact three public Engine fixtures pass through typed binding and one-shot compute without
  inward-artifact construction or CPU-internal imports.
- CPU capability remains strict and no Engine/CPU production behavior changes.
- Public/package inventories, dependency declarations, and architecture tests remain unchanged.
- The task stays `Ready` until implementation and the independent documentation pass complete;
  Engine 0008 stays `Blocked` until this task is `Complete` with its positive fixtures.

## Tests and validation

Focused Compiler tests must prove:

- static Conv2d and Conv3d canonical closure, exact descriptor fields, and graph-query visibility;
- the direct Conv2d-to-`SQUEEZE(axis=2)` view rule and visible Conv1d result layout;
- partially dynamic Conv2d/Conv3d and dynamic Conv1d composition remain unresolved;
- unrelated squeeze axes, squeeze inputs not newly closed by the pass, and other layout operations
  remain unchanged;
- already resolved and non-convolution values remain unchanged, including exact input identity
  when nothing qualifies;
- deterministic canonical-layout overflow failure with retained cause, plus checked-factory use
  and contextual failure handling for squeeze-view construction;
- exact preservation of identities, topology, operations, constants, bindable IDs, constraints,
  phases, derivatives, publications, CSE/DCE results, and order; and
- complete compilation builds each `OperationCapabilityQuery` from the final closed descriptors;
  the existing Conv3d query test must assert that final descriptor instead of the unresolved
  Model producer descriptor.

Integration tests must execute the three exact numerical fixtures above through the public Engine
typed lifecycle and `compute`, including reversed supplied input order for reusable runs, detached
materialization, expected Shape/layout, and exact FLOAT32 arrays. No backend-conformance or CPU
test is added because backend semantics and strict capability are unchanged.

Future implementation task-tier validation:

```bash
./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.ConvolutionLogicalLayoutClosureTest --tests io.github.pho001.synaptik.compiler.GraphCompilerTest --tests io.github.pho001.synaptik.compiler.Conv3dCompilerTest
./gradlew :testing:integration-tests:test --tests io.github.pho001.synaptik.testing.integration.EngineConvolutionIntegrationTest
./gradlew :modules:compiler:test
./gradlew :testing:integration-tests:test
./gradlew :modules:compiler:javadoc
git diff --check
```

The clean documentation pass validates Markdown links, anchors, heading uniqueness, fences,
whitespace/final newlines, exact twelve-path scope, package/public inventories, and task/master/
roadmap status and order. Do not rerun successful Java suites in that pass unless it changes Java
or finds a concrete executable discrepancy. Validation tier: affected Compiler module, affected
integration-test suite, Compiler Javadoc, and documentation; no repository-wide or performance
checkpoint is justified.

## Risks

- Closing every static unresolved descriptor would silently create a general default-layout
  policy; eligibility must remain tied to final Conv2d/Conv3d producer occurrences.
- Closing the Conv2d value without its exact direct squeeze would leave the public Conv1d result
  unresolved; propagating beyond that squeeze would broaden the task into unrelated inference.
- Rerunning inference after closure would compare against Model's intentionally unresolved
  convolution descriptor and conflate validation with final descriptor choice.
- Rebuilding graph objects carelessly could change node/operation identity, publications,
  bindable Tensor IDs, derivative graph ownership, or final order even when topology looks equal.
- A fixture using forged descriptors or CPU internals could pass without proving the public
  compile/plan/prepare/run path; the integration allowlist and import boundary must be enforced.
- Treating static logical layout as physical layout could leak backend behavior into Compiler or
  weaken CPU admission; the pass must create no representation or capability fact.
- Marking Engine 0008 Ready before 0006B6 and all three public fixtures complete would recreate
  the checkpoint's original acceptance gap.
- The recorded full Compiler failure shows the pre-existing Conv3d Planning-query test still
  expects the unresolved Model producer descriptor even though the selected closure correctly
  supplies the final canonical contiguous descriptor. That exact stale assertion must be updated,
  so its test path is the required twelfth allowlisted path rather than an implicit scope overrun.

## Dependencies

- Compiler 0006B, 0006B3, 0006B4, and 0006B5 — Complete.
- Current Model Conv1d composition and Conv2d/Conv3d unresolved construction semantics — stable.
- Current Planning capability-query handoff and strict CPU convolution capability — stable.
- Engine 0001–0007 and CPU dimensional-convolution execution — Complete.
- Engine 0008 — Blocked on this task and its public positive fixtures.

This task is a user-authorized prerequisite-order exception: after selection of blocked Engine
0008, work returns to the Compiler plan for one bounded owning prerequisite. Within Compiler it is
inserted in normal order immediately after 0006B5 and before Draft 0006C; it does not reorder or
implement 0006C.

## Documentation-focused clean-context pass

After Java and integration evidence stabilizes, hand the exact diff, commands/results, descriptor
rules, preservation evidence, and baseline scope to a distinct clean documentation-focused
context. Apply the General, API/Javadoc, and Planning profiles. Finalize both changed production
Javadocs and `docs/api/compile-api.md`, synchronize this task, both master plans, Engine 0008, and
the roadmap, and review glossary impact. Record reasoned no-change conclusions for Tensor, Public,
Runtime, and Training API guides; Model/Planning/Prepare/Runtime/Engine/CPU contracts;
architecture/ADRs; architecture and conformance tests; Gradle; and other modules.

## Follow-up tasks

- Engine 0008 becomes `Ready` only after this task is `Complete` and the exact public fixtures are
  passing; its documentation-only repository checkpoint then consumes, rather than recreates,
  that evidence.
- NN convolution integration remains ordered after Engine 0008.
- Compiler 0006C remains the separate Draft Conv3d gradient expressibility boundary.
- Compiler 0007 remains a separate Draft algebra side branch.

## Architecture impact

Expected impact: None. The task fills a final Compiler logical-descriptor fact within existing
ownership and passes it through the existing Planning query boundary. No authoritative contract,
dependency direction, public API, or ADR changes. Stop rather than edit architecture if
implementation evidence contradicts this conclusion.

## Implementation prompt

```text
Work in /Users/phujka/IdeaProjects/Synaptik on Compiler task 0006B6. Do not use GSD, commit, or
push. Use a separate clean implementation context. Read AGENTS.md, ARCHITECTURE.md, the current
architecture plan, planning guide, roadmap, Compiler and Engine master plans, this task, Engine
0008, completed Compiler 0006B/0006B3/0006B4/0006B5, documentation rules/profiles, and every
directly relevant Model/Compiler/Planning/CPU/Engine/integration source and test. Preserve the
dirty worktree.

Implement exactly the twelve-path allowlist. Add the package-private final convolution logical-
layout closure after optimization and published-constant closure. Close only fully static
unresolved Conv2d/Conv3d outputs to LayoutDescriptor.contiguous(shape), then propagate only the
direct newly closed Conv2d -> SQUEEZE(axis=2) result through the existing checked view rule. Keep
dynamic Shapes unresolved. Preserve exact IDs, topology, operation instances, values, constants,
bindings, constraints, phases, derivatives, publications, optimization results, and order. Add no
fake Conv1d operation, general inference mode, physical fact, backend predicate, downstream
production change, dependency, architecture edit, or work on 0006C.

Add the focused negative/preservation/overflow/query tests and the three exact public Engine
numerical fixtures specified by the task. Use only public Tensor/Engine typed binding and compute
in integration tests; do not forge inward artifacts or import CPU internals. In the existing
`Conv3dCompilerTest.passesExactPublicationDiagnosticsAndPlanningQueryWithoutProviderClaim`, update
only the stale query-output expectation to the final compiled Conv3d node output descriptor;
retain the unresolved Model producer contract and all other assertions. Run the specified focused/
module/integration/Javadoc/scope gates once. Then hand stable evidence to a distinct clean
documentation-focused context. Mark 0006B6 Complete and return Engine 0008 to Ready only after all
implementation, public fixtures, documentation, and status gates pass.
```

## Local decisions

- Use a dedicated post-validation final rewrite, not controlled inference, because current
  inference must continue to validate unresolved Model construction exactly.
- Run after 0006B5 closure so one ordered final-descriptor pipeline feeds every downstream
  artifact; the predicates do not overlap.
- Close a fresh convolution result from its fully static output Shape alone. Input capability
  remains independently checked by Planning/backend queries.
- Propagate only direct Conv2d-to-height-squeeze geometry needed by the ordinary public Conv1d
  result. The pass never classifies a sequence as Conv1d.
- Keep CPU rejection of unresolved/dynamic descriptors observable and unchanged.
- Add public cross-module numerical evidence in integration tests while keeping all production
  ownership in Compiler.

## Known limitations

Dynamic and partially dynamic convolution results remain unresolved and therefore are not
admitted by the current CPU provider. Layout propagation beyond the exact direct axis-two squeeze
is not supplied. Conv3d remains forward-only. This task does not add mixed-backend execution,
physical-layout selection, or NN integration.

## Validation evidence

Empty until implementation.

## Implementation notes

Empty until implementation.

## Completion summary

Empty until implementation.
