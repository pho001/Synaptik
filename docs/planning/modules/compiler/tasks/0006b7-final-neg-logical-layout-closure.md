# Task 0006B7: Final NEG Logical-[Layout](../../../../glossary.md#layout) Closure

## Status

Complete

Historical implementation note: task 0006B11 supersedes the dedicated pass and test named below
with an [operation](../../../../glossary.md#operation)-aware static-result closure. NEG remains an explicitly proven materialized-result
family, while affine/[view](../../../../glossary.md#term-view-meanings) results [fail closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation). The following sections intentionally preserve the
0006B7 completion evidence rather than describing the current source tree.

## Goal

Close the one newly proved [Compiler](../../../../glossary.md#term-compiler) prerequisite for [Metal](../../../../glossary.md#term-metal) 0002: after final [validation](../../../../glossary.md#term-validation),
optimization, and the existing final [descriptor](../../../../glossary.md#term-descriptor) closures, assign [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous) logical
[layout](../../../../glossary.md#layout) only to fully static still-unresolved values in these exact roles:

1. the output of an exact [unary](../../../../glossary.md#term-unary) `NEG` [occurrence](../../../../glossary.md#term-occurrence); and
2. a [compile](../../../../glossary.md#compile)-time splat [graph](../../../../glossary.md#graph) input directly consumed by an exact unary `NEG` occurrence.

The package-private pass runs before [publication](../../../../glossary.md#term-publication) [binding](../../../../glossary.md#term-binding), [constant](../../../../glossary.md#term-constant)/bindable source projection,
capability queries, [ownership](../../../../glossary.md#term-ownership-meanings) and partitioning, and logical-memory derivation in both forward-only
and backward-capable compile branches. It preserves [Model](../../../../glossary.md#term-model) construction and
`ElementwiseInference` unresolved-layout semantics and creates no general [elementwise](../../../../glossary.md#term-elementwise) or default
layout policy.

## [Scope](../../../../glossary.md#term-scope)

- Add one package-private stateless final [Compiler](../../../../glossary.md#term-compiler) pass dedicated to exact
  `UnaryElementwiseKind.NEG` occurrences with `NoOperationAttrs.INSTANCE`, one input, and one
  output.
- Invoke it in both [graph](../../../../glossary.md#graph)-stage [compile](../../../../glossary.md#compile) branches after final optimization,
  `PublishedCompileTimeConstantDescriptorClosure`, and `ConvolutionLogicalLayoutClosure`, and
  before either branch returns the final `GraphCompilation` consumed by complete [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names).
- Close an eligible NEG output only when its final descriptor has a fully static Shape and an
  unresolved layout.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [Shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout))</span>
- Close an eligible NEG input only when it is a graph input, has an explicit entry in the
  validated compile-time splat sidecar, is consumed directly by that NEG occurrence, has a fully
  static Shape, and still has unresolved layout.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Use `LayoutDescriptor.contiguous(shape)` for every selected value, including rank-zero scalar
  Shapes and Shapes containing zero extents. This is a logical Compiler fact; Metal 0002 retains
  its stricter positive-shape, rank-<span data-glossary-navigation="true"> ([rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [scalar glossary](../../../../glossary.md#term-scalar); [Metal glossary](../../../../glossary.md#term-metal); [rank glossary](../../../../glossary.md#term-rank))</span>`1..16` execution domain.
- If the same eligible splat feeds multiple NEG occurrences, close its one exact `GraphValue`
  once. Do not require it to be source-only or published.
- Preserve the exact graph topology, node and operation references, IDs, graph input/output
  order, node phases, constants, bindable Tensor identities, constraints, derivative-order
  metadata, publications, and optimization results. Replace only changed <span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [node glossary](../../../../glossary.md#node); [operation glossary](../../../../glossary.md#operation); [constants glossary](../../../../glossary.md#term-constant); [Tensor glossary](../../../../glossary.md#tensor); [derivative glossary](../../../../glossary.md#term-derivative); [metadata glossary](../../../../glossary.md#term-metadata))</span>`GraphValue`
  descriptors, and return the exact input `ValidatedGraph` when nothing qualifies.
- Retain checked layout-arithmetic failure as the cause and add deterministic operation/node/value
  role context.
- Add focused preservation, failure, both-branch, and final capability-query tests.
- Finalize affected Compiler Javadoc and the Compile API in a separate clean documentation pass,
  and synchronize this task, the Compiler and Metal master plans, Metal 0002, and the roadmap when
  implementation completes.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- changing [Model](../../../../glossary.md#term-model) [Tensor](../../../../glossary.md#tensor) construction, `Tensor.neg()`, `ElementwiseInference`, or any other
  inference rule<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
- closing another unary, binary, scalar, activation, comparison, logical, classification, cast,
  layout, convolution, reduction, or structured-operation descriptor<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [scalar glossary](../../../../glossary.md#term-scalar); [activation glossary](../../../../glossary.md#term-activation); [layout glossary](../../../../glossary.md#layout); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [operation glossary](../../../../glossary.md#operation); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>
- a general elementwise propagation rule, default logical-layout rule, rerun of inference, or
  backend-driven Compiler policy<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise); [backend glossary](../../../../glossary.md#backend); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- closing a bindable graph input, an unregistered constant inferred from storage/provenance, a
  splat not directly consumed by NEG, or any dynamic/partially dynamic value<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [constant glossary](../../../../glossary.md#term-constant); [provenance glossary](../../../../glossary.md#provenance))</span>
- physical allocation, constant materialization, upload, backend selection, MPSGraph lowering,
  Runtime/Prepare changes, or a dependency on Metal<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [lowering glossary](../../../../glossary.md#lowering); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [dependency glossary](../../../../glossary.md#term-dependency); [Metal glossary](../../../../glossary.md#term-metal))</span>
- relaxing Metal 0002's positive-extent, non-scalar, FLOAT32, resolved dense-contiguous
  capability domain<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [FLOAT32 glossary](../../../../glossary.md#term-float32); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>
- changing graph structure, adding identity/contiguous operations, or modifying optimization,
  publication, source-role, capability, partition, or memory algorithms<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [publication glossary](../../../../glossary.md#term-publication); [partition glossary](../../../../glossary.md#partition))</span>
- work on Compiler 0006C, Compiler 0007, another Metal operation/type/layout, or public Engine
  Metal composition<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Compiler ownership and the
  compile lifecycle<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [compile glossary](../../../../glossary.md#compile); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Compiler master plan](../master-plan.md)
- [Compiler 0006B5](0006b5-published-compile-time-constant-descriptor-closure.md)
- [Compiler 0006B6](0006b6-final-convolution-logical-layout-closure.md)
- [Metal master plan](../../../backends/metal/master-plan.md)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>
- [Metal 0002](../../../backends/metal/tasks/0002-mpsgraph-prepared-execution-route.md)

## Architecture constraints

- Compiler owns final immutable logical graph descriptors and must complete this fact before
  constructing <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [graph glossary](../../../../glossary.md#graph); [descriptors glossary](../../../../glossary.md#term-descriptor))</span>[`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) values and invoking [backend](../../../../glossary.md#backend)-neutral [Planning](../../../../glossary.md#term-planning).
- [Model](../../../../glossary.md#term-model) continues to express semantic non-[view](../../../../glossary.md#term-view-meanings) [elementwise](../../../../glossary.md#term-elementwise) results with unresolved [layout](../../../../glossary.md#layout).
- Planning sees only final descriptors and occurrence semantics; it gains no constant-provenance
  or Metal-route policy.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [constant glossary](../../../../glossary.md#term-constant); [provenance glossary](../../../../glossary.md#provenance); [Metal glossary](../../../../glossary.md#term-metal))</span>
- Metal remains responsible for its exact capability, constant materialization, lowering,
  physical storage, native execution, and stricter executable geometry.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [lowering glossary](../../../../glossary.md#lowering))</span>
- [Compiler](../../../../glossary.md#term-compiler) must not depend on [Runtime](../../../../glossary.md#term-runtime), [Prepare](../../../../glossary.md#prepare), [Engine](../../../../glossary.md#term-engine), or any [concrete backend](../../../../glossary.md#concrete-backend).
- No architecture authority, dependency direction, public API, or ADR changes are expected. Stop
  if implementation evidence requires one.<span data-glossary-navigation="true"> ([dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>

## Package impact

Existing package used:

- `io.github.pho001.synaptik.compiler` — owns the package-private final graph-descriptor pass and
  its same-package tests.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [descriptor glossary](../../../../glossary.md#term-descriptor))</span>

No package or public type is added.

Type placement:

- `io.github.pho001.synaptik.compiler.NegLogicalLayoutClosure` — package-private final
  transformation beside the existing published-constant and convolution logical-layout closures.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [layout glossary](../../../../glossary.md#layout))</span>

## Affected files

Exact implementation allowlist:

1. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/NegLogicalLayoutClosure.java`
   (new)
2. `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompiler.java`
3. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/NegLogicalLayoutClosureTest.java`
   (new)
4. `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`
5. `docs/api/compile-api.md`
6. this task
7. `docs/planning/modules/compiler/master-plan.md`
8. `docs/planning/backends/metal/tasks/0002-mpsgraph-prepared-execution-route.md`
9. `docs/planning/backends/metal/master-plan.md`
10. `docs/planning/roadmap.md`

Review without editing [Model](../../../../glossary.md#term-model), [Planning](../../../../glossary.md#term-planning), [Prepare](../../../../glossary.md#prepare), [Runtime](../../../../glossary.md#term-runtime), [Engine](../../../../glossary.md#term-engine), [Metal](../../../../glossary.md#term-metal) implementation, [native
code](../../../../glossary.md#native-numerical-library-and-instruction-architecture), [Gradle](../../../../glossary.md#term-gradle), tests outside [Compiler](../../../../glossary.md#term-compiler), architecture/ADRs, glossary, and other documentation. If an
eleventh path is required, stop and replan rather than [widening](../../../../glossary.md#term-widening) [scope](../../../../glossary.md#term-scope) implicitly.

## Maximum [scope](../../../../glossary.md#term-scope)

At most the exact ten paths above: two [Compiler](../../../../glossary.md#term-compiler) production/[Javadoc](../../../../glossary.md#term-javadoc) paths, two Compiler test paths,
the [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api), and five [planning](../../../../glossary.md#term-planning) paths. No executable [module](../../../../glossary.md#term-module) outside Compiler changes.

## Acceptance criteria

- Both compile branches apply the pass after optimization and both existing relevant closures and
  before every final publication, constant/bindable source, capability, ownership, partition, and
  logical-memory derivation.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [publication glossary](../../../../glossary.md#term-publication); [constant glossary](../../../../glossary.md#term-constant); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [partition glossary](../../../../glossary.md#partition))</span>
- Exact [unary](../../../../glossary.md#term-unary) NEG outputs close if and only if they remain unresolved and fully static.
- Exact compile-time splat graph inputs close if and only if they remain unresolved, are fully
  static, and are directly consumed by an exact unary NEG occurrence.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Rank-zero and zero-extent static eligible values close logically; Metal remains free and
  required to reject them under its narrower capability contract.<span data-glossary-navigation="true"> ([Rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [extent glossary](../../../../glossary.md#term-extent); [Metal glossary](../../../../glossary.md#term-metal))</span>
- Dynamic/partially dynamic, already resolved, bindable, non-splat, indirectly consumed,
  non-NEG, wrong-attribute, and malformed/unrelated values remain unchanged.
- [Model](../../../../glossary.md#term-model) construction and `ElementwiseInference` tests continue to observe unresolved unary-result
  semantics.
- The pass creates no operation, node, edge, source role, physical fact, backend fact, or general
  layout policy and preserves all graph sidecars and identities exactly.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [node glossary](../../../../glossary.md#node); [backend glossary](../../../../glossary.md#backend); [layout glossary](../../../../glossary.md#layout))</span>
- Recording capability-provider tests prove final eligible descriptors are resolved before query
  construction in forward-only and backward-capable compilation; ineligible descriptors remain
  unresolved in their final queries.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
- Checked canonical-layout failure identifies whether the NEG input or output failed and retains
  the arithmetic cause.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>
- Focused and full [Compiler](../../../../glossary.md#term-compiler) tests, Compiler [Javadoc](../../../../glossary.md#term-javadoc), Markdown/status/[scope](../../../../glossary.md#term-scope) checks, and
  `git diff --check` pass.
- The task becomes `Complete` only after implementation evidence and the separate clean
  documentation-focused pass are recorded. Metal 0002 then changes from `Blocked` to [`Ready`](../../../../glossary.md#term-ready); it
  must not be marked `Complete` by this task.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused tests must prove:

- static unresolved NEG output closure for ordinary caller inputs;
- static unresolved direct NEG splat-input closure, including shared consumption;
- [scalar](../../../../glossary.md#term-scalar) and zero-[extent](../../../../glossary.md#term-extent) logical closure;
- both forward-only and backward-capable [graph](../../../../glossary.md#graph) branches;
- already resolved [identity](../../../../glossary.md#term-identity-meanings) preservation and exact no-op object [identity](../../../../glossary.md#term-reference-identity) when nothing qualifies;
- dynamic/partially dynamic, bindable, indirect-splat, non-NEG, and unrelated-value preservation;
- exact topology, IDs, operation references, graph boundaries/order, constants, bindable Tensor
  IDs, constraints, phases, derivative metadata, and publications;<span data-glossary-navigation="true"> ([topology glossary](../../../../glossary.md#term-topology); [operation glossary](../../../../glossary.md#operation); [constants glossary](../../../../glossary.md#term-constant); [Tensor glossary](../../../../glossary.md#tensor); [derivative glossary](../../../../glossary.md#term-derivative); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- contextual checked-[layout](../../../../glossary.md#layout) arithmetic failure; and
- final [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) visibility before [ownership](../../../../glossary.md#term-ownership-meanings)/[partition](../../../../glossary.md#partition)/memory [planning](../../../../glossary.md#term-planning).

[Run](../../../../glossary.md#term-run) once after executable Java stabilizes:

```bash
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.NegLogicalLayoutClosureTest \
  --tests io.github.pho001.synaptik.compiler.GraphCompilerTest
./gradlew :modules:compiler:test
```

Documentation-focused pass:

```bash
./gradlew :modules:compiler:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

Validate local Markdown links and anchors, unique effective anchors, balanced fences, final
newlines, trailing whitespace, exact ten-path [scope](../../../../glossary.md#term-scope), package/type placement, no staged files, and
task/master/roadmap/[Metal](../../../../glossary.md#term-metal)-0002 status and ordering. The documentation pass reuses successful Java
test evidence unless it changes executable Java behavior or identifies a concrete reason to rerun.
Repository-wide validation is deferred to [CI](../../../../glossary.md#term-continuous-integration-ci) because this task changes one [module](../../../../glossary.md#term-module) and no shared
contract, [dependency](../../../../glossary.md#term-dependency), or build configuration.

## Dependencies

- [Compiler](../../../../glossary.md#term-compiler) 0006B3, 0006B4, 0006B5, and 0006B6 — Complete.
- Current explicit [compile](../../../../glossary.md#compile)-time splat sidecar and final `ValidatedGraph` contracts — stable.
- Current [Planning](../../../../glossary.md#term-planning) capability-query [handoff](../../../../glossary.md#term-handoff) — stable.
- Metal 0002 independent review — identified this exact prerequisite; this task is now Complete,
  so Metal 0002 is Ready.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>

This is a user-authorized owning-prerequisite ordering exception from the active [Metal](../../../../glossary.md#term-metal) frontier.
Within Compiler it is ordered normally after 0006B6 and before Draft 0006C. It does not implement,
reorder, or otherwise change 0006C or 0007.

## Follow-up tasks

- [Metal](../../../../glossary.md#term-metal) 0002 is [`Ready`](../../../../glossary.md#term-ready) and owns implementation of the exact positive-shape FLOAT32 NEG route,
  including run-owned splat initialization.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [FLOAT32 glossary](../../../../glossary.md#term-float32); [run glossary](../../../../glossary.md#term-run))</span>
- [Compiler](../../../../glossary.md#term-compiler) 0006C and 0007 remain independent Draft side branches.

## Architecture impact

Expected impact: None.

The pass fills one final logical-[descriptor](../../../../glossary.md#term-descriptor) fact inside existing [Compiler](../../../../glossary.md#term-compiler) [ownership](../../../../glossary.md#term-ownership-meanings) and exposes it
through the existing final [graph](../../../../glossary.md#graph) and capability-query boundary. It changes no authoritative
contract, [dependency direction](../../../../glossary.md#java-platform-and-build-tools), public [API](../../../../glossary.md#term-application-programming-interface-api), or [backend](../../../../glossary.md#backend) responsibility. Stop rather than edit
architecture if implementation evidence contradicts this conclusion.

## Implementation prompt

```text
You are the separate clean implementation agent for Synaptik Compiler task 0006B7. Work in
/Users/phujka/IdeaProjects/Synaptik on the existing dirty worktree. Do not use GSD. Do not commit
or push.

Read AGENTS.md, ARCHITECTURE.md, docs/architecture/current-architecture-plan.md,
docs/planning/planning-guide.md, docs/planning/roadmap.md,
docs/planning/modules/compiler/master-plan.md, this task, completed Compiler 0006B5 and 0006B6,
the Metal master plan and Metal 0002, documentation rules/profiles, and all directly relevant
Compiler/Model/Planning source and tests. Preserve unrelated and partial Metal changes.

Implement exactly the ten-path allowlist and this specification. Add the package-private final
NEG logical-layout closure after optimization and existing final closures in both compile
branches. Close only fully static still-unresolved exact NEG outputs and fully static
still-unresolved registered compile-time splat graph inputs directly consumed by exact NEG to
LayoutDescriptor.contiguous(shape). Preserve Model/ElementwiseInference unresolved semantics,
graph structure, identities, every sidecar, scalar/zero-extent logical support, and downstream
ordering. Add no general layout policy, physical/backend work, Metal dependency, or work on
0006C/0007.

Run the specified focused/full Compiler tests. Then hand the stable diff and evidence to a
distinct clean documentation-focused agent/thread. That pass must follow the documentation rules,
apply the General/API-Javadoc/Planning profiles, finalize affected Javadocs and the Compile API,
review glossary impact, synchronize Compiler/Metal/task/roadmap status, validate exact scope and
formatting, and avoid rerunning stable Java tests unless executable behavior changes or a concrete
risk requires it. Stop for an architecture, dependency, public-surface, or scope conflict. Mark
0006B7 Complete and Metal 0002 Ready only after every gate passes; do not mark Metal 0002
Complete.
```

## Local decisions

- Use a dedicated final descriptor transformation, not a new inference mode, so Model and shared
  elementwise inference retain their established unresolved semantics.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [inference glossary](../../../../glossary.md#term-inference); [Model glossary](../../../../glossary.md#term-model); [elementwise glossary](../../../../glossary.md#term-elementwise))</span>
- Run after both existing closures. Their predicates do not overlap this exact NEG occurrence and
  direct-splat rule, and the ordered final-descriptor pipeline remains explicit.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Eligibility follows explicit constant sidecar membership and direct graph consumption only;
  storage, provenance absence, Tensor factory history, and scalar equality never infer a constant.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [graph glossary](../../../../glossary.md#graph); [provenance glossary](../../../../glossary.md#provenance); [Tensor factory glossary](../../../../glossary.md#tensor-factory); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Scalar and zero-extent values are valid logical closure cases even though Metal 0002 rejects
  them at its exact capability boundary.<span data-glossary-navigation="true"> ([extent glossary](../../../../glossary.md#term-extent); [Metal glossary](../../../../glossary.md#term-metal))</span>

## Known limitations

The pass does not resolve another [elementwise](../../../../glossary.md#term-elementwise) result or propagate [layout](../../../../glossary.md#layout) beyond exact NEG. Dynamic
[Shapes](../../../../glossary.md#shape) remain unresolved. It does not materialize [constants](../../../../glossary.md#term-constant) or make [Metal](../../../../glossary.md#term-metal) executable; those remain
Metal 0002 responsibilities after this prerequisite completes.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context completed the exact four Compiler Java paths and reported the initial
  focused command<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
  `./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.NegLogicalLayoutClosureTest --tests io.github.pho001.synaptik.compiler.GraphCompilerTest`
  passing 41 tests with no failures or errors.
- The same implementation context ran `./gradlew :modules:compiler:test` before the independent
  review correction. It passed 281 tests across 40 suites with no failures or errors. No
  post-correction full-module count is claimed.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module))</span>
- The implementation context ran `./gradlew :modules:compiler:javadoc` successfully. The later
  review correction changed only a test, and the documentation pass found both production
  Javadocs accurate without modification, so Javadoc generation was not repeated.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- This mandatory clean documentation-focused completion context independently read the
  architecture contract, documentation rules and selected General/API-Javadoc/Planning profiles,
  planning guide, complete directly affected plans and API reference, and all four final Compiler
  implementation/test paths. It compared the implementation with the acceptance criteria and
  verified the independent review correction that proves the nearest legal wrong-attributes and
  wrong-cardinality variants remain exact no-ops. The corrected<span data-glossary-navigation="true"> ([architecture contract glossary](../../../../glossary.md#architecture-contract); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Planning glossary](../../../../glossary.md#term-planning); [cardinality glossary](../../../../glossary.md#term-cardinality))</span>
  `NegLogicalLayoutClosureTest` rerun passed; this evidence intentionally records no invented
  post-correction aggregate Compiler count.
- The independent review confirmed exact pass ordering in both compile branches: optimization,
  published-constant closure, convolution closure, then NEG closure before <span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [constant glossary](../../../../glossary.md#term-constant); [convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>`GraphCompilation`
  and all publication, source projection, capability-query, ownership, partition, and logical-
  memory work. It also confirmed that only fully static unresolved exact NEG outputs and direct
  explicit-splat graph inputs close, while bindable, indirect, dynamic, resolved, non-NEG, and
  near-match values remain unchanged.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [partition glossary](../../../../glossary.md#partition); [graph glossary](../../../../glossary.md#graph))</span>
- `NegLogicalLayoutClosure` and `GraphCompiler` Javadocs were reviewed in full. They accurately
  document eligibility, exclusions, ordering, preservation, failure context, and the absence of a
  general elementwise policy; no Java change was needed or permitted by the documentation-pass
  allowlist.<span data-glossary-navigation="true"> ([elementwise glossary](../../../../glossary.md#term-elementwise))</span>
- `docs/api/compile-api.md` now documents the third bounded final descriptor closure, its exact
  eligibility and exclusions, both-branch ordering, preservation contract, scalar/zero-extent
  logical behavior, overflow failure, and the distinction from general inference, physical
  state, materialization, and backend capability.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [scalar glossary](../../../../glossary.md#term-scalar); [extent glossary](../../../../glossary.md#term-extent); [overflow glossary](../../../../glossary.md#term-overflow); [inference glossary](../../../../glossary.md#term-inference); [materialization glossary](../../../../glossary.md#materialization); [backend capability glossary](../../../../glossary.md#backend-capability))</span>
- Glossary: no change. The implementation introduces no reusable domain term; `NEG`, logical
  layout, graph input, and compile-time splat already have established meanings, and the bounded
  rule is explained locally in the Compile API.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout))</span>
- Architecture and focused architecture documentation: no change. The pass fills a final logical
  descriptor inside existing Compiler ownership, changes no module responsibility, lifecycle,
  dependency direction, or ADR decision, and remains before backend-neutral Planning.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [backend glossary](../../../../glossary.md#backend))</span>
- Architecture tests: no change because no dependency or module-boundary rule changed.
  Backend-conformance and integration tests: no change because this task adds no backend behavior
  or supported Engine lifecycle. Build configuration: no change because no dependency, plugin,
  toolchain, or task configuration changed. Other modules: no change because Model inference,
  Planning algorithms, Prepare, Runtime, Engine, CPU, Metal implementation, and native code are
  outside this bounded Compiler prerequisite.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [plugin glossary](../../../../glossary.md#term-plugin); [toolchain glossary](../../../../glossary.md#term-toolchain); [Model glossary](../../../../glossary.md#term-model); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [native code glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture))</span>
- Final documentation validation passed: local Markdown targets and heading anchors, unique
  effective anchors, balanced fences, final newlines, trailing whitespace, task/master/roadmap
  status synchronization, exact six-path documentation-pass allowlist, no staged files, and<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
  `git diff --check`. `git diff --cached --check` also passed with no staged diff.

## Implementation notes

- Added package-private stateless `NegLogicalLayoutClosure` and invoked it after the two existing
  final descriptor closures in both graph-stage branches.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [graph glossary](../../../../glossary.md#graph))</span>
- The pass rebuilds only changed `GraphValue` descriptors and the graph-owning immutable
  sidecars, preserves every node and operation reference, and returns the exact input when no
  value qualifies.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [node glossary](../../../../glossary.md#node); [operation glossary](../../../../glossary.md#operation))</span>
- Focused tests cover shared direct splats, scalar and zero extents, bindable and indirect inputs,
  dynamic and resolved descriptors, non-NEG and nearest legal guard variants, exact no-op
  identity, graph sidecars, contextual arithmetic failures, both compile branches, and final
  capability-query visibility.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [guard glossary](../../../../glossary.md#term-guard); [identity glossary](../../../../glossary.md#term-identity-meanings); [compile glossary](../../../../glossary.md#compile))</span>
- The documentation-focused review changed only the six authorized documentation/planning paths.
  All existing Metal implementation, test, native, build, conformance, and architecture-test
  changes were preserved unchanged.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Metal glossary](../../../../glossary.md#term-metal); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## Completion summary

- Completed changes: implemented the bounded final exact-NEG logical-layout closure in both
  compile branches and finalized its API and planning documentation.<span data-glossary-navigation="true"> ([layout glossary](../../../../glossary.md#layout); [compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: the four exact [Compiler](../../../../glossary.md#term-compiler) implementation/test paths plus
  `docs/api/compile-api.md`, this task, the Compiler master plan, Metal task 0002, the Metal master
  plan, and the roadmap.
- Tests and validation: initial focused 41-test pass; pre-correction full Compiler pass of 281
  tests across 40 suites; successful Compiler Javadoc; successful corrected closure-test rerun;
  final Markdown, scope, status, staged-file, newline, fence, anchor, link, and whitespace checks.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation-agent review: completed in this mandatory clean documentation-focused context;
  it independently verified the near-match guard correction and finalized the six documentation/
  planning paths without changing executable behavior.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard))</span>
- Documentation impact: Compile API and synchronized task/master/roadmap/Metal prerequisite
  status updated. Architecture, focused architecture, glossary, backend guides, and other API
  references remain accurate for the reasons recorded above.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>
- [Javadoc](../../../../glossary.md#term-javadoc) review: `NegLogicalLayoutClosure` and `GraphCompiler` are accurate; no change required.
- Glossary impact: none; no reusable term or existing definition changed.
- Unresolved issues: None for Compiler 0006B7. [Metal](../../../../glossary.md#term-metal) 0002 remains unimplemented and uncommitted.
- Follow-up required: implement and validate Metal 0002 as the next [`Ready`](../../../../glossary.md#term-ready) frontier.

Status: Complete
