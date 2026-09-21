# Task 0006B7: Final NEG Logical-Layout Closure

## Status

Complete

## Goal

Close the one newly proved Compiler prerequisite for Metal 0002: after final validation,
optimization, and the existing final descriptor closures, assign canonical contiguous logical
layout only to fully static still-unresolved values in these exact roles:

1. the output of an exact unary `NEG` occurrence; and
2. a compile-time splat graph input directly consumed by an exact unary `NEG` occurrence.

The package-private pass runs before publication binding, constant/bindable source projection,
capability queries, ownership and partitioning, and logical-memory derivation in both forward-only
and backward-capable compile branches. It preserves Model construction and
`ElementwiseInference` unresolved-layout semantics and creates no general elementwise or default
layout policy.

## Scope

- Add one package-private stateless final Compiler pass dedicated to exact
  `UnaryElementwiseKind.NEG` occurrences with `NoOperationAttrs.INSTANCE`, one input, and one
  output.
- Invoke it in both graph-stage compile branches after final optimization,
  `PublishedCompileTimeConstantDescriptorClosure`, and `ConvolutionLogicalLayoutClosure`, and
  before either branch returns the final `GraphCompilation` consumed by complete compilation.
- Close an eligible NEG output only when its final descriptor has a fully static Shape and an
  unresolved layout.
- Close an eligible NEG input only when it is a graph input, has an explicit entry in the
  validated compile-time splat sidecar, is consumed directly by that NEG occurrence, has a fully
  static Shape, and still has unresolved layout.
- Use `LayoutDescriptor.contiguous(shape)` for every selected value, including rank-zero scalar
  Shapes and Shapes containing zero extents. This is a logical Compiler fact; Metal 0002 retains
  its stricter positive-shape, rank-`1..16` execution domain.
- If the same eligible splat feeds multiple NEG occurrences, close its one exact `GraphValue`
  once. Do not require it to be source-only or published.
- Preserve the exact graph topology, node and operation references, IDs, graph input/output
  order, node phases, constants, bindable Tensor identities, constraints, derivative-order
  metadata, publications, and optimization results. Replace only changed `GraphValue`
  descriptors, and return the exact input `ValidatedGraph` when nothing qualifies.
- Retain checked layout-arithmetic failure as the cause and add deterministic operation/node/value
  role context.
- Add focused preservation, failure, both-branch, and final capability-query tests.
- Finalize affected Compiler Javadoc and the Compile API in a separate clean documentation pass,
  and synchronize this task, the Compiler and Metal master plans, Metal 0002, and the roadmap when
  implementation completes.

## Out of scope

- changing Model Tensor construction, `Tensor.neg()`, `ElementwiseInference`, or any other
  inference rule
- closing another unary, binary, scalar, activation, comparison, logical, classification, cast,
  layout, convolution, reduction, or structured-operation descriptor
- a general elementwise propagation rule, default logical-layout rule, rerun of inference, or
  backend-driven Compiler policy
- closing a bindable graph input, an unregistered constant inferred from storage/provenance, a
  splat not directly consumed by NEG, or any dynamic/partially dynamic value
- physical allocation, constant materialization, upload, backend selection, MPSGraph lowering,
  Runtime/Prepare changes, or a dependency on Metal
- relaxing Metal 0002's positive-extent, non-scalar, FLOAT32, resolved dense-contiguous
  capability domain
- changing graph structure, adding identity/contiguous operations, or modifying optimization,
  publication, source-role, capability, partition, or memory algorithms
- work on Compiler 0006C, Compiler 0007, another Metal operation/type/layout, or public Engine
  Metal composition

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially Compiler ownership and the
  compile lifecycle
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Planning guide](../../../planning-guide.md)
- [Compiler master plan](../master-plan.md)
- [Compiler 0006B5](0006b5-published-compile-time-constant-descriptor-closure.md)
- [Compiler 0006B6](0006b6-final-convolution-logical-layout-closure.md)
- [Metal master plan](../../../backends/metal/master-plan.md)
- [Metal 0002](../../../backends/metal/tasks/0002-mpsgraph-prepared-execution-route.md)

## Architecture constraints

- Compiler owns final immutable logical graph descriptors and must complete this fact before
  constructing `OperationCapabilityQuery` values and invoking backend-neutral Planning.
- Model continues to express semantic non-view elementwise results with unresolved layout.
- Planning sees only final descriptors and occurrence semantics; it gains no constant-provenance
  or Metal-route policy.
- Metal remains responsible for its exact capability, constant materialization, lowering,
  physical storage, native execution, and stricter executable geometry.
- Compiler must not depend on Runtime, Prepare, Engine, or any concrete backend.
- No architecture authority, dependency direction, public API, or ADR changes are expected. Stop
  if implementation evidence requires one.

## Package impact

Existing package used:

- `io.github.pho001.synaptik.compiler` — owns the package-private final graph-descriptor pass and
  its same-package tests.

No package or public type is added.

Type placement:

- `io.github.pho001.synaptik.compiler.NegLogicalLayoutClosure` — package-private final
  transformation beside the existing published-constant and convolution logical-layout closures.

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

Review without editing Model, Planning, Prepare, Runtime, Engine, Metal implementation, native
code, Gradle, tests outside Compiler, architecture/ADRs, glossary, and other documentation. If an
eleventh path is required, stop and replan rather than widening scope implicitly.

## Maximum scope

At most the exact ten paths above: two Compiler production/Javadoc paths, two Compiler test paths,
the Compile API, and five planning paths. No executable module outside Compiler changes.

## Acceptance criteria

- Both compile branches apply the pass after optimization and both existing relevant closures and
  before every final publication, constant/bindable source, capability, ownership, partition, and
  logical-memory derivation.
- Exact unary NEG outputs close if and only if they remain unresolved and fully static.
- Exact compile-time splat graph inputs close if and only if they remain unresolved, are fully
  static, and are directly consumed by an exact unary NEG occurrence.
- Rank-zero and zero-extent static eligible values close logically; Metal remains free and
  required to reject them under its narrower capability contract.
- Dynamic/partially dynamic, already resolved, bindable, non-splat, indirectly consumed,
  non-NEG, wrong-attribute, and malformed/unrelated values remain unchanged.
- Model construction and `ElementwiseInference` tests continue to observe unresolved unary-result
  semantics.
- The pass creates no operation, node, edge, source role, physical fact, backend fact, or general
  layout policy and preserves all graph sidecars and identities exactly.
- Recording capability-provider tests prove final eligible descriptors are resolved before query
  construction in forward-only and backward-capable compilation; ineligible descriptors remain
  unresolved in their final queries.
- Checked canonical-layout failure identifies whether the NEG input or output failed and retains
  the arithmetic cause.
- Focused and full Compiler tests, Compiler Javadoc, Markdown/status/scope checks, and
  `git diff --check` pass.
- The task becomes `Complete` only after implementation evidence and the separate clean
  documentation-focused pass are recorded. Metal 0002 then changes from `Blocked` to `Ready`; it
  must not be marked `Complete` by this task.

## Tests / validation

Focused tests must prove:

- static unresolved NEG output closure for ordinary caller inputs;
- static unresolved direct NEG splat-input closure, including shared consumption;
- scalar and zero-extent logical closure;
- both forward-only and backward-capable graph branches;
- already resolved identity preservation and exact no-op object identity when nothing qualifies;
- dynamic/partially dynamic, bindable, indirect-splat, non-NEG, and unrelated-value preservation;
- exact topology, IDs, operation references, graph boundaries/order, constants, bindable Tensor
  IDs, constraints, phases, derivative metadata, and publications;
- contextual checked-layout arithmetic failure; and
- final `OperationCapabilityQuery` visibility before ownership/partition/memory planning.

Run once after executable Java stabilizes:

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
newlines, trailing whitespace, exact ten-path scope, package/type placement, no staged files, and
task/master/roadmap/Metal-0002 status and ordering. The documentation pass reuses successful Java
test evidence unless it changes executable Java behavior or identifies a concrete reason to rerun.
Repository-wide validation is deferred to CI because this task changes one module and no shared
contract, dependency, or build configuration.

## Dependencies

- Compiler 0006B3, 0006B4, 0006B5, and 0006B6 — Complete.
- Current explicit compile-time splat sidecar and final `ValidatedGraph` contracts — stable.
- Current Planning capability-query handoff — stable.
- Metal 0002 independent review — identified this exact prerequisite; this task is now Complete,
  so Metal 0002 is Ready.

This is a user-authorized owning-prerequisite ordering exception from the active Metal frontier.
Within Compiler it is ordered normally after 0006B6 and before Draft 0006C. It does not implement,
reorder, or otherwise change 0006C or 0007.

## Follow-up tasks

- Metal 0002 is `Ready` and owns implementation of the exact positive-shape FLOAT32 NEG route,
  including run-owned splat initialization.
- Compiler 0006C and 0007 remain independent Draft side branches.

## Architecture impact

Expected impact: None.

The pass fills one final logical-descriptor fact inside existing Compiler ownership and exposes it
through the existing final graph and capability-query boundary. It changes no authoritative
contract, dependency direction, public API, or backend responsibility. Stop rather than edit
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
  elementwise inference retain their established unresolved semantics.
- Run after both existing closures. Their predicates do not overlap this exact NEG occurrence and
  direct-splat rule, and the ordered final-descriptor pipeline remains explicit.
- Eligibility follows explicit constant sidecar membership and direct graph consumption only;
  storage, provenance absence, Tensor factory history, and scalar equality never infer a constant.
- Scalar and zero-extent values are valid logical closure cases even though Metal 0002 rejects
  them at its exact capability boundary.

## Known limitations

The pass does not resolve another elementwise result or propagate layout beyond exact NEG. Dynamic
Shapes remain unresolved. It does not materialize constants or make Metal executable; those remain
Metal 0002 responsibilities after this prerequisite completes.

## Validation evidence

- Implementation context completed the exact four Compiler Java paths and reported the initial
  focused command
  `./gradlew :modules:compiler:test --tests io.github.pho001.synaptik.compiler.NegLogicalLayoutClosureTest --tests io.github.pho001.synaptik.compiler.GraphCompilerTest`
  passing 41 tests with no failures or errors.
- The same implementation context ran `./gradlew :modules:compiler:test` before the independent
  review correction. It passed 281 tests across 40 suites with no failures or errors. No
  post-correction full-module count is claimed.
- The implementation context ran `./gradlew :modules:compiler:javadoc` successfully. The later
  review correction changed only a test, and the documentation pass found both production
  Javadocs accurate without modification, so Javadoc generation was not repeated.
- This mandatory clean documentation-focused completion context independently read the
  architecture contract, documentation rules and selected General/API-Javadoc/Planning profiles,
  planning guide, complete directly affected plans and API reference, and all four final Compiler
  implementation/test paths. It compared the implementation with the acceptance criteria and
  verified the independent review correction that proves the nearest legal wrong-attributes and
  wrong-cardinality variants remain exact no-ops. The corrected
  `NegLogicalLayoutClosureTest` rerun passed; this evidence intentionally records no invented
  post-correction aggregate Compiler count.
- The independent review confirmed exact pass ordering in both compile branches: optimization,
  published-constant closure, convolution closure, then NEG closure before `GraphCompilation`
  and all publication, source projection, capability-query, ownership, partition, and logical-
  memory work. It also confirmed that only fully static unresolved exact NEG outputs and direct
  explicit-splat graph inputs close, while bindable, indirect, dynamic, resolved, non-NEG, and
  near-match values remain unchanged.
- `NegLogicalLayoutClosure` and `GraphCompiler` Javadocs were reviewed in full. They accurately
  document eligibility, exclusions, ordering, preservation, failure context, and the absence of a
  general elementwise policy; no Java change was needed or permitted by the documentation-pass
  allowlist.
- `docs/api/compile-api.md` now documents the third bounded final descriptor closure, its exact
  eligibility and exclusions, both-branch ordering, preservation contract, scalar/zero-extent
  logical behavior, overflow failure, and the distinction from general inference, physical
  state, materialization, and backend capability.
- Glossary: no change. The implementation introduces no reusable domain term; `NEG`, logical
  layout, graph input, and compile-time splat already have established meanings, and the bounded
  rule is explained locally in the Compile API.
- Architecture and focused architecture documentation: no change. The pass fills a final logical
  descriptor inside existing Compiler ownership, changes no module responsibility, lifecycle,
  dependency direction, or ADR decision, and remains before backend-neutral Planning.
- Architecture tests: no change because no dependency or module-boundary rule changed.
  Backend-conformance and integration tests: no change because this task adds no backend behavior
  or supported Engine lifecycle. Build configuration: no change because no dependency, plugin,
  toolchain, or task configuration changed. Other modules: no change because Model inference,
  Planning algorithms, Prepare, Runtime, Engine, CPU, Metal implementation, and native code are
  outside this bounded Compiler prerequisite.
- Final documentation validation passed: local Markdown targets and heading anchors, unique
  effective anchors, balanced fences, final newlines, trailing whitespace, task/master/roadmap
  status synchronization, exact six-path documentation-pass allowlist, no staged files, and
  `git diff --check`. `git diff --cached --check` also passed with no staged diff.

## Implementation notes

- Added package-private stateless `NegLogicalLayoutClosure` and invoked it after the two existing
  final descriptor closures in both graph-stage branches.
- The pass rebuilds only changed `GraphValue` descriptors and the graph-owning immutable
  sidecars, preserves every node and operation reference, and returns the exact input when no
  value qualifies.
- Focused tests cover shared direct splats, scalar and zero extents, bindable and indirect inputs,
  dynamic and resolved descriptors, non-NEG and nearest legal guard variants, exact no-op
  identity, graph sidecars, contextual arithmetic failures, both compile branches, and final
  capability-query visibility.
- The documentation-focused review changed only the six authorized documentation/planning paths.
  All existing Metal implementation, test, native, build, conformance, and architecture-test
  changes were preserved unchanged.

## Completion summary

- Completed changes: implemented the bounded final exact-NEG logical-layout closure in both
  compile branches and finalized its API and planning documentation.
- Files changed or created: the four exact Compiler implementation/test paths plus
  `docs/api/compile-api.md`, this task, the Compiler master plan, Metal task 0002, the Metal master
  plan, and the roadmap.
- Tests and validation: initial focused 41-test pass; pre-correction full Compiler pass of 281
  tests across 40 suites; successful Compiler Javadoc; successful corrected closure-test rerun;
  final Markdown, scope, status, staged-file, newline, fence, anchor, link, and whitespace checks.
- Documentation-agent review: completed in this mandatory clean documentation-focused context;
  it independently verified the near-match guard correction and finalized the six documentation/
  planning paths without changing executable behavior.
- Documentation impact: Compile API and synchronized task/master/roadmap/Metal prerequisite
  status updated. Architecture, focused architecture, glossary, backend guides, and other API
  references remain accurate for the reasons recorded above.
- Javadoc review: `NegLogicalLayoutClosure` and `GraphCompiler` are accurate; no change required.
- Glossary impact: none; no reusable term or existing definition changed.
- Unresolved issues: None for Compiler 0006B7. Metal 0002 remains unimplemented and uncommitted.
- Follow-up required: implement and validate Metal 0002 as the next `Ready` frontier.

Status: Complete
