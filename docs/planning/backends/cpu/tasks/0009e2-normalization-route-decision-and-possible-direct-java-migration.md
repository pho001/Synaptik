# Task 0009E2: Normalization Route Decision and Possible Direct-Java Migration

## Status

Complete

## Goal

Decide whether the complete current CPU trailing Layer/RMS-normalization route remains bounded
generated execution or becomes finite direct typed Java. Select only the route with demonstrably
lower total implementation-and-verification cost; this is not a benchmark or performance task.

## Scope

Decide exactly one first-class, fully static `LAYER_NORM` or `RMS_NORM` node admitted today:
Layer no-affine `[input]`, affine Layer `[input, scale, bias]`, RMS `[input]`, or scaled RMS
`[input, scale]`, all with positive trailing `normalizedShape`. Preserve the current
`BFLOAT16`/`FLOAT32`/`FLOAT64` promotion/result/epsilon boundary, gradient eligibility,
positive-rank static Shape, matching affine/scale Shape, resolved non-negative layouts, and
distinct injective output.

Preserve canonical leading-slice and trailing-domain traversal; dense-linear/general-odometer
geometry; non-injective input reads/injective output writes; ordered first-occurrence boundary
deduplication; typed heap-array/native-order `MemorySegment` and ordered mixed carriers; checked
spans/access/alignment/non-overlap; scalar or disjoint caller-parallel complete-slice ranges;
immutable invocation; and one final typed store per output. Empty leading or normalized extent
remains a no-read/no-write/no-workspace/no-invocation/no-submission output.

Preserve the exact existing numerical algorithms and special-value behavior. Layer remains its
three-pass finite/nonconstant mean, corrected compensated centered-square, then standardize and
optional affine body; it retains finite-constant positive-zero standardization, NaN/infinity slice
propagation, epsilon inside the root, only negative roundoff-residue correction, exact-mean state,
and result-format operation/narrowing boundaries. RMS remains its two-pass scaled-square then
divide/optional-scale body, including epsilon/root construction and its NaN/infinity,
overflow/underflow, and signed-zero behavior. Layer retains exactly one disjoint max-size
`AGGREGATE_EXACT_STATE` slice per simultaneously used range; RMS retains zero workspace.

Cold-reject before output mutation or worker submission every currently excluded semantic,
lowering, binding, and invocation fact: not exactly one admitted first-class occurrence;
decomposed graph; wrong kind/attrs/arity/value identity/type/result/epsilon/gradient/Shape/rank/
trailing normalized Shape/layout; unsupported carrier, inaccessible/undersized/misaligned span,
negative/overflowing geometry, non-injective output, output/input overlap, invalid range/workers/
resources, or unsupported empty/nonempty resource arrangement. Confirm the current boundary from
code before implementation; do not add a finite-input rejection policy or any silent fallback.

## Out of scope

`BATCH_NORM_INFERENCE` and `BATCH_NORM_TRAINING` (later CPU 0009F), loss, softmax, attention,
ordinary reductions, partial/combine protocols, vectors, native routes, fusion, decomposed-kind
recognition, Model/Compiler semantics or gradients, public/shared Prepare/Runtime, resources,
architecture/modules/build, benchmarks, and performance claims. Do not create E3, F, or G tasks.
Legacy is read-only behavioral context only.

## Architecture references

Read [architecture](../../../../../ARCHITECTURE.md), [current architecture plan](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md), [documentation rules](../../../../developer-guide/documentation-rules.md), [CPU master plan](../master-plan.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [E1](0009e1-partial-integral-reduction-direct-java-migration.md) through [E1C](0009e1c-softmax-style-reduction-route-decision.md), and completed [0007F](0007f-portable-layer-and-rms-normalization-coverage.md).

## Architecture constraints

`ARCHITECTURE.md` is authoritative. CPU Prepare owns lowering, finite cold route selection,
resource declaration, and post-slot finalization; Runtime invokes immutable prepared work only.
Keep borrowed carriers, isolated `RunState`, prepared-region ownership, and cold validation
before writes/submission. No public/shared API, generic normalization executor, resource kind,
dependency, architecture, module, or build change: stop and report any need for one.

## Package impact

Existing CPU-private packages only: `internal.ir` and `internal.lowering` retain normalization
identity and geometry; `internal.codegen.emit` retains both emitters if generation remains;
`internal.prepare`, `internal.executable`, `internal.cache`, and `route.portable` retain
cold selection, validation, artifact/binding, finalization, and invocation; `internal.reference`
retains the independent clean-Java oracle. Direct Java may add narrowly named Layer/RMS private
executable owners, never a generic service or public type.

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.ir`
- `io.github.pho001.synaptik.backend.cpu.internal.lowering`
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`
- `io.github.pho001.synaptik.backend.cpu.internal.prepare`
- `io.github.pho001.synaptik.backend.cpu.internal.executable`
- `io.github.pho001.synaptik.backend.cpu.internal.cache`
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable`
- `io.github.pho001.synaptik.backend.cpu.internal.reference`

Packages added or changed: None.

Type placement:

- Retained generation adds no production type.
- If direct Java wins the strict cost rule, add only
  `internal.executable.CpuLayerNormDirectExecutable` and
  `internal.executable.CpuRmsNormDirectExecutable`; each owns one finite typed family and no
  generic normalization dispatch.

## Affected files

Expected planning paths are this task plus the CPU master plan, parent CPU 0009, and roadmap.
Review-only seams: capability; trailing-normalization IR/lowering; both emitters/classfile
generator; exact-state access; preparation/finalization/executable; portable route; artifact/cache;
reference oracle; generated coverage inventory/evidence; and focused tests.

## Maximum scope

This task may create or modify at most 32 paths: 16 production/Javadoc, 12 test/resource, and
four planning paths; at most two narrowly named CPU-private direct executable types. A 33rd path,
third type, shared/public/resource/build/architecture/conformance/integration change,
materialization, partial/combine state, or generic element dispatch is a stop-and-replan
condition.

## Acceptance criteria

- The final decision covers all four Layer/RMS forms and every current admitted/rejected fact by
  total implementation-plus-verification cost, without a benchmark gate or performance claim.
- The selected route preserves exact algorithms, epsilon, axes/trailing shape, affine/scale
  behavior, types, dense/general geometry, carrier matrix, complete-slice ownership, workspace,
  immutable invocation, cold no-write failure, and final store shape.
- Direct Java, if selected, is finite typed cold-selected carrier/geometry code with immutable
  range-owning calls and no per-element type/carrier/layout/kind dispatch, allocation, boxing,
  reflection, map/string lookup, or avoidable virtual semantic dispatch. It fully retires every
  selected generated emitter/classfile, selection, artifact/cache, invocation, inventory, and
  evidence route.
- Retention records typed carrier/layout, Layer exact-state/RMS zero-workspace, numerical,
  validator/binding, artifact/cache, invocation, oracle, and retirement costs that make replacement
  larger. It retains an optimal clean-Java oracle with matching pass/dataflow/store shape.
- Retention includes generated Class-File and hot-loop comparison with that oracle, demonstrating
  no hidden Synaptik helper call or avoidable hot-path allocation, boxing, reflection, generic, or
  virtual semantic dispatch. Literal-bytecode, JIT, and performance assertions are not required.
- Focused tests cover forms/types, dense/general layouts, all carrier pairings, axes/ranks,
  empty/nonempty geometry, numerical/special/narrowing/one-store behavior, workspace, ranges,
  canaries/no-write failure, lowering/prepare/finalization/invocation/artifact/cache/reference,
  and selected-route retirement or retention inventory.
- A separate clean documentation-focused context finalizes Javadocs, explanatory/planning status,
  glossary impact, and documentation validation in the same change.

## Route-selection decision rule

First establish that either candidate preserves the complete Scope boundary; an unsupported
candidate is not comparable. Record an auditable non-performance comparison: (1) retained
generation's emitter/Class-File inspection, typed carrier/layout binding, Layer workspace/RMS
zero-workspace, validator, artifact/cache, invocation, oracle, and focused-evidence maintenance;
(2) direct Java's finite typed implementation and cold selection/binding; and (3) complete
generated-route retirement plus semantic, rejection, invocation, artifact/cache, inventory,
oracle, and hot-loop verification. Select direct Java only when (2) plus (3) is demonstrably
strictly lower than (1). Equal, uncertain, incomplete, or non-auditable evidence retains
generation. Record facts, buckets, comparison, selected route, and retirement consequence in
validation evidence. This rule supplies no metric, benchmark, or performance claim.

## Tests / validation

After executable stabilization run the focused current normalization-owner tests: capability and
package inventory; trailing-normalization IR/lowering; Layer/RMS generated or replacement
direct-kernel tests; normalization semantic-closure/evidence/coverage inventory; reference; and
affected preparer/finalizer/prepared-executable/artifact-cache tests. Record exact commands and XML
counts; add retirement tests before deletion. Do not run a benchmark or repository-wide suite.

The documentation context reuses stable Java evidence unless it changes executable Java; it checks
links, anchors, fences, terminology/glossary, exact scope, dependencies/status/frontier, then runs:

```bash
git diff --check
git status --short -uall
```

Architecture, conformance, integration, and repository validation defer to 0009G/CI because no
shared or end-to-end boundary may change.

## Dependencies

- E1 through E1C are Complete retained-generated-route decisions.
- Complete 0007F supplies the semantic identity, numerical algorithms, static geometry, resource
  boundary, emitters, binding, artifact/cache, and independent oracle.

## Follow-up tasks

- E3 loss remains Draft and follows E2.
- Batch normalization remains exclusively under Draft 0009F; no E2 child is authorized.

## Architecture impact

Expected impact: None. Stop and report any public, dependency, resource, module, or architecture
conflict.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009E2. Do not commit, push, stage,
reset, revert, delete, or modify unrelated work; do not use GSD. Read AGENTS.md,
ARCHITECTURE.md, the current architecture plan, Planning Guide, CPU master plan, CPU 0009,
E1/E1A/E1B1/E1B2/E1B3/E1C/0007F, this task, current normalization source/tests, documentation
rules, and General/Planning profiles. Implement exactly this route decision. Choose direct finite
typed Java only when its complete implementation and verification cost is strictly lower;
otherwise retain generation with concrete evidence and Class-File/hot-loop comparison to the
clean-Java oracle. Preserve the confirmed boundary; stop on conflict. Hand the stabilized diff and
exact test evidence to a separate clean documentation context. No benchmarks/repository-wide
tests/commit/push/stage.
```

## Rationale

Layer and RMS share static trailing-slice lowering and cold multi-carrier binding but have distinct
pass counts, special-value behavior, and workspace. Their existing dedicated emitters and oracle
make one bounded route decision coherent without absorbing batch normalization or adjacent
reduction families.

## Local decisions

Retain the complete bounded generated CPU Layer/RMS-normalization route. Direct Java is selected
only when its implementation, verification, and generated-route-retirement cost is demonstrably
strictly lower than retention; equality, uncertainty, or incomplete evidence retains generation.
The current evidence does not satisfy that strict condition. Layer retains one disjoint
exact-state workspace slice per simultaneously used range; RMS retains zero workspace. Batch
normalization remains exclusively later CPU 0009F.

## Known limitations

No speed, vector, scaling, bytecode-identity, JIT, or performance conclusion follows. Dynamic
Shapes/layouts, non-first-class/decomposed forms, partial/combine execution, native execution, and
batch normalization remain excluded.

## Validation evidence

The implementation context recorded this evidence before the separate documentation context
finalized the task. Inspection of the current `CpuTrailingNormalizationIr` and
`CpuTrailingNormalizationLowering` confirms the complete E2 boundary: exactly one first-class
`LAYER_NORM` or `RMS_NORM` occurrence, the four ordered forms, supported floating promotion and
typed positive finite epsilon, positive static rank and matching trailing normalized Shape,
resolved non-negative layouts, an injective distinct output, checked spans, and ordered
first-occurrence input-boundary deduplication. The lowerer emits complete leading-slice ranges;
an empty leading or normalized extent emits no range and no Layer scratch. It neither recognizes
decompositions nor adds an input-finiteness policy. Batch normalization is not inspected as an
E2 candidate and remains 0009F.

The three non-performance cost buckets retain generated execution for the complete Layer and RMS
families. (1) Retention already has two dedicated finite typed emitters, cold-specialized heap,
native-order `MemorySegment`, and mixed-carrier entries, packed dense/general geometry, immutable
prepared invocation, generated-artifact identity/cache, and the independent
`CpuTrailingNormalizationReferenceKernel`. Layer additionally has its disjoint per-active-range
exact-state slice; RMS declares zero workspace. (2) A direct replacement would need two new
finite typed executable owners, each reproducing its distinct numerical passes, type/store
boundaries, all carrier/layout combinations, cold validators, and immutable range bindings.
(3) It would then also need complete retirement proof and changes to generator selection,
artifact/cache identity, finalization, invocation, inventory, generated evidence, semantic and
rejection tests, and the oracle comparison. Buckets (2) plus (3) are not demonstrably strictly
smaller than (1); under the task rule, the retained generated route is selected. This is neither
a benchmark nor a performance or JIT conclusion.

Source and actual generated-Class-File test inspection supports that retained decision.
`CpuLayerNormEmitter` emits the three-pass exact-mean, corrected centered-square, standardize,
optional-affine body with its one final typed store; `CpuRmsNormEmitter` emits the two-pass
scaled-square/root, optional-scale body with one final typed store. The clean-Java oracle has the
same family-specific pass/dataflow/store shape while remaining independent of lowering geometry,
generated invocation data, and Layer exact-state storage. The focused generated-artifact tests
construct and parse actual class bytes for every form and result type: entries are typed static,
have no fields, and contain no Synaptik member reference; the RMS evidence confirms `Math.hypot`.
The cross-family Class-File closure test further rejects fields, object-typed entry arguments,
bootstrap methods, method handles, dynamic constants, reflection/invoke/collection members, and
unexpected Synaptik helper ownership. Together with the emitter inspection this establishes no
hidden Synaptik semantic helper call, allocation, boxing, reflection, map/string/generic dispatch,
or virtual semantic dispatch in the generated hot body. It does not assert literal Class-File
identity, JIT behavior, or performance.

The implementation context ran only the focused normalization owners on 2026-09-09:

```bash
./gradlew :backends:cpu:test --rerun-tasks \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.CpuInternalPackageInventoryTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuTrailingNormalizationLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuLayerNormGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuRmsNormGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuNormalizationSemanticClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuTrailingNormalizationEvidenceTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedDirectEvidenceClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedCoverageEvidenceTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuTrailingNormalizationReferenceTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelPersistenceEvidenceTest
```

The 16 selected XML reports record 169 tests, zero failures, zero errors, and one expected
opt-in persistence-evidence skip. They cover capability/inventory, IR/lowering admission,
all forms/types, dense/general and heap/segment/mixed carriers, numerical and special-value
classes, range/canary/no-write behavior, Layer scratch/RMS zero workspace, preparation,
finalization, immutable invocation, artifact/cache, reference, and generated evidence. No
benchmark or repository-wide suite ran.

## Implementation notes

Retain the existing bounded generated route without executable changes. The Layer and RMS routes
remain separate numerical owners but one cohesive E2 decision: their retained cold binding and
artifact mechanics are shared, while Layer exact-state isolation and RMS zero-workspace make a
full direct replacement and generated-route retirement more costly than the already-auditable
generated route. Documentation finalization, including any final planning-status change, remains
outside this implementation context.

## Completion summary

- Completed changes: Finalized the retained generated-route decision for all four current
  first-class Layer/RMS forms; no executable Java change was warranted. Updated this task, the CPU
  master plan, parent CPU 0009, and roadmap so E2 is Complete and E3 is the sole next Draft
  summary frontier.
- Files changed: `docs/planning/backends/cpu/tasks/0009e2-normalization-route-decision-and-possible-direct-java-migration.md`,
  `docs/planning/backends/cpu/master-plan.md`,
  `docs/planning/backends/cpu/tasks/0009-portable-generated-coverage-closure-checkpoint.md`, and
  `docs/planning/roadmap.md`. No production, test, Javadoc, glossary, explanatory, architecture,
  build, architecture-test, backend-conformance, or integration-test path changed.
- Validation: Reused the implementation context's focused 16-suite CPU result recorded on
  2026-09-09: 169 tests, zero failures, zero errors, and one expected opt-in
  persistence-evidence skip. Independently inspected the current IR/lowering, both emitters,
  reference oracle, generated-artifact structural tests, and their XML reports; checked the four
  planning records for status/frontier consistency, local links/anchors, code fences,
  terminology, final newlines, exact paths, `git diff --check`, and `git status --short -uall`.
  No Java suite was rerun because no executable Java changed and the recorded evidence was
  available and consistent with the inspected source/tests.
- Documentation/Javadoc review: Applied the General and Planning profiles. Existing CPU-private
  Javadocs accurately describe the retained emitters, lowering, and independent oracle; no public
  API or behavior changed. Explanatory documentation and the glossary already define Layer/RMS
  normalization, exact-state workspace, and epsilon accurately, so no change was required.
  Architecture and architecture tests remain accurate because no boundary changed;
  backend-conformance and integration coverage remain deferred to CPU 0009G/CI because no
  cross-backend or end-to-end contract changed.
- Unresolved issues: None for E2. CPU 0009 remains Ready and incomplete.
- Follow-up required: E3 loss is the sole next Draft summary frontier; no detailed E3 task is
  created by this decision. Batch normalization remains under Draft CPU 0009F.

Status: Complete
