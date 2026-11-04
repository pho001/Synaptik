# Task 0009E2: [Normalization](../../../../glossary.md#term-normalization) Route Decision and Possible Direct-Java Migration

## Reader context

This task records retention of generated execution for four first-class trailing Layer/RMS forms,
not a normalization performance result. `CpuTrailingNormalizationIr` and
`CpuTrailingNormalizationLowering` define admitted identities and complete-slice geometry;
`CpuLayerNormEmitter` and `CpuRmsNormEmitter` own distinct numerical bodies.
`CpuTrailingNormalizationReferenceKernel` is the independent semantic oracle. Generated Class-File
tests supply structural evidence about entry signatures and forbidden dependencies, not numerical
correctness by themselves.

Here, CPU Prepare owns cold selection, validation, resource declaration, and post-slot finalization;
Runtime invokes immutable prepared work. Typed arrays and native-order MemorySegments are invocation
storage, while Shape/layout facts describe admitted geometry. Layer exact-state scratch belongs to
simultaneously active run ranges, not persistent model state; RMS declares none. Direct Java would
also require retiring artifact/cache, invocation, and evidence machinery.

Current-topic navigation (not a replacement for the historical record):

- [current trailing Layer-normalization form](../../../../reference/tensor/normalization.md#layer-normalization-expressions) — Explains normalized Shape, ordered inputs, epsilon, and centered population normalization.
- [current RMS-normalization form](../../../../reference/tensor/normalization.md#rms-normalization-expressions) — Contrasts uncentered mean square and optional scale with Layer normalization.

## Status

Complete

## Goal

Decide whether the complete current [CPU](../../../../glossary.md#term-cpu) trailing Layer/RMS-[normalization](../../../../glossary.md#term-normalization) route remains bounded
generated execution or becomes [finite](../../../../glossary.md#term-finite) direct typed Java. Select only the route with demonstrably
lower total implementation-and-verification cost; this is not a benchmark or performance task.

## [Scope](../../../../glossary.md#term-scope)

Decide exactly one first-class, fully static `LAYER_NORM` or `RMS_NORM` [node](../../../../glossary.md#node) admitted today:
Layer no-affine `[input]`, affine Layer `[input, scale, bias]`, RMS `[input]`, or scaled RMS
`[input, scale]`, all with positive trailing `normalizedShape`. Preserve the current
[`BFLOAT16`](../../../../glossary.md#term-bfloat16)/[`FLOAT32`](../../../../glossary.md#term-float32)/[`FLOAT64`](../../../../glossary.md#term-float64) promotion/result/[epsilon](../../../../glossary.md#epsilon) boundary, [gradient](../../../../glossary.md#term-gradient) eligibility,
positive-[rank](../../../../glossary.md#term-rank) static [Shape](../../../../glossary.md#shape), matching affine/scale Shape, resolved non-negative [layouts](../../../../glossary.md#layout), and
distinct [injective](../../../../glossary.md#term-injective) output.

Preserve [canonical](../../../../glossary.md#term-canonical-form) leading-[slice](../../../../glossary.md#slice) and trailing-domain traversal; dense-[linear](../../../../glossary.md#term-linear-meanings)/general-odometer
geometry; non-injective input reads/injective output writes; ordered first-[occurrence](../../../../glossary.md#term-occurrence) boundary
deduplication; typed [heap](../../../../glossary.md#term-heap)-array/native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment) and ordered mixed carriers; checked
spans/access/[alignment](../../../../glossary.md#term-alignment)/non-overlap; [scalar](../../../../glossary.md#term-scalar) or disjoint caller-parallel complete-slice ranges;
[immutable](../../../../glossary.md#term-immutable) invocation; and one final typed store per output. Empty leading or normalized [extent](../../../../glossary.md#term-extent)
remains a no-read/no-write/no-workspace/no-invocation/no-submission output.

Preserve the exact existing numerical algorithms and special-value behavior. Layer remains its
three-pass [finite](../../../../glossary.md#term-finite)/nonconstant [mean](../../../../glossary.md#term-mean), corrected compensated centered-square, then standardize and
optional affine body; it retains finite-[constant](../../../../glossary.md#term-constant) positive-zero standardization, [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity) slice
propagation, epsilon inside the root, only negative roundoff-residue correction, exact-mean state,
and result-format [operation](../../../../glossary.md#operation)/[narrowing](../../../../glossary.md#term-narrowing) boundaries. RMS remains its two-pass scaled-square then
divide/optional-scale body, including epsilon/root construction and its NaN/infinity,
[overflow](../../../../glossary.md#term-overflow)/[underflow](../../../../glossary.md#term-underflow), and [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero) behavior. Layer retains exactly one disjoint max-size
`AGGREGATE_EXACT_STATE` slice per simultaneously used range; RMS retains zero workspace.

Cold-reject before output mutation or worker submission every currently excluded semantic,
[lowering](../../../../glossary.md#lowering), [binding](../../../../glossary.md#term-binding), and invocation fact: not exactly one admitted first-class occurrence;
decomposed [graph](../../../../glossary.md#graph); wrong kind/attrs/[arity](../../../../glossary.md#term-arity)/value [identity](../../../../glossary.md#term-identity-meanings)/type/result/epsilon/gradient/Shape/rank/
trailing [normalized Shape](../../../../glossary.md#normalized-shape)/layout; unsupported [carrier](../../../../glossary.md#term-storage-carrier), inaccessible/undersized/misaligned span,
negative/overflowing geometry, non-injective output, output/input overlap, invalid range/workers/
resources, or unsupported empty/nonempty resource arrangement. Confirm the current boundary from
code before implementation; do not add a finite-input rejection policy or any silent fallback.

## Out of [scope](../../../../glossary.md#term-scope)

`BATCH_NORM_INFERENCE` and `BATCH_NORM_TRAINING` (later [CPU](../../../../glossary.md#term-cpu) 0009F), [loss](../../../../glossary.md#loss), [softmax](../../../../glossary.md#softmax--log-softmax), attention,
ordinary reductions, partial/combine protocols, [vectors](../../../../glossary.md#term-vector), native routes, [fusion](../../../../glossary.md#term-fusion), decomposed-kind
recognition, [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) semantics or [gradients](../../../../glossary.md#term-gradient), public/shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime), resources,
architecture/[modules](../../../../glossary.md#term-module)/build, benchmarks, and performance claims. Do not create E3, F, or G tasks.
Legacy is read-only behavioral context only.

## Architecture references

Read [architecture](../../../../../ARCHITECTURE.md), [current architecture plan](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>, [documentation rules](../../../../developer-guide/documentation-rules.md), [CPU master plan](../master-plan.md)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>, parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [E1](0009e1-partial-integral-reduction-direct-java-migration.md) through [E1C](0009e1c-softmax-style-reduction-route-decision.md), and completed [0007F](0007f-portable-layer-and-rms-normalization-coverage.md).

## Architecture constraints

`ARCHITECTURE.md` is authoritative. [CPU](../../../../glossary.md#term-cpu) [Prepare](../../../../glossary.md#prepare) owns [lowering](../../../../glossary.md#lowering), [finite](../../../../glossary.md#term-finite) cold route selection,
resource declaration, and post-slot finalization; [Runtime](../../../../glossary.md#term-runtime) invokes [immutable](../../../../glossary.md#term-immutable) prepared work only.
Keep [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) carriers, isolated [`RunState`](../../../../glossary.md#run-state--runstate), prepared-region [ownership](../../../../glossary.md#term-ownership-meanings), and cold [validation](../../../../glossary.md#term-validation)
before writes/submission. No public/shared [API](../../../../glossary.md#term-application-programming-interface-api), generic [normalization](../../../../glossary.md#term-normalization) executor, resource kind,
[dependency](../../../../glossary.md#term-dependency), architecture, [module](../../../../glossary.md#term-module), or build change: stop and report any need for one.

## Package impact

Existing [CPU](../../../../glossary.md#term-cpu)-private packages only: `internal.ir` and `internal.lowering` retain [normalization](../../../../glossary.md#term-normalization)
[identity](../../../../glossary.md#term-identity-meanings) and geometry; `internal.codegen.emit` retains both emitters if generation remains;
`internal.prepare`, `internal.executable`, `internal.cache`, and `route.portable` retain
cold selection, [validation](../../../../glossary.md#term-validation), [artifact](../../../../glossary.md#term-artifact)/[binding](../../../../glossary.md#term-binding), finalization, and invocation; `internal.reference`
retains the independent clean-Java [oracle](../../../../glossary.md#term-test-oracle). Direct Java may add narrowly named Layer/RMS private
executable [owners](../../../../glossary.md#term-owner), never a generic service or public type.

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
  generic normalization dispatch.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>

## Affected files

Expected [planning](../../../../glossary.md#term-planning) paths are this task plus the [CPU](../../../../glossary.md#term-cpu) master plan, parent CPU 0009, and roadmap.
Review-only seams: capability; trailing-[normalization](../../../../glossary.md#term-normalization) [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering); both emitters/classfile
generator; exact-state access; preparation/finalization/executable; portable route; [artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache);
reference [oracle](../../../../glossary.md#term-test-oracle); generated coverage inventory/evidence; and focused tests.

## Maximum [scope](../../../../glossary.md#term-scope)

This task may create or modify at most 32 paths: 16 production/[Javadoc](../../../../glossary.md#term-javadoc), 12 test/resource, and
four [planning](../../../../glossary.md#term-planning) paths; at most two narrowly named [CPU](../../../../glossary.md#term-cpu)-private direct executable types. A 33rd path,
third type, shared/public/resource/build/architecture/[conformance](../../../../glossary.md#term-conformance)/integration change,
[materialization](../../../../glossary.md#materialization), partial/combine state, or generic [element](../../../../glossary.md#term-element) [dispatch](../../../../glossary.md#term-dispatch) is a stop-and-replan
condition.

## Acceptance criteria

- The final decision covers all four Layer/RMS forms and every current admitted/rejected fact by
  total implementation-plus-verification cost, without a benchmark gate or performance claim.
- The selected route preserves exact algorithms, epsilon, axes/trailing shape, affine/scale
  behavior, types, dense/general geometry, carrier matrix, complete-slice ownership, workspace,
  immutable invocation, cold no-write failure, and final store shape.<span data-glossary-navigation="true"> ([epsilon glossary](../../../../glossary.md#epsilon); [axes glossary](../../../../glossary.md#term-axis); [shape glossary](../../../../glossary.md#shape); [carrier glossary](../../../../glossary.md#term-storage-carrier); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [slice glossary](../../../../glossary.md#slice); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- Direct Java, if selected, is finite typed cold-selected carrier/geometry code with immutable
  range-owning calls and no per-element type/carrier/layout/kind dispatch, allocation, boxing,
  reflection, map/string lookup, or avoidable virtual semantic dispatch. It fully retires every
  selected generated emitter/classfile, selection, artifact/cache, invocation, inventory, and
  evidence route.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [element glossary](../../../../glossary.md#term-element); [layout glossary](../../../../glossary.md#layout); [dispatch glossary](../../../../glossary.md#term-dispatch); [lookup glossary](../../../../glossary.md#term-lookup); [artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache))</span>
- Retention records typed carrier/layout, Layer exact-state/RMS zero-workspace, numerical,
  validator/binding, artifact/cache, invocation, oracle, and retirement costs that make replacement
  larger. It retains an optimal clean-Java oracle with matching pass/dataflow/store shape.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [oracle glossary](../../../../glossary.md#term-test-oracle); [dataflow glossary](../../../../glossary.md#term-dataflow))</span>
- Retention includes generated Class-File and hot-loop comparison with that oracle, demonstrating
  no hidden Synaptik helper call or avoidable hot-path allocation, boxing, reflection, generic, or
  virtual semantic dispatch. Literal-bytecode, JIT, and performance assertions are not required.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit))</span>
- Focused tests cover forms/types, dense/general layouts, all carrier pairings, axes/ranks,
  empty/nonempty geometry, numerical/special/narrowing/one-store behavior, workspace, ranges,
  canaries/no-write failure, lowering/prepare/finalization/invocation/artifact/cache/reference,
  and selected-route retirement or retention inventory.<span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing); [lowering glossary](../../../../glossary.md#lowering); [prepare glossary](../../../../glossary.md#prepare))</span>
- A separate clean documentation-focused context finalizes Javadocs, explanatory/planning status,
  glossary impact, and documentation validation in the same change.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [validation glossary](../../../../glossary.md#term-validation))</span>

## Route-selection decision rule

First establish that either candidate preserves the complete [Scope](../../../../glossary.md#term-scope) boundary; an unsupported
candidate is not comparable. Record an auditable non-performance comparison: (1) retained
generation's emitter/Class-File inspection, typed [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout) [binding](../../../../glossary.md#term-binding), Layer workspace/RMS
zero-workspace, validator, [artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache), invocation, [oracle](../../../../glossary.md#term-test-oracle), and focused-evidence maintenance;
(2) direct Java's [finite](../../../../glossary.md#term-finite) typed implementation and cold selection/binding; and (3) complete
generated-route retirement plus semantic, rejection, invocation, artifact/cache, inventory,
oracle, and hot-loop verification. Select direct Java only when (2) plus (3) is demonstrably
strictly lower than (1). Equal, uncertain, incomplete, or non-auditable evidence retains
generation. Record facts, buckets, comparison, selected route, and retirement consequence in
[validation](../../../../glossary.md#term-validation) evidence. This rule supplies no [metric](../../../../glossary.md#term-metric), benchmark, or performance claim.

## Tests / [validation](../../../../glossary.md#term-validation)

After executable stabilization [run](../../../../glossary.md#term-run) the focused current [normalization](../../../../glossary.md#term-normalization)-[owner](../../../../glossary.md#term-owner) tests: capability and
package inventory; trailing-normalization [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering); Layer/RMS generated or replacement
direct-[kernel](../../../../glossary.md#kernel) tests; normalization semantic-closure/evidence/coverage inventory; reference; and
affected preparer/finalizer/prepared-executable/[artifact](../../../../glossary.md#term-artifact)-[cache](../../../../glossary.md#term-cache) tests. Record exact commands and XML
counts; add retirement tests before deletion. Do not run a benchmark or repository-wide suite.

The documentation context reuses stable Java evidence unless it changes executable Java; it checks
links, anchors, fences, terminology/glossary, exact [scope](../../../../glossary.md#term-scope), dependencies/status/frontier, then runs:

```bash
git diff --check
git status --short -uall
```

Architecture, [conformance](../../../../glossary.md#term-conformance), integration, and repository validation defer to 0009G/[CI](../../../../glossary.md#term-continuous-integration-ci) because no
shared or end-to-end boundary may change.

## Dependencies

- E1 through E1C are Complete retained-generated-route decisions.
- Complete 0007F supplies the semantic identity, numerical algorithms, static geometry, resource
  boundary, emitters, binding, artifact/cache, and independent oracle.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [binding glossary](../../../../glossary.md#term-binding); [artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>

## Follow-up tasks

- E3 [loss](../../../../glossary.md#loss) remains Draft and follows E2.
- [Batch normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training) remains exclusively under Draft 0009F; no E2 child is authorized.

## Architecture impact

Expected impact: None. Stop and report any public, [dependency](../../../../glossary.md#term-dependency), resource, [module](../../../../glossary.md#term-module), or architecture
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

Layer and RMS share static trailing-[slice](../../../../glossary.md#slice) [lowering](../../../../glossary.md#lowering) and cold multi-[carrier](../../../../glossary.md#term-storage-carrier) [binding](../../../../glossary.md#term-binding) but have distinct
pass counts, special-value behavior, and workspace. Their existing dedicated emitters and [oracle](../../../../glossary.md#term-test-oracle)
make one bounded route decision coherent without absorbing [batch normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training) or adjacent
reduction families.

## Local decisions

Retain the complete bounded generated [CPU](../../../../glossary.md#term-cpu) Layer/RMS-[normalization](../../../../glossary.md#term-normalization) route. Direct Java is selected
only when its implementation, verification, and generated-route-retirement cost is demonstrably
strictly lower than retention; equality, uncertainty, or incomplete evidence retains generation.
The current evidence does not satisfy that strict condition. Layer retains one disjoint
exact-state workspace [slice](../../../../glossary.md#slice) per simultaneously used range; RMS retains zero workspace. [Batch
normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training) remains exclusively later CPU 0009F.

## Known limitations

No speed, [vector](../../../../glossary.md#term-vector), scaling, [bytecode](../../../../glossary.md#term-bytecode)-[identity](../../../../glossary.md#term-identity-meanings), [JIT](../../../../glossary.md#term-just-in-time-compilation-jit), or performance conclusion follows. Dynamic
[Shapes](../../../../glossary.md#shape)/[layouts](../../../../glossary.md#layout), non-first-class/decomposed forms, partial/combine execution, native execution, and
[batch normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training) remain excluded.

## [Validation](../../../../glossary.md#term-validation) evidence

The implementation context recorded this evidence before the separate documentation context
finalized the task. Inspection of the current `CpuTrailingNormalizationIr` and
`CpuTrailingNormalizationLowering` confirms the complete E2 boundary: exactly one first-class
`LAYER_NORM` or `RMS_NORM` [occurrence](../../../../glossary.md#term-occurrence), the four ordered forms, supported floating promotion and
typed positive [finite](../../../../glossary.md#term-finite) [epsilon](../../../../glossary.md#epsilon), positive static [rank](../../../../glossary.md#term-rank) and matching trailing [normalized Shape](../../../../glossary.md#normalized-shape),
resolved non-negative [layouts](../../../../glossary.md#layout), an [injective](../../../../glossary.md#term-injective) distinct output, checked spans, and ordered
first-occurrence input-boundary deduplication. The lowerer emits complete leading-[slice](../../../../glossary.md#slice) ranges;
an empty leading or normalized [extent](../../../../glossary.md#term-extent) emits no range and no Layer scratch. It neither recognizes
decompositions nor adds an input-finiteness policy. [Batch normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training) is not inspected as an
E2 candidate and remains 0009F.

The three non-performance cost buckets retain generated execution for the complete Layer and RMS
families. (1) Retention already has two dedicated finite typed emitters, cold-specialized [heap](../../../../glossary.md#term-heap),
native-order [`MemorySegment`](../../../../glossary.md#term-memorysegment), and mixed-[carrier](../../../../glossary.md#term-storage-carrier) entries, packed dense/general geometry, [immutable](../../../../glossary.md#term-immutable)
prepared invocation, generated-[artifact](../../../../glossary.md#term-artifact) [identity](../../../../glossary.md#term-identity-meanings)/[cache](../../../../glossary.md#term-cache), and the independent
`CpuTrailingNormalizationReferenceKernel`. Layer additionally has its disjoint per-active-range
exact-state slice; RMS declares zero workspace. (2) A direct replacement would need two new
finite typed executable [owners](../../../../glossary.md#term-owner), each reproducing its distinct numerical passes, type/store
boundaries, all carrier/layout combinations, cold validators, and immutable range [bindings](../../../../glossary.md#term-binding).
(3) It would then also need complete retirement proof and changes to generator selection,
artifact/cache identity, finalization, invocation, inventory, generated evidence, semantic and
rejection tests, and the [oracle](../../../../glossary.md#term-test-oracle) comparison. Buckets (2) plus (3) are not demonstrably strictly
smaller than (1); under the task rule, the retained generated route is selected. This is neither
a benchmark nor a performance or [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) conclusion.

Source and actual generated-Class-File test inspection supports that retained decision.
`CpuLayerNormEmitter` emits the three-pass exact-[mean](../../../../glossary.md#term-mean), corrected centered-square, standardize,
optional-affine body with its one final typed store; `CpuRmsNormEmitter` emits the two-pass
scaled-square/root, optional-scale body with one final typed store. The clean-Java oracle has the
same family-specific pass/[dataflow](../../../../glossary.md#term-dataflow)/store [shape](../../../../glossary.md#shape) while remaining independent of [lowering](../../../../glossary.md#lowering) geometry,
generated invocation data, and Layer exact-state storage. The focused generated-artifact tests
construct and parse actual class bytes for every form and result type: entries are typed static,
have no fields, and contain no Synaptik member reference; the RMS evidence confirms `Math.hypot`.
The cross-family Class-File closure test further rejects fields, object-typed entry [arguments](../../../../glossary.md#term-argument),
bootstrap methods, method handles, dynamic [constants](../../../../glossary.md#term-constant), reflection/invoke/collection members, and
unexpected Synaptik helper [ownership](../../../../glossary.md#term-ownership-meanings). Together with the emitter inspection this establishes no
hidden Synaptik semantic helper call, allocation, boxing, reflection, map/string/generic [dispatch](../../../../glossary.md#term-dispatch),
or virtual semantic dispatch in the generated hot body. It does not assert literal Class-File
identity, JIT behavior, or performance.

The implementation context ran only the focused [normalization](../../../../glossary.md#term-normalization) owners on 2026-09-09:

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
opt-in [persistence](../../../../glossary.md#term-persistence)-evidence skip. They cover capability/inventory, [IR](../../../../glossary.md#term-intermediate-representation-ir)/lowering [admission](../../../../glossary.md#term-admission),
all forms/types, dense/general and heap/segment/mixed carriers, numerical and special-value
classes, range/[canary](../../../../glossary.md#term-canary)/no-write behavior, Layer scratch/RMS zero workspace, preparation,
finalization, immutable invocation, artifact/cache, reference, and generated evidence. No
benchmark or repository-wide suite ran.

## Implementation notes

Retain the existing bounded generated route without executable changes. The Layer and RMS routes
remain separate numerical [owners](../../../../glossary.md#term-owner) but one cohesive E2 decision: their retained cold [binding](../../../../glossary.md#term-binding) and
[artifact](../../../../glossary.md#term-artifact) mechanics are shared, while Layer exact-state isolation and RMS zero-workspace make a
full direct replacement and generated-route retirement more costly than the already-auditable
generated route. Documentation finalization, including any final [planning](../../../../glossary.md#term-planning)-status change, remains
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
  build, architecture-test, backend-conformance, or integration-test path changed.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Validation: Reused the implementation context's focused 16-suite CPU result recorded on
  2026-09-09: 169 tests, zero failures, zero errors, and one expected opt-in
  persistence-evidence skip. Independently inspected the current IR/lowering, both emitters,
  reference oracle, generated-artifact structural tests, and their XML reports; checked the four
  planning records for status/frontier consistency, local links/anchors, code fences,
  terminology, final newlines, exact paths, <span data-glossary-navigation="true"> ([Validation glossary](../../../../glossary.md#term-validation); [persistence glossary](../../../../glossary.md#term-persistence); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [oracle glossary](../../../../glossary.md#term-test-oracle); [artifact glossary](../../../../glossary.md#term-artifact); [planning glossary](../../../../glossary.md#term-planning))</span>`git diff --check`, and `git status --short -uall`.
  No Java suite was rerun because no executable Java changed and the recorded evidence was
  available and consistent with the inspected source/tests.
- Documentation/Javadoc review: Applied the General and Planning profiles. Existing CPU-private
  Javadocs accurately describe the retained emitters, lowering, and independent oracle; no public
  API or behavior changed. Explanatory documentation and the glossary already define Layer/RMS
  normalization, exact-state workspace, and epsilon accurately, so no change was required.
  Architecture and architecture tests remain accurate because no boundary changed;
  backend-conformance and integration coverage remain deferred to CPU 0009G/CI because no
  cross-backend or end-to-end contract changed.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [RMS normalization glossary](../../../../glossary.md#rms-normalization); [epsilon glossary](../../../../glossary.md#epsilon); [CI glossary](../../../../glossary.md#term-continuous-integration-ci))</span>
- Unresolved issues: None for E2. [CPU](../../../../glossary.md#term-cpu) 0009 remains [Ready](../../../../glossary.md#term-ready) and incomplete.
- Follow-up required: E3 loss is the sole next Draft summary frontier; no detailed E3 task is
  created by this decision. Batch normalization remains under Draft CPU 0009F.<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [Batch normalization glossary](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training))</span>

Status: Complete
