# Task 0009E1C: [Softmax](../../../../glossary.md#softmax--log-softmax)-Style Reduction Route Decision

## Reader context

This task records retaining bounded generated SOFTMAX and LOG_SOFTMAX for first-class
shape-preserving nodes, not recognizing decomposed graphs. `CpuSoftmaxIr` identifies the operation;
lowering fixes complete normalization slices, and `CpuSoftmaxEmitter` specializes the three-pass
body. `CpuSoftmaxInputValidator` enforces finite represented inputs and shifts before output
mutation or worker submission, rather than introducing a NaN/infinity result policy.

Here, CPU analysis selects structural heap/segment carrier and layout forms; finalization constructs
an immutable prepared executable recipe. Runtime later calls the CPU binder on current run storage,
creating a per-run bound invocation for complete slices with zero workspace. Generated Class-File and
hot-loop checks are structural oracles; `CpuSoftmaxReferenceKernel` independently checks stable numerical dataflow. The recorded cost rule includes direct implementation and full
generated-route retirement, not a speed comparison.

Current-topic navigation (not a replacement for the historical record):

- [current first-class softmax expression meaning](../../../../reference/tensor/normalization.md#softmax-expressions) — Separates metadata construction and ideal slice meaning from backend-specific admission and execution.
- [backend ownership versus implementation route](../../../../design/notes/cpu-kernel-strategy.md#strategy) — Separates Planning ownership, CPU route choice, finalization, and Runtime invocation.

## Status

Complete

## Goal

Decide whether the complete [CPU](../../../../glossary.md#term-cpu) [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax)/`LOG_SOFTMAX` route remains bounded generated execution or becomes [finite](../../../../glossary.md#term-finite) direct typed Java, choosing only the lower total implementation-and-verification-cost route. No benchmark, performance, or numerical redesign is authorized.

## [Scope](../../../../glossary.md#term-scope)

Decide the complete current first-class one-input/one-output `SoftmaxKind.SOFTMAX` and `SoftmaxKind.LOG_SOFTMAX` family with `SoftmaxAttrs`: identical [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), or [`BFLOAT16`](../../../../glossary.md#term-bfloat16) input/result types and [gradient](../../../../glossary.md#term-gradient) eligibility; normalized in-[rank](../../../../glossary.md#term-rank) positive-width [axis](../../../../glossary.md#term-axis); result [Shape](../../../../glossary.md#shape) equal to input Shape. A [normalization](../../../../glossary.md#term-normalization) [slice](../../../../glossary.md#slice) fixes all non-selected [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix).

Preserve static resolved non-negative [layouts](../../../../glossary.md#layout), dense-[linear](../../../../glossary.md#term-linear-meanings)/general-odometer geometry, non-[injective](../../../../glossary.md#term-injective) input reads/injective output writes, [heap](../../../../glossary.md#term-heap)-array/[`MemorySegment`](../../../../glossary.md#term-memorysegment)/ordered mixed carriers, checked span/access/[alignment](../../../../glossary.md#term-alignment)/non-overlap, [scalar](../../../../glossary.md#term-scalar) or disjoint caller-parallel ranges owning complete slices, [immutable](../../../../glossary.md#term-immutable) prepared [binding](../../../../glossary.md#term-binding), zero scratch/workspace, and one final typed store per output. Preserve stable three-pass maximum, compensated shifted-exponential sum, final result, and represented [narrowing](../../../../glossary.md#term-narrowing): [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax) divides exponentials by sum; `LOG_SOFTMAX` subtracts its logarithm. Preserve empty non-selected-[extent](../../../../glossary.md#term-extent) no-op behavior, [finite](../../../../glossary.md#term-finite) input/shift [validation](../../../../glossary.md#term-validation) before write or submission, and rejection rather than a new [NaN](../../../../glossary.md#term-nan)/[infinity](../../../../glossary.md#term-infinity) policy.

Cold-reject without output mutation or submitted work every current excluded fact: not exactly one supported first-class [node](../../../../glossary.md#node); decomposed [graph](../../../../glossary.md#graph); wrong kind/attrs/[arity](../../../../glossary.md#term-arity)/value [identity](../../../../glossary.md#term-identity-meanings)/type/gradient eligibility/Shape/rank/axis/layout; non-floating/non-identical type; unresolved, negative, overflowing, inaccessible, undersized, or misaligned layout/[carrier](../../../../glossary.md#term-storage-carrier); non-injective output; zero selected extent; input/output overlap; invalid bases/range/workers/resources; non-finite represented input or maximum shift; and scratch/workspace request. Confirm this code boundary before implementation; no silent fallback.

## Out of [scope](../../../../glossary.md#term-scope)

Layer/RMS/[batch normalization](../../../../glossary.md#batch-normalization--batch-normalization-inference-and-training), [loss](../../../../glossary.md#loss), attention, other reductions, partial/combine execution, [vector](../../../../glossary.md#term-vector)/native routes, [fusion](../../../../glossary.md#term-fusion) or decomposed-[softmax](../../../../glossary.md#softmax--log-softmax) recognition, [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) semantics or [gradients](../../../../glossary.md#term-gradient), public/shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime), architecture/[modules](../../../../glossary.md#term-module)/build, benchmarks, and performance claims. Do not create detailed E2, E3, F, or G tasks. Legacy is read-only evidence only.

## Architecture references and constraints

Read [architecture](../../../../../ARCHITECTURE.md), [current architecture plan](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>, [documentation rules](../../../../developer-guide/documentation-rules.md), [CPU master plan](../master-plan.md), parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [E1](0009e1-partial-integral-reduction-direct-java-migration.md), [E1A](0009e1a-masked-reduction-route-decision.md), [E1B1](0009e1b1-log-sum-exp-route-decision.md), [E1B2](0009e1b2-statistical-reduction-route-decision.md), [E1B3](0009e1b3-norm-reduction-route-decision.md), and [0007E](0007e-portable-stable-softmax-and-log-softmax-coverage.md).

`ARCHITECTURE.md` is authoritative. [CPU](../../../../glossary.md#term-cpu) [Prepare](../../../../glossary.md#prepare) owns [lowering](../../../../glossary.md#lowering), [finite](../../../../glossary.md#term-finite) route selection, resource declaration, and finalization after slot assignment; [Runtime](../../../../glossary.md#term-runtime) invokes [immutable](../../../../glossary.md#term-immutable) prepared work only. Preserve [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) carriers, isolated [`RunState`](../../../../glossary.md#run-state--runstate), cold [validation](../../../../glossary.md#term-validation) before writes/submission, and prepared-region [ownership](../../../../glossary.md#term-ownership-meanings). No public/shared [API](../../../../glossary.md#term-application-programming-interface-api), generic [normalization](../../../../glossary.md#term-normalization) executor, resource kind, [dependency](../../../../glossary.md#term-dependency), architecture, or build change; stop on a need for one.

## Package impact

Existing [CPU](../../../../glossary.md#term-cpu)-private packages only: `internal.ir`/`internal.lowering` retain [identity](../../../../glossary.md#term-identity-meanings)/geometry; `internal.codegen.emit` retains `CpuSoftmaxEmitter` if generated; `internal.prepare`, `internal.executable`, `internal.cache`, and `route.portable` retain selection, validator, [artifact](../../../../glossary.md#term-artifact)/[binding](../../../../glossary.md#term-binding), finalization, and invocation; `internal.reference` retains the independent clean-Java [oracle](../../../../glossary.md#term-test-oracle). Retention adds no type. Direct Java may add one narrowly named CPU-private [softmax](../../../../glossary.md#softmax--log-softmax) executable, never a generic service.

## Affected files and maximum [scope](../../../../glossary.md#term-scope)

Expected [planning](../../../../glossary.md#term-planning) paths are this task plus the three synchronized records. Review-only seams: capability, `CpuSoftmaxIr`, [lowering](../../../../glossary.md#lowering), emitter/classfile generator, validator, [prepare](../../../../glossary.md#prepare)/finalization/executable, route, [artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache), reference [oracle](../../../../glossary.md#term-test-oracle), and focused tests. At most 28 paths (14 production/[Javadoc](../../../../glossary.md#term-javadoc), ten test/resource, four planning) and one new [CPU](../../../../glossary.md#term-cpu)-private type. A 29th path, second type, shared/public/resource/build/architecture/[conformance](../../../../glossary.md#term-conformance)/integration change, [materialization](../../../../glossary.md#materialization), partial state, or generic [element](../../../../glossary.md#term-element) [dispatch](../../../../glossary.md#term-dispatch) is stop-and-replan.

## Acceptance criteria

- The final decision covers the complete [matrix](../../../../glossary.md#term-matrix-meanings) by total implementation plus verification cost, without benchmark gate or performance claim.
- It preserves both kinds/types, [Shape](../../../../glossary.md#shape)/[axis](../../../../glossary.md#term-axis)/[layout](../../../../glossary.md#layout) [admission](../../../../glossary.md#term-admission), dense/general geometry, [heap](../../../../glossary.md#term-heap)/segment/mixed carriers, [scalar](../../../../glossary.md#term-scalar)/disjoint complete-[slice](../../../../glossary.md#slice) [ownership](../../../../glossary.md#term-ownership-meanings), zero workspace, [immutable](../../../../glossary.md#term-immutable) invocation, cold no-write failure, and one store.
- It preserves stable three-pass/compensation/[narrowing](../../../../glossary.md#term-narrowing), empty non-selected [extent](../../../../glossary.md#term-extent), [finite](../../../../glossary.md#term-finite)-input/shift rejection, and special-value boundary.
- Direct Java is finite typed cold-selected [carrier](../../../../glossary.md#term-storage-carrier)/geometry code with immutable range-owning calls; it has no generic per-[element](../../../../glossary.md#term-element) carrier/type/layout/kind [dispatch](../../../../glossary.md#term-dispatch), allocation, boxing, reflection, map/string, or avoidable virtual semantic dispatch; it fully retires generated selection, [artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache), invocation, inventory/evidence, and references.
- Retention records typed carrier/layout, numerical, validator, [binding](../../../../glossary.md#term-binding), artifact/cache, invocation, [oracle](../../../../glossary.md#term-test-oracle), and retirement costs making replacement larger, and retains an optimal clean-Java oracle with matching stable [dataflow](../../../../glossary.md#term-dataflow)/store shape.
- Retention requires generated Class-File/hot-loop inspection against that oracle, proving no hidden Synaptik helper calls or avoidable allocations, boxing, reflection, or generic/virtual semantic dispatch. No literal-[bytecode](../../../../glossary.md#term-bytecode), [JIT](../../../../glossary.md#term-just-in-time-compilation-jit), or performance assertion is required.
- Focused tests cover kinds/types, dense/general, carrier pairings, axis positions, empty/nonempty geometry, finite/narrowing/special values/one-store, range/[canary](../../../../glossary.md#term-canary)/no-write failures, [lowering](../../../../glossary.md#lowering)/validator/[prepare](../../../../glossary.md#prepare)/finalization/invocation/artifact/cache/reference/inventory/retirement.
- A separate clean documentation-focused context finalizes Javadocs, explanatory/[planning](../../../../glossary.md#term-planning) status, glossary impact, and documentation [validation](../../../../glossary.md#term-validation).

## Route-selection decision rule

First establish that either candidate preserves every [Scope](../../../../glossary.md#term-scope) semantic and support boundary; an
unsupported candidate is not comparable. Record a bounded, auditable cost comparison with these
non-performance buckets: (1) generated retention's emitter/Class-File inspection, typed
[carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout) [binding](../../../../glossary.md#term-binding), validator, [artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache), invocation, [oracle](../../../../glossary.md#term-test-oracle), and focused-evidence
maintenance; (2) direct Java's [finite](../../../../glossary.md#term-finite) typed implementation and cold selection/binding work; and
(3) direct Java's complete generated-route retirement plus semantic, rejection, invocation,
artifact/cache, inventory, oracle, and hot-loop verification work. Select direct Java only when
the documented sum of buckets (2) and (3) is demonstrably strictly lower than bucket (1).
Equal, uncertain, incomplete, or non-auditable evidence retains generation. Record the candidate
facts, each bucket, comparison, selected route, and retirement consequence in [validation](../../../../glossary.md#term-validation) evidence.
This rule supplies neither a performance [metric](../../../../glossary.md#term-metric) nor a benchmark or performance claim.

## Tests / [validation](../../../../glossary.md#term-validation)

After executable stabilization [run](../../../../glossary.md#term-run) focused current [softmax](../../../../glossary.md#softmax--log-softmax)-[owner](../../../../glossary.md#term-owner) tests: capability, package inventory, softmax [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/generated/input-validator/reference, generated [matrix](../../../../glossary.md#term-matrix-meanings)/direct-evidence/coverage inventory, and affected preparer/finalizer/prepared-executable/[artifact](../../../../glossary.md#term-artifact)-[cache](../../../../glossary.md#term-cache) tests. Record exact commands/XML counts; add retirement tests before deletion. No benchmark or repository-wide suite.

Documentation context reuses stable Java evidence unless it changes executable Java; it validates links, anchors, fences, terminology/glossary, exact [scope](../../../../glossary.md#term-scope), dependencies/status/frontier, then runs:

```bash
git diff --check
git status --short -uall
```

Architecture, [conformance](../../../../glossary.md#term-conformance), integration, and repository validation defer to 0009G/[CI](../../../../glossary.md#term-continuous-integration-ci) because no shared/end-to-end boundary may change.

## Dependencies and follow-up tasks

- E1, E1A, E1B1, E1B2, and E1B3 are Complete retained-generated-route decisions.
- Complete 0007E supplies semantic [identity](../../../../glossary.md#term-identity-meanings), [stable algorithm](../../../../glossary.md#conditioning-and-numerical-stability), geometry, validator, emitter, [artifact](../../../../glossary.md#term-artifact)/[binding](../../../../glossary.md#term-binding), and reference [oracle](../../../../glossary.md#term-test-oracle).
- E2, E3, F, and G remain Draft summaries only. E2 follows this decision; do not create it here.

## Architecture impact

Expected impact: None. Stop and report any public, [dependency](../../../../glossary.md#term-dependency), resource, [module](../../../../glossary.md#term-module), or architecture conflict.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009E1C. Do not commit, push, stage, reset, revert, delete, or modify unrelated work; do not use GSD. Read AGENTS.md, ARCHITECTURE.md, current architecture plan, Planning Guide, CPU master plan, CPU 0009, E1/E1A/E1B1/E1B2/E1B3/0007E, this task, current softmax source/tests, documentation rules, and General/Planning profiles. Implement exactly this route decision. Choose direct finite typed Java only if total implementation and verification cost is lower; otherwise retain generation with concrete evidence and Class-File/hot-loop comparison to the clean-Java oracle. Preserve the confirmed boundary; stop on conflict. Hand the stabilized diff and exact test evidence to a separate clean documentation context. No benchmarks/repository-wide tests/commit/push/stage.
```

## Rationale

Both kinds share complete-[slice](../../../../glossary.md#slice) geometry, cold [carrier](../../../../glossary.md#term-storage-carrier)/[binding](../../../../glossary.md#term-binding), [finite](../../../../glossary.md#term-finite) prevalidation, and three stable passes; only the final division versus logarithm/subtraction differs. This follows advanced reductions and precedes [normalization](../../../../glossary.md#term-normalization) without authorizing either adjacent family.

## Local decisions

Empty until implemented.

## Known limitations

No speed, [vector](../../../../glossary.md#term-vector), scaling, [bytecode](../../../../glossary.md#term-bytecode)-parity, or performance conclusion follows. Dynamic [Shapes](../../../../glossary.md#shape)/[layouts](../../../../glossary.md#layout), zero selected width, non-[finite](../../../../glossary.md#term-finite) input, decomposed [graphs](../../../../glossary.md#graph), [fusion](../../../../glossary.md#term-fusion), [materialization](../../../../glossary.md#materialization), partial/combine, and native execution remain excluded.

## [Validation](../../../../glossary.md#term-validation) evidence for [planning](../../../../glossary.md#term-planning)

Before creating this [Ready](../../../../glossary.md#term-ready) task, the planning context inspected required architecture/planning/documentation records; 0007E and E1--E1B3; [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) [softmax](../../../../glossary.md#softmax--log-softmax) references; `CpuCapabilityProvider`, `CpuSoftmaxIr`, [lowering](../../../../glossary.md#lowering), emitter, validator, reference [oracle](../../../../glossary.md#term-test-oracle), preparation/finalization/invocation, [artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache), and focused evidence tests. It confirmed the [Scope](../../../../glossary.md#term-scope) rejection boundary and no architecture conflict. This planning-only change runs no [Gradle](../../../../glossary.md#term-gradle) or benchmark.

## Implementation notes

Draft implementation decision: retain the bounded generated route unchanged for both
`SoftmaxKind.SOFTMAX` and `SoftmaxKind.LOG_SOFTMAX`.

The confirmed admitted [matrix](../../../../glossary.md#term-matrix-meanings) is exactly one first-class [shape](../../../../glossary.md#shape)-preserving [node](../../../../glossary.md#node) with
`SoftmaxAttrs`, identical [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), or [`BFLOAT16`](../../../../glossary.md#term-bfloat16) boundaries, a [normalized axis](../../../../glossary.md#normalized-axis) with
positive selected width, static non-negative resolved [layouts](../../../../glossary.md#layout), non-[injective](../../../../glossary.md#term-injective) input reads and
injective output writes, [heap](../../../../glossary.md#term-heap)/segment/ordered-mixed carriers, and [scalar](../../../../glossary.md#term-scalar) or disjoint
complete-[slice](../../../../glossary.md#slice) parallel ranges. It has zero workspace and [immutable](../../../../glossary.md#term-immutable) invocation [binding](../../../../glossary.md#term-binding). An empty
non-selected [extent](../../../../glossary.md#term-extent) is a no-op. Cold [lowering](../../../../glossary.md#lowering), [carrier](../../../../glossary.md#term-storage-carrier)/span/[alignment](../../../../glossary.md#term-alignment)/overlap checks, range and
worker [validation](../../../../glossary.md#term-validation), and `CpuSoftmaxInputValidator` reject every excluded semantic or binding fact,
including non-[finite](../../../../glossary.md#term-finite) represented inputs and maximum shifts, before output mutation or worker
submission.

This satisfies the task's comparable-candidate prerequisite: the existing route preserves the
three full slice passes (maximum, compensated shifted-exponential sum, final result), represented
[narrowing](../../../../glossary.md#term-narrowing), one final typed store per output, and the distinct final forms (division for [softmax](../../../../glossary.md#softmax--log-softmax);
one logarithm then subtraction for [log-softmax](../../../../glossary.md#softmax--log-softmax)). A direct candidate would need to preserve all of
those facts before it could be compared.

Auditable non-performance cost comparison:

1. Retained-generation maintenance is bounded: `CpuSoftmaxEmitter` already cold-specializes the
   kind, three represented types, typed heap/segment/mixed carriers, and dense/general geometry;
   `CpuSoftmaxInputValidator`, lowering geometry, immutable generated binding, artifact/cache,
   and <span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache))</span>`CpuSoftmaxReferenceKernel` already own validation, invocation, and independent-oracle
   evidence.
2. Direct Java would require a new finite CPU-private owner plus cold typed carrier and
   dense/general selection/binding for both kinds and all three representations, while duplicating
   the stable three-pass/Kahan/narrowing/store bodies and preserving the no-write rejection
   boundary.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
3. Direct retirement would additionally require removing generated selection and proving the
   replacement across artifact/cache identity, finalization, immutable invocation, package and
   coverage inventory, oracle, rejection/canary, and generated-route retirement evidence.<span data-glossary-navigation="true"> ([canary glossary](../../../../glossary.md#term-canary))</span>

Buckets 2 plus 3 are not demonstrably strictly lower than bucket 1; they are materially larger
because all currently verified finite route and retirement seams would need replacement. The rule
therefore selects retention. This is an implementation-and-verification-cost decision only, not a
benchmark or performance claim.

Generated-route inspection found a generation-time kind-specialized, typed static entry with the
same three-pass [dataflow](../../../../glossary.md#term-dataflow) and final-store shape as the independent clean-Java [oracle](../../../../glossary.md#term-test-oracle). The actual
representative generated Class-Files parsed by `CpuSoftmaxGeneratedKernelTest` have one typed
entry, no fields, and no `io/github/pho001/synaptik` member reference; the [log](../../../../glossary.md#term-log) form alone refers
to `java/lang/Math.log`. The emitted hot loops directly perform typed carrier loads, maximum,
compensated exponential accumulation, and one typed output store; no hidden Synaptik helper call,
allocation, boxing, reflection, map/collection, string/generic [dispatch](../../../../glossary.md#term-dispatch), or virtual semantic
dispatch occurs in those loops. This inspection makes no literal Class-File, [JIT](../../../../glossary.md#term-just-in-time-compilation-jit), or performance
[identity](../../../../glossary.md#term-identity-meanings) assertion.

No executable Java or [Javadoc](../../../../glossary.md#term-javadoc) changed. The implementation context ran the following focused
validation on 2026-09-09 after an earlier broader invocation selected environment-owned
coverage-ledger/[checkpoint](../../../../glossary.md#term-checkpoint) tests without their [canonical](../../../../glossary.md#term-canonical-form) resources:

```bash
./gradlew --no-daemon --no-parallel --no-configuration-cache :backends:cpu:test --rerun-tasks \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.CpuInternalPackageInventoryTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuSoftmaxGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuSoftmaxInputValidatorTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuSoftmaxReferenceTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuSpecializedGeneratedMatrixTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedDirectEvidenceClosureTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelArtifactStoreTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratedKernelPersistenceEvidenceTest
```

The 14 selected XML reports record 164 tests, zero failures, zero errors, and one expected
opt-in [persistence](../../../../glossary.md#term-persistence)-evidence skip. The first broader attempt also selected the two
environment-owned coverage-ledger/checkpoint tests; they failed only because their canonical
ledger/gap-matrix resources were absent from that invocation, so they are not recorded as route
validation. No benchmark or repository-wide suite ran.

The separate clean documentation-focused context independently read the governing contracts, the
focused source/Javadocs/tests, and fresh softmax XML. It applied the General and [Planning](../../../../glossary.md#term-planning) profiles.
The current XML for `CpuSoftmaxGeneratedKernelTest` records five tests with zero failures or
errors; the implementation context's recorded 14-report serialized [run](../../../../glossary.md#term-run) remains the route-validation
evidence described above. The initial broader run is deliberately retained as a non-passing,
environment-owned coverage-ledger/checkpoint invocation, not presented as route validation.

No Javadoc, explanatory guide, glossary, architecture, build, architecture-test,
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), or integration-test change is required: this decision changes neither
executable Java nor a public, cross-backend, shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime), [dependency](../../../../glossary.md#term-dependency), [module](../../../../glossary.md#term-module), or
configuration contract. Existing [CPU](../../../../glossary.md#term-cpu)-private Javadocs accurately state the generated emitter,
validator, and independent oracle boundaries; existing terminology adds no reusable term or
changed meaning. Architecture and architecture tests remain accurate because no boundary changed;
backend conformance and integration remain deferred to CPU 0009/[CI](../../../../glossary.md#term-continuous-integration-ci) because no cross-backend or
end-to-end promise changed. The existing [Gradle](../../../../glossary.md#term-gradle)/Java configuration remains accurate because no
build input changed.

Documentation validation checked the four permitted planning records for synchronized status and
frontier, local Markdown links and heading anchors, code fences, terminology, final newlines, and
whitespace. `git diff --check` passed; `git status --short -uall` confirmed that this planning
change is the only uncommitted work. No Java test was rerun because executable Java did not change
and the recorded focused evidence is not contradicted by the fresh softmax XML.

## Completion summary

- Completed changes: Retained the complete bounded generated [CPU](../../../../glossary.md#term-cpu) [`SOFTMAX`](../../../../glossary.md#softmax--log-softmax)/`LOG_SOFTMAX` route;
  no executable route change was warranted. The verified admission/rejection, stable three-pass,
  zero-workspace, immutable invocation, carrier/layout, complete-slice, one-store, and generated
  hot-loop/oracle boundaries remain intact.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [immutable glossary](../../../../glossary.md#term-immutable); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [slice glossary](../../../../glossary.md#slice); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Files changed or created: this task plus the synchronized CPU master plan, parent CPU 0009 task,
  and roadmap; no source, Javadoc, glossary, explanatory, architecture, build, or test path changed.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Tests and validation: Reused the implementation context's serialized 14-report focused run
  (164 tests, zero failures/errors, one expected opt-in skip); accurately retained its preceding
  overbroad resource-dependent failure as non-route evidence. Independently inspected fresh<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [run glossary](../../../../glossary.md#term-run))</span>
  `CpuSoftmaxGeneratedKernelTest` XML (5 tests, zero failures/errors), source/tests, links,
  anchors, fences, terminology, status/frontier, exact paths, `git diff --check`, and status.
- Documentation-agent review: the separate clean documentation-focused context applied the General
  and Planning profiles;
  Javadocs, explanatory documentation, and glossary were reviewed and remain accurate unchanged.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Architecture impact: None.
- Unresolved issues: None for E1C. CPU 0009 remains [Ready](../../../../glossary.md#term-ready) and incomplete; E1B remains Complete.
- Follow-up required: E2 normalization is the sole next Draft summary frontier; no detailed E2
  task is created here.<span data-glossary-navigation="true"> ([normalization glossary](../../../../glossary.md#term-normalization))</span>

Status: Complete
