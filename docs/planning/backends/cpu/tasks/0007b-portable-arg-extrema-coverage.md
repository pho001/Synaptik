# Task 0007B: Portable Arg-Extrema Coverage

## Status

Complete

## Goal

Execute exactly one fully static, resolved-layout, one-axis `ARG_MIN` or `ARG_MAX` occurrence on
the portable CPU route. Return the deterministic zero-based logical coordinate along the selected
axis as an `INT64` tensor while preserving the current Model ordering, first/last tie policy,
Shape form, carrier, layout, resource, lifecycle, and fail-closed contracts in scalar and
parallel-scalar execution.

## Scope

- Recognize exactly one compiled `ARG_MIN` or `ARG_MAX` node with one numeric input, one `INT64`
  output, `ArgExtremaAttrs(axis, keepDimensions, tiePolicy)`, fully static Shapes, and fully
  resolved legal input/output layouts.
- Accept exactly current Model input types `FLOAT64`, `FLOAT32`, `BFLOAT16`, `INT32`, and `INT64`;
  reject `BOOL`, any other type, a non-`INT64` output, or a differentiable output.
- Consume the already-normalized, non-negative Model axis without renormalizing it, and validate
  it against the resolved input rank once during cold lowering. Require its selected input extent
  to be positive. With `keepDimensions == false`, remove that Dimension; with
  `keepDimensions == true`, retain it with extent one. Require the compiled output Shape to match
  exactly.
- Store the zero-based logical coordinate in the selected input Dimension, not a physical offset,
  storage ordinal, flattened index, or selected value. Arbitrary legal resolved strides,
  offsets, zero-stride input reads, and injective output layouts must not change that meaning.
- Apply signed ordering for `INT32` and `INT64`. For every floating type, a represented NaN is
  preferred to every non-NaN for both `ARG_MIN` and `ARG_MAX`; multiple represented NaNs tie;
  negative zero orders below positive zero; and infinities follow their ordinary numeric order.
  BFLOAT16 comparisons widen only the represented raw 16-bit value for comparison and never
  reinterpret a non-BFLOAT16 source value.
- On equal ordered values, including repeated NaNs and equal signed zeros, `FIRST_INDEX` returns
  the smallest logical selected-axis coordinate and `LAST_INDEX` the largest. A canonical
  increasing logical-axis traversal may retain on equality for `FIRST_INDEX` and replace on
  equality for `LAST_INDEX`.
- Reject a selected axis with static extent zero during capability/lowering before resource
  declaration. Permit zero extents only in unselected Dimensions; then the output contains zero
  cells and execution performs no generated invocation, worker submission, read, or write.
- Lower output-cell ordinals to complete selected-axis domains. Every scalar or parallel-scalar
  range owns disjoint complete output cells; no selected domain may be split, partially reduced,
  combined, or shared between workers.
- Support matching typed heap or `MemorySegment` input carriers and `long[]` or
  `MemorySegment` output carriers in heap/segment/mixed combinations. Validate accessibility,
  alignment, complete referenced spans, output injectivity, and all input/output physical overlap
  before the first write or worker submission.
- Declare zero workspace and zero materialization. Retain only immutable prepared geometry and
  invocation-private primitive coordinate state; borrow but never close the configured worker
  group and never close or retain ownership of caller buffers.
- Emit one final, field-free typed generated class with exactly one public static `invoke` entry
  whose carrier parameters are concrete, followed by primitive geometry and half-open output-cell
  range parameters. Advance `CpuGeneratorSchema.CURRENT_VERSION` exactly once from 43 to 44
  because the new compatible generated-byte family is admitted; schema-43 and older envelopes are
  incompatible safe misses with regeneration, never migration, aliasing, or dual-schema reuse.
- Treat an optimal clean primitive Java loop for the exact specialization as the generated-code
  design, structural-review, and performance oracle. The generated selected-domain loop must be
  algorithmically and structurally equivalent in traversal, comparison, tie handling, carrier
  work, address work, branch shape, and store count.

## Out of scope

- Full reduction, multiple axes, a runtime axis, unresolved Shape or layout, dynamic selected
  extent, values-plus-indices output, top-K, sort/argsort, pairwise extrema, masked arg extrema,
  or any arg-extrema spelling not currently owned by Model.
- `BOOL`, additional tie policies, altered NaN placement, payload ordering, numerical
  canonicalization, a new empty-axis identity, changed signed-zero rules, or any invented Model
  semantics.
- Vector selected-domain computation, partial/combine reduction, scratch, materialization,
  selected-domain reordering, shared mutable state, per-element/per-output allocation, or a
  generic fallback that claims unsupported dynamic forms.
- A generated bridge to `CpuScalarReferenceKernel` or another helper; Runtime semantic dispatch;
  generic `Object` carrier descriptors; reflection; method handles; `invokedynamic`; collection,
  map, or string dispatch; boxing; or avoidable virtual indirection in generated hot work.
- Public API, Model, Compiler, Training, Runtime, Prepare contract, backend contract, dependency,
  module boundary, Gradle/toolchain, architecture, conformance, integration, native route, tuning,
  tracing, or persistence-format changes beyond the CPU-private current generator schema.
- Reopening CPU 0007A1D, changing completed CPU 0007A2 behavior, planning CPU 0007C, or executing
  later Draft CPU work.

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0007A ordinary extrema`](0007a-portable-ordinary-extrema-and-boolean-reductions.md)
- [`CPU 0007A1 generated numerical aggregates`](0007a1-portable-ordinary-numerical-aggregate-reductions.md)
- [`CPU 0007A1A generated scalar self-containment`](0007a1a-generated-scalar-body-self-containment.md)
- [`CPU 0007A1C evidence closure`](0007a1c-generated-direct-evidence-closure.md)
- [`CPU 0007A1D native-order review`](0007a1d-native-order-segment-layout-hoisting.md)
- [`CPU 0007A1N aggregate-loop evidence`](0007a1n-multi-axis-min-residual-parity.md)
- [`CPU 0007A1O ledger reconciliation`](0007a1o-pointwise-ledger-evidence-reconciliation.md)
- [`CPU 0007A2 binding-aware SUM`](0007a2-portable-binding-aware-sum-to-shape-reduction.md)
- [`Model 0018U1 arg-extrema normalization`](../../../modules/model/tasks/0018u1-integral-reductions-and-arg-min-normalization.md)
- [`Compiler 0005B reduction and normalization gradients`](../../../modules/compiler/tasks/0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)

## Architecture constraints

- Model exclusively owns operation meaning, type/Shape inference, normalized attributes, logical
  index semantics, floating/integral ordering, tie policy, empty-axis rejection, and gradient
  eligibility. Compiler capture and verification remain authoritative; CPU must consume the
  verified occurrence and fail closed when its exact bounded executable subset is not proved.
- CPU analysis owns capability truthfulness, lowering, access validation, strategy, and exact
  zero-resource declarations before shared assignment. CPU finalization owns generated artifact
  realization. Runtime receives only an immutable prepared recipe, direct carriers, cold geometry,
  and disjoint output-cell ranges and does not interpret arg-extrema semantics.
- Generated code must preserve the optimal direct clean-Java algorithm, hot-loop/dataflow shape,
  and avoidable-overhead profile. A semantically correct helper-dispatch loop is not sufficient.
- Capability publication must name only the exact static one-node matrix implemented here. Every
  unsupported type, Shape, layout, attribute, carrier, resource, or occurrence form remains a
  deterministic rejection rather than a hidden fallback or optimistic support claim.
- Any need to change an authoritative architecture rule, shared module, public/backend contract,
  dependency direction, or lifecycle owner is architectural uncertainty: stop without editing
  that boundary and request clarification.

## Package impact

Existing CPU-private packages changed:

- `io.github.pho001.synaptik.backend.cpu` — exact capability admission.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — focused immutable arg-extrema IR.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — focused validation and static
  output-cell/selected-axis geometry.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — focused typed direct emitter and
  existing generator dispatch.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — one-unit portable plan.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — zero-resource declaration and
  geometry propagation/finalization.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — cold binding, overlap validation,
  invocation-private state, and scalar/parallel range invocation.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent scalar differential
  oracle.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema-44 compatibility assertions.

Packages added: None.

Type placement:

- Add `CpuArgExtremaIr`, `CpuArgExtremaLowering`, and `CpuArgExtremaEmitter` in the three focused
  packages above. Do not add `ARG_MIN`/`ARG_MAX` to `CpuAggregateIr`,
  `CpuAggregateLowering`, or `CpuAggregateEmitter`: those owners assume same-typed aggregate
  values and already contain the ordinary/multi-axis and exact-state aggregate responsibilities,
  while arg extrema has a numeric-input/`INT64`-output boundary, one-axis-only geometry, and an
  explicit logical-index tie policy.
- Add no facade, manager, helper, registry, or public type. Reuse existing carrier access,
  prepared-executable, worker, specialization, and artifact-store infrastructure only at their
  current responsibility boundaries.

## Affected files

Expected production paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
  and its package Javadoc if the published family inventory requires synchronization;
- `internal/ir/CpuPortableKernelIr.java`, new `internal/ir/CpuArgExtremaIr.java`, and
  `internal/ir/package-info.java`;
- `internal/lowering/CpuPartitionLowering.java`, new
  `internal/lowering/CpuArgExtremaLowering.java`, and `internal/lowering/package-info.java`;
- `internal/codegen/emit/CpuClassFileKernelGenerator.java`, new
  `internal/codegen/emit/CpuArgExtremaEmitter.java`, and `internal/codegen/emit/package-info.java`;
- `internal/cache/CpuGeneratorSchema.java` and directly affected cache package Javadoc;
- `internal/route/portable/CpuPortableRoutePlan.java` and its package Javadoc;
- `internal/prepare/CpuPartitionPreparationPlan.java`, `CpuPartitionPreparer.java`,
  `CpuPartitionFinalizer.java`, and their package Javadoc;
- `internal/executable/CpuPreparedExecutable.java` and its package Javadoc; and
- `internal/reference/CpuScalarReferenceKernel.java` and its package Javadoc.

Expected test paths:

- `CpuCapabilityProviderTest.java`, `CpuCapabilityProviderPublicShapeTest.java`, and
  `CpuInternalPackageInventoryTest.java`;
- new `CpuArgExtremaIrTest.java`, `CpuArgExtremaLoweringTest.java`, and
  `CpuArgExtremaGeneratedKernelTest.java` in their matching internal packages;
- `CpuClassFileKernelGeneratorTest.java`, `CpuGeneratedKernelArtifactStoreTest.java`,
  `CpuKernelSpecializationTest.java`, and `CpuPointwiseLedgerEvidenceTest.java` for generation,
  schema-44/current-versus-historical compatibility, and ledger preservation;
- `CpuPartitionPreparerTest.java`, `CpuPartitionFinalizerTest.java`, and
  `CpuPreparedExecutableTest.java`; and
- `CpuReferenceDifferentialTest.java`; executable tests own the family-specific overlap cases,
  while generic `CpuBufferBindingTest.java` remains review-only unless its existing contract is
  proved stale and the task stops to replan the path exchange.

Expected documentation/planning paths:

- `docs/backend-guide/cpu-backend.md`, `docs/glossary.md`, this task, the CPU master plan, and the
  global roadmap. Architecture, API, capability, Compiler, Training, conformance, integration,
  and build documents are review-only unless the task encounters a stated stop condition.

## Maximum scope

The complete implementation, tests, and documentation may modify or create at most 42 repository
paths: at most 23 production/Javadoc paths, 14 test paths, and the five named documentation and
planning paths. Exactly three new production types are permitted: `CpuArgExtremaIr`,
`CpuArgExtremaLowering`, and `CpuArgExtremaEmitter`.

A 43rd path, a fourth production type, a change to existing aggregate IR/lowering/emission, any
shared-module/public/build/architecture/conformance/integration path, workspace or materialization,
or partial/combine design is a stop/replan condition. An existing optional path may be omitted
when its contract remains accurate; omission does not authorize a replacement unrelated path.

## Acceptance criteria

- Capability returns supported exactly for one fully static resolved-layout `ARG_MIN` or
  `ARG_MAX` occurrence with one accepted numeric input, exact `INT64` non-gradient output, valid
  `ArgExtremaAttrs`, positive selected extent, matching keep/remove-Dimension output Shape, legal
  input layout, and injective legal output layout. Every adjacent unsupported form rejects.
- Lowering validates the already-normalized non-negative axis against the resolved input rank and
  records it unchanged with the kind, accepted input type, keep-Dimension form, tie policy, exact
  input/output Shapes and layouts, positive selected-axis extent, output cell count, selected-axis
  input stride, and sufficient immutable geometry for arbitrary legal resolved address calculation
  without Runtime semantic interpretation.
- Static selected extent zero rejects before resource declaration. An unselected zero extent
  yields zero output cells, zero generated calls, zero worker submissions, zero reads/writes, and
  otherwise valid preparation.
- Scalar and parallel-scalar outputs match the independent reference for both kinds, both tie
  policies, both keep-Dimension forms, all five input types, ranks one and greater, selected axes
  at the first/middle/last position, dense and general layouts, zero-stride input reads, partial
  and empty legal output-cell ranges, and heap/segment/mixed carrier patterns.
- Floating cases cover finite values, both infinities, multiple raw NaN encodings, NaN versus
  non-NaN, equal NaNs under both tie policies, positive/negative zero ordering and ties, and raw
  BFLOAT16 encodings. Integral cases cover signed extrema and repeated equal values. Every result
  is the exact logical selected-axis coordinate.
- Cold binding rejects insufficient, inaccessible, misaligned, overflowing, non-injective-output,
  mismatched-carrier, or physically overlapping input/output spans before mutation or submission.
  Rejected and failed invocations leave input/output canaries unchanged and submit no worker work.
- Resource declarations contain exactly two boundaries, one numeric input and one `INT64` output,
  with zero workspace and no materialization. Scalar and parallel execution borrow the worker
  group without closing it; prepared recipes and generated artifacts are immutable and safe for
  concurrent invocation with distinct buffers.
- Parallel chunks are deterministic, disjoint half-open output-cell ranges. Each chunk completes
  every selected-axis traversal for its cells, uses invocation-private primitive state, and
  produces raw-output parity with scalar execution across repeated and concurrent runs.
- Generated classes are deterministic, final, field-free, constructor-free, and expose exactly
  one typed static `invoke` entry. Descriptors contain the concrete numeric input carrier,
  `long[]` or `MemorySegment` output carrier, primitive geometry, and `long start/end`; they do not
  contain `Object` or a bridge.
- Complete `javap -c -p` and `javap -v -p` inspection proves direct typed load, compare/tie,
  logical-index, address, and store loops equivalent to the frozen optimal clean-Java oracle. The
  selected-domain hot work contains no Synaptik helper/reference call, allocation, boxing,
  reflection, method handle, `invokedynamic`, generic dispatch, collection/map/string lookup,
  avoidable division/remainder, repeated cold decision, or more than one result store per output
  cell.
- A stable forbidden-reference allowlist permits only the JDK primitive raw-bit/widening and typed
  `MemorySegment`/native-order layout operations proved necessary by the inspected specialization;
  any unexpected owner/member is a failure, not an allowlist expansion without review.
- Generated bytes and specialization keys are deterministic for the same facts and differ for
  every kind/type/form/tie/carrier/access fact that changes emitted behavior. Schema advances
  exactly once from 43 to 44; schema-43 and older envelopes safely miss and regenerate, while the
  immutable historical schema-42 ledger/evidence remains unchanged and explicitly historical.
- A frozen performance matrix includes at least: dense array `FLOAT64 ARG_MIN/FIRST` without kept
  Dimension; dense segment `FLOAT32 ARG_MAX/LAST` with kept Dimension; general-layout mixed
  `BFLOAT16 ARG_MIN/LAST`; general-layout mixed `INT32 ARG_MAX/FIRST`; dense segment `INT64
  ARG_MIN/LAST`; and general-layout array `FLOAT64 ARG_MAX/FIRST`. It spans both kinds, policies,
  forms, every input type, dense/general addressing, array/segment/mixed carriers, signed zeros,
  NaNs, and raw BFLOAT16 while timing non-empty complete output-cell ranges.
- Every frozen generated case and its optimal direct primitive Java counterpart use the same
  algorithm, traversal, carrier/layout/range work, comparison branches, index selection, stores,
  and anti-dead-code-elimination checksum. Each generated/direct ratio passes `<= 1.15x` in every
  one of five fresh isolated forks and the cross-fork median of fork medians. The benchmark oracle
  is never intentionally slowed, padded, routed through helpers, or given different work.
- The implementation context passes focused owners, one final uncached CPU suite, CPU Javadoc,
  semantic/Class-File/forbidden-reference/determinism/checksum/five-fork evidence, exact scope,
  and Git whitespace checks. A distinct clean documentation context then finalizes Javadocs,
  guide/glossary/planning impact and evidence without repeating successful Java suites.

## Tests / validation

The implementation context runs focused tests after code and test stabilization, then one final
CPU suite:

```bash
./gradlew :backends:cpu:test \
  --tests io.github.pho001.synaptik.backend.cpu.CpuCapabilityProviderTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.ir.CpuArgExtremaIrTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuArgExtremaLoweringTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuArgExtremaGeneratedKernelTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionFinalizerTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutableTest \
  --tests io.github.pho001.synaptik.backend.cpu.internal.reference.CpuReferenceDifferentialTest
./gradlew :backends:cpu:test --rerun-tasks
./gradlew :backends:cpu:javadoc
```

Retain one immutable evidence bundle outside the repository containing exact source/probe inputs,
environment and JVM version, generated class bytes, specialization facts, complete `javap -c -p`
and `javap -v -p`, member/forbidden-reference reports, raw semantic outputs, canary/overlap/range
reports, five sequential isolated `-Xms1g -Xmx1g` fork outputs, rejected whole samples, summaries,
inventory, and SHA-256 manifest. Freeze the direct oracles before timing. Each fork uses at least
five randomized warmup rounds, nine randomized measured rounds, adaptive batches of at least 25
ms, deterministic case inputs, randomized generated/direct order, and raw-output/checksum
verification before timing. Reject and retain a whole fork if any case, control, checksum, or
environment gate fails.

The distinct clean documentation context reads the stabilized diff, source Javadocs, guide,
glossary, planning records, retained evidence, and final CPU XML. It does not rerun the successful
CPU tests, generation probe, decompilation, or five-fork benchmark unless it changes executable
Java/test behavior or records a concrete stale-evidence reason. It runs:

```bash
git diff --check
git diff --cached --check
git status --short -uall
```

It also validates local Markdown links and anchors, balanced fences, final newlines, trailing
whitespace, exact path/type ceilings, schema 44 and historical schema-42 wording, task status,
CPU 0007A2 Complete, CPU 0007A1D Review needed, CPU 0007B Complete, CPU 0007C as the next Draft
task, no detailed 0007C specification, and empty staging. Repository-wide,
architecture, backend-conformance, and integration suites remain deferred to CPU 0009 or CI
because this task must not change shared boundaries or multi-module behavior.

## Dependencies

- Complete CPU 0007A supplies arbitrary-layout output-cell reduction ownership, deterministic
  extrema ordering precedent, zero-resource execution, overlap rejection, and scalar/parallel
  orchestration.
- Complete CPU 0007A1A, CPU 0007A1C, and CPU 0007A1N supply the permanent self-contained generated
  loop, optimal-direct-Java, Class-File, forbidden-reference, and five-fork evidence disciplines.
  CPU 0007A1D independently remains `Review needed`; this task neither depends on its failed local
  performance result nor changes that status.
- Complete CPU 0007A2 supplies the current schema-43 preparation/finalization/cache frontier and
  historical-schema ledger coexistence rule.
- Complete Model 0018U1 owns the exact current arg-extrema attributes, accepted types, ordering,
  tie, Shape, result-type, and empty-selected-axis contracts. Completed Compiler 0005B and current
  compiler verification/autograd code supply capture, deferred/static constraint, non-gradient,
  and fail-closed downstream contracts.

## Follow-up tasks

- CPU 0007C remains ordered Draft masked-reduction work and receives no detailed specification in
  this change.
- CPU 0007D and later tasks remain Draft in existing order. CPU 0009 or CI retains repository-wide
  capability/conformance/integration closure.
- CPU 0007A1D remains historical `Review needed`; resolving it requires its own separately scoped
  evidence/review task and is not a prerequisite for truthful ARG_MIN/ARG_MAX execution.

## Architecture impact

Expected impact: None.

This task adds only a CPU-private execution family behind existing backend, Compiler/Prepare, and
Runtime contracts. It adds no dependency, module, public API, architecture rule, resource kind,
build change, backend-conformance surface, or integration boundary. If implementation proves any
such change necessary, stop and report the uncertainty instead of editing that boundary.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007B. Work on the existing worktree
without committing, pushing, staging, resetting, reverting, deleting, or modifying unrelated
work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide, CPU task 0007B,
CPU tasks 0007/0007A/0007A1/0007A1A/0007A1C/0007A1D/0007A1N/0007A1O/0007A2, Model 0018U1,
Compiler 0005B and current capture/verification/autograd contracts, the CPU master plan, current
capability/IR/lowering/emitter/reference/route/prepare/finalization/executable/cache source and
tests, documentation rules, and the General/API-Javadoc/Planning profiles.

Implement exactly one fully static resolved-layout one-axis ARG_MIN or ARG_MAX occurrence through
new focused CpuArgExtremaIr, CpuArgExtremaLowering, and CpuArgExtremaEmitter owners. Preserve the
Model's exact five-type input, INT64 output, already-normalized non-negative axis,
keep/remove-Dimension, FIRST/LAST, NaN-preferred, signed-zero, empty-selected-axis, logical-index,
and non-gradient contracts. Validate that axis against the resolved input rank once in cold
lowering; do not renormalize it. Use
arbitrary legal layouts and heap/segment/mixed carriers, disjoint complete output-cell ranges,
scalar/parallel-scalar execution, zero workspace/materialization, pre-write overlap/span/resource
validation, immutable recipes, invocation-private state, and fail-closed unsupported forms. Do
not add arg extrema to the existing aggregate IR/lowering/emitter or introduce partial/combine
state.

Generate direct typed bytecode structurally and algorithmically equivalent to an optimal clean
primitive Java loop. Permit no bridge/helper dispatch or avoidable per-element allocation,
boxing, reflection, map/string dispatch, or virtual indirection in hot work. Advance schema 43 to
44 exactly once, preserving older-envelope safe misses and the historical schema-42 ledger. Run
all semantic, range/carrier/layout/parallel/resource/lifecycle, Class-File, forbidden-reference,
determinism, checksum, focused/full CPU, Javadoc, and reproducible five-fork generated/direct
gates in the task. Never intentionally slow the direct oracle.

Stop on the 43rd path, fourth new production type, existing aggregate-owner change,
workspace/materialization or partial/combine need, shared/public/build/architecture/conformance/
integration change, or unresolved semantic/architectural uncertainty. Hand the stabilized diff,
CPU XML, and retained evidence to a distinct clean documentation-focused context. That context
must independently finalize affected Javadocs, CPU guide, glossary, task, master plan, and roadmap
without duplicating successful Java/performance work. Do not mark Complete until every gate
passes. Do not commit, push, or stage.
```

## Local decisions

- Current Model semantics correct the older CPU row's shorthand: this is a shared `ARG_MIN` and
  `ARG_MAX` family, not arg-max-only; `ArgExtremaAttrs.axis` is already normalized and non-negative
  in the compiled occurrence received by CPU, while the attribute record constructor itself does
  not normalize; both kinds prefer NaN to non-NaN; negative zero is below positive zero; and static
  empty selected axes reject rather than return an identity. Unselected zero extents remain valid
  zero-output execution.
- Focused arg-extrema owners are required because the current aggregate IR/lowering/emitter owns
  same-typed value reductions, multiple-axis/full forms, and exact numerical state, whereas this
  family crosses numeric input to `INT64` output and has one-axis logical-index/tie behavior.
- One output-cell ordinal is the only parallel ownership unit. The selected-axis traversal is
  sequential in increasing logical-coordinate order, so FIRST/LAST policy is explicit and no
  partial/combine reduction or deterministic merge rule is needed.
- The algorithm needs no scratch or representation copy: arbitrary resolved input/output layouts
  are addressable directly, the accumulator is one represented value plus one logical index, and
  output cells are disjoint. Therefore workspace and materialization are exactly zero.
- Schema 44 is required because a newly admitted generated family changes current compatible
  bytes. Cache keys must distinguish every emitted kind/type/form/tie/carrier/access fact; runtime
  extents and worker identity remain cold facts under existing specialization policy.
- The permanent performance requirement remains unchanged: generated code matches optimal clean
  Java, and the benchmark oracle is never intentionally slowed to manufacture parity.

## Known limitations

- This task supports only fully static, resolved-layout, one-axis Model forms. It makes no claim
  for dynamic axes/extents/layouts, multi-axis arg extrema, values-plus-indices, or vectorized
  selected-domain work.
- The five-fork gate is evidence for the frozen cases, host, JVM, and protocol, not a universal
  speed guarantee or production tuning input.
- Parallelism is only across independent output cells. A result with fewer than two eligible
  chunks executes on the invoking thread even when parallel capacity is available.

## Validation evidence

Planning context: `/root/cpu_0007b_planning`.

The planning pass read the governing architecture/planning/documentation contracts, completed CPU
and Model prerequisites, current source/tests, and the same-typed aggregate-owner boundary. It
found no architectural or semantic uncertainty requiring a stop.

Implementation context `/root` ran the exact focused owner command, one fresh uncached
`./gradlew :backends:cpu:test --rerun-tasks`, a final metadata-refresh CPU suite after the required
generated entry visibility correction, and CPU Javadoc. The retained final XML reports 58 suites,
384 tests, zero failures, zero errors, and one existing skip. Executable Java and tests did not
change after that retained suite.

Retained evidence root `/tmp/synaptik-cpu-0007b-final.crCuar` verifies through `SHA256SUMS` and
records `VERIFIED,6`, six deterministic final field-free constructor-free generated classes,
exactly one typed public static `invoke` entry per class, complete `javap -c -p` and `javap -v -p`,
the stable permitted JDK member set, no forbidden reference, and byte-identical repeated
generation. All five isolated fixed-heap forks were accepted in full. Every one of the 30 case-
fork ratios and all six aggregate ratios passed `<= 1.15x`; the largest fork ratio was
`1.038305072x` for general-layout mixed-carrier INT32 ARG_MAX/FIRST.

The mandatory clean documentation context had no available context ID. It independently reviewed
the stabilized diff, final source/tests, Model arg-extrema contracts, CPU package and member
Javadocs, CPU guide, glossary, task/master/roadmap records, retained XML, checksums, member reports,
and generated decompilation. It changed no executable Java or test. Final CPU Javadoc passed with
no task-introduced documentation warning; only the two Java Vector API incubating-module notices
remain. Local Markdown target/anchor and balanced-fence checks, terminology/status checks, final-newline and
whitespace checks, exact scope/type/schema/owner checks, and empty-staging checks passed. The final
change has 40 paths, exactly three new production types, schema 44, and no staged path. CPU 0007A1D
remains `Review needed`, CPU 0007A2 remains `Complete`, and CPU 0007C is the next `Draft` task with
no detailed specification.

## Implementation notes

Implementation context `/root` added the three focused private owners and threaded their immutable
geometry through capability, route, preparation, finalization, execution, and the independent
scalar oracle. Generated code uses direct typed carriers, increasing logical-axis traversal,
raw-bit NaN/signed-zero selection, one INT64 coordinate store per output cell, zero workspace and
materialization, complete-span overlap rejection, and scalar or complete-cell parallel ownership.
Current generator schema is 44; schema-43 and schema-42 test envelopes miss, while the historical
schema-42 ledger wording remains explicit.

The final performance correction identified repeated cold selected-axis stride loads as the
remaining generated/direct discrepancy. The emitter now supplies a structurally proved unit-
stride body and a runtime-guarded stride-two body for the affected BFLOAT16/INT32 forms, while
retaining the arbitrary-stride direct typed fallback. The accepted generated classes preserve the
frozen straightforward Java oracle's algorithm, traversal, comparisons, address work, and one
store per output cell.

The documentation pass finalized constructor/method Javadocs for the new IR, lowering geometry,
emitter, and reference contract; synchronized package summaries; added the current CPU arg-extrema
guide with a concrete Shape/tie/ordering walkthrough; updated existing glossary entries; and
synchronized this task, the CPU master plan, and roadmap. No Javadoc or explanatory text claims
fusion, compiler/Model changes, gradients, multi-axis behavior, native/vector selected-domain
execution, or broader backend readiness.

## Completion summary

- Completed changes: implemented and documented CPU-private portable ARG_MIN/ARG_MAX across the
  exact five numeric input types, logical INT64 results, both Shape forms and tie policies,
  arbitrary legal layouts/carriers/ranges, scalar and parallel-scalar execution, direct typed
  generated loops, zero resources/materialization, and schema 44.
- Files changed or created: 40 paths, including exactly three new production types, three new test
  types, affected CPU implementation/package Javadocs, the CPU guide, glossary, this task, CPU
  master plan, and roadmap.
- Tests and validation: retained focused owners passed; retained final CPU XML has 384 tests, zero
  failures/errors, and one existing skip; all five performance forks and aggregates passed; final
  CPU Javadoc had no task-introduced documentation warning; Markdown/static checks, SHA-256
  verification, scope/schema/owner checks, and Git whitespace/staging checks passed.
- Documentation-agent review: completed in this clean documentation context; no context ID was
  available, and no executable Java or test changed.
- Documentation impact: CPU guide and package summaries now describe only the exact implemented
  static portable subset and its execution/resource/generated-code boundaries.
- Javadoc review: all changed Java contracts were reviewed; the new constructor/member contracts
  now document every input, result, and expected failure in scope.
- Glossary impact: existing aggregate-reduction and arg-extrema tie-policy entries now distinguish
  Model meaning from the exact current CPU realization; no new reusable term was needed.
- No-change conclusions: architecture/current plan/ADR and architecture tests are unchanged because
  no authority, dependency, or lifecycle boundary changed; public Model/Compiler/Training/API and
  capabilities documents are unchanged because semantics and gradients did not change; backend
  conformance/integration, build/Gradle, shared modules, unrelated modules, and existing
  `CpuAggregateIr`/`CpuAggregateLowering`/`CpuAggregateEmitter` remain unchanged because the task is
  CPU-private and uses focused owners behind existing contracts.
- Evidence: `/tmp/synaptik-cpu-0007b-final.crCuar`.
- Unresolved issues: None within task 0007B. Historical CPU 0007A1D remains `Review needed`.
- Follow-up required: None for 0007B. CPU 0007C remains the next ordered `Draft` task without a
  detailed specification.

Status: Complete
