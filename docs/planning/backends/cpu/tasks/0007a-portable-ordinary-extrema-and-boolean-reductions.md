# Task 0007A: Portable Ordinary Extrema and Boolean Reductions

## Status

Complete

## Goal

Add the next bounded CPU frontier after completed CPU 0007: execute exactly one fully static,
resolved-layout current Model ordinary `MIN`, `MAX`, `ALL`, or `ANY` occurrence through the
portable generated-kernel route.

This task establishes one shared full/single-axis/multi-axis aggregate geometry without also
choosing a floating summation or product algorithm. Each output cell owns one complete reduction
domain, visits that domain in canonical logical row-major order, and may execute independently in
parallel. No worker splits or combines one reduction domain, and the task declares no workspace,
partial buffer, or hidden accumulator resource.

## Why CPU 0007A was split

The former row combined three independently implementable boundaries:

1. ordinary selector and boolean folds (`MIN`, `MAX`, `ALL`, and `ANY`);
2. ordinary numerical accumulation (`SUM`, `MEAN`, and `PROD`); and
3. binding-aware target-Shape `SUM`.

The first boundary has one output-domain geometry, represented-value extrema/boolean state, fixed
empty identities, and zero scratch. The second must separately implement Model's exact-real
floating `SUM`/`MEAN` target, product special values, finite-result rounding, and any required
declared accumulator resource; reusing CPU 0007's per-step typed scan policy would not satisfy the
ordinary `SUM` contract. The third adds right-aligned bound-Shape compatibility and a different
axis-selection geometry. Combining them would pre-decide two numerical/resource designs and one
binding-aware geometry in a task whose executable foundation does not require them.

CPU 0007A therefore owns only the first cohesive boundary. Draft CPU 0007A1 follows with ordinary
`SUM`/`MEAN`/`PROD`; Draft CPU 0007A2 follows with binding-aware `SUM_TO_SHAPE`. This refinement
does not change the ownership or order of CPU 0007B–0007F.

## Scope

### Exact occurrence and attribute matrix

- Admit exactly one CPU-owned node whose kind is `AggregateReductionKind.MIN`, `MAX`, `ALL`, or
  `ANY`, with one ordered input and one output.
- Admit exactly these current ordinary attribute forms:
  - exact `NoOperationAttrs.INSTANCE`, meaning full reduction over every input axis with the
    canonical scalar output;
  - exact `AxisReductionAttrs`, carrying one already normalized input axis and its exact
    `keepDimensions` flag; or
  - exact `MultiAxisReductionAttrs`, carrying distinct already normalized axes and the exact
    `keepDimensions` flag. An empty axis list selects a one-element point domain at every input
    position; it is not a full reduction.
- Reject `SUM`, `MEAN`, `PROD`, `ARG_MIN`, `ARG_MAX`, masked, statistical, target-Shape, and every
  other kind or attributes class. Do not reinterpret another signature as ordinary reduction.
- Revalidate the current compiler-produced occurrence: exactly one input/output, exact attributes,
  fully static descriptors, exact Model result Shape and type, and resolved layouts. CPU prepare
  must not assume that Tensor construction was the only producer of the captured node.

### Exact data-type and result matrix

| Kind | Accepted input types | Output type |
|---|---|---|
| `MIN` | FLOAT64, FLOAT32, BFLOAT16, INT32, INT64 | exact input type |
| `MAX` | FLOAT64, FLOAT32, BFLOAT16, INT32, INT64 | exact input type |
| `ALL` | BOOL only | BOOL |
| `ANY` | BOOL only | BOOL |

There is no promotion, widened result, cast, integer truth conversion, or floating truth
conversion. BFLOAT16 comparisons use the exact represented BFLOAT16 value widened only for
comparison; the selected output remains an exact BFLOAT16 represented value. BOOL input follows
the current canonical one-byte representation and every output is canonical false or true.

### Shape, domains, and empty behavior

- Full reduction selects every input axis and produces the canonical scalar Shape, including for
  rank-zero input. A rank-zero scalar contributes its one value. An input Shape with any
  zero-extent axis has an empty full domain and still has one scalar output identity.
- Single-axis reduction selects exactly the normalized axis. `keepDimensions=false` removes it;
  `true` retains it with extent one. A zero selected extent creates an empty domain independently
  for every output cell.
- Multi-axis reduction selects axis membership, not attribute-list order. Removal drops every
  selected axis; retention replaces each selected extent with one. Empty axes preserve the input
  Shape and create a point domain containing exactly the corresponding input value, irrespective
  of `keepDimensions`.
- An output cell exists for each row-major coordinate over unselected input axes. If any
  unselected extent is zero, the output has zero logical elements: no identity is materialized,
  no generated call occurs, and no worker is submitted.
- Selected zero extents make each otherwise existing domain empty. Materialize these exact
  identities:

| Kind/type | Empty-domain result |
|---|---|
| floating `MIN` | positive infinity in the result format |
| floating `MAX` | negative infinity in the result format |
| INT32 `MIN` / INT64 `MIN` | `Integer.MAX_VALUE` / `Long.MAX_VALUE` |
| INT32 `MAX` / INT64 `MAX` | `Integer.MIN_VALUE` / `Long.MIN_VALUE` |
| BOOL `ALL` | canonical true |
| BOOL `ANY` | canonical false |

Zero-element input/output Shapes are valid when these rules permit them. All element counts,
domain counts, coordinate products, referenced spans, and byte sizes are checked during cold
lowering/binding; overflow fails before execution.

### Traversal, extrema, and determinism

- For each output cell, traverse selected coordinates in canonical input logical row-major order:
  the greatest input-axis index changes fastest. Physical offsets and strides never determine
  logical order. Attribute order for multi-axis reduction does not alter traversal.
- INT32 and INT64 extrema use exact signed order. BOOL uses conjunction/disjunction over each
  logical value and may short-circuit only within the same canonical domain; short-circuiting must
  not change any observable result, failure, write, or worker behavior.
- Floating non-NaN extrema use numerical order with infinities ordered normally. `MIN` selects
  negative zero when either zero sign occurs; `MAX` selects positive zero when either occurs.
- A NaN anywhere in a floating domain produces NaN. The CPU-private deterministic policy selects
  the represented bits of the first NaN in canonical logical traversal order; it does not promise
  that payload choice across backends or future schema versions. Non-NaN selected values retain
  their exact represented bits.
- Scalar and parallel-scalar execution use the same per-output-cell traversal and must be bitwise
  identical. Floating arithmetic is not performed, BFLOAT16 is never rounded or accumulated, and
  integers never overflow because extrema only compare represented values.

### Generated execution, parallelism, and resources

- Add focused CPU-private `CpuAggregateIr`, `CpuAggregateLowering`, and `CpuAggregateEmitter`
  owners in the existing IR, lowering, and Class-File emission packages. A different equally
  focused `ExtremaBoolean` name is acceptable only if every integration seam uses it consistently.
- Lower one occurrence to one computation unit and one generated artifact. The declared boundary
  and buffer order is exactly `[input, output]`. There is no materialization, workspace, partial
  result, combine buffer, per-domain heap object, persistent accumulator, or shared mutable state.
- Generated primitive `start`/`end` bounds denote a contiguous range of independent flattened
  output-cell ordinals. Scalar execution covers all output cells; parallel-scalar execution
  partitions only this output domain into deterministic disjoint ranges. Never split a reduction
  domain across workers and never perform partial/combine work.
- The generated entry is a direct Class-File bridge to a CPU-owned static aggregate body, following
  current scan/fold structure. The task does not require the reduction loop to be embedded in the
  generated bytecode and must not document the bridge as if it were embedded emission.
- Cold per-range geometry may allocate one invocation-private primitive `long[]` coordinate pack,
  as the current scan/fold routes do. That pack is prepared invocation state, not numerical
  workspace. The body allocates no object per output cell or selected element.
- Vector and parallel-vector bodies are out of scope. Empty outputs perform no generated call;
  one output cell is scalar. Parallelism requires multiple independent output cells.

### Layouts, carriers, and pre-mutation validation

- Support arbitrary currently supported fully resolved input/output layouts with non-negative
  storage offsets and strides: dense, offset, positive-strided, interleaved, transposed, and
  zero-strided reads; and arbitrary proven-injective writable output layouts.
- Use exact current carriers: `double[]`, `float[]`, `short[]`, `int[]`, `long[]`, or `byte[]` for
  the matching type, accessible native-order `MemorySegment`, and compatible heap/segment mixed
  boundaries. Retain current ownership, alignment, accessibility, writability, and size rules.
- Require a distinct output value/binding and an injective output layout. Use the completed bounded
  injectivity decision, including exact enumeration for the currently permitted small-layout
  bound and the monotone-span sufficient proof above it; do not replace it with a stricter dense
  requirement or an unbounded address set.
- Validate complete physical input/output referenced spans, not range prefixes. Reject any actual
  complete-span overlap before coordinate-pack mutation, output initialization/write, generated
  invocation, or worker submission. Disjoint slices of one underlying allocation remain legal
  when their complete referenced byte spans do not intersect.
- Every carrier, layout, Shape, type, address, span, overlap, and resource failure occurs before
  the first observable output mutation. No failure may expose a partially reduced output.

### IR, specialization, compatibility, and capability

- `CpuAggregateIr` is the canonical CPU-private semantic structure. It includes kind, represented
  data type, ordinary form (`FULL`, `SINGLE_AXIS`, or `MULTI_AXIS`), canonical selected-axis
  membership, retention policy, structural input/output access plans, deterministic floating-NaN
  selection policy, output-domain range meaning, and zero-workspace policy.
- Canonicalize multi-axis membership in increasing normalized-axis order for geometry and
  compatible artifact identity; retain/revalidate that the original attributes contain distinct
  normalized axes. Attribute list order is not a traversal or result fact.
- Specialization/fingerprint/persistence identity additionally includes ranks, boundary roles and
  count, carrier pattern, compute mode, generated entry descriptor, and every other fact that
  changes generated bytes or compatibility. Concrete compatible extents, offsets, stride
  magnitudes, element/domain counts, slots, carrier instances, addresses, run/worker identity, and
  selected range count remain cold when they do not change emitted bytes.
- Advance generated compatibility from schema 20 to schema 21 with no migration reader. A schema-
  20 artifact is incompatible; corrupt/incompatible artifacts remain rejected before definition.
- Extend the independent `CpuScalarReferenceKernel` without calling the production aggregate body,
  generated bridge, lowering helper, or production coordinate packer. The oracle follows logical
  Shape coordinates and the exact identities/NaN/zero rules above.
- `CpuCapabilityProvider.supports` may return true only when the complete one-node partition can
  pass lowering, declarations, assignment/finalization, artifact generation, binding, complete
  overlap validation, and execution for the exact matrix. Occurrence-local truth must not imply
  that a mixed or multi-node partition is supported before complete partition lowering exists.

## Out of scope

- ordinary `SUM`, `MEAN`, or `PROD`, including any exact/reproducible floating accumulator,
  reciprocal division, product parity implementation, partial reduction, combine tree, or scratch
- binding-aware `SumToShapeAttrs`/`SUM_TO_SHAPE`, right-aligned target-Shape axis derivation, or
  unresolved/dynamic bound Shapes
- `ARG_MIN`, `ARG_MAX`, masked reductions, `LOG_SUM_EXP`, variance, standard deviation, L1/L2
  norms, softmax/log-softmax, layer/RMS/batch normalization, attention, or fusion
- more than one node, general partition-DAG decomposition, reduction epilogues, materialized
  splits, in-place execution, output overlap, negative offsets/strides, unresolved layouts, or
  runtime-bound axes/extents
- vector reduction, within-domain parallelism, partial/combine work, atomics, hidden resources,
  native/provider routes, tuning, benchmarks, relaxed numerics, or performance claims
- Model, Compiler, shared Prepare/Runtime, Config, Trace, Engine, NN, training, gradient, public API,
  architecture, ADR, dependency, Gradle/Java-version, or build-structure changes
- new architecture-test, backend-conformance, or integration infrastructure; portable closure and
  repository-wide conformance remain owned by CPU 0009

## Architecture references

- [Architecture contract](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Runtime/prepare/backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [CPU master plan](../master-plan.md)
- [CPU 0005A partition-kernel reset](0005a-atomic-partition-kernel-architecture-reset.md)
- [CPU 0005C vector and parallel strategies](0005c-vector-and-parallel-portable-strategies.md)
- [CPU 0005D evidence gate](0005d-materialization-specialization-and-persistence-evidence-gate.md)
- [CPU 0005E typed portable family expansion](0005e-portable-pointwise-types-carriers-and-semantic-family-expansion.md)
- [CPU 0006 static affine views](0006-portable-static-affine-views-and-boundary-materialization.md)
- [CPU 0006B2 overlap fold](0006b2-portable-overlap-fold.md)
- [CPU 0006C stable ordering](0006c-portable-stable-ordering-and-selection.md)
- [CPU 0007 cumulative scans](0007-portable-cumulative-scan-coverage.md)
- [Model 0018U1 integral reductions](../../../modules/model/tasks/0018u1-integral-reductions-and-arg-min-normalization.md)
- [Model 0018V multi-axis reductions](../../../modules/model/tasks/0018v-multi-axis-and-statistical-reductions.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [CPU backend guide](../../../../backend-guide/cpu-backend.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- Model owns operation identity, attribute/Shape/type eligibility, extrema/boolean meaning, empty
  identities, and signed-zero/NaN requirements. CPU owns only revalidation, private IR/geometry,
  deterministic realization, resource declarations, generated route, binding, and execution.
- CPU analysis declares exactly two buffers and zero workspace before CPU-blind shared assignment.
  Finalization validates assignments before realizing exactly one compatible generated artifact.
- Prepared recipes are immutable and reusable. Concurrent runs use distinct `RunState` values and
  invocation-private coordinate packs; no prepared accumulator or mutable reduction state is
  shared across runs.
- Runtime receives one direct prepared invocation and no operation, graph, axis list, route choice,
  layout discovery, resource discovery, or semantic dispatcher.
- Work remains inside `backends/cpu`, adds no dependency/package/public supported type, and
  preserves the existing generated portable route and optional persistence lifecycle.
- Capability truth may not exceed complete-partition lowering and executable evidence. Stop if
  exact behavior requires Model/compiler/shared-contract changes, numerical accumulation, hidden
  scratch, within-domain combination, or another semantic family.

## Package impact

No package is added, removed, moved, or exported.

- `io.github.pho001.synaptik.backend.cpu` remains the sole supported fail-closed capability owner.
- `internal.ir.CpuAggregateIr` owns immutable ordinary-form/kind/type/access/determinism identity.
- `internal.lowering.CpuAggregateLowering` owns one-node revalidation, selected-axis membership,
  checked layouts, output/domain counts, declarations, and compact cold geometry.
- `internal.codegen.emit.CpuAggregateEmitter` owns the direct generated bridge and CPU-owned static
  scalar body with represented-value comparison/load/store mechanics.
- Existing prepare, portable-route, cache, executable, memory, and reference packages gain only
  the focused integrations required by this family.
- Tests mirror the corresponding production packages. Add no generic reduction registry,
  accumulator abstraction, manager, service, cursor, or utility package.

## Affected files

Expected production/package paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuPortableKernelIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAggregateIr.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuPartitionLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAggregateLowering.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecialization.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/CpuPortableRoutePlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/portable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutable.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/package-info.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuScalarReferenceKernel.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/reference/package-info.java`

Expected CPU tests:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuAggregateIrTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuAggregateLoweringTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuAggregateGeneratedKernelTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGeneratorTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuKernelSpecializationTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratedKernelArtifactStoreTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedExecutableTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/reference/CpuReferenceDifferentialTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/memory/CpuBufferBindingTest.java`

Expected documentation/planning paths during implementation:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

No Model/compiler/shared Prepare/Runtime/API/architecture/Gradle/conformance/integration path is
expected to change.

## Maximum scope

This task may create or modify at most 42 paths: 24 production/package paths, 13 CPU test paths,
and 5 documentation/planning paths. Stop before a 43rd path, a new package, or any path outside the
listed CPU module and documentation/planning set. An existing listed integration or inventory test
may remain unchanged, but an unlisted replacement must not increase the ceiling.

If implementation requires numerical accumulation, target-Shape geometry, a shared contract,
another workspace, another semantic family, or more paths, stop and report the exact conflict. Do
not conceal incomplete coverage behind capability truth or a follow-up.

## Acceptance criteria

- Capability and lowering admit exactly one fully static resolved-layout ordinary `MIN`, `MAX`,
  `ALL`, or `ANY` occurrence with exact full/single-/multi-axis attributes and type matrix.
- Full, removed/retained single-axis, ordered-but-membership-equivalent multi-axis, and empty-axis
  point forms derive the exact current Model output Shape and domain geometry.
- FLOAT64/FLOAT32/BFLOAT16 extrema, INT32/INT64 extrema, and BOOL folds produce exact output types,
  empty identities, infinity/NaN/signed-zero/signed-order behavior, and canonical BOOL results.
- Every output cell follows canonical logical row-major selected-domain traversal; first-logical-
  NaN selection and scalar/parallel-scalar output bits are deterministic.
- Parallelism is only across complete independent output cells. There is zero declared workspace,
  no partial/combine work, no hidden accumulator, and no vector route.
- Dense, offset, positive/zero-strided, transposed/interleaved reads; injective non-dense writes;
  heap/native/mixed carriers; scalars; and selected/unselected zero extents match an independent
  reference oracle that does not invoke production aggregate helpers.
- Complete physical overlap and every binding/layout/resource failure occur before coordinate
  mutation, output writes, generated calls, or worker submission.
- The Class-File route is documented and tested as a generated direct bridge to a CPU-owned static
  body. Hot loops allocate no per-cell/per-element object and inspect no Model/graph/runtime policy.
- Schema 21 rejects schema-20 artifacts. Canonical IR, fingerprint, specialization, persistence,
  and cold-geometry inclusion/exclusion rules match this specification.
- `SUM`, `MEAN`, `PROD`, target-Shape SUM, later reduction/normalization families, multi-node
  partitions, unsupported types/attrs, and dynamic/unresolved forms remain fail-closed.
- Existing pointwise, movement, indexing, scatter, fold, ordering, random, scan, cache, persistence,
  worker, and resource behavior remains unchanged.
- A separate clean documentation-focused context finalizes affected Javadocs/package summaries,
  CPU guide, glossary impact, task/master/roadmap evidence, and explicit API/architecture/
  conformance no-change conclusions before this task becomes Complete.

## Tests / validation

During implementation, run focused tests covering capability negatives/positives, aggregate IR,
all three attribute forms, Shape/domain geometry, every type/kind, identities, empty selected and
unselected axes, empty multi-axis selection, scalar input, NaN payload selection, signed zero,
infinities, signed integral extrema, canonical BOOL, arbitrary layouts/carriers, bounded
injectivity, preparation/finalization, complete pre-mutation overlap, scalar/parallel parity,
schema/cache identity, and an independent reference comparison.

After executable Java stabilizes, run one final module suite:

```bash
./gradlew :backends:cpu:test
```

The separate clean documentation pass reuses that successful evidence unless it changes
executable Java and runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
```

It also validates affected Markdown links/anchors, balanced fences, one terminal newline,
trailing whitespace, rendered Javadocs, exact changed-path membership/ceiling, package/type
placement, schema 21, type/attribute/empty-domain matrix, task/master/roadmap status/dependency
coherence, and absence of detailed CPU 0007A1/0007A2 task files.

Repository-wide tests, architecture tests, backend conformance, and integration tests are deferred
to CPU 0009 or continuous integration. Run them here only if implementation unexpectedly changes
a repository-wide, dependency, architecture, or reusable cross-backend contract; that is outside
scope and normally requires stopping first.

## Dependencies

- CPU 0005A–0007: Complete, including typed carriers, resolved-layout access plans, bounded
  injectivity, direct Class-File bridge patterns, scalar/parallel-scalar workers, exact resource
  declarations, complete overlap validation, independent reference patterns, and schema 20.
- Model 0018U1 and 0018V: Complete current ordinary full/single-/multi-axis signatures, five-type
  numeric extrema, BOOL ALL/ANY, Shape/zero-domain identities, signed-zero/NaN semantics, and
  empty-axis multi-axis meaning.
- Current Compiler capture/inference: Complete and unchanged; produces revalidated exact
  one-input/one-output descriptors and normalized axes before CPU prepare.
- Current shared Prepare/Runtime contracts: Complete and unchanged; carry CPU declarations and
  invoke only the finalized direct prepared executable.

## Follow-up tasks

- Draft CPU 0007A1 owns ordinary full/single-/multi-axis `SUM`, `MEAN`, and `PROD`, including the
  exact floating numerical target, integral modular behavior, product special values,
  deterministic strategy, and explicit scratch/accumulator design.
- Draft CPU 0007A2 depends on CPU 0007A1 and owns binding-aware `SUM` with `SumToShapeAttrs`, exact
  right-aligned bound-Shape compatibility, leading/aligned reduction geometry, and truthful
  resources. Neither follow-up has a detailed specification yet.
- CPU 0007B–0007F retain arg-extrema, masked, advanced statistical/norm, stable softmax, and
  normalization ownership. CPU 0009 retains portable closure and conformance.

## Architecture impact

Expected impact: None.

The task uses existing backend ownership of lowering, route selection, generated artifacts,
resources, binding, and execution. It changes no module responsibility or dependency direction.
If implementation requires architecture, another module, or a shared contract change, stop and
report the exact conflict instead of editing around it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
Implement Synaptik CPU task 0007A exactly from its Ready specification. Do not use GSD. Read
AGENTS.md, ARCHITECTURE.md, the current architecture plan and directly relevant architecture
documents, documentation rules/profiles, planning guide, roadmap, CPU master plan, task 0007A,
completed CPU tasks 0005A/0005B/0005C/0005D/0005E/0006/0006B2/0006C/0006D/0007, Model tasks
0018U1/0018V and current reduction source/tests, Compiler reduction inference, and every affected
CPU scan/fold/capability/lowering/IR/generator/prepare/executable/reference source and test in full
before editing.

Deliver exactly one fully static resolved-layout ordinary MIN/MAX/ALL/ANY occurrence through the
portable generated route for exact NoOperationAttrs, AxisReductionAttrs, and
MultiAxisReductionAttrs forms. Implement the specified five-type extrema and BOOL matrix, exact
Shape/empty-domain identities, logical row-major first-NaN/signed-zero behavior, arbitrary
supported layouts/carriers, injective outputs, complete pre-mutation overlap rejection,
output-cell-only scalar/parallel-scalar work, zero workspace/partial/combine state, an independent
reference oracle, truthful complete-partition capability, and schema 21. The generated entry is a
bridge to a CPU-owned static body; do not overclaim embedded bytecode. Preserve all exclusions and
the 42-path ceiling. Stop on any architecture, shared-contract, Model-semantic, numerical,
resource, or scope conflict.

Run focused tests and one final ./gradlew :backends:cpu:test after executable Java stabilizes. Do
not commit or push. Then hand the uncommitted diff and exact Java evidence to a distinct clean
documentation-focused context following docs/developer-guide/documentation-rules.md. That pass
must independently finalize affected Javadocs/package summaries, CPU guide, glossary impact,
task/master/roadmap, no-change conclusions, Javadoc, rendered pages, Markdown, exact-scope, and
whitespace validation without repeating the successful Java suite unless executable behavior
changes. Do not mark Complete until both passes and every acceptance criterion succeed. Return a
completion summary with completed changes, exact files, tests/validation, documentation and
Javadoc/glossary impact, unresolved issues/follow-up, own CODEX_THREAD_ID when available, and
Status: Complete or Status: Incomplete with Follow-up required: <specific follow-up>.
```

## Local decisions

- Split the old ordinary-reduction row at numerical/resource and binding-geometry boundaries.
  Extrema/boolean folds are first because they establish complete output/domain geometry with no
  floating accumulation algorithm or scratch dependency.
- Canonical logical row-major traversal makes behavior independent of physical layout and gives a
  deterministic CPU-private first-NaN payload policy without adding public semantics.
- Parallelize only across independent output cells. Keeping each complete domain within one worker
  eliminates partial/combine state, reassociation, and scratch from this task.
- Treat empty selected domains by identity and empty unselected/output domains as no work.
- Reject physical overlap rather than introduce in-place alias/traversal semantics absent from the
  current Model and CPU prepared boundary.

## Known limitations

- Coverage is one fully static, resolved-layout occurrence with non-negative physical strides.
- Floating NaN payload choice is deterministic only for CPU schema 21 and is not a cross-backend
  promise. No floating arithmetic or accuracy/performance claim is made.
- One full reduction or any occurrence with only one output cell remains scalar, even for a large
  domain. This task has no within-domain parallelism.
- Numerical aggregates and target-Shape SUM remain fail-closed pending CPU 0007A1 and 0007A2.

## Validation evidence

- Planning context `019ffe96-884f-7bb0-be25-56ccfeeaa7e3` split the former CPU 0007A row and
  produced this Ready implementation specification after reviewing the architecture/planning
  contracts, completed CPU foundations, current Model/Compiler reduction contracts, and CPU
  scan/fold/resource/determinism patterns.
- Local planning links, balanced fences, terminal newlines, trailing whitespace, every required
  planning-guide section, exact three-path CPU planning inventory, Ready/Draft dependency/frontier
  consistency, one new detailed task, absence of 0007A1/0007A2 specifications, and
  `git diff --check` passed. The planning thread applied patches only to this task, the CPU master
  plan, and CPU-specific roadmap text; concurrent NN planning/source/test work remained outside
  its patch targets and was neither staged nor rewritten.
- Implementation context `019ffea4-7930-70c2-9773-ef9c76fefc17` completed the executable slice
  in 32 CPU paths: 23 production/package paths and 9 test paths, all in this task's allowlist.
  Its focused aggregate run passed 10 tests; its broader preparation/cache/integration run passed
  50 tests. Its final `./gradlew :backends:cpu:test` run on OpenJDK 26.0.1+8-34 with Gradle 9.6.1
  passed 53 suites and 303 tests with zero failures and zero errors. No executable Java or test
  changed after that run.
- Documentation context `019ffeb8-d37c-7c31-9c89-26a0264258d8` independently recounted the
  preserved final JUnit XML as 53 suites, 303 tests, 0 failures, 0 errors, and 1 skip. The sole
  skip is
  `CpuGeneratedKernelPersistenceEvidenceTest.recordsFixedForkedPersistenceEvidence()`, whose
  existing assumption reports that explicit persistence evidence is disabled. This context did
  not rerun a Java test suite.
- The documentation context reviewed the final implementation diff, all 23 affected production/
  package paths and 9 test paths, the current Model aggregate contracts/tests, generated pages,
  and the required architecture, lifecycle, performance, planning, API, backend-guide, Javadoc,
  developer-guide, example, and glossary contracts. It finalized detailed aggregate IR/lowering/
  emitter/executable Javadocs, affected package summaries, the CPU backend guide, this existing
  glossary term, and CPU planning/status records without changing executable statements or tests.
- `./gradlew :backends:cpu:javadoc` passed after the final Java documentation edit. Earlier
  successful generations exposed comment-only rendered-page defects; the context corrected them
  and reran as permitted. The final build emitted only the two expected incubating Vector API
  warnings. Rendered inspection covered `CpuAggregateIr` and both nested enums,
  `CpuAggregateLowering`, `Geometry`, `Layout`, `CpuAggregateEmitter`,
  `CpuCapabilityProvider`, `CpuPreparedExecutable`, `CpuGeneratorSchema`,
  `CpuPortableRoutePlan`, `CpuScalarReferenceKernel`, and affected package summaries.
- Final targeted validation confirmed local Markdown targets and generated heading anchors,
  balanced backtick/tilde fences, the worked INT32 calculation, glossary terminology, terminal
  newlines, trailing whitespace, package/type placement, exact type/attribute/empty-domain
  coverage, schema 21 plus explicit schema-20 rejection, direct-bridge wording, status/dependency
  coherence, absence of detailed 0007A1/0007A2 files, concurrent NN-scope preservation, the exact
  37-path CPU-attributable allowlist, the 42-path ceiling, and `git diff --check`.
- Tensor API: no change. The public full/single-/multi-axis reduction meanings, Shapes, types,
  identities, and attributes were already documented; this task adds only a concrete CPU
  realization and no public Tensor contract.
- Compile API: no change. Compiler capture/inference and reduction semantics are unchanged, and
  CPU lowering consumes the existing projected occurrence without adding a compiler promise.
- Runtime API: no change. The existing prepared-executable, cold-binding, `RunState`, ownership,
  and schedule contracts already cover this backend-private recipe; no shared Runtime API or hot-
  path policy changed.
- Training API: no change. The task adds no gradient, optimizer, parameter, training-session, or
  backend-specific training contract.
- Model/compiler/shared capabilities or master plans outside CPU: no change. Model semantics and
  Compiler production remain inputs to CPU revalidation; capability truth changes only in the
  concrete CPU provider and complete one-node CPU lowering.
- Architecture, architecture decision records, and architecture tests: no change. Backend-owned
  lowering, artifact realization, binding, and execution remain within the existing contract,
  with no dependency or module-boundary change.
- Backend conformance and integration tests: no change. This bounded CPU unit/integration slice
  does not introduce shared conformance infrastructure; portable closure remains CPU 0009 work.
- Gradle, dependencies, Java version, build structure, Config, Trace, Engine, NN, training, and
  every other module: no change. The implementation uses existing Java 26 CPU facilities and
  adds no dependency or cross-module surface.

## Implementation notes

- Added CPU-private `CpuAggregateIr`, `CpuAggregateLowering`, and `CpuAggregateEmitter` owners and
  integrated them through capability, portable-route specialization, schema/cache, prepare,
  finalization, direct prepared execution, and independent scalar reference evaluation.
- Full, normalized single-axis, and canonical-membership multi-axis forms use one checked output/
  domain geometry. Empty axis membership is a one-value point domain; selected zero extents write
  identities, while unselected zero extents produce no output work.
- The exact matrix is five represented numeric types for `MIN`/`MAX` and canonical BOOL for
  `ALL`/`ANY`. Logical input row-major traversal selects the first represented NaN, enforces
  signed-zero extrema, compares BFLOAT16 after widening while copying selected raw bits, and uses
  signed integral order.
- Scalar and parallel-scalar execution partition only independent complete output cells. The plan
  declares exactly input/output buffers and zero workspace/materialization/partial/combine state.
- Binding validates carriers, layouts, injective output, canonical BOOL input, and complete
  physical overlap before mutation or submission. Schema 21 has no migration reader and treats
  schema 20 as incompatible.
- The generated class contains a direct bridge to the CPU-owned static body; documentation does
  not describe the reduction loop as embedded bytecode.

## Completion summary

- Completed changes: Implemented exact one-node portable ordinary `MIN`, `MAX`, `ALL`, and `ANY`
  execution across the specified forms, types, layouts, carriers, empty domains, deterministic
  selection rules, output-cell ranges, resources, reference oracle, capability, and schema 21.
- Files changed or created: 37 CPU-attributable paths: 23 production/package paths, 9 CPU test
  paths, and 5 documentation/planning paths. All are task-allowlisted and below the 42-path limit.
- Tests and validation: Reused implementation-context focused 10-test, broader 50-test, and final
  53-suite/303-test CPU evidence; independently recounted 0 failures, 0 errors, and the 1 expected
  opt-in persistence skip. Final CPU Javadoc, rendered-page, Markdown, scope, schema, status,
  package, concurrency-preservation, and whitespace validation passed.
- Documentation-agent review: Clean context `019ffeb8-d37c-7c31-9c89-26a0264258d8` completed the
  mandatory independent documentation finalization without executable or test changes.
- Documentation impact: Updated affected Javadocs/package summaries, the CPU backend guide, this
  task, CPU master plan, and CPU-specific roadmap evidence. Public API and architecture documents
  remain unchanged for the reasons recorded above.
- Javadoc review: Finalized aggregate IR/enums, lowering/Geometry/Layout, emitter, capability,
  executable, schema, route, reference, and package contracts; final generation and rendered-page
  inspection passed.
- Glossary impact: Updated the existing CPU portable-route entry for current schema 21 and the
  implemented ordinary aggregate boundary; no unnecessary new term was introduced.
- Unresolved issues: None.
- Follow-up required: None for CPU 0007A. CPU 0007A1 is the next Draft planning frontier; CPU
  0007A2 remains Draft and dependent on 0007A1. Neither has a detailed specification.

Status: Complete
