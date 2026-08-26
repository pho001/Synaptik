# Task 0008A: Portable Channels-First Dimensional Convolution Closure

## Status

Complete

## Goal

Close the current portable channels-first forward-convolution slice in the CPU backend. Validate
NCW Conv1d from its actual visible two-branch `EXPAND_DIMS(2) -> CONV2D -> SQUEEZE(2)` compiled
composition through CPU analysis, finalization, generated execution, and result publication, and
add direct grouped NCDHW `CONV3D` execution with truthful resources and bounded scalar parallelism.

Keep both paths rank-specific. Reuse the completed Conv2d implementation only where source and
tests prove an identical CPU-private responsibility; do not create public or backend-generic
`ConvNd`, dynamic-rank machinery, or general partition-directed-acyclic-graph (DAG) handling.

## Motivation and mental model

The repository exposes two different rank-extension mechanisms:

```text
Conv1d convenience                       first-class Conv3d

NCW input ---- EXPAND_DIMS(2) --\        NCDHW input --\
                                  CONV2D                 CONV3D -> NCDHW result
NCW weight --- EXPAND_DIMS(2) --/        5-D weight ----/
                                  |
                             SQUEEZE(2) -> NCW result
```

Conv1d must remain those ordinary visible graph nodes. CPU analysis may recognize exactly that
family-owned topology and keep its singleton views virtual, but it must not rewrite the compiled
graph or claim general DAG fusion. Conv3d is already one ordinary flat Compiler node, so CPU owns
rank-five validation, lowering, resources, generation, binding, and direct execution.

## Scope

### Exact NCW Conv1d composition

- Admit one CPU-owned partition containing exactly four semantic occurrences: two independent
  `EXPAND_DIMS` occurrences at literal axis `2`, one `CONV2D` consuming both expanded values and
  optional original rank-one bias, and one `SQUEEZE` at literal axis `2` consuming the Conv2d
  result. Permit either deterministic topological order of the two independent expansions; admit
  no other node, edge, output, or fan-out.
- Require the external input Shape `[N, C_in, W]`, external weight Shape
  `[C_out, C_in/groups, K_w]`, optional bias `[C_out]`, and final output
  `[N, C_out, W_out]`. Verify the visible intermediate Shapes
  `[N, C_in, 1, W]`, `[C_out, C_in/groups, 1, K_w]`, and
  `[N, C_out, 1, W_out]` exactly.
- Require the Conv2d attributes to be the exact Model mapping
  `(strideHeight=1, strideWidth=stride, paddingHeight=0, paddingWidth=padding,
  dilationHeight=1, dilationWidth=dilation, groups=groups)` and require both rank edits to carry
  the exact literal axis. Similar-looking arbitrary rank-edit/Conv2d graphs fail this recognizer.
- Require all three intermediate values to be single-use, unpublished, and virtual according to
  the complete memory requirements. Declare only original input, original weight, optional bias,
  and final NCW output buffers. Declare zero workspace and no intermediate materialization.
- Synthesize only the CPU-private singleton-height access geometry needed by the existing direct
  Conv2d unit. Preserve the original rank-three carrier bases, offsets, spans, and non-negative
  strides; a singleton-axis stride has no storage extent and must not trigger copying. Write the
  final result directly through the rank-three output carrier using the equivalent singleton-
  height rank-four address interpretation.
- Keep the exact captured/compiled graph visible in fixtures and assertions. Recognition is a
  bounded lowering fact, not a Model operation, Compiler rewrite, Planning rule, or public
  capability called `CONV1D`.
- Exercise the composition from complete `PrepareContext` analysis through resource assignment,
  finalization, cold binding, scalar and eligible parallel-scalar invocation, and final NCW
  result. No Engine-facing claim is made before Engine 0004.
- Reuse the existing `CpuConv2dIr`, emitter, reference semantics, cache identity, and execution
  body when the synthesized rank-four geometry is indistinguishable from direct singleton-height
  Conv2d. Do not create a second Conv1d numerical kernel or silently materialize either view.

### Exact admitted grouped NCDHW Conv3d occurrence

- Add truthful `CpuCapabilityProvider` admission for exactly `Conv3dKind.CONV3D` with
  `Conv3dAttrs`, ordered `[input, weight]` or `[input, weight, bias]`, and one output.
- Admit only fully static Shapes with present resolved layouts. Require input
  `[N, C_in, D, H, W]`, weight `[C_out, C_in/groups, K_d, K_h, K_w]`, optional bias `[C_out]`,
  and output `[N, C_out, D_out, H_out, W_out]` with the exact Model/Compiler formulas.
- Require positive static kernel extents, stride, dilation, and groups; non-negative symmetric
  padding; checked channel divisibility; exact `weightChannelsPerGroup * groups == C_in`; exact
  bias length; exact output geometry; matching gradient-request metadata; and an injective output.
- Admit FLOAT64, FLOAT32, and BFLOAT16 boundaries in any ordered combination allowed by Model
  promotion. Reject BOOL, INT32, INT64, every unresolved Shape/layout, and any future floating type
  until an explicit semantic and CPU route exists.
- Support groups `1`, intermediate groups, and depthwise geometry; zero or empty legal output
  domains; typed primitive arrays, native-order `MemorySegment` values, and mixed carriers; and
  arbitrary legal resolved non-negative offsets and strides through the general scalar form.

### Conv3d semantic and numerical contract

- Implement grouped NCDHW cross-correlation without reversing the stored kernel.
- For each complete output cell, visit its contiguous group-local input channels, kernel depth,
  kernel height, then kernel width in increasing logical order. Padded coordinates are conceptual
  positive zero and still participate in ordinary multiplication, including multiplication by
  infinity.
- Initialize from optional intrinsic bias exactly once. FLOAT64 output accumulates in FLOAT64;
  FLOAT32 and BFLOAT16 output accumulate in FLOAT32; BFLOAT16 narrows once at the final store.
  An empty contraction starts from positive zero before optional bias.
- Preserve Model permission for reassociation and fused multiply-add without claiming cross-
  backend bitwise identity. For every generated specialization, the independent optimal clean
  Java implementation and emitted code must use the same semantic algorithm, loop/dataflow shape,
  rounding boundaries, address progression, validation envelope, and avoidable-overhead profile.
- Cover NaN, infinity, signed zero, conceptual padding, mixed promotion, bias, empty contraction,
  groups/depthwise, stride, dilation, padding, and arbitrary legal layout behavior explicitly.

### Conv3d epilogue boundary

- The only Conv3d epilogue in this task is the optional intrinsic rank-one bias carried by the
  `CONV3D` occurrence.
- A Conv3d-led external ADD, activation, clamp, publication fan-out, or any other suffix is not
  directly fused and does not select CPU 0008's Conv2d-only materialized-suffix form. Such a
  multi-node Conv3d partition fails this task's specialized lowering closed.
- Do not generalize `CpuConv2dIr.Epilogue`, `CpuConv2dLowering`, or
  `CpuPreparedExecutableSequence` into an all-rank convolution policy. CPU 0008B owns general DAG
  unit decomposition/materialized fallback, and CPU 0008C owns later closed typed epilogue
  recognition.

### Resources, generation, and bounded parallelism

- Direct Conv3d declares exactly input, weight, optional intrinsic bias, and output buffers with
  checked spans, sizes, access modes, and alignments. It declares zero workspace, zero
  materializations, one computation unit, one portable route plan, one generated artifact, and
  one prepared executable.
- No im2col, packing, reorder, partial-sum, combine, provider, native, or persistent prepared
  resource is permitted. If correctness or performance requires one, stop and replan.
- Generated scalar work owns a half-open range of complete output cells in logical
  `N, C_out, D_out, H_out, W_out` order. Decode each cell once, then run the complete group-local
  contraction. Empty output invokes no generated entry and submits no worker work.
- Parallel-scalar execution partitions only those complete output-cell ranges through the current
  CPU worker owner and existing bounded cold heuristic. A cell has exactly one writer; workers
  share no partial state, atomics, workspace, combine phase, or nested submission. Scalar and
  parallel-scalar use the same generated body and per-cell algorithm.
- Cold binding validates exact carriers, native accessibility/order/alignment, spans, output
  writability, worker capacity, and every output/input overlap before the first write or worker
  submission. Input/input sharing is allowed where all reads remain valid; every output/input
  overlap is rejected.
- Add one rank-specific immutable `CpuConv3dIr`, one focused lowerer, one focused emitter, and one
  independent direct Java reference owner. Advance the generator schema exactly once and make all
  older envelopes safe misses.
- Keep generated classes deterministic, final, field-free, and constructor-free, with one typed
  static entry and primitive cold-supplied geometry/range arguments. Generated hot code contains
  no graph/layout/operation lookup, generic dispatch, route selection, cache access, allocation,
  boxing, reflection, collections, synchronization, or Synaptik numerical helper calls.

### Proved reuse boundary

- Reuse existing `CpuAccessPlan`, carrier bindings, specialization/cache machinery, resource
  assignment, finalization, prepared-executable binding, worker orchestration, and complete-output-
  cell range infrastructure because those responsibilities are already rank-neutral in source.
- Reuse the Conv2d numerical unit for exact Conv1d composition because the Model mapping proves a
  singleton-height Conv2d with identical semantic work.
- Keep Conv3d IR, lowering, emitter, and oracle rank-specific. Extract a new CPU-private shared
  component from rank two and rank three only if both completed implementations otherwise contain
  the same invariant and algorithmic responsibility, the extraction reduces duplication, and its
  name does not imply arbitrary rank. Record the proof and both consumers in `Local decisions`.
- Do not share merely because fields have similar names. No universal convolution geometry,
  epilogue, IR, rank loop, public facade, registry, or generic helper is planned.

### Generated-code and performance evidence

- Freeze a machine-readable task-owned generated-form ledger before final measurement. A form is
  distinct whenever result/accumulator type, intrinsic-bias control flow, carrier access topology,
  address-progression body, or another emitted hot-loop/dataflow fact changes class bytes or
  executed instructions. Every distinct hot form admitted by this task must have a ledger row;
  any unledgered form is either added and evidenced within scope or rejected closed.
- The ledger must cover every new Conv3d emitted form and every Conv1d composition form whose
  lowering changes the previously evidenced Conv2d entry or invocation shape. Cache keys that
  generate byte-identical bodies may share a row only when an automated equivalence check proves
  identical bytes and typed entry semantics; record every covered key in that row.
- At minimum the ledger includes unbiased and biased Conv1d; grouped and depthwise Conv1d;
  FLOAT64, FLOAT32, and BFLOAT16 result/accumulator forms; dense heap, all-segment, selected mixed-
  carrier, and arbitrary-layout access forms; unbiased and biased Conv3d; groups `1`, intermediate
  groups, and depthwise Conv3d; padded/strided/dilated Conv3d; and scalar plus parallel-scalar
  orchestration. Add an unchanged direct Conv2d control and its materialized-suffix control.
- For every retained generated class in the ledger, save deterministic bytes and checksum, parse
  the complete Class-File, and inspect complete `javap -c -v` output. Automated member/reference
  scans must prove the typed entry, expected loop/address/multiply/add/conversion shape, exact
  allowed primitive/JDK calls, and absence of fields, constructors, method handles,
  `invokedynamic`, dynamic constants, bootstrap methods, allocation, boxing, reflection,
  collection/map dispatch, graph/layout/operation references, hidden bridge/interpreter calls,
  and Synaptik-owned per-cell or per-contribution helpers.
- Every ledger row representing a distinct hot specialized form must compare generated execution
  with its own optimal clean Java implementation in five fresh accepted fixed-heap Java 26 forks.
  Each fork uses deterministic inputs, randomized generated/direct order, at least five warmup
  batches, nine measured rounds, adaptive batches of at least 25 ms, and semantic equality before
  timing. Every per-fork ratio and aggregate median must be `<= 1.15x`.
- Reject a whole fork only before measurement for a recorded environment, classpath, or control
  failure. Discard no measured ratio. Retain per-row ratios, medians, environment, source and class
  digests, accepted/rejected forks, inspection reports, and one evidence-manifest digest. Claims
  remain bounded to those exact forms and environment.

### Documentation and Javadoc

- Update meaningful Javadoc for every changed Java type/method contract, including inputs,
  returns, failures, ownership, resources, range semantics, and nullability where applicable.
- Update `docs/backend-guide/cpu-backend.md` with current Conv1d-composition and direct Conv3d
  behavior, resources, epilogue boundary, schema, validation, and bounded evidence. Do not write
  planned Engine or cross-backend behavior as current.
- Update `docs/glossary.md` only if the implementation introduces or changes a reusable term;
  otherwise record a reasoned no-change conclusion.
- Finalize this task, the CPU master plan, and roadmap from actual evidence. CPU 0008B remains
  Draft and becomes next only after this task is Complete.

## Out of scope

- Any public, Model, Compiler, Planning, Prepare, Runtime, Engine, Backend Contract, Config, Trace,
  or backend-generic `ConvNd` API; dynamic rank; or a new operation kind.
- Symbolic/dynamic Shape or unresolved-layout CPU execution.
- Conv3d backward, training, saved values, gradients, or Compiler 0006C.
- General partition-DAG decomposition, fusion, recognition, profitability, materialization,
  representation reuse, or decision facts from CPU 0008B–0008E.
- External Conv3d epilogues and generalization of the Conv2d-only two-unit sequence.
- MATMUL/linear, pooling, attention, or loss execution from CPU 0008F–0008I.
- Vector convolution, native/provider route, packing, reorder, im2col, autotuning, cache
  measurement, hardware-specific specialization, or shared architecture changes.
- Runtime route selection, scheduling policy, graph inspection, validity-policy changes, or public
  end-to-end Engine integration.
- Unrelated refactoring, dependency/build changes, backend conformance, integration-test
  infrastructure, or architecture-contract changes.

## Architecture references

- `ARCHITECTURE.md`: Model semantic ownership; Compiler flat validated graph; Planning ownership;
  concrete-backend analysis, exact pre-assignment declarations, post-assignment finalization,
  generated-code ownership, Runtime prepared-recipe execution, and performance oracle discipline.
- `docs/architecture/current-architecture-plan.md`: explanatory compile/plan/prepare/run sequence
  and current module boundaries.
- `docs/planning/planning-guide.md`: Ready-task completeness, scope, validation tiers, evidence,
  and clean implementation/documentation context requirements.
- `docs/planning/backends/cpu/master-plan.md`: CPU package map and ordered 0008A–0008I frontier.
- Complete Model tasks 0025G and 0025H, Compiler task 0006B, and CPU task 0008.

## Architecture constraints

- Model remains the sole semantic owner. CPU consumes the exact Conv1d composition and Conv3d
  contract without redefining Shapes, promotion, numerical meaning, or operation identity.
- Compiler retains one visible flat graph. CPU-private Conv1d recognition changes neither captured
  nodes nor publication identity.
- CPU analysis/lowering owns family legality, exact resources, route choice, specialization, and
  generated artifacts before Runtime. Finalization only resolves already declared assignments.
- Runtime invokes prepared executables and owns validity/publication transitions; it does not
  inspect operations, select a route, allocate undeclared resources, or decompose a graph.
- Generated code follows the optimal clean Java oracle's specialized loop/dataflow and
  avoidable-overhead profile. Performance claims require the complete task ledger and five-fork
  evidence above.
- No dependency direction, shared contract, module boundary, public API, or architecture rule may
  change. If one is required, stop and report the conflict.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu` — exact occurrence capability admission.
- `io.github.pho001.synaptik.backend.cpu.internal.ir` — immutable rank-specific code identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — exact topology and geometry owners.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — direct Class-File emission.
- `io.github.pho001.synaptik.backend.cpu.internal.reference` — independent clean Java oracle.
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — schema/compatibility identity.
- `io.github.pho001.synaptik.backend.cpu.internal.route.portable` — existing selected-route facts.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — declaration and finalization facts.
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — existing cold binding/execution.

Packages added or changed:

- No package is added. Only the existing CPU-private packages above may change.

Type placement:

- `io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv3dIr` — immutable rank-five geometry,
  type, access, bias, algorithm, and code-shaping identity.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv1dCompositionLowering` — exact
  four-node/two-branch recognition and singleton-view folding; not a general graph lowerer.
- `io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv3dLowering` — exact static NCDHW
  geometry, boundaries, resources, and direct-family lowering.
- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuConv3dEmitter` — direct typed
  complete-output-cell Class-File body.
- `io.github.pho001.synaptik.backend.cpu.internal.reference.CpuConv3dReferenceKernel` — independent
  optimal clean Java semantic and performance oracle.

Tests mirror production packages. The generated-form ledger and performance harness remain test-
only evidence, not production tuning infrastructure.

## Affected files

Expected production and Javadoc paths:

- `CpuCapabilityProvider.java` and CPU package Javadoc;
- `internal/ir/CpuPortableKernelIr.java`, new `CpuConv3dIr.java`, and IR package Javadoc;
- `internal/lowering/CpuPartitionLowering.java`, new `CpuConv1dCompositionLowering.java`, new
  `CpuConv3dLowering.java`, and lowering package Javadoc;
- `internal/codegen/emit/CpuClassFileKernelGenerator.java`, new `CpuConv3dEmitter.java`, and emitter
  package Javadoc;
- `internal/reference/CpuConv3dReferenceKernel.java` and reference package Javadoc;
- `internal/cache/CpuGeneratorSchema.java` and existing specialization/artifact compatibility
  owners only where rank-five identity requires them;
- existing portable-route, preparation-plan, preparer, finalizer, prepared-executable, and package
  Javadoc owners only where the new rank-specific IR, geometry, declarations, or typed fields
  require integration; and
- an optional single new narrowly named CPU-private common component only after the reuse proof in
  this specification is recorded. It must have exactly the two rank-specific consumers.

Expected test/resource paths:

- focused Conv1d composition topology/lowering/end-execution tests;
- focused Conv3d capability, IR, lowering, reference, generated-kernel, preparation/finalization,
  executable/binding, schema/cache, package-inventory, evidence, and performance tests/resources;
- negative topology, rank/axis/attribute, descriptor, geometry, type, layout, publication/fan-out,
  resource, carrier/span/alignment, alias, and validation-before-write matrices; and
- unchanged direct and split Conv2d controls.

Expected documentation/planning paths:

- `docs/backend-guide/cpu-backend.md`;
- `docs/glossary.md` only if terminology changes;
- this task;
- `docs/planning/backends/cpu/master-plan.md`; and
- `docs/planning/roadmap.md`.

## Maximum scope

This task may create or modify at most:

- 24 production/Javadoc paths, including exactly the five planned new rank-specific types and at
  most one optional proved-common private type;
- 17 CPU test/resource paths;
- the five named documentation/planning paths, with the glossary optional; and
- 46 total paths.

The ceiling exceeds the ordinary 12–18-file guardrail because one ordered cohesive capability
must validate an existing four-node composition and add the rank-five family through the same
CPU lifecycle, generated-code, and evidence boundaries. No Model/Compiler test source, shared
Java, Gradle, architecture, conformance, integration, later task specification, or vendor path is
permitted. If the work needs another production owner or exceeds a category, stop and replan.

## Acceptance criteria

- The exact visible Conv1d graph, including two independent `EXPAND_DIMS(2)` producers, one mapped
  Conv2d occurrence, and one `SQUEEZE(2)`, is asserted and executes correctly through analysis,
  assignment, finalization, binding, scalar/parallel invocation, and NCW output.
- Conv1d keeps all three intermediate graph values virtual and single-use, declares only external
  boundaries, uses zero workspace/materialization, and reuses the existing Conv2d generated unit.
  Wrong axes, mapping, ranks, edges, fan-out, publication, or added nodes fail closed.
- CPU capability is true exactly for the admitted static resolved-layout grouped NCDHW Conv3d
  matrix and false for every excluded signature/type/descriptor/geometry case.
- Unbiased and intrinsic-biased Conv3d produce correct results for groups `1`, intermediate
  groups, depthwise, unit/non-unit stride/dilation, zero/non-zero padding, legal empty domains,
  all three result types, mixed promotion, arrays, segments, mixed carriers, and general layouts.
- Results preserve the exact current Model/Compiler Shape, grouping, traversal, promotion,
  conceptual-padding, bias, accumulation, special-value, and final-conversion contracts.
- Conv3d accepts intrinsic bias only. Every external suffix or multi-node Conv3d-led partition
  fails the specialized path and never selects the Conv2d-only fused/split epilogue machinery.
- Analysis declares exact Conv3d buffers and zero workspace/materialization before assignment;
  finalization introduces no requirement. Empty output performs no call or worker submission.
- Scalar and bounded parallel-scalar execution own disjoint complete output cells, use identical
  per-cell generated work, and require no atomics, scratch, partials, or combine phase.
- All cold carrier/span/alignment/accessibility/overlap/worker validation completes before output
  mutation or submission. Failures expose no partially valid result.
- The generator schema advances once; old artifacts are safe misses; identical specializations
  regenerate byte-identical classes.
- The generated-form ledger is closed over every distinct new hot emitted form. Complete Class-
  File and decompilation inspection passes for every retained class with all required positive
  and forbidden-member/reference checks.
- Every distinct ledger hot form and both unchanged Conv2d controls pass all five per-fork and
  aggregate generated/direct `<= 1.15x` gates against optimal clean Java oracles. No measured
  sample is discarded.
- Existing CPU families and the direct/fused/split Conv2d contract remain green and unchanged in
  meaning. No general DAG, external Conv3d epilogue, native/vector route, or later family appears.
- No public API, shared module, dependency, build, architecture, Runtime policy, conformance, or
  integration contract changes.
- Production/test types match the package map and all category/path ceilings.
- A separate clean documentation-focused agent finalizes Javadocs, CPU guide, glossary impact,
  planning status, links, and evidence in the same overall change.
- CPU 0008A is marked Complete only from actual evidence. CPU 0008B becomes the sole next Ready
  CPU task; no 0008B detailed specification is created here.

## Tests / validation

During implementation, run focused tests for affected owners, for example:

```bash
./gradlew :modules:model:test --tests '*TensorConv1dExpressionTest' --tests '*Conv3d*'
./gradlew :modules:compiler:test --tests '*Conv3dCompilerTest'
./gradlew :backends:cpu:test \
  --tests '*Conv1d*' \
  --tests '*Conv2d*' \
  --tests '*Conv3d*' \
  --tests '*CpuCapabilityProviderTest' \
  --tests '*CpuPartitionPreparerTest' \
  --tests '*CpuPartitionFinalizerTest' \
  --tests '*CpuPreparedExecutableTest' \
  --tests '*CpuInternalPackageInventoryTest'
```

The Model and Compiler commands verify the unchanged visible composition and first-class Conv3d
forward contracts; they do not authorize changes in those modules. After CPU executable code
stabilizes, run exactly one authoritative uncached CPU suite:

```bash
./gradlew :backends:cpu:test --rerun-tasks
```

Run the task-owned evidence harness in five fresh isolated Java 26 fixed-heap forks with the exact
protocol in Scope. Retain machine-readable XML counts, the form ledger, semantic results, ratios,
class bytes/checksums, complete Class-File and `javap -c -v` reports, forbidden-reference scan,
accepted/rejected fork log, and manifest digest.

The final clean documentation-focused context receives the stable diff and exact Java/evidence
results. It does not repeat successful Java or performance runs unless it changes executable
behavior or records a concrete stale-evidence risk. After final Javadoc edits it runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short -uall
```

It also checks Markdown links/anchors, heading order, balanced fences, final newlines, terminology,
generated Javadoc pages, form-ledger/evidence digests, exact path/type ceilings, schema/status/order,
absence of a 0008B task file, and empty staging.

Repository-wide validation remains deferred to CPU 0009 or continuous integration because this
task changes no shared dependency or architecture boundary. Backend conformance remains CPU 0009:
the current module has only its foundation and this task promises a CPU-private subset, not a
cross-backend result contract. Engine 0004 owns public lifecycle integration. If implementation
changes those facts, stop and replan with proportionate validation.

## Dependencies

- Complete CPU 0008 and its direct grouped NCHW Conv2d, bounded Conv2d-only epilogue/split,
  generated-schema-51, resource, preparation/finalization, worker, binding, and evidence contracts.
- Complete Model 0025G, which defines Conv1d solely as the exact visible composition consumed here.
- Complete Model 0025H, which owns first-class Conv3d Shape, grouping, promotion, numerical, bias,
  and provenance semantics.
- Complete Compiler 0006B, which adopts one flat Conv3d forward node and independently validates
  its exact descriptors/obligations. Draft Compiler 0006C is not a forward prerequisite.
- Existing Planning, Prepare, Runtime, Backend Contract, Config, Trace, cache, and worker contracts
  are sufficient and remain unchanged.

## Follow-up tasks

- CPU 0008B: general partition-DAG computation-unit decomposition and bounded fusion; sole Ready
  CPU task after completion.
- CPU 0008C: closed typed specialized-subgraph and epilogue recognition, including any selected
  external Conv3d epilogue.
- CPU 0008D: bounded fusion profitability and typed cold decision facts.
- CPU 0008E: bounded multi-input materialization and representation reuse.
- CPU 0008F–0008I: MATMUL/linear, Pool2d, attention, and loss execution.
- CPU 0009: generated-coverage/backend-conformance checkpoint.
- Compiler 0006C: separate Conv3d gradient closure.
- Engine 0004: public Conv1d/Conv2d/Conv3d lifecycle integration.

## Architecture impact

Expected impact: None.

This task consumes existing concrete-backend analysis, exact declaration, finalization, generated-
artifact, worker, cold-binding, and prepared-execution boundaries. Exact Conv1d recognition and
rank-specific Conv3d are CPU implementation facts. If implementation requires a shared/public
contract, new dependency, Runtime policy, or architecture rule, stop and report the conflict.

## Implementation prompt

Use this prompt in a separate clean implementation task/thread:

```text
You are the isolated implementation agent for Synaptik CPU task 0008A.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md,
docs/planning/backends/cpu/master-plan.md, and
docs/planning/backends/cpu/tasks/0008a-portable-channels-first-dimensional-convolution-closure.md.
Read the task's directly referenced Model, Compiler, CPU source/tests, generated-code/performance,
and documentation contracts. Implement exactly the Ready specification. Do not implement later
CPU tasks. Stop and report any architecture, shared-contract, or maximum-scope conflict.

After implementation and recorded Java/Class-File/performance validation, hand the stable diff
and evidence to a separate clean documentation-focused agent. That pass must follow
docs/developer-guide/documentation-rules.md, independently finalize Javadocs, the CPU guide,
glossary impact, planning evidence/status, and documentation checks in the same overall change,
and must not repeat successful Java/performance work unless executable behavior changes or a
concrete risk is recorded.

Do not commit or push unless the coordinating user explicitly authorizes it. Update this task's
local decisions, known limitations, validation evidence, implementation notes, completion summary,
and final status only from actual results.
```

## Local decisions

- Planning fixes the Conv1d lowering boundary to one exact four-node/two-branch topology. This is
  the smallest source-backed way to validate the visible composition without requiring CPU 0008B's
  general DAG decomposition.
- Planning fixes Conv3d epilogues to intrinsic bias only. CPU 0008's external ADD/RELU and two-unit
  materialized suffix are rank-two family contracts, while 0008B–0008C explicitly own their later
  generalization.
- No new shared convolution component is pre-approved by resemblance alone. The implementation
  may add at most one narrowly named private component only after recording the concrete rank-two/
  rank-three responsibility and both consumers.
- Direct complete-output-cell traversal with zero workspace is the only admitted Conv3d algorithm.
  A packing, im2col, partial, combine, or native requirement is a replanning trigger.

## Known limitations

- Conv1d CPU execution is limited to the exact current visible composition with virtual singleton
  views and no extra publication, fan-out, or suffix.
- Conv3d CPU execution is forward-only, fully static, resolved-layout, scalar/parallel-scalar, and
  has intrinsic bias only.
- Public Engine integration, cross-backend conformance, gradients, general DAG/epilogue handling,
  vector/native routes, and later heavy families remain the named follow-ups above.

## Validation evidence

- Clean planning context `01a03e57-b2a0-7001-ab6e-f566b85c4f1b` started from clean `main` at
  `e0494057`. It read `AGENTS.md`, the full architecture contract and current architecture plan,
  the planning guide and roadmap, the CPU master plan, complete CPU 0008, Model 0025G/0025H, and
  Compiler 0006B task specifications, plus directly relevant Model/Compiler/CPU source, tests,
  CPU/Tensor/Compile guides, documentation rules/profiles, and commit history.
- Source inspection confirmed Conv1d is exactly the visible two-expansion/Conv2d/squeeze graph;
  Conv3d is one flat forward node; current CPU lowering has one Conv2d-first special path and an
  otherwise straight-line boundary; Conv2d external epilogue and sole two-unit materialized split
  are rank-two-specific; direct Conv2d uses complete-output-cell ranges, zero workspace, schema 51,
  direct generated scalar code, and an independent clean Java oracle. These facts determine this
  task's exact Conv1d recognizer and intrinsic-bias-only Conv3d epilogue boundary.
- Planning-only checks passed: canonical 20 task headings in order; exactly one CPU task-table
  `Ready` row. At that planning checkpoint, 0008A was linked and Ready while 0008B was unlinked,
  Draft, next, and without a task file; both
  introduced local Markdown links resolve; ten balanced fences; terminal line feed; no carriage
  returns or trailing whitespace; exactly the task, CPU master plan, and roadmap changed; empty
  staging; and tracked `git diff --check` plus new-file whitespace checks. No Java, Gradle,
  Javadoc, Class-File, or performance command was run for this planning-only change.
- Documentation context `01a03e87-9b14-7003-a6e9-e984bcf17144` applied the General,
  API-and-Javadoc, Developer-guide, Backend-guide, Planning, and Example profiles. It changed
  Javadoc in `CpuGeneratorSchema`, `CpuConv3dIr`, `CpuPartitionLowering`,
  `CpuPartitionPreparationPlan`, and `CpuConv3dLowering`; updated the CPU backend guide; and
  finalized this task, the CPU master plan, and the roadmap. It independently reviewed the
  remaining changed/new CPU contracts, including capability, route, preparation/finalization,
  prepared execution, the Conv1d recognizer, Conv3d emitter/reference oracle, tests, and ledger;
  their existing or implementation-drafted Javadocs remained accurate and complete, so no
  further edits were needed.
- The same pass reviewed `ARCHITECTURE.md`, the current architecture explanation, package
  summaries, Tensor/Compiler convolution contracts, and the glossary without change. The task
  changes no architecture owner, dependency, public API, package boundary, build, backend-
  conformance, integration, or Engine contract. It introduces no reusable term and changes no
  existing term's semantic meaning, so the glossary correctly remains unchanged.
- Final `./gradlew :backends:cpu:javadoc` completed successfully: 11 tasks, 2 executed and 9
  up-to-date, with only the two expected incubating Vector API warnings and no Javadoc content
  warning. Generated pages for the Conv3d IR, lowerer and geometry, and reference oracle were
  inspected for the finalized contracts. The targeted local checker reported
  `Markdown validation passed for 4 files`, including local links, anchors, and balanced fences.
- The retained evidence manifest contains 230 checksummed entries; `sha256sum -c` reported every
  entry `OK`, and its own digest is
  `dd185e49c691fb9761414c414c6bdf88507c463c109600deaeb62448c8c2292c`. Final scope is exactly
  35 paths: 15 production, 16 test/resource, and 4 documentation/planning paths. New-file final
  newline and tracked/untracked whitespace checks passed. `git diff --check` and
  `git diff --cached --check` produced no diagnostic. Status checks found 0008A Complete, exactly
  one CPU Ready row for 0008B, no 0008B detailed spec, and zero staged paths.
- Successful Model, Compiler, CPU, Class-File, and performance evidence was reused rather than
  rerun because this pass changed documentation and Javadoc only, not executable Java behavior.
  No validation limitation or unresolved documentation issue remains.

## Implementation notes

- Implementation context `01a03e65-a069-71b0-811b-ce55dcf875aa` worked from HEAD
  `e0494057` with the planning-only task/master-plan/roadmap diff preserved and unstaged.
- Exact Conv1d recognition accepts only the two axis-2 expansions feeding the third-node Conv2d
  and the axis-2 squeeze of that Conv2d result. It proves private single-use intermediates and
  address-preserving singleton descriptors, then delegates the rank-four boundary descriptors to
  the existing Conv2d lowering, geometry, emitter, and complete-output-cell execution unit. The
  three intermediate values remain virtual; no new Conv1d generated body or entry shape exists.
- Direct Conv3d uses the new rank-specific `CpuConv3dIr`, `CpuConv3dLowering`,
  `CpuConv3dEmitter`, and independent `CpuConv3dReferenceKernel`. The emitted NCDHW grouped loop
  owns complete output cells, applies optional intrinsic bias once, loads weights even for
  conceptual padding, multiplies positive zero by that weight, preserves FLOAT64/FLOAT32
  accumulation and final BFLOAT16 conversion, and uses zero workspace/materialization. Dense heap
  carriers use proved dense integer address progression; segment, mixed-carrier, and arbitrary
  resolved layouts use exact long element addressing.
- Conv3d reuses only existing rank-neutral carrier emission, specialization/cache, immutable
  artifact, range orchestration, worker, binding, overlap, and buffer infrastructure. Its IR,
  layout geometry, lowerer, emitter, and oracle remain rank-specific; no shared ConvNd component
  was introduced. Conv1d reuses the completed rank-two numerical unit because its singleton-height
  mapping is exact.
- Conv3d-led multi-node partitions and external epilogues fail in the one-node Conv3d lowerer.
  Vector/native selection, materialization, and Conv2d fused/split epilogues remain unavailable to
  Conv3d. Generator schema advanced once from 51 to 52.
- The implementation changed 15 CPU production paths and 16 CPU test/resource paths, including
  five new production types, six new test types, and one generated-form-ledger resource; the three
  planning paths are the pre-existing planning change plus this implementation evidence. These
  counts remain within the task ceilings and no other module source, dependency, build, public API,
  architecture, conformance, or integration path changed.
- The frozen 12-row machine-readable ledger maps three unchanged Conv1d composition forms to their
  previously evidenced Conv2d bodies, closes seven distinct Conv3d generated/performance forms,
  and records direct plus materialized-suffix Conv2d controls. Seven deterministic Conv3d classes,
  compatibility bytes, SHA-256 values, parsed Class-File member scans, and complete `javap -c -v`
  reports passed typed-entry, loop/arithmetic/conversion, allowed-call, and forbidden-reference
  inspection. No fields, constructors, method handles, `invokedynamic`, dynamic/bootstrap
  constants, allocation, boxing, reflection, collection dispatch, graph/layout operation
  references, or Synaptik-owned hot helpers were present.
- After the ledger was frozen, five fresh accepted Java 26 `-Xms1g -Xmx1g` forks passed every
  per-fork `<= 1.15x` gate with no rejected protocol fork. Aggregate generated/direct medians were:
  dense FLOAT32 `0.667123855x`; grouped biased FLOAT32 `0.817644722x`; grouped biased FLOAT64
  `1.066896079x`; depthwise BFLOAT16 `1.062009415x`; all-segment FLOAT32 `0.817853071x`;
  arbitrary-layout mixed carrier `0.860624087x`; and parallel-scalar FLOAT32 `0.683170212x`.
  Three pre-ledger measured tuning samples are retained separately as developmental calibration and
  are not part of the frozen protocol. The unchanged Conv2d control aggregates also passed:
  direct dense `0.992271187x`, grouped `1.035211861x`, depthwise `0.939897066x`, general mixed
  `0.971556498x`, fused `0.988055832x`, parallel `0.987585697x`, materialized split
  `0.991748527x`, and pointwise add control `0.988852838x`.
- Focused validation passed for Model Conv1d/Conv3d (21 tests), Compiler Conv3d (4 tests), and the
  prescribed CPU families/boundaries (123 tests after correcting a test-only artifact-cache key
  collision). The single authoritative `./gradlew :backends:cpu:test --rerun-tasks` passed 494
  tests with zero failures/errors and three established skips. Retained XML, forks, ledger,
  classes, checksums, inspection reports, controls, developmental samples, and manifest live at
  `/private/tmp/synaptik-cpu-0008a-retained-evidence-20260826`; the manifest-file SHA-256 is
  `dd185e49c691fb9761414c414c6bdf88507c463c109600deaeb62448c8c2292c`.
- Documentation context `01a03e87-9b14-7003-a6e9-e984bcf17144` independently reviewed the actual
  production, test, resource, and planning diff against `AGENTS.md`, `ARCHITECTURE.md`, the current
  architecture plan, documentation rules, the General/API-and-Javadoc/Developer-guide/Backend-
  guide/Planning/Example profiles, the CPU guide and glossary, the planning guide/roadmap/CPU
  master plan, this task, complete CPU 0008, Model 0025G/0025H, and Compiler 0006B contracts.
- The documentation pass finalized schema-52 and new Conv3d contract Javadocs, including the
  immutable IR, exact topology/lowering boundary, cold rank-five layouts, packed geometry,
  compatibility constructors, ownership, nullability, ranges, results, and failure conditions.
  It updated the CPU guide with the exact visible Conv1d recognizer, direct Conv3d numerical and
  resource behavior, scalar/parallel-scalar range ownership, carrier and overlap validation,
  intrinsic-bias-only epilogue boundary, schema 52, a checked geometry example, and bounded
  generated-code/performance evidence. It synchronized this task, the CPU master plan, and the
  roadmap so 0008A is Complete and 0008B is the sole Ready CPU task without a detailed spec.
- The glossary was reviewed without change. The implementation changes the current execution
  status of already defined Conv1d composition, Conv3d, generated-kernel, and schema concepts; it
  introduces no reusable term and changes none of those terms' semantic meanings. Current-status
  detail therefore remains in the CPU guide and planning evidence rather than duplicating it in
  the central terminology dictionary.

## Completion summary

CPU 0008A is complete. It recognizes only the exact visible NCW Conv1d two-expansion/Conv2d/
squeeze composition while keeping its singleton views virtual, and it adds direct grouped NCDHW
Conv3d generated CPU execution with optional intrinsic bias, arbitrary admitted non-negative
resolved layouts, heap/segment/mixed carriers, zero workspace/materialization, and scalar or
bounded parallel-scalar complete-output-cell work. External Conv3d epilogues and general DAG
handling fail closed. Generator schema is 52. The recorded implementation tests, generated-code
inspection, five-fork performance evidence, finalized Javadocs, CPU guide, planning status, and
documentation validation all pass without changing architecture or shared contracts. CPU 0008B
is the sole Ready CPU task and has no detailed task specification.

Status: Complete
