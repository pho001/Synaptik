# Task 0007A1L: [Pointwise](../../../../glossary.md#term-pointwise) general-loop [residual](../../../../glossary.md#term-residual-meanings) parity

## Reader context

The pointwise residual was a remaining timing gap, not saved differentiation state. Its row
fused FLOAT32 DIV, SIGMOID, and MUL but still had broadcast and strided boundaries. The class
generator recognized the exact structural instruction/carrier pattern; the loop emitter checked
concrete geometry and range state before selecting a bounded direct cursor loop. All other cases
retained the existing general-long state machine.

The optimization preserved each arithmetic site: FLOAT32 division, the stable sign-dependent
sigmoid using binary64 exponential and final narrowing, then FLOAT32 multiplication. Fusing
storage did not authorize changing that sequence or sharing run values. Runtime still bound
current inputs and the distinct output. Current typed IR and generation ownership explain the
division between code-shaping facts and per-invocation geometry.

Current reading alongside this historical record:

- [CPU typed intermediate representation](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalir)
  — distinguishes code-shaping structure from concrete geometry and live values.
- [CPU generated-code ownership](../../../../explanation/packages/cpu.md#iogithubpho001synaptikbackendcpuinternalcodegenemit)
  — separates generation-time emitters, emitted typed work, and JVM compilation.

## Status

Complete

## Goal

Correct exactly the persistent `P-SCALAR-GENERAL` generated/direct [residual](../../../../glossary.md#term-residual-meanings) through the cohesive
[pointwise](../../../../glossary.md#term-pointwise) loop-shaping boundary in `CpuClassFileKernelGenerator` and `CpuLoopEmitter`. Preserve the
frozen [CPU](../../../../glossary.md#term-cpu) 0007A1C [FLOAT32](../../../../glossary.md#term-float32) `DIV -> SIGMOID -> MUL` semantics, mixed carriers, broadcast input,
strided output, arbitrary legal [half-open](../../../../glossary.md#term-half-open-interval) ranges, and unchanged general-long fallback while
passing the frozen five-[fork](../../../../glossary.md#term-fork-meanings) `<= 1.15x` gate.

## [Scope](../../../../glossary.md#term-scope)

- Use the [immutable](../../../../glossary.md#term-immutable) A1C probe sources and the accepted A1K evidence bundle at
  `/private/tmp/synaptik-cpu-0007a1k-tSVmcoQh`.
- Correct only `P-SCALAR-GENERAL`: the [fused](../../../../glossary.md#specialization-and-fusion) [FLOAT32](../../../../glossary.md#term-float32) `DIV -> SIGMOID -> MUL` chain over [Shape](../../../../glossary.md#shape)
  `[512,512]`, with ordered carriers [`MemorySegment`](../../../../glossary.md#term-memorysegment), `float[]`, `MemorySegment`, `float[]`;
  dense first and third inputs; a last-axis broadcast second input; and an offset, two-strided
  general output.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>
- Preserve exact FLOAT32 division and multiplication and the existing stable SIGMOID formula,
  including its binary64 exponential work, sign branch, and one final narrowing. Do not substitute
  a different activation, approximation, vector body, formula order, or numerical policy.<span data-glossary-navigation="true"> ([SIGMOID glossary](../../../../glossary.md#term-sigmoid); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [narrowing glossary](../../../../glossary.md#term-narrowing); [activation glossary](../../../../glossary.md#term-activation); [vector glossary](../../../../glossary.md#term-vector))</span>
- Preserve arbitrary legal half-open output ranges, input immutability, canaries, output-domain
  ownership, complete overlap rejection, scalar or parallel-scalar reuse, and zero workspace.<span data-glossary-navigation="true"> ([half-open glossary](../../../../glossary.md#term-half-open-interval); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Treat the frozen optimal direct primitive Java loop as the generated-code design and review
  oracle. The guarded body may derive the row and column from an integer ordinal using the proved
  power-of-two Shape, then use direct typed carrier loads/stores and the existing scalar semantic
  emitter.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Require complete runtime geometry, range, and sentinel guards before entering any narrowed body.
  Every unproved pointwise topology, opcode sequence, carrier pattern, access form, extent, stride,
  offset, or range must retain the existing typed general-long state machine.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [pointwise glossary](../../../../glossary.md#term-pointwise); [topology glossary](../../../../glossary.md#term-topology); [opcode glossary](../../../../glossary.md#term-opcode); [extent glossary](../../../../glossary.md#term-extent); [stride glossary](../../../../glossary.md#term-stride))</span>
- Keep the generated [target](../../../../glossary.md#term-target) final and field-free with exactly one typed static `invoke`, no
  constructor or bridge, and no allocation, reflection, dynamic construct, generic `Object`
  descriptor, Runtime operation dispatch, or new Synaptik hot-path call.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [operation glossary](../../../../glossary.md#operation); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Advance the generator schema exactly once if emitted bytes change, propagate it to current
  schema assertions, and treat schema 39 and older envelopes as incompatible safe misses without
  migration or dual-schema reuse.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- Retain immutable inputs, exact commands, environment, semantics, generated classes, complete
  `javap -c` and `javap -v`, structural/hot-loop reports, five accepted forks, rejected samples,
  XML summaries, context identifiers, inventories, and checksums.<span data-glossary-navigation="true"> ([forks glossary](../../../../glossary.md#term-fork-meanings))</span><span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>

## Out of [scope](../../../../glossary.md#term-scope)

- Changing the frozen probe, case, Shape, carriers, comparator, JVM flags, fork protocol,
  threshold, direct oracle, or semantics.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [fork glossary](../../../../glossary.md#term-fork-meanings); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Correcting `S-GENERAL-MIN`, `X-MIN-MULTI`, or any completed row.
- Changing `CpuScalarEmitter` formulas, `CpuCarrierEmitter`, another family emitter, pointwise IR
  or lowering, fusion eligibility, access-plan meaning, validation, resources, materialization,
  orchestration, route selection, Prepare/Runtime behavior, capabilities, public API,
  dependencies, build structure, conformance, or integration.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lowering glossary](../../../../glossary.md#lowering); [fusion glossary](../../../../glossary.md#term-fusion); [validation glossary](../../../../glossary.md#term-validation); [materialization glossary](../../../../glossary.md#materialization); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Adding a generic fixed-Shape pointwise framework, per-element address table, workspace,
  materialization route, vector/native implementation, dynamic Shape/layout support, tuning
  input, or universal performance claim.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [vector glossary](../../../../glossary.md#term-vector); [layout glossary](../../../../glossary.md#layout))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [`current architecture plan`](../../../../architecture/current-architecture-plan.md)
- [`planning guide`](../../../planning-guide.md)
- [`documentation rules`](../../../../developer-guide/documentation-rules.md)
- [`CPU backend guide`](../../../../backend-guide/cpu-backend.md)
- [`glossary`](../../../../glossary.md)
- [`CPU master plan`](../master-plan.md)
- [`CPU 0005B access plans`](0005b-universal-access-plans-and-right-aligned-broadcasting.md)
- [`CPU 0005H unary closure`](0005h-portable-unary-transcendental-and-activation-closure.md)
- [`CPU 0007A0 generated parity`](0007a0-generated-hot-path-parity-correction.md)
- [`CPU 0007A1C evidence closure`](0007a1c-generated-direct-evidence-closure.md)
- [`CPU 0007A1K affine-copy residual parity`](0007a1k-affine-copy-residual-parity.md)

## Architecture constraints

- Model owns pointwise meaning and numerical semantics. This task changes only CPU-private
  generated loop realization and evidence.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [pointwise glossary](../../../../glossary.md#term-pointwise); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- CPU analysis retains fusion, access-plan lowering, strategy, resource declaration, and route
  selection before shared assignment; CPU finalization realizes one artifact afterward; Runtime
  only invokes prepared direct carriers, cold geometry, and range bounds.<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion); [lowering glossary](../../../../glossary.md#lowering); [artifact glossary](../../../../glossary.md#term-artifact); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Generated code must preserve the optimal clean Java algorithm and hot-loop/dataflow shape while
  keeping the general-long form for all unproved cases.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow))</span>
- Benchmark observations accept or reject this bounded generated body and never select or mutate
  production settings.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit` — existing pointwise class
  assembly, access-loop state machines, scalar semantic emission, and generated-kernel tests.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [kernel glossary](../../../../glossary.md#kernel))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.cache` — [schema](../../../../glossary.md#term-schema) compatibility when bytes change.
- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — existing schema assertion and
  current-version package wording only if the version advances.

Packages added or changed: None.

Type placement:

- `CpuClassFileKernelGenerator` remains the pointwise body assembler and may select only the
  completely proved frozen form before delegating its address state.
- `CpuLoopEmitter` remains the [owner](../../../../glossary.md#term-owner) of generated [pointwise](../../../../glossary.md#term-pointwise) loop/address state.
- `CpuScalarEmitter` remains the unchanged owner of the `DIV`, stable [`SIGMOID`](../../../../glossary.md#term-sigmoid), and `MUL`
  formulas.
- No helper, public type, package, or [module](../../../../glossary.md#term-module) may be added.

## Affected files

Expected implementation paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuClassFileKernelGenerator.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuLoopEmitter.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/cache/CpuGeneratorSchema.java`
- directly affected existing [cache](../../../../glossary.md#term-cache) and [prepare](../../../../glossary.md#prepare) package Javadocs only as required;
- `CpuPointwiseGeneratedKernelTest.java` and `CpuGeneratedDirectEvidenceClosureTest.java` only for
  stable semantic/Class-File assertions;
- `CpuGeneratedKernelArtifactStoreTest.java`, `CpuKernelSpecializationTest.java`,
  `CpuPartitionPreparerTest.java`, and `CpuPartitionFinalizerTest.java` only when schema advances;
  and<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema))</span>
- CPU backend guide, glossary, this task, CPU 0007A1C, CPU master plan, roadmap, and the sole next
  detailed residual task created only after this task completes.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend); [residual glossary](../../../../glossary.md#term-residual-meanings))</span>

## Maximum [scope](../../../../glossary.md#term-scope)

At most two production code-shaping [owners](../../../../glossary.md#term-owner) may change: the [pointwise](../../../../glossary.md#term-pointwise) body assembler and loop
state [owner](../../../../glossary.md#term-owner) named above. With [schema](../../../../glossary.md#term-schema)/package propagation, pointwise/closure tests, four schema
assertion owners, and documentation/[planning](../../../../glossary.md#term-planning), the overall change must remain at or below 18
repository paths. If `CpuScalarEmitter`, another production owner, [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/resource code, a
new helper, another [module](../../../../glossary.md#term-module), or a broader path set is required, stop and replan.

## Acceptance criteria

- Frozen sources recompile byte-identically and exact semantics remain `VERIFIED,20`.
- `P-SCALAR-GENERAL` passes `<= 1.15x` in every one of five fresh isolated forks and in the
  cross-fork median.
- Controls `P-VECTOR-SEGMENT`, `P-INTEGRAL-MIXED`, and `O-ARGSORT` pass every [fork](../../../../glossary.md#term-fork-meanings) and [median](../../../../glossary.md#term-median).
- The unchanged full probe exits nonzero only for `S-GENERAL-MIN` and `X-MIN-MULTI`.
- Exact output bits, input immutability, canaries, broadcast behavior, strided output mapping,
  arbitrary legal range ownership, and zero-resource contracts remain unchanged.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Target Class-File and complete disassembly prove the completely guarded typed pointwise loop and
  retained fallback, with no avoidable general odometer, repeated geometry lookup, coordinate
  division/remainder, generic dispatch, allocation, or helper call in the proved address loop.
  Calls required by the unchanged stable SIGMOID formula remain permitted.<span data-glossary-navigation="true"> ([Target glossary](../../../../glossary.md#term-target); [lookup glossary](../../../../glossary.md#term-lookup); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [dispatch glossary](../../../../glossary.md#term-dispatch); [SIGMOID glossary](../../../../glossary.md#term-sigmoid))</span>
- Focused [pointwise](../../../../glossary.md#term-pointwise)/closure/[schema](../../../../glossary.md#term-schema) [owners](../../../../glossary.md#term-owner) pass, followed by one uncached [CPU](../../../../glossary.md#term-cpu) suite.
- A distinct clean documentation context finalizes affected Javadocs, guide/glossary impact,
  evidence/status records, and documentation validation without rerunning stable Java tests.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>

## Tests / [validation](../../../../glossary.md#term-validation)

[Run](../../../../glossary.md#term-run) the focused `CpuPointwiseGeneratedKernelTest`, `CpuGeneratedDirectEvidenceClosureTest`, and
four [schema](../../../../glossary.md#term-schema)-assertion [owners](../../../../glossary.md#term-owner), followed once by:

```bash
./gradlew :backends:cpu:test --rerun-tasks
```

Recompile and [checksum](../../../../glossary.md#term-checksum) the frozen sources, run `VERIFIED,20`, retain the [target](../../../../glossary.md#term-target) and all required
control/deferred generated classes and complete disassembly, run five sequential fresh [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) [forks](../../../../glossary.md#term-jvm-fork)
with the unchanged `-Xms1g -Xmx1g` protocol, and regenerate the evidence inventory/checksums. A
complete [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) is accepted only if the target, controls, and every non-deferred row pass; retain
and reject the whole sample otherwise.

The documentation pass runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git diff --cached --check
git status --short
```

It also checks Markdown links/anchors/fences/final newlines, schema propagation, exact path bounds,
exactly one next [Ready](../../../../glossary.md#term-ready) task, A1C incomplete, A1L Complete only after every gate, [CPU](../../../../glossary.md#term-cpu) 0007A2
Draft/blocked, and empty staging. Repository-wide, architecture, [conformance](../../../../glossary.md#term-conformance), and integration
suites remain deferred to CPU 0009 or [CI](../../../../glossary.md#term-continuous-integration-ci) because no shared contract may change.

## Dependencies

- [CPU](../../../../glossary.md#term-cpu) 0007A1C supplies the [immutable](../../../../glossary.md#term-immutable) twenty-row corpus and remains incomplete.
- CPU 0005B and CPU 0007A0 supply the general access state machine and generated pointwise parity
  foundation.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- CPU 0005H supplies the exact stable [SIGMOID](../../../../glossary.md#term-sigmoid) formula and exceptional-value behavior.
- CPU 0007A1D supplies stable invocation-local segment [layout](../../../../glossary.md#layout) handling.
- CPU 0007A1K is Complete at [schema](../../../../glossary.md#term-schema) 39 and supplies the accepted five-[fork](../../../../glossary.md#term-fork-meanings) [residual](../../../../glossary.md#term-residual-meanings) frontier.

## Follow-up tasks

- Detailed [CPU 0007A1M scatter-MIN residual parity](0007a1m-scatter-min-residual-parity.md) is
  the sole next Ready task and owns <span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>`S-GENERAL-MIN`. `X-MIN-MULTI` remains Draft and unassigned.
- Resume [CPU](../../../../glossary.md#term-cpu) 0007A1C closure only after both remaining persistent rows close.
- CPU 0007A2 remains Draft and blocked behind CPU 0007A1C and all corrective [residual](../../../../glossary.md#term-residual-meanings) work.

## Architecture impact

Expected impact: None.

If the task requires an architecture, [dependency](../../../../glossary.md#term-dependency), public [API](../../../../glossary.md#term-application-programming-interface-api), semantic, numerical-policy,
resource, route, [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime), build, [IR](../../../../glossary.md#term-intermediate-representation-ir), [lowering](../../../../glossary.md#lowering), [conformance](../../../../glossary.md#term-conformance), or integration change, stop
and report it.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0007A1L. Work on the existing dirty
worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying
unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, CPU task 0007A1L, CPU tasks
0005B/0005H/0007A0/0007A1C/0007A1K, the relevant pointwise loop/generator/cache/package source and
tests, and the complete A1K evidence bundle. Implement exactly the bounded
P-SCALAR-GENERAL correction through CpuClassFileKernelGenerator and CpuLoopEmitter. Preserve the
immutable probe, exact FLOAT32 DIV/stable-SIGMOID/MUL formula, mixed carriers, last-axis
broadcast, strided output, arbitrary legal ranges, zero resources, typed fallback, and optimal
direct-clean-Java algorithm/dataflow shape. Stop before changing CpuScalarEmitter, a third
production owner, or any IR/lowering/resource/architecture/lifecycle contract.

Run the specified focused/full CPU, semantic/Class-File/five-fork, forbidden-reference, schema,
scope, and preservation gates. Hand the stabilized diff and exact evidence to a distinct clean
documentation context. Do not mark Complete until the target and all controls pass every gate.
Do not commit, push, or stage.
```

## Local decisions

- Accepted A1K [forks](../../../../glossary.md#term-fork-meanings) measure `P-SCALAR-GENERAL` at `2.641514165x`, `2.609853653x`,
  `2.607747319x`, `2.638802714x`, and `2.642773957x`, with [median](../../../../glossary.md#term-median) `2.638802714x`; the row is
  persistent rather than a one-fork anomaly.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>
- This row is selected before the two remaining residuals because it is first in the frozen
  residual order and has one cohesive pointwise loop-shaping boundary. The unchanged scalar
  semantic emitter already matches the direct formula; the defect evidence is confined to the
  generated general address state around that formula.<span data-glossary-navigation="true"> ([residual glossary](../../../../glossary.md#term-residual-meanings); [pointwise glossary](../../../../glossary.md#term-pointwise); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Scatter MIN and multi-axis aggregate MIN retain distinct copy/update/reduction and selected-
  domain algorithms and must not enter this task.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>

## Known limitations

- Ratios apply only to the frozen case, JVM, host, and protocol; they are not a universal
  pointwise, general-layout, or activation performance claim.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [host glossary](../../../../glossary.md#term-host); [pointwise glossary](../../../../glossary.md#term-pointwise); [layout glossary](../../../../glossary.md#layout); [activation glossary](../../../../glossary.md#term-activation))</span>
- This task does not close A1C or authorize [CPU](../../../../glossary.md#term-cpu) 0007A2.

## [Validation](../../../../glossary.md#term-validation) evidence

- Implementation context `01a02438-4fe2-7b40-b70e-2664d29a4095` produced the immutable evidence
  bundle at <span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>`/private/tmp/synaptik-cpu-0007a1l-PY33M3lJ` from repository base
  `d51dcfc8c00a1886596dd95545f99d1dc607c7fb`. The complete manifests, checksums, frozen-source and
  probe preservation reports, XML summaries, five raw fork outputs and summary, generated target
  class, complete <span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings))</span>`javap -c` and `javap -v`, direct oracle, schema transition, implementation
  patch, context, commands, environment, and unchanged-owner reports were independently reviewed.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [schema glossary](../../../../glossary.md#term-schema); [owner glossary](../../../../glossary.md#term-owner))</span>
- Frozen semantics passed exactly (`VERIFIED,20`). The focused invocation passed 62 tests in six
  suites. The one and only full CPU rerun passed 356 of 357 tests in 54 suites with zero failures
  or errors and one expected skip. No Java suite, semantic probe, timing fork, benchmark, or
  generated-class corpus was rerun by the documentation context.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
- Five accepted isolated forks measured `P-SCALAR-GENERAL` at `1.006810905x`, `1.005707592x`,
  `1.006301612x`, `1.004126974x`, and `1.006512287x`; [median](../../../../glossary.md#term-median) `1.006301612x`. All three controls
  passed every fork and median. Every accepted full fork leaves exactly `S-GENERAL-MIN` and
  `X-MIN-MULTI`; no rejected [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) occurred in this final [run](../../../../glossary.md#term-run).
- Complete [target](../../../../glossary.md#term-target) disassembly proves a final, field-free class with one typed static
  `invoke(MemorySegment, float[], MemorySegment, float[], long[], long, long)`. Complete guards
  precede the ordinal loop; its address body has shifts and masks but no integer or long division
  or remainder, performs direct <span data-glossary-navigation="true"> ([masks glossary](../../../../glossary.md#term-boolean-mask))</span>`JAVA_FLOAT_UNALIGNED` loads and a strided `float[]` store, and
  preserves FLOAT32 division/multiplication around the sign-branch binary64<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
  `StrictMath.exp` sigmoid with one final narrowing. The unchanged typed general-long body begins
  at the failed-guard target in the same method.<span data-glossary-navigation="true"> ([sigmoid glossary](../../../../glossary.md#term-sigmoid); [narrowing glossary](../../../../glossary.md#term-narrowing); [guard glossary](../../../../glossary.md#term-guard))</span>
- Documentation context `/root/doc_0007a1l` independently finalized affected internal Javadocs,
  package summaries, the CPU guide, glossary, and synchronized planning without changing
  executable Java or tests. It ran CPU Javadoc exactly once after edits and passed Markdown,
  schema/current-only wording, status/order, exact-scope, frozen-evidence, implementation-path
  preservation, staging, and whitespace checks recorded in the completion summary.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Javadoc glossary](../../../../glossary.md#term-javadoc); [scope glossary](../../../../glossary.md#term-scope))</span>

## Implementation notes

- `CpuClassFileKernelGenerator` recognizes only the exact frozen scalar general pointwise IR:
  FLOAT32 <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [pointwise glossary](../../../../glossary.md#term-pointwise); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [FLOAT32 glossary](../../../../glossary.md#term-float32))</span>`[512,512]`, `DIV -> SIGMOID -> MUL`, ordered carriers [`MemorySegment`](../../../../glossary.md#term-memorysegment), `float[]`,
  `MemorySegment`, `float[]`, dense first/third inputs, a last-axis-broadcast second input, and one
  offset two-strided output. <span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis))</span>`CpuLoopEmitter` admits the direct body only after the complete
  24-value geometry, ordered range, start coordinate/address, extent, stride, and sentinel proof.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [extent glossary](../../../../glossary.md#term-extent); [stride glossary](../../../../glossary.md#term-stride))</span>
- The proved loop narrows the bounded ordinal and end once, derives row/column with shift/mask
  operations, and performs direct typed carrier access. It preserves the existing stable sigmoid:
  FLOAT32 division, sign branch, binary64 <span data-glossary-navigation="true"> ([mask glossary](../../../../glossary.md#term-boolean-mask); [carrier glossary](../../../../glossary.md#term-storage-carrier); [sigmoid glossary](../../../../glossary.md#term-sigmoid); [binary64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>`StrictMath.exp`, one final FLOAT32 narrowing, then
  FLOAT32 multiplication. Every legal half-open range is supported; empty ranges return without
  work, and every failed proof retains the unchanged typed general-long state machine.<span data-glossary-navigation="true"> ([narrowing glossary](../../../../glossary.md#term-narrowing); [half-open glossary](../../../../glossary.md#term-half-open-interval))</span>
- Generated compatibility advances exactly once to schema 40. Schema 39 and older envelopes are
  incompatible safe misses; no migration, alias, converter, or dual-schema reuse was added.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- Architecture/current-plan/ADR/architecture-test conclusion: the bounded CPU-private generated
  body changes no architecture rule, module boundary, dependency direction, or shared contract,
  so no architecture contract, explanatory architecture page, ADR, or architecture test changes.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [CPU glossary](../../../../glossary.md#term-cpu); [module glossary](../../../../glossary.md#term-module); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [architecture contract glossary](../../../../glossary.md#architecture-contract); [architecture test glossary](../../../../glossary.md#term-architecture-test))</span>
- API/capability/lifecycle conclusion: public APIs and capability reporting, IR/lowering,
  materialization and resource declarations, route selection, Prepare/Runtime ownership, and
  unproved forms are unchanged. Backend conformance and integration tests therefore need no edit.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [lifecycle glossary](../../../../glossary.md#lifecycle); [lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Build/documentation conclusion: no dependency, Gradle/toolchain, shared-module, native/vector,
  example, or public API documentation change is required. The CPU guide and glossary record only
  the bounded current schema-40 behavior; affected internal Javadocs preserve its exact guard,
  numerical, range, addressing, and fallback boundaries.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [toolchain glossary](../../../../glossary.md#term-toolchain); [vector glossary](../../../../glossary.md#term-vector); [guard glossary](../../../../glossary.md#term-guard))</span>

## Completion summary

- Completed changes: added the completely guarded frozen pointwise ordinal body at schema 40,
  retained the typed fallback, synchronized schema assertions, and finalized directly affected
  Javadocs, package summaries, CPU guide, glossary, and planning records.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [schema glossary](../../../../glossary.md#term-schema); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: ten implementation-owned production/test paths plus eight
  documentation/Javadoc/planning paths, for 18 combined paths. No executable Java or test changed
  in the documentation pass.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Tests and [validation](../../../../glossary.md#term-validation): reused `VERIFIED,20`, six suites/62 focused tests, 54 suites/357 CPU tests
  with 356 passes and one expected skip, complete Class-File inspection, and five accepted target
  and control forks. The documentation pass ran CPU Javadoc exactly once and passed all required
  documentation, scope, preservation, staging, and whitespace checks.<span data-glossary-navigation="true"> ([forks glossary](../../../../glossary.md#term-fork-meanings))</span><span data-glossary-navigation="true"> ([target glossary](../../../../glossary.md#term-target); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation impact: internal generator/loop/schema/cache/code-generation/prepare Javadocs,
  CPU guide, glossary, this task, A1C, CPU master plan, roadmap, and the sole next detailed task
  are synchronized. No new public API or architecture term was introduced.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [prepare glossary](../../../../glossary.md#prepare); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Unresolved issues: exactly `S-GENERAL-MIN` and `X-MIN-MULTI` remain.
- Follow-up required: execute Ready CPU 0007A1M, then plan only the remaining evidence-supported
  owner-cohesive <span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [owner glossary](../../../../glossary.md#term-owner))</span>`X-MIN-MULTI` correction. [CPU](../../../../glossary.md#term-cpu) 0007A2 remains Draft and blocked until A1C closes.

Status: Complete
