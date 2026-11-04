# Task CPU0029: Recurrent [scan](../../../../glossary.md#term-cumulative-scan) [scalar](../../../../glossary.md#term-scalar) type and [layout](../../../../glossary.md#layout) coverage

## Reader context — historical integration boundary

This extends the fixed RNN/GRU/LSTM family, not the captured-body general Scan family. A Tensor
descriptor supplies logical coordinates and strides; borrowed heap/native storage contains values at
those addresses. Strided or repeated read views therefore need exact span checks without a carrier
copy, while outputs remain distinct dense run-owned buffers. Low-precision inputs work at the
specified FLOAT32 sites and narrow stored state as recorded; later time steps read that stored
state. Runtime owns each run's writable state, not the input owner or reusable recipe. The timing
comparison includes different route/oracle entry checks and cannot isolate kernel speed by itself.

Current reading: [Fixed recurrent execution](../../../../backend-guide/cpu-backend.md#fixed-recurrent-forward-execution).

## Status

Complete — accepted 2026-10-21 after independent Class C code/docs APPROVE and
main's source-matched native-enabled root test/[Javadoc](../../../../glossary.md#term-javadoc) [checkpoint](../../../../glossary.md#term-checkpoint).

## Change class

Class C — numerical, [binding](../../../../glossary.md#term-binding), scratch and hot-loop changes; clean implementation and independent
targeted code/documentation review mandatory.

## Goal and [scope](../../../../glossary.md#term-scope)

Complete direct-Java [scalar](../../../../glossary.md#term-scalar) fixed [runtime](../../../../glossary.md#term-runtime)-length scans; preserve CPU0028's [lifecycle](../../../../glossary.md#lifecycle).

- All six biased/bias-free signatures, FORWARD/REVERSE, exact Model equations/packing. One common
  FLOAT32/FLOAT64/BFLOAT16/FLOAT16 type for all floating inputs/outputs; lengths remain non-gradient
  INT64 <span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [Model glossary](../../../../glossary.md#term-model); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [BFLOAT16 glossary](../../../../glossary.md#term-bfloat16); [FLOAT16 glossary](../../../../glossary.md#term-float16); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>`[B]`. Floating [gradient](../../../../glossary.md#term-gradient) flags are accepted with exact output OR, not backward permission.
- Static input `[T,B,I]`, states `[B,H]`, [weights](../../../../glossary.md#term-weight) `[G*H,I]`/`[G*H,H]`, optional [bias](../../../../glossary.md#term-bias) `[G*H]`,
  G=1/3/4. I/H positive, T/B nonnegative. Every dimension, packed extent, logical boundary count
  and referenced storage-element span fits <span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension); [extent glossary](../../../../glossary.md#term-extent); [element glossary](../../../../glossary.md#term-element))</span>`int`; checked `long` address/byte/scratch arithmetic.
- Every input, including lengths, admits Model-valid resolved nonnegative strides/offsets:
  dense, offset, permuted/transposed, sliced/gapped, positive-stride overlapping reads and
  zero-stride repeated reads. Non-singleton zero-stride repetition requires Model's view flag.
  Address = offset + sum(coordinate*stride); span is not logical count. Empty layouts have span
  zero regardless of unused offset and must not be accessed. Negative storage strides/offsets
  remain unrepresentable/rejected; REVERSE changes traversal, not layout. Unresolved/dynamic
  layouts remain unsupported. Outputs stay canonical zero-offset dense <span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [singleton glossary](../../../../glossary.md#term-singleton); [view glossary](../../../../glossary.md#term-view-meanings); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [layouts glossary](../../../../glossary.md#layout); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>`[T,B,H]`/`[B,H]`.

## Non-goals

No [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt)/backward/tape, static [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) migration, NN/Data workflow [widening](../../../../glossary.md#term-widening), [Metal](../../../../glossary.md#term-metal)/[CUDA](../../../../glossary.md#term-cuda), [vector](../../../../glossary.md#term-vector)/parallel/
native [scan](../../../../glossary.md#term-cumulative-scan), active-row packing, tuning/default changes, public or primitive APIs, [Engine](../../../../glossary.md#term-engine)
orchestration, [dependency](../../../../glossary.md#term-dependency)/format changes, generated scan, generic loop/region infrastructure,
new execution protocol or speculative abstraction. No architecture/authority edits.

## Contracts

Start at root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>.
Read these exact authoritative headings, not completed task histories:

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>,
  [Explicit non-goals](../../../../../ARCHITECTURE.md#explicit-non-goals), [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Recurrent [scan](../../../../glossary.md#term-cumulative-scan) [Fixed recurrent scan without graph regions](../../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions)<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
  and all five subheadings: [Fixed family and current Model surface](../../../../architecture/contracts/recurrent-scan.md#fixed-family-and-current-model-surface)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model))</span>,
  [Static Shape and runtime-value boundary](../../../../architecture/contracts/recurrent-scan.md#static-shape-and-runtime-value-boundary)<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>,
  [Valid lengths, traversal, and outputs](../../../../architecture/contracts/recurrent-scan.md#valid-lengths-traversal-and-outputs),
  [Purity and lifecycle ownership](../../../../architecture/contracts/recurrent-scan.md#purity-and-lifecycle-ownership)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>,
  [Performance and migration invariants](../../../../architecture/contracts/recurrent-scan.md#performance-and-migration-invariants).
- [Backend](../../../../glossary.md#backend) execution [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>,
  [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>,
  [Low-precision capability evidence](../../../../architecture/contracts/backend-execution.md#low-precision-capability-evidence)<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision))</span>.
- Foundational [Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([Profile-free numerical semantics glossary](../../../../glossary.md#profile-free-numerical-semantics))</span>:
  [Family formulas and unary sites](../../../../architecture/contracts/foundational-modules.md#family-formulas-and-unary-sites)<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary))</span>,
  [Arithmetic-site DAZ/FTZ](../../../../architecture/contracts/foundational-modules.md#arithmetic-site-dazftz)<span data-glossary-navigation="true"> ([DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz))</span>,
  [Exact and discrete floor](../../../../architecture/contracts/foundational-modules.md#exact-and-discrete-floor),
  [Floating arithmetic primitives](../../../../architecture/contracts/foundational-modules.md#floating-arithmetic-primitives),
  [Aggregate floor](../../../../architecture/contracts/foundational-modules.md#aggregate-floor)<span data-glossary-navigation="true"> ([Aggregate floor glossary](../../../../glossary.md#term-aggregate-floor))</span>,
  [Composite inheritance](../../../../architecture/contracts/foundational-modules.md#composite-inheritance),
  [Model low-precision contract](../../../../architecture/contracts/foundational-modules.md#model-low-precision-contract).
- [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare) [modules/runtime](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
  [modules/prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>.

Stop on missing/conflicting/ambiguous authority; do not invent numerical rules. No blocker found.

## Dependencies and integration

- Depends on: CPU0028 and CPU0022, Complete ([scan](../../../../glossary.md#term-cumulative-scan) baseline and existing low working-site policy).
- Conflicts with: scan admission/lowering/kernel/binding/resources, numerical policy, current
  ledger and shared explanatory/planning edits; serialize these scopes.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [binding glossary](../../../../glossary.md#term-binding))</span>
- Parallel group: None; sole serial [CPU](../../../../glossary.md#term-cpu) workstream, no concurrent writer authorized.
- Common base revision: `f918d7d7dab3fff0b144747f56d629733bd443c7` plus this [planning](../../../../glossary.md#term-planning)-only packet.
- Integration order: main launch verification → clean CPU executor → independent Class C
  code/docs review/corrections → main source-matched checkpoint → main status/frontier closure.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Integration [validation](../../../../glossary.md#term-validation): focused [conformance](../../../../glossary.md#term-conformance)/public [Engine](../../../../glossary.md#term-engine) below, then main's root `test javadoc`;
  reuse worker evidence without duplicate review suites.
- Shared-document integration owner: main; executor drafts explanation/Javadoc,
  independent review finalizes serially; main alone owns shared planning/closure.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

Verified 2026-10-21: HEAD equals base; CPU0028/CPU0022 Complete with accepted reviews/validation.
NN0022A/B remain Complete; no new [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [dependency](../../../../glossary.md#term-dependency). CPU0029 was user-selected sole [Ready](../../../../glossary.md#term-ready) CPU frontier;
CPU0023 Draft, conflicts inactive. Main rechecked before launching the sole serial implementation executor.
Preserve user-owned `THIRD_PARTY_NOTICES.md` and
`modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastCleanJavaOracle.java` unchanged.

## Files and symbols

[CPU](../../../../glossary.md#term-cpu) paths use `modules/backend/cpu/src/{main,test}/java/io/github/pho001/synaptik/backend/cpu/`.

- `internal/lowering/CpuRecurrentScanLowering.java` — Geometry/Facts, role/[layout](../../../../glossary.md#layout)/span [bindings](../../../../glossary.md#term-binding);
  `CpuCapabilityProvider`, `internal/prepare/{CpuPartitionPreparer,CpuPartitionPreparationPlan,CpuPartitionFinalizer}`
  — admission, scratch bytes/alignment and finalization only.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- `internal/executable/CpuRecurrentScanPreparedExecutable.java` — cold bindings; retain existing
  `internal/route/portable/CpuRecurrentScanKernel.java` for FLOAT32. Added three internal concrete
  types in that same package: <span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32))</span>`CpuRecurrentScanFloat64Kernel`, `CpuRecurrentScanBFloat16Kernel`,
  `CpuRecurrentScanFloat16Kernel`; addressing remains in existing lowering/binding types.
  No common execution protocol or per-element dtype dispatch.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [element glossary](../../../../glossary.md#term-element); [dtype glossary](../../../../glossary.md#data-type--datatype); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Read `RecurrentScanKind`, `LayoutDescriptor`/`LayoutGeometry`, `BFloat16Bits`/`Float16Bits`,
  `CpuMatmulReferenceKernel`, `CpuScalarReferenceKernel.lowActivation`/scan sigmoid.
  Reuse conversions/math, not boxed execution, emitters or a matmul protocol.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [sigmoid glossary](../../../../glossary.md#term-sigmoid))</span>
- Extend `CpuRecurrentScanLoweringTest`/`CpuRecurrentScanExecutionTest`; add test-only
  `internal/route/portable/CpuRecurrentScanCleanJavaOracle.java` and
  `internal/route/portable/CpuRecurrentScanScalarTimingTest.java`.
- `testing/backend-conformance`: `CpuRecurrentScanConformanceTest`, `LowPrecisionCapabilityLedgerTest`,
  `src/test/resources/low-precision-capability-ledger-v2.tsv` — independent current dtype answers
  and negative predicates; never rewrite historical v1.
- `testing/integration-tests`: `EngineCpuRecurrentScanIntegrationTest`; retain FluentModelBuilder/
  NumericSequenceInference regressions.

## Acceptance criteria

1. Provider/lowerer agree on scoped positives and role/shape/flag/mixed-type/layout/overflow
   negatives. Deduplicate reads without losing roles/siblings. Non-INT64 float[] fixtures become
   dtype-correct double[]/float[]/raw short[]/long[]; Engine fresh values retain full referenced span.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [layout glossary](../../../../glossary.md#layout); [overflow glossary](../../../../glossary.md#term-overflow); [INT64 glossary](../../../../glossary.md#data-type--datatype); [dtype glossary](../../../../glossary.md#data-type--datatype); [Engine glossary](../../../../glossary.md#term-engine); [referenced span glossary](../../../../glossary.md#referenced-element-span))</span>
2. Preserve heap/native-order segment/sliced/mixed bindings and read-only inputs. Reject wrong
   observable carriers, dead/wrong-thread, short, misaligned and read-only writable bindings.
   Real read/read aliasing is legal; retain conservative whole-selected-representation overlap
   rejection for read/write, sibling writes and scratch/any boundary. No gap-reuse expansion.
   Separate physical slice base/layout offset; preserve prefix/gaps/capacity tails.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [bindings glossary](../../../../glossary.md#term-binding); [aliasing glossary](../../../../glossary.md#term-aliasing); [slice glossary](../../../../glossary.md#slice))</span>
3. Binding reads no runtime lengths. Execution rechecks access and validates all logically strided
   lengths after producers, before padding/state copies/any scan write. Last-row -1/T+1/
   Long.MAX_VALUE leaves every sibling unchanged, then repaired invocation/preparation is reusable.
   Earlier producers need not roll back. Poisoned gaps must not be mistaken for length values.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [padding glossary](../../../../glossary.md#padding); [scan glossary](../../../../glossary.md#term-cumulative-scan))</span>
4. Preserve original-time placement, reverse valid-prefix order, exact +0 padding and empty T/B.
   Gather zero-length/zero-time initial states raw (NaN payload, -0, subnormal) into final states.
   Padded NaN/Inf never enters dots/gates/updates. Each role gets offset/strided/zero-stride tests
   with poisoned gaps and positive-stride overlapping reads.<span data-glossary-navigation="true"> ([Gather glossary](../../../../glossary.md#gather); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [subnormal glossary](../../../../glossary.md#term-subnormal); [NaN glossary](../../../../glossary.md#term-nan); [stride glossary](../../../../glossary.md#term-stride))</span>
5. FLOAT64 uses double work/accumulation, no float downcast or DAZ/FTZ; FLOAT32 retains its sites.
   Low inputs decode exactly, work/accumulate in FLOAT32, narrow once ties-to-even per published
   value, and store hidden/cell state exactly in declared type for subsequent state reads.
   No wider carry, premature low contributor/accumulator/gate narrowing or implicit double work.
   Respect stored-state barriers including LSTM cell→hidden consumption, and named sigmoid/EXP/TANH
   sites; do not narrow every implicit composite subexpression. Test F64 <span data-glossary-navigation="true"> ([FLOAT64 glossary](../../../../glossary.md#term-float64); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [FLOAT32 glossary](../../../../glossary.md#term-float32); [ties-to-even glossary](../../../../glossary.md#term-ties-to-even); [carry glossary](../../../../glossary.md#term-carry); [accumulator glossary](../../../../glossary.md#accumulator-type); [narrowing glossary](../../../../glossary.md#term-narrowing); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm); [sigmoid glossary](../../../../glossary.md#term-sigmoid); [EXP glossary](../../../../glossary.md#term-expx); [TANH glossary](../../../../glossary.md#term-tanh); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>`1+2^-30`, Double.MIN_VALUE
   and 8-byte scratch; multi-step low GRU/LSTM with off-diagonal recurrent weights distinguishes
   premature gate narrowing, double work and unrounded carry, including LSTM cell→hidden.<span data-glossary-navigation="true"> ([GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru); [weights glossary](../../../../glossary.md#term-weight))</span>
6. One immutable scalar recipe/occurrence, graph/preparation size constant in T, isolated run state.
   Declare exact two-packed-row scratch before slots: width/alignment 8 for FLOAT64, otherwise 4;
   independent of T/lengths, omitted for empty T/B. No prototype storage/per-time executable growth.
   No generated schema bump for this untuned direct route; stop if serialized compatibility must change.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [scalar glossary](../../../../glossary.md#term-scalar); [occurrence glossary](../../../../glossary.md#term-occurrence); [graph glossary](../../../../glossary.md#graph); [constant glossary](../../../../glossary.md#term-constant); [run state glossary](../../../../glossary.md#run-state--runstate); [alignment glossary](../../../../glossary.md#term-alignment); [prototype glossary](../../../../glossary.md#term-prototype); [schema glossary](../../../../glossary.md#term-schema))</span>
7. Select dtype/family/carrier/layout cold. Direct primitive fixed-rank loops have no per-element
   dispatch/boxing/allocation, generic reader/virtual arithmetic, maps/strings, reflection, graph
   inspection or input materialization. Finish prior-hidden projections before replacement.
   Independent optimal clean Java oracle expresses same-case algorithm/dataflow, not production calls.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [rank glossary](../../../../glossary.md#term-rank); [element glossary](../../../../glossary.md#term-element); [dispatch glossary](../../../../glossary.md#term-dispatch); [materialization glossary](../../../../glossary.md#materialization); [oracle glossary](../../../../glossary.md#term-test-oracle); [dataflow glossary](../../../../glossary.md#term-dataflow))</span>
8. Qualify each family/direction/bias/type with test-only finite oracles, explicit metrics,
   thresholds, fixture sizes and cancellation bounds, separate from exact state/publication,
   NaN/Inf/overflow/underflow/subnormal/sign checks. Public compile/prepare/run/runValues covers
   all types, representative strided roles, changed/produced lengths, finalCell-only/reordered
   ports, downstream consumers, earlier open results and concurrent isolation. Preserve forward
   flags, fail-closed Compiler backward admission and static NN/inference-only workflows.<span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias); [finite glossary](../../../../glossary.md#term-finite); [metrics glossary](../../../../glossary.md#machine-learning-and-hyperparameters); [fixture glossary](../../../../glossary.md#term-fixture); [cancellation glossary](../../../../glossary.md#term-cancellation); [publication glossary](../../../../glossary.md#term-publication); [underflow glossary](../../../../glossary.md#term-underflow); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [Compiler glossary](../../../../glossary.md#term-compiler); [admission glossary](../../../../glossary.md#term-admission); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [inference glossary](../../../../glossary.md#term-inference))</span>

## [Validation](../../../../glossary.md#term-validation)

Verified [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26.0.1 / [Gradle](../../../../glossary.md#term-gradle) 9.6.1 and real [Metal](../../../../glossary.md#term-metal) library. Gradle requires `require_escalated`
for [cache](../../../../glossary.md#term-cache) access. No planner Gradle; reviewer reuses evidence.

```bash
export JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home
./gradlew :backends:cpu:test --tests '*CpuRecurrentScan*' --tests '*CpuCapabilityProviderTest'
./gradlew :backends:cpu:test
./gradlew :testing:backend-conformance:test --tests '*CpuRecurrentScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest' :testing:integration-tests:test --tests '*EngineCpuRecurrentScanIntegrationTest' --tests '*FluentModelBuilderIntegrationTest' --tests '*NumericSequenceInference*'
SYNAPTIK_CPU_RECURRENT_SCAN_TIMING=true ./gradlew :backends:cpu:test --tests '*CpuRecurrentScanScalarTimingTest' --rerun-tasks
./gradlew :backends:cpu:javadoc
git diff --check
```

Timing is opt-in diagnostic only: at most eight fixed small/moderate cases (four [dtypes](../../../../glossary.md#data-type--datatype), dense
and one strided representative, covering three families), one warmed [JVM](../../../../glossary.md#term-java-virtual-machine-jvm), 120-second measurement
deadline. Compare prebound route and specialized clean Java on identical carriers/equations,
outside setup/allocation; consume checksums, retain all [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)/JDK/[target](../../../../glossary.md#term-target)/[shapes](../../../../glossary.md#shape)/ratios.
No timing threshold, speed/parity claim, tuning or expanding benchmark [matrix](../../../../glossary.md#term-review-matrix). Inspect source and
`javap -c -p` of compiled [kernel](../../../../glossary.md#kernel)/[oracle](../../../../glossary.md#term-test-oracle) for [dispatch](../../../../glossary.md#term-dispatch)/allocation/downcasts, not generated-code parity.
Record inconclusive timing honestly; semantic/structural acceptance remains mandatory.

After final independent review, main runs one source-matched [checkpoint](../../../../glossary.md#term-checkpoint):

```bash
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc
```

## Documentation and review impact

Update affected [CPU](../../../../glossary.md#term-cpu) Javadocs, CPU guide's `Fixed recurrent forward execution`, boundary explanation's
`Fixed recurrent scan handoff`, fixed-[scan](../../../../glossary.md#term-cumulative-scan) glossary/current execution claims and
`docs/api/tensor-api.md#fixed-recurrent-scan-expressions`. Review running-models/[graph](../../../../glossary.md#graph)-[checkpoint](../../../../glossary.md#term-checkpoint) CPU
claims; no [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[API](../../../../glossary.md#term-application-programming-interface-api)/[persistence](../../../../glossary.md#term-persistence)/training [widening](../../../../glossary.md#term-widening). Apply general + [backend](../../../../glossary.md#backend)/API profiles and targeted
glossary search. Independent Class C review checks final
code/tests, low/state boundaries, strided [atomic](../../../../glossary.md#resource-transaction-and-rollback) [preflight](../../../../glossary.md#term-preflight), resources/[aliasing](../../../../glossary.md#term-aliasing), clean Java hot loops
and final docs/links. Generate CPU [Javadoc](../../../../glossary.md#term-javadoc) after final edits; reuse executable evidence. No [ADR](../../../../glossary.md#term-architecture-decision-record-adr) or
architecture-test changes expected. Docs-only NN [scope](../../../../glossary.md#term-scope): `ModelBuilder` class Javadoc in
`extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModelBuilder.java` and, if needed,
`RuntimeLengthScanModelTest` comment: remove stale CPU [FLOAT32](../../../../glossary.md#term-float32)-only claims, no NN code/API change.

## Result

Implementation frozen and accepted 2026-10-21; Raman's independent Class C final code/docs
review: **APPROVE**, zero P1/P2 findings; 499 link checks complete. Main's root gate passed.

- Reviewed [CPU](../../../../glossary.md#term-cpu) production: `CpuRecurrentScanLowering`, `CpuRecurrentScanPreparedExecutable`,
  `CpuPartitionPreparer`, `CpuPartitionPreparationPlan`, `CpuRecurrentScanKernel`,
  `CpuRecurrentScanFloat64Kernel`, `CpuRecurrentScanBFloat16Kernel`, `CpuRecurrentScanFloat16Kernel`.
  Final changes since core review were Javadocs and removal of an unused cold local; hot dataflow is unchanged.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow))</span>
- Reviewed tests: `CpuRecurrentScanExecutionTest`, `CpuRecurrentScanLoweringTest`, `CpuInternalPackageInventoryTest`,
  `CpuRecurrentScanCleanJavaOracle`, `CpuRecurrentScanScalarTimingTest`, `CpuRecurrentScanConformanceTest`,
  `LowPrecisionCapabilityLedgerTest`, `EngineCpuRecurrentScanIntegrationTest`; current v2 changes only the
  three CPU scan low answers (117 true per low type); historical v1, Metal and unrelated dirty files unchanged.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [Metal glossary](../../../../glossary.md#term-metal))</span>
- Reused successful executor commands below with [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26.0.1 and escalated [cache](../../../../glossary.md#term-cache) access; no reviewer reruns:

```bash
./gradlew :backends:cpu:test --tests '*CpuRecurrentScan*' --tests '*CpuCapabilityProviderTest' :testing:backend-conformance:test --tests '*CpuRecurrentScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuRecurrentScanIntegrationTest' --tests '*FluentModelBuilderIntegrationTest' --tests '*NumericSequenceInference*'
./gradlew :backends:cpu:test :backends:cpu:javadoc
SYNAPTIK_CPU_RECURRENT_SCAN_TIMING=true ./gradlew :backends:cpu:test --tests '*CpuRecurrentScanScalarTimingTest' --tests '*CpuInternalPackageInventoryTest' --rerun-tasks
```

- Complete CPU/Javadoc PASS, 6m08s: 210 suites, 1,109 tests, 31 optional/conditional skips, no failures/errors;
  14 execution + 5 lowering + 29 provider + 2 inventory pass. Javadoc: 94 existing warnings, no errors.
  Conformance 2 + ledger 7 and Engine 5 + Fluent 5 + numeric inference 3 pass, all zero skips.
  Earlier inventory allowlist and fixed low rounding-discriminator fixtures were corrected before these gates.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [lowering glossary](../../../../glossary.md#lowering); [Conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [inference glossary](../../../../glossary.md#term-inference); [rounding glossary](../../../../glossary.md#term-rounding))</span>
- Timing PASS: exactly the same 8/8 cases, 1.864s, JDK 26.0.1+8-34 on macOS/[aarch64](../../../../glossary.md#term-aarch64).
  [Session-local raw eight-case XML](/private/tmp/synaptik-cpu0029-evidence.aUysNh/timing-prebound-route-v1.xml)
  retains all five samples/side, shapes, dtypes, layouts, checksums and ratios (0.88–5.84).
  Mutable carriers/slicing/spans/alignment and native scratch match; setup/allocation are outside timing.
  Route includes BoundInvocation lease/access/length checks; bare preflighted oracle does not. Sample phase
  shifts leave warm-up/JIT effects unresolved. No causal attribution, parity/speed claim or tuning conclusion.
  No third pure-kernel column or new measurements were performed.<span data-glossary-navigation="true"> ([samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [shapes glossary](../../../../glossary.md#shape); [dtypes glossary](../../../../glossary.md#data-type--datatype); [layouts glossary](../../../../glossary.md#layout); [Mutable glossary](../../../../glossary.md#term-mutable); [alignment glossary](../../../../glossary.md#term-alignment); [BoundInvocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [lease glossary](../../../../glossary.md#term-lease); [oracle glossary](../../../../glossary.md#term-test-oracle); [warm-up glossary](../../../../glossary.md#benchmark-measurement-vocabulary); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit); [kernel glossary](../../../../glossary.md#kernel))</span>
- Finalized CPU/API/architecture/user/glossary explanations, CPU kernel/binding Javadocs, ModelBuilder class
  Javadoc and two test comments. Applied general, API/Javadoc, backend/developer, user, architecture,
  example and planning profiles. No executable/API/authority/ABI/schema/dependency or NN/Data scope change.
  Link/anchor and whitespace checks plus comment-stripped Java/protected-file hash checks pass.
  Main's root Javadoc covers final comment-only edits; successful executable evidence is reused.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [binding glossary](../../../../glossary.md#term-binding); [backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [dependency glossary](../../../../glossary.md#term-dependency); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [scope glossary](../../../../glossary.md#term-scope); [hash glossary](../../../../glossary.md#term-hash))</span>
- Main's exact source-matched root command:

```bash
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc
```

- Root PASS: exit 0, BUILD SUCCESSFUL in 5m27s; 119 tasks, 21 executed/98 UP-TO-DATE.
  Current XML includes cached/UP-TO-DATE evidence, not all freshly run: CPU 210 suites/1,109 tests/31 skips;
  Architecture 25/0 skips, Conformance 52/0, Integration 187/1, NN 524/0, Engine 136/0;
  zero failures/errors. The sole integration skip is<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
  `EngineModelAutotuningIntegrationTest.eligibleCpuAlternativesCompleteBothPublicTuningPhases`:
  host fixture has no eligible local-workload alternatives, unrelated to scan. CPU skips are
  optional/conditional, including normally disabled timing; all eight opt-in timing cases ran earlier.<span data-glossary-navigation="true"> ([host glossary](../../../../glossary.md#term-host); [fixture glossary](../../../../glossary.md#term-fixture))</span>
- Main verified all 11 source/protected [SHA-256](../../../../glossary.md#term-sha-256) [hashes](../../../../glossary.md#term-cryptographic-hash) match before/after the gate, unchanged HEAD
  `f918d7d7dab3fff0b144747f56d629733bd443c7`, and passing `git diff --check`.
  Source remains frozen; no accepted work remains. CPU frontier None; no automatic successor launch.
- Size exception: retained scoped contracts, atomic Class C acceptance and exact validation commands make
  this brief exceed 200 lines/15 KB; compact Result points to existing raw evidence, keeping it below 25 KiB.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [validation glossary](../../../../glossary.md#term-validation))</span>

Status: Complete
