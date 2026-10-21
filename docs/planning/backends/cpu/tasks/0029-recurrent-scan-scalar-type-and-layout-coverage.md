# Task CPU0029: Recurrent scan scalar type and layout coverage

## Status

Complete — accepted 2026-10-21 after independent Class C code/docs APPROVE and
main's source-matched native-enabled root test/Javadoc checkpoint.

## Change class

Class C — numerical, binding, scratch and hot-loop changes; clean implementation and independent
targeted code/documentation review mandatory.

## Goal and scope

Complete direct-Java scalar fixed runtime-length scans; preserve CPU0028's lifecycle.

- All six biased/bias-free signatures, FORWARD/REVERSE, exact Model equations/packing. One common
  FLOAT32/FLOAT64/BFLOAT16/FLOAT16 type for all floating inputs/outputs; lengths remain non-gradient
  INT64 `[B]`. Floating gradient flags are accepted with exact output OR, not backward permission.
- Static input `[T,B,I]`, states `[B,H]`, weights `[G*H,I]`/`[G*H,H]`, optional bias `[G*H]`,
  G=1/3/4. I/H positive, T/B nonnegative. Every dimension, packed extent, logical boundary count
  and referenced storage-element span fits `int`; checked `long` address/byte/scratch arithmetic.
- Every input, including lengths, admits Model-valid resolved nonnegative strides/offsets:
  dense, offset, permuted/transposed, sliced/gapped, positive-stride overlapping reads and
  zero-stride repeated reads. Non-singleton zero-stride repetition requires Model's view flag.
  Address = offset + sum(coordinate*stride); span is not logical count. Empty layouts have span
  zero regardless of unused offset and must not be accessed. Negative storage strides/offsets
  remain unrepresentable/rejected; REVERSE changes traversal, not layout. Unresolved/dynamic
  layouts remain unsupported. Outputs stay canonical zero-offset dense `[T,B,H]`/`[B,H]`.

## Non-goals

No BPTT/backward/tape, static NN migration, NN/Data workflow widening, Metal/CUDA, vector/parallel/
native scan, active-row packing, tuning/default changes, public or primitive APIs, Engine
orchestration, dependency/format changes, generated scan, generic loop/region infrastructure,
new execution protocol or speculative abstraction. No architecture/authority edits.

## Contracts

Start at root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Read these exact authoritative headings, not completed task histories:

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules),
  [Explicit non-goals](../../../../../ARCHITECTURE.md#explicit-non-goals), [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Recurrent scan [Fixed recurrent scan without graph regions](../../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions)
  and all five subheadings: [Fixed family and current Model surface](../../../../architecture/contracts/recurrent-scan.md#fixed-family-and-current-model-surface),
  [Static Shape and runtime-value boundary](../../../../architecture/contracts/recurrent-scan.md#static-shape-and-runtime-value-boundary),
  [Valid lengths, traversal, and outputs](../../../../architecture/contracts/recurrent-scan.md#valid-lengths-traversal-and-outputs),
  [Purity and lifecycle ownership](../../../../architecture/contracts/recurrent-scan.md#purity-and-lifecycle-ownership),
  [Performance and migration invariants](../../../../architecture/contracts/recurrent-scan.md#performance-and-migration-invariants).
- Backend execution [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules),
  [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes),
  [Low-precision capability evidence](../../../../architecture/contracts/backend-execution.md#low-precision-capability-evidence).
- Foundational [Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics):
  [Family formulas and unary sites](../../../../architecture/contracts/foundational-modules.md#family-formulas-and-unary-sites),
  [Arithmetic-site DAZ/FTZ](../../../../architecture/contracts/foundational-modules.md#arithmetic-site-dazftz),
  [Exact and discrete floor](../../../../architecture/contracts/foundational-modules.md#exact-and-discrete-floor),
  [Floating arithmetic primitives](../../../../architecture/contracts/foundational-modules.md#floating-arithmetic-primitives),
  [Aggregate floor](../../../../architecture/contracts/foundational-modules.md#aggregate-floor),
  [Composite inheritance](../../../../architecture/contracts/foundational-modules.md#composite-inheritance),
  [Model low-precision contract](../../../../architecture/contracts/foundational-modules.md#model-low-precision-contract).
- Runtime/Prepare [modules/runtime](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
  [modules/prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle).

Stop on missing/conflicting/ambiguous authority; do not invent numerical rules. No blocker found.

## Dependencies and integration

- Depends on: CPU0028 and CPU0022, Complete (scan baseline and existing low working-site policy).
- Conflicts with: scan admission/lowering/kernel/binding/resources, numerical policy, current
  ledger and shared explanatory/planning edits; serialize these scopes.
- Parallel group: None; sole serial CPU workstream, no concurrent writer authorized.
- Common base revision: `f918d7d7dab3fff0b144747f56d629733bd443c7` plus this planning-only packet.
- Integration order: main launch verification → clean CPU executor → independent Class C
  code/docs review/corrections → main source-matched checkpoint → main status/frontier closure.
- Integration validation: focused conformance/public Engine below, then main's root `test javadoc`;
  reuse worker evidence without duplicate review suites.
- Shared-document integration owner: main; executor drafts explanation/Javadoc,
  independent review finalizes serially; main alone owns shared planning/closure.

Verified 2026-10-21: HEAD equals base; CPU0028/CPU0022 Complete with accepted reviews/validation.
NN0022A/B remain Complete; no new NN dependency. CPU0029 was user-selected sole Ready CPU frontier;
CPU0023 Draft, conflicts inactive. Main rechecked before launching the sole serial implementation executor.
Preserve user-owned `THIRD_PARTY_NOTICES.md` and
`modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuOrdinaryPointwiseCastCleanJavaOracle.java` unchanged.

## Files and symbols

CPU paths use `modules/backend/cpu/src/{main,test}/java/io/github/pho001/synaptik/backend/cpu/`.

- `internal/lowering/CpuRecurrentScanLowering.java` — Geometry/Facts, role/layout/span bindings;
  `CpuCapabilityProvider`, `internal/prepare/{CpuPartitionPreparer,CpuPartitionPreparationPlan,CpuPartitionFinalizer}`
  — admission, scratch bytes/alignment and finalization only.
- `internal/executable/CpuRecurrentScanPreparedExecutable.java` — cold bindings; retain existing
  `internal/route/portable/CpuRecurrentScanKernel.java` for FLOAT32. Added three internal concrete
  types in that same package: `CpuRecurrentScanFloat64Kernel`, `CpuRecurrentScanBFloat16Kernel`,
  `CpuRecurrentScanFloat16Kernel`; addressing remains in existing lowering/binding types.
  No common execution protocol or per-element dtype dispatch.
- Read `RecurrentScanKind`, `LayoutDescriptor`/`LayoutGeometry`, `BFloat16Bits`/`Float16Bits`,
  `CpuMatmulReferenceKernel`, `CpuScalarReferenceKernel.lowActivation`/scan sigmoid.
  Reuse conversions/math, not boxed execution, emitters or a matmul protocol.
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
   dtype-correct double[]/float[]/raw short[]/long[]; Engine fresh values retain full referenced span.
2. Preserve heap/native-order segment/sliced/mixed bindings and read-only inputs. Reject wrong
   observable carriers, dead/wrong-thread, short, misaligned and read-only writable bindings.
   Real read/read aliasing is legal; retain conservative whole-selected-representation overlap
   rejection for read/write, sibling writes and scratch/any boundary. No gap-reuse expansion.
   Separate physical slice base/layout offset; preserve prefix/gaps/capacity tails.
3. Binding reads no runtime lengths. Execution rechecks access and validates all logically strided
   lengths after producers, before padding/state copies/any scan write. Last-row -1/T+1/
   Long.MAX_VALUE leaves every sibling unchanged, then repaired invocation/preparation is reusable.
   Earlier producers need not roll back. Poisoned gaps must not be mistaken for length values.
4. Preserve original-time placement, reverse valid-prefix order, exact +0 padding and empty T/B.
   Gather zero-length/zero-time initial states raw (NaN payload, -0, subnormal) into final states.
   Padded NaN/Inf never enters dots/gates/updates. Each role gets offset/strided/zero-stride tests
   with poisoned gaps and positive-stride overlapping reads.
5. FLOAT64 uses double work/accumulation, no float downcast or DAZ/FTZ; FLOAT32 retains its sites.
   Low inputs decode exactly, work/accumulate in FLOAT32, narrow once ties-to-even per published
   value, and store hidden/cell state exactly in declared type for subsequent state reads.
   No wider carry, premature low contributor/accumulator/gate narrowing or implicit double work.
   Respect stored-state barriers including LSTM cell→hidden consumption, and named sigmoid/EXP/TANH
   sites; do not narrow every implicit composite subexpression. Test F64 `1+2^-30`, Double.MIN_VALUE
   and 8-byte scratch; multi-step low GRU/LSTM with off-diagonal recurrent weights distinguishes
   premature gate narrowing, double work and unrounded carry, including LSTM cell→hidden.
6. One immutable scalar recipe/occurrence, graph/preparation size constant in T, isolated run state.
   Declare exact two-packed-row scratch before slots: width/alignment 8 for FLOAT64, otherwise 4;
   independent of T/lengths, omitted for empty T/B. No prototype storage/per-time executable growth.
   No generated schema bump for this untuned direct route; stop if serialized compatibility must change.
7. Select dtype/family/carrier/layout cold. Direct primitive fixed-rank loops have no per-element
   dispatch/boxing/allocation, generic reader/virtual arithmetic, maps/strings, reflection, graph
   inspection or input materialization. Finish prior-hidden projections before replacement.
   Independent optimal clean Java oracle expresses same-case algorithm/dataflow, not production calls.
8. Qualify each family/direction/bias/type with test-only finite oracles, explicit metrics,
   thresholds, fixture sizes and cancellation bounds, separate from exact state/publication,
   NaN/Inf/overflow/underflow/subnormal/sign checks. Public compile/prepare/run/runValues covers
   all types, representative strided roles, changed/produced lengths, finalCell-only/reordered
   ports, downstream consumers, earlier open results and concurrent isolation. Preserve forward
   flags, fail-closed Compiler backward admission and static NN/inference-only workflows.

## Validation

Verified JDK 26.0.1 / Gradle 9.6.1 and real Metal library. Gradle requires `require_escalated`
for cache access. No planner Gradle; reviewer reuses evidence.

```bash
export JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home
./gradlew :backends:cpu:test --tests '*CpuRecurrentScan*' --tests '*CpuCapabilityProviderTest'
./gradlew :backends:cpu:test
./gradlew :testing:backend-conformance:test --tests '*CpuRecurrentScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest' :testing:integration-tests:test --tests '*EngineCpuRecurrentScanIntegrationTest' --tests '*FluentModelBuilderIntegrationTest' --tests '*NumericSequenceInference*'
SYNAPTIK_CPU_RECURRENT_SCAN_TIMING=true ./gradlew :backends:cpu:test --tests '*CpuRecurrentScanScalarTimingTest' --rerun-tasks
./gradlew :backends:cpu:javadoc
git diff --check
```

Timing is opt-in diagnostic only: at most eight fixed small/moderate cases (four dtypes, dense
and one strided representative, covering three families), one warmed JVM, 120-second measurement
deadline. Compare prebound route and specialized clean Java on identical carriers/equations,
outside setup/allocation; consume checksums, retain all samples/JDK/target/shapes/ratios.
No timing threshold, speed/parity claim, tuning or expanding benchmark matrix. Inspect source and
`javap -c -p` of compiled kernel/oracle for dispatch/allocation/downcasts, not generated-code parity.
Record inconclusive timing honestly; semantic/structural acceptance remains mandatory.

After final independent review, main runs one source-matched checkpoint:

```bash
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc
```

## Documentation and review impact

Update affected CPU Javadocs, CPU guide's `Fixed recurrent forward execution`, boundary explanation's
`Fixed recurrent scan handoff`, fixed-scan glossary/current execution claims and
`docs/api/tensor-api.md#fixed-recurrent-scan-expressions`. Review running-models/graph-checkpoint CPU
claims; no NN/API/persistence/training widening. Apply general + backend/API profiles and targeted
glossary search. Independent Class C review checks final
code/tests, low/state boundaries, strided atomic preflight, resources/aliasing, clean Java hot loops
and final docs/links. Generate CPU Javadoc after final edits; reuse executable evidence. No ADR or
architecture-test changes expected. Docs-only NN scope: `ModelBuilder` class Javadoc in
`extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModelBuilder.java` and, if needed,
`RuntimeLengthScanModelTest` comment: remove stale CPU FLOAT32-only claims, no NN code/API change.

## Result

Implementation frozen and accepted 2026-10-21; Raman's independent Class C final code/docs
review: **APPROVE**, zero P1/P2 findings; 499 link checks complete. Main's root gate passed.

- Reviewed CPU production: `CpuRecurrentScanLowering`, `CpuRecurrentScanPreparedExecutable`,
  `CpuPartitionPreparer`, `CpuPartitionPreparationPlan`, `CpuRecurrentScanKernel`,
  `CpuRecurrentScanFloat64Kernel`, `CpuRecurrentScanBFloat16Kernel`, `CpuRecurrentScanFloat16Kernel`.
  Final changes since core review were Javadocs and removal of an unused cold local; hot dataflow is unchanged.
- Reviewed tests: `CpuRecurrentScanExecutionTest`, `CpuRecurrentScanLoweringTest`, `CpuInternalPackageInventoryTest`,
  `CpuRecurrentScanCleanJavaOracle`, `CpuRecurrentScanScalarTimingTest`, `CpuRecurrentScanConformanceTest`,
  `LowPrecisionCapabilityLedgerTest`, `EngineCpuRecurrentScanIntegrationTest`; current v2 changes only the
  three CPU scan low answers (117 true per low type); historical v1, Metal and unrelated dirty files unchanged.
- Reused successful executor commands below with JDK 26.0.1 and escalated cache access; no reviewer reruns:

```bash
./gradlew :backends:cpu:test --tests '*CpuRecurrentScan*' --tests '*CpuCapabilityProviderTest' :testing:backend-conformance:test --tests '*CpuRecurrentScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuRecurrentScanIntegrationTest' --tests '*FluentModelBuilderIntegrationTest' --tests '*NumericSequenceInference*'
./gradlew :backends:cpu:test :backends:cpu:javadoc
SYNAPTIK_CPU_RECURRENT_SCAN_TIMING=true ./gradlew :backends:cpu:test --tests '*CpuRecurrentScanScalarTimingTest' --tests '*CpuInternalPackageInventoryTest' --rerun-tasks
```

- Complete CPU/Javadoc PASS, 6m08s: 210 suites, 1,109 tests, 31 optional/conditional skips, no failures/errors;
  14 execution + 5 lowering + 29 provider + 2 inventory pass. Javadoc: 94 existing warnings, no errors.
  Conformance 2 + ledger 7 and Engine 5 + Fluent 5 + numeric inference 3 pass, all zero skips.
  Earlier inventory allowlist and fixed low rounding-discriminator fixtures were corrected before these gates.
- Timing PASS: exactly the same 8/8 cases, 1.864s, JDK 26.0.1+8-34 on macOS/aarch64.
  [Session-local raw eight-case XML](/private/tmp/synaptik-cpu0029-evidence.aUysNh/timing-prebound-route-v1.xml)
  retains all five samples/side, shapes, dtypes, layouts, checksums and ratios (0.88–5.84).
  Mutable carriers/slicing/spans/alignment and native scratch match; setup/allocation are outside timing.
  Route includes BoundInvocation lease/access/length checks; bare preflighted oracle does not. Sample phase
  shifts leave warm-up/JIT effects unresolved. No causal attribution, parity/speed claim or tuning conclusion.
  No third pure-kernel column or new measurements were performed.
- Finalized CPU/API/architecture/user/glossary explanations, CPU kernel/binding Javadocs, ModelBuilder class
  Javadoc and two test comments. Applied general, API/Javadoc, backend/developer, user, architecture,
  example and planning profiles. No executable/API/authority/ABI/schema/dependency or NN/Data scope change.
  Link/anchor and whitespace checks plus comment-stripped Java/protected-file hash checks pass.
  Main's root Javadoc covers final comment-only edits; successful executable evidence is reused.
- Main's exact source-matched root command:

```bash
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc
```

- Root PASS: exit 0, BUILD SUCCESSFUL in 5m27s; 119 tasks, 21 executed/98 UP-TO-DATE.
  Current XML includes cached/UP-TO-DATE evidence, not all freshly run: CPU 210 suites/1,109 tests/31 skips;
  Architecture 25/0 skips, Conformance 52/0, Integration 187/1, NN 524/0, Engine 136/0;
  zero failures/errors. The sole integration skip is
  `EngineModelAutotuningIntegrationTest.eligibleCpuAlternativesCompleteBothPublicTuningPhases`:
  host fixture has no eligible local-workload alternatives, unrelated to scan. CPU skips are
  optional/conditional, including normally disabled timing; all eight opt-in timing cases ran earlier.
- Main verified all 11 source/protected SHA-256 hashes match before/after the gate, unchanged HEAD
  `f918d7d7dab3fff0b144747f56d629733bd443c7`, and passing `git diff --check`.
  Source remains frozen; no accepted work remains. CPU frontier None; no automatic successor launch.
- Size exception: retained scoped contracts, atomic Class C acceptance and exact validation commands make
  this brief exceed 200 lines/15 KB; compact Result points to existing raw evidence, keeping it below 25 KiB.

Status: Complete
