# CPU0071: Common [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [binding](../../../../glossary.md#term-binding) and setup foundation

## Reader context

This completed extraction separates **generating code** from **running that code**. During CPU
preparation, ordinary MATMUL lowering and Scan body lowering call `CpuMatmulLowering` to check
matrix dimensions. Their emitters then call the common contraction-emission methods. The
ordinary emitter supplies entry operands and ranges; `CpuGeneralScanEmitter` supplies the
current time slice, carry version and activity condition. These are internal framework callers,
not separate application calls for every time step.

A generation-time callback writes bytecode into the class being built; it is not retained as a
callback invoked by the generated loop. Shared math therefore does not create a second Runtime
owner or a new per-step allocation. Read [current Scan composition](../../../../backend-guide/cpu-backend.md#current-general-scan-composition)
for the present relationship between ordinary units and Scan, and
[Scan source roles](../../../../reference/tensor/general-scan.md#formals-actual-positions-and-source-closure)
for what carry, sequence and invariant inputs describe. The timings and CPU0072 launch status
below belong to this checkpoint; they are not renewed by those current-reference links.

## Status and change class

Complete, 2026-10-29. Independent corrected source/[Javadoc](../../../../glossary.md#term-javadoc) review APPROVE; MAIN accepted
implementation and [validation](../../../../glossary.md#term-validation). Source writer closed; no active source or measurement frontier.
CPU0072 stays Draft/unlaunched. Class C: generated-code hot paths and temporal/resource proofs.

## Goal and [scope](../../../../glossary.md#term-scope)

Make existing ordinary [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) and current [scalar](../../../../glossary.md#term-scalar) Scan use common family [normalization](../../../../glossary.md#term-normalization)
and bound [contraction](../../../../glossary.md#contraction-dimension) emission. Separate invocation setup from [operation](../../../../glossary.md#operation) regions in existing
[owners](../../../../glossary.md#term-owner); retain ordinary direct/tiled scalar/[SIMD](../../../../glossary.md#simd) and CPU0065 [TANH](../../../../glossary.md#term-tanh). Move Scan's existing indexed/
cursor variants and activity semantics into common [ownership](../../../../glossary.md#term-ownership-meanings), then retire its MATMUL subemitter.
[Binding](../../../../glossary.md#term-binding) [model](../../../../glossary.md#term-model) and [non-fused example](../existing-emitter-composition-plan.md#non-fused-matmul-to-operation-boundary)<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion))</span>
remain in the replacement plan.

No all-family rewrite, new Scan family/[dtype](../../../../glossary.md#data-type--datatype)/[layout](../../../../glossary.md#layout)/route, ordinary symbolic [F64](../../../../glossary.md#floating-point-precision-and-rounding) MATMUL, second
body [graph](../../../../glossary.md#graph), [runtime](../../../../glossary.md#term-runtime) access layer, per-step [callback](../../../../glossary.md#term-callback)/allocation, worker change or new Runtime [owner](../../../../glossary.md#term-owner).
Existing Scan [pointwise](../../../../glossary.md#term-pointwise)/reduction math is retained pending CPU0072, not falsely reported retired.

Priority remains CPU0071 → separately approved CPU0072. Scan retains traversal/temporal bindings;
[`CpuWorkerGroup`](../../../../glossary.md#cpu-worker-group--cpuworkergroup) and per-submission Job stay unchanged. Optional Job reuse is not a prerequisite.
Later [synchronous](../../../../glossary.md#term-synchronous) worker composition needs the [separate gates](../existing-emitter-composition-plan.md#simd-admission-and-parallel-composition-gates).

## Contracts and [Ready](../../../../glossary.md#term-ready) gates

Read AGENTS and affected source/tests. Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
and [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span> route to [backend](../../../../glossary.md#backend)
[CPU routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> and
[Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>;
Scan [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>,
[Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates)<span data-glossary-navigation="true"> ([Qualification glossary](../../../../glossary.md#term-qualification))</span>,
[Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic),
[Qualification and mechanism gates](../../../../architecture/contracts/recurrent-scan.md#qualification-and-mechanism-gates),
[Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
and [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates)<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>;
[Runtime](../../../../glossary.md#term-runtime) [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>.
The implemented compatibility exception is owned only by backend execution
[CPU0071 approved static MATMUL compatibility cutover](../../../../architecture/contracts/backend-execution.md#cpu0071-approved-static-matmul-compatibility-cutover)<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>.
This brief records implementation/evidence, not authority or a successor launch.

No new production type/public [API](../../../../glossary.md#term-application-programming-interface-api)/framework or Runtime [handoff](../../../../glossary.md#term-handoff) was added; test fixtures are separate.
Future boundary changes require exact package/name/purpose/[signature](../../../../glossary.md#term-signature) and USER approval.

## Dependencies and integration

- Depends on: CPU0070 independently reviewed/MAIN accepted; retained CPU0064/0065 and accepted
  Compiler0015/0016/Prepare0015; explicit user implementation approval and Ready gates above.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Conflicts with: [CPU](../../../../glossary.md#term-cpu) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[carrier](../../../../glossary.md#term-storage-carrier)/loop/Scan/[cache](../../../../glossary.md#term-cache)/resource and CPU0048–0052 writers.
- Parallel group: None; one clean implementation writer, then independent targeted review.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty work;
  MAIN's preservation receipt below verifies this accepted base, not an assumed merged base.
- Integration order: this cutover → independent authority review + MAIN Ready/GO → fresh clean
  code worker → independent code/docs review → CPU module/Javadoc + named conformance/Engine
  gates → optional single measurement/evidence reuse → MAIN acceptance; CPU0072 stays Draft.<span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Integration validation: worker and MAIN commands below, documentation links/anchors,
  whitespace, changed-file/hash preservation and exact retained-capability checks.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [hash glossary](../../../../glossary.md#term-hash))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; final eight-document pass delegated to Ptolemy.

MAIN verified accepted predecessor/authority/[signature](../../../../glossary.md#term-signature) gates and the 2518-file `cpu71-code-start.json`
base; accepted dirty work is not merged history. [Immutable](../../../../glossary.md#term-immutable) before-source:
`/private/tmp/synaptik-cpu0071-main.wsdT4h/cpu-src`; MAIN owns preservation and the final join.

## Files and symbols

Source prefix: `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`.

- `lowering/CpuMatmulLowering.lower`/`BatchGeometry`, `ir/CpuMatmulIr`/`CpuAccessPlan`: checked [normalization](../../../../glossary.md#term-normalization).
- `codegen/emit/` [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[carrier](../../../../glossary.md#term-storage-carrier)/loop emitters and generator: separate entry setup from [contraction](../../../../glossary.md#contraction-dimension).
- Scan [lowering](../../../../glossary.md#lowering) `analyzeShaped/analyzeSymbolic`, [IR](../../../../glossary.md#term-intermediate-representation-ir) and `CpuGeneralScanEmitter.Shaped` adapters:
  temporal adaptation; preserve `CpuGeneralScanStageSchedule.order`, resources and existing fallback.
- `cache/CpuKernelSpecialization`/`CpuGeneratorSchema`, IR identity and named tests/oracles below:
  both identity streams, frozen artifacts and generated/direct evidence.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [artifacts glossary](../../../../glossary.md#term-artifact))</span>

## Implemented seams and approved policy

Existing `CpuMatmulLowering.lower` and Scan [lowering](../../../../glossary.md#lowering) now call package-local `validateRankTwo`
for exact [Shape](../../../../glossary.md#shape)/[Dimension](../../../../glossary.md#dimension) equations, never sampled B. `BatchGeometry` retains [F32](../../../../glossary.md#floating-point-precision-and-rounding) guards.
`CpuMatmulEmitter.emitCursorContraction`/`emitIndexedContraction` own increasing full-K control
and represented math. Generation-only callbacks inline [bytecode](../../../../glossary.md#term-bytecode), never [runtime](../../../../glossary.md#term-runtime) calls; cursor
loaders fill typed locals and indexed loaders leave operands on stack. Callers retain typed +0/
address initialization and known-empty static K omission. Scan supplies temporal/activity/store
[bindings](../../../../glossary.md#term-binding) without passing Shaped/Stage to common math. No public [signature](../../../../glossary.md#term-signature) or production type added.

`CpuCarrierEmitter.prepareSegmentLayouts` still reserves six [layout](../../../../glossary.md#layout) locals first; combined entries
omit unused initialization. Compatible [vector species](../../../../glossary.md#term-vector-species)/[byte order](../../../../glossary.md#text-encoding-and-file-boundaries) remain invocation-local.
`Shaped.prepareMatmulStrides` binds static [constants](../../../../glossary.md#term-constant) outside time, not evolving [scalar](../../../../glossary.md#term-scalar)→dense
factors. Symbolic/combined selected [stride](../../../../glossary.md#term-stride) handles are unchanged. The approved exception is
implemented in both [specialization](../../../../glossary.md#term-specialization) streams; retained [identities](../../../../glossary.md#term-identity-meanings)/bytes remain [validation](../../../../glossary.md#term-validation) [targets](../../../../glossary.md#term-target),
not universal proof. Unexpected changed namespaces still require review.

## Acceptance criteria

1. Ordinary and Scan rank-two callers reach common MATMUL normalization and contraction owners.
   No math entry accepts Scan <span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [normalization glossary](../../../../glossary.md#term-normalization); [contraction glossary](../../../../glossary.md#contraction-dimension))</span>`Shaped`/Stage as its semantic owner or delegates back to a renamed
   Scan-only implementation. Existing unrelated high-rank/low/integral ordinary forms retain
   their guards and code; no universal-loop refactor is required.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [rank glossary](../../../../glossary.md#term-rank))</span>
2. CodeBuilder alone allocates [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) locals. Distinguish those locals from packed `long[]` indices
   and shared Runtime selections. Layout block is reserved once; geometry/species/layout and
   invariant-stride setup dominates use outside time, including empty/tail/fallback paths.
   Compatible dtype/access setup is reused; F32/F64/mask species cannot be conflated. Repeated
   generation-time prepare calls must not emit repeated runtime setup within time.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Layout glossary](../../../../glossary.md#layout); [invariant glossary](../../../../glossary.md#term-invariant); [stride glossary](../../../../glossary.md#term-stride); [dtype glossary](../../../../glossary.md#data-type--datatype); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [mask glossary](../../../../glossary.md#term-boolean-mask); [prepare glossary](../../../../glossary.md#prepare))</span>
3. Common activity/proof inputs express ROW output gating, ROW(1) active K and SHARED any-active
   work. Active-K guard precedes <span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard))</span>**both** operand loads and dependent math; ordinary unguarded
   bindings remain valid. Preserve increasing contributors, full K, K=0 typed zero, old/next
   carry simultaneous lifetimes, original-time histories and selected-forward exact saved bits.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding); [carry glossary](../../../../glossary.md#term-carry))</span>
4. Non-fused MATMUL → operation uses its declared intermediate/demanded port, completes the
   producer domain before consumers/cut reuse and adds no mandatory copy. No callback, nested
   prepared executable, geometry repack, graph lookup/re-lowering or allocation per step,
   including tensor buffers; no T-fold graph/code expansion.<span data-glossary-navigation="true"> ([fused glossary](../../../../glossary.md#specialization-and-fusion); [operation glossary](../../../../glossary.md#operation); [producer glossary](../../../../glossary.md#term-producer); [callback glossary](../../../../glossary.md#term-callback); [prepared executable glossary](../../../../glossary.md#prepared-executable--preparedexecutable); [graph glossary](../../../../glossary.md#graph); [lookup glossary](../../../../glossary.md#term-lookup); [lowering glossary](../../../../glossary.md#lowering); [tensor glossary](../../../../glossary.md#tensor); [buffers glossary](../../../../glossary.md#term-buffer))</span>
5. Preserve v9 stable carriers/temporal bases, explicit-B affine/zero-base proofs, cell sharing,
   indexed active-row/feature contractions and bank-free schedule/fallback, inclusive all-B cut
   lifetimes and physical final-port hazards. Retain CPU0065 vector TANH/tails and ordinary
   worker/range behavior. No Scan SIMD/parallel admission follows from common math.<span data-glossary-navigation="true"> ([feature glossary](../../../../glossary.md#term-feature); [vector glossary](../../../../glossary.md#term-vector); [TANH glossary](../../../../glossary.md#term-tanh); [SIMD glossary](../../../../glossary.md#simd); [admission glossary](../../../../glossary.md#term-admission))</span>
6. Qualify all retained static/symbolic/length forward and actual Compiler reverse occurrences
   independently: both traversals/seeds, F32/F64 only where already admitted, ties/two carries,
   shared parameter sums, poison padding, zero/full/mixed/invalid lengths, T=0/B=0/1/empty K/
   zero features, 64→40→1→64 reuse, dense/scalar reads, alias/overflow/cleanup/retry/concurrency.
   Do not weaken ordinary batch F32-only guards or admit symbolic all-valid F64.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [parameter glossary](../../../../glossary.md#term-parameter); [padding glossary](../../../../glossary.md#padding); [scalar glossary](../../../../glossary.md#term-scalar); [alias glossary](../../../../glossary.md#aliasing-and-arena); [overflow glossary](../../../../glossary.md#term-overflow); [concurrency glossary](../../../../glossary.md#term-concurrency); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
7. Remove `CpuGeneralScanMatmulEmitter` only after its callers qualify. Preserve SUM algorithms
   unchanged and unaffected keys/bytes; list every changed artifact, advance both affected
   identity streams and reject stale inputs per the approved policy. No actual B/L/addresses/
   slots enter canonical identity. No new mutable prepared state or Runtime handoff.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings); [canonical glossary](../../../../glossary.md#term-canonical-form); [mutable glossary](../../../../glossary.md#term-mutable); [handoff glossary](../../../../glossary.md#term-handoff))</span>

## [Validation](../../../../glossary.md#term-validation)

Recorded validation commands below completed; this historical record requests no rerun:

```bash
./gradlew :backends:cpu:test --tests '*CpuMatmulBindingSnapshotTest' :backends:cpu:javadoc
./gradlew :backends:cpu:test :backends:cpu:javadoc
```

MAIN capability-preservation [checkpoint](../../../../glossary.md#term-checkpoint), once after independent review:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*CpuSymbolicScanConformanceTest' --tests '*CpuMatmulStrategyConformanceTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest' --tests '*EngineCpuSymbolicScanIntegrationTest' --tests '*EngineCpuMatmulStrategyIntegrationTest'
```

ClassFile/clean-Java checks cover loop/[dataflow](../../../../glossary.md#term-dataflow), setup dominance, activity-before-load and v9 proofs.
Historical evidence remains [immutable](../../../../glossary.md#term-immutable); v9 timings do not qualify changed static [shapes](../../../../glossary.md#shape).
After source review and named gates MAIN launched `run-conditional.sh --main-prelaunch-go` once.
The USER-approved protocol used one measured [JVM](../../../../glossary.md#term-java-virtual-machine-jvm), eight rows
([F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) × static all-valid/static-length × forward/actual [Compiler](../../../../glossary.md#term-compiler) reverse), original traversal
forward and sibling reverse, T4/B7/F3/H2. All-valid: `next=emit=h+x@W`; length mode:
`next=emit=tanh(x@W+h@U+bias)`, L=`[4,0,1,3,2,4,1]`. Baseline/new/direct-Java arms;
10000 [warmup](../../../../glossary.md#term-warmup) calls per arm/row, 15 measured triples × 1000 calls, cyclic arm order; hard 60-second
JVM watchdog deadline, no retry/repeat/additional [fork](../../../../glossary.md#term-fork-meanings). Storage was preallocated with before/after
checks. Cold [compile](../../../../glossary.md#compile)/generate/define, allocation/[binding](../../../../glossary.md#term-binding) and warmed entry are separate; [Engine](../../../../glossary.md#term-engine)
[admission](../../../../glossary.md#term-admission) was not measured. Generated arms include [MethodHandle](../../../../glossary.md#native-numerical-library-and-instruction-architecture)/spreader versus direct-call costs.
Legacy timing selectors, including dynamicBatch.performance, stay disabled; no root/native rebuild.

## Result

Completed 2026-10-29: common [normalization](../../../../glossary.md#term-normalization)/cursor/indexed math and static [stride](../../../../glossary.md#term-stride) setup in existing
[owners](../../../../glossary.md#term-owner); `CpuGeneralScanMatmulEmitter` deleted after caller [qualification](../../../../glossary.md#term-qualification). Five test-only
`CpuMatmulBinding*` classes provide [snapshot](../../../../glossary.md#term-snapshot)/[binding](../../../../glossary.md#term-binding) tests, [fixture](../../../../glossary.md#term-fixture), clean Java and explicit harness.
Source freeze: 31 [CPU](../../../../glossary.md#term-cpu)-src paths (14 modified, 16 added, one retired), plus CPU guide/brief.
No new production type, numerical rule, resource/[Runtime](../../../../glossary.md#term-runtime) [handoff](../../../../glossary.md#term-handoff), worker/Job or Scan route.

Both streams implement only static all-valid [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) v4→v5 and retained-MATMUL static-length
v1→length-aware-matmul v2, with stale definition/persisted-[cache](../../../../glossary.md#term-cache) rejection. Non-MATMUL shaped-v4/
length-v1, symbolic-v3, combined-v9, [pointwise](../../../../glossary.md#term-pointwise)/fixed and ordinary controls retain [identities](../../../../glossary.md#term-identity-meanings)/bytes
in the recorded corpus; [canonical](../../../../glossary.md#term-canonical-form) projections/[schema](../../../../glossary.md#term-schema) 71/packs/resources remain unchanged.
[Finite](../../../../glossary.md#term-finite)-corpus preservation is evidence, not proof for every [specialization](../../../../glossary.md#term-specialization).

Evidence root: `/private/tmp/synaptik-cpu0071-code.BvOdPR`; original [captures](../../../../glossary.md#term-capture-meanings)/raw [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) are [immutable](../../../../glossary.md#term-immutable).

- Frozen matrix: 226 cases, 68 approved changes/158 exact retained; 194 geometry witnesses and
  eight actual old ClassFiles. Another 21 ordinary controls cover batch-v1 F32 forward/dX/dW and
  typed-v1 F64 arithmetic/SUM/MEAN/AD, recovered unmeasured from immutable before-source, not a
  current-source self-baseline. Vector golden bytes require captured 128-bit preferred species;
  other hosts explicitly report unqualified vector rows without dropping scalar checks.<span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-matrix-meanings); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [MEAN glossary](../../../../glossary.md#term-mean); [AD glossary](../../../../glossary.md#autograd); [Vector glossary](../../../../glossary.md#term-vector); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Full CPU/Javadoc PASS: 1385 tests, zero failures/errors, 53 reported skips; 94 pre-existing
  Javadoc warnings (<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`final-cpu-test-javadoc.log`). Subsequent species-test/two-Javadoc correction:
  three snapshot tests, no skips/failures/errors, Javadoc PASS (`review-correction-validation.log`).
  Independent targeted re-review APPROVE, no unresolved correctness/documentation findings.
- MAIN joins PASS: conformance 19/Engine 26 tests, no skips/failures/errors
  (<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>`main-named-joins-authorized.log`). Initial sandbox cache-lock failure preceded tests
  (`main-named-joins.log`); authorized existing-cache launch succeeded in six seconds.
- Sole measurement exit 0, eight before/after checks PASS (`conditional-performance.log`).
  Budget consumed/closed: no repeat, retry, additional fork or inherited CPU0072 budget.
  New/old median ratios 0.9881–1.0313 establish no speedup or steady-state parity.
  F64 length-forward direct samples transition from about 30 µs through 16.311 µs to 0.83 µs;
  its aggregate ratio is not valid performance evidence. F32 length-reverse new samples also
  transition from about 32 µs to 0.98 µs; its direct arm has an early transient.
  Fixed warmup was insufficient for some arms; no compilation logs prove a C2 or other cause.
  Length-reverse new/direct medians about 1.50 F32/1.42 F64 remain an unresolved performance
  question, not correctness failure or proof of a new regression. All-valid ratios are likewise
  harness diagnostics, not optimal-Java or emitted-inner-loop parity proof.<span data-glossary-navigation="true"> ([fork glossary](../../../../glossary.md#term-fork-meanings); [median glossary](../../../../glossary.md#term-median); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [warmup glossary](../../../../glossary.md#term-warmup); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [C2 glossary](../../../../glossary.md#term-c2); [regression glossary](../../../../glossary.md#term-regression))</span>

Documentation impact: affected Javadocs and CPU/Scan explanations updated; root/owning exception
and [planning](../../../../glossary.md#term-planning) statuses synchronized. General/Architecture/Planning profiles applied; targeted glossary
review found no new reusable meaning. No new [ADR](../../../../glossary.md#term-architecture-decision-record-adr)/architecture edge. Final docs reuse executable
evidence without JVMs. No required foundation follow-up; CPU0072 stays Draft and unlaunched.

Status: Complete
