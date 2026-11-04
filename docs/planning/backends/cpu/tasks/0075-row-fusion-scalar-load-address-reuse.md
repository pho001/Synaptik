# Task CPU0075: [Scalar](../../../../glossary.md#term-scalar) row-[fusion](../../../../glossary.md#term-fusion) load and address reuse

## Reader context

The generator is optimizing repeated **reads of the same cell within one active row**, not
copying every input into a new tensor. It may put a previously loaded represented value in a
typed JVM local and reuse it only while the address and stored value are proved unchanged.
“Store epoch” distinguishes periods separated by a write that might change that cell; an
uncertain alias or intervening write requires another checked read.

`CpuRowMapReductionLowering` and preparation prove the lifetime and local-slot budget before
generation; the ordinary generator and Scan emitter consume those proofs. JVM local slots are
generated method variables, not Runtime buffer slots and not a count of physical CPU registers.
The [load/address reuse explanation](../../../../backend-guide/cpu-backend.md#implemented-scalar-loadaddress-reuse-cutover)
gives a cell-reuse example. Compare [later invariant staging](../../../../backend-guide/cpu-backend.md#cpu0076-implemented-scalar-invariant-staging)
for a deliberately wider lifetime: this task alone does not hoist matrix loads outside rows
or retain carry values across time. Old/new timing ratios below retain their unequal-JIT and
entry-wrapper limitations; fewer load sites alone do not prove a speedup.

## Status

Complete — 2026-10-30, independent final code/[Javadoc](../../../../glossary.md#term-javadoc)/docs APPROVE and MAIN owning gate PASS.
The sole comparison and source frontier are closed; no successor or further timing is authorized.

## Change class

Class C — changes generated hot-loop [dataflow](../../../../glossary.md#term-dataflow) and selective [artifact](../../../../glossary.md#term-artifact) compatibility, not numerics.
Clean implementation and an independent targeted code/[Javadoc](../../../../glossary.md#term-javadoc)/documentation review are required.

## Goal

Replace affected ordinary/general-Scan [scalar](../../../../glossary.md#term-scalar) row emission with within-row external-load reuse
and [constant](../../../../glossary.md#term-constant) affine address folding/sharing: generic reuse, not F3/H2 hardcoding or old/new emitters.

## [Scope](../../../../glossary.md#term-scope) and non-goals

- Existing F32/F64 scalar and parallel-scalar row bodies where already admitted; ordinary array
  and checked-segment eligibility, Scan checked segments only. Keep static, named-B, length and
  combined scopes: ordinary symbolic F64 MATMUL/all-valid symbolic F64 Scan remain unqualified.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [scalar glossary](../../../../glossary.md#term-scalar); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- Only existing private emitter/binder, specialization and necessary preparer/lowering proof
  owners change. No new top-level production type, common helper seam/signature, runtime object,
  entry/helper ABI parameter, public policy, API, numerical rule, global schema or projection.
  Use private helpers inside existing owners; stop before expansion.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [lowering glossary](../../../../glossary.md#lowering); [signature glossary](../../../../glossary.md#term-signature); [runtime glossary](../../../../glossary.md#term-runtime); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [parameter glossary](../../../../glossary.md#term-parameter); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [schema glossary](../../../../glossary.md#term-schema))</span>
- No W/U hoisting outside the row, carry promotion across T, new fuser/recognizer, dtype/layout
  capability, SIMD-body rewrite, exact-SUM change, worker/reservation/scheduling change or rollout.
  Preserve all unrelated dirty work. No stage/commit/push; historical campaigns remain closed.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [dtype glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [SIMD glossary](../../../../glossary.md#simd); [commit glossary](../../../../glossary.md#term-commit))</span>

## Contracts

Read the root [scope](../../../../glossary.md#term-scope) index and these exact headings:

- [Root — Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>,
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span> and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [Backend](../../../../glossary.md#backend) execution, under [CPU0073 accepted row-map and multiple-terminal reduction fusion](../../../../architecture/contracts/backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion)<span data-glossary-navigation="true"> ([fusion glossary](../../../../glossary.md#term-fusion))</span>:
  `Physical region and admission`, `Cold proof, resources and range`,
  `Exact types, signatures and generation lifetime`, `Selective compatibility and retention`.
- Backend execution, under [CPU0074 selected row-region execution exception](../../../../architecture/contracts/backend-execution.md#cpu0074-selected-row-region-execution-exception):
  `Bounded selection and numerical boundary`, `Exact new internal seams`,
  `Resources, dispatch lifetime and compatibility`, and
  [CPU0075 selective scalar load and address reuse](../../../../architecture/contracts/backend-execution.md#cpu0075-selective-scalar-load-and-address-reuse)<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar))</span>.
- [Backend execution — Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [CPU glossary](../../../../glossary.md#term-cpu))</span>
  — retain existing named-B eligibility and numeric binding.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- Scan [owner](../../../../glossary.md#term-owner) — [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification)<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>,
  [CPU0074 selected terminal grouping exception](../../../../architecture/contracts/recurrent-scan.md#cpu0074-selected-terminal-grouping-exception),
  [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>,
  [First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse),
  [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [preflight glossary](../../../../glossary.md#term-preflight))</span>.
- [Runtime owner — CPU0074 run-owned synchronous reservation](../../../../architecture/contracts/runtime-prepare-engine.md#cpu0074-run-owned-synchronous-reservation)<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [synchronous glossary](../../../../glossary.md#term-synchronous))</span>
  — preserve existing dispatch, joins and run lifetime; no owner change.<span data-glossary-navigation="true"> ([dispatch glossary](../../../../glossary.md#term-dispatch); [lifetime glossary](../../../../glossary.md#term-lifetime))</span>

Stop on missing/ambiguous authority; do not invent an internal protocol to bridge a gap.

## Dependencies and integration

- Depends on: CPU0073/0074 Complete with retained baseline; explicit user approval of the
  selective emission/cache replacement and bounded experiment, with MAIN-fixed finite budget.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [finite glossary](../../../../glossary.md#term-finite))</span>
- Conflicts with: any writer to CPU row emission, specialization, cold proof or owning authority;
  all CPU0074/earlier timing campaigns are closed, not concurrent work. CPU0048–0052 held.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
- Parallel group: None; one serial implementation writer, then independent targeted review.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty
  2574-path inventory, not a clean checkout or merged predecessor.
- Integration order: this authority/Ready packet → MAIN source-bound GO/freeze confirmation →
  implementation and targeted preflight → MAIN one-JVM release → bounded observation →
  improvement review/module gate → independent review → MAIN acceptance.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [preflight glossary](../../../../glossary.md#term-preflight); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [release glossary](../../../../glossary.md#term-release))</span>
- Integration [validation](../../../../glossary.md#term-validation): targeted [CPU](../../../../glossary.md#term-cpu) commands and one final CPU [module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc) gate below.
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; independent final docs pass after source writer closure.

Verified launch, 2026-10-30: CPU0073/0074 Complete; user approved the replacement and MAIN-fixed
budget. MAIN's `/private/tmp/synaptik-cpu0075-run.zwSo9i/baseline/` froze 512 CPU files, 32 old
classes/[logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) and original Java classes against the 2574-path inventory before edits. Writer closed;
final source/harness/evidence freeze has 62 entries.

## Files and symbols

Production files, relative to `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`:

- `codegen/emit/CpuClassFileKernelGenerator.java`: `emitRowRegion`, `regionLoad`, `regionAddress`.
- `codegen/emit/CpuGeneralScanEmitter.java`: `emitRegionalStage`, `regionalLoad`, `load` and private
  regional address binding only; nonregional uses of <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>`load` retain literal emission.
- `cache/CpuKernelSpecialization.java`: `compatibilityBytes`, `classIdentityBytes`,
  `scalarRowReuseIdentity`; no new record component or entry [ABI](../../../../glossary.md#term-application-binary-interface-abi).
- `prepare/CpuPartitionPreparer.java`: `rowEmission` and necessary cold live-slot accounting;
  `lowering/CpuRowMapReductionLowering.java`: existing pre-resource row proof only if needed.

Focused tests: `CpuRowMapReductionGeneratedKernelTest` (especially
`assertFusedCombinedStructure`), `CpuRowFusionSimdParallelTest`,
`CpuRowMapReductionLoweringTest`, `CpuGeneralScanLoweringTest`, `CpuKernelSpecializationTest`,
`CpuGeneratedKernelArtifactStoreTest`. Added `CpuRowScalarReuseTest` and test-only
`CpuRowScalarReuseComparison`; changed [scalar](../../../../glossary.md#term-scalar)-retention expectations in `CpuRowFusionSimdParallelTest`.
Comparison uses `CpuRowFusionSimdParallelTimingTest.ordinary()/scan()` and unchanged
`CpuRowFusionSimdParallelJavaOracle`, not permission to rerun CPU0074.

## Acceptance criteria

1. Use authenticated source/storage origin plus the complete effective affine cell recipe,
   including B coefficients, view offsets/strides, row/time/bank binding and store epoch.
   Never key just by value ordinal or sampled B. Prove equality for all admitted B and preserve
   binding alias guards; a possibly aliasing retained/public/scratch write invalidates reuse.
   Unknown mapping keeps a checked direct read. Activity and first-load dominance hold on
   every branch; no reuse crosses rows, active times, banks, workers or changed storage epochs.<span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [strides glossary](../../../../glossary.md#term-stride); [binding glossary](../../../../glossary.md#term-binding); [epoch glossary](../../../../glossary.md#term-epoch-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
2. Cache repeated cells only as needed, with exact last-use intervals and a type-compatible
   free pool; release/recycle value and address locals after last use. Do not keep all 20 input
   cells live. Supply actual intervals, not an assumed F3/H2-specific cap. Include existing
   producer/accumulator/binding/control slots and
   double/long widths in the pre-slot 128-live-slot/method-size proof. Report max_locals separately
   from simultaneous JVM-slot liveness and hardware registers. No hot discovery or new allocation.<span data-glossary-navigation="true"> ([Cache glossary](../../../../glossary.md#term-cache); [release glossary](../../../../glossary.md#term-release); [producer glossary](../../../../glossary.md#term-producer); [accumulator glossary](../../../../glossary.md#accumulator-type); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [liveness glossary](../../../../glossary.md#term-liveness))</span>
3. Cold proof precedes cuts/resources/slots, including [scalar](../../../../glossary.md#term-scalar) without `RowFusionEmission` and
   parallel-scalar caller/helper phases separately. If a reuse/size proof fails, keep unshared
   accesses in the replacement or the existing cold fallback; never a late route/resource
   decision or parallel maintained old emitter. No new type/common seam is presumed.
4. For the actual ordinary F32 F3/H2 capture, prove the load/store mapping and reduce 32 → 20
   external scalar load sites if its epoch/dominance proof permits all twelve duplicates.
   Explain any justified smaller reduction; do not weaken guards to meet a count. Preserve the
   ordered 28 MUL/22 ADD/2 SUB sites and seven retained row stores, terminal publication and
   separate exact bias SUM. Scan analogously removes proved duplicates while retaining ordered
   24 MUL/24 ADD/2 SUB, five row stores and twelve distinct after-row per-time carry ADDs in
   the existing actual combined fixture. Generic fixtures beyond F3/H2 must exercise reuse.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [capture glossary](../../../../glossary.md#term-capture-meanings); [publication glossary](../../../../glossary.md#term-publication); [bias glossary](../../../../glossary.md#term-bias); [carry glossary](../../../../glossary.md#term-carry); [fixture glossary](../../../../glossary.md#term-fixture))</span>
5. Preserve raw/special/cancellation behavior and producer-ordered complete length preflight,
   B=0/1, T=0, empty/all-inactive and mixed lengths, poisoned inactive sources, saved histories,
   +0 padding, singleton/no-axis raw movement, alias rejection and concurrent mixed-B reuse.
   Address folding/sharing keeps checked alignment/span/overflow and nominal-view geometry.<span data-glossary-navigation="true"> ([cancellation glossary](../../../../glossary.md#term-cancellation); [preflight glossary](../../../../glossary.md#term-preflight); [padding glossary](../../../../glossary.md#padding); [singleton glossary](../../../../glossary.md#term-singleton); [axis glossary](../../../../glossary.md#term-axis); [alignment glossary](../../../../glossary.md#term-alignment); [overflow glossary](../../../../glossary.md#term-overflow); [nominal glossary](../../../../glossary.md#term-nominal))</span>
6. Append the authority's scoped ordinary/Scan scalar-reuse marker in BOTH identity streams
   only for affected scalar/parallel-scalar cases, including emission without exec-v1 metadata.
   Keep original namespaces, canonical keys, schema 71 and projections 72/71. Old affected
   artifacts must be stale under new keys; verify generate/load/reuse and B/L-independent keys.
   Historical <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [metadata glossary](../../../../glossary.md#term-metadata); [canonical glossary](../../../../glossary.md#term-canonical-form); [schema glossary](../../../../glossary.md#term-schema); [artifacts glossary](../../../../glossary.md#term-artifact))</span>`.tsv`/`.gz.b64` controls are NEVER regenerated from candidate code. Test affected
   scalar snapshots as stale; unaffected SIMD/nonregional keys/ClassFiles/resources stay literal
   against authentic frozen evidence. Shared scalar helpers used by SIMD must not drift its bytes.<span data-glossary-navigation="true"> ([SIMD glossary](../../../../glossary.md#simd))</span>

## [Validation](../../../../glossary.md#term-validation)

Targeted semantics, structural and both-stream/[cache](../../../../glossary.md#term-cache) tests use [immutable](../../../../glossary.md#term-immutable) stale/literal controls.
Keep loop/activity/store/[carry](../../../../glossary.md#term-carry) checks and prove reuse/dominance/last-use instead of incidental
no-local-copy assertions. The completed focused command was:

```bash
./gradlew :backends:cpu:test --tests '*CpuRowMapReductionGeneratedKernelTest' --tests '*CpuRowFusionSimdParallelTest' --tests '*CpuRowMapReductionLoweringTest' --tests '*CpuGeneralScanLoweringTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest' --offline
```

Approved protocol, now exhausted: exactly one [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) with an external 90-second watchdog; four
[F32](../../../../glossary.md#floating-point-precision-and-rounding) cases (ordinary/actual Scan reverse, B32/B256, F3/H2, Scan T=4). A is authentic old [bytecode](../../../../glossary.md#term-bytecode),
B the replacement, C unchanged independent Java. A/B use fresh hidden handles, identical raw
[ABI](../../../../glossary.md#term-application-binary-interface-abi)/spreader wrappers, read-only saved inputs and separate writable outputs. Entry resets are
inside each call; cold [binding](../../../../glossary.md#term-binding)/inspection/equality controls are outside windows. C's different
wrapper, W/U staging and across-T carry locals are disclosed, not claimed as [dataflow](../../../../glossary.md#term-dataflow) parity.
Exactly 30000 warm calls/arm/case in three 10000-call blocks, then 3×5000 [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) calls; both
phases use ABC/BCA/CAB order. MAIN approved the harness before launch. Incremental samples and
uptime bounds accompany the same JVM's [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) [log](../../../../glossary.md#term-log): distinguish [C1](../../../../glossary.md#term-c1) failure, normal [C2](../../../../glossary.md#term-c2), [on-stack
replacement](../../../../glossary.md#term-on-stack-replacement-osr) ([OSR](../../../../glossary.md#just-in-time-compilation)) and unresolved/[inlining](../../../../glossary.md#term-inlining) attribution. Fixed counts never prove C2. No retry,
restart, [fork](../../../../glossary.md#term-fork-meanings), expansion or diagnostic JVM; expiry would retain [censored](../../../../glossary.md#term-censored) partial evidence.

After reviewing the [finite](../../../../glossary.md#term-finite) result, MAIN released exactly one final affected-[module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc) gate:

```bash
./gradlew :backends:cpu:test :backends:cpu:javadoc --offline
```

Profitability is not a source-acceptance prerequisite. Structural savings and timing conclusions
remain distinct. No additional [conformance](../../../../glossary.md#term-conformance)/integration/root [run](../../../../glossary.md#term-run) was required: dependencies,
worker/resource [ownership](../../../../glossary.md#term-ownership) and [API](../../../../glossary.md#term-application-programming-interface-api)/ABI boundaries are unchanged. No budget remains open.

## Documentation and review impact

Root/owning retention, ADR0054, [CPU](../../../../glossary.md#term-cpu) guide/boundary and task/master/roadmap are synchronized.
Independent review finalized code/[Javadoc](../../../../glossary.md#term-javadoc)/documentation acceptance using recorded tests; no
review rerun or Java edit. General, architecture, [backend](../../../../glossary.md#backend), [ADR](../../../../glossary.md#term-architecture-decision-record-adr), [planning](../../../../glossary.md#term-planning) and Javadoc profiles
applied; targeted glossary review found no new term or changed public meaning.

## Result

Status: Complete — independent final APPROVE, 0 open P1/P2; MAIN owning gate passed, 2026-10-30.

- Changed five production [owners](../../../../glossary.md#term-owner), three test/harness files and eight Markdown docs; no new production type/seam.
- Root/affine/store-[epoch](../../../../glossary.md#term-epoch-meanings) reuse, last-use typed pool and LONG offset retain cold 128-slot/checked fallback.
- SUM-address P1 fixed/closed; nine typed/static/admitted-symbolic B1/3/7 [regression](../../../../glossary.md#term-regression) scenarios passed.
- Ordinary loads 32→20, MUL/ADD/SUB 28/22/2, stores 7; Scan gets 27→22, math 24/24/2, stores 5, [carry](../../../../glossary.md#term-carry) ADDs 12 unchanged.
- Peak ordinary duplicate intervals 4; live-slot bounds ordinary 82/Scan 98, distinct from max_locals 87/128 and HW registers.
- Both [identity](../../../../glossary.md#term-identity-meanings) streams stale only affected [scalar](../../../../glossary.md#term-scalar) [artifacts](../../../../glossary.md#term-artifact); historical resources and [SIMD](../../../../glossary.md#simd)/nonregional controls retained.
- Focused command above: 83 tests, no failures/errors/skips; later P1 regression and four-case/eight-SIMD [preflight](../../../../glossary.md#term-preflight) passed.
- MAIN final command: exit 0, BUILD SUCCESSFUL 5m54s; 233 suites/1440 tests, 0 failures/errors, 56 skips (1384 executed).
- Row-[fusion](../../../../glossary.md#term-fusion) 18/0 skip; reuse 2/1 skip (baseline-opt-in passed separately, P1 executed); [Javadoc](../../../../glossary.md#term-javadoc) PASS, 94 warnings.
- Post-gate freeze 62/62 PASS; HEAD/index/unrelated work preserved. No review test rerun or Java edit.
- Sole [JVM](../../../../glossary.md#term-java-virtual-machine-jvm): exit 0 in 27.855s, 72 windows (36 warm/36 [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)), raw equality PASS; no censor/retry/extra JVM.
- B32/B256 ordinary A/B 172.8×/216.3×: A32 no own [nmethod](../../../../glossary.md#term-nmethod), A256 only [C2](../../../../glossary.md#term-c2) [OSR](../../../../glossary.md#term-on-stack-replacement-osr); B normal C2 before samples. Not matched-[JIT](../../../../glossary.md#term-just-in-time-compilation-jit) speedups.
- Scan A/B/C normal C2 before samples; A/B 1.007×/1.005×: no material speedup. New/Java ordinary 1.36×/1.27×, Scan 1.68×/1.79×.
- [C1](../../../../glossary.md#term-c1) failures persist, including candidates; no C1 fix/proved machine-code cause. Java staging/carry/wrapper gaps remain.
- Evidence root `/private/tmp/synaptik-cpu0075-run.zwSo9i/`: `SOURCE-HANDOFF.txt`, `SOURCE-FREEZE.sha256`.
- Preflight `source-preflight.8794461838743973694/`; final XML/receipt `final-owning.AYrZUC/`; samples/events/JIT/`ANALYSIS.json` `measurement.7v__8dmc/`.
- Javadocs reviewed, eight docs synchronized; glossary: no new term; local links/anchors/whitespace checked.
- No required follow-up. Frontier None; Scan/Java gap disclosed, defaults unchanged; no new task/budget/rollout or stage/[commit](../../../../glossary.md#term-commit)/push.
