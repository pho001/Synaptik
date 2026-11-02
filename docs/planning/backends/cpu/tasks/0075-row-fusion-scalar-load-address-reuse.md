# Task CPU0075: Scalar row-fusion load and address reuse

## Status

Complete — 2026-10-30, independent final code/Javadoc/docs APPROVE and MAIN owning gate PASS.
The sole comparison and source frontier are closed; no successor or further timing is authorized.

## Change class

Class C — changes generated hot-loop dataflow and selective artifact compatibility, not numerics.
Clean implementation and an independent targeted code/Javadoc/documentation review are required.

## Goal

Replace affected ordinary/general-Scan scalar row emission with within-row external-load reuse
and constant affine address folding/sharing: generic reuse, not F3/H2 hardcoding or old/new emitters.

## Scope and non-goals

- Existing F32/F64 scalar and parallel-scalar row bodies where already admitted; ordinary array
  and checked-segment eligibility, Scan checked segments only. Keep static, named-B, length and
  combined scopes: ordinary symbolic F64 MATMUL/all-valid symbolic F64 Scan remain unqualified.
- Only existing private emitter/binder, specialization and necessary preparer/lowering proof
  owners change. No new top-level production type, common helper seam/signature, runtime object,
  entry/helper ABI parameter, public policy, API, numerical rule, global schema or projection.
  Use private helpers inside existing owners; stop before expansion.
- No W/U hoisting outside the row, carry promotion across T, new fuser/recognizer, dtype/layout
  capability, SIMD-body rewrite, exact-SUM change, worker/reservation/scheduling change or rollout.
  Preserve all unrelated dirty work. No stage/commit/push; historical campaigns remain closed.

## Contracts

Read the root scope index and these exact headings:

- [Root — Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Module-ownership routing](../../../../../ARCHITECTURE.md#module-ownership-routing),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules) and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Backend execution, under [CPU0073 accepted row-map and multiple-terminal reduction fusion](../../../../architecture/contracts/backend-execution.md#cpu0073-accepted-row-map-and-multiple-terminal-reduction-fusion):
  `Physical region and admission`, `Cold proof, resources and range`,
  `Exact types, signatures and generation lifetime`, `Selective compatibility and retention`.
- Backend execution, under [CPU0074 selected row-region execution exception](../../../../architecture/contracts/backend-execution.md#cpu0074-selected-row-region-execution-exception):
  `Bounded selection and numerical boundary`, `Exact new internal seams`,
  `Resources, dispatch lifetime and compatibility`, and
  [CPU0075 selective scalar load and address reuse](../../../../architecture/contracts/backend-execution.md#cpu0075-selective-scalar-load-and-address-reuse).
- [Backend execution — Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract)
  — retain existing named-B eligibility and numeric binding.
- Scan owner — [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification),
  [CPU0074 selected terminal grouping exception](../../../../architecture/contracts/recurrent-scan.md#cpu0074-selected-terminal-grouping-exception),
  [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories),
  [First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse),
  [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates).
- [Runtime owner — CPU0074 run-owned synchronous reservation](../../../../architecture/contracts/runtime-prepare-engine.md#cpu0074-run-owned-synchronous-reservation)
  — preserve existing dispatch, joins and run lifetime; no owner change.

Stop on missing/ambiguous authority; do not invent an internal protocol to bridge a gap.

## Dependencies and integration

- Depends on: CPU0073/0074 Complete with retained baseline; explicit user approval of the
  selective emission/cache replacement and bounded experiment, with MAIN-fixed finite budget.
- Conflicts with: any writer to CPU row emission, specialization, cold proof or owning authority;
  all CPU0074/earlier timing campaigns are closed, not concurrent work. CPU0048–0052 held.
- Parallel group: None; one serial implementation writer, then independent targeted review.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty
  2574-path inventory, not a clean checkout or merged predecessor.
- Integration order: this authority/Ready packet → MAIN source-bound GO/freeze confirmation →
  implementation and targeted preflight → MAIN one-JVM release → bounded observation →
  improvement review/module gate → independent review → MAIN acceptance.
- Integration validation: targeted CPU commands and one final CPU module/Javadoc gate below.
- Shared-document integration owner: MAIN; independent final docs pass after source writer closure.

Verified launch, 2026-10-30: CPU0073/0074 Complete; user approved the replacement and MAIN-fixed
budget. MAIN's `/private/tmp/synaptik-cpu0075-run.zwSo9i/baseline/` froze 512 CPU files, 32 old
classes/logs and original Java classes against the 2574-path inventory before edits. Writer closed;
final source/harness/evidence freeze has 62 entries.

## Files and symbols

Production files, relative to `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`:

- `codegen/emit/CpuClassFileKernelGenerator.java`: `emitRowRegion`, `regionLoad`, `regionAddress`.
- `codegen/emit/CpuGeneralScanEmitter.java`: `emitRegionalStage`, `regionalLoad`, `load` and private
  regional address binding only; nonregional uses of `load` retain literal emission.
- `cache/CpuKernelSpecialization.java`: `compatibilityBytes`, `classIdentityBytes`,
  `scalarRowReuseIdentity`; no new record component or entry ABI.
- `prepare/CpuPartitionPreparer.java`: `rowEmission` and necessary cold live-slot accounting;
  `lowering/CpuRowMapReductionLowering.java`: existing pre-resource row proof only if needed.

Focused tests: `CpuRowMapReductionGeneratedKernelTest` (especially
`assertFusedCombinedStructure`), `CpuRowFusionSimdParallelTest`,
`CpuRowMapReductionLoweringTest`, `CpuGeneralScanLoweringTest`, `CpuKernelSpecializationTest`,
`CpuGeneratedKernelArtifactStoreTest`. Added `CpuRowScalarReuseTest` and test-only
`CpuRowScalarReuseComparison`; changed scalar-retention expectations in `CpuRowFusionSimdParallelTest`.
Comparison uses `CpuRowFusionSimdParallelTimingTest.ordinary()/scan()` and unchanged
`CpuRowFusionSimdParallelJavaOracle`, not permission to rerun CPU0074.

## Acceptance criteria

1. Use authenticated source/storage origin plus the complete effective affine cell recipe,
   including B coefficients, view offsets/strides, row/time/bank binding and store epoch.
   Never key just by value ordinal or sampled B. Prove equality for all admitted B and preserve
   binding alias guards; a possibly aliasing retained/public/scratch write invalidates reuse.
   Unknown mapping keeps a checked direct read. Activity and first-load dominance hold on
   every branch; no reuse crosses rows, active times, banks, workers or changed storage epochs.
2. Cache repeated cells only as needed, with exact last-use intervals and a type-compatible
   free pool; release/recycle value and address locals after last use. Do not keep all 20 input
   cells live. Supply actual intervals, not an assumed F3/H2-specific cap. Include existing
   producer/accumulator/binding/control slots and
   double/long widths in the pre-slot 128-live-slot/method-size proof. Report max_locals separately
   from simultaneous JVM-slot liveness and hardware registers. No hot discovery or new allocation.
3. Cold proof precedes cuts/resources/slots, including scalar without `RowFusionEmission` and
   parallel-scalar caller/helper phases separately. If a reuse/size proof fails, keep unshared
   accesses in the replacement or the existing cold fallback; never a late route/resource
   decision or parallel maintained old emitter. No new type/common seam is presumed.
4. For the actual ordinary F32 F3/H2 capture, prove the load/store mapping and reduce 32 → 20
   external scalar load sites if its epoch/dominance proof permits all twelve duplicates.
   Explain any justified smaller reduction; do not weaken guards to meet a count. Preserve the
   ordered 28 MUL/22 ADD/2 SUB sites and seven retained row stores, terminal publication and
   separate exact bias SUM. Scan analogously removes proved duplicates while retaining ordered
   24 MUL/24 ADD/2 SUB, five row stores and twelve distinct after-row per-time carry ADDs in
   the existing actual combined fixture. Generic fixtures beyond F3/H2 must exercise reuse.
5. Preserve raw/special/cancellation behavior and producer-ordered complete length preflight,
   B=0/1, T=0, empty/all-inactive and mixed lengths, poisoned inactive sources, saved histories,
   +0 padding, singleton/no-axis raw movement, alias rejection and concurrent mixed-B reuse.
   Address folding/sharing keeps checked alignment/span/overflow and nominal-view geometry.
6. Append the authority's scoped ordinary/Scan scalar-reuse marker in BOTH identity streams
   only for affected scalar/parallel-scalar cases, including emission without exec-v1 metadata.
   Keep original namespaces, canonical keys, schema 71 and projections 72/71. Old affected
   artifacts must be stale under new keys; verify generate/load/reuse and B/L-independent keys.
   Historical `.tsv`/`.gz.b64` controls are NEVER regenerated from candidate code. Test affected
   scalar snapshots as stale; unaffected SIMD/nonregional keys/ClassFiles/resources stay literal
   against authentic frozen evidence. Shared scalar helpers used by SIMD must not drift its bytes.

## Validation

Targeted semantics, structural and both-stream/cache tests use immutable stale/literal controls.
Keep loop/activity/store/carry checks and prove reuse/dominance/last-use instead of incidental
no-local-copy assertions. The completed focused command was:

```bash
./gradlew :backends:cpu:test --tests '*CpuRowMapReductionGeneratedKernelTest' --tests '*CpuRowFusionSimdParallelTest' --tests '*CpuRowMapReductionLoweringTest' --tests '*CpuGeneralScanLoweringTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest' --offline
```

Approved protocol, now exhausted: exactly one JVM with an external 90-second watchdog; four
F32 cases (ordinary/actual Scan reverse, B32/B256, F3/H2, Scan T=4). A is authentic old bytecode,
B the replacement, C unchanged independent Java. A/B use fresh hidden handles, identical raw
ABI/spreader wrappers, read-only saved inputs and separate writable outputs. Entry resets are
inside each call; cold binding/inspection/equality controls are outside windows. C's different
wrapper, W/U staging and across-T carry locals are disclosed, not claimed as dataflow parity.
Exactly 30000 warm calls/arm/case in three 10000-call blocks, then 3×5000 sample calls; both
phases use ABC/BCA/CAB order. MAIN approved the harness before launch. Incremental samples and
uptime bounds accompany the same JVM's JIT log: distinguish C1 failure, normal C2, on-stack
replacement (OSR) and unresolved/inlining attribution. Fixed counts never prove C2. No retry,
restart, fork, expansion or diagnostic JVM; expiry would retain censored partial evidence.

After reviewing the finite result, MAIN released exactly one final affected-module/Javadoc gate:

```bash
./gradlew :backends:cpu:test :backends:cpu:javadoc --offline
```

Profitability is not a source-acceptance prerequisite. Structural savings and timing conclusions
remain distinct. No additional conformance/integration/root run was required: dependencies,
worker/resource ownership and API/ABI boundaries are unchanged. No budget remains open.

## Documentation and review impact

Root/owning retention, ADR0054, CPU guide/boundary and task/master/roadmap are synchronized.
Independent review finalized code/Javadoc/documentation acceptance using recorded tests; no
review rerun or Java edit. General, architecture, backend, ADR, planning and Javadoc profiles
applied; targeted glossary review found no new term or changed public meaning.

## Result

Status: Complete — independent final APPROVE, 0 open P1/P2; MAIN owning gate passed, 2026-10-30.

- Changed five production owners, three test/harness files and eight Markdown docs; no new production type/seam.
- Root/affine/store-epoch reuse, last-use typed pool and LONG offset retain cold 128-slot/checked fallback.
- SUM-address P1 fixed/closed; nine typed/static/admitted-symbolic B1/3/7 regression scenarios passed.
- Ordinary loads 32→20, MUL/ADD/SUB 28/22/2, stores 7; Scan gets 27→22, math 24/24/2, stores 5, carry ADDs 12 unchanged.
- Peak ordinary duplicate intervals 4; live-slot bounds ordinary 82/Scan 98, distinct from max_locals 87/128 and HW registers.
- Both identity streams stale only affected scalar artifacts; historical resources and SIMD/nonregional controls retained.
- Focused command above: 83 tests, no failures/errors/skips; later P1 regression and four-case/eight-SIMD preflight passed.
- MAIN final command: exit 0, BUILD SUCCESSFUL 5m54s; 233 suites/1440 tests, 0 failures/errors, 56 skips (1384 executed).
- Row-fusion 18/0 skip; reuse 2/1 skip (baseline-opt-in passed separately, P1 executed); Javadoc PASS, 94 warnings.
- Post-gate freeze 62/62 PASS; HEAD/index/unrelated work preserved. No review test rerun or Java edit.
- Sole JVM: exit 0 in 27.855s, 72 windows (36 warm/36 sample), raw equality PASS; no censor/retry/extra JVM.
- B32/B256 ordinary A/B 172.8×/216.3×: A32 no own nmethod, A256 only C2 OSR; B normal C2 before samples. Not matched-JIT speedups.
- Scan A/B/C normal C2 before samples; A/B 1.007×/1.005×: no material speedup. New/Java ordinary 1.36×/1.27×, Scan 1.68×/1.79×.
- C1 failures persist, including candidates; no C1 fix/proved machine-code cause. Java staging/carry/wrapper gaps remain.
- Evidence root `/private/tmp/synaptik-cpu0075-run.zwSo9i/`: `SOURCE-HANDOFF.txt`, `SOURCE-FREEZE.sha256`.
- Preflight `source-preflight.8794461838743973694/`; final XML/receipt `final-owning.AYrZUC/`; samples/events/JIT/`ANALYSIS.json` `measurement.7v__8dmc/`.
- Javadocs reviewed, eight docs synchronized; glossary: no new term; local links/anchors/whitespace checked.
- No required follow-up. Frontier None; Scan/Java gap disclosed, defaults unchanged; no new task/budget/rollout or stage/commit/push.
