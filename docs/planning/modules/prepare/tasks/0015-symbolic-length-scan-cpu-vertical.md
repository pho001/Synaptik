# Task Prepare0015: Symbolic-length Scan CPU vertical

## Status

Complete — MAIN accepted 2026-10-27 after independent Class C code/docs APPROVE (0 P1/P2), owning validation, the sole bounded timing probe and configured root test/Javadoc PASS.
Ready admission: independent planning APPROVE, 0 open P1/P2; MAIN accepted 2026-10-26. User “pokracuj” authorizes bounded complete implementation including this gate.
PlanningAuthorized=true; AuthorityAuthorized=true (bounded qualification); ImplementationAuthorized=true.
Active frontier: None. JavaLaunchAuthorized=true records the completed clean implementation admission, not a successor launch.
Size exception: 241 lines/~21KB (<25KB); one atomic cross-owner qualification keeps support gates, typed seams,
acceptance and validation evidence together. The pre-Java documentation phase changed no Java work.

## Change class

Class C — cross-owner geometry/generated hot paths; clean implementation and independent code/docs review.

## Goal and scope

One named B + runtime discrete non-gradient INT64[B] lengths, fixed T/features, homogeneous FLOAT32/FLOAT64 (F32/F64)
forward AND actual Compiler-generated first-order reverse: compile once → prepare once → fresh
runValues descriptors/resources. First qualify bounded outer F64/control/ROW(1) inside this task,
then combined projection/recipes/execution/publication. All-valid symbolic F64 is not assumed or
required; its current F32-only qualification stays separate.

New API/type/package/session/executor/memo/dependency/representation: None. Retain static/all-valid
routes. NN/Training, cold/copy/wire/v5, native/OpenBLAS/Metal/mixed-owner, variable T/features,
low/mixed dtypes, vector/parallel/tuning, fluent migration and higher-order/createGraph stay held.

## Contracts

Start at [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Exact required headings (including applicable subordinate rules):

- Root: `Core lifecycle`, `Core invariants`, `Dependency rules`, `Testing requirements`.
- [recurrent-scan.md](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan): `General differentiable scan`, `Accepted symbolic-batch scan`, `Length-aware whole-batch scan`, especially `Runtime preflight and implementation gates`.
- [runtime-prepare-engine.md](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare): `modules/runtime`, `modules/prepare`, `modules/engine`, `Accepted symbolic-batch resource contract`, `Prepare lifecycle`, `Run lifecycle`, `Public reusable prepared execution`.
- [backend-execution.md](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes): `CPU backend routes`, `Accepted ordinary symbolic-batch CPU contract`.
- [compiler-autograd.md](../../../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding): `Accepted symbolic-batch metadata binding`; no new AD/binder algorithm/API.

Design context only: [length-aware proposal](../../../length-aware-general-scan.md#proposed-symbolic-b-plus-lengths-integration),
`Minimal coordinated delta and serial capability order`, `Numeric resources, admission and compatibility`,
`Proportional future validation, not runs now`.

## Dependencies and integration

- Depends on: Model0044; Compiler0015; Prepare0011/0013/0014; CPU0042 (retaining CPU0033/0037/0041); Runtime0016; Engine0021; Planning0008 — Complete.
- Conflicts with: Scan projection/roles/demand; CPU typed closure/IR/emission/resources/identities; Engine binding/publication/lifecycle; owning contracts/shared docs. Serialize, no concurrent writer.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus the accepted 2483-file dirty worktree (`prepare0015-baseline.json`), not merged predecessors; index empty.
- Integration order: MAIN review/Ready → user approval → in-task pre-Java review/MAIN acceptance → clean implementation → independent code/docs review → MAIN combined gate/acceptance.
- Integration validation: the single MAIN combined checkpoint under Validation, reusing successful worker evidence.
- Shared-document integration owner: MAIN; exclusive serial final docs/Javadoc delegation.

Launch verification 2026-10-26: MAIN reverified final Ready planning unchanged, Complete predecessors/
reviews/gates, sole frontier/inactive conflicts, no other writer. This clean context confirmed predecessor
Status/Result, all 2483 baseline hashes and HEAD/origin/index unchanged. MAIN accepted the frozen 19-file qualification after independent APPROVE; sole code-writer launch follows with stable base/conflicts.

## Pre-Java support-gate qualification (same task)

Enacted qualification is implemented and validated at these exact owning headings. Independent
pre-Java review and MAIN acceptance passed on 2026-10-26, before the clean implementation launch.

- Root `Core lifecycle`, `Core invariants`, `Scope-indexed normative contracts`: matching bounded support/status links, successors held.
- recurrent-scan `Accepted symbolic-batch scan` / `Qualification and mechanism gates`, `Length-aware whole-batch scan` / `Runtime preflight and implementation gates`: combined projection/execution only; unchanged header/equations/roles/AD guards/saved sites/preflight/resource algorithms.
- runtime-prepare-engine `modules/prepare`, `Prepare lifecycle`, `Run lifecycle`, `Public reusable prepared execution`: static-or-one-B length projection/admission/publication; unchanged handoff/lifecycle, no memo.
- backend-execution `CPU backend routes`, `Accepted ordinary symbolic-batch CPU contract`: combined F32/F64 and bounded outer/control/ROW(1); retain old F32 bytes/static reduction sizing.
- compiler-autograd `Accepted symbolic-batch metadata binding`: downstream status only.

Same-change explanations/decision evidence; no authority-only successor. A different clean code context
followed review/MAIN acceptance. No missing contract, unavoidable new type or substantive decision remained.

## Files and symbols

Prefix `modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/`: analysis/PrepareContext validateScan/validateRowShape; GraphPreparation projection/demand/handoff.
CPU prefix `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`: CpuCapabilityProvider; below `internal/`, existing owners by directory (all basenames `.java`):

- `lowering/`: CpuPartitionLowering supportsOrdinaryBatch/batchByteSizes/lowerOrdinaryBatch;
  CpuAggregateLowering.BatchGeometry; CpuGeneralScanLowering analyze/lower/Facts.pack/row proof.
- `prepare/`: CpuPartitionPreparer, CpuPartitionFinalizer; `ir/`: CpuGeneralScanIr, CpuPortableKernelIr;
  `cache/`: CpuKernelSpecialization non-F32 ordinary gate and BOTH identity streams;
  `memory/`: CpuRepresentationRecipes; `route/portable/`: CpuPortableRoutePlan.
- `codegen/emit/`: CpuGeneralScanEmitter **separate combined branch preserving old bytes**;
  CpuGeneralScanReductionEmitter, CpuGeneralScanMatmulEmitter; CpuAggregateEmitter float locals/five-limb/
  carrier assumptions; CpuExactSumEmitter, CpuClassFileKernelGenerator.validate (currently ordinary-batch-v1 only), scalar/carrier emitters.
- `executable/`: CpuPreparedExecutable.bindOrdinaryBatch F32 span/base hooks;
  CpuGeneralScanPreparedExecutable.executeBound static hooks/stages.batch → run-aware spans/explicit B
  even `[B,0]`; CpuPreparedPartitionExecutable orchestration. No new representation/Runtime protocol.

Engine prefix `modules/engine/src/main/java/io/github/pho001/synaptik/engine/` → PreparedExecution
numeric/publication indexing, AdvancedEngine fresh admission, RunResult actual descriptors; retain dtype-generic mechanics.
Extend existing tests below; Compiler0015 fixtures are actual-graph references, not redesign. Keep invalid raw-IR guards.

## Acceptance criteria

1. Qualify outer typed seed/zero/one splats, ADD ties/cross-occurrence sums, required MUL/DIV,
   SUM/MEAN/sumToShape, expand, feature-only reshape/rank edits, transpose/contiguous/same-dtype CAST.
   Explicit family/dtype/rank/axis/attribute/access predicates admit `[T,B,F]` and ROW(1) `[F,B]`/
   `[T,F,B]` literals/zeros, not arbitrary rank-three or ordinary F64 MATMUL. Positive/negative
   provider tests INSIDE/OUTSIDE Scan give identical answers, no origin tag. Only non-gradient
   `[B]` F32/F64→INT64 CAST adds control production; direct L needs no producer.
2. Ordinary F64 exact SUM/MEAN uses existing typed exact arithmetic: signedBits=2099+ceil(log2(N)),
   zero cardinality term at N≤1; int-bounded N fits fixed 34 limbs, 280-byte single-range scratch
   when outputCount>0, else zero with slot/creator retained. F32 keeps five limbs/48 bytes;
   static F64 N=3 stays 33 limbs/272 bytes. Test rational MEAN, subnormals, ±0, NaN/Inf/empties.
   Scan producing-site SUM remains typed sequential, never this ordinary exact algorithm.
3. Qualify complete forward/actual opposite-traversal Compiler siblings independently, both dtypes;
   no hand-authored reverse substitute. Preserve every-original-carry/sequence ROW(0) AD guards,
   generated mixed ROW(0)/ROW(1)/SHARED histories/constants/shared transpose, joint/missing seeds,
   ties/cross-occurrence sums, false-flag propagation and exact saved TANH/SIGMOID. No replay or
   weakened residual/higher-order preflight.
4. Before analyzers validate all outer/local declarations/bindings/latent ports and complete demand,
   named-axis identity not F=B/H=B coincidence. L is separate eight-byte control, no body formal.
   Preindex numeric domains; before borrow/allocation check fixed-feature products behind empties,
   int extents/counts/strides/spans, long bytes/alignment/padding/aggregate overflow. No L-based shapes.
5. Use existing fixed/run-size recipes and explicit batch-required mode even for `[B,0]`. Carry B
   explicitly in fresh numeric packing, never derive it from frame/features. Declare every bank,
   cut/temporary/output/history/zero-byte workspace before slots, with actual B/typed widths.
   Row history T×B×row-elements, SHARED T×P, dead history zero bytes/stores; accumulator/contribution
   banks O(P), never B×P. Prove cut lifetime/size dominance for every B or keep disjoint regions;
   raw carry backing stays independent of padded histories.
6. After producers recheck run-aware typed spans/access/alignment/overlap and ALL B lengths in
   [0,T] before first Scan output/padding/history/scratch write, including invalid-last/T=0;
   B=0 has no lengths. Scan-local rejection, not upstream rollback. Activity dominates sequence/
   seed/history padding loads AND arithmetic, including hoisted/shared work. Inactive carry/dInitial
   carry is raw initial/final-seed identity, emissions/dX raw +0. Reduce active contributors in
   increasing b, then one distinct represented shared-carry ADD per active time; inactive bypass.
   Preserve CPU0042 fusion/SUM-epilogue and exact saved-site guards.
7. One prepared Engine.runValues witness per dtype/direction/direct-or-produced ingress: T=4/F=3/H=2,
   `next=tanh(xStep@Wx+h@Wh+bias)`, joint final/emission seeds, nonuniform data; B=32→32→7→1→32,
   L respectively `b%5`, `4-b%5`, `b%5`, `[0]`, all 4, plus repeated L. Compare outputs/actual gradients
   to independent active-prefix unrolling. Final `[B,2]`, emissions `[4,B,2]`, dX `[4,B,3]`; fixed
   parameter gradients; unchanged code/routes/slots/topology, actual-sized resources.
8. EVERY call, even same B: fresh complete Compiler bind/private obligations, unused/tied inputs,
   all forward/reverse/latent ports, current input-backed descriptors/publications/storage. No memo.
   One RunState, fresh mutable packed/access/carry arrays/resources/result; numeric-only Runtime.
   No per-B prepare/recompile/cache, graph walk, route/topology choice, per-step allocation/callback,
   unsafe FFM, boxing/reflection/map/string dispatch or avoidable fallback overhead.
9. Extend existing tests: B=0/1, separate fixed-T=0 and T=0/B=0, zero features, fixed-product/byte/
   aggregate overflow, raw ±0/NaN carry, poisoned padding/seeds, zero/mixed/full/invalid-last L,
   live/dead/shared histories, aliases/read views/write guards, cleanup/failure/retry/close/lease
   precedence, concurrent mixed-B/results. Empty Scan skips no other required work. Mean uses B×H,
   not sum(L): empty mean NaN, empty row gradients, nonempty typed-zero parameter sums. Retain held gates.
10. Separate combined canonical `:symbolic-length-v1` and baked `|scan-symbolic-length-v1=` in BOTH
    specialization streams, including dtype/roles/terminal/demand/control/numeric-pack/recipe facts;
    exclude actual B/L/bytes/slots/addresses. Added outer F64/control/new rank-three cases use
    `ordinary-batch-typed-v1` in IR and both specialization streams, including exact dtype/carrier/
    mapping/reduction-state facts; positive/negative validated-generation and artifact load/reuse/stale
    rejection tests cover the typed namespace. Old F32 `ordinary-batch-v1`, schema/family71, static
    stagedv4, all-valid symbolic canonicalv1/bakedv3, static-length v1, fixed/ordinaryF32 keys/bytes stay identical.

## Validation

Executed commands and outcomes are recorded in Result. Settings/builds map :backends:cpu to modules/backend/cpu.
Focused development extends PrepareContext/GraphPreparation/BackendPartitionPreparer and EngineSymbolicScan
tests; use test filters, not repeated full suites. CPU focused example:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScan*Test' --tests '*CpuSymbolicScanGeneratedKernelTest' --tests '*CpuOrdinaryDynamicBatchTest' -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=2 --console=plain
```

After focused checks and executable freeze, once: `./gradlew :modules:prepare:test :backends:cpu:test :modules:engine:test :modules:prepare:javadoc :backends:cpu:javadoc :modules:engine:javadoc -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=2 --console=plain`. MAIN owns the sole campaign below after independent structural review, avoiding measurement of code still under correction.

Existing optimal clean-Java/ClassFile oracles check changed hot paths only: combined loops/binding
and F64 exact outer reductions; algorithm/hot-loop/dataflow and hidden calls/allocation/dispatch.
One added CpuGeneralScanTimingTest method, one campaign/JVM: fixed witness/B32/7, both typed forward/
actual reverse and new exact reductions, equal carriers/inputs/saved sites/preallocation. Separate
cold generation/classload/JIT, entry/preflight and kernel costs; disclose censoring/variance.
Budget: ≤60s wall including setup; per case 20 warmup pairs, then 3 rounds ×30 measured pairs,
100 calls/side/pair, alternating order. Stop admitting work at deadline; report overshoot/censoring
and non-steady-state limits. No warm-to-stability, extra fork or retry campaign.
Executed sole method command: `./gradlew :backends:cpu:test --tests '*.CpuGeneralScanTimingTest.symbolicLengthForwardReversePrepare0015' -Dsynaptik.cpu.generalScan.performance=true -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-prepare0015.gVDqlx/performance -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=1 --console=plain`.
No five-fork/full-matrix, JIT-cause or parity/speed promise; reuse unchanged evidence.

MAIN runs ONE combined conformance/integration/architecture/root gate. Extend CpuGeneralScanConformanceTest/
CpuSymbolicScanConformanceTest, EngineCpuGeneralScanIntegrationTest/EngineCpuSymbolicScanIntegrationTest;
reuse architecture guards and successful worker tasks:

```bash
SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib ./gradlew test javadoc -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=2 --console=plain
```

Existing explicit native environment only, no rebuild/build/gating/skip edits; report blockers.
Reuse successes, no duplicate reviewer suites; rerun only for changed code/stale evidence/concrete risk.
Final Javadoc follows comments. Check links/anchors/size/whitespace and baseline-relative scope/bytes.

## Documentation and review impact

Phase 1 enacts gates, focused explanations, existing ADR amendments and glossary support status; no Javadoc/Java/API change.
Final independent docs context updates implemented support/Javadocs/API examples and `docs/user-guide/running-models.md`
after executable acceptance. Targeted glossary: symbolic Scan/row role/primal binding; meanings unchanged. MAIN integration.

## Result

Status: Complete. MAIN accepted 2026-10-27; active Prepare/CPU/Engine frontiers None.
Delivered one named B plus runtime INT64[B] lengths, fixed T/features, homogeneous F32/F64 forward
and actual Compiler-generated first-order reverse through existing Prepare/CPU/Engine owners.
The same prepared owner supports changed/repeated B/L, actual-sized resources and publications;
bounded ordinary F64/control/ROW(1) prerequisites are included. No new production type/API/protocol,
dependency, session, memo, native change or production deletion. All ten acceptance criteria passed.

Changed: 19 production Java paths (four comment-only), 18 test Java paths (two new test-only types),
the current v2 capability ledger, and 21 Markdown paths. Independent code/structural review (Rawls)
and documentation/Javadoc review (Plato) APPROVE, 0 P1/P2. Final R1 row-copy activity is once per row;
R2 F64 workspace uses eight-byte cells in both geometry and packing. Six core Scan semantic sections
are unchanged. Old v1 ledger/all 272 old v2 rows and 28 static-length key/both-stream/ClassFile cases
remain exact. Generated code was compared with typed optimal Java; 16/16 measured Scan ClassFiles
equal the structurally reviewed artifacts. No hidden allocation/helper dispatch or activation replay
was found in the inspected generated hot loops.

Validation: owning command above exit 0, 1,524 executed/46 skipped (1,570 discovered), three Javadocs
PASS; focused conformance 23 and integration 24 PASS. MAIN's sole configured root command above
exit 0: 4,988 executed/57 skipped (5,045 discovered), zero failures/errors, all Javadocs PASS;
119 tasks, 47 executed/72 UP-TO-DATE. Existing optional diagnostics/performance/observer skips were
retained, not bypassed. Root XML was archived before the filtered timing test replaced CPU reports.
No native rebuild or Git mutation. Final links/anchors/whitespace and baseline-relative scope PASS;
HEAD/origin/index and the other 2,426 accepted files remain unchanged.

Sole timing campaign: exit 0, 24/24 complete cases, 20 warmup pairs + 3 rounds ×30 measured pairs,
100 calls/side/pair; 10.117149416 seconds, no censoring/overshoot. Ratios below are paired medians
of generated entry plus fixture list/spreader/MethodHandle overhead versus a typed direct Java call;
both include Runnable/deadline-clock overhead. They are not isolated inner-loop measurements.

| Cases | Generated/direct paired-median range |
|---|---|
| B=7, forward, both dtypes/traversals | 0.921–1.137 |
| B=7, actual reverse, both dtypes/traversals | 1.344–1.462 |
| B=32, forward, both dtypes/traversals | 0.811–1.059 |
| B=32, actual reverse, both dtypes/traversals | 41.129–78.837 |
| F32 exact SUM/MEAN, both B | 1.091–46.116 |
| F64 exact SUM/MEAN, both B | 1.020–1.081 |

The large B=32 reverse gaps are real observations in this short series and remain unexplained.
Their round ratios change sharply; neither steady-state parity nor a JIT cause is established.
No slow sample was discarded, oracle weakened, extra fork/retry/profiling campaign run, or speed
qualification inferred. A separate bounded diagnosis is needed before any performance-parity claim;
it is not an unimplemented acceptance item in this explicitly non-parity capability task.
Detailed evidence is retained in `/private/tmp/synaptik-prepare0015.gVDqlx/`: implementation handoff,
classfile manifest, root log/XML/counts, full timing report, `30-performance-summary.json` and
`31-measured-classfile-identity.json`. Final docs/examples/glossary support status and affected
Javadocs are synchronized; no further glossary meaning or Java behavior change was needed.
No required follow-up remains for this bounded capability. NN/Training, cold/wire/copy, low/native,
variable T/features, higher order and all-valid symbolic F64 remain held; no successor is authorized.
