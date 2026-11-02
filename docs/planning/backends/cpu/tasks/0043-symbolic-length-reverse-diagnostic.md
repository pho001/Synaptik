# Task CPU0043: Symbolic-length actual reverse diagnostic

## Status

Complete — MAIN accepted frozen diagnostic source/evidence 2026-10-27 after independent Class C APPROVE, zero open P1/P2, and verified module/preservation gates; final documentation closeout delegated exclusively to the independent review context, with MAIN retaining integration ownership.

## Change class

Class C — generated-code/hot-path diagnosis and conditional correction require clean implementation and independent targeted code/documentation review.

## Goal

Separate entry/warmup effects from a generated reverse-kernel defect in Prepare0015's B32 gap. Correct only a demonstrated local cause; justified no-change or bounded unknown is valid, not a parity claim.

## Scope and non-goals

One CPU-only witness: FLOAT32, original traversal=false, T4/F3/H2, actual Compiler-generated reverse, B32 problem and B7 control. Reuse one Family/class per JVM and both batch sizes; forward runs only establish/check saved sites, not another timing workload. Model source → actual Compiler sibling → generated entry remains unchanged.

Isolate original fixture entry versus prebound entry and typed Java reference; control batch order in fresh JVMs; correlate measurements with actual just-in-time (JIT) compilation levels, inlining and code size. Inspect emitted bytecode/dataflow against optimal specialized Java before considering existing emitter/executable correction. No inferred JIT parity or causal claim from timing alone.

No new production named type, API, protocol, memo, configuration, dependency, native work, Engine/Prepare/Compiler change, architecture/ADR, benchmark framework or certificate. No packing, algorithm/numerical/oracle weakening, saved-site elision, unsafe access, workflow expansion, F64 timing matrix or speed claim. Stop for architecture uncertainty, required new production type, or repair outside these existing contracts. Preserve all old timing methods, oracle algorithms and historical results; no Git mutation.

## Contracts

Read the root scope index and exact headings below, not the full API or predecessor histories:

- [Root](../../../../../ARCHITECTURE.md): `Authority, incorporation, and precedence`, `Core lifecycle`, `Core invariants`, `Scope-indexed normative contracts`, `Testing requirements`.
- [CPU owner](../../../../architecture/contracts/backend-execution.md): `Concrete backend modules`, `Performance evidence and optimization tooling`, `CPU backend routes`, `Accepted ordinary symbolic-batch CPU contract` (entry-cost disclosure and retained identities).
- [Scan owner](../../../../architecture/contracts/recurrent-scan.md): `Qualification, performance and migration gates`; `Accepted symbolic-batch scan` and its `Actual first-order closure and arithmetic`, `Empty domains and numeric resources`, `Qualification and mechanism gates`; `Length-aware whole-batch scan` including `Header, bindings and source admission`, `Bounded row-role analysis`, `Activity, raw identity and retained histories`, `First-order length-mode reverse`, `Runtime preflight and implementation gates`.
- [Lifecycle owner](../../../../architecture/contracts/runtime-prepare-engine.md): `modules/runtime`, `Run lifecycle` — cold typed binding, producer-ordered preflight and isolated run state remain unchanged.

Prepare0015's [accepted Result](../../../modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result), MAIN accepted 2026-10-27, fulfills stale pending-acceptance receipts. Contracts explicitly implement qualification; no semantic conflict or authority edit follows. Six core Scan semantic sections stay unchanged. Any prior receipt cleanup is outside CPU0043.

## Dependencies and integration

- Depends on: Prepare0015; CPU0041; CPU0042 — all Complete with accepted reviews/validation, retained dependencies satisfied.
- Conflicts with: CPU Scan generation/identity/executable/test harness and shared planning/docs; serialize all writers in that semantic scope.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus accepted dirty worktree in `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/prepare0015-final.json` (2485 files, empty index); not a HEAD diff or merged predecessors.
- Integration order: MAIN Ready/base verification → clean executor diagnostic/conditional repair → independent targeted code/docs review → MAIN acceptance/status maps.
- Integration validation: CPU0043 focused and final CPU test/Javadoc commands below; MAIN reuses source-matched evidence and reviews scope/links/identities once. No root rerun absent an actual cross-module change (which needs new scope approval).
- Shared-document integration owner: MAIN; executor/review provide bounded findings and Result, MAIN owns final status maps and any separately authorized shared-document edits.

MAIN launch verification 2026-10-27: Ready independently checked; user approval and accepted Complete dependencies verified. Only three planned Markdown paths differ from the accepted base; all 2483 other files/HEAD/origin/index unchanged. Planner closed; conflicts inactive, sole writer. CPU0043 alone is authorized; historical evidence manifest protects 1779 files.

## Files and symbols

Paths below are relative to `modules/backend/cpu/src/`; no new production package/type is selected.

- `test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneralScanTimingTest.java` — retain `symbolicLengthForwardReversePrepare0015`, `prepare0015Pairs`, `prepare0015Batch`; add only opt-in `symbolicLengthReverseCpu0043` if needed, using existing performance/evidenceRoot properties and launcher style.
- Same test package: `CpuSymbolicLengthScanFixture` (`Family`, `generated`, `verify`, typed direct closures) and `CpuSymbolicLengthScanCleanJavaOracle` (`reverseFloat32/64`) — original oracle/timing behavior retained; new test-local prebinding/control entry only if necessary. No per-call List.get in the prebound control; disclose remaining MethodHandle/spreader/Runnable/clock costs symmetrically, without subtracting an empty-call estimate as kernel time.
- Same test package: `CpuSymbolicScanGeneratedKernelTest` and `CpuOrdinaryDynamicBatchTest` — existing structural/numerical/compatibility regressions. Their hard-coded Prepare0015 evidence roots must gain test-local opt-in redirection through existing evidenceRoot before reruns; keep historical files untouched, default assertions/algorithms intact.
- `main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneralScanEmitter.java` — verified `Shaped.emitSymbolicLength()` calls `bindSymbolicGeometry()` then `emitLengthAware()`; inspect carrier/cut bindings and active loops. `CpuGeneralScanMatmulEmitter`, `CpuGeneralScanReductionEmitter` only for demonstrated cause.
- `main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuGeneralScanPreparedExecutable.java` — verified `bindCompatible(RunState, BufferRepresentation[], WorkspaceRepresentation[])`, private `Invocation.executeBound()`; conditional correction only if proved, preserving physical/length preflight.
- `main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuGeneralScanIr.java` and `internal/cache/CpuKernelSpecialization.java` — inspect compatibility for any actual emitted-byte change; apply existing stale-artifact rejection requirements, preserve unrelated identities/bytes. Stop if correction needs an authority change rather than silently freezing changed bytes under an old identity.

Explicit unresolved evidence only: Prepare0015 Result above; `/private/tmp/synaptik-prepare0015.gVDqlx/30-performance-summary.json`, `performance/prepare0015-report.txt`, verified `classfiles/`, `classfile-manifest.json`, `31-measured-classfile-identity.json`. Do not read completed CPU0041/0042 histories. Put new samples, commands, ClassFiles/hashes, JIT logs and interpretation in `/private/tmp/synaptik-cpu0043.LHBDZm/`.

## Acceptance criteria

1. Preserve historical evidence and explicitly identify the original confounds: B32 before B7 in the same class/JVM; 20 warmup pairs ×100 then 3×30 measured pairs ×100; generated wrapper's three list lookups/spreader/MethodHandle versus typed static Java. Original B32 reverse p50 range 41.129–78.837× and B7 1.344–1.462× are observations, not diagnosed causes.
2. Baseline runs differ only in B32→B7 versus B7→B32 order. Each uses the same F32/non-reversed Family and actual reverse, unchanged L[row]=row%5/nonuniform inputs/seeds, independent outputs/scratch and exact saved sites. Verify before/after measurements; retain poisoned-padding and raw saved-value checks outside timing.
3. Within each run compare original generated/direct entry with a prebound generated/control entry, holding kernel bytes, work, resources, checked segment ABI and fixed call counts constant. Report cold compile/generate/define/first entry separately; do not relabel wrapper-inclusive costs isolated inner-loop time. Separate any preflight/binding sample, never silently remove it from production.
4. Capture actual generated method identity, bytecode length, compilation level/transitions, inlining decisions/rejections and available compiled-code size for generated entry, wrapper and Java reverse at measurement windows. Log JVM uptime/phase boundaries for correlation; warmup counts never prove C2. Distinguish missing evidence from absent compilation; unavailable assembly tooling permits no extra campaign.
5. Compare actual measured ClassFiles (hash manifest) and optimal Java algorithm, hot loops, contributor order, carry/cut dataflow and hidden calls/allocation/boxing/reflection/dispatch. A local repair requires a specific demonstrated cause and regression protection, not a timing hypothesis. Never weaken oracle or change algorithm to look faster.
6. At most two baseline fresh-JVM order/control runs plus one conditional post-fix confirmation, each ≤60 seconds test-body admission including cold setup/checks, total ≤180 seconds. Fixed 20 warmup pairs then 3×30 measured pairs, 100 calls/side/pair for each of the two entry comparisons and two B values; alternate side order. No warm-to-stability, retries, five forks, matrix or extra JIT campaign. Check deadline before setup stages/calls; report censoring/overshoot and stop admitting work. Failed starts consume a run slot. No measurement during ordinary regression suites.
7. Third probe only after both baselines demonstrate a causal harness/production defect and a corresponding repair is made; repeat B32-first with identical protocol/logging. Creating diagnostic code or redirecting evidence roots is not such a repair. Otherwise stop after two baselines with justified no-change/bounded unknown; no forced rewrite, F64 speed or parity claim.
8. Any generator/executable repair passes both-dtype/both-traversal semantic and structural regressions, old identities/bytes and cache rejection checks; retained six Scan semantics, activity/preflight, saved rounding, O(parameter) accumulation and resources stay intact. Clean independent code/docs review is mandatory; MAIN alone accepts completion/status.

## Validation

Use MAIN's verified OpenJDK 26.0.1/logging support (metadata check, no run slot or causal proof). Preserve default compiler policy: no -Xcomp, forced thresholds or disabling DontCompileHugeMethods as a fix. Commands run from repository root after historical evidence redirection; regressions disable performance:

```bash
./gradlew :backends:cpu:test --tests '*.CpuSymbolicScanGeneratedKernelTest' --tests '*.CpuGeneralScanExecutionTest' --tests '*.CpuGeneralScanLoweringTest' --tests '*.CpuGeneratedKernelArtifactStoreTest' -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0043.LHBDZm/regression -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=1 --console=plain
./gradlew :backends:cpu:test :backends:cpu:javadoc -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0043.LHBDZm/final -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=1 --console=plain
```

Existing evidenceRoot's final directory selects order: `baseline-b32-first`, `baseline-b7-first`, conditional `postfix-b32-first`. Create external ephemeral `/private/tmp/synaptik-cpu0043.LHBDZm/probe.init.gradle` only: target `:backends:cpu`'s `Test` task `test`, set `outputs.upToDateWhen { false }`, `outputs.cacheIf { false }`, `maxParallelForks=1`, `forkEvery=1`; add only worker `jvmArgs` for UnlockDiagnosticVMOptions, LogCompilation, LogFile=<evidenceRoot>/hotspot_%p.log, PrintCompilation and PrintInlining. No global JVM environment instrumentation, upstream rerun flag or repository build edits. Each launch must verify JUnit XML: exactly 1 executed/0 skipped/0 failures/errors; record fresh worker PID/JDK/options and distinct order/root. Failed/missing execution consumes its slot, never retry. Exact commands after init creation:

```bash
run_cpu0043() {
  local run_dir="/private/tmp/synaptik-cpu0043.LHBDZm/$1"
  mkdir -p "$run_dir"
  ./gradlew -I /private/tmp/synaptik-cpu0043.LHBDZm/probe.init.gradle :backends:cpu:test --tests '*.CpuGeneralScanTimingTest.symbolicLengthReverseCpu0043' -Dsynaptik.cpu.generalScan.performance=true -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.evidenceRoot="$run_dir" --max-workers=1 --console=plain > "$run_dir/console.log" 2>&1
}
run_cpu0043 baseline-b32-first
run_cpu0043 baseline-b7-first
# Only after baseline-proved causal repair: run_cpu0043 postfix-b32-first
```

Inspect ClassFiles with existing inspection style/`javap -c -p`; record paths/commands and JIT correlations, without extra timing. Changed backend semantics require a stop; semantic-preserving repair reuses accepted conformance/integration evidence plus new CPU regressions. No root/native gate. Review reuses owning tests unless code/evidence changes require rerun.

## Documentation and review impact

Apply documentation rules/general/planning profiles and matching affected Javadoc/explanation profiles. Review Javadoc after Java edits; update changed contracts or justify accuracy. No new glossary meaning; targeted Scan/residual/hot-path search only. Justify any no-documentation-change conclusion. Independent Class C code/docs review covers diff/contracts, oracle fairness, logs/limits, links and historical preservation, reusing tests. MAIN owns integration/status; Prepare0015 stays accepted. Prior receipt cleanup is outside CPU0043.

Size exception: this completed brief exceeds 15 KB but remains below 25 KB to retain the issued protocol/commands and compact Result together as one atomic diagnostic; no scope expansion follows.

## Result

Status: Complete — justified no production change; exact historical cause and residual entry-inclusive overhead remain bounded unknowns.

- Changed only four CPU test files: `CpuGeneralScanTimingTest`, `CpuSymbolicLengthScanFixture`, `CpuSymbolicScanGeneratedKernelTest` and `CpuOrdinaryDynamicBatchTest`; added the opt-in diagnostic, captured controls, semantic/structural regression and evidence-root redirection. No production repair or identity change.
- Evidence root `/private/tmp/synaptik-cpu0043.LHBDZm/`: [executor handoff](/private/tmp/synaptik-cpu0043.LHBDZm/handoff.txt), [diagnosis and limits](/private/tmp/synaptik-cpu0043.LHBDZm/diagnosis.txt), raw samples/JIT XML, manifests and exact commands in `commands.txt`.
- Both actual workers reuse one F32/false Family for B32/B7, actual Compiler reverse, unchanged inputs/seeds, independent writable carriers and fixed 20+3×30 pairs ×100 calls/side; output, poisoned-padding/seed and raw saved-site checks pass outside timing.
- Original-entry aggregate paired p50: B32-first B32 **1.311890×**, then B7 **1.455868×**; B7-first B7 **82.564374×**, then B32 **1.868603×**. These crossover observations are not steady-state kernel ratios or a speed/parity claim.
- Actual PID23028/PID23879 logs show C1 level-3 on-stack-replacement (OSR) failures, `out of virtual registers in LIR generator`, for BOTH generated and Java reverse; invocation counts do not prove matching compilation.
- Normal C2 entry nmethod stamps: B32-first generated/Java **2.340/2.497s**, straddling changing B32 rounds; B7-first Java **1.128s** precedes large B7 windows, generated **1.996s** follows them and precedes later B32 measured rounds.
- OSR entries, wrapper/caller inlining and runtime deoptimization records were reviewed; installation/window correlation is not a controlled causal intervention or per-call tracing. Missing event stamps, logging perturbation and absent historical JIT traces limit attribution.
- Original always precedes prebound; inherited profiles, captured arguments/end bounds and constant traversal confound lookup-cost attribution. Later B32 prebound round p50 remains about 1.8–1.9×; B7-first prebound B7 aggregate remains 73.637662×. Residual overhead is not allocated to wrapper, kernel or JIT.
- Generated reverse retains activity-dominated loads, increasing row contributors, saved TANH sites, distinct contribution/carry ADD and O(parameter) storage. Extra binding/address dataflow and scalar derivative-input reads versus Java's literal `1f` do not alone prove avoidability or justify rewriting the oracle/emitter.
- All four measured forward/reverse ClassFiles and BOTH identity streams equal historical artifacts; old timing-method source and oracle are unchanged. No hidden allocation/boxing/reflection/graph dispatch or activation replay was found in measured reverse bytecode.
- Two actual baseline slots, each JUnit **1 executed/0 skipped/0 failures/errors**: uncensored bodies **2.181233125s + 1.689178375s = 3.8704115s**, no overshoot/retry/third probe. Neither control creation nor launcher correction justified a causal repair/confirmation.
- Launcher deviation: first launch retained **583 empty candidate workers plus PID23028**, 584 logs; empty logs are not kernel evidence. MAIN authorized only external `Test.include('**/CpuGeneralScanTimingTest.class')` before run two, which retained one worker/log. Archived init/options show identical actual-worker flags except evidence root; startup overhead is separate from body time.
- Focused command above passed **117 executed/0 skipped/0 failures/errors** after a pre-start wrapper-lock permission failure; no timing slot was consumed by that regression launch. Final CPU test/Javadoc command above passed **1276 executed +45 skipped/1321 discovered, 0 failures/errors**; Javadoc **UP-TO-DATE**, no production-source change or test-source Javadoc generation claimed.
- Source/scope/history audits pass: **1779 historical artifacts** hash-unchanged, **2479 protected repository files** unchanged against the recorded audit base; HEAD/origin/index unchanged. Independent review reused executable evidence; no further suite/probe.
- Documentation impact: only this task, CPU master plan and roadmap closeout; no explanatory/authority/glossary change because production semantics/API/lifecycle/terminology are unchanged. Added test Javadocs and retained contracts reviewed as accurate under general/planning/API-and-Javadoc profiles and targeted glossary search.
- Independent Class C source/evidence/Javadoc and final documentation review: **APPROVE, zero open P1/P2**. MAIN accepted frozen source/evidence and retains final integration/receipt acceptance. Active frontier **None**; no automatic successor or further timing. Prior receipt cleanup remains outside scope.
