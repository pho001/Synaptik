# Task CPU0038: Repair symbolic Scan timing methodology

## Status

Complete — MAIN accepted 2026-10-25 after Sagan's final independent APPROVE (zero open P1/P2)
and the single post-freeze CPU probe. Worker/reviewer closed; CPU test and production frontiers
None. No successor or further probe authorized. Validation provenance and limits are in Result.

## Change class

Class A — bounded test methodology/regressions and focused documentation only; no production
semantics, numerics, API, lifecycle, generated code, identity, dependency or hot-path contract changes.
Separate planning/implementation and final independent read-only review are explicitly required.

## Goal

Measure same-method generated forward/reverse reuse across runtime B versus fixed-T/F/H clean
Java; separate cold setup/fixed warmup from observational windows. Equal Class-File bytes alone
do not establish actual method/JIT-profile reuse.

## Scope

Repair only the CPU symbolic Scan probe and deterministic default regressions. Timed families are
T3/F4/H3 and T4/F32/H32, FLOAT32 TANH, with runtime B. Retain generic fixture/oracle behavior for
other T/F/H and SIGMOID consumers. Use the existing generated-kernel seam, a small test-local
prepared-family holder and per-B fixture binding; no general benchmark framework or runtime cache.

## Non-goals

No production emitter/IR/binder/resource/cache/type/API/architecture/ADR/build changes, JVM tuning,
C2 eligibility gate, Engine edits/remeasurement, native build, other CPU probes, root suite,
adaptive warm-until-winning, extra forks/campaign, commit/push or downstream launch.
Do not rewrite completed Prepare0014 evidence or the static baseline TSV.

## Contracts

- [ARCHITECTURE.md](../../../../../ARCHITECTURE.md): `Scope-indexed normative contracts`,
  `Core lifecycle`, `Core invariants`, `Testing requirements` — unchanged ownership/lifecycles.
- [Backend execution](../../../../architecture/contracts/backend-execution.md):
  `Performance evidence and optimization tooling`, `CPU backend routes` — observational evidence,
  specialized clean-Java/Class-File comparison, no production selection or hidden hot-loop overhead.
- [Recurrent scan](../../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan):
  `Accepted symbolic-batch scan` and its subheadings — fixed T/features, actual reverse, saved sites,
  empty B and immutable code with isolated invocation resources. Stop on ambiguity or scope expansion.

## Dependencies and integration

- Depends on: [Prepare0014](../../../modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md#result)
  and [CPU0037](0037-general-scan-dense-read-views.md#result), both Complete/accepted.
- Conflicts with: symbolic-Scan fixture/oracle/probe and shared CPU docs; serialize production Scan
  changes. No competing writer; MAIN admission audit read-only.
- Parallel group: None; sole serial writer, no concurrent-write worktree.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus ACCEPTED
  DIRTY tree, not merged predecessors. Exact baseline:
  `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/cpu0038-pre-plan.json` and
  `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/cpu0038-pre-plan-files/`.
- Integration order: MAIN admission → clean worker/tests → independent read-only review/corrections
  → freeze → MAIN single probe → closure; all complete. No plan-review gate or worker opt-in.
- Integration validation: one pre-correction CPU-module run plus final corrected focused tests;
  independent review, MAIN baseline/scope/link checks and single post-freeze probe. No root checkpoint.
- Shared-document integration owner: MAIN; worker Result/guide ownership ended at implementation
  closure. Planning closure: brief, CPU master, roadmap only.

MAIN admission 2026-10-25 19:46:43 UTC: Ready/sole TEST frontier; current Status/Result verified
Prepare0014 accepted 2026-10-25, CPU0037 accepted 2026-10-24, both Complete/reviewed/gated;
CPU0030–0036 Status Complete. CPU0038 unused; prior frontier None. Exact 2468-file pre-plan match,
unchanged HEAD/origin/empty index, all 1462 existing Java executable pins frozen; accepted dirty work
preserved, no production authorization.

## Files and symbols

Exclusive executable write allowlist (all in
`modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/`):

- `CpuSymbolicScanTimingTest.java`: bounded probe, measure/sample/packing helpers and Fixture;
  small nested prepared-family holder may retain the two CpuGeneratedKernel artifacts and recipes.
  Preserve `Fixture(int,int,int,int)` and `Fixture(int,int,int,int,boolean)` plus generic `direct()`.
- New `CpuSymbolicScanTimingMethodologyTest.java`: default identity/compatibility/reset/resource/
  five-output/saved-site regressions, not guarded by the performance property.
- New `CpuSymbolicScanCleanJavaOracle.java`: two fixed-T/F/H TANH methods, B runtime; compile-time
  specialization for timing, separate from the old generic direct method.
- `CpuSymbolicScanGeneratedKernelTest.java`: only focused regression assertions/adaptation if needed;
  preserve existing saved-activation injection, static compatibility and generic-fixture tests.

Read-only seams: `CpuGeneralScanGeneratedKernelTest.generated/specialization`,
`CpuGeneratedKernel.hiddenClass/entryPoint/classBytes/specialization`, and
`CpuSymbolicScanGeneratedKernelTest.inspect`. No reflective wrapper or production edit.
Read-only resource: the sibling test-resources `symbolic-scan-static-baseline.tsv`, byte-for-byte.
Documentation write scope: this brief; MAIN-only `../master-plan.md` and `../../../roadmap.md`;
worker-only `docs/backend-guide/cpu-backend.md#symbolic-scan-cpu-execution`, one short linking
paragraph for CPU0038, preserving the original Prepare0014 timings/limits (currently lines4318–4331).

## Acceptance criteria

1. Prepare/compile/lower/generate/define exactly one forward + actual Compiler reverse pair per
   timed family per probe body, before per-B bindings. Reuse the SAME hidden Class and direct entry
   handle for both phases across B; retain artifacts, not just equal bytes/hashes/names. Per-B
   binding performs no generation/definition. No global/static mutable cache or per-B specialization.
2. Default regressions for BOTH families cover 64→40→1→64→0, assertSame hiddenClass/entryPoint
   for each phase, stable bytes/compatibility, separate fresh packed arrays/native writable storage,
   deterministic resets/repeated calls and closing one fixture without invalidating another.
   Family mismatch (T/F/H or activation/meaning/ordered ABI/compatibility) rejects before kernel
   invocation/writes; test both dimensional and TANH-vs-SIGMOID mismatch, not hash equality alone.
3. Before AND after each timed case, verify all five outputs (final carry, emission history,
   dInitialCarry, dX, dW), represented saved activation sites and matched mandatory read/write byte
   footprints, distinct public/saved buffers and zero workspace for these proved fixtures. Default
   B=0 tests still check nonempty parameter gradients. Preserve existing finite tolerance, exact/raw
   saved-site tests and all old generic constructors/semantics, including T1/B1/F4/H4 TANH/SIGMOID.
4. Timed oracle uses fixed T/F/H literals in two clean Java methods, runtime B, same native storage,
   joint seeds, mandatory histories/outputs, constituent FLOAT32 rounding and contributor order.
   Match actual forward/reverse stage order (including dW/dX), no activation replay, scratch omission,
   algebraic shortcut, per-element allocation/boxing/reflection/dispatch or runtime shape selection.
   Select the family outside hot loops; the old generic `Fixture.direct()` is NOT the timed oracle.
5. Source plus actual generated/oracle Class-File inspection checks algorithm/loop/dataflow,
   represented arithmetic and hidden helper/allocation/dispatch risks; retain existing automated
   generated inspection. Disclose generated two-entry versus direct-call entry costs, not pure-loop
   or full Engine cost. Report numeric/access packing separately, never label it full Engine BIND.
6. Fixed protocol below, off by default, no timing/speed/C2-threshold assertions. Report all sample
   means/order/counts and limitations; no parity, negligible-B-overhead or steady-state guarantee.

## Validation

Original plan: focused regressions then ONE whole-CPU run on stabilized source, opt-ins disabled.
Actual full run preceded corrections; MAIN accepts it with final corrected focused evidence, NOT
a full suite on the final freeze (see Result). Recorded worker commands; no further run authorized:

```bash
JDK_JAVA_OPTIONS='--add-modules=jdk.incubator.vector' ./gradlew --no-daemon --no-parallel --max-workers=1 :backends:cpu:test --tests '*CpuSymbolicScanTimingMethodologyTest' --tests '*CpuSymbolicScanGeneratedKernelTest' -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --no-build-cache --console=plain
JDK_JAVA_OPTIONS='--add-modules=jdk.incubator.vector' ./gradlew --no-daemon --no-parallel --max-workers=1 :backends:cpu:test --rerun -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --no-build-cache --console=plain
git diff --check
```

Worker ran NO opt-in probe. After independent review/corrections and source freeze, MAIN alone
ran exactly ONE final opt-in CPU probe, forced execution, one measured test JVM/body:

```bash
JDK_JAVA_OPTIONS='--add-modules=jdk.incubator.vector -XX:+UnlockDiagnosticVMOptions -XX:+LogCompilation -XX:LogFile=/private/tmp/synaptik-cpu0038-main.MVvZxq/hotspot-%p.xml' ./gradlew --no-daemon --no-parallel --max-workers=1 :backends:cpu:test --rerun --tests '*CpuSymbolicScanTimingTest' -Dsynaptik.cpu.dynamicBatch.performance=true -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --no-build-cache --console=plain
```

One monotonic 30-second TOTAL body deadline includes both families' cold setup, bindings, checks,
startup/warmup, samples and packing measurement; not 30s per family. Check it within bounded work
loops and at phase boundaries; report any noninterruptible-setup overshoot honestly. Stop/censor
unfinished phases/remaining cases at cap, not successful zero/partial samples; no retry or increased
budget without approval. Build/Javadoc duration is separate from the body cap.
Cold compile/lower/generate/define and per-B allocation/binding timings/counts are separate from
kernel windows. Startup warms BOTH variants at the same representative B64: 20,000 alternating
pairs for T3/F4/H3; 3,000 for T4/F32/H32 (larger contractions justify fewer calls). Then small-family
timed cases B1/40/64 and large-family B64 each receive 1,000 fixed alternating warmup pairs,
followed by three paired 300-call samples, generated/direct order G-D, D-G, G-D. No reset of
defined methods between B values. Packing-only observation keeps fixed 100 + 3×300 calls under
the SAME deadline; omit/censor it if budget exhausted. Counts never imply compilation readiness.

Retain final XML/system-out, cold/startup/warm/sample boundaries and JVM/options plus Class/entry
identity correlation. LogCompilation is diagnostic observation only: inspect actual forward AND
reverse normal-entry C2 versus OSR and their timing relative to windows, never eligibility or a
reason to extend/retry. If logging is unsupported, preserve that limitation without a replacement
performance run/build edit. If absent/late/unknown, say JIT readiness unverified/transitional;
call windows post-fixed-warmup observations, not proven steady state. No special compiler thresholds.
Compare all repository files to the exact pre-plan SHA-256 manifest: only this allowlist/planning
delta is permitted; in particular every production Java/native/build file and static TSV stays
byte-identical. Record unchanged HEAD/origin/index, new-file whitespace and local link/anchor checks.

## Documentation and review impact

Test Javadocs cover inputs/results/failures, ownership/reset/timing; documentation-rules.md and
general/planning/API-and-Javadoc/backend/developer-guide profiles applied. Sagan's final read-only
review covered identity/fairness/cap, actual Class-Files, Javadocs and guide link, reusing tests.
Targeted glossary review: symbolic batch binding, Symbolic Scan, generated CPU computation kernel;
no meaning changes. Historical evidence preserved; unchanged production/API/Javadoc permits reuse
of accepted Prepare0014 module/root generation; test Javadocs source-reviewed. No new review gate.

## Result

Status: Complete — MAIN accepted 2026-10-25; Sagan final read-only APPROVE, zero open P1/P2.
Changed: TimingTest, new CleanJavaOracle/TimingMethodologyTest, guide paragraph and three planning
files; GeneratedKernelTest/static TSV unchanged. True Class/handle reuse per family across B,
isolated resources and default regressions verified; generic/raw saved-activation tests retained.
Both P2s fixed: immediate post-bind cap guards; semantic literal `1f`/`0f + sum`, not forced
generator-only constant reads (52/4100 bytes disclosed separately). Four emitted Class-Files unchanged.
ONE pre-correction full CPU run PASS: 1259 tests, 40 skips, zero failures/errors; archived full XML.
Final corrected focused PASS: 15 tests, zero skips/failures/errors. NOT a full suite on final freeze.
MAIN's single post-review/source-freeze probe PASS: forced/noncached, 14s Gradle, one test/zero skips,
failures/errors/censoring; body 8.755825291s<30s, all four cases/24 complete 300-call windows and
pre/post all-five-output/saved-site/footprint checks complete. No worker opt-in or review reruns.
Median window means, generated/direct µs (ratio): T3/F4/H3 B1 0.523333/0.509583 (1.026983),
B40 11.010973/8.646390 (1.273476), B64 18.599030/13.662500 (1.361320);
T4/F32/H32 B64 994.624167/385.224027 (2.581937). No row/sample discarded.
All six normal-entry C2 methods available before windows, no recorded invalidations of those IDs;
availability is not exact executed assembly or universal steady state. Old 26× is not valid steady-
state evidence; this repairs methodology, NOT production acceleration/parity. Larger 2.58× gap remains.
MAIN evidence: `/private/tmp/synaptik-cpu0038-main.MVvZxq/{result.txt,analysis.json,probe.xml}`;
worker: `/private/tmp/synaptik-cpu0038-worker.1r31WF/{worker-report.txt,java-source-freeze.sha256}`.
MAIN's four source pins matched before/after probe; all 726 production Java/non-allowlisted files
unchanged, exactly seven task paths, HEAD/origin/index unchanged. Javadocs/guide/glossary review final;
unchanged production module/root Javadoc reused. No unresolved repair issue/required follow-up;
worker/reviewer closed, CPU test/production frontiers None, no successor or further probe authorized.
