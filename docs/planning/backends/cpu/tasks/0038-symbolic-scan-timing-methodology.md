# Task CPU0038: Repair [symbolic Scan](../../../../glossary.md#symbolic-scan) timing methodology

## Reader context — historical integration boundary

A symbolic-batch prepared family keeps the same generated methods while B changes per invocation;
reconstructing equal class bytes in a new class would not establish the same JIT profile. This
test-only repair therefore distinguishes family/method reuse, per-B binding, fixed warmup and
measured windows. The generated entry and specialized Java reference must perform matching
public/saved writes, while generator-only constant storage is disclosed separately. Neither
post-warmup sampling nor C2 availability guarantees steady state. The older large ratio is retained
as invalid steady-state evidence, not silently relabeled as a production speedup.

Current reading: [Symbolic-batch Scan execution](../../../../backend-guide/cpu-backend.md#symbolic-scan-cpu-execution).

## Status

Complete — MAIN accepted 2026-10-25 after Sagan's final independent APPROVE (zero open P1/P2)
and the single post-freeze [CPU](../../../../glossary.md#term-cpu) probe. Worker/reviewer closed; CPU test and production frontiers
None. No successor or further probe authorized. [Validation](../../../../glossary.md#term-validation) [provenance](../../../../glossary.md#provenance) and limits are in Result.

## Change class

Class A — bounded test methodology/regressions and focused documentation only; no production
semantics, numerics, [API](../../../../glossary.md#term-application-programming-interface-api), [lifecycle](../../../../glossary.md#lifecycle), generated code, [identity](../../../../glossary.md#term-identity-meanings), [dependency](../../../../glossary.md#term-dependency) or hot-path contract changes.
Separate [planning](../../../../glossary.md#term-planning)/implementation and final independent read-only review are explicitly required.

## Goal

Measure same-method generated forward/reverse reuse across [runtime](../../../../glossary.md#term-runtime) B versus fixed-T/F/H clean
Java; separate cold setup/fixed [warmup](../../../../glossary.md#term-warmup) from observational windows. Equal Class-File bytes alone
do not establish actual method/[JIT](../../../../glossary.md#term-just-in-time-compilation-jit)-profile reuse.

## [Scope](../../../../glossary.md#term-scope)

Repair only the [CPU](../../../../glossary.md#term-cpu) [symbolic Scan](../../../../glossary.md#symbolic-scan) probe and deterministic default regressions. Timed families are
T3/F4/H3 and T4/[F32](../../../../glossary.md#floating-point-precision-and-rounding)/H32, [FLOAT32](../../../../glossary.md#term-float32) [TANH](../../../../glossary.md#term-tanh), with [runtime](../../../../glossary.md#term-runtime) B. Retain generic [fixture](../../../../glossary.md#term-fixture)/[oracle](../../../../glossary.md#term-test-oracle) behavior for
other T/F/H and [SIGMOID](../../../../glossary.md#term-sigmoid) consumers. Use the existing generated-[kernel](../../../../glossary.md#kernel) seam, a small test-local
prepared-family holder and per-B fixture [binding](../../../../glossary.md#term-binding); no general benchmark framework or runtime [cache](../../../../glossary.md#term-cache).

## Non-goals

No production emitter/[IR](../../../../glossary.md#term-intermediate-representation-ir)/binder/resource/[cache](../../../../glossary.md#term-cache)/type/[API](../../../../glossary.md#term-application-programming-interface-api)/architecture/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/build changes, [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) tuning,
[C2](../../../../glossary.md#term-c2) eligibility gate, [Engine](../../../../glossary.md#term-engine) edits/remeasurement, native build, other [CPU](../../../../glossary.md#term-cpu) probes, root suite,
adaptive warm-until-winning, extra [forks](../../../../glossary.md#term-fork-meanings)/campaign, [commit](../../../../glossary.md#term-commit)/push or downstream launch.
Do not rewrite completed Prepare0014 evidence or the static baseline TSV.

## Contracts

- [ARCHITECTURE.md](../../../../../ARCHITECTURE.md): `Scope-indexed normative contracts`,
  `Core lifecycle`, `Core invariants`, `Testing requirements` — unchanged [ownership](../../../../glossary.md#term-ownership-meanings)/lifecycles.
- [Backend execution](../../../../architecture/contracts/backend-execution.md)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>:
  `Performance evidence and optimization tooling`, `CPU backend routes` — observational evidence,
  specialized clean-Java/Class-File comparison, no production selection or hidden hot-loop overhead.
- [Recurrent scan](../../../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan):
  `Accepted symbolic-batch scan` and its subheadings — fixed T/features, actual reverse, saved sites,
  empty B and immutable code with isolated invocation resources. Stop on ambiguity or scope expansion.<span data-glossary-navigation="true"> ([features glossary](../../../../glossary.md#term-feature); [immutable glossary](../../../../glossary.md#term-immutable); [scope glossary](../../../../glossary.md#term-scope))</span>

## Dependencies and integration

- Depends on: [Prepare0014](../../../modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md#result)
  and [CPU0037](0037-general-scan-dense-read-views.md#result), both Complete/accepted.
- Conflicts with: symbolic-Scan fixture/oracle/probe and shared CPU docs; serialize production Scan
  changes. No competing writer; MAIN admission audit read-only.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [oracle glossary](../../../../glossary.md#term-test-oracle); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Parallel group: None; sole serial writer, no concurrent-write worktree.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` plus ACCEPTED
  DIRTY tree, not merged predecessors. Exact baseline:
  `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/cpu0038-pre-plan.json` and
  `/private/tmp/synaptik-dynamic-batch-cutover.Of4Q8D/cpu0038-pre-plan-files/`.
- Integration order: MAIN admission → clean worker/tests → independent read-only review/corrections
  → freeze → MAIN single probe → closure; all complete. No plan-review gate or worker opt-in.
- Integration validation: one pre-correction CPU-module run plus final corrected focused tests;
  independent review, MAIN baseline/scope/link checks and single post-freeze probe. No root checkpoint.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module); [run glossary](../../../../glossary.md#term-run); [scope glossary](../../../../glossary.md#term-scope); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Shared-document integration owner: MAIN; worker Result/guide ownership ended at implementation
  closure. Planning closure: brief, CPU master, roadmap only.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Planning glossary](../../../../glossary.md#term-planning))</span>

MAIN [admission](../../../../glossary.md#term-admission) 2026-10-25 19:46:43 UTC: [Ready](../../../../glossary.md#term-ready)/sole TEST frontier; current Status/Result verified
Prepare0014 accepted 2026-10-25, CPU0037 accepted 2026-10-24, both Complete/reviewed/gated;
CPU0030–0036 Status Complete. CPU0038 unused; prior frontier None. Exact 2468-file pre-plan match,
unchanged HEAD/origin/empty index, all 1462 existing Java executable pins frozen; accepted dirty work
preserved, no production authorization.

## Files and symbols

Exclusive executable write allowlist (all in
`modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/`):

- `CpuSymbolicScanTimingTest.java`: bounded probe, measure/sample/packing helpers and Fixture;
  small nested prepared-family holder may retain the two CpuGeneratedKernel artifacts and recipes.
  Preserve <span data-glossary-navigation="true"> ([sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [Fixture glossary](../../../../glossary.md#term-fixture); [artifacts glossary](../../../../glossary.md#term-artifact))</span>`Fixture(int,int,int,int)` and `Fixture(int,int,int,int,boolean)` plus generic `direct()`.
- New `CpuSymbolicScanTimingMethodologyTest.java`: default identity/compatibility/reset/resource/
  five-output/saved-site regressions, not guarded by the performance property.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- New `CpuSymbolicScanCleanJavaOracle.java`: two fixed-T/F/H TANH methods, B runtime; compile-time
  specialization for timing, separate from the old generic direct method.<span data-glossary-navigation="true"> ([TANH glossary](../../../../glossary.md#term-tanh); [runtime glossary](../../../../glossary.md#term-runtime); [compile glossary](../../../../glossary.md#compile); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- `CpuSymbolicScanGeneratedKernelTest.java`: only focused regression assertions/adaptation if needed;
  preserve existing saved-activation injection, static compatibility and generic-fixture tests.<span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression); [activation glossary](../../../../glossary.md#term-activation))</span>

Read-only seams: `CpuGeneralScanGeneratedKernelTest.generated/specialization`,
`CpuGeneratedKernel.hiddenClass/entryPoint/classBytes/specialization`, and
`CpuSymbolicScanGeneratedKernelTest.inspect`. No reflective wrapper or production edit.
Read-only resource: the sibling test-resources `symbolic-scan-static-baseline.tsv`, byte-for-byte.
Documentation write [scope](../../../../glossary.md#term-scope): this brief; MAIN-only `../master-plan.md` and `../../../roadmap.md`;
worker-only `docs/backend-guide/cpu-backend.md#symbolic-scan-cpu-execution`, one short linking
paragraph for CPU0038, preserving the original Prepare0014 timings/limits (currently lines4318–4331).

## Acceptance criteria

1. Prepare/compile/lower/generate/define exactly one forward + actual Compiler reverse pair per
   timed family per probe body, before per-B bindings. Reuse the SAME hidden Class and direct entry
   handle for both phases across B; retain artifacts, not just equal bytes/hashes/names. Per-B
   binding performs no generation/definition. No global/static mutable cache or per-B specialization.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [compile glossary](../../../../glossary.md#compile); [Compiler glossary](../../../../glossary.md#term-compiler); [bindings glossary](../../../../glossary.md#term-binding); [hidden Class glossary](../../../../glossary.md#term-hidden-class); [artifacts glossary](../../../../glossary.md#term-artifact); [mutable glossary](../../../../glossary.md#term-mutable); [cache glossary](../../../../glossary.md#term-cache); [specialization glossary](../../../../glossary.md#term-specialization))</span>
2. Default regressions for BOTH families cover 64→40→1→64→0, assertSame hiddenClass/entryPoint
   for each phase, stable bytes/compatibility, separate fresh packed arrays/native writable storage,
   deterministic resets/repeated calls and closing one fixture without invalidating another.
   Family mismatch (T/F/H or activation/meaning/ordered ABI/compatibility) rejects before kernel
   invocation/writes; test both dimensional and TANH-vs-SIGMOID mismatch, not hash equality alone.<span data-glossary-navigation="true"> ([fixture glossary](../../../../glossary.md#term-fixture); [activation glossary](../../../../glossary.md#term-activation); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [kernel glossary](../../../../glossary.md#kernel); [TANH glossary](../../../../glossary.md#term-tanh); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [hash glossary](../../../../glossary.md#term-hash))</span>
3. Before AND after each timed case, verify all five outputs (final carry, emission history,
   dInitialCarry, dX, dW), represented saved activation sites and matched mandatory read/write byte
   footprints, distinct public/saved buffers and zero workspace for these proved fixtures. Default
   B=0 tests still check nonempty parameter gradients. Preserve existing finite tolerance, exact/raw
   saved-site tests and all old generic constructors/semantics, including T1/B1/F4/H4 TANH/SIGMOID.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [buffers glossary](../../../../glossary.md#term-buffer); [parameter glossary](../../../../glossary.md#term-parameter); [gradients glossary](../../../../glossary.md#term-gradient); [finite glossary](../../../../glossary.md#term-finite); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
4. Timed oracle uses fixed T/F/H literals in two clean Java methods, runtime B, same native storage,
   joint seeds, mandatory histories/outputs, constituent FLOAT32 rounding and contributor order.
   Match actual forward/reverse stage order (including dW/dX), no activation replay, scratch omission,
   algebraic shortcut, per-element allocation/boxing/reflection/dispatch or runtime shape selection.
   Select the family outside hot loops; the old generic <span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [FLOAT32 glossary](../../../../glossary.md#term-float32); [rounding glossary](../../../../glossary.md#term-rounding); [element glossary](../../../../glossary.md#term-element); [dispatch glossary](../../../../glossary.md#term-dispatch); [shape glossary](../../../../glossary.md#shape))</span>`Fixture.direct()` is NOT the timed [oracle](../../../../glossary.md#term-test-oracle).
5. Source plus actual generated/oracle Class-File inspection checks algorithm/loop/dataflow,
   represented arithmetic and hidden helper/allocation/dispatch risks; retain existing automated
   generated inspection. Disclose generated two-entry versus direct-call entry costs, not pure-loop
   or full Engine cost. Report numeric/access packing separately, never label it full Engine BIND.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [Engine glossary](../../../../glossary.md#term-engine))</span>
6. Fixed protocol below, off by default, no timing/speed/C2-threshold assertions. Report all sample
   means/order/counts and limitations; no parity, negligible-B-overhead or steady-state guarantee.<span data-glossary-navigation="true"> ([C2 glossary](../../../../glossary.md#term-c2); [sample glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>

## [Validation](../../../../glossary.md#term-validation)

Original plan: focused regressions then ONE whole-[CPU](../../../../glossary.md#term-cpu) [run](../../../../glossary.md#term-run) on stabilized source, opt-ins disabled.
Actual full run preceded corrections; MAIN accepts it with final corrected focused evidence, NOT
a full suite on the final freeze (see Result). Recorded worker commands; no further run authorized:

```bash
JDK_JAVA_OPTIONS='--add-modules=jdk.incubator.vector' ./gradlew --no-daemon --no-parallel --max-workers=1 :backends:cpu:test --tests '*CpuSymbolicScanTimingMethodologyTest' --tests '*CpuSymbolicScanGeneratedKernelTest' -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --no-build-cache --console=plain
JDK_JAVA_OPTIONS='--add-modules=jdk.incubator.vector' ./gradlew --no-daemon --no-parallel --max-workers=1 :backends:cpu:test --rerun -Dsynaptik.cpu.dynamicBatch.performance=false -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --no-build-cache --console=plain
git diff --check
```

Worker ran NO opt-in probe. After independent review/corrections and source freeze, MAIN alone
ran exactly ONE final opt-in CPU probe, forced execution, one measured test [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)/body:

```bash
JDK_JAVA_OPTIONS='--add-modules=jdk.incubator.vector -XX:+UnlockDiagnosticVMOptions -XX:+LogCompilation -XX:LogFile=/private/tmp/synaptik-cpu0038-main.MVvZxq/hotspot-%p.xml' ./gradlew --no-daemon --no-parallel --max-workers=1 :backends:cpu:test --rerun --tests '*CpuSymbolicScanTimingTest' -Dsynaptik.cpu.dynamicBatch.performance=true -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false --no-build-cache --console=plain
```

One monotonic 30-second TOTAL body deadline includes both families' cold setup, [bindings](../../../../glossary.md#term-binding), checks,
startup/[warmup](../../../../glossary.md#term-warmup), [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) and packing measurement; not 30s per family. Check it within bounded work
loops and at phase boundaries; report any noninterruptible-setup overshoot honestly. Stop/censor
unfinished phases/remaining cases at cap, not successful zero/partial samples; no retry or increased
budget without approval. Build/[Javadoc](../../../../glossary.md#term-javadoc) duration is separate from the body cap.
Cold [compile](../../../../glossary.md#compile)/lower/generate/define and per-B allocation/binding timings/counts are separate from
[kernel](../../../../glossary.md#kernel) windows. Startup warms BOTH variants at the same representative B64: 20,000 alternating
pairs for T3/F4/H3; 3,000 for T4/[F32](../../../../glossary.md#floating-point-precision-and-rounding)/H32 (larger contractions justify fewer calls). Then small-family
timed cases B1/40/64 and large-family B64 each receive 1,000 fixed alternating warmup pairs,
followed by three paired 300-call samples, generated/direct order G-D, D-G, G-D. No reset of
defined methods between B values. Packing-only observation keeps fixed 100 + 3×300 calls under
the SAME deadline; omit/censor it if budget exhausted. Counts never imply [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) readiness.

Retain final XML/system-out, cold/startup/warm/sample boundaries and JVM/options plus Class/entry
[identity](../../../../glossary.md#term-identity-meanings) correlation. LogCompilation is diagnostic observation only: inspect actual forward AND
reverse normal-entry [C2](../../../../glossary.md#term-c2) versus [OSR](../../../../glossary.md#term-on-stack-replacement-osr) and their timing relative to windows, never eligibility or a
reason to extend/retry. If logging is unsupported, preserve that limitation without a replacement
performance run/build edit. If absent/late/unknown, say [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) readiness unverified/transitional;
call windows post-fixed-warmup observations, not proven [steady state](../../../../glossary.md#term-steady-state). No special [compiler](../../../../glossary.md#term-compiler) thresholds.
Compare all repository files to the exact pre-plan [SHA-256](../../../../glossary.md#term-sha-256) [manifest](../../../../glossary.md#term-manifest): only this allowlist/[planning](../../../../glossary.md#term-planning)
delta is permitted; in particular every production Java/native/build file and static TSV stays
byte-identical. Record unchanged HEAD/origin/index, new-file whitespace and local link/anchor checks.

## Documentation and review impact

Test Javadocs cover inputs/results/failures, [ownership](../../../../glossary.md#term-ownership-meanings)/reset/timing; documentation-rules.md and
general/[planning](../../../../glossary.md#term-planning)/[API](../../../../glossary.md#term-application-programming-interface-api)-and-[Javadoc](../../../../glossary.md#term-javadoc)/[backend](../../../../glossary.md#backend)/developer-guide profiles applied. Sagan's final read-only
review covered [identity](../../../../glossary.md#term-identity-meanings)/fairness/cap, actual Class-Files, Javadocs and guide link, reusing tests.
Targeted glossary review: [symbolic batch binding](../../../../glossary.md#symbolic-batch-binding), [Symbolic Scan](../../../../glossary.md#symbolic-scan), [generated CPU computation kernel](../../../../glossary.md#generated-cpu-computation-kernel);
no meaning changes. Historical evidence preserved; unchanged production/API/Javadoc permits reuse
of accepted Prepare0014 [module](../../../../glossary.md#term-module)/root generation; test Javadocs source-reviewed. No new review gate.

## Result

Status: Complete — MAIN accepted 2026-10-25; Sagan final read-only APPROVE, zero open P1/P2.
Changed: TimingTest, new CleanJavaOracle/TimingMethodologyTest, guide paragraph and three [planning](../../../../glossary.md#term-planning)
files; GeneratedKernelTest/static TSV unchanged. True Class/handle reuse per family across B,
isolated resources and default regressions verified; generic/raw saved-[activation](../../../../glossary.md#term-activation) tests retained.
Both P2s fixed: immediate post-bind cap guards; semantic literal `1f`/`0f + sum`, not forced
generator-only [constant](../../../../glossary.md#term-constant) reads (52/4100 bytes disclosed separately). Four emitted Class-Files unchanged.
ONE pre-correction full [CPU](../../../../glossary.md#term-cpu) [run](../../../../glossary.md#term-run) PASS: 1259 tests, 40 skips, zero failures/errors; archived full XML.
Final corrected focused PASS: 15 tests, zero skips/failures/errors. NOT a full suite on final freeze.
MAIN's single post-review/source-freeze probe PASS: forced/noncached, 14s [Gradle](../../../../glossary.md#term-gradle), one test/zero skips,
failures/errors/[censoring](../../../../glossary.md#benchmark-measurement-vocabulary); body 8.755825291s<30s, all four cases/24 complete 300-call windows and
pre/post all-five-output/saved-site/footprint checks complete. No worker opt-in or review reruns.
[Median](../../../../glossary.md#term-median) window means, generated/direct µs (ratio): T3/F4/H3 B1 0.523333/0.509583 (1.026983),
B40 11.010973/8.646390 (1.273476), B64 18.599030/13.662500 (1.361320);
T4/[F32](../../../../glossary.md#floating-point-precision-and-rounding)/H32 B64 994.624167/385.224027 (2.581937). No row/[sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) discarded.
All six normal-entry [C2](../../../../glossary.md#term-c2) methods available before windows, no recorded invalidations of those IDs;
availability is not exact executed assembly or universal [steady state](../../../../glossary.md#term-steady-state). Old 26× is not valid steady-
state evidence; this repairs methodology, NOT production acceleration/parity. Larger 2.58× gap remains.
MAIN evidence: `/private/tmp/synaptik-cpu0038-main.MVvZxq/{result.txt,analysis.json,probe.xml}`;
worker: `/private/tmp/synaptik-cpu0038-worker.1r31WF/{worker-report.txt,java-source-freeze.sha256}`.
MAIN's four source pins matched before/after probe; all 726 production Java/non-allowlisted files
unchanged, exactly seven task paths, HEAD/origin/index unchanged. Javadocs/guide/glossary review final;
unchanged production [module](../../../../glossary.md#term-module)/root [Javadoc](../../../../glossary.md#term-javadoc) reused. No unresolved repair issue/required follow-up;
worker/reviewer closed, CPU test/production frontiers None, no successor or further probe authorized.
