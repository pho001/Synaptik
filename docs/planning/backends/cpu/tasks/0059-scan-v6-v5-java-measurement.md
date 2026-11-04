# Task CPU0059: Bounded Scan v6/v5/Java entry comparison

## Reader context

G5 and G6 are different generated artifacts; “standard” and “typed” are two ways the external
launcher calls those artifacts, not different Scan algorithms. The standard action uses a
captured segment-array spreader; the typed action supplies the ordered segments, geometry and
range directly to `MethodHandle.invokeExact`. Reusing a code target across mechanisms borrows
code, not writable state: each target/mechanism keeps independent outputs and scratch.
E remains the unchanged direct-Java comparator in both mechanisms.

Read [class bytes versus JIT code](../../../../backend-guide/cpu-backend.md#generated-class-byte-persistence-and-evidence)
before interpreting pins or compilation windows, and [current Scan composition](../../../../backend-guide/cpu-backend.md#current-general-scan-composition)
for current support. The alignment repair and every failed or successful slot below stay part
of this historical experiment rather than instructions to rerun it.

## Status

Complete — repaired [preflight](../../../../glossary.md#term-preflight) and three measured JVMs passed; independent result review APPROVE,
0 P1/P2, MAIN accepted 2026-10-28. All slots consumed; Java parity remains unresolved.
This grant is distinct from closed CPU0053/0055/0056 grants; preserve their markers/results.

## Change class

Class C conservatively: generated hot-path/[application binary interface](../../../../glossary.md#term-application-binary-interface-abi) ([ABI](../../../../glossary.md#application-binary-interface-and-native-bridge)) performance evidence.
Use a clean off-tree executor and independent prelaunch and actual-result review; no optimization.

## Goal

Compare current generated v6 (G6), retained generated v5 (G5), and unchanged optimal Java (E)
for the actual [Compiler](../../../../glossary.md#term-compiler) reverse of the original-forward symbolic-length Scan family (`false`).
Separate two entry mechanisms; this is entry-inclusive diagnostic evidence, not pure-[kernel](../../../../glossary.md#kernel) or [Engine](../../../../glossary.md#term-engine) end-to-end cost.

## [Scope](../../../../glossary.md#term-scope) and non-goals

[FLOAT32](../../../../glossary.md#term-float32), fixed T=4/F=3/H=2, invocation B=7 then B=32, lengths `row % 5`, joint final/emission [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings).
Fresh repair root: `/private/tmp/synaptik-cpu0059-repair.M0v6kS`, created with `mktemp -d`.
Preserve the entire original `/private/tmp/synaptik-cpu0059.AB8i7S` source/classes/freeze/[preflight](../../../../glossary.md#term-preflight) unchanged.
Only semantic harness delta: original Java line 152 [heap](../../../../glossary.md#term-heap) `poisonExpected` store, `JAVA_INT` → `JAVA_INT_UNALIGNED`.
Adapt paths/[provenance](../../../../glossary.md#provenance)/grant mechanically; all repaired launcher, protocol, [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) and analysis go in the fresh root.
No repository source/test/build/architecture/[oracle](../../../../glossary.md#term-test-oracle) edits, [Gradle](../../../../glossary.md#term-gradle), repository [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), [module](../../../../glossary.md#term-module)/full tests,
native build, [commit](../../../../glossary.md#term-commit)/push, optimization, other benchmark, automatic CPU0048 successor or historical rewrite.

## Contracts

- [Architecture root](../../../../../ARCHITECTURE.md#core-invariants): `Core invariants` and [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>.
- [Backend execution](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>: `CPU backend routes` and [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling).
- [Recurrent scan](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates): `Qualification, performance and migration gates`, [Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic), and [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>.

Read AGENTS, this brief, these exact headings, affected source and focused tests. Stop on ambiguity or conflict.

## Dependencies and integration

- Depends on: CPU0058 Complete, independent APPROVE (0 P1/P2), [CPU](../../../../glossary.md#term-cpu)/[Javadoc](../../../../glossary.md#term-javadoc)/[conformance](../../../../glossary.md#term-conformance)/[Engine](../../../../glossary.md#term-engine) gates passed. Reuse, do not rerun.
- Conflicts with: any source/class/[oracle](../../../../glossary.md#term-test-oracle) writer, build, test [run](../../../../glossary.md#term-run) or other benchmark; no concurrent repository writers during execution.
- Parallel group: None; worker writes only off-tree, so no isolated worktree is required.
- Common base revision: HEAD=origin `07a281edd3086efde2853211b71b48288db027e1`, accepted 2501-file dirty tree, empty index; MAIN's `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0059-repair-ready.json`. Earlier snapshots remain [provenance](../../../../glossary.md#provenance).
- Integration order: planner → MAIN preservation/[Ready](../../../../glossary.md#term-ready) verification and plan integration → clean executor repair/[compile](../../../../glossary.md#compile)/freeze then STOP → fresh independent targeted prelaunch APPROVE → MAIN fresh [hash](../../../../glossary.md#term-hash)-bound [release](../../../../glossary.md#term-release) → one added untimed [preflight](../../../../glossary.md#term-preflight) → on success only, original three unused serial measured JVMs → independent result review → MAIN closure.
- Integration [validation](../../../../glossary.md#term-validation): frozen-input and launcher checks; raw correctness, schedule/count/disposition reconciliation, reproducible offline analysis, independent result APPROVE, links/whitespace and MAIN preservation [checkpoint](../../../../glossary.md#term-checkpoint). No repeated executable suites.
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN alone owns master/roadmap, Ready verification and Result; planner updates only this brief, executor edits no repository files.

Ready verification (MAIN, 2026-10-28): CPU0058 gates accepted; protected inputs/ordered [classpath](../../../../glossary.md#term-classpath)
and base/HEAD/origin/index unchanged; no conflicting writer/build/benchmark.
User approved only the one-line repair, ONE added untimed preflight and, conditional on success,
the ORIGINAL THREE UNUSED measured slots. MAIN integrated plans; slots were unreleased at [admission](../../../../glossary.md#term-admission).
Source/authority frontiers and CPU0048–0052 stay held.
Historical release accepted revision-2 freeze `8388bfb91ab716811799c3897bd5ad79c8b880507c8f71136290eaa261db24c8`
and independent APPROVE (0 P1/P2); it is provenance only, not a repaired-harness release.

## Files and [immutable](../../../../glossary.md#term-immutable) inputs

Repository paths below are read-only; emit package is `io/github/pho001/synaptik/backend/cpu/internal/codegen/emit`.
Read `modules/backend/cpu/src/test/java/<emit-package>/CpuSymbolicLengthScanFixture.java`
(`Family`, `preboundEquivalentReverseCpu0045`, `capturedReverseCpu0044`) and
`CpuSymbolicLengthScanEquivalentJavaOracle.java`; focused test is
`CpuSymbolicScanGeneratedKernelTest.combinedTypedActualReverseMatchesSpecializedJavaAndRetainsClassFileEvidence`.
Read generator/emitter/[IR](../../../../glossary.md#term-intermediate-representation-ir) and [specialization](../../../../glossary.md#term-specialization) [identity](../../../../glossary.md#term-identity-meanings) only as needed for pin/origin/[ABI](../../../../glossary.md#term-application-binary-interface-abi) checks.

CPU0058 root is `/private/tmp/synaptik-cpu0058.Eo998v`; immutable reverse [artifacts](../../../../glossary.md#term-artifact) are:

- G5 `prechange-v5/FLOAT32-false-1.class`, [SHA-256](../../../../glossary.md#term-sha-256) `30477241d7603cfec1c2f46079739a5ac10c2b1292dba3fb58aeaa2b05884330`.
- G6 `current-evidence/classfiles/FLOAT32-false-1.class`, SHA-256 `8f2ca20dd90ea82caf37f0aa17af503cbd54d8a035972afaff7f64807c588883`.
- Keep corresponding identity/compatibility bytes and structure text; reuse CPU0058 review/receipt/structural comparison: only three address quartets and baked v5→v6 identity differ. Do not repeat the audit.
- E class SHA-256 `79896bfa961e39abf2b3a18f5e40ff708cacd36d39de05bab41ed3024716521f`; source SHA-256 `2712ae1da5a5af7136a03a3ac650a60a274a5253fb6f51fee47c6da96b896d61`.

Exact [JDK](../../../../glossary.md#term-java-development-kit-jdk): `/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home`.
Reuse `/private/tmp/synaptik-scan-measurement-verified.LLDIqd/runtime-classpath.txt` only after ordered origins/[hashes](../../../../glossary.md#term-hash) and current [runtime](../../../../glossary.md#term-runtime)-class [validation](../../../../glossary.md#term-validation).
MAIN captured `preserve-inputs.mjs` / `inputs-baseline.json` in the original CPU0059 root: **22 ordered [classpath](../../../../glossary.md#term-classpath) entries, 3913 protected inputs** (not 23).
It protects CPU0058, JDK, runtime classes, source/oracles and emitter facts. Use its check mode; never recapture or duplicate MAIN's [manifest](../../../../glossary.md#term-manifest).

## Acceptance criteria and fixed protocol

1. Freeze a versioned protocol, launcher source/compiled bytes, analysis and exact commands before prelaunch review.
   Reject environment/agent/selector injection; retain exact JDK/VM flags, classpath ordering and class origins/hashes.
   Generate and retain fresh G6 bytes and both identity streams with the same pin before timing in each attempt.
   Inspect retained G5's exact typed entry, all 19 ordered segment slots, 202-long numeric pack/layout,
   scratch <span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [classpath glossary](../../../../glossary.md#term-classpath); [identity glossary](../../../../glossary.md#term-identity-meanings); [layout glossary](../../../../glossary.md#layout))</span>`8*B` bytes and range `[0,2*B)` before diagnostic-only hidden-class loading. Stale G5 must not enter the production [cache](../../../../glossary.md#term-cache).
2. For each B use six disjoint fixtures/writable packs (three targets × two mechanisms), equal actual reads/seeds/scalars and exact forward-produced saved histories, never replayed activations.
   Standard mechanism: cached segment-array spreader, captured once outside calls, using the fixture's existing seam.
   Typed mechanism: raw <span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [scalars glossary](../../../../glossary.md#term-scalar); [activations glossary](../../../../glossary.md#term-activation))</span>`MethodHandle.invokeExact` with all 19 [`MemorySegment`](../../../../glossary.md#term-memorysegment) operands, `long[]`, `long`, `long`;
   capture every operand outside the call (no per-call array/list/fixture lookup). Borrow the same raw target per generated version across mechanisms.
   Per-version static-final raw handles may expose stable code targets to JIT; operands, scalar values and geometry stay invocation data.
   E is <span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [lookup glossary](../../../../glossary.md#term-lookup); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit))</span>`fixture.preboundEquivalentReverseCpu0045()` direct static action in **both** mechanisms, unchanged.
   No `insertArguments` candidate, slow E, per-call [binding](../../../../glossary.md#term-binding) experiment or timing subtraction.
3. ONE added untimed [preflight](../../../../glossary.md#term-preflight): raw checks only, **zero clocked [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) and zero [warmup](../../../../glossary.md#term-warmup) rounds**.
   Total budget: one preserved failed preflight + one new untimed preflight + three original unused measured opportunities.
   Only on preflight success, exactly three fresh serial measured JVMs; all targets/mechanisms, B7 before B32.
   For each B: 100 warmup + 300 measured rounds, 100 calls/action/round, all six actions every round.
   For round r=0..399, mechanisms are standard→typed when r is even, typed→standard when odd;
   within each mechanism target position p=0..2 selects `[G5,G6,E][(r+p)%3]`. Same schedule in each repeat.
   This is 480000 scheduled action calls/JVM including warmup, 1.44M across three; untimed correctness calls are separately enumerated.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm))</span>
4. Each JVM attempt, including preflight, has a 120-second wall deadline including cold setup and persistence.
   Check admission outside each 100-call clocked batch; no per-call clock, allocation or added guard in that loop.
   Mark attempts exclusively before launch; no replacement, retry, adaptive warmup or extra process to gather evidence.
   Preserve every failure/censored/partial attempt and exit disposition; added preflight failure means STOP, no measured launch.
   A failed measured slot remains consumed; proceed only if correctness/input integrity remains valid, otherwise stop and report unused slots.
   No extra benchmark/runs are authorized; do not kill unrelated idle Gradle or other processes.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [admission glossary](../../../../glossary.md#term-admission); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [guard glossary](../../../../glossary.md#term-guard); [gather glossary](../../../../glossary.md#gather); [censored glossary](../../../../glossary.md#term-censored); [integrity glossary](../../../../glossary.md#term-integrity); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
5. Before/after calls, outside clocks, compare raw bytes for all five cotangents, forward saved-site histories and unchanged inputs;
   explicitly check inactive exact positive zero, empty-row raw final-seed identity, and poison padding without contamination.
   Retain/check the latest produced outputs <span data-glossary-navigation="true"> ([positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [seed glossary](../../../../glossary.md#term-seed); [padding glossary](../../../../glossary.md#padding))</span>**before** any rerun [validation](../../../../glossary.md#term-validation) can overwrite them; never substitute [fixture](../../../../glossary.md#term-fixture) `verify()` for this observation.
   Observe each sample outside its clocked loop; keep diagnostic poisoning/check calls separate from the fixed timed schedule.
6. Record cold compile/analyze, generation, definition and binding separately, never subtract from action timings.
   Retain fresh generated/hidden names, pins, source/class/launcher hashes, exact commands and all samples, warmup included:
   repeat, phase, mechanism, B, round, mechanism/target positions, target, calls, start/end nanoseconds and JVM uptime.
   Retain per-attempt console/exit/timeout/disposition and full JIT (just-in-time compiler) XML, including incomplete logs on failure.
   Match normal compilation completion times to samples before claiming C2 readiness; missing assembly is not native causal proof.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [compiler glossary](../../../../glossary.md#term-compiler); [logs glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [C2 glossary](../../../../glossary.md#term-c2))</span>
7. Offline analysis reports per-repeat/B/mechanism latencies, paired G6/G5, G6/E and G5/E ratios and variability.
   Keep every observation; no silent outlier/failed-slot censorship, historical pooling or universal parity claim.
   Mechanism comparisons are diagnostic only; distinguish incomplete/compilation-confounded results from established steady-state evidence.<span data-glossary-navigation="true"> ([outlier glossary](../../../../glossary.md#term-outlier); [pooling glossary](../../../../glossary.md#term-pooling))</span>

## [Validation](../../../../glossary.md#term-validation) and review gates

Preparation compiles only `Cpu0059ScanMeasurement.java` off-tree with the exact [JDK](../../../../glossary.md#term-java-development-kit-jdk)'s `javac`,
`--add-modules=jdk.incubator.vector -proc:none`, unchanged reused [classpath](../../../../glossary.md#term-classpath) and output under the fresh repair root.
Adapt a new `run.sh` with modes [`compile`](../../../../glossary.md#compile), `preflight preflight`, `measure measured-1` through `measured-3`;
Freeze expanded commands, compiled bytes, original-to-repaired diff/[provenance](../../../../glossary.md#provenance) and attempt handling, then STOP.
Fresh independent targeted prelaunch review checks the sole semantic delta, mechanical adaptations,
freeze/compiled bytes, unchanged pins/[ABI](../../../../glossary.md#term-application-binary-interface-abi)/call shape/schedule, zero-clock [preflight](../../../../glossary.md#term-preflight), limits and logging.
MAIN requires fresh preservation/conflict checks and a [release](../../../../glossary.md#term-release) bound to the new freeze [hash](../../../../glossary.md#term-hash) before any [run](../../../../glossary.md#term-run) mode.
The release covers only the added preflight and, conditional on success, the original unused three slots; no retry/additional measurement.
Use `node /private/tmp/synaptik-cpu0059.AB8i7S/preserve-inputs.mjs check` before/after execution and analysis.
Independent result review reuses frozen source/test evidence, reconciles raw [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) and [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) timelines, and checks reproducible offline analysis.
MAIN performs final preservation/shared-document integration; check local links/anchors, new-file whitespace and `git diff --check`.

## Documentation and review impact

[Planning](../../../../glossary.md#term-planning) profile plus general style apply. Only this brief is a planner repository write.
No [API](../../../../glossary.md#term-application-programming-interface-api)/semantics/authority/[Javadoc](../../../../glossary.md#term-javadoc)/glossary change; [fixture](../../../../glossary.md#term-fixture)/[oracle](../../../../glossary.md#term-test-oracle) contracts stay accurate.
Independent Class C prelaunch/result approvals are recorded below.

## Result

Original root retains failed [preflight](../../../../glossary.md#term-preflight) PID 35566: exit 1 at [heap](../../../../glossary.md#term-heap) [alignment](../../../../glossary.md#term-alignment), 429.085 ms, zero [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters);
its source, revision-2 freeze, [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) and 175 evidence files remain unchanged. Failure review is retained.
Fresh-root javac passed; source delta is only the heap raw-int alignment and mechanical BASE path.
Five nested classes are byte-identical; independent prelaunch APPROVE (0 P1/P2) and MAIN [release](../../../../glossary.md#term-release) bind
freeze `e8f3dc35664a3178d93b22a4d20424193ff0798380029a645e26f69eb3c9ea0f`.
Runs via `/bin/bash`: `<repair-root>/run.sh preflight preflight` and `run.sh measure measured-N`, N=1,2,3.
New preflight PID 44344 passed in 476.270 ms, zero samples/[warmup](../../../../glossary.md#term-warmup); measured PIDs 44366/44374/44381
passed in 3069.751/2967.300/3068.830 ms. No retry, [censoring](../../../../glossary.md#benchmark-measurement-vocabulary), partial/unchecked row or additional [JVM](../../../../glossary.md#term-java-virtual-machine-jvm).
Each measured JVM has 4800 rows/480000 scheduled calls including warmup: total 14400/1440000;
each successful JVM separately checks 36 reverse/24 forward calls. Raw checks and [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) PID/XML pass.
Frozen `analysis-v1.mjs` ran offline; `analysis-result-v1.json` [SHA-256](../../../../glossary.md#term-sha-256)
`710b96ab6ecdbf8faf4d06d64a9391ce6cb62a21b055a8eeaaae4158140a53a3`; statistics/commands in `execution-receipt-v1.json`.
Per-repeat paired medians (R1/R2/R3), not pooled: B7 standard G6/G5=1.001/1.006/0.998,
G6/E=1.177/1.171/1.165; B7 typed 1.009/0.997/0.998 and 1.178/1.160/1.156.
B32 standard G6/G5=0.993/0.999/1.000, G6/E=1.210/1.207/1.209;
B32 typed 0.997/0.994/1.000 and 1.212/1.202/1.207. No consistent G6 speedup is demonstrated.
Normal [kernel](../../../../glossary.md#kernel) [C2](../../../../glossary.md#term-c2) precedes measured B7; [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-loop C2 arrives during measured B32. Full-path [steady
state](../../../../glossary.md#term-steady-state)/native cause are unproved; these local entry-inclusive results do not establish [Engine](../../../../glossary.md#term-engine) or universal parity.
MAIN postgroup preservation passed: 3913 + 4136 + 2636 inputs and all 175 original files unchanged;
repository base/HEAD/origin/index unchanged before [planning](../../../../glossary.md#term-planning) closure. Only three planning files change;
no repository executable, [API](../../../../glossary.md#term-application-programming-interface-api), [Javadoc](../../../../glossary.md#term-javadoc), glossary, [ADR](../../../../glossary.md#term-architecture-decision-record-adr) or architecture change. `review-results.md/json`
APPROVE (0 P1/P2) reproduces analysis byte-exact and approves planning [metrics](../../../../glossary.md#machine-learning-and-hyperparameters)/caveats. MAIN accepted.

Status: Complete
