# Task CPU0056: [Scan](../../../../glossary.md#term-cumulative-scan) invocation [binding](../../../../glossary.md#term-binding)

## Status

Complete — reviewed rejected optimization; CPU0056-only candidate withdrawn exactly on 2026-10-28.
User approved the narrow implementation and short A/B comparison; no [commit](../../../../glossary.md#term-commit) or push.

## Change class

Class C — changes the cold-[bound invocation](../../../../glossary.md#bound-invocation--boundinvocation)/hot-entry boundary and [run](../../../../glossary.md#term-run)-owned [bindings](../../../../glossary.md#term-binding).
Use a clean implementation context and an independent clean targeted code/documentation review.

## Goal and [scope](../../../../glossary.md#term-scope)

Remove [Scan](../../../../glossary.md#term-cumulative-scan)'s execute-time segment-array/geometry [argument](../../../../glossary.md#term-argument) passing without changing generated code.
Retain the raw typed `CpuGeneratedKernel.entryPoint()` in `CpuGeneralScanPreparedExecutable`.
On each cold bind, use `MethodHandles.insertArguments` to bind every [`MemorySegment`](../../../../glossary.md#term-memorysegment) argument,
including optional scratch, then bind `(Object) long[]` geometry as one argument, not varargs.
The existing `Invocation` owns a final `(long,long)void` [target](../../../../glossary.md#term-target) and calls it exactly once with
the existing range. Replace the prepared `asSpreader` and execute-time array-plus-pack passing.
Ordinary `CpuPreparedExecutable.callFor` already demonstrates this mechanism; inspect, do not edit it.

## Non-goals

No new [Runtime](../../../../glossary.md#term-runtime) [API](../../../../glossary.md#term-application-programming-interface-api), production type or helper; no emitter, [fixture](../../../../glossary.md#term-fixture), [oracle](../../../../glossary.md#term-test-oracle), [schema](../../../../glossary.md#term-schema), build,
architecture, numerical, resource, route or geometry-recipe changes. No [Engine](../../../../glossary.md#term-engine) measurement,
benchmark apparatus, freeze/[release](../../../../glossary.md#term-release)/nonce layers, native builds, broad campaign or successor work.
Keep all old evidence, grants and exhausted budgets untouched; CPU0053 remains Cancelled.

## Contracts

- [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([normative glossary](../../../../glossary.md#term-normative))</span>: [immutable](../../../../glossary.md#term-immutable) preparations and isolated [run state](../../../../glossary.md#run-state--runstate).
- [Backend](../../../../glossary.md#backend) execution: [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes) and [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling): existing [CPU](../../../../glossary.md#term-cpu) [ownership](../../../../glossary.md#term-ownership-meanings) and observational evidence only.
- [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)/[Engine](../../../../glossary.md#term-engine): [`modules/runtime`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime) and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>: cold typed [binding](../../../../glossary.md#term-binding), one [RunState](../../../../glossary.md#run-state--runstate), [producer](../../../../glossary.md#term-producer) ordering and [resource lifetime](../../../../glossary.md#ownership-borrowing-and-lease).
- Recurrent [Scan](../../../../glossary.md#term-cumulative-scan): [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates)<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span> and [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates)<span data-glossary-navigation="true"> ([Qualification glossary](../../../../glossary.md#term-qualification))</span>: preserve guards/support and separate cold/entry/[kernel](../../../../glossary.md#kernel) evidence.

Read these exact headings, affected source and focused tests; do not read predecessor histories.
Stop for MAIN clarification if a contract is missing/ambiguous or this [scope](../../../../glossary.md#term-scope) requires expansion.

## Dependencies and integration

- Depends on: CPU0055 and Prepare0015, Complete; MAIN supplied prior actual-result independent APPROVE and stable accepted source verification.
- Conflicts with: [Scan](../../../../glossary.md#term-cumulative-scan) executable, geometry and generated-[artifact](../../../../glossary.md#term-artifact) writers; reported inactive, recheck before launch.
- Parallel group: None; serial implementation and review, no concurrent writer.
- Common base revision: HEAD=origin/main `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty work captured in `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0056-base.json` (2496 files); not merged predecessors.
- Integration order: brief → source + focused tests → independent review → [module](../../../../glossary.md#term-module) + named join → bounded performance diagnostic → Result/MAIN acceptance.
- Integration [validation](../../../../glossary.md#term-validation): worker [CPU](../../../../glossary.md#term-cpu) test/[Javadoc](../../../../glossary.md#term-javadoc) once; MAIN named `CpuGeneralScanConformanceTest` and `EngineCpuGeneralScanIntegrationTest` [checkpoint](../../../../glossary.md#term-checkpoint) below.
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; worker owns only the exact guide paragraph delegated below.

Launch check (MAIN, 2026-10-28): CPU0056 is [Ready](../../../../glossary.md#term-ready) at the synchronized sole CPU source/test and
observational frontier. CPU0055's final independent result review and accepted Prepare0015 satisfy
dependencies. Contracts are stable; HEAD/origin match the base, index is empty, and the retained
[manifest](../../../../glossary.md#term-manifest) differs only by this brief and authorized master/roadmap updates. Conflicts are inactive.
One clean implementation context exclusively owns source/tests/guide; MAIN performs no concurrent
repository writes. Separate diagnostic preparation may write only its off-tree evidence directory.

## Files and symbols

- `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuGeneralScanPreparedExecutable.java`: constructor entry retention, `bindCompatible`, existing `Invocation`, affected implementation [Javadoc](../../../../glossary.md#term-javadoc).
- Focused tests under `modules/backend/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/`: `executable/CpuGeneralScanExecutionTest`, `prepare/CpuGeneralScanPreparationTest`, `codegen/emit/CpuGeneralScanGeneratedKernelTest`, `codegen/emit/CpuSymbolicScanGeneratedKernelTest`.
- `docs/backend-guide/cpu-backend.md:4866`: only the paragraph currently describing the retained typed [MethodHandle](../../../../glossary.md#native-numerical-library-and-instruction-architecture) spreader.
- New off-tree diagnostic source/results only: `/private/tmp/synaptik-scan-binding.D2bvwa`; reuse current fixtures/[oracle](../../../../glossary.md#term-test-oracle) read-only.

## Acceptance criteria

- No prepared spreader; every cold [binding](../../../../glossary.md#term-binding) creates its own final two-long [target](../../../../glossary.md#term-target) with all segments/scratch and geometry captured. No [mutable](../../../../glossary.md#term-mutable) [run](../../../../glossary.md#term-run) binding moves into prepared state.
- Preserve typed access, [dtype](../../../../glossary.md#data-type--datatype), [alignment](../../../../glossary.md#term-alignment), full-capacity overlap including tails, read/read [aliases](../../../../glossary.md#aliasing-and-arena), [liveness](../../../../glossary.md#term-liveness), [lease](../../../../glossary.md#term-lease) cleanup/retry and [producer](../../../../glossary.md#term-producer)-ordered all-B length checks before any [Scan](../../../../glossary.md#term-cumulative-scan)-local write, including invalid-last/T=0/B=0 cases.
- Preserve current fixed, symbolic and length-mode [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) support exactly, including actual reverse; do not infer all-valid symbolic F64 support. Generated bytes, both [identity](../../../../glossary.md#term-identity-meanings) streams and [cache](../../../../glossary.md#term-cache) identities remain exact.
- Repeated/fresh-input/concurrent runs retain one [RunState](../../../../glossary.md#run-state--runstate) per logical run and isolated arrays, storage and bound targets; scratch-present/absent and empty domains retain existing semantics.
- Focused regressions verify invocation [capture](../../../../glossary.md#term-capture-meanings)/isolation and preserved guards; retain generated-byte/identity assertions without changing their fixtures or expected identities.
- Report hot entry and repeated cold-bind-plus-entry separately. If cold cost outweighs hot gain or no benefit is demonstrated, withdraw only CPU0056's exact candidate changes (including associated candidate-only docs/tests), preserve accepted dirty work and record the rejected optimization honestly. Do not broaden emitter work.

## [Validation](../../../../glossary.md#term-validation)

Use [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26 (`/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home`).
For every [Gradle](../../../../glossary.md#term-gradle) [run](../../../../glossary.md#term-run) set all opt-in performance properties false, including
`synaptik.cpu.generalScan.performance` and `synaptik.cpu.dynamicBatch.performance`.
Worker focused selection: the four named focused classes above (`--tests '*<ClassName>'` each).
Final owning command after executable/review corrections and final [Javadoc](../../../../glossary.md#term-javadoc):

```bash
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home ./gradlew -I /private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0047-module-filter.gradle -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.dynamicBatch.performance=false :backends:cpu:test :backends:cpu:javadoc
```

The existing external filter excludes ONLY the two known unconditional historical timing methods
of `CpuPartitionDagGeneratedEvidenceTest`; no other exclusion, filter or build edit is authorized.
MAIN [checkpoint](../../../../glossary.md#term-checkpoint): `./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'`
with the same JDK/performance-off environment. No root suite or native rebuild for this bounded task.
Reviewer reuses successful source-matched executable gates; rerun only for changed behavior or concrete risk.

## Bounded diagnostic

One untimed correctness [preflight](../../../../glossary.md#term-preflight), then at most three fresh serial JVMs total, with no retry budget.
Compare the identical current generated hidden method through old cached spreader versus per-[run](../../../../glossary.md#term-run)
bound [target](../../../../glossary.md#term-target), alongside unchanged direct equivalent Java E. Use B=7/B=32, T=4, F=3, H=2;
100 [warmup](../../../../glossary.md#term-warmup) + 300 measured rounds, 100 calls per target per round, cyclic target order.
Also measure repeated cold-bind-plus-entry because [Runtime](../../../../glossary.md#term-runtime) binds every run; do not report only
amortized hot-call gains. Retain raw correctness checks, [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), exact commands/[JDK](../../../../glossary.md#term-java-development-kit-jdk) and limits.
This is a short entry/[binding](../../../../glossary.md#term-binding) diagnostic, not [Engine](../../../../glossary.md#term-engine) timing, parity proof or a production selector.

## Documentation and review impact

Update affected implementation [Javadoc](../../../../glossary.md#term-javadoc) and only the spreader paragraph, using documentation rules,
general/[backend](../../../../glossary.md#backend)-guide/Javadoc profiles and targeted glossary search; no new terminology is expected.
Independent clean targeted code/docs review is mandatory for Class C and must inspect the actual
diff, [capture](../../../../glossary.md#term-capture-meanings)/[guard](../../../../glossary.md#term-guard)/isolation coverage, byte/[identity](../../../../glossary.md#term-identity-meanings) preservation and final diagnostic conclusions.
Architecture impact: none; stop instead of changing authority or introducing a new boundary.

## Result

MAIN accepted the rejected optimization on 2026-10-28 after independent final APPROVE, no open P1/P2.
The candidate changed only the executable, a production-path [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) [capture](../../../../glossary.md#term-capture-meanings)/isolation test and the
guide paragraph. All three were withdrawn with `apply_patch`, byte-identical to the verified base;
no production, test, [Javadoc](../../../../glossary.md#term-javadoc) or guide change remains. Only this brief, [CPU](../../../../glossary.md#term-cpu) master and roadmap changed.
Candidate [validation](../../../../glossary.md#term-validation): named four-class gate 154 passed/1 skipped; owning CPU command above 1,303
passed/52 standard skips, no failures, with only the two authorized historical timing exclusions.
CPU Javadoc passed (94 existing warnings); [conformance](../../../../glossary.md#term-conformance) 12/12 and [Engine](../../../../glossary.md#term-engine) integration 22/22 passed.
The first sandboxed join failed during [Gradle](../../../../glossary.md#term-gradle) native-service initialization before tests; the same
command outside the sandbox passed. After withdrawal, the four-class gate passed 152/1 skipped;
full-[module](../../../../glossary.md#term-module)/Javadoc baseline evidence was reused by exact source equivalence, not rerun.
One untimed [preflight](../../../../glossary.md#term-preflight) and three serial JVMs completed: 100 warm/300 measured rounds, 100 actions
per [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), all 16,800 raw rows retained and schedule-checked. Same computational ClassFile and
both [identities](../../../../glossary.md#term-identity-meanings), unchanged Java E, raw-equal results and unmodified reads passed independently.
Hot paired N/S medians: B7 0.992–1.001, B32 0.991–0.996. Cold capture plus one entry: B7
2.163–2.678, B32 1.295–1.337; [binding](../../../../glossary.md#term-binding)-only N medians 848–1,182 ns. Per-[run](../../../../glossary.md#term-run) binding loses much
more than hot calls gain, so the original prepared spreader remains. These are mechanism costs,
not Engine timings; they neither diagnose the remaining Java gap nor establish performance parity.
Evidence: `/private/tmp/synaptik-scan-binding.D2bvwa/{summary.json,review-final.txt,integration.log}`;
worker commands/[logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) and exact candidate/withdrawal snapshots: `/private/tmp/cpu0056-worker.IGQTO8/`.
Documentation/Javadoc/glossary reviewed; restored descriptions remain accurate, no new terminology.
All 4,136 protected historical/source inputs and HEAD/origin/index are unchanged; no [commit](../../../../glossary.md#term-commit)/push,
extra diagnostic [JVM](../../../../glossary.md#term-java-virtual-machine-jvm), broader optimization or successor was launched. Remaining Java gap stays open.

Status: Complete
