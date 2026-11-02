# Task CPU0058: Symbolic-length Scan zero scratch-base omission

## Status

Complete — MAIN accepted the independently reviewed repair and named checkpoint on 2026-10-28.

## Change class

Class C — generated hot-path address proof and combined baked-artifact compatibility change.
Use a clean implementation context and an independent clean targeted code/documentation review.

## Goal

Extend existing `zeroBases` access proof to combined symbolic-length Scan scratch `Cut` intermediates
whose full offset is zero for every B; preserve inclusive lifetimes, arithmetic and resources.

## Scope

- In `Shaped.emitLengthAware`, recognize only combined, scratch-backed, nonbanked cuts:
  `ir.symbolic() && ir.lengthAware()`, `cut.port() < 0`, `cut.carry() < 0`,
  `cut.offset() == 0` and `cut.offsetPerBatch() == 0`. Current `batchCuts` uses exactly
  `offset + offsetPerBatch * B`; confirm this remains the complete recipe before editing.
- Prove the carrier/base stays valid through every use in the inclusive cut lifetime,
  including derived views, aliases and branch/iteration transitions. Register only the exact
  origin whose zero base is proved; retain cell-scoped relative-offset caching.
- Advance only the combined baked payload to `|scan-symbolic-length-v6=` through
  `CpuGeneralScanIr.bakedSpecializationIdentity()`, shared by both specialization streams.
  Add focused semantic, structural and compatibility regressions and narrow current docs.

## Non-goals

No sampled/runtime-B proof, runtime branch, per-B specialization, changed allocation/lifetime/bank,
cursor gating, constant baking, copy cleanup, type/API/protocol, broader capability or arithmetic/resource
change. No unrelated refactor, measurement, benchmark, commit or push.
CPU0048–0052 and all exhausted/cancelled measurement grants remain held.

## Contracts

- [Architecture root](../../../../../ARCHITECTURE.md#core-invariants): **Core invariants** and
  [**Scope-indexed normative contracts**](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
- [Backend execution](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes):
  **CPU backend routes** and
  [**Performance evidence and optimization tooling**](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling).
- [Recurrent Scan](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates):
  **Qualification, performance and migration gates**;
  [**Actual first-order closure and arithmetic**](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic);
  [**Activity, raw identity and retained histories**](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories);
  [**Runtime preflight and implementation gates**](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates).

This bounded cleanup is not renewed performance qualification; historical measurements remain
limited baseline evidence, never proof of speed, parity or native execution cost.

## Dependencies and integration

- Depends on: CPU0057 independently reviewed Complete; MAIN verified acceptance. User approved
  its preferred zero-base repair, not the cursor alternative or new measurement.
- Conflicts with: any writer to the emitter/combined identity/proof or named current docs;
  no competing writer reported. Serialize implementation, documentation and final review.
- Parallel group: None.
- Common base revision: accepted `HEAD = origin = 07a281edd3086efde2853211b71b48288db027e1`
  plus accepted dirty tree; index empty. MAIN's `cpu0057-accepted` audit is clean;
  `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0058-base.json` records 2499 files.
  Preserve approximately 412 existing dirty entries; do not reset/stash/clean unrelated work.
- Integration order: MAIN verifies frontier/dependencies and marks Ready; clean worker implements;
  independent targeted code/docs review; one final CPU/Javadoc gate after final edits;
  MAIN integrates Result/master/roadmap and runs the named checkpoint. Review reuses success.
- Integration validation: MAIN runs `:testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest'`
  and `:testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'` once,
  with the evidence redirection and performance-disabled environment below; no repository-wide rerun.
- Shared-document integration owner: MAIN owns this brief's Result, master and roadmap.
  Under serialized handoff only, worker owns narrow current-proof/v6 factual updates to root,
  backend contract and CPU guide listed below; reviewer finalizes them. No other shared edits.

Launch verification (MAIN, 2026-10-28): CPU0057 is independently reviewed Complete with its
offline validation accepted. CPU0058 is the sole authorized source/test frontier; measurement,
offline-observation and separate authority frontiers stay None. Conflicts are inactive and
writers are serialized. The accepted 2499-file base is unchanged except this brief and
synchronized master/roadmap; HEAD/origin/index match. Historical checks passed 4136 files,
and the additional 2636-file snapshot is captured. The user approved only this bounded repair;
no benchmark or Git publication follows. Worker may now launch from this verified packet.

## Files and symbols

Paths below are relative to `modules/backend/cpu/src/` unless stated otherwise.

- `main/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuGeneralScanEmitter.java`:
  `Shaped.emitLengthAware` cut setup (~554–558), `bindStableLengthCarry` (~1117),
  `load` (~1427), `store` (~1462); inspect `batchCuts`, `zeroBases` and lifetime/cache scopes.
- `main/java/io/github/pho001/synaptik/backend/cpu/internal/ir/CpuGeneralScanIr.java`:
  `Cut`, `bakedSpecializationIdentity()` (exact current line 585) and affected Javadoc.
- Existing tests under `test/java/io/github/pho001/synaptik/backend/cpu/internal/`:
  `codegen/emit/CpuSymbolicScanGeneratedKernelTest.java`, `CpuGeneralScanGeneratedKernelTest.java`;
  `executable/CpuGeneralScanExecutionTest.java`; `prepare/CpuGeneralScanPreparationTest.java`;
  `cache/CpuKernelSpecializationTest.java`, `CpuGeneratedKernelArtifactStoreTest.java`.
  Extend these tests as needed; no new production type.
- Read only [CPU0057 preferred proof recommendation](../0057-scan-body-java-audit.md#one-preferred-bounded-next-step),
  not predecessor histories. Retain optimal specialized Java oracle and existing fixtures unchanged.
- Narrow current documentation writes: `ARCHITECTURE.md` **Core invariants**,
  `docs/architecture/contracts/backend-execution.md` **CPU backend routes** and
  `docs/backend-guide/cpu-backend.md`. No recurrent-owner edit: its v1 namespace is historical
  initial acceptance, not a stale current-version claim.

## Acceptance criteria

1. Generation proves the full recipe zero for every admitted B, not `cut.offset()==0` alone,
   a resolved runtime fact or one sample. Nonzero constant, zero-constant/nonzero-B coefficient,
   bank-backed and port/time-backed cuts do not acquire this scratch proof. Existing lifetime,
   bank selection and alias guards remain intact; no temporal offset cache escapes its cell.
2. Current F32 T=4/F=3/H=2 actual reverse witness at B=7 and B=32 loses the two static
   base-load/add sequences at work store and db load. Historical v5 has local 108 assigned
   zero, store BCIs 823/825 and load 1785/1788; these identify the witness, not brittle new BCI
   expectations. The corresponding 44/244 dynamic additions disappear by structural accounting,
   not timing. Assert load/store dataflow and checked access, not merely total opcode counts.
3. F32/F64 actual Compiler forward/reverse, both traversals, B=0/positive (including 7/32),
   T=0 and zero/mixed/full lengths retain outputs, raw inactive carry/+0, exact saved sites,
   poisoned-padding skipping, invalid-last-length fail-before-write, reuse and resources.
   Existing independent active-prefix/optimal-Java oracles remain unchanged.
4. Both specialization streams advance only combined baked v5→v6, reject stale combined
   v1–v5 artifacts and reuse valid v6 artifacts. Canonical `:symbolic-length-v1`, numeric pack
   `typed-explicit-b-v1`, schema/family 71, all noncombined identities and generated bytes
   remain exact. Verify existing noncombined byte/identity pins, including all-valid symbolic,
   static-length, static shaped/MATMUL, pointwise/fixed and ordinary paths.
5. Search current `v5` assertions and advance only non-timing tests as needed. In particular,
   `CpuGeneralScanTimingTest` v5 assertions around 1856 belong to the frozen opt-in CPU0045
   campaign: do not rewrite or run it. Do not update historical fixture/oracle/pins/artifacts,
   old reports, protocols or measurements to make tests pass.
6. No added helper call, allocation, boxing, reflection, dispatch, arithmetic or memory traffic
   in the hot loop; compare generated Class-File algorithm/dataflow to retained optimal Java.
   Independent code/docs review approves the exact final diff and preservation evidence.

## Validation

Worker inspects writable evidence destinations before testing. Every test invocation MUST redirect
`-Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0058.Eo998v/current-evidence`.
`modules/backend/cpu/build.gradle.kts` already forwards this property to test JVMs. This protects
the historical Prepare0015 default used by
`combinedTypedActualReverseMatchesSpecializedJavaAndRetainsClassFileEvidence` and
`CpuOrdinaryDynamicBatchTest`; inspect/override any other writable historical destination before
running, without editing frozen tests/build config or broadening the external filter.

Use JDK26 and installed Gradle; clear inherited opt-in performance/generalScan selectors,
`JAVA_TOOL_OPTIONS`, `JDK_JAVA_OPTIONS` and Gradle/JVM option injection before invocation.
All opt-in performance/diagnostic/C2 campaigns must remain disabled (also unset
`SYNAPTIK_CPU_LOSS_PERFORMANCE`). The evidenceRoot property is the sole intentional
generalScan evidence-output option, not a measurement grant. Commands use shell array expansion:

```bash
export JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home
cpu0058_gradle=/Users/phujka/.gradle/wrapper/dists/gradle-9.6.1-bin/4ticwg1pgcbps2hj28r8so764/gradle-9.6.1/bin/gradle
cpu0058_options=(--no-daemon --console=plain -I /private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0047-module-filter.gradle
  -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0058.Eo998v/current-evidence
  -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false)
"$cpu0058_gradle" "${cpu0058_options[@]}" :backends:cpu:test \
  --tests '*CpuSymbolicScanGeneratedKernelTest' --tests '*CpuGeneralScanGeneratedKernelTest' \
  --tests '*CpuGeneralScanExecutionTest' --tests '*CpuGeneralScanPreparationTest' \
  --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest'
# ONE final affected-module run after final executable/Javadoc edits:
"$cpu0058_gradle" "${cpu0058_options[@]}" :backends:cpu:test :backends:cpu:javadoc
node /private/tmp/synaptik-cpu0058.Eo998v/preserve.mjs check
git diff --check
```

The unchanged filter excludes only the two historical unconditional timing campaigns; record exact
commands/counts/skips and structural findings. Preserve both baselines: the existing 4136-file check
and new 2636-file `preserve.mjs` snapshot (CPU0056/0057/Prepare0015). Run both checks after tests,
never recapture either. MAIN uses the same options; stop if output safety cannot be established.

## Documentation and review impact

Use general/planning profiles for this brief and architecture/backend-guide/Javadoc profiles for edits.
Update affected Java contract Javadoc for the stronger scratch proof and combined v6 identity;
record reviewed unchanged Javadoc explicitly. Root/backend-owner/CPU-guide updates are factual
current v6/proof maintenance, not a new architectural decision; no ADR is needed unless an
actual conflict appears, in which case stop. Search glossary terms narrowly; no terminology
change is intended. Validate links/anchors and final diff. Independent review reuses successful
tests unless code or a concrete risk requires rerun; Javadoc generation follows final Javadoc edits.

## Result

MAIN accepted independent APPROVE (0 open P1/P2) and the named checkpoint. Nine reviewed files
cover three production files, three tests and three docs; MAIN updated the three planning records.
The complete zero recipe and nonbanked scratch origin reuse existing proof-aware access. All four
typed/traversal reverse streams lose three static address sequences, including the alternative B1
raw read; F32 witness entry 2147→2126 bytes, 952→940 instructions. At B7/B32 the two active paths
remove 44/244 zero additions by structural accounting. Oracle bytes and other dataflow are unchanged.
Both streams advance only combined baked v5→v6; stale v1–v5 reject, current cache reuse passes,
and 20 noncombined snapshot occurrences retain identities/bytes. Focused six-filter gate: 185 PASS.
Single final CPU/Javadoc command above with additional output-only `-I evidence-redirect.gradle`:
1305 PASS, 51 skips, 0 failures/errors; the unchanged filter excludes two historical timing cases.
Seven unrelated tests relocate path constants only; assertion checks and numerical code stay intact.
Javadoc passed with 94 existing warnings; post-review Javadoc-only `:backends:cpu:javadoc` passed
after correcting two current identity descriptions. No executable tests were repeated for that fix.
Named MAIN conformance/Engine checkpoint: 12+22 PASS, no skips/errors. No new performance run.
Docs/Javadoc/glossary reviewed; no term/ADR; links/anchors/whitespace passed. 4136+2636 historical
inputs and unrelated work preserved; HEAD/origin/index unchanged, no commit/push.
Exact commands, snapshots, redirected-output audit, structural results and independent review:
`/private/tmp/synaptik-cpu0058.Eo998v/{receipt.md,review-final.md,main-integration.log}`.
No speed/native-cost/parity claim or implementation follow-up; the timing-gap cause remains open.

Status: Complete
