# Task CPU0061: Scan feature-contraction affine addresses

## Status

Cancelled — the sole candidate failed its frozen retention gate on 2026-10-28 and was restored exactly to accepted dirty v7. The user-requested performance-only experiment is complete; no retained implementation, functional suites, retry or successor. All CPU frontiers are None; older budgets stay closed.

## Change class

Class C — generated hot-loop/dataflow and baked identity; clean implementation/independent review.

## Goal

ONE combined symbolic-B/runtime-length candidate: prove eligibility → replace feature-K cursors with int-K affine addresses → confirm Scan/optimal-Java benefit → functional suites/retention.

## Scope and non-goals

- ROW(0) dH/dX feature contractions, FLOAT32/FLOAT64 combined only. No hardcoded F3/H2/B/T, sampled proofs, new type/infrastructure/audit layer or route/resource/lifetime change.
- Preserve CPU0060 direct ROW(1) @ ROW(0) contractions, ineligible cursor code and exact noncombined identities/ClassFiles. No other module implementation, native build, GSD, commit/push, unrelated edit or automatic successor/campaign.
- Planner writes only this brief; MAIN owns master/roadmap/frontier/shared integration.

## Contracts

- [Root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts): `Core invariants`, `Module-ownership routing` — generation ownership, immutable preparations, run state/identity.
- [Backend execution](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes): `CPU backend routes`, `Accepted ordinary symbolic-batch CPU contract`, `Performance evidence and optimization tooling` — selected bindings, optimal Java oracle, checked access/diagnostic evidence.
- [Scan owner](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic): `Actual first-order closure and arithmetic`, `Activity, raw identity and retained histories`, `First-order length-mode reverse`, `Qualification, performance and migration gates`, `Qualification and mechanism gates`, `Runtime preflight and implementation gates` — guarded actual reverse, represented arithmetic/resources.
- Stop/report missing proof/conflict. Cursor wording describes v7; update retained v8 status, not arithmetic/architecture.

## Dependencies and integration

- Depends on: CPU0060; Prepare0015 Complete; MAIN verifies Result/review/gates before Ready.
- Conflicts with: CPU generation/identity/test/benchmark/shared-contract writers, builds and probes; serialize.
- Parallel group: None.
- Common base revision: HEAD=origin/main `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty source; empty index; request manifest `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0061-request.json` (2502 files).
- Integration order: MAIN admission → clean candidate/compile → independent candidate/compiled-launcher approval → MAIN hash release → sole benchmark/result review → positive-only tests/docs/final review → MAIN checkpoint/acceptance.
- Integration validation: positive-only CPU0061 General Scan conformance + Engine integration checkpoint (baseline 12+22), not root; command below.
- Shared-document integration owner: MAIN; final-docs delegation allowed. MAIN verified dependencies/base/index/preservation, no competing Java/build/probe; 2286 CPU0060 files protected by workroot `preserve-previous.mjs`. Recheck release/closure.

## Files and symbols

Paths below use `modules/backend/cpu/src/{main,test}/java/io/github/pho001/synaptik/backend/cpu/`; no new production type/package.

- Production: `main/.../internal/codegen/emit/CpuGeneralScanMatmulEmitter.java` (`emit`, relevant Javadoc); `main/.../internal/ir/CpuGeneralScanIr.java` (`bakedSpecializationIdentity`, Javadoc); `main/.../internal/cache/CpuKernelSpecialization.java` (both identity-stream Javadocs only).
- Read-only: `CpuAccessPlan.BatchBinding`; `CpuGeneralScanEmitter.Shaped` (`batchStrides`, `longConstants`, `activeLoops`, `extentFact`, `loadByteFact`). No unapproved expansion.
- Positive-only tests: `test/.../internal/codegen/emit/CpuSymbolicScanGeneratedKernelTest.java`; `test/.../internal/cache/{CpuKernelSpecializationTest,CpuGeneratedKernelArtifactStoreTest}.java`. Retain noncombined baseline resource.
- Read-only `test/.../internal/codegen/emit/`: `CpuSymbolicLengthScanEquivalentJavaOracle.java` (`reverseFloat32` dX/dH), `CpuSymbolicLengthScanFixture.java`. E uses int-feature affine addresses; emitter advances two long cursors.
- Positive-only docs: root `Core invariants`, backend `CPU backend routes`, `docs/backend-guide/cpu-backend.md` combined identity/CPU0060 sections, this brief. MAIN integrates; no ADR/redesign.

## Acceptance criteria

- Generation-only eligibility: combined symbolic+length, output/left ROW(0), right SHARED; fixed int-bounded K, not B (`BatchBinding.extentBatchFactors().get(1)==0`, checked `extentConstants`, corresponding right-axis agreement); selected leftK/rightK `longConstants==width`. Choose minimal existing proof, never nominal-stride/sample-B equality. Unproved/changing/zero/noncontiguous strides retain fallback.
- Int K, widened long affine addresses from selected cell bases; no two cursor recurrences. Activity dominates operand reads/arithmetic. Preserve carriers/base versions, cuts, checked access, epilogues/saved stores, ordered typed zero/product/ADD and DISTINCT per-time carry ADD. Empty K publishes typed +0. No hidden helpers/allocation/boxing/reflection/dispatch.
- Combined baked v7→v8 in BOTH streams; reject stale v7 and earlier artifacts. Canonical `:symbolic-length-v1`, pack `typed-explicit-b-v1`, schema/family 71/resources/noncombined bytes unchanged.
- Before edits capture dirty source/G7 ClassFile/identity/ABI/resources. G7 SHA-256 `fe48de1f10b6dc257c548d716fad0a1ccf97653bc1899a3e3499beb986b008f9`. Reuse authenticated CPU0060 20-case noncombined baseline; compare fresh generation-only outputs, no numerical test-method calls or semantic re-derivation/skips. Fallback code equal modulo combined names/identity.
- Frozen diagnostic retention rule: median of same-round G8/G7 ratios <0.99 in all four B×mechanism strata; <1.0 in each stratum's zero-based early/middle/late windows 1000–1099/1100–1199/1200–1299 (0–999 warmup). Require valid raw checks, settled actual timed paths including inlined code and no contradictory JIT/order trend; otherwise benefit is unconfirmed. This is a conservative retention gate, not statistical significance, general speed/parity, pure-kernel timing or JIT causality; retain all rows.

## Validation — PERFORMANCE FIRST

Workroot `/private/tmp/synaptik-cpu0061.CFjsUt`; copy `/private/tmp/synaptik-cpu0060.uNRs5H/measurement.D6is3Y` mechanics off-tree, NO old-root writes. MAIN preservation/runner only; JDK26.0.1/Gradle9.6.1, injection/campaigns disabled.

1. Capture baseline/candidate; Stage A compile ONLY: `env -i PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin HOME=/Users/phujka USER=phujka TMPDIR=/private/tmp /bin/bash /private/tmp/synaptik-cpu0061.CFjsUt/run-validation.sh :backends:cpu:compileTestJava`. NO test task before benefit.
2. Generation-only JVM: Family construction, ClassFile extraction/verify/decompilation, identity/resources/ABI and G8 pin; NO forward/reverse/kernel calls. Inspect G7/G8/E loops/guards/bases. Compile off-tree benchmark; independent candidate AND compiled-launcher/protocol/manifest approval before MAIN hash release.
3. ONE benchmark JVM/attempt, 120-second wall including setup/checks: matched G7/G8/unchanged E, B7/B32,T4/F3/H2; standard+typed, six actions,100 calls/action,FIXED 1000 warmup+300 measured rounds/B. Totals 1,560,000 scheduled calls,15,600 samples (12,000 warmup/3,600 measured); untimed checks separate. Retain inherited balanced ordering, counters, setup/entry-cost disclosures, all outputs/samples.
4. Numeric ABI/raw-output, poison/padding/latest checks ONLY inside that JVM, before overwrite. Normal JIT logs; no forced compilation/adaptive warmup/separate numeric preflight/retries/extra campaigns. Longer fixed warmup addresses prior measured-wrapper C2 completion, NOT guaranteed full-path steady state; disclose measured-window compilation/failures.
5. No coherent benefit or invalid/confounded/failed evidence: STOP, no functional suites. Preserve results; `apply_patch` ONLY task-authored candidate/identity deltas back to captured dirty v7, or explicitly await user decision. No unvalidated stable-production cutover, reset of accepted work or retry. Record Incomplete/follow-up.
6. ONLY independent positive-result confirmation + MAIN retention unlock targeted tests, ONE final CPU test/Javadoc and MAIN's ONE checkpoint. Audit legacy destinations; runner writes new-root evidence, immutable agent reused only for tests. Preserve CPU0047 two-method timing exclusion; no added skips/root/duplicate successful suites.

Positive-only task arguments to Stage A's runner:

```bash
:backends:cpu:test --tests '*CpuSymbolicScanGeneratedKernelTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest'
:backends:cpu:test :backends:cpu:javadoc
:testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'
```

Cover F32/F64 forward/actual reverse, both traversals; K=0/1/other widths,B=0/1/feature collisions,T=0/zero/mixed lengths, poisoned padding, raw seeds/saved sites,reuse, selected-bank/noncontiguous/zero-stride fallbacks and CPU0060 row contractions. Assert inner-loop shape, stale-v7 rejection/v8 reuse, exact 20-case noncombined equality. Check links/anchors/whitespace/preservation and `git diff --check`.

## Documentation and review impact

Independent Class C review prelaunch/final; reuse successful tests. If retained, update emitter/IR/cache Javadocs, root/backend/guide v8 explanations; record unchanged-Javadoc review. Apply documentation rules/general/planning and later relevant profiles; targeted glossary search. Final Javadoc after final edits.

## Result

The user's performance-only continuation produced one candidate, not an accepted capability.
Only `CpuGeneralScanMatmulEmitter.java` and `CpuGeneralScanIr.java` changed temporarily, including
Javadoc and experimental v8. Compile-only passed; generation-only verification covered eight
combined and 20 exact noncombined cases, unchanged resources/ABI and both identity projections.
No numerical action ran before the sole benchmark; no functional suite or Javadoc task ran.

Independent Hegel prelaunch approval bound freeze `a90bee833d11b39c666f3ac9dbf498dca1f4c5a22044b46d4b3b8d2484f29631`.
The sole JVM passed 15,600 samples/1,560,000 scheduled calls plus 36 reverse/24 forward raw checks
in 7.339s, without malformed, partial, unchecked or schedule-error rows, failure, timeout or retry.
Full same-round median G8/G7 ratios: B7 standard 0.995136, typed 0.982322; B32 standard
0.996031, typed 0.995263. Only 1/4 full groups passed <0.99 and 11/12 windows passed <1.0;
B32 standard middle was 1.000383. Independent actual-result review confirms benefit NO.
Normal C2 kernels completed by 0.900s and wrappers by 1.509s, before B7/B32 measurement at
1.804s/6.100s; complete adapter-path readiness remains unresolved. No parity/JIT-cause claim.

MAIN rejected retention. `apply_patch` restored ONLY those two files byte-exact to captured
dirty v7; compile-only passed again and all 12 selected runtime classes match their baseline.
Candidate source/classes, raw rows, analysis, review bindings and restoration receipts remain in
`/private/tmp/synaptik-cpu0061.CFjsUt`; 610 benchmark/G8 files and 2286 prior files stayed unchanged.
Hegel independently approved the actual result, exact restoration and planning closure with no open findings.
The published base, empty index and unrelated dirty work are preserved. No commit/push occurred.
Accepted v7 documentation/Javadocs remain accurate; planning-only closure records this negative
result. No new terminology, glossary/ADR change or functional rerun is required for restored source.
The residual Java gap remains; any different candidate needs separate authorization, not a retry.
Completion below refers to the performance experiment/restoration; the implementation is Cancelled.

Status: Complete
