# Task CPU0045: Equivalent clean Java for symbolic-length Scan reverse

## Status

Complete — MAIN accepted the independent source/evidence reviews and actual documentation diff
on 2026-10-28. The bounded result does not establish a speedup or parity with direct Java.

## Change class

Class C conservatively: generated-code/hot-loop analysis and a possible narrow emission fix.
Clean implementation and independent targeted code/documentation review of the actual result are required.

## Goal

Create an executable, genuinely equivalent optimal clean-Java reference for the accepted v4
symbolic-B/runtime-length general Scan actual reverse. Compare actual ClassFiles, hot loops and
dataflow with that reference and retained successful just-in-time (JIT) evidence. Fix only
specifically demonstrated avoidable overhead, if any. Preserve the unchanged fastest literal
T4/0/1 Java reference as the second target; a no-specific-overhead conclusion is valid.

## Scope and non-goals

- Primary witness: homogeneous F32, original forward traversal and actual Compiler-generated
  opposite-traversal reverse, F3/H2, invocation B and lengths; retained timing cases T4/B7/B32.
- New reference retains runtime T and actual scalar operands/initializers, not substituted
  literal 4/0/1. This models the existing entry, not variable-T prepared capability.
- Use existing owners; add only a focused test-only oracle. Extend existing tests/fixture and,
  only if warranted, add a CPU0045-only opt-in timing/preflight entry. Keep old entries immutable.
- No algorithm, public/supported API, application binary interface (ABI), storage/resource contract,
  capability, dependency, native, OpenBLAS, unsafe access, route/tuning or numerical-policy change.
  No new production abstraction, broad cleanup/campaign, replay, compaction or relaxed semantics.
- CPU0044 and its repeat evidence/history/protocols are read-only. Their budgets are not reopened.
  Stop for MAIN on contract ambiguity, a new baked fact or any required scope expansion.

## Contracts

Start at [ARCHITECTURE.md — Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
Read these exact applicable headings, not unrelated contracts or whole historical tasks:

- Root [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules).
- `backend-execution.md`: [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules),
  [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling),
  [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes),
  [Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract).
- `recurrent-scan.md`: [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings),
  [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates),
  [Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic),
  [Empty domains and numeric resources](../../../../architecture/contracts/recurrent-scan.md#empty-domains-and-numeric-resources),
  [Bounded row-role analysis](../../../../architecture/contracts/recurrent-scan.md#bounded-row-role-analysis),
  [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories),
  [First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse),
  [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates).

## Dependencies and integration

- Depends on: [CPU0044](0044-symbolic-length-scan-code-shape.md#result),
  [Prepare0015](../../../modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result),
  [CPU0041](0041-symbolic-scan-reverse-carrier-specialization.md#result),
  [CPU0042](0042-length-aware-general-scan-execution.md#result) — all Complete.
- Conflicts with: any writer/measurement changing Scan emission, carriers, resources, identities,
  oracle/harness or shared contracts/planning; serialize implementation then result review.
- Parallel group: None; one fresh executor, no concurrent writer or measurement.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  PLUS the accepted dirty tree in `/private/tmp/synaptik-cpu0045.ICUu7M/request-base.json`
  (2487 repository paths, empty index); HEAD alone is not the implementation baseline.
- Integration order: MAIN packet review → fresh executor → independent actual-result code/docs
  review → MAIN acceptance/status integration. Conditional measurement follows the freeze gate below.
- Integration validation: reuse final focused/CPU/Javadoc and structural evidence below;
  MAIN checks the combined diff, links, baseline/evidence preservation and review disposition once.
- Shared-document integration owner: MAIN; owns roadmap/master and any required narrow
  current-version wording. Executor owns this brief's compact Result, not completed CPU0044 records.

Ready verification, 2026-10-27: MAIN verified predecessor Status/Result, accepted reviews/gates,
unchanged HEAD/origin, empty index and no competing writer/measurement. Planner independently
matched all 2487 request-base paths and protected repair/repeat hashes before editing these three
planning documents. This request authorizes CPU0045, not a fifth CPU0044 slot. MAIN captures the
post-plan source-executor freeze before launch; recheck that freeze, not HEAD, on entry.

## Files and symbols

Paths below are relative to `modules/backend/cpu/src/`; production prefix is
`main/java/io/github/pho001/synaptik/backend/cpu/internal/`, test prefix is the same under `test/java/`.

- Production `codegen/emit/CpuGeneralScanEmitter.java`: `Shaped.emitSymbolicLength`,
  `emitLengthAware`, `bindSymbolicGeometry`, `cacheLengthScalars`, `arithmeticCell`,
  `copyInitial`, `lengthTimeProduct`, `load`/`store`; analysis first, edit only a proved redundancy.
- Production `codegen/emit/CpuCarrierEmitter.java`: `prepareSegmentLayouts`, `loadBytes`/`storeBytes`;
  `codegen/emit/CpuClassFileKernelGenerator.java`: `generateClassBytes`/general-Scan entry setup.
- Production `ir/CpuGeneralScanIr.java`: `Stages`, `canonicalIdentity`, `bakedSpecializationIdentity`;
  `cache/CpuKernelSpecialization.java`: `compatibilityBytes`/`classIdentityBytes`.
- Read-only production ABI reference: `lowering/CpuGeneralScanLowering.java`, `Facts.pack(long)`.
- Test `codegen/emit/CpuSymbolicLengthScanCleanJavaOracle.java`: existing `reverseFloat32`
  and complete literal-reference source/class stay unchanged.
- Add test-only `codegen/emit/CpuSymbolicLengthScanEquivalentJavaOracle.java` in existing
  `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`; direct typed reverse reference,
  not an interpreter or a transcription retaining artificial generated overhead.
- Test `codegen/emit/CpuSymbolicLengthScanFixture.java`: `Family`, independent carriers/packing,
  captured reverse actions; `CpuSymbolicScanGeneratedKernelTest.java`: equivalence/structural cases;
  `CpuGeneralScanTimingTest.java`: new CPU0045-only entries if needed, old methods unchanged.
- If emission changes: existing `cache/CpuKernelSpecializationTest.java` and
  `cache/CpuGeneratedKernelArtifactStoreTest.java` for identity/stale-load/preservation checks.

## Frozen baseline and retained evidence

Artifact root: `/private/tmp/synaptik-cpu0045.ICUu7M`; write new evidence only beneath it.
`protected-evidence.json` protects all 1074 repair and 102 repeat files; never overwrite them.
Retained v4 reverse `abi-preflight/measured-1.class` SHA-256:
`beee86e2474d5a276a731726eff6b895c13d814b9f5788db65a58045ee60428c`.
Unchanged literal oracle `.java` SHA-256 `8350c3163c31580cd034abc634ae603e676b714322664e4febaf16f02fcb2eb2`;
`.class` SHA-256 `57d30a13da1a58836a1a89463295543c05ec2ff45d208c945d3fbb89242351f9`.

Only explicitly needed predecessor evidence, under `/private/tmp/synaptik-cpu0044-repair.FodFPm/`:
`code-comparison.txt`, `javap/v4.txt`, `abi-preflight/`, `comparison/final-analysis.json` and
`comparison/hotspot_pid61708.log`. `RepairHotLoopsEquivalent.java` is labeled PSEUDOCODE,
not executable reference or actual decompilation. Successful common-live normal-C2 evidence
reported v4/Java medians B7 760.205/552.295 ns and B32 3080.835/1750.210 ns; this is not causal proof.
Under `/private/tmp/synaptik-cpu0044-repeat.K6Qi8L/`, `repeat-result-v2.txt`/`analysis-v2.json`
report raw B7 783.955/576.455 and B32 3044.585/1832.290 ns; JIT XML is missing, so no strict JIT claim.

## Acceptance criteria

1. New compiled Java reference uses runtime T/B and actual derivative scalar/shared initializers;
   keep fixed F3/H2 specialization and checked typed accesses. Preserve producing/rounding sites,
   saved primals, original-time indices, length gates before loads/arithmetic, increasing-b
   contributors and one distinct shared-carry ADD per active time. Preserve singleton/raw SUM,
   inactive raw carry, +0 padding, T=0/B=0 and all-zero-length behavior; no activation replay.
2. Qualify against actual Compiler reverse with nonuniform inputs, joint seeds, mixed/empty/full
   activity, B0/1/7/32 and separately compiled fixed T0/1/4 cases. Vary actual scalar initializers
   and derivative operand so literal substitutions fail; check raw special/identity boundaries.
   Retain existing independent active-prefix semantic evidence; do not weaken the fastest oracle.
3. Retain actual generated and javac ClassFiles/disassembly plus a concise loop/dataflow mapping.
   Distinguish semantic requirements, structural redundancies and unknown JIT effects. Check hidden
   helpers/allocation/boxing/reflection/dispatch. Do not infer check/address cost, register pressure
   or foreign-memory API causality from bytecode count, timing ratios or pseudocode.
4. Record each specific overhead proof before a narrow edit, or conclude none demonstrated and
   leave production unchanged. Do not redo the four accepted v4 fixes. Runtime T is not a baked
   key fact: do not bake 4. If bytes change, advance only the combined baked identity in both
   streams, reject stale artifacts and have MAIN integrate required current-version wording;
   preserve canonical v1, numeric pack/ABI, cuts/scratch and every noncombined identity/byte.
5. Any shared-emitter change retains F32/F64, both traversals, empty/special cases, checked access
   and existing fixed/static/all-valid behavior. Independent actual-result review approves the
   reference, source diff, evidence limitations and targeted documentation before completion.

## Validation and conditional measurement budget

Executor runs focused qualification, then one final affected-module test/Javadoc after executable
changes stabilize. Performance flags stay false for ordinary validation; reviewers reuse evidence.

```bash
./gradlew :backends:cpu:test --tests '*CpuSymbolicScanGeneratedKernelTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest' -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=1 --console=plain
./gradlew :backends:cpu:test :backends:cpu:javadoc -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=1 --console=plain
```

No repository-wide rerun, native rebuild or Gradle/build-file edit is authorized. MAIN reuses
accepted integration/conformance evidence unless a concrete changed-path risk requires a decision.

- No proved fix means no measurement is needed. Only ONE actual measured body is available if
  warranted, in one normal-tiered JDK 26.0.1 JVM: retained v4 versus candidate, equivalent Java
  and unchanged fastest literal Java, F32/original-forward/T4/F3/H2, B7 then B32. No v3 campaign.
- Ceiling: 100 warmup + 300 measured rounds per B, 100 calls/target/round, at most four targets
  (320000 clocked calls including warmup), 60 seconds whole body including cold setup/checks.
  One no-timing launcher dry-run/ABI preflight is allowed; no warmup or timed batches there.
- Before measured launch MAIN accepts the exact source/classes, own carrier/pack mapping,
  selector, command/JVM flags and compact fixed schedule after independent targeted review.
  Use CPU0045-only methods and already-forwarded properties; never rerun old opt-in entries.
  Create the JIT XML output directory BEFORE test-JVM start, not inside the test body.
- Retain raw sides, phase brackets, hashes, JUnit and complete/censored/deadline disposition.
  Preserve independent outputs/scratch, equal work and balanced fixed target order. Separate cold
  setup/admission from entry-inclusive kernel timing; disclose MethodHandle/spreader versus direct
  entry, no empty-call subtraction. Use retained successful XML for existing JIT analysis.
  New strict JIT claims require actual normal-nmethod live intervals and wrapper transitions;
  missing XML or PrintCompilation alone cannot establish them. No retry, repeat or further campaign.

## Documentation and review impact

Apply [documentation rules](../../../../developer-guide/documentation-rules.md), General style
and Planning profile to this brief; use API/Javadoc and Backend-guide profiles for affected text.
Review affected Javadoc and the [CPU guide's combined repair section](../../../../backend-guide/cpu-backend.md#cpu0044-combined-code-shape-repair).
Update narrowly for an actual new implementation fact, otherwise record a reasoned no-change
conclusion. MAIN owns any root/backend-owner current-version wording; no semantic contract change.
Search glossary terms locally; no new terminology is intended. Independent Class C result review
checks actual source/tests/ClassFiles, documentation and links, reusing successful executable gates.

## Result

- Added executable `CpuSymbolicLengthScanEquivalentJavaOracle` with runtime T/B and actual scalar operands;
  changed Emitter/IR, Fixture/GeneratedKernel/Timing tests and specialization/artifact-store tests.
- Combined-only stable own-final-port/no-bank binding preserves raw initial copies and each view's nominal
  recipe, enabling cell-scoped offset/zero-base sharing; baked v5 alone advances both streams.
- Actual v5 Class-File map proves four shared read/write offset pairs; canonical v1, numeric pack/ABI,
  resources and noncombined bytes remain unchanged. Literal Java source/class and old entries are intact.
- Reused focused 55 PASS, 216-case equivalence, typed/traversal/raw/empty, cache/bank/view/alias checks.
- Full CPU run: 1343 reported, 1290 passed, one task-caused selector collision, 52 skipped. Repaired
  class: 31 PASS; successful unaffected results reused, not a clean post-repair full-module rerun.
- CPU Javadoc: successful generation with 94 existing warnings; final documentation-only command/outcome
  is recorded in `/private/tmp/synaptik-cpu0045.ICUu7M/doc-closeout-receipt.txt`. No tests duplicated here.
- Sole measured body: 1/1 consumed, uncensored 1.129084042 s including persistence, 320000 clocked calls.
  Raw median paired v5/v4 latency: B7 1.032334, B32 1.011950; no demonstrated v5 speedup.
- Normal-C2 kernel window retains 300 rounds/B; B7 wrapper transitions require a separate 204-round
  normal-C1-host segment (v5/v4 1.028667), not a blanket all-wrapper-C2 or steady-state claim.
- Unequal entry/harness costs remain included; no pure-inner-loop, causal, parity or significance claim.
- Evidence root `/private/tmp/synaptik-cpu0045.ICUu7M`: `validation.txt`, `loop-dataflow-mapping.md`,
  `measurement-result.txt` and independent semantic/prelaunch/measurement receipts; Q1–Q3 are closed.
- Docs/Javadoc: root/backend-owner current-v5 receipts, CPU guide and specialization Javadoc updated;
  glossary reviewed, no new term, API/ABI/capability/semantics/architecture decision or ADR change.
- Frozen protocol is retained atomically despite this brief exceeding 15 KB; detailed evidence stays outside it.

MAIN closeout: actual five-file documentation/Javadoc diff, 22 frozen pins, 157 local links,
132 anchors, accepted source baseline and protected predecessor evidence checked successfully;
see `doc-closeout-receipt.txt`. Current plan statuses are integrated. The measurement budget is
exhausted; remaining Scan performance analysis is deferred behind the approved structural repairs.

Status: Complete
