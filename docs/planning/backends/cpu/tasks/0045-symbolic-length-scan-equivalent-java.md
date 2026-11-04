# Task CPU0045: Equivalent clean Java for symbolic-length [Scan](../../../../glossary.md#term-cumulative-scan) reverse

## Reader context — historical integration boundary

An equivalent Java oracle must retain the generated entry's actual scalar inputs and runtime T/B,
rather than substituting a literal fixed-time/zero/one algorithm that happens to match one fixture.
That runtime-T argument does not grant variable-T prepared execution. CPU's bounded production
change shares proved read/write offsets only for a carry with its own final output and no changing
bank; raw initial copies and other views retain their required mappings. The literal fastest Java
reference remains a separate target. Structural work removal is therefore distinct from measured
improvement: this record demonstrated no v5 speedup and did not close the remaining parity question.

Current reading: [CPU0045 equivalent Java boundary](../../../../backend-guide/cpu-backend.md#cpu0045-equivalent-java-and-stable-carry-offsets); [Symbolic batch plus runtime lengths](../../../../backend-guide/cpu-backend.md#symbolic-length-scan-cpu-execution).

## Status

Complete — MAIN accepted the independent source/evidence reviews and actual documentation diff
on 2026-10-28. The bounded result does not establish a speedup or parity with direct Java.

## Change class

Class C conservatively: generated-code/hot-loop analysis and a possible narrow emission fix.
Clean implementation and independent targeted code/documentation review of the actual result are required.

## Goal

Create an executable, genuinely equivalent optimal clean-Java reference for the accepted v4
symbolic-B/[runtime](../../../../glossary.md#term-runtime)-length [general Scan](../../../../glossary.md#general-differentiable-scan) actual reverse. Compare actual ClassFiles, hot loops and
[dataflow](../../../../glossary.md#term-dataflow) with that reference and retained successful just-in-time ([JIT](../../../../glossary.md#term-just-in-time-compilation-jit)) evidence. Fix only
specifically demonstrated avoidable overhead, if any. Preserve the unchanged fastest literal
T4/0/1 Java reference as the second [target](../../../../glossary.md#term-target); a no-specific-overhead conclusion is valid.

## [Scope](../../../../glossary.md#term-scope) and non-goals

- Primary witness: homogeneous F32, original forward traversal and actual Compiler-generated
  opposite-traversal reverse, F3/H2, invocation B and lengths; retained timing cases T4/B7/B32.<span data-glossary-navigation="true"> ([homogeneous glossary](../../../../glossary.md#term-homogeneous); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- New reference retains runtime T and actual scalar operands/initializers, not substituted
  literal 4/0/1. This models the existing entry, not variable-T prepared capability.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [scalar glossary](../../../../glossary.md#term-scalar))</span>
- Use existing owners; add only a focused test-only oracle. Extend existing tests/fixture and,
  only if warranted, add a CPU0045-only opt-in timing/preflight entry. Keep old entries immutable.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [fixture glossary](../../../../glossary.md#term-fixture); [preflight glossary](../../../../glossary.md#term-preflight); [immutable glossary](../../../../glossary.md#term-immutable))</span>
- No algorithm, public/supported API, application binary interface (ABI), storage/resource contract,
  capability, dependency, native, OpenBLAS, unsafe access, route/tuning or numerical-policy change.
  No new production abstraction, broad cleanup/campaign, replay, compaction or relaxed semantics.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#application-binary-interface-and-native-bridge))</span><span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [application binary interface glossary](../../../../glossary.md#term-application-binary-interface-abi); [dependency glossary](../../../../glossary.md#term-dependency); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- CPU0044 and its repeat evidence/history/protocols are read-only. Their budgets are not reopened.
  Stop for MAIN on contract ambiguity, a new baked fact or any required scope expansion.

## Contracts

Start at [ARCHITECTURE.md — Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>.
Read these exact applicable headings, not unrelated contracts or whole historical tasks:

- Root [Authority, incorporation, and precedence](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>.
- `backend-execution.md`: [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>,
  [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling),
  [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>,
  [Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>.
- `recurrent-scan.md`: [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings)<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [primal glossary](../../../../glossary.md#term-primal); [bindings glossary](../../../../glossary.md#term-binding))</span>,
  [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates)<span data-glossary-navigation="true"> ([Qualification glossary](../../../../glossary.md#term-qualification))</span>,
  [Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic),
  [Empty domains and numeric resources](../../../../architecture/contracts/recurrent-scan.md#empty-domains-and-numeric-resources),
  [Bounded row-role analysis](../../../../architecture/contracts/recurrent-scan.md#bounded-row-role-analysis),
  [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>,
  [First-order length-mode reverse](../../../../architecture/contracts/recurrent-scan.md#first-order-length-mode-reverse),
  [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [preflight glossary](../../../../glossary.md#term-preflight))</span>.

## Dependencies and integration

- Depends on: [CPU0044](0044-symbolic-length-scan-code-shape.md#result),
  [Prepare0015](../../../modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result),
  [CPU0041](0041-symbolic-scan-reverse-carrier-specialization.md#result),
  [CPU0042](0042-length-aware-general-scan-execution.md#result) — all Complete.
- Conflicts with: any writer/measurement changing Scan emission, carriers, resources, identities,
  oracle/harness or shared contracts/planning; serialize implementation then result review.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([Scan glossary](../../../../glossary.md#term-cumulative-scan); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
- Parallel group: None; one fresh executor, no concurrent writer or measurement.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  PLUS the accepted dirty tree in `/private/tmp/synaptik-cpu0045.ICUu7M/request-base.json`
  (2487 repository paths, empty index); HEAD alone is not the implementation baseline.
- Integration order: MAIN packet review → fresh executor → independent actual-result code/docs
  review → MAIN acceptance/status integration. Conditional measurement follows the freeze gate below.
- Integration validation: reuse final focused/CPU/Javadoc and structural evidence below;
  MAIN checks the combined diff, links, baseline/evidence preservation and review disposition once.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [CPU glossary](../../../../glossary.md#term-cpu); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration owner: MAIN; owns roadmap/master and any required narrow
  current-version wording. Executor owns this brief's compact Result, not completed CPU0044 records.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

[Ready](../../../../glossary.md#term-ready) verification, 2026-10-27: MAIN verified predecessor Status/Result, accepted reviews/gates,
unchanged HEAD/origin, empty index and no competing writer/measurement. Planner independently
matched all 2487 request-base paths and protected repair/repeat [hashes](../../../../glossary.md#term-hash) before editing these three
[planning](../../../../glossary.md#term-planning) documents. This request authorizes CPU0045, not a fifth CPU0044 slot. MAIN [captures](../../../../glossary.md#term-capture-meanings) the
post-plan source-executor freeze before launch; recheck that freeze, not HEAD, on entry.

## Files and symbols

Paths below are relative to `modules/backend/cpu/src/`; production prefix is
`main/java/io/github/pho001/synaptik/backend/cpu/internal/`, test prefix is the same under `test/java/`.

- Production `codegen/emit/CpuGeneralScanEmitter.java`: `Shaped.emitSymbolicLength`,
  `emitLengthAware`, `bindSymbolicGeometry`, `cacheLengthScalars`, `arithmeticCell`,
  `copyInitial`, `lengthTimeProduct`, `load`/`store`; analysis first, edit only a proved redundancy.
- Production `codegen/emit/CpuCarrierEmitter.java`: `prepareSegmentLayouts`, `loadBytes`/`storeBytes`;
  `codegen/emit/CpuClassFileKernelGenerator.java`: `generateClassBytes`/general-[Scan](../../../../glossary.md#term-cumulative-scan) entry setup.
- Production `ir/CpuGeneralScanIr.java`: `Stages`, `canonicalIdentity`, `bakedSpecializationIdentity`;
  `cache/CpuKernelSpecialization.java`: `compatibilityBytes`/`classIdentityBytes`.
- Read-only production [ABI](../../../../glossary.md#term-application-binary-interface-abi) reference: `lowering/CpuGeneralScanLowering.java`, `Facts.pack(long)`.
- Test `codegen/emit/CpuSymbolicLengthScanCleanJavaOracle.java`: existing `reverseFloat32`
  and complete literal-reference source/class stay unchanged.
- Add test-only `codegen/emit/CpuSymbolicLengthScanEquivalentJavaOracle.java` in existing
  `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`; direct typed reverse reference,
  not an interpreter or a transcription retaining artificial generated overhead.
- Test `codegen/emit/CpuSymbolicLengthScanFixture.java`: `Family`, independent carriers/packing,
  captured reverse actions; `CpuSymbolicScanGeneratedKernelTest.java`: equivalence/structural cases;
  `CpuGeneralScanTimingTest.java`: new CPU0045-only entries if needed, old methods unchanged.
- If emission changes: existing `cache/CpuKernelSpecializationTest.java` and
  `cache/CpuGeneratedKernelArtifactStoreTest.java` for [identity](../../../../glossary.md#term-identity-meanings)/stale-load/preservation checks.

## Frozen baseline and retained evidence

[Artifact](../../../../glossary.md#term-artifact) root: `/private/tmp/synaptik-cpu0045.ICUu7M`; write new evidence only beneath it.
`protected-evidence.json` protects all 1074 repair and 102 repeat files; never overwrite them.
Retained v4 reverse `abi-preflight/measured-1.class` [SHA-256](../../../../glossary.md#term-sha-256):
`beee86e2474d5a276a731726eff6b895c13d814b9f5788db65a58045ee60428c`.
Unchanged literal [oracle](../../../../glossary.md#term-test-oracle) `.java` SHA-256 `8350c3163c31580cd034abc634ae603e676b714322664e4febaf16f02fcb2eb2`;
`.class` SHA-256 `57d30a13da1a58836a1a89463295543c05ec2ff45d208c945d3fbb89242351f9`.

Only explicitly needed predecessor evidence, under `/private/tmp/synaptik-cpu0044-repair.FodFPm/`:
`code-comparison.txt`, `javap/v4.txt`, `abi-preflight/`, `comparison/final-analysis.json` and
`comparison/hotspot_pid61708.log`. `RepairHotLoopsEquivalent.java` is labeled [PSEUDOCODE](../../../../glossary.md#term-pseudocode),
not executable reference or actual [decompilation](../../../../glossary.md#term-decompilation). Successful common-live normal-[C2](../../../../glossary.md#term-c2) evidence
reported v4/Java medians B7 760.205/552.295 ns and B32 3080.835/1750.210 ns; this is not causal proof.
Under `/private/tmp/synaptik-cpu0044-repeat.K6Qi8L/`, `repeat-result-v2.txt`/`analysis-v2.json`
report raw B7 783.955/576.455 and B32 3044.585/1832.290 ns; [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) XML is missing, so no strict JIT claim.

## Acceptance criteria

1. New compiled Java reference uses runtime T/B and actual derivative scalar/shared initializers;
   keep fixed F3/H2 specialization and checked typed accesses. Preserve producing/rounding sites,
   saved primals, original-time indices, length gates before loads/arithmetic, increasing-b
   contributors and one distinct shared-carry ADD per active time. Preserve singleton/raw SUM,
   inactive raw carry, +0 padding, T=0/B=0 and all-zero-length behavior; no activation replay.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [derivative glossary](../../../../glossary.md#term-derivative); [scalar glossary](../../../../glossary.md#term-scalar); [specialization glossary](../../../../glossary.md#term-specialization); [rounding glossary](../../../../glossary.md#term-rounding); [carry glossary](../../../../glossary.md#term-carry); [singleton glossary](../../../../glossary.md#term-singleton); [padding glossary](../../../../glossary.md#padding); [activation glossary](../../../../glossary.md#term-activation))</span>
2. Qualify against actual Compiler reverse with nonuniform inputs, joint seeds, mixed/empty/full
   activity, B0/1/7/32 and separately compiled fixed T0/1/4 cases. Vary actual scalar initializers
   and derivative operand so literal substitutions fail; check raw special/identity boundaries.
   Retain existing independent active-prefix semantic evidence; do not weaken the fastest oracle.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [identity glossary](../../../../glossary.md#term-identity-meanings); [oracle glossary](../../../../glossary.md#term-test-oracle))</span>
3. Retain actual generated and javac ClassFiles/disassembly plus a concise loop/dataflow mapping.
   Distinguish semantic requirements, structural redundancies and unknown JIT effects. Check hidden
   helpers/allocation/boxing/reflection/dispatch. Do not infer check/address cost, register pressure
   or foreign-memory API causality from bytecode count, timing ratios or pseudocode.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit); [dispatch glossary](../../../../glossary.md#term-dispatch); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [bytecode glossary](../../../../glossary.md#term-bytecode); [pseudocode glossary](../../../../glossary.md#term-pseudocode))</span>
4. Record each specific overhead proof before a narrow edit, or conclude none demonstrated and
   leave production unchanged. Do not redo the four accepted v4 fixes. Runtime T is not a baked
   key fact: do not bake 4. If bytes change, advance only the combined baked identity in both
   streams, reject stale artifacts and have MAIN integrate required current-version wording;
   preserve canonical v1, numeric pack/ABI, cuts/scratch and every noncombined identity/byte.<span data-glossary-navigation="true"> ([artifacts glossary](../../../../glossary.md#term-artifact); [canonical glossary](../../../../glossary.md#term-canonical-form); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
5. Any shared-emitter change retains F32/F64, both traversals, empty/special cases, checked access
   and existing fixed/static/all-valid behavior. Independent actual-result review approves the
   reference, source diff, evidence limitations and targeted documentation before completion.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>

## [Validation](../../../../glossary.md#term-validation) and conditional measurement budget

Executor runs focused [qualification](../../../../glossary.md#term-qualification), then one final affected-[module](../../../../glossary.md#term-module) test/[Javadoc](../../../../glossary.md#term-javadoc) after executable
changes stabilize. Performance flags stay false for ordinary validation; reviewers reuse evidence.

```bash
./gradlew :backends:cpu:test --tests '*CpuSymbolicScanGeneratedKernelTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest' -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=1 --console=plain
./gradlew :backends:cpu:test :backends:cpu:javadoc -Dsynaptik.cpu.generalScan.performance=false -Dsynaptik.cpu.generalScan.pointwisePerformance=false -Dsynaptik.cpu.dynamicBatch.performance=false --max-workers=1 --console=plain
```

No repository-wide rerun, native rebuild or [Gradle](../../../../glossary.md#term-gradle)/build-file edit is authorized. MAIN reuses
accepted integration/[conformance](../../../../glossary.md#term-conformance) evidence unless a concrete changed-path risk requires a decision.

- No proved fix means no measurement is needed. Only ONE actual measured body is available if
  warranted, in one normal-tiered JDK 26.0.1 JVM: retained v4 versus candidate, equivalent Java
  and unchanged fastest literal Java, F32/original-forward/T4/F3/H2, B7 then B32. No v3 campaign.<span data-glossary-navigation="true"> ([JDK glossary](../../../../glossary.md#term-java-development-kit-jdk); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- Ceiling: 100 warmup + 300 measured rounds per B, 100 calls/target/round, at most four targets
  (320000 clocked calls including warmup), 60 seconds whole body including cold setup/checks.
  One no-timing launcher dry-run/ABI preflight is allowed; no warmup or timed batches there.<span data-glossary-navigation="true"> ([warmup glossary](../../../../glossary.md#term-warmup); [target glossary](../../../../glossary.md#term-target); [run glossary](../../../../glossary.md#term-run); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [preflight glossary](../../../../glossary.md#term-preflight); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
- Before measured launch MAIN accepts the exact source/classes, own carrier/pack mapping,
  selector, command/JVM flags and compact fixed schedule after independent targeted review.
  Use CPU0045-only methods and already-forwarded properties; never rerun old opt-in entries.
  Create the JIT XML output directory BEFORE test-JVM start, not inside the test body.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit))</span>
- Retain raw sides, phase brackets, hashes, JUnit and complete/censored/deadline disposition.
  Preserve independent outputs/scratch, equal work and balanced fixed target order. Separate cold
  setup/admission from entry-inclusive kernel timing; disclose MethodHandle/spreader versus direct
  entry, no empty-call subtraction. Use retained successful XML for existing JIT analysis.
  New strict JIT claims require actual normal-nmethod live intervals and wrapper transitions;
  missing XML or PrintCompilation alone cannot establish them. No retry, repeat or further campaign.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([JUnit glossary](../../../../glossary.md#term-junit); [censored glossary](../../../../glossary.md#term-censored); [admission glossary](../../../../glossary.md#term-admission); [kernel glossary](../../../../glossary.md#kernel); [MethodHandle glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture); [nmethod glossary](../../../../glossary.md#term-nmethod))</span>

## Documentation and review impact

Apply [documentation rules](../../../../developer-guide/documentation-rules.md), General style
and [Planning](../../../../glossary.md#term-planning) profile to this brief; use [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc) and [Backend](../../../../glossary.md#backend)-guide profiles for affected text.
Review affected Javadoc and the [CPU guide's combined repair section](../../../../backend-guide/cpu-backend.md#cpu0044-combined-code-shape-repair)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>.
Update narrowly for an actual new implementation fact, otherwise record a reasoned no-change
conclusion. MAIN owns any root/backend-[owner](../../../../glossary.md#term-owner) current-version wording; no semantic contract change.
Search glossary terms locally; no new terminology is intended. Independent Class C result review
checks actual source/tests/ClassFiles, documentation and links, reusing successful executable gates.

## Result

- Added executable `CpuSymbolicLengthScanEquivalentJavaOracle` with runtime T/B and actual scalar operands;
  changed Emitter/IR, Fixture/GeneratedKernel/Timing tests and specialization/artifact-store tests.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [scalar glossary](../../../../glossary.md#term-scalar); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [Fixture glossary](../../../../glossary.md#term-fixture); [specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- Combined-only stable own-final-port/no-bank binding preserves raw initial copies and each view's nominal
  recipe, enabling cell-scoped offset/zero-base sharing; baked v5 alone advances both streams.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [nominal glossary](../../../../glossary.md#term-nominal))</span>
- Actual v5 Class-File map proves four shared read/write offset pairs; canonical v1, numeric pack/ABI,
  resources and noncombined bytes remain unchanged. Literal Java source/class and old entries are intact.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Reused focused 55 PASS, 216-case equivalence, typed/traversal/raw/empty, [cache](../../../../glossary.md#term-cache)/bank/[view](../../../../glossary.md#term-view-meanings)/[alias](../../../../glossary.md#aliasing-and-arena) checks.
- Full CPU run: 1343 reported, 1290 passed, one task-caused selector collision, 52 skipped. Repaired
  class: 31 PASS; successful unaffected results reused, not a clean post-repair full-module rerun.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [run glossary](../../../../glossary.md#term-run); [module glossary](../../../../glossary.md#term-module))</span>
- CPU Javadoc: successful generation with 94 existing warnings; final documentation-only command/outcome
  is recorded in `/private/tmp/synaptik-cpu0045.ICUu7M/doc-closeout-receipt.txt`. No tests duplicated here.
- Sole measured body: 1/1 consumed, uncensored 1.129084042 s including persistence, 320000 clocked calls.
  Raw median paired v5/v4 latency: B7 1.032334, B32 1.011950; no demonstrated v5 speedup.<span data-glossary-navigation="true"> ([persistence glossary](../../../../glossary.md#term-persistence); [median glossary](../../../../glossary.md#term-median); [latency glossary](../../../../glossary.md#term-latency))</span>
- Normal-C2 kernel window retains 300 rounds/B; B7 wrapper transitions require a separate 204-round
  normal-C1-host segment (v5/v4 1.028667), not a blanket all-wrapper-C2 or steady-state claim.<span data-glossary-navigation="true"> ([C2 glossary](../../../../glossary.md#term-c2); [kernel glossary](../../../../glossary.md#kernel); [C1 glossary](../../../../glossary.md#term-c1); [host glossary](../../../../glossary.md#term-host))</span>
- Unequal entry/harness costs remain included; no pure-inner-loop, causal, parity or significance claim.
- Evidence root `/private/tmp/synaptik-cpu0045.ICUu7M`: `validation.txt`, `loop-dataflow-mapping.md`,
  `measurement-result.txt` and independent semantic/prelaunch/measurement receipts; Q1–Q3 are closed.
- Docs/Javadoc: root/backend-owner current-v5 receipts, CPU guide and specialization Javadoc updated;
  glossary reviewed, no new term, API/ABI/capability/semantics/architecture decision or ADR change.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [owner glossary](../../../../glossary.md#term-owner); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- Frozen protocol is retained atomically despite this brief exceeding 15 KB; detailed evidence stays outside it.

MAIN closeout: actual five-file documentation/[Javadoc](../../../../glossary.md#term-javadoc) diff, 22 frozen pins, 157 local links,
132 anchors, accepted source baseline and protected predecessor evidence checked successfully;
see `doc-closeout-receipt.txt`. Current plan statuses are integrated. The measurement budget is
exhausted; remaining [Scan](../../../../glossary.md#term-cumulative-scan) performance analysis is deferred behind the approved structural repairs.

Status: Complete
