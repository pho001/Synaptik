# Task CPU0034: General-[scan](../../../../glossary.md#term-cumulative-scan) [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) loops and [carry](../../../../glossary.md#term-carry) transition

## Reader context — historical integration boundary

There are two independent changes in this record: coordinate induction controls body-cell traversal,
and carry rebinding selects which stored time version the next step reads. CPU's generator may use
an int coordinate only under a proved extent bound and must widen it before byte-address arithmetic.
A carry backed by a demanded emission can be rebound after the complete step only when old-state
reads and final-next views remain correct; otherwise separate storage/selection stays necessary.
Runtime does not choose these mappings. The bounded measurement compares one generated/Java workload
and does not requalify earlier campaign results.

Current reading: [Retained static Scan resources](../../../../backend-guide/cpu-backend.md#retained-static-scan-qualification-and-resources).

## Status

Complete — main accepted the independently reviewed implementation, [runtime](../../../../glossary.md#term-runtime) gates and bounded evidence on 2026-10-23.

## Change class

Class C — generated hot-loop/[dataflow](../../../../glossary.md#term-dataflow) changes and generated-[artifact](../../../../glossary.md#term-artifact) [cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings).
Separate clean [planning](../../../../glossary.md#term-planning), implementation, and independent code/documentation-review contexts;
all writes serial. Main verifies the frozen baseline before launching the executor.

## Goal

Make proved small-static-[extent](../../../../glossary.md#term-extent) `Shaped` [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) loops follow optimal clean Java induction,
and move proved emission-backed [carry](../../../../glossary.md#term-carry) selection to a post-step transition. Preserve the complete
declared [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) forward and actual [Compiler](../../../../glossary.md#term-compiler)-generated reverse domain, numerics and [ownership](../../../../glossary.md#term-ownership-meanings).
The intended sequence is initial [binding](../../../../glossary.md#term-binding) → step/read/save/write → next-carry binding.

## [Scope](../../../../glossary.md#term-scope) and non-goals

Use only existing [CPU](../../../../glossary.md#term-cpu) [scan](../../../../glossary.md#term-cumulative-scan) emission, derived [IR](../../../../glossary.md#term-intermediate-representation-ir) and [specialization](../../../../glossary.md#term-specialization)/[cache](../../../../glossary.md#term-cache) seams. No added or moved
top-level production type, package, public [API](../../../../glossary.md#term-application-programming-interface-api), [backend](../../../../glossary.md#backend), [dependency](../../../../glossary.md#term-dependency), [AD](../../../../glossary.md#autograd) or execution layer.
Preserve capability, predeclared resources, entry [application binary interface](../../../../glossary.md#term-application-binary-interface-abi) ([ABI](../../../../glossary.md#application-binary-interface-and-native-bridge)), temporal
lifetimes and [run](../../../../glossary.md#term-run) isolation. No other families, fixed-scan redirection, low types, [vector](../../../../glossary.md#term-vector),
parallel, [OpenBLAS](../../../../glossary.md#term-openblas), [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) or [Checkpoint](../../../../glossary.md#term-checkpoint) advancement. No global `CpuLoopEmitter` refactor; [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) K
and other independent loops stay unchanged unless a concrete [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)-[binding](../../../../glossary.md#term-binding) proof requires it.

## Contracts

- [Root](../../../../../ARCHITECTURE.md): **Authority, incorporation, and precedence**;
  **Core invariants**; **[Scope](../../../../glossary.md#term-scope)-indexed [normative](../../../../glossary.md#term-normative) contracts**; **[Dependency](../../../../glossary.md#term-dependency) rules**.
- [Backend execution](../../../../architecture/contracts/backend-execution.md): **[CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) routes**;
  **Performance evidence and optimization tooling**; **Profile-free [backend capability](../../../../glossary.md#backend-capability) and [identity](../../../../glossary.md#term-identity-meanings)**.
- [Scan owner](../../../../architecture/contracts/recurrent-scan.md)<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>: **[General differentiable scan](../../../../glossary.md#general-differentiable-scan)**;
  **[Canonical](../../../../glossary.md#term-canonical-form) [residual](../../../../glossary.md#scan-residual-port-and-history-demand) ports and demand**; **Reverse [scan](../../../../glossary.md#term-cumulative-scan) and exact [primal](../../../../glossary.md#term-primal) [bindings](../../../../glossary.md#term-binding)**;
  **Traversal, empty domains and declared [qualification](../../../../glossary.md#term-qualification)**; **Qualification, performance and migration gates**.

These plans add no architecture authority. Stop on missing/ambiguous contracts, incompatible
architecture, or required scope expansion; do not invent a layer or numerical permission.

## Dependencies and integration

- Depends on: CPU0033 and low-level native join Complete; Compiler0012 Complete as accepted
  compatibility baseline (including their already accepted predecessors/reviews/validation).<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation))</span>
- Conflicts with: all CPU general-scan emission/IR/cache identity, qualification/resources,
  shared scan explanation/CPU guide/planning writes; none active at packet preparation.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [cache glossary](../../../../glossary.md#term-cache); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- Parallel group: None; sole serial writer, no concurrent implementation/reviewer writes.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus all 263 accepted dirty paths, empty index, then this three-file planning packet;
  preserve every accepted change. This is not a committed or merged predecessor snapshot.<span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot))</span>
- Integration order: main frozen-base/Ready verification → clean CPU executor → separate independent
  Class C code/docs review/corrections → final worker gates → main checkpoint while campaign stopped
  → harness refresh/review → recovery campaign → main closure. Sole serial stream; no worktree/commit/push.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit))</span>
- Integration validation: reuse main's passed conformance/Engine checkpoint under Validation on
  unchanged reviewed runtime; no rerun or full root gate for this harness-only amendment.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [runtime glossary](../../../../glossary.md#term-runtime))</span>
- Shared-document integration owner: main; executor may draft affected Javadocs, independent reviewer
  exclusively finalizes targeted explanations/Javadocs while worker pauses, main owns planning closure.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

Launch verification (2026-10-23): main checked [Ready](../../../../glossary.md#term-ready), contracts/dependencies, sole [CPU](../../../../glossary.md#term-cpu) frontier,
frozen HEAD/origin, empty index, preserved 263 accepted dirty paths and inactive conflicts.
[Planning](../../../../glossary.md#term-planning) changed only this brief, CPU master plan and roadmap. The recorded recovery amendment
kept dependencies/base/[identity](../../../../glossary.md#term-identity-meanings) unchanged. [Checkpoint](../../../../glossary.md#term-checkpoint)/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and all successors remain held.

## Files and symbols

Production paths below are relative to `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`:

- `codegen/emit/CpuGeneralScanEmitter.java`: `Shaped.emit` [carry](../../../../glossary.md#term-carry) setup/transition, `loops`/`loop`,
  `load`/`offset`/`denseAddress`, `copyDense`; [coordinate](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)-local kind and [widening](../../../../glossary.md#term-widening) must be explicit.
- `codegen/emit/CpuGeneralScanReductionEmitter.java`: coordinate copies and restored [epilogue](../../../../glossary.md#term-epilogue) [axes](../../../../glossary.md#term-axis);
  `CpuGeneralScanMatmulEmitter.java` in that directory: M/N coordinate loads only as needed.
- `ir/CpuGeneralScanIr.java`: `bakedSpecializationIdentity`; leave `canonicalStages` geometry-free.
  `cache/CpuKernelSpecialization.java`: both identity streams/structural key;
  existing generator/artifact-store seams for stale-identity tests, not a new cache layer.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [cache glossary](../../../../glossary.md#term-cache))</span>

Focused tests in the [CPU](../../../../glossary.md#term-cpu) test tree: `CpuGeneralScanGeneratedKernelTest`, `CpuGeneralScanTimingTest`,
`CpuGeneralScanCleanJavaOracle`, [lowering](../../../../glossary.md#lowering)/execution/preparation tests, `CpuKernelSpecializationTest`
and affected [artifact](../../../../glossary.md#term-artifact)-store tests. Main uses existing `CpuGeneralScanConformanceTest` and
`EngineCpuGeneralScanIntegrationTest`; no new benchmark subsystem or shared build changes.

## [Identity](../../../../glossary.md#term-identity-meanings) decision

The existing sole versioned baked [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) is `CpuGeneralScanIr.bakedSpecializationIdentity()`.
Advance its `|scan-shaped-v1=` and `|scan-matmul-v1=` discriminants to `v2` for the changed
`Shaped` generator (both staged families consume it), retaining all keyed geometry/resource facts.
`CpuKernelSpecialization.compatibilityBytes()` and `classIdentityBytes()` both append it;
`structuralKey()` [hashes](../../../../glossary.md#term-hash) the latter, so both compatibility streams, [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) names and [cache](../../../../glossary.md#term-cache) keys advance.
Keep global envelope/family 71 and [canonical](../../../../glossary.md#term-canonical-form) `:shaped-v1:`/`:matmul-v1:` projections unchanged.
Keep absent-stage [pointwise](../../../../glossary.md#term-pointwise), fixed and all unrelated keys/bytes unchanged. CPU0033's v1/[F32](../../../../glossary.md#floating-point-precision-and-rounding)
retention was historical acceptance, not a requirement to reuse identities for changed [bytecode](../../../../glossary.md#term-bytecode).
Update affected frozen staged expectations with explicit v1→v2 [provenance](../../../../glossary.md#provenance) and stale-v1 rejection;
do not silently refresh unrelated hashes or require changed staged bytes to remain frozen.

## Acceptance criteria

- Select int induction only for statically proved nonnegative extents ≤ `Integer.MAX_VALUE`.
  Prove endpoint arithmetic for 0, 1, MAX and MAX+1; no overflow at the last increment/compare.
  Keep the constant-empty guard and legal larger-extent long-counter fallback. Widen int coordinates
  with <span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [constant glossary](../../../../glossary.md#term-constant); [guard glossary](../../../../glossary.md#term-guard); [extent glossary](../../../../glossary.md#term-extent); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`i2l` before 64-bit offset/stride products; all addresses, byte cursors/bases and time remain long.
  Adapt every coordinate consumer, including reduction bindings and MATMUL M/N; never <span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [bindings glossary](../../../../glossary.md#term-binding); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>`lload`
  an int local. Conservative local/method-size qualification remains safe. Large-extent proof uses
  generation/structure/qualification tests, not giant allocations or billion-iteration executions.<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>
- Initialize carries before time, then update proved exact emission-backed carries after the step,
  removing their unnecessary first-iteration test from the hot time loop. Generation-time proof must
  preserve both directions, T=0 raw initial carry, T=1, multiple/tied carries, scalar-initializer→dense
  factors, forwarded views, save-before-update previous-state histories and exact saved rounding.
  Keep old/new bank lifetimes and all unproved cases' safe fallback; no generic unsafe alias rewrite.
  State the clean-Java oracle and technical reason for retained fallback/deviation. Code remains
  O(body), with no new helper/dispatch/allocation overhead or unproved FMA/reassociation change.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [scalar glossary](../../../../glossary.md#term-scalar); [views glossary](../../../../glossary.md#term-view-meanings); [rounding glossary](../../../../glossary.md#term-rounding); [alias glossary](../../../../glossary.md#aliasing-and-arena); [oracle glossary](../../../../glossary.md#term-test-oracle); [dispatch glossary](../../../../glossary.md#term-dispatch); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [reassociation glossary](../../../../glossary.md#term-reassociation))</span>
- Focused semantic and structural tests exercise the optimized path and fallback in F32/F64,
  actual Compiler reverse, special/raw/empty domains and reduction/MATMUL coordinate bindings.
  Inspect/verify actual generated Class-Files for induction/widening and carry-branch removal,
  helper/allocation absence and local/method budgets. Existing resource, ABI and isolation checks pass.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [Compiler glossary](../../../../glossary.md#term-compiler); [widening glossary](../../../../glossary.md#term-widening); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Test both v2 compatibility streams and structural keys, old affected artifact rejection through
  actual definition/cache seams, compatible-T reuse, all baked-fact separation, geometry-free canonical
  IR, and unchanged pointwise/fixed/unrelated identities. New hashes describe new generator bytes.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache); [canonical glossary](../../../../glossary.md#term-canonical-form); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [pointwise glossary](../../../../glossary.md#term-pointwise))</span>
- Complete the bounded evidence protocol below; report all results and limitations, not a universal
  performance-parity threshold. Independent review must approve code, identity, oracle/dataflow and docs.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [dataflow glossary](../../../../glossary.md#term-dataflow))</span>

## Bounded new evidence

Use only [F64](../../../../glossary.md#floating-point-precision-and-rounding) shaped `next=h+xStep*w`, T=8, B=4, H=8 and existing
`CpuGeneralScanCleanJavaOracle.broadcastForwardFloat64`, with identical producing sites, values,
final-[carry](../../../../glossary.md#term-carry)/stacked-emission demand and zero workspace. Predeclare **both shared** native input/output
scopes, distinct nonaliasing outputs and identical [scope](../../../../glossary.md#term-scope) policy before [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); no shared/confined asymmetry.
Generate/define the actual new production class in each of exactly 3 valid fresh sequential JVMs, no [CPU](../../../../glossary.md#term-cpu) parallel
campaign. Per [JVM](../../../../glossary.md#term-java-virtual-machine-jvm), 100 [warmup](../../../../glossary.md#term-warmup) and 300 measured iterations, each 10,000 calls **per side**;
alternate paired side order. Retain every one of the 300 paired samples and full-[distribution](../../../../glossary.md#term-distribution) stats
(including early/late windows), exact output checks before/after measurement, normal [C2](../../../../glossary.md#term-c2) [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names)
state evidence for generated entry/[oracle](../../../../glossary.md#term-test-oracle), [JDK](../../../../glossary.md#term-java-development-kit-jdk)/flags, [schema](../../../../glossary.md#term-schema)/[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication) [identities](../../../../glossary.md#term-identity-meanings), class [hash](../../../../glossary.md#term-hash)/bytes
and disassembly. Separate cold generation/load/first entry from warmed [kernel](../../../../glossary.md#kernel) costs; do not infer
[binding](../../../../glossary.md#term-binding) or public [Engine](../../../../glossary.md#term-engine) cost by subtraction. No forced C2, JDK patch, frozen-byte patching,
full [matrix](../../../../glossary.md#term-matrix-meanings)/five-[fork](../../../../glossary.md#term-fork-meanings) campaign, or extra [F32](../../../../glossary.md#floating-point-precision-and-rounding) timing campaign; F32 semantic coverage is required.

Recorded one-time exception: the first launch completed 100 warmup pairs, zero measured pairs;
children 2/3 never launched. Native samples establish optional `jcmd Compiler.codelist`
pipe/[compiler](../../../../glossary.md#term-compiler)-lock [deadlock](../../../../glossary.md#term-deadlock), not kernel slowdown. Failed setup remains preserved at
`/private/tmp/synaptik-cpu0034-FgGCiI/fork-1`; only the verified owned diagnostic received SIGTERM,
then workload/launcher unwound. Only optional snapshots/helper were removed, with no new layer.
[Java Flight Recorder](../../../../glossary.md#term-java-flight-recorder-jfr) ([JFR](../../../../glossary.md#memory-management-atomic-counters-and-complexity)), LogCompilation and clock/block correlation retain the required evidence.
After harness [compile](../../../../glossary.md#compile)/syntax and independent review, one authorized recovery completed exactly
3 valid fresh sequential JVMs in `measurement-v2/measured-fork-1..3`, without altering production,
measured body, counts or fairness. No pilot, result-driven retry, extra matrix or fourth completed measurement.

## [Validation](../../../../glossary.md#term-validation)

Worker gates already passed after reviewed [runtime](../../../../glossary.md#term-runtime) freeze:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScanGeneratedKernelTest' --tests '*CpuGeneralScanLoweringTest' --tests '*CpuGeneralScanExecutionTest' --tests '*CpuGeneralScanPreparationTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc
```

The timing-only diagnostic removal reused successful runtime/[Javadoc](../../../../glossary.md#term-javadoc) evidence. Only invalidated
harness evidence was refreshed: `./gradlew :backends:cpu:compileTestJava`, launcher/analyzer syntax
checks and independent targeted recovery review. No full-suite rerun; production/runtime stayed frozen.
Exact launch, [logs](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), failed setup and all three valid [captures](../../../../glossary.md#term-capture-meanings) are recorded in the Result [handoff](../../../../glossary.md#term-handoff).

Main [checkpoint](../../../../glossary.md#term-checkpoint) ran once on unchanged reviewed runtime while the campaign was stopped, before
harness refresh: [conformance](../../../../glossary.md#term-conformance) 7 tests/1s PASS and [Engine](../../../../glossary.md#term-engine) 17 tests/4s PASS. No duplicate rerun:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'
git diff --check
```

Check changed Markdown links/anchors and untracked-file whitespace. No full repository/native
build or benchmark campaign unless a concrete expanded-[scope](../../../../glossary.md#term-scope) risk is reported and authorized.

## Documentation and review impact

Independent serial Class C and recovery reviews finalized affected Javadocs under general/[API](../../../../glossary.md#term-application-programming-interface-api)
profiles, the [scan](../../../../glossary.md#term-cumulative-scan) explainer's generated-code/[qualification](../../../../glossary.md#term-qualification) sections and the [CPU](../../../../glossary.md#term-cpu) guide's bounded-scan/
entry/[validation](../../../../glossary.md#term-validation) sections. `ARCHITECTURE.md` and `backend-execution.md` record descriptive current
v2 [identity](../../../../glossary.md#term-identity-meanings) only; no architecture rule/decision changed, so no [ADR](../../../../glossary.md#term-architecture-decision-record-adr) was required. Targeted glossary
review found no new term. Local links/anchors and tracked/untracked whitespace passed; main owns closure.

## Result

Status: Complete

Main accepted 2026-10-23 after independent code/docs and recovery APPROVE, zero open findings.
Implemented int/long [coordinates](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) and [widening](../../../../glossary.md#term-widening), reduction/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [bindings](../../../../glossary.md#term-binding), safe post-step [carry](../../../../glossary.md#term-carry) proof/
fallback and staged v2 [identity](../../../../glossary.md#term-identity-meanings). The forwarded-final P1 was reproduced before correction and [regression](../../../../glossary.md#term-regression)-tested.
Changed the five production files above; tests `CpuGeneralScanGeneratedKernelTest`, `CpuGeneralScanTimingTest`,
`CpuKernelSpecializationTest`, `CpuGeneratedKernelArtifactStoreTest`; new `general-scan-v1-{shaped,matmul}.class.gz.b64`
fixtures; the four reviewed docs above and three [planning](../../../../glossary.md#term-planning) files. No new production type/[API](../../../../glossary.md#term-application-programming-interface-api)/capability.
[Validation](../../../../glossary.md#term-validation): focused 97 PASS; [module](../../../../glossary.md#term-module) 1,169 executed/38 skipped, zero failures/errors; [Javadoc](../../../../glossary.md#term-javadoc) PASS
(94 warnings in unchanged APIs); harness [compile](../../../../glossary.md#compile) PASS; [conformance](../../../../glossary.md#term-conformance) 7/7 and [Engine](../../../../glossary.md#term-engine) 17/17 PASS.
Launch: `CPU0034_REVIEW_APPROVED=yes zsh /private/tmp/synaptik-cpu0034-FgGCiI/run-bounded.zsh`.
All 900 primary pairs and 2,400 [warmup](../../../../glossary.md#term-warmup)/measured blocks independently audited; outputs matched before/after.

| [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) | Generated [median](../../../../glossary.md#term-median) ns | Java median ns | Ratio of medians |
| --- | ---: | ---: | ---: |
| 1 | 372.82085 | 384.18960 | 0.97041 |
| 2 | 378.16250 | 382.00210 | 0.98995 |
| 3 | 378.45830 | 390.43330 | 0.96933 |

Exact normal-entry [C2](../../../../glossary.md#term-c2) confirmed; no observed [target](../../../../glossary.md#term-target) deopt/invalidation. Strict +/-100 ms guards qualify
288/287/287 pairs; final 12/13/13 lack recorded tail, not excluded from primary statistics. One [F64](../../../../glossary.md#floating-point-precision-and-rounding)/local
workload, not universal parity or a matched v1 speedup claim; no per-output bit dump, paired byte checks passed.
Full files/commands/[hashes](../../../../glossary.md#term-hash)/raw paths/limits: `/private/tmp/synaptik-cpu0034-FgGCiI/FINAL-VALIDATION-HANDOFF.txt`.
All 263 initial dirty paths preserved, only 18 scoped changes; HEAD/origin/index unchanged. No [commit](../../../../glossary.md#term-commit)/push/successor.
