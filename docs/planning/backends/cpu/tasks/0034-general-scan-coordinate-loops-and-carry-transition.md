# Task CPU0034: General-scan coordinate loops and carry transition

## Status

Complete — main accepted the independently reviewed implementation, runtime gates and bounded evidence on 2026-10-23.

## Change class

Class C — generated hot-loop/dataflow changes and generated-artifact cache identity.
Separate clean planning, implementation, and independent code/documentation-review contexts;
all writes serial. Main verifies the frozen baseline before launching the executor.

## Goal

Make proved small-static-extent `Shaped` coordinate loops follow optimal clean Java induction,
and move proved emission-backed carry selection to a post-step transition. Preserve the complete
declared F32/F64 forward and actual Compiler-generated reverse domain, numerics and ownership.
The intended sequence is initial binding → step/read/save/write → next-carry binding.

## Scope and non-goals

Use only existing CPU scan emission, derived IR and specialization/cache seams. No added or moved
top-level production type, package, public API, backend, dependency, AD or execution layer.
Preserve capability, predeclared resources, entry application binary interface (ABI), temporal
lifetimes and run isolation. No other families, fixed-scan redirection, low types, vector,
parallel, OpenBLAS, NN or Checkpoint advancement. No global `CpuLoopEmitter` refactor; MATMUL K
and other independent loops stay unchanged unless a concrete coordinate-binding proof requires it.

## Contracts

- [Root](../../../../../ARCHITECTURE.md): **Authority, incorporation, and precedence**;
  **Core invariants**; **Scope-indexed normative contracts**; **Dependency rules**.
- [Backend execution](../../../../architecture/contracts/backend-execution.md): **CPU backend routes**;
  **Performance evidence and optimization tooling**; **Profile-free backend capability and identity**.
- [Scan owner](../../../../architecture/contracts/recurrent-scan.md): **General differentiable scan**;
  **Canonical residual ports and demand**; **Reverse scan and exact primal bindings**;
  **Traversal, empty domains and declared qualification**; **Qualification, performance and migration gates**.

These plans add no architecture authority. Stop on missing/ambiguous contracts, incompatible
architecture, or required scope expansion; do not invent a layer or numerical permission.

## Dependencies and integration

- Depends on: CPU0033 and low-level native join Complete; Compiler0012 Complete as accepted
  compatibility baseline (including their already accepted predecessors/reviews/validation).
- Conflicts with: all CPU general-scan emission/IR/cache identity, qualification/resources,
  shared scan explanation/CPU guide/planning writes; none active at packet preparation.
- Parallel group: None; sole serial writer, no concurrent implementation/reviewer writes.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`
  plus all 263 accepted dirty paths, empty index, then this three-file planning packet;
  preserve every accepted change. This is not a committed or merged predecessor snapshot.
- Integration order: main frozen-base/Ready verification → clean CPU executor → separate independent
  Class C code/docs review/corrections → final worker gates → main checkpoint while campaign stopped
  → harness refresh/review → recovery campaign → main closure. Sole serial stream; no worktree/commit/push.
- Integration validation: reuse main's passed conformance/Engine checkpoint under Validation on
  unchanged reviewed runtime; no rerun or full root gate for this harness-only amendment.
- Shared-document integration owner: main; executor may draft affected Javadocs, independent reviewer
  exclusively finalizes targeted explanations/Javadocs while worker pauses, main owns planning closure.

Launch verification (2026-10-23): main checked Ready, contracts/dependencies, sole CPU frontier,
frozen HEAD/origin, empty index, preserved 263 accepted dirty paths and inactive conflicts.
Planning changed only this brief, CPU master plan and roadmap. The recorded recovery amendment
kept dependencies/base/identity unchanged. Checkpoint/NN and all successors remain held.

## Files and symbols

Production paths below are relative to `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/`:

- `codegen/emit/CpuGeneralScanEmitter.java`: `Shaped.emit` carry setup/transition, `loops`/`loop`,
  `load`/`offset`/`denseAddress`, `copyDense`; coordinate-local kind and widening must be explicit.
- `codegen/emit/CpuGeneralScanReductionEmitter.java`: coordinate copies and restored epilogue axes;
  `CpuGeneralScanMatmulEmitter.java` in that directory: M/N coordinate loads only as needed.
- `ir/CpuGeneralScanIr.java`: `bakedSpecializationIdentity`; leave `canonicalStages` geometry-free.
  `cache/CpuKernelSpecialization.java`: both identity streams/structural key;
  existing generator/artifact-store seams for stale-identity tests, not a new cache layer.

Focused tests in the CPU test tree: `CpuGeneralScanGeneratedKernelTest`, `CpuGeneralScanTimingTest`,
`CpuGeneralScanCleanJavaOracle`, lowering/execution/preparation tests, `CpuKernelSpecializationTest`
and affected artifact-store tests. Main uses existing `CpuGeneralScanConformanceTest` and
`EngineCpuGeneralScanIntegrationTest`; no new benchmark subsystem or shared build changes.

## Identity decision

The existing sole versioned baked payload is `CpuGeneralScanIr.bakedSpecializationIdentity()`.
Advance its `|scan-shaped-v1=` and `|scan-matmul-v1=` discriminants to `v2` for the changed
`Shaped` generator (both staged families consume it), retaining all keyed geometry/resource facts.
`CpuKernelSpecialization.compatibilityBytes()` and `classIdentityBytes()` both append it;
`structuralKey()` hashes the latter, so both compatibility streams, binary names and cache keys advance.
Keep global envelope/family 71 and canonical `:shaped-v1:`/`:matmul-v1:` projections unchanged.
Keep absent-stage pointwise, fixed and all unrelated keys/bytes unchanged. CPU0033's v1/F32
retention was historical acceptance, not a requirement to reuse identities for changed bytecode.
Update affected frozen staged expectations with explicit v1→v2 provenance and stale-v1 rejection;
do not silently refresh unrelated hashes or require changed staged bytes to remain frozen.

## Acceptance criteria

- Select int induction only for statically proved nonnegative extents ≤ `Integer.MAX_VALUE`.
  Prove endpoint arithmetic for 0, 1, MAX and MAX+1; no overflow at the last increment/compare.
  Keep the constant-empty guard and legal larger-extent long-counter fallback. Widen int coordinates
  with `i2l` before 64-bit offset/stride products; all addresses, byte cursors/bases and time remain long.
  Adapt every coordinate consumer, including reduction bindings and MATMUL M/N; never `lload`
  an int local. Conservative local/method-size qualification remains safe. Large-extent proof uses
  generation/structure/qualification tests, not giant allocations or billion-iteration executions.
- Initialize carries before time, then update proved exact emission-backed carries after the step,
  removing their unnecessary first-iteration test from the hot time loop. Generation-time proof must
  preserve both directions, T=0 raw initial carry, T=1, multiple/tied carries, scalar-initializer→dense
  factors, forwarded views, save-before-update previous-state histories and exact saved rounding.
  Keep old/new bank lifetimes and all unproved cases' safe fallback; no generic unsafe alias rewrite.
  State the clean-Java oracle and technical reason for retained fallback/deviation. Code remains
  O(body), with no new helper/dispatch/allocation overhead or unproved FMA/reassociation change.
- Focused semantic and structural tests exercise the optimized path and fallback in F32/F64,
  actual Compiler reverse, special/raw/empty domains and reduction/MATMUL coordinate bindings.
  Inspect/verify actual generated Class-Files for induction/widening and carry-branch removal,
  helper/allocation absence and local/method budgets. Existing resource, ABI and isolation checks pass.
- Test both v2 compatibility streams and structural keys, old affected artifact rejection through
  actual definition/cache seams, compatible-T reuse, all baked-fact separation, geometry-free canonical
  IR, and unchanged pointwise/fixed/unrelated identities. New hashes describe new generator bytes.
- Complete the bounded evidence protocol below; report all results and limitations, not a universal
  performance-parity threshold. Independent review must approve code, identity, oracle/dataflow and docs.

## Bounded new evidence

Use only F64 shaped `next=h+xStep*w`, T=8, B=4, H=8 and existing
`CpuGeneralScanCleanJavaOracle.broadcastForwardFloat64`, with identical producing sites, values,
final-carry/stacked-emission demand and zero workspace. Predeclare **both shared** native input/output
scopes, distinct nonaliasing outputs and identical scope policy before samples; no shared/confined asymmetry.
Generate/define the actual new production class in each of exactly 3 valid fresh sequential JVMs, no CPU parallel
campaign. Per JVM, 100 warmup and 300 measured iterations, each 10,000 calls **per side**;
alternate paired side order. Retain every one of the 300 paired samples and full-distribution stats
(including early/late windows), exact output checks before/after measurement, normal C2 compilation
state evidence for generated entry/oracle, JDK/flags, schema/payload identities, class hash/bytes
and disassembly. Separate cold generation/load/first entry from warmed kernel costs; do not infer
binding or public Engine cost by subtraction. No forced C2, JDK patch, frozen-byte patching,
full matrix/five-fork campaign, or extra F32 timing campaign; F32 semantic coverage is required.

Recorded one-time exception: the first launch completed 100 warmup pairs, zero measured pairs;
children 2/3 never launched. Native samples establish optional `jcmd Compiler.codelist`
pipe/compiler-lock deadlock, not kernel slowdown. Failed setup remains preserved at
`/private/tmp/synaptik-cpu0034-FgGCiI/fork-1`; only the verified owned diagnostic received SIGTERM,
then workload/launcher unwound. Only optional snapshots/helper were removed, with no new layer.
Java Flight Recorder (JFR), LogCompilation and clock/block correlation retain the required evidence.
After harness compile/syntax and independent review, one authorized recovery completed exactly
3 valid fresh sequential JVMs in `measurement-v2/measured-fork-1..3`, without altering production,
measured body, counts or fairness. No pilot, result-driven retry, extra matrix or fourth completed measurement.

## Validation

Worker gates already passed after reviewed runtime freeze:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScanGeneratedKernelTest' --tests '*CpuGeneralScanLoweringTest' --tests '*CpuGeneralScanExecutionTest' --tests '*CpuGeneralScanPreparationTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc
```

The timing-only diagnostic removal reused successful runtime/Javadoc evidence. Only invalidated
harness evidence was refreshed: `./gradlew :backends:cpu:compileTestJava`, launcher/analyzer syntax
checks and independent targeted recovery review. No full-suite rerun; production/runtime stayed frozen.
Exact launch, logs, failed setup and all three valid captures are recorded in the Result handoff.

Main checkpoint ran once on unchanged reviewed runtime while the campaign was stopped, before
harness refresh: conformance 7 tests/1s PASS and Engine 17 tests/4s PASS. No duplicate rerun:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'
git diff --check
```

Check changed Markdown links/anchors and untracked-file whitespace. No full repository/native
build or benchmark campaign unless a concrete expanded-scope risk is reported and authorized.

## Documentation and review impact

Independent serial Class C and recovery reviews finalized affected Javadocs under general/API
profiles, the scan explainer's generated-code/qualification sections and the CPU guide's bounded-scan/
entry/validation sections. `ARCHITECTURE.md` and `backend-execution.md` record descriptive current
v2 identity only; no architecture rule/decision changed, so no ADR was required. Targeted glossary
review found no new term. Local links/anchors and tracked/untracked whitespace passed; main owns closure.

## Result

Status: Complete

Main accepted 2026-10-23 after independent code/docs and recovery APPROVE, zero open findings.
Implemented int/long coordinates and widening, reduction/MATMUL bindings, safe post-step carry proof/
fallback and staged v2 identity. The forwarded-final P1 was reproduced before correction and regression-tested.
Changed the five production files above; tests `CpuGeneralScanGeneratedKernelTest`, `CpuGeneralScanTimingTest`,
`CpuKernelSpecializationTest`, `CpuGeneratedKernelArtifactStoreTest`; new `general-scan-v1-{shaped,matmul}.class.gz.b64`
fixtures; the four reviewed docs above and three planning files. No new production type/API/capability.
Validation: focused 97 PASS; module 1,169 executed/38 skipped, zero failures/errors; Javadoc PASS
(94 warnings in unchanged APIs); harness compile PASS; conformance 7/7 and Engine 17/17 PASS.
Launch: `CPU0034_REVIEW_APPROVED=yes zsh /private/tmp/synaptik-cpu0034-FgGCiI/run-bounded.zsh`.
All 900 primary pairs and 2,400 warmup/measured blocks independently audited; outputs matched before/after.

| JVM | Generated median ns | Java median ns | Ratio of medians |
| --- | ---: | ---: | ---: |
| 1 | 372.82085 | 384.18960 | 0.97041 |
| 2 | 378.16250 | 382.00210 | 0.98995 |
| 3 | 378.45830 | 390.43330 | 0.96933 |

Exact normal-entry C2 confirmed; no observed target deopt/invalidation. Strict +/-100 ms guards qualify
288/287/287 pairs; final 12/13/13 lack recorded tail, not excluded from primary statistics. One F64/local
workload, not universal parity or a matched v1 speedup claim; no per-output bit dump, paired byte checks passed.
Full files/commands/hashes/raw paths/limits: `/private/tmp/synaptik-cpu0034-FgGCiI/FINAL-VALIDATION-HANDOFF.txt`.
All 263 initial dirty paths preserved, only 18 scoped changes; HEAD/origin/index unchanged. No commit/push/successor.
