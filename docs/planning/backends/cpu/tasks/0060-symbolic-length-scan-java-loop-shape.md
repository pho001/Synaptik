# Task CPU0060: Symbolic-length Scan Java loop shape

## Reader context

“Loop shape” here means generated control flow and address placement, not a Tensor's dimensions.
`CpuGeneralScanEmitter.Shaped` arranges initialization, activity guards and temporal bindings;
the MATMUL and reduction subemitters named below generated the historical arithmetic loops.
Hoisting an address is legal only while its selected carrier and base remain stable. Skipping
body work at zero maximum length must still publish initialized carries and required padding.
Neither action transfers ownership of the run's buffers into the prepared recipe.

The [current general Scan overview](../../../../backend-guide/cpu-backend.md#current-general-scan-composition)
separates later support from this combined-mode repair; [common MATMUL binding](../../../../backend-guide/cpu-backend.md#common-rank-two-matmul-binding)
explains today's division between temporal bindings and common arithmetic generation.
The v7 measurements below remain entry-inclusive observations of their fixed witness.

## Status

Complete — MAIN accepted the bounded implementation, independent code/docs and actual-result
reviews, owning/named gates and sole diagnostic on 2026-10-28. No successor or further measurement is authorized.

## Change class

Class C — generated hot-loop/[dataflow](../../../../glossary.md#term-dataflow) invariants and baked-[artifact](../../../../glossary.md#term-artifact) compatibility. Clean executor
and independent clean targeted code/documentation review are mandatory; MAIN owns integration.

## Goal

Make combined symbolic-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) plus length-gated Scan follow optimal equivalent Java loop shape,
without changing arithmetic or semantics. Cover [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) ([F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding)), forward and actual [Compiler](../../../../glossary.md#term-compiler) reverse,
both source traversals. Derive generic proofs, never hardcode the reference's F=3/H=2 geometry.

## [Scope](../../../../glossary.md#term-scope) and acceptance criteria

1. Flatten scalar initialization/fills to byte-cursor loops when a constant dense domain is proved;
   omit constant-true guards. Preserve raw scalar bits and nonempty shared initialization at T=0/B=0.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [constant glossary](../../../../glossary.md#term-constant))</span>
2. Exit the invocation at maxLength=0 only after required raw initialization and padding fills;
   use the existing EMPTY publication/init path, not a blind return; publish every demanded carry/output.
   Include bank-backed and non-own-final-port carries.
   Do not enter body work or dereference body-only closed scalar/storage on empty/inactive paths.<span data-glossary-navigation="true"> ([padding glossary](../../../../glossary.md#padding); [publication glossary](../../../../glossary.md#term-publication); [carry glossary](../../../../glossary.md#term-carry))</span>
3. Hoist pointwise row/time-invariant affine address components to the proper enclosing loop,
   not each column. Prove dominance, selected stride/origin stability and branch/iteration lifetime;
   do not extend cell-offset caching across mutable bindings or move guarded loads outside activity.<span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [invariant glossary](../../../../glossary.md#term-invariant); [stride glossary](../../../../glossary.md#term-stride); [lifetime glossary](../../../../glossary.md#term-lifetime); [mutable glossary](../../../../glossary.md#term-mutable); [bindings glossary](../../../../glossary.md#term-binding))</span>
4. For proved length-gated ROW(1) @ ROW(0) contractions, emit direct affine per-active-row addresses
   matching E, rather than updating two operand cursors on inactive rows. Keep increasing-row
   contributors, typed zero/product/ADD sites and the distinct per-time shared-carry ADD unchanged.
   Retain the non-row-contiguous cursor path and generic fallback when the direct-address proof fails.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous))</span>
5. For B=1 SUM-to-Shape raw-singleton cases, remove the one-contributor loop and redundant length
   checks only when the singleton and enclosing active-time/any-active proofs establish that row's
   activity. Preserve fused producing/saved sites and epilogues; B>1 with one active row is NOT raw SUM.<span data-glossary-navigation="true"> ([SUM-to-Shape glossary](../../../../glossary.md#sum-to-shape); [singleton glossary](../../../../glossary.md#term-singleton); [fused glossary](../../../../glossary.md#specialization-and-fusion))</span>
6. Remove provably dead setup only where touched. Optionally derive known 8*B/12*B byte facts from
   runtime B if existing numeric coefficients prove it straightforwardly; no pack change or baked B.
   If this needs a separate mechanism or broader rewrite, flag it without implementing it.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>
7. Advance combined baked v6→v7 append-only through both specialization streams. Reject stale
   combined v1..v6 artifacts; reuse valid v7. Preserve canonical <span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [artifacts glossary](../../../../glossary.md#term-artifact); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>`:symbolic-length-v1`, numeric pack
   `typed-explicit-b-v1`, [schema](../../../../glossary.md#term-schema)/family 71, and ALL noncombined [identities](../../../../glossary.md#term-identity-meanings)/generated bytes exactly.

## Non-goals

No broad cleanup, [fusion](../../../../glossary.md#term-fusion)/stage reorder, per-B [specialization](../../../../glossary.md#term-specialization), new production type/[API](../../../../glossary.md#term-application-programming-interface-api), [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare)
[handoff](../../../../glossary.md#term-handoff), native change, arithmetic/resource/[lifetime](../../../../glossary.md#term-lifetime)/bank change, capability expansion or [Engine](../../../../glossary.md#term-engine) [memo](../../../../glossary.md#cache-and-memoization).
No historical [fixture](../../../../glossary.md#term-fixture)/[oracle](../../../../glossary.md#term-test-oracle)/protocol/result edits, [commit](../../../../glossary.md#term-commit), push, [rollback](../../../../glossary.md#term-rollback) or unrelated dirty edits.

## Contracts

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>.
- [Backend](../../../../glossary.md#backend) execution: [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
  and [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling).
- Recurrent Scan: [Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic),
  [Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>,
  [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates)<span data-glossary-navigation="true"> ([Qualification glossary](../../../../glossary.md#term-qualification))</span>.

Stop on missing/ambiguous contracts or an architectural conflict; do not invent an exception.

## Dependencies and integration

- Depends on: CPU0058/CPU0059 Complete, independently accepted with 0 P1/P2; existing Prepare0015 unchanged.
- Conflicts with: all writers to affected source, identities/formats, tests/evidence or benchmarks;
  serialize source, review and shared-document writes. No parallel writer or benchmark runner.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span>
- Parallel group: None; serial work, no concurrent-write worktree required.
- Common base revision: HEAD=origin/main `07a281edd3086efde2853211b71b48288db027e1`, empty index,
  plus 2501 accepted dirty-tree files in `/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0060-request.json`.
  MAIN fresh `cpu0059-complete` audit PASS, changes=[]; HEAD/origin/index unchanged (2026-10-28).
- Integration order: planner → MAIN Ready/frontier verification → clean executor → independent
  reviewer → MAIN closure. MAIN synchronized master/roadmap before executor launch.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Integration validation: one final CPU module/Javadoc gate plus MAIN's named conformance/Engine
  checkpoint below; reuse successful evidence outside changed branches, no full-root validation.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [CPU glossary](../../../../glossary.md#term-cpu); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Shared-document integration owner: MAIN; root/owning contract/CPU-guide factual v7 maintenance
  lands in the same overall change under exclusive serial handoff. Planner edits this brief only.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [handoff glossary](../../../../glossary.md#term-handoff))</span>

MAIN launch verification: CPU0058/0059 and Prepare0015 are accepted Complete, including independent
reviews and named gates. Fresh request-base audit passed with HEAD/origin/index unchanged; only this
brief and MAIN's synchronized plans differ. No competing Java/build/writer/benchmark is active.
Historical protection captured 3524 files and existing 4136/2636 checks passed. CPU0060 is the sole
source/test frontier; CPU0048–0052 and all other source/authority frontiers remain held.

## Files and symbols

Paths use `modules/backend/cpu/src/{main,test}/java/io/github/pho001/synaptik/backend/cpu/`.

- `main/.../internal/codegen/emit/CpuGeneralScanEmitter.java`: `Shaped.emitLengthAware`, `copyInitial`,
  `activeLoops`/`lengthLoop`, `emitArithmetic`, `load`/`store`/`offset`; bounded init/exit/address proofs.
- Same emitter package: `CpuGeneralScanMatmulEmitter.emit` and
  `CpuGeneralScanReductionEmitter.emitSymbolic`/`emitSymbolicFold`; active [contraction](../../../../glossary.md#contraction-dimension)/[singleton](../../../../glossary.md#term-singleton) paths.
- `main/.../internal/ir/CpuGeneralScanIr.bakedSpecializationIdentity` and
  `main/.../internal/cache/CpuKernelSpecialization.{compatibilityBytes,classIdentityBytes}`: v7/[Javadoc](../../../../glossary.md#term-javadoc).
- `test/.../internal/codegen/emit/CpuSymbolicScanGeneratedKernelTest` and focused general-Scan tests;
  `test/.../internal/cache/CpuKernelSpecializationTest`, `CpuGeneratedKernelArtifactStoreTest`.
- Read-only E: `test/.../internal/codegen/emit/CpuSymbolicLengthScanEquivalentJavaOracle.java`;
  keep it and `CpuSymbolicLengthScanFixture` unchanged. Add cases in existing focused test [owners](../../../../glossary.md#term-owner).

Current reference: `/private/tmp/synaptik-scan-code-comparison.7TV1H1/decompiled/fresh-G6.java` plus E,
not CPU0057's historical G5 report. G6 [SHA-256](../../../../glossary.md#term-sha-256):
`8f2ca20dd90ea82caf37f0aa17af503cbd54d8a035972afaff7f64807c588883`.
Static G6/E: 940/660 instructions, 2126/1406 code bytes, 293/52 locals; no [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) cause proved.

## [Validation](../../../../glossary.md#term-validation)

Before implementation, [capture](../../../../glossary.md#term-capture-meanings) a fresh non-skipped baseline of all 20 noncombined test cases'
[identities](../../../../glossary.md#term-identity-meanings)/ClassFiles; after changes require exact equality, not missing-baseline skips. Use workroot
`/private/tmp/synaptik-cpu0060.uNRs5H`. Preserve historical 4136/2636 checks, [immutable](../../../../glossary.md#term-immutable) roots/results and
G6/E pins; MAIN's workroot `preserve-evidence.mjs` protects CPU0058, both CPU0059 roots and comparison
(3524 files captured before edits). CPU0059's 3913 [runtime](../../../../glossary.md#term-runtime)/build-input [manifest](../../../../glossary.md#term-manifest)
legitimately changes: authenticate authorized new build origins, not unchanged 3913 after implementation.

Before tests, audit redirection for all seven legacy output roots. The immutable CPU0058 agent jar
may be reused READ-ONLY: `EvidenceRedirect.java` dynamically consumes generalScan.evidenceRoot.
Create a new init pointing that property to CPU0060 `current-evidence`/`disabled-attention` destinations;
never use the old init/property writing CPU0058. No new Java agent [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) is needed. Reuse unchanged
`/private/tmp/synaptik-cpu-repairs.8TWk5O/cpu0047-module-filter.gradle` (only two unconditional timing
methods excluded). MAIN's output-safe runner is `/private/tmp/synaptik-cpu0060.uNRs5H/run-validation.sh`,
[SHA-256](../../../../glossary.md#term-sha-256) `ed0d26b3fccfed586e1a799a1599c0b80c95d1b815fa5ad94dbf8ccacad39b99`; its new
`validation-output.gradle` init has SHA-256 `2c294ed5aff962622f381cdab1342765be5ed3596e91fa985f749350ae914cfe`.
Invoke via `env -i PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin HOME=/Users/phujka USER=phujka TMPDIR=/private/tmp /bin/bash <runner>`.
Use [Gradle](../../../../glossary.md#term-gradle) 9.6.1/[JDK](../../../../glossary.md#term-java-development-kit-jdk) 26.0.1; clear inherited option injection/performance/diagnostic/[C2](../../../../glossary.md#term-c2) selectors.
All opt-in campaigns disabled. With audited options, [run](../../../../glossary.md#term-run) focused tests, then ONE final [module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc)
run after final edits, then MAIN's named [checkpoint](../../../../glossary.md#term-checkpoint) (review reuses successful evidence):

```bash
gradle :backends:cpu:test --tests '*CpuSymbolicScanGeneratedKernelTest' --tests '*CpuGeneralScanGeneratedKernelTest' --tests '*CpuGeneralScanExecutionTest' --tests '*CpuGeneralScanPreparationTest' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest'
gradle :backends:cpu:test :backends:cpu:javadoc
gradle :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'
git diff --check
```

Extend focused [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) forward/actual-reverse tests for both traversals, T=0/B=0/all-zero lengths,
B=1 raw special values, raw [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), exact saved sites, poisoned active/[padding](../../../../glossary.md#padding) boundaries, closed body
[scalars](../../../../glossary.md#term-scalar), bank/fallback/[stride](../../../../glossary.md#term-stride) counterexamples and changed operands across reusable runs. Inspect
ClassFiles/[decompilation](../../../../glossary.md#term-decompilation) for all five fixes, verifier dominance, dead setup, checked access, no hidden
helpers/avoidable allocation/boxing/reflection/[dispatch](../../../../glossary.md#term-dispatch), typed [ABI](../../../../glossary.md#term-application-binary-interface-abi) and unchanged pack/[cache](../../../../glossary.md#term-cache) boundaries.

Exactly ONE fresh MEASURED [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) after source freeze/module gate: before-G6/after-G7/E in both
CPU0059 mechanisms, B7/B32, 100 [warmup](../../../../glossary.md#term-warmup) + 300 measured rounds, 100 calls/action/round: 480000 calls.
One attempt, 120-second deadline; raw prechecks run inside it, no standalone [preflight](../../../../glossary.md#term-preflight) JVM. MAIN
freezes protocol/order/commands and obtains independent review before launch. This distinct CPU0060
normal-fix validation grant does not resume old grants. Reuse CPU0059 mechanics outside old roots,
with three-way ABI verification and latest raw checks BEFORE overwrite/rerun. No ABI-shape change,
[Engine](../../../../glossary.md#term-engine)/full [forks](../../../../glossary.md#term-fork-meanings), retries, extra [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) campaign or measurement cascade. Retain all [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)/limits:
diagnostic evidence only, no parity proof, timing-based selection policy or JIT-cause claim.

## Documentation and review impact

Update affected implementation/[identity](../../../../glossary.md#term-identity-meanings) Javadocs and narrow root/[backend](../../../../glossary.md#backend)-contract/[CPU](../../../../glossary.md#term-cpu)-guide current
v7 explanations in the same change; no new architecture decision/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) intended. Reviewer uses actual
diff/contracts and documentation profiles, validates links/anchors and targeted glossary impact,
and reuses successful tests unless behavior changes or a concrete risk warrants rerun. Generate final
[Javadoc](../../../../glossary.md#term-javadoc) after final edits; record reviewed unchanged Javadocs explicitly.

## Result

All five bounded fixes are implemented for combined [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) forward and actual reverse, both
traversals: flat raw [scalar](../../../../glossary.md#term-scalar) fills; initialized EMPTY [publication](../../../../glossary.md#term-publication) at maxLength=0; [binding](../../../../glossary.md#term-binding)-scoped
[pointwise](../../../../glossary.md#term-pointwise) address prefixes with full same-cell reuse; proved active-row affine contractions;
and direct raw-[singleton](../../../../glossary.md#term-singleton) [SUM-to-Shape](../../../../glossary.md#sum-to-shape) with producing stores/epilogues retained. Other [matrix](../../../../glossary.md#term-matrix-meanings) paths
keep their cursors. Both baked streams advance to v7; actual v6 [artifacts](../../../../glossary.md#term-artifact) reject. The 20 noncombined
[snapshot](../../../../glossary.md#term-snapshot) cases ran without skips and retained both [identities](../../../../glossary.md#term-identity-meanings) and complete generated bytes.

The twelve-file source/test/doc delta is retained in `/private/tmp/synaptik-cpu0060.uNRs5H/task-only.diff`
and `HANDOFF.txt`; no [fixture](../../../../glossary.md#term-fixture)/[oracle](../../../../glossary.md#term-test-oracle), [API](../../../../glossary.md#term-application-programming-interface-api), arithmetic, pack, resource or [lifecycle](../../../../glossary.md#lifecycle) changes occurred.
Actual reference ClassFile G6→G7: 940→910 instructions, 2126→2086 code bytes, 293→300 local slots.
These are static facts, not executed/native instructions, allocation, spilling or speed evidence.

[Validation](../../../../glossary.md#term-validation): the named focused command passed 189 tests; the added cursor-fallback test passed.
The sole final [CPU](../../../../glossary.md#term-cpu) test/[Javadoc](../../../../glossary.md#term-javadoc) command passed in 5m52s: 1310 executed, 51 skipped, zero failures/errors;
the unchanged two-method timing filter and all disabled-test details are in `final-validation.json`.
Javadoc succeeded with 94 existing warnings, none on changed production files. MAIN's named
[conformance](../../../../glossary.md#term-conformance)/[Engine](../../../../glossary.md#term-engine) command passed 12+22 tests with zero skips/failures/errors in 12s.

The independently released sole [JVM](../../../../glossary.md#term-java-virtual-machine-jvm) completed 4800 [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)/480000 scheduled calls plus 36 reverse
and 24 forward untimed raw checks; no malformed, partial, unchecked or schedule-error rows.
Same-round [median](../../../../glossary.md#term-median) elapsed ratios (300 pairs each): B7 standard G7/G6=0.9658, G7/E=1.1196;
B7 typed 0.9687/1.1073; B32 standard 0.9716/1.1657; B32 typed 0.9758/1.1562.
Thus this entry-inclusive [run](../../../../glossary.md#term-run) observed 2.4–3.4% lower elapsed time than G6, still 10.7–16.6% above E.
[Kernel](../../../../glossary.md#kernel) [C2](../../../../glossary.md#term-c2) completions precede measured B7, but timed wrappers reach C2 during measured B32;
no uniformly settled full path, parity, general speed or [JIT](../../../../glossary.md#term-just-in-time-compilation-jit)-cause conclusion follows. All rows remain.
Protocol, approvals, exact commands, raw outputs/JIT [log](../../../../glossary.md#term-log) and `analysis-results.json` are retained in
`/private/tmp/synaptik-cpu0060.uNRs5H/measurement.D6is3Y`; the one slot is consumed, no retry.

Independent Ohm code/docs, compiled-launcher and actual-result reviews approve with no open P1/P2.
Root/[backend](../../../../glossary.md#backend)-contract/CPU-guide v7 facts and affected Javadocs were finalized together; other
affected Javadocs remain accurate. General, architecture, backend-guide, API/Javadoc and [planning](../../../../glossary.md#term-planning)
profiles applied; targeted glossary review required no new term. No [ADR](../../../../glossary.md#term-architecture-decision-record-adr)/authority change was needed.
Historical 3524/4136/2636-file checks and `git diff --check` pass; HEAD/origin/index and unrelated dirty
files remain unchanged. Optional byte-fact derivation and broader dead setup, including the redundant
dominated scalar-[cache](../../../../glossary.md#term-cache) [guard](../../../../glossary.md#term-guard), are deferred; the [residual](../../../../glossary.md#term-residual-meanings) Java gap remains open, not this task's parity claim.

Status: Complete
