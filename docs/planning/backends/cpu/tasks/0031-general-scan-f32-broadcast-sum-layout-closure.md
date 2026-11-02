# Task CPU0031: F32 general-scan broadcast, SUM and layout closure

## Status

Complete — main accepted the bounded capability on 2026-10-22, 14:38+ UTC, after final
independent code/docs APPROVE (0 P1/P2), worker module validation and the named checkpoint PASS.
CPU frontier: None; other area frontiers None. Mandatory MATMUL/F64 successor gates remain Draft.

## Change class

Class C — generated hot-loop/dataflow, temporal carry ownership and exact pre-slot resources.

## Goal

Execute homogeneous FLOAT32 shape-changing scan algebra and its actual Compiler-generated
first-order reverse: body proof → resource/liveness plan → one emitted time loop.
Close broadcast parameter gradients without assuming every carry or body value has one common N.

## Scope

- Static T≥0, all-valid axis-zero sequences; ≥1 carry/sequence, ≥0 invariants/emissions.
  Each carry has its own fixed shape; each sequence its own fixed step shape and common T.
  Local ranks 0..16, stacked rank up to 17; checked element/byte geometry, including zeros.
  Canonical outer boundaries/formals, plus CPU0030's proved all-zero-stride read bindings for
  Compiler splats/initializers; no general outer-stride/offset expansion. Heap/native/mixed
  inputs remain checked MemorySegment bindings, not new entry carrier strategies.
- Exact closed algebra: tensor/scalar ADD/SUB/MUL/DIV, NEG/TANH/SIGMOID;
  right-aligned Model broadcasting; SUM with NoOperationAttrs, AxisReductionAttrs,
  MultiAxisReductionAttrs or SumToShapeAttrs (keep-dimensions true/false where applicable);
  EXPAND/RESHAPE with TargetShapeAttrs, EXPAND_DIMS/SQUEEZE with AxisTransformAttrs,
  rank-two PERMUTE `[1,0]`, and CONTIGUOUS. Only valid existing Model guards/layout inference.
  RESHAPE requires proved reshape-compatible storage or an explicit CONTIGUOUS input, not
  an implicit reorder. Derived zero-stride/permuted local views are not canonical outer storage.
- Qualify forward and actual reverse independently, optimized/unoptimized, using existing
  ScanGradientRules and ordinary rule owners. Reverse closure includes positional ADD accumulation,
  broadcast SUM-to-Shape, restoration EXPAND_DIMS/EXPAND/CONTIGUOUS, inverse views and closed
  typed zero/one invariant splats. Homogeneity needs no cross-dtype CAST; same-type casts are
  identity only if actually retained by Model/capture. Reject every type-changing CAST.

## Non-goals

MATMUL (including its reverse), FLOAT64/low/mixed types, new AD/rules, Compiler production changes,
padding/lengths, NN/persistence admission, Engine/Runtime/shared-Prepare changes, native/vector/
parallel scan routes, tuning, arbitrary regions/control flow, or a broad performance campaign.
Retain fixed four-dtype/layout routes and CPU0030's admitted pointwise domain without regression.

## Contracts

- Root [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules),
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan),
  ALL subordinate headings — source/scopes, canonical residual demand, exact primal binding,
  traversal/empties, qualification/performance/migration gates.
- Backend [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules),
  [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes),
  [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling),
  [Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity).
- Lifecycle [`modules/runtime`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
  [`modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle),
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution).
- Foundational [Profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries)
  and [Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics):
  Family formulas and unary sites, Arithmetic-site DAZ/FTZ, Exact and discrete floor,
  Floating arithmetic primitives, Aggregate floor, Composite inheritance.
- Compiler [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
  only for actual generated forms/primal/constants; no new AD ownership or algebra.

Missing/ambiguous applicable authority or need for a new boundary: stop and ask main.

## Dependencies and integration

- Depends on: CPU0030; its accepted Model0034/0035, Planning0008, Compiler0011, Prepare0010
  and CPU0022/0027/0028/0029 baseline, all Complete per current area/program status and main's handoff.
- Conflicts with: general-scan admission/lowering/IR/emission/binding/resources, generated/cache
  identity, numerical policy/current ledgers, shared planning/explanations; serialize these scopes.
- Parallel group: None; one writer. Read-only review may overlap; no concurrent write scope.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`, plus
  248 accepted dirty paths through final CPU0030 and this planning packet; not merged predecessors.
- Integration order: independent plan review → main final acceptance/Ready verification → clean
  CPU executor → independent targeted Class C code/docs review/corrections → main checkpoint.
- Integration validation: worker commands below; main's CPU0031 focused cross-module checkpoint;
  one source-matched native-enabled root `test javadoc` remains at the later low-level program join.
- Shared-document integration owner: main; delegate focused implementation explanations/Javadoc
  exclusively to the serial worker/reviewer, retain shared planning/status integration in main.

Historical main launch verification (2026-10-22): accepted independent plan APPROVE (0 P1/P2),
all dependencies Complete, matching HEAD/origin/main at the common base, empty index, inactive
conflicts and preserved accepted dirty work/source guards. User's “ok, pokracuj” authorized the
bounded successor; CPU0031 was the sole Ready frontier. Seven integration fields remained unchanged.
Preserve THIRD_PARTY_NOTICES.md, CpuOrdinaryPointwiseCastCleanJavaOracle.java and the Superseded
Compiler0010 brief byte-for-byte; no reset/checkout/stash, commit/push, or unrelated cleanup.

## Files and symbols

CPU prefix: `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.
Reuse `CpuCapabilityProvider`, `internal/lowering/CpuGeneralScanLowering` (analyze/lower/Facts/Geometry),
`internal/ir/CpuGeneralScanIr`, `internal/codegen/emit/CpuGeneralScanEmitter`,
`internal/executable/CpuGeneralScanPreparedExecutable`, existing partition/portable/cache seams.
Read `CpuAggregateLowering`/`CpuAggregateIr`/`CpuAggregateEmitter`, `CpuCarrierEmitter`,
`CpuLoopEmitter`/`CpuScalarEmitter`, and `CpuMatmulEmitter` to distinguish reusable code from entry ABI.
Read Compiler `ScanGradientRules`, Elementwise/Layout/Reduction/LinearAlgebraGradientRules and
`ScanGradientRulesTest`/`GeneralScanCompilerTest`; they already demonstrate broadcast/SUM/view
closure and rank-two MATMUL in F32/F64, not CPU execution capability.

Only added top-level production type, now delivered: `CpuGeneralScanReductionEmitter` in existing package
`io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`, for step-bound SUM loops rather than
the ordinary aggregate emitter's fixed two-entry ABI/exact-limb workspace. Reuse scalar/carrier/loop
seams; keep ordinary aggregate routes unchanged. Derived stage/binding/resource facts stay nested
in the four existing scan types; no second semantic body, generic stage executor or new public type.
Any additional production type/package requires main's packet update before implementation.

Tests: extend existing `CpuGeneralScan*` capability/lowering/preparation/generated/execution/oracle/
timing tests, `CpuKernelSpecializationTest`/artifact-store tests and current generated coverage ledger.
Extend `testing/backend-conformance/.../CpuGeneralScanConformanceTest` with Model-only fixtures;
NO Compiler dependency there. Actual AD belongs to CPU tests (existing test-only Compiler edge)
and `testing/integration-tests/.../EngineCpuGeneralScanIntegrationTest`. No build-edge change.

## Acceptance criteria

1. Provider proof validates whole header/local inference/role inventory/demand AND binder,
   including unsupported dead interiors at T=0 or empty coordinates. Preserve current budgets
   (128 nodes/256 local values/64 formals/256 ports, ≤248 boundaries and ≤254 entry slots),
   plus fail-closed generated method/local budgets. A forward true answer never promises reverse.
2. Invocation binding validates dtype/alignment/capacity/complete-capacity overlap once, creates
   run-owned geometry/carrier arrays and hoists invariant layout/stride/constant setup outside time.
   Step binding changes only required carriers/bases/time offsets/carry versions/ranges; typed locals
   and generation-time emitter calls only. No graph, dispatch, allocation, geometry-array creation,
   MethodHandle/callback per op/step or T-fold bytecode. One RunState and one prepared entry remain.
3. Before slots, derive each value's shape/access and use lifetime; each carry's temporal forwarding
   closure and simultaneous old/new versions; exact checked output/live-history bytes and all
   staging, carry banks, body temporaries and reduction scratch with alignment/offset/lifetime.
   Prefer direct sequence/emission/residual offsets and demanded frames as backing. A transpose
   or reduction may read across coordinates: CPU0030's in-place per-coordinate bank is not safe
   by default. Overwrite only after the last old-value consumer; use separately justified banks/
   ping-pong when necessary. Reuse scratch after last use, never a live saved site or aliased role.
   Document why every materialization/copy exists; no blanket per-op buffers or input staging.
4. Previous-carry histories store BEFORE the step, produced histories at exact stored sites, all at
   original time indices. Distinct demanded wrappers remain distinct even for aliased roots.
   Undemanded forward AND reverse residuals allocate/store zero history. Internal temporaries are
   not histories. T-dependent publication bytes remain; O(body) recipe/code excludes requested data.
5. Baked/keyed facts: dtype, ordered body/op/immediate dataflow, normalized axes/keep-dims/target
   mapping, view/access/rank relations, role/alias/demand maps, carry version/storage decisions,
   stage ordering, scratch/entry ABI and every specialized extent/stride. Runtime facts: T,
   compatible extents/stride magnitudes, per-value frame sizes/ranges/addresses, slots/representations
   and run buffers. Key any geometry actually baked; never put slots/IDs/storage into CpuKernelIr.
   Current envelope/family are 71: explicitly advance incompatible identity append-only, reject
   stale 71 artifacts if meaning/ABI changes, and synchronize identity references through main.
   Test independent key mutations, stale rejection and identical-code reuse for compatible geometry.
6. Test the complete shape/op + actual AD matrix: scalar and `[H]` trainable broadcast invariants
   into `[B,H]`, rank growth `[H]`/`[B,1]`, different-shaped multiple carries and emission shapes;
   both traversals, two carries, T=0/1/≥3, both/missing seed classes, false-gradient h0 with trainable
   x/parameters, tied roles/roots, unused inputs and reused templates/cross-occurrence sums.
   Exercise full/single/multi-axis SUM (both keep forms), target-one/leading/equal-shape sumToShape,
   view restoration/transpose/CONTIGUOUS and every declared arithmetic/scalar kind. Compare actual
   emitted reverse to unrolled ordinary Tensor AD and specialized represented-site Java oracles.
7. Empty axes do not skip nonempty work: `[0,H] -> [H]` SUM/parameter gradients produce typed zero
   even with NaN/Inf invariant values; zero selected axes/no-reduction sumToShape preserve bits.
   T=0 touches no body: raw initial final carry, empty histories/emissions, dInitial=dFinal seed,
   zero invariant sums. Include permuted/cross-coordinate carry hazards and forwarding chains.
   Pin NaN/±Inf/±0, zero divisors, overflow/underflow/subnormals; DIV reverse remains ordinary
   unmasked formula, activation reverse consumes saved values, movement stays raw exact. Reduced
   SUM may use a rounded all-contributors-once F32 tree per Model; do not inherit exact-limb
   overhead merely from the ordinary route. Explicit oracle/metric/threshold for finite arithmetic.
8. Overflow/budget/malformed body/unsupported MATMUL/F64/cast/view rejections occur before writes;
   heap/native/mixed, overlap tails, cleanup/failure/retry, open-result fresh runs and concurrent
   prepared reuse retain existing behavior. Structural bytecode tests count real history/bank/
   staging stores and inspect loops/calls/allocation. Keep old pointwise no-extra-copy hot path.
9. Oracle is optimal clean specialized Java for the SAME algorithm, loops/dataflow, dtype/carrier,
   saved sites and work, not a deliberately generic fallback. Class-File/decompilation inspection
   and one short matched timing diagnostic cover a small broadcast reverse and a larger reduction
   case; separate cold generation/load/first-JIT opportunity, entry/preflight and kernel/step costs.
   Repeat the old CPU0030 small mixed-backward timing only if pointwise executable shape changes
   or a concrete regression risk justifies it. Otherwise retain structural old-path regression
   checks and short matched timing of the new broadcast/reduction cases. Transparently retain the
   historical ~7× gap with unproved origin; parity is NOT inherited. Fix or justify concrete
   diagnosed avoidable overhead; report unresolved timing/JIT confounds without speed promise,
   parity/C2 gate, five forks or waiting for C2.

## Validation

Worker validation: focused iterations, then one final affected-module run; evidence in Result.

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScan*' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc
./gradlew :backends:cpu:test --tests '*CpuGeneralScanTimingTest' -Dsynaptik.cpu.generalScan.performance=true -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0031-evidence
```

Main checkpoint once after ordered review/integration; reuse successful worker evidence:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'
./gradlew :testing:architecture-tests:test
git diff --check
```

Root/native validation remains at the low-level join, not a repeated worker gate.

## Follow-up

Mandatory serial gates after reviewed CPU0031: rank-two MATMUL + its actual reverse on the new
bindings, then homogeneous FLOAT64 qualification across the COMPLETE declared algebra/closure.
Reassess cohesive task size at that frontier; no far-future brief or Ready grant now. F64 requires
typed double locals/eight-byte resources, full numerical/saved-site tests and NO DAZ/FTZ.

## Documentation and review impact

Planning profile applies. Affected Javadocs, targeted CPU/backend/scan explanations and current
ledger evidence are finalized; main owns shared status integration.
No architecture decision/edge or new glossary meaning changed. Targeted terms: general scan,
formal/primal binding, residual/history demand, cotangent normalization and sum-to-Shape.
Independent plan and final Class C code/docs reviews accepted; successful executable evidence reused.
Closure changes planning metadata only; no further explanatory/Javadoc/glossary update is needed.

## Result

Main accepted 2026-10-22, 14:38+ UTC: bounded F32 broadcast/SUM/layout forward + actual reverse Complete.
Files: scan IR/lowering/emitter/executable seams, new CpuGeneralScanReductionEmitter, existing specialization,
scan/cache/oracle/timing tests, CPU build opt-ins, conformance/Engine fixtures, ledger v2 and focused docs/Javadocs.
Rawls final code APPROVE (0 P1/P2), R1–R4 closed/manifests verified; Herschel independent docs APPROVE (0),
preserving Rawls's 12 Markdown/5 Java files and finalizing five docs; six Java files executable-token-neutral,
30 normalized compiled-class dumps identical; 83 targeted links/anchors PASS. No new API/architecture/edge/lifecycle/glossary meaning.
Geometry-free canonical CpuKernelIr and global/family 71 retained; conditional typed scan-shaped-v1 payload
in BOTH existing specialization byte projections; old 12 actual ClassFiles unchanged. No current generated
inventory refresh; v1 ledger immutable, v2 adds two actual query rows. Fixed four-type/layout routes unchanged.
Final `./gradlew :backends:cpu:test :backends:cpu:javadoc` PASS (5m27s): 216 suites, 1148 tests,
1115 passed/33 skips/0 failures/errors; fresh XML 14:07:54–14:13:21 UTC; later comments proved neutral.
Javadoc PASS, 94 baseline warnings (untouched compatibility-constructor parameters/incubator), not zero-warning.
Focused: 58 tests/2 skips/0 failures, including 72 actual AD/unrolled variants and 36 R4 regressions;
40 vs 52 gets, 32 B/16 F32 stores and unused 0 B; R3 LMUL 34→11. Safe-cell fusion retains necessary full-stage cuts.
Main checkpoint: `./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest' --console=plain` PASS (1s), 2 suites/10 tests, 14:37:51 UTC;
`./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest' --console=plain` PASS (3s), 1 suite/7 tests, 14:36:33 UTC;
`./gradlew :testing:architecture-tests:test --console=plain` PASS (2s), 11 suites/25 tests, 14:36:58–59 UTC: all 42 pass, no skips/failures.
Only conformance rerun: stale expected F32 seed count 118→119 for new seed 130-SCAN/broadcast-sum-layout;
all independent BF16/F16 negatives retained; v1/v2/production unchanged. Final source freeze verified; no duplicate CPU run.
Matched timing: two new cases, one JVM/five samples, `/private/tmp/synaptik-cpu0031-r3-evidence/summary.txt`;
T8B2H8 generated 26.2–27.8 µs vs Java 22.8–24.7 µs; T8B64H128 0.357–1.890 ms vs 119–132 µs.
Transition/gap origin unproved; historical CPU0030 ~7× mixed gap retained, no parity/C2 guarantee or new performance gate.
Java/ClassFile oracle algorithms, time→H→B loops, carries and load/store counts match; no hidden helpers/graph/dispatch/allocation.
Follow-up: serial rank-two MATMUL + actual F32 reverse Draft → full homogeneous F64 algebra/closure Draft → root/native low-level join.
No full F32-domain, NN/padding/training/persistence/low/vector/parallel grant. No worker remains; no successor launched.
Status: Complete

Size exception: acceptance criteria and the compact final evidence remain together above 200 lines/15 KB;
splitting this bounded generated-code/resource capability would separate its validation from its acceptance.
