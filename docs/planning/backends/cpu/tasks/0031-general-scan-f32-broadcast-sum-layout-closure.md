# Task CPU0031: [F32](../../../../glossary.md#floating-point-precision-and-rounding) general-[scan](../../../../glossary.md#term-cumulative-scan) broadcast, SUM and [layout](../../../../glossary.md#layout) closure

## Status

Complete — main accepted the bounded capability on 2026-10-22, 14:38+ UTC, after final
independent code/docs APPROVE (0 P1/P2), worker [module](../../../../glossary.md#term-module) [validation](../../../../glossary.md#term-validation) and the named [checkpoint](../../../../glossary.md#term-checkpoint) PASS.
[CPU](../../../../glossary.md#term-cpu) frontier: None; other area frontiers None. Mandatory [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) successor gates remain Draft.

## Change class

Class C — generated hot-loop/[dataflow](../../../../glossary.md#term-dataflow), temporal [carry](../../../../glossary.md#term-carry) [ownership](../../../../glossary.md#term-ownership-meanings) and exact pre-slot resources.

## Goal

Execute [homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT32](../../../../glossary.md#term-float32) [shape](../../../../glossary.md#shape)-changing [scan](../../../../glossary.md#term-cumulative-scan) algebra and its actual [Compiler](../../../../glossary.md#term-compiler)-generated
first-order reverse: body proof → resource/[liveness](../../../../glossary.md#term-liveness) plan → one emitted time loop.
Close broadcast [parameter](../../../../glossary.md#term-parameter) [gradients](../../../../glossary.md#term-gradient) without assuming every [carry](../../../../glossary.md#term-carry) or body value has one common N.

## [Scope](../../../../glossary.md#term-scope)

- Static T≥0, all-valid axis-zero sequences; ≥1 carry/sequence, ≥0 invariants/emissions.
  Each carry has its own fixed shape; each sequence its own fixed step shape and common T.
  Local ranks 0..16, stacked rank up to 17; checked element/byte geometry, including zeros.
  Canonical outer boundaries/formals, plus CPU0030's proved all-zero-stride read bindings for
  Compiler splats/initializers; no general outer-stride/offset expansion. Heap/native/mixed
  inputs remain checked MemorySegment bindings, not new entry carrier strategies.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [carry glossary](../../../../glossary.md#term-carry); [shape glossary](../../../../glossary.md#shape); [rank glossary](../../../../glossary.md#term-rank); [element glossary](../../../../glossary.md#term-element); [Canonical glossary](../../../../glossary.md#term-canonical-form); [stride glossary](../../../../glossary.md#term-stride); [bindings glossary](../../../../glossary.md#term-binding); [Compiler glossary](../../../../glossary.md#term-compiler); [Heap glossary](../../../../glossary.md#term-heap); [MemorySegment glossary](../../../../glossary.md#term-memorysegment); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Exact closed algebra: tensor/scalar ADD/SUB/MUL/DIV, NEG/TANH/SIGMOID;
  right-aligned Model broadcasting; SUM with NoOperationAttrs, AxisReductionAttrs,
  MultiAxisReductionAttrs or SumToShapeAttrs (keep-dimensions true/false where applicable);
  EXPAND/RESHAPE with TargetShapeAttrs, EXPAND_DIMS/SQUEEZE with AxisTransformAttrs,
  rank-two PERMUTE <span data-glossary-navigation="true"> ([tensor glossary](../../../../glossary.md#tensor); [scalar glossary](../../../../glossary.md#term-scalar); [TANH glossary](../../../../glossary.md#term-tanh); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [Model glossary](../../../../glossary.md#term-model); [broadcasting glossary](../../../../glossary.md#broadcasting); [NoOperationAttrs glossary](../../../../glossary.md#nooperationattrs); [dimensions glossary](../../../../glossary.md#dimension); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`[1,0]`, and CONTIGUOUS. Only valid existing Model guards/layout inference.
  RESHAPE requires proved reshape-compatible storage or an explicit CONTIGUOUS input, not
  an implicit reorder. Derived zero-stride/permuted local views are not canonical outer storage.<span data-glossary-navigation="true"> ([CONTIGUOUS glossary](../../../../glossary.md#term-contiguous); [layout glossary](../../../../glossary.md#layout); [inference glossary](../../../../glossary.md#term-inference); [views glossary](../../../../glossary.md#term-view-meanings))</span>
- Qualify forward and actual reverse independently, optimized/unoptimized, using existing
  ScanGradientRules and ordinary rule owners. Reverse closure includes positional ADD accumulation,
  broadcast SUM-to-Shape, restoration EXPAND_DIMS/EXPAND/CONTIGUOUS, inverse views and closed
  typed zero/one invariant splats. Homogeneity needs no cross-dtype CAST; same-type casts are
  identity only if actually retained by Model/capture. Reject every type-changing CAST.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([SUM-to-Shape glossary](../../../../glossary.md#sum-to-shape); [invariant glossary](../../../../glossary.md#term-invariant); [dtype glossary](../../../../glossary.md#data-type--datatype); [identity glossary](../../../../glossary.md#term-identity-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>

## Non-goals

[MATMUL](../../../../glossary.md#matrix-multiplication--matmul) (including its reverse), [FLOAT64](../../../../glossary.md#term-float64)/low/mixed types, new [AD](../../../../glossary.md#autograd)/rules, [Compiler](../../../../glossary.md#term-compiler) production changes,
[padding](../../../../glossary.md#padding)/lengths, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[persistence](../../../../glossary.md#term-persistence) [admission](../../../../glossary.md#term-admission), [Engine](../../../../glossary.md#term-engine)/[Runtime](../../../../glossary.md#term-runtime)/shared-[Prepare](../../../../glossary.md#prepare) changes, native/[vector](../../../../glossary.md#term-vector)/
parallel [scan](../../../../glossary.md#term-cumulative-scan) routes, tuning, arbitrary regions/control flow, or a broad performance campaign.
Retain fixed four-[dtype](../../../../glossary.md#data-type--datatype)/[layout](../../../../glossary.md#layout) routes and CPU0030's admitted [pointwise](../../../../glossary.md#term-pointwise) domain without [regression](../../../../glossary.md#term-regression).

## Contracts

- Root [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>,
  [Core lifecycle](../../../../../ARCHITECTURE.md#core-lifecycle),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>,
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>,
  ALL subordinate headings — source/scopes, canonical residual demand, exact primal binding,
  traversal/empties, qualification/performance/migration gates.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [primal glossary](../../../../glossary.md#term-primal); [binding glossary](../../../../glossary.md#term-binding); [qualification glossary](../../../../glossary.md#term-qualification))</span>
- [Backend](../../../../glossary.md#backend) [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>,
  [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>,
  [Performance evidence and optimization tooling](../../../../architecture/contracts/backend-execution.md#performance-evidence-and-optimization-tooling),
  [Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity)<span data-glossary-navigation="true"> ([backend capability glossary](../../../../glossary.md#backend-capability); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>.
- [Lifecycle](../../../../glossary.md#lifecycle) [`modules/runtime`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
  [`modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare))</span>,
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>,
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>.
- Foundational [Profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries)
  and [Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([Profile-free numerical semantics glossary](../../../../glossary.md#profile-free-numerical-semantics))</span>:
  Family formulas and unary sites, Arithmetic-site DAZ/FTZ, Exact and discrete floor,
  Floating arithmetic primitives, Aggregate floor, Composite inheritance.<span data-glossary-navigation="true"> ([unary glossary](../../../../glossary.md#term-unary); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [Aggregate floor glossary](../../../../glossary.md#term-aggregate-floor))</span>
- [Compiler](../../../../glossary.md#term-compiler) [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd))</span>
  only for actual generated forms/primal/constants; no new AD ownership or algebra.<span data-glossary-navigation="true"> ([constants glossary](../../../../glossary.md#term-constant); [AD glossary](../../../../glossary.md#autograd); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>

Missing/ambiguous applicable authority or need for a new boundary: stop and ask main.

## Dependencies and integration

- Depends on: CPU0030; its accepted Model0034/0035, Planning0008, Compiler0011, Prepare0010
  and CPU0022/0027/0028/0029 baseline, all Complete per current area/program status and main's handoff.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- Conflicts with: general-scan admission/lowering/IR/emission/binding/resources, generated/cache
  identity, numerical policy/current ledgers, shared planning/explanations; serialize these scopes.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [admission glossary](../../../../glossary.md#term-admission); [lowering glossary](../../../../glossary.md#lowering); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [binding glossary](../../../../glossary.md#term-binding); [cache glossary](../../../../glossary.md#term-cache); [identity glossary](../../../../glossary.md#term-identity-meanings); [planning glossary](../../../../glossary.md#term-planning))</span>
- Parallel group: None; one writer. Read-only review may overlap; no concurrent write [scope](../../../../glossary.md#term-scope).
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40`, plus
  248 accepted dirty paths through final CPU0030 and this planning packet; not merged predecessors.
- Integration order: independent plan review → main final acceptance/Ready verification → clean
  CPU executor → independent targeted Class C code/docs review/corrections → main checkpoint.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Integration validation: worker commands below; main's CPU0031 focused cross-module checkpoint;
  one source-matched native-enabled root <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [module glossary](../../../../glossary.md#term-module))</span>`test javadoc` remains at the later low-level program join.
- Shared-document integration owner: main; delegate focused implementation explanations/Javadoc
  exclusively to the serial worker/reviewer, retain shared planning/status integration in main.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

Historical main launch verification (2026-10-22): accepted independent plan APPROVE (0 P1/P2),
all dependencies Complete, matching HEAD/origin/main at the common base, empty index, inactive
conflicts and preserved accepted dirty work/source guards. User's “ok, pokracuj” authorized the
bounded successor; CPU0031 was the sole [Ready](../../../../glossary.md#term-ready) frontier. Seven integration fields remained unchanged.
Preserve THIRD_PARTY_NOTICES.md, CpuOrdinaryPointwiseCastCleanJavaOracle.java and the Superseded
Compiler0010 brief byte-for-byte; no reset/checkout/stash, [commit](../../../../glossary.md#term-commit)/push, or unrelated cleanup.

## Files and symbols

[CPU](../../../../glossary.md#term-cpu) prefix: `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.
Reuse `CpuCapabilityProvider`, `internal/lowering/CpuGeneralScanLowering` (analyze/lower/Facts/Geometry),
`internal/ir/CpuGeneralScanIr`, `internal/codegen/emit/CpuGeneralScanEmitter`,
`internal/executable/CpuGeneralScanPreparedExecutable`, existing [partition](../../../../glossary.md#partition)/portable/[cache](../../../../glossary.md#term-cache) seams.
Read `CpuAggregateLowering`/`CpuAggregateIr`/`CpuAggregateEmitter`, `CpuCarrierEmitter`,
`CpuLoopEmitter`/`CpuScalarEmitter`, and `CpuMatmulEmitter` to distinguish reusable code from entry [ABI](../../../../glossary.md#term-application-binary-interface-abi).
Read [Compiler](../../../../glossary.md#term-compiler) `ScanGradientRules`, [Elementwise](../../../../glossary.md#term-elementwise)/[Layout](../../../../glossary.md#layout)/Reduction/LinearAlgebraGradientRules and
`ScanGradientRulesTest`/`GeneralScanCompilerTest`; they already demonstrate broadcast/SUM/[view](../../../../glossary.md#term-view-meanings)
closure and [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) in [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding), not CPU execution capability.

Only added top-level production type, now delivered: `CpuGeneralScanReductionEmitter` in existing package
`io.github.pho001.synaptik.backend.cpu.internal.codegen.emit`, for step-bound SUM loops rather than
the ordinary aggregate emitter's fixed two-entry ABI/exact-[limb](../../../../glossary.md#exact-superaccumulator-and-limbs) workspace. Reuse [scalar](../../../../glossary.md#term-scalar)/[carrier](../../../../glossary.md#term-storage-carrier)/loop
seams; keep ordinary aggregate routes unchanged. Derived stage/[binding](../../../../glossary.md#term-binding)/resource facts stay nested
in the four existing [scan](../../../../glossary.md#term-cumulative-scan) types; no second semantic body, generic stage executor or new public type.
Any additional production type/package requires main's packet update before implementation.

Tests: extend existing `CpuGeneralScan*` capability/[lowering](../../../../glossary.md#lowering)/preparation/generated/execution/[oracle](../../../../glossary.md#term-test-oracle)/
timing tests, `CpuKernelSpecializationTest`/[artifact](../../../../glossary.md#term-artifact)-store tests and current generated coverage ledger.
Extend `testing/backend-conformance/.../CpuGeneralScanConformanceTest` with [Model](../../../../glossary.md#term-model)-only fixtures;
NO Compiler [dependency](../../../../glossary.md#term-dependency) there. Actual [AD](../../../../glossary.md#autograd) belongs to CPU tests (existing test-only Compiler edge)
and `testing/integration-tests/.../EngineCpuGeneralScanIntegrationTest`. No build-edge change.

## Acceptance criteria

1. Provider proof validates whole header/local inference/role inventory/demand AND binder,
   including unsupported dead interiors at T=0 or empty coordinates. Preserve current budgets
   (128 nodes/256 local values/64 formals/256 ports, ≤248 boundaries and ≤254 entry slots),
   plus fail-closed generated method/local budgets. A forward true answer never promises reverse.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [coordinates glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [nodes glossary](../../../../glossary.md#node); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
2. Invocation binding validates dtype/alignment/capacity/complete-capacity overlap once, creates
   run-owned geometry/carrier arrays and hoists invariant layout/stride/constant setup outside time.
   Step binding changes only required carriers/bases/time offsets/carry versions/ranges; typed locals
   and generation-time emitter calls only. No graph, dispatch, allocation, geometry-array creation,
   MethodHandle/callback per op/step or T-fold bytecode. One RunState and one prepared entry remain.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [dtype glossary](../../../../glossary.md#data-type--datatype); [alignment glossary](../../../../glossary.md#term-alignment); [run glossary](../../../../glossary.md#term-run); [carrier glossary](../../../../glossary.md#term-storage-carrier); [invariant glossary](../../../../glossary.md#term-invariant); [layout glossary](../../../../glossary.md#layout); [stride glossary](../../../../glossary.md#term-stride); [constant glossary](../../../../glossary.md#term-constant); [carry glossary](../../../../glossary.md#term-carry); [graph glossary](../../../../glossary.md#graph); [dispatch glossary](../../../../glossary.md#term-dispatch); [MethodHandle glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture); [callback glossary](../../../../glossary.md#term-callback); [bytecode glossary](../../../../glossary.md#term-bytecode); [RunState glossary](../../../../glossary.md#run-state--runstate))</span>
3. Before slots, derive each value's shape/access and use lifetime; each carry's temporal forwarding
   closure and simultaneous old/new versions; exact checked output/live-history bytes and all
   staging, carry banks, body temporaries and reduction scratch with alignment/offset/lifetime.
   Prefer direct sequence/emission/residual offsets and demanded frames as backing. A transpose
   or reduction may read across coordinates: CPU0030's in-place per-coordinate bank is not safe
   by default. Overwrite only after the last old-value consumer; use separately justified banks/
   ping-pong when necessary. Reuse scratch after last use, never a live saved site or aliased role.
   Document why every materialization/copy exists; no blanket per-op buffers or input staging.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [transpose glossary](../../../../glossary.md#term-transpose); [materialization glossary](../../../../glossary.md#materialization); [buffers glossary](../../../../glossary.md#term-buffer))</span>
4. Previous-carry histories store BEFORE the step, produced histories at exact stored sites, all at
   original time indices. Distinct demanded wrappers remain distinct even for aliased roots.
   Undemanded forward AND reverse residuals allocate/store zero history. Internal temporaries are
   not histories. T-dependent publication bytes remain; O(body) recipe/code excludes requested data.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
5. Baked/keyed facts: dtype, ordered body/op/immediate dataflow, normalized axes/keep-dims/target
   mapping, view/access/rank relations, role/alias/demand maps, carry version/storage decisions,
   stage ordering, scratch/entry ABI and every specialized extent/stride. Runtime facts: T,
   compatible extents/stride magnitudes, per-value frame sizes/ranges/addresses, slots/representations
   and run buffers. Key any geometry actually baked; never put slots/IDs/storage into CpuKernelIr.
   Current envelope/family are 71: explicitly advance incompatible identity append-only, reject
   stale 71 artifacts if meaning/ABI changes, and synchronize identity references through main.
   Test independent key mutations, stale rejection and identical-code reuse for compatible geometry.<span data-glossary-navigation="true"> ([dataflow glossary](../../../../glossary.md#term-dataflow); [target glossary](../../../../glossary.md#term-target); [view glossary](../../../../glossary.md#term-view-meanings); [alias glossary](../../../../glossary.md#aliasing-and-arena); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [extent glossary](../../../../glossary.md#term-extent); [Runtime glossary](../../../../glossary.md#term-runtime); [identity glossary](../../../../glossary.md#term-identity-meanings); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
6. Test the complete [shape](../../../../glossary.md#shape)/op + actual [AD](../../../../glossary.md#autograd) [matrix](../../../../glossary.md#term-matrix-meanings): [scalar](../../../../glossary.md#term-scalar) and `[H]` trainable broadcast invariants
   into `[B,H]`, [rank](../../../../glossary.md#term-rank) growth `[H]`/`[B,1]`, different-shaped multiple carries and emission shapes;
   both traversals, two carries, T=0/1/≥3, both/missing seed classes, false-gradient h0 with trainable
   x/parameters, tied roles/roots, unused inputs and reused templates/cross-occurrence sums.
   Exercise full/single/multi-axis SUM (both keep forms), target-one/leading/equal-shape sumToShape,
   view restoration/transpose/CONTIGUOUS and every declared arithmetic/scalar kind. Compare actual
   emitted reverse to unrolled ordinary Tensor AD and specialized represented-site Java oracles.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [gradient glossary](../../../../glossary.md#term-gradient); [parameters glossary](../../../../glossary.md#term-parameter); [occurrence glossary](../../../../glossary.md#term-occurrence); [CONTIGUOUS glossary](../../../../glossary.md#term-contiguous); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [Tensor glossary](../../../../glossary.md#tensor))</span>
7. Empty [axes](../../../../glossary.md#term-axis) do not skip nonempty work: `[0,H] -> [H]` SUM/parameter gradients produce typed zero
   even with NaN/Inf invariant values; zero selected axes/no-reduction sumToShape preserve bits.
   T=0 touches no body: raw initial final carry, empty histories/emissions, dInitial=dFinal seed,
   zero invariant sums. Include permuted/cross-coordinate carry hazards and forwarding chains.
   Pin NaN/±Inf/±0, zero divisors, overflow/underflow/subnormals; DIV reverse remains ordinary
   unmasked formula, activation reverse consumes saved values, movement stays raw exact. Reduced
   SUM may use a rounded all-contributors-once F32 tree per Model; do not inherit exact-limb
   overhead merely from the ordinary route. Explicit oracle/metric/threshold for finite arithmetic.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [subnormals glossary](../../../../glossary.md#term-subnormal); [activation glossary](../../../../glossary.md#term-activation); [all-contributors-once glossary](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [Model glossary](../../../../glossary.md#term-model); [limb glossary](../../../../glossary.md#exact-superaccumulator-and-limbs); [oracle glossary](../../../../glossary.md#term-test-oracle); [metric glossary](../../../../glossary.md#term-metric); [finite glossary](../../../../glossary.md#term-finite))</span>
8. Overflow/budget/malformed body/unsupported MATMUL/F64/cast/view rejections occur before writes;
   heap/native/mixed, overlap tails, cleanup/failure/retry, open-result fresh runs and concurrent
   prepared reuse retain existing behavior. Structural bytecode tests count real history/bank/
   staging stores and inspect loops/calls/allocation. Keep old pointwise no-extra-copy hot path.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [heap glossary](../../../../glossary.md#term-heap); [pointwise glossary](../../../../glossary.md#term-pointwise); [hot path glossary](../../../../glossary.md#term-hot-path))</span>
9. Oracle is optimal clean specialized Java for the SAME algorithm, loops/dataflow, dtype/carrier,
   saved sites and work, not a deliberately generic fallback. Class-File/decompilation inspection
   and one short matched timing diagnostic cover a small broadcast reverse and a larger reduction
   case; separate cold generation/load/first-JIT opportunity, entry/preflight and kernel/step costs.
   Repeat the old CPU0030 small mixed-backward timing only if pointwise executable shape changes
   or a concrete regression risk justifies it. Otherwise retain structural old-path regression
   checks and short matched timing of the new broadcast/reduction cases. Transparently retain the
   historical ~7× gap with unproved origin; parity is NOT inherited. Fix or justify concrete
   diagnosed avoidable overhead; report unresolved timing/JIT confounds without speed promise,
   parity/C2 gate, five forks or waiting for C2.<span data-glossary-navigation="true"> ([forks glossary](../../../../glossary.md#term-fork-meanings))</span><span data-glossary-navigation="true"> ([decompilation glossary](../../../../glossary.md#term-decompilation); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit); [preflight glossary](../../../../glossary.md#term-preflight); [kernel glossary](../../../../glossary.md#kernel); [regression glossary](../../../../glossary.md#term-regression); [C2 glossary](../../../../glossary.md#term-c2))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker validation: focused iterations, then one final affected-[module](../../../../glossary.md#term-module) [run](../../../../glossary.md#term-run); evidence in Result.

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScan*' --tests '*CpuKernelSpecializationTest' --tests '*CpuGeneratedKernelArtifactStoreTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc
./gradlew :backends:cpu:test --tests '*CpuGeneralScanTimingTest' -Dsynaptik.cpu.generalScan.performance=true -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0031-evidence
```

Main [checkpoint](../../../../glossary.md#term-checkpoint) once after ordered review/integration; reuse successful worker evidence:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'
./gradlew :testing:architecture-tests:test
git diff --check
```

Root/native validation remains at the low-level join, not a repeated worker gate.

## Follow-up

Mandatory serial gates after reviewed CPU0031: [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) + its actual reverse on the new
[bindings](../../../../glossary.md#term-binding), then [homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT64](../../../../glossary.md#term-float64) [qualification](../../../../glossary.md#term-qualification) across the COMPLETE declared algebra/closure.
Reassess cohesive task size at that frontier; no far-future brief or [Ready](../../../../glossary.md#term-ready) grant now. [F64](../../../../glossary.md#floating-point-precision-and-rounding) requires
typed double locals/eight-byte resources, full numerical/saved-site tests and NO [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz).

## Documentation and review impact

[Planning](../../../../glossary.md#term-planning) profile applies. Affected Javadocs, targeted [CPU](../../../../glossary.md#term-cpu)/[backend](../../../../glossary.md#backend)/[scan](../../../../glossary.md#term-cumulative-scan) explanations and current
ledger evidence are finalized; main owns shared status integration.
No architecture decision/edge or new glossary meaning changed. Targeted terms: [general scan](../../../../glossary.md#general-differentiable-scan),
formal/[primal](../../../../glossary.md#term-primal) [binding](../../../../glossary.md#term-binding), [residual](../../../../glossary.md#scan-residual-port-and-history-demand)/[history demand](../../../../glossary.md#scan-residual-port-and-history-demand), [cotangent normalization](../../../../glossary.md#cotangent-normalization) and [sum-to-Shape](../../../../glossary.md#sum-to-shape).
Independent plan and final Class C code/docs reviews accepted; successful executable evidence reused.
Closure changes planning [metadata](../../../../glossary.md#term-metadata) only; no further explanatory/[Javadoc](../../../../glossary.md#term-javadoc)/glossary update is needed.

## Result

Main accepted 2026-10-22, 14:38+ UTC: bounded [F32](../../../../glossary.md#floating-point-precision-and-rounding) broadcast/SUM/[layout](../../../../glossary.md#layout) forward + actual reverse Complete.
Files: [scan](../../../../glossary.md#term-cumulative-scan) [IR](../../../../glossary.md#term-intermediate-representation-ir)/[lowering](../../../../glossary.md#lowering)/emitter/executable seams, new CpuGeneralScanReductionEmitter, existing [specialization](../../../../glossary.md#term-specialization),
scan/[cache](../../../../glossary.md#term-cache)/[oracle](../../../../glossary.md#term-test-oracle)/timing tests, [CPU](../../../../glossary.md#term-cpu) build opt-ins, [conformance](../../../../glossary.md#term-conformance)/[Engine](../../../../glossary.md#term-engine) fixtures, ledger v2 and focused docs/Javadocs.
Rawls final code APPROVE (0 P1/P2), R1–R4 closed/manifests verified; Herschel independent docs APPROVE (0),
preserving Rawls's 12 Markdown/5 Java files and finalizing five docs; six Java files executable-[token](../../../../glossary.md#term-token)-neutral,
30 normalized compiled-class dumps identical; 83 targeted links/anchors PASS. No new [API](../../../../glossary.md#term-application-programming-interface-api)/architecture/edge/[lifecycle](../../../../glossary.md#lifecycle)/glossary meaning.
Geometry-free [canonical](../../../../glossary.md#term-canonical-form) CpuKernelIr and global/family 71 retained; conditional typed scan-shaped-v1 [payload](../../../../glossary.md#artifact-envelope-and-atomic-publication)
in BOTH existing specialization byte projections; old 12 actual ClassFiles unchanged. No current generated
inventory refresh; v1 ledger [immutable](../../../../glossary.md#term-immutable), v2 adds two actual query rows. Fixed four-type/layout routes unchanged.
Final `./gradlew :backends:cpu:test :backends:cpu:javadoc` PASS (5m27s): 216 suites, 1148 tests,
1115 passed/33 skips/0 failures/errors; fresh XML 14:07:54–14:13:21 UTC; later comments proved neutral.
[Javadoc](../../../../glossary.md#term-javadoc) PASS, 94 baseline warnings (untouched compatibility-constructor [parameters](../../../../glossary.md#term-parameter)/incubator), not zero-warning.
Focused: 58 tests/2 skips/0 failures, including 72 actual [AD](../../../../glossary.md#autograd)/[unrolled](../../../../glossary.md#backpropagation-through-time-and-recurrent-state) variants and 36 R4 regressions;
40 vs 52 gets, 32 B/16 F32 stores and unused 0 B; R3 LMUL 34→11. Safe-cell [fusion](../../../../glossary.md#term-fusion) retains necessary full-stage cuts.
Main [checkpoint](../../../../glossary.md#term-checkpoint): `./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest' --console=plain` PASS (1s), 2 suites/10 tests, 14:37:51 UTC;
`./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest' --console=plain` PASS (3s), 1 suite/7 tests, 14:36:33 UTC;
`./gradlew :testing:architecture-tests:test --console=plain` PASS (2s), 11 suites/25 tests, 14:36:58–59 UTC: all 42 pass, no skips/failures.
Only conformance rerun: stale expected F32 [seed](../../../../glossary.md#term-seed) count 118→119 for new seed 130-SCAN/broadcast-sum-layout;
all independent [BF16](../../../../glossary.md#floating-point-precision-and-rounding)/[F16](../../../../glossary.md#floating-point-precision-and-rounding) negatives retained; v1/v2/production unchanged. Final source freeze verified; no duplicate CPU [run](../../../../glossary.md#term-run).
Matched timing: two new cases, one [JVM](../../../../glossary.md#term-java-virtual-machine-jvm)/five [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters), `/private/tmp/synaptik-cpu0031-r3-evidence/summary.txt`;
T8B2H8 generated 26.2–27.8 µs vs Java 22.8–24.7 µs; T8B64H128 0.357–1.890 ms vs 119–132 µs.
Transition/gap origin unproved; historical CPU0030 ~7× mixed gap retained, no parity/[C2](../../../../glossary.md#term-c2) guarantee or new performance gate.
Java/ClassFile oracle algorithms, time→H→B loops, carries and load/store counts match; no hidden helpers/[graph](../../../../glossary.md#graph)/[dispatch](../../../../glossary.md#term-dispatch)/allocation.
Follow-up: serial [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) + actual F32 reverse Draft → full [homogeneous](../../../../glossary.md#term-homogeneous) [F64](../../../../glossary.md#floating-point-precision-and-rounding) algebra/closure Draft → root/native low-level join.
No full F32-domain, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[padding](../../../../glossary.md#padding)/training/[persistence](../../../../glossary.md#term-persistence)/low/[vector](../../../../glossary.md#term-vector)/parallel grant. No worker remains; no successor launched.
Status: Complete

Size exception: acceptance criteria and the compact final evidence remain together above 200 lines/15 KB;
splitting this bounded generated-code/resource capability would separate its [validation](../../../../glossary.md#term-validation) from its acceptance.
