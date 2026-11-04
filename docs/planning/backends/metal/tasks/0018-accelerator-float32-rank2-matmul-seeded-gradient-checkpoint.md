# Task 0018: [ACCELERATOR](../../../../glossary.md#term-accelerator) [FLOAT32](../../../../glossary.md#term-float32) [Rank-Two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) Seeded-[Gradient](../../../../glossary.md#term-gradient) [Checkpoint](../../../../glossary.md#term-checkpoint)

## Reader context

The passing smoke did not complete the planned matrix: the first full single-term contraction
failed the then-required zero-sign rule. The later Model decision and successor task are separate
events, not a reopening of this blocked experiment.

Compiler would generate the two operand-gradient graphs from an explicit caller-supplied
cotangent seed, meaning the output sensitivity to propagate backward. Metal would execute those
ordinary operations. The local PERMUTE description changed logical operand order only when
analysis proved its producer edge; it did not authorize arbitrary borrowed view storage or
hidden packing. This proposed seeded-gradient checkpoint was narrower than general training.
The [current admission and matrix-multiplication explanation](../../../../backend-guide/metal-backend.md#capability-whole-partition-analysis-and-route-selection)
connects those descriptor checks to today's bounded routes, not to unperformed acceptance work here.

## Status

Blocked

[Metal](../../../../glossary.md#term-metal) 0020 is Complete at implementation `9ddb75f6` plus documentation remediation `5b77c742`.
[Model](../../../../glossary.md#term-model) 0027, Config 0006, and [Engine](../../../../glossary.md#term-engine) 0018 are Complete. Historical Metal 0009 and Metal 0017 remain
Blocked records, not dependencies; Metal 0016 has no active restart, and Model 0026 remains an
independent Draft. The mandatory fresh real-M3 probe ran from clean base
`0625a18727ac2cf59220cfa7db5de93e1cac2651` and stopped before any production edit. The [smoke](../../../../glossary.md#term-smoke-test)
passed, but the first full direct context-zero/optimization-zero `K=1` cell evaluated
`+0.0f * -1.0f` as [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), while this task's unchanged Model set permits only [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero).
`MPSGraphReducedPrecisionFastMathNone` set/read and the selector, direct [targets](../../../../glossary.md#term-target), and controls were
valid. Zero is not [subnormal](../../../../glossary.md#term-subnormal), and a one-term [contraction](../../../../glossary.md#contraction-dimension) has no [reassociation](../../../../glossary.md#term-reassociation) or multiply-add
choice that changes the required sign. The probe was removed and the tree returned clean. [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4
still has exactly thirteen exports, [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema) is seven, and private [identities](../../../../glossary.md#term-identity-meanings) are version eight.
No production/schema/capability change from this task exists.

## Change class

Class C — this adds profile-qualified [contraction](../../../../glossary.md#contraction-dimension) capability, local affine-[view](../../../../glossary.md#term-view-meanings) [topology](../../../../glossary.md#term-topology), native
[schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings), [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution), and public [Engine](../../../../glossary.md#term-engine) forward/backward behavior across the
native boundary. A fresh real-[device](../../../../glossary.md#term-device) [oracle](../../../../glossary.md#term-test-oracle) and independent final review are mandatory.

## Goal

Add only positive, static, same-type [`FLOAT32`](../../../../glossary.md#term-float32) [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) under
`NumericalProfile.ACCELERATOR`, including exact local rank-two [transpose](../../../../glossary.md#term-transpose) composition and
[Compiler](../../../../glossary.md#term-compiler)-generated explicitly seeded first-order [gradients](../../../../glossary.md#term-gradient) for both operands. Admit only the
[Model](../../../../glossary.md#term-model)-owned [reassociation](../../../../glossary.md#term-reassociation)/[FMA](../../../../glossary.md#term-fused-multiply-add-fma) plus row-scoped [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz) result set. Preserve every [contraction](../../../../glossary.md#contraction-dimension) term,
never use reduced [precision](../../../../glossary.md#term-precision), and keep strict [Metal](../../../../glossary.md#term-metal) `MATMUL` false.

## Exact capability and [topology](../../../../glossary.md#term-topology)

- Admit exactly `MatmulKind.MATMUL` with `NoOperationAttrs.INSTANCE`, two ordered inputs, and one
  output. Shapes are <span data-glossary-navigation="true"> ([Shapes glossary](../../../../glossary.md#shape))</span>`[M,K]`, `[K,N]`, and `[M,N]`, with positive static `M`, `K`, and `N`; all
  element/byte counts and native conversions use checked arithmetic.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element))</span>
- Inputs and output are same-type [`FLOAT32`](../../../../glossary.md#term-float32). The canonical output is dense, zero-offset, and
  non-view; its <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [view glossary](../../../../glossary.md#term-view-meanings))</span>`requiresGrad` is the logical OR of the input flags. Input flags need not match.
- Each operand is either a canonical dense zero-offset non-view or the exact rank-two descriptor
  produced by one earlier in-partition <span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [descriptor glossary](../../../../glossary.md#term-descriptor); [partition glossary](../../../../glossary.md#partition))</span>`PERMUTE [1,0]` from a canonical source. Authenticate the
  exact producer, axes, value identity, topology, and represented logical order. The transpose
  cannot be a partition feed, foreign transfer, constant, arbitrary view, other affine form, or
  hidden host/materialization step.<span data-glossary-navigation="true"> ([producer glossary](../../../../glossary.md#term-producer); [axes glossary](../../../../glossary.md#term-axis); [identity glossary](../../../../glossary.md#term-identity-meanings); [transpose glossary](../../../../glossary.md#term-transpose); [constant glossary](../../../../glossary.md#term-constant); [host glossary](../../../../glossary.md#term-host); [materialization glossary](../../../../glossary.md#materialization))</span>
- The exact local transpose may occur on the left, right, or both operands. This includes public
  no-bias <span data-glossary-navigation="true"> ([bias glossary](../../../../glossary.md#term-bias))</span>[`linear`](../../../../glossary.md#term-linear-meanings), whose visible chain is `PERMUTE [1,0] -> MATMUL`; no `LINEAR` operation or
  pattern rewrite is introduced.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- A canonical MATMUL output may be published or consumed by already-admitted accelerator nodes.
  No cross-node contraction is allowed. Complete-partition analysis must reject malformed local
  transpose provenance before native creation.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [accelerator glossary](../../../../glossary.md#term-accelerator); [nodes glossary](../../../../glossary.md#node); [contraction glossary](../../../../glossary.md#contraction-dimension); [provenance glossary](../../../../glossary.md#provenance))</span>
- Strict Metal rejects every MATMUL occurrence. Accelerator rejects rank other than two, vectors,
  batches/broadcasts, zero/dynamic extents, other types/promotions, wrong attrs/Shape/layout,
  nonlocal views, unchecked geometry, and malformed topology.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [occurrence glossary](../../../../glossary.md#term-occurrence); [rank glossary](../../../../glossary.md#term-rank); [vectors glossary](../../../../glossary.md#term-vector); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [layout glossary](../../../../glossary.md#layout))</span>

## Numerical contract and mandatory fresh Apple M3 [oracle](../../../../glossary.md#term-test-oracle)

Use only `matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:`. Before production edits,
create a new disposable [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) probe under `/tmp`, [compile](../../../../glossary.md#compile) it with `xcrun clang` against
Foundation/[Metal](../../../../glossary.md#term-metal)/[MPSGraph](../../../../glossary.md#term-mpsgraph), [run](../../../../glossary.md#term-run) it on the real M3, and remove source and [binary](../../../../glossary.md#term-binary-descriptor-and-metadata). Do not reuse the
historical Metal 0009 probe or results.

Require macOS 26's `reducedPrecisionFastMath` control and set it to
`MPSGraphReducedPrecisionFastMathNone` at optimization levels zero and one; inability to set that
exact control blocks the task. Use two fresh contexts, three independently created executables per
cell per context, eight runs per executable, guarded caller-supplied direct [targets](../../../../glossary.md#term-target), permuted feed/
target [bindings](../../../../glossary.md#term-binding), retained native names/[Shapes](../../../../glossary.md#shape), and cell-scoped autorelease pools. First pass one
non-square direct [smoke](../../../../glossary.md#term-smoke-test) cell. Then exercise direct, left-transposed, right-transposed, and both-
transposed [topology](../../../../glossary.md#term-topology) forms with at least eight Shape/data cells per form: non-square ordinary,
`M=1`, `K=1`, `N=1`, exact integers, all-term sentinels, [cancellation](../../../../glossary.md#term-cancellation)/tree-sensitive values, and
[subnormal](../../../../glossary.md#term-subnormal)/normal/special-value boundaries. This is at least 384 executables and 3,072 runs, in
addition to the smoke cell and run; report exact totals.

For every bounded output [dot product](../../../../glossary.md#term-dot-product), an independent raw-bit oracle must enumerate the complete
[Model](../../../../glossary.md#term-model)-permitted set: every declared `k` contribution exactly once; every [reassociation](../../../../glossary.md#term-reassociation)/binary tree;
[`FLOAT32`](../../../../glossary.md#term-float32) [rounding](../../../../glossary.md#term-rounding); only corresponding multiply-add [contraction](../../../../glossary.md#contraction-dimension) by [FMA](../../../../glossary.md#term-fused-multiply-add-fma); row-scoped [DAZ](../../../../glossary.md#term-daz) of declared
subnormal arithmetic inputs; and optional [FTZ](../../../../glossary.md#term-ftz) of [finite](../../../../glossary.md#term-finite) subnormal arithmetic results. Include both
unfused and every legal [fused](../../../../glossary.md#specialization-and-fusion) placement. An FMA cannot cross [nodes](../../../../glossary.md#node) or erase an observable value.
No contribution may be dropped, duplicated, replaced, or treated as an implicit reduced-[precision](../../../../glossary.md#term-precision)
term. [Mantissa](../../../../glossary.md#floating-point-precision-and-rounding)-sensitive cells must distinguish full FLOAT32 products/accumulation from reduced
precision. Term-sentinel cells must make omission of each individual `k` term observable.

Cover signed zeros; minimum/ordinary/maximum subnormals; minimum normals and neighbors; ordinary
and maximum finite values; [overflow](../../../../glossary.md#term-overflow); [infinities](../../../../glossary.md#nan-infinity-and-signed-zero); signed quiet/signaling NaNs; `K=1` multiplication
by one; cancellation and tree-sensitive permutations. Require raw-bit membership when the allowed
set is finite and exact result-class membership only where current Model [NaN](../../../../glossary.md#term-nan) representation is
unconstrained. [Epsilon](../../../../glossary.md#epsilon), [CPU](../../../../glossary.md#term-cpu)-output comparison, [tolerance](../../../../glossary.md#term-tolerance), term-[loss](../../../../glossary.md#loss) acceptance, reciprocal or
algebraic substitution, undocumented fast math, and special-value repair are forbidden. Any
selector, Shape, binding, [canary](../../../../glossary.md#term-canary), input-preservation, [transpose](../../../../glossary.md#term-transpose), full-term, precision, or oracle
failure blocks 0018 without production/[schema](../../../../glossary.md#term-schema)/capability edits.

## Native [schema](../../../../glossary.md#term-schema), [identity](../../../../glossary.md#term-identity-meanings), [lifecycle](../../../../glossary.md#lifecycle), and [cache](../../../../glossary.md#term-cache)

- Keep ABI v4, statuses, create/run signatures, and exactly thirteen exports. Evolve the fixed
  160-byte node schema <span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [run glossary](../../../../glossary.md#term-run); [signatures glossary](../../../../glossary.md#term-signature); [node glossary](../../../../glossary.md#node))</span>`7 -> 8`; retain wires `1..14`; append only `MATMUL=15` with two ordered
  inputs, `NONE` attributes, required sentinels, zero count/reserved/[payload](../../../../glossary.md#artifact-envelope-and-atomic-publication), and [canonical](../../../../glossary.md#term-canonical-form) output.
- Java and native schema-eight validation must agree on rank-two Shapes, exact contraction,
  positive checked geometry, profile, value states, and authenticated local transpose provenance.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [Shapes glossary](../../../../glossary.md#shape); [contraction glossary](../../../../glossary.md#contraction-dimension); [transpose glossary](../../../../glossary.md#term-transpose); [provenance glossary](../../../../glossary.md#provenance))</span>
  [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) produces canonical state. Unknown wire `16+`, malformed records, or profile-incompatible
  programs reject before persistent resource publication.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication))</span>
- Advance workload-signature, exact-policy, candidate, compatibility, route-policy, and codec
  identities <span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec))</span>`8 -> 9`. Identity includes profile/schema, ordered Shapes/descriptors/states,
  transpose provenance, gradient flags, topology/targets, ABI, and splats. Old, cross-profile,
  changed-Shape, corrupt, stale, or foreign data rejects or misses safely.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [gradient glossary](../../../../glossary.md#term-gradient); [topology glossary](../../../../glossary.md#term-topology); [targets glossary](../../../../glossary.md#term-target))</span>
- The whole-partition MPSGraph route is the only MATMUL route. Analysis declares exact boundary
  buffers and address workspace; finalization transactionally creates one reusable Shape-
  specialized executable; cold binding validates representations; hot execution performs one
  synchronous submission. Preserve direct targets, leases, rollback, cleanup, reuse, concurrency,
  session/context isolation, and close rejection. Add no fallback, retry, late route choice, copy,
  packing, or hidden host GEMM/transpose.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [buffers glossary](../../../../glossary.md#term-buffer); [binding glossary](../../../../glossary.md#term-binding); [synchronous glossary](../../../../glossary.md#term-synchronous); [leases glossary](../../../../glossary.md#term-lease); [rollback glossary](../../../../glossary.md#term-rollback); [concurrency glossary](../../../../glossary.md#term-concurrency); [host glossary](../../../../glossary.md#term-host); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>

## Forward and explicitly seeded backward evidence

Through public [`Engine`](../../../../glossary.md#term-engine) with only the [Metal](../../../../glossary.md#term-metal) integration and [`ACCELERATOR`](../../../../glossary.md#term-accelerator), execute and materialize:

1. a direct non-square [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) forward [MATMUL](../../../../glossary.md#matrix-multiplication--matmul);
2. no-[bias](../../../../glossary.md#term-bias) [`linear`](../../../../glossary.md#term-linear-meanings) with its exact local right `PERMUTE [1,0] -> MATMUL` chain; and
3. `engine.compile(List.of(output), List.of(seed), List.of(left, right))` for canonical rank-two
   operands and a caller-supplied canonical <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form))</span>`[M,N]` [cotangent seed](../../../../glossary.md#primal-adjoint-and-differentiation-seed).

Inspect the [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel) and prove the two requested [gradients](../../../../glossary.md#term-gradient) are exactly the current [Compiler](../../../../glossary.md#term-compiler)
formulas `seed @ transpose(right)` and `transpose(left) @ seed`, with only rank-two local
`PERMUTE [1,0]` and MATMUL [nodes](../../../../glossary.md#node), no reduction/unbroadcast, hidden copy, [CPU](../../../../glossary.md#term-cpu) [partition](../../../../glossary.md#partition), or implicit
[seed](../../../../glossary.md#term-seed). Execute the real [dylib](../../../../glossary.md#term-dylib), materialize forward plus both gradient publications, and compare each
[contraction](../../../../glossary.md#contraction-dimension) to the same permitted-set [oracle](../../../../glossary.md#term-test-oracle). Cover repeated and independent sessions, direct
[publication](../../../../glossary.md#term-publication), input preservation, close rejection, and strict-profile [ownership](../../../../glossary.md#term-ownership-meanings) failure.

This is a seeded first-order execution [checkpoint](../../../../glossary.md#term-checkpoint), not [scalar](../../../../glossary.md#term-scalar)-[objective](../../../../glossary.md#term-objective) training. Do not claim
[`TrainingSession`](../../../../glossary.md#training-session--trainingsession), [optimizer](../../../../glossary.md#term-optimizer) execution, convergence, implicit scalar [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings), [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) unbroadcasting,
or general Metal backward support.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([automatic differentiation glossary](../../../../glossary.md#autograd))</span>
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span> and
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>

If the local-[transpose](../../../../glossary.md#term-transpose) rule requires a shared representation/capability change, or implementation
requires reduced [precision](../../../../glossary.md#term-precision), term [loss](../../../../glossary.md#loss), a new export, wider transfer, [host](../../../../glossary.md#term-host) fallback, [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace)
profile state, or a [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler) semantic change, stop and report the conflict.

## Dependencies and integration

- Depends on: [Metal](../../../../glossary.md#term-metal) 0020; [Model](../../../../glossary.md#term-model) 0027; Config 0006; [Engine](../../../../glossary.md#term-engine) 0018; not Metal 0009/0016/0017
- Conflicts with: any Metal 0016 restart and every Metal capability/preparation/native schema/
  candidate/codec/materialization/public Engine scope<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None
- Common base revision: `5b77c74239f379e424f8ac0c0c22d65e9a21bab0` plus this [planning](../../../../glossary.md#term-planning)-only authorization
- Integration order: Serial after Metal 0020; fresh oracle before production edits; schema/identity,
  capability/topology, lifecycle, real forward/backward Engine proof, docs, independent review<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [identity glossary](../../../../glossary.md#term-identity-meanings); [topology glossary](../../../../glossary.md#term-topology); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Integration [validation](../../../../glossary.md#term-validation): [accelerator](../../../../glossary.md#term-accelerator) full-[contraction](../../../../glossary.md#contraction-dimension) [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) Class C [checkpoint](../../../../glossary.md#term-checkpoint)
- Shared-document integration [owner](../../../../glossary.md#term-owner): task implementer, finalized by independent Class C review

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalNegPreparationPlan` — exact
  profile/rank/Shape/state/topology domain and lowering.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [Shape glossary](../../../../glossary.md#shape); [topology glossary](../../../../glossary.md#term-topology); [lowering glossary](../../../../glossary.md#lowering))</span>
- `MetalMpsGraphProgram`, `MetalNativeApi`, `MetalDeviceContext`, native foundation — schema eight,
  MATMUL wire 15, selector, validation, ABI-v4 stability, and real execution.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [validation glossary](../../../../glossary.md#term-validation); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Metal candidate/tuning/codec/decision types — version-nine deterministic identity and stale
  rejection; prepared resource/finalizer/executable owners retain the current lifecycle.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [identity glossary](../../../../glossary.md#term-identity-meanings); [prepared resource glossary](../../../../glossary.md#prepared-resource--preparedresource); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Focused capability/[schema](../../../../glossary.md#term-schema)/raw-native/prepared/candidate tests, [Metal](../../../../glossary.md#term-metal) [conformance](../../../../glossary.md#term-conformance),
  `GradientRulesTest`, and `EngineExplicitCompositionMetalIntegrationTest` — fail-closed boundaries,
  current formulas, and real seeded forward/backward evidence.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- Affected Javadocs, Metal/native guides, API/user/capability/preparer status, glossary, scoped
  backend-execution current-capability text, and synchronized planning. Update only the Metal
  current-set statement in the owning scoped contract after executable behavior passes; no Model
  semantics, architecture root/rule, ADR, Runtime, Trace, dependency, or public API signature
  changes are authorized.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [backend glossary](../../../../glossary.md#backend); [planning glossary](../../../../glossary.md#term-planning); [Model glossary](../../../../glossary.md#term-model); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Runtime glossary](../../../../glossary.md#term-runtime); [Trace glossary](../../../../glossary.md#trace); [dependency glossary](../../../../glossary.md#term-dependency); [signature glossary](../../../../glossary.md#term-signature))</span>

## Acceptance criteria

1. The fresh real-M3 probe passes the full direct/transpose/Shape/numerical/control matrix and
   complete permitted-set oracle before edits; exact totals and artifact removal are recorded.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [oracle glossary](../../../../glossary.md#term-test-oracle); [artifact glossary](../../../../glossary.md#term-artifact))</span>
2. Strict MATMUL stays false. Accelerator admits exactly positive static same-type FLOAT32 rank-two
   MATMUL and authenticated local <span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [Accelerator glossary](../../../../glossary.md#term-accelerator); [FLOAT32 glossary](../../../../glossary.md#term-float32); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`[1,0]` [transpose](../../../../glossary.md#term-transpose) inputs; every excluded form fails closed.
3. Every output contains every declared contraction term exactly once and belongs to the Model
   DAZ/FTZ/reassociation/FMA set; no reduced precision, tolerance, term loss, cross-node
   contraction, fallback, or repair exists.<span data-glossary-navigation="true"> ([contraction glossary](../../../../glossary.md#contraction-dimension); [Model glossary](../../../../glossary.md#term-model); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [reassociation glossary](../../../../glossary.md#term-reassociation); [FMA glossary](../../../../glossary.md#term-fused-multiply-add-fma); [precision glossary](../../../../glossary.md#term-precision); [tolerance glossary](../../../../glossary.md#term-tolerance); [loss glossary](../../../../glossary.md#loss); [node glossary](../../../../glossary.md#node))</span>
4. ABI v4 retains thirteen exports. Schema eight appends only wire 15; version-nine identity rejects
   old, cross-profile, malformed, corrupt, stale, changed-Shape, and foreign data.<span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [Schema glossary](../../../../glossary.md#term-schema); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
5. Native/Java lifecycle proof covers direct and local-transpose forms, targets, input preservation,
   reuse, concurrency, sessions/contexts, rollback, cleanup, and close rejection.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [targets glossary](../../../../glossary.md#term-target); [concurrency glossary](../../../../glossary.md#term-concurrency); [rollback glossary](../../../../glossary.md#term-rollback))</span>
6. A real CPU-free Engine executes oracle-checked direct/linear forward and explicitly seeded
   backward publications for both operands. Graph inspection proves only the planned rank-two
   PERMUTE/MATMUL formulas; strict ownership fails.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [linear glossary](../../../../glossary.md#term-linear-meanings); [Graph glossary](../../../../glossary.md#graph); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
7. Focused/native/conformance/Compiler/Engine/architecture/full-build and documentation/diff checks
   pass; independent Class C review has no unresolved finding.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Compiler glossary](../../../../glossary.md#term-compiler))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker validation:

```bash
xcrun clang -fobjc-arc -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph /tmp/synaptik-metal-0018-matmul-probe.m -o /tmp/synaptik-metal-0018-matmul-probe
/tmp/synaptik-metal-0018-matmul-probe
rm -f /tmp/synaptik-metal-0018-matmul-probe.m /tmp/synaptik-metal-0018-matmul-probe
./native/metal-macos-arm64/build.sh
nm -gU native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :backends:metal:test :backends:metal:javadoc
./gradlew :modules:compiler:test --tests '*GradientRulesTest*'
./gradlew :testing:backend-conformance:test --tests '*Metal*'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" ./gradlew :testing:integration-tests:test --tests '*EngineExplicitCompositionMetalIntegrationTest*'
./gradlew :testing:architecture-tests:test
./gradlew build
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

Record probe totals/removal, [oracle](../../../../glossary.md#term-test-oracle) cardinalities and term [masks](../../../../glossary.md#term-boolean-mask), exports, [schema](../../../../glossary.md#term-schema)/[identity](../../../../glossary.md#term-identity-meanings) versions,
test counts/skips, strict/profile negatives, changed paths, and status/frontier checks. The worker
owns one full validation [checkpoint](../../../../glossary.md#term-checkpoint); review reruns executable checks only for changed code or
concretely stale evidence.

## Documentation and review impact

Update affected Javadocs, [Metal](../../../../glossary.md#term-metal)/native guides, [API](../../../../glossary.md#term-application-programming-interface-api)/user/capability/preparer status, targeted
glossary, the owning [backend](../../../../glossary.md#backend)-execution contract's current Metal capability set, and synchronized
[planning](../../../../glossary.md#term-planning) after executable behavior passes. No architecture rule or [ADR](../../../../glossary.md#term-architecture-decision-record-adr) changes. Independent Class
C review inspects the fresh [oracle](../../../../glossary.md#term-test-oracle), all-term/reduced-[precision](../../../../glossary.md#term-precision) controls, strict-false [matrix](../../../../glossary.md#term-matrix-meanings), local
[transpose](../../../../glossary.md#term-transpose) [authentication](../../../../glossary.md#term-authentication), [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings), [lifecycle](../../../../glossary.md#lifecycle) and cleanup, real seeded two-operand
[Engine](../../../../glossary.md#term-engine) proof, [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) absence, changed [scope](../../../../glossary.md#term-scope), and probe removal.

## Result

Blocked by the mandatory fresh Apple M3 [oracle](../../../../glossary.md#term-test-oracle) before production edits. The non-square direct
[smoke](../../../../glossary.md#term-smoke-test) passed. In the first full direct context-zero/optimization-zero `K=1` cell,
`+0.0f * -1.0f` published [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero), outside this task's unchanged permitted set containing
only [negative zero](../../../../glossary.md#nan-infinity-and-signed-zero). The reduced-[precision](../../../../glossary.md#term-precision)-fast-math-none set/read, selector, direct [target](../../../../glossary.md#term-target),
[bindings](../../../../glossary.md#term-binding), and controls were valid; the failure is neither [subnormal](../../../../glossary.md#term-subnormal) handling nor a K=1
[reassociation](../../../../glossary.md#term-reassociation)/[FMA](../../../../glossary.md#term-fused-multiply-add-fma) choice. The disposable probe was removed, the tree is clean, and no production,
test, native, [schema](../../../../glossary.md#term-schema), capability, [identity](../../../../glossary.md#term-identity-meanings), or documentation implementation change remains.
[Model](../../../../glossary.md#term-model) 0029 later completed the separate semantic decision at `30826783`. Reverified [Metal](../../../../glossary.md#term-metal) 0021,
not this task, is now the [Ready](../../../../glossary.md#term-ready) implementation successor and requires a completely fresh full
oracle from the beginning. This historical task remains Blocked and is not reopened.

Status: Incomplete
Follow-up required: execute Ready Metal 0021 under the refined final-[publication](../../../../glossary.md#term-publication) contract and its
completely fresh full-oracle gate; do not reopen this historical task.
