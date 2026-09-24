# Task 0018: ACCELERATOR FLOAT32 Rank-Two MATMUL Seeded-Gradient Checkpoint

## Status

Blocked

Metal 0020 is Complete at implementation `9ddb75f6` plus documentation remediation `5b77c742`.
Model 0027, Config 0006, and Engine 0018 are Complete. Historical Metal 0009 and Metal 0017 remain
Blocked records, not dependencies; Metal 0016 has no active restart, and Model 0026 remains an
independent Draft. The mandatory fresh real-M3 probe ran from clean base
`0625a18727ac2cf59220cfa7db5de93e1cac2651` and stopped before any production edit. The smoke
passed, but the first full direct context-zero/optimization-zero `K=1` cell evaluated
`+0.0f * -1.0f` as positive zero, while this task's unchanged Model set permits only negative zero.
`MPSGraphReducedPrecisionFastMathNone` set/read and the selector, direct targets, and controls were
valid. Zero is not subnormal, and a one-term contraction has no reassociation or multiply-add
choice that changes the required sign. The probe was removed and the tree returned clean. ABI v4
still has exactly thirteen exports, node schema is seven, and private identities are version eight.
No production/schema/capability change from this task exists.

## Change class

Class C — this adds profile-qualified contraction capability, local affine-view topology, native
schema/cache identity, prepared execution, and public Engine forward/backward behavior across the
native boundary. A fresh real-device oracle and independent final review are mandatory.

## Goal

Add only positive, static, same-type `FLOAT32` rank-two `MATMUL` under
`NumericalProfile.ACCELERATOR`, including exact local rank-two transpose composition and
Compiler-generated explicitly seeded first-order gradients for both operands. Admit only the
Model-owned reassociation/FMA plus row-scoped DAZ/FTZ result set. Preserve every contraction term,
never use reduced precision, and keep strict Metal `MATMUL` false.

## Exact capability and topology

- Admit exactly `MatmulKind.MATMUL` with `NoOperationAttrs.INSTANCE`, two ordered inputs, and one
  output. Shapes are `[M,K]`, `[K,N]`, and `[M,N]`, with positive static `M`, `K`, and `N`; all
  element/byte counts and native conversions use checked arithmetic.
- Inputs and output are same-type `FLOAT32`. The canonical output is dense, zero-offset, and
  non-view; its `requiresGrad` is the logical OR of the input flags. Input flags need not match.
- Each operand is either a canonical dense zero-offset non-view or the exact rank-two descriptor
  produced by one earlier in-partition `PERMUTE [1,0]` from a canonical source. Authenticate the
  exact producer, axes, value identity, topology, and represented logical order. The transpose
  cannot be a partition feed, foreign transfer, constant, arbitrary view, other affine form, or
  hidden host/materialization step.
- The exact local transpose may occur on the left, right, or both operands. This includes public
  no-bias `linear`, whose visible chain is `PERMUTE [1,0] -> MATMUL`; no `LINEAR` operation or
  pattern rewrite is introduced.
- A canonical MATMUL output may be published or consumed by already-admitted accelerator nodes.
  No cross-node contraction is allowed. Complete-partition analysis must reject malformed local
  transpose provenance before native creation.
- Strict Metal rejects every MATMUL occurrence. Accelerator rejects rank other than two, vectors,
  batches/broadcasts, zero/dynamic extents, other types/promotions, wrong attrs/Shape/layout,
  nonlocal views, unchecked geometry, and malformed topology.

## Numerical contract and mandatory fresh Apple M3 oracle

Use only `matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:`. Before production edits,
create a new disposable Objective-C probe under `/tmp`, compile it with `xcrun clang` against
Foundation/Metal/MPSGraph, run it on the real M3, and remove source and binary. Do not reuse the
historical Metal 0009 probe or results.

Require macOS 26's `reducedPrecisionFastMath` control and set it to
`MPSGraphReducedPrecisionFastMathNone` at optimization levels zero and one; inability to set that
exact control blocks the task. Use two fresh contexts, three independently created executables per
cell per context, eight runs per executable, guarded caller-supplied direct targets, permuted feed/
target bindings, retained native names/Shapes, and cell-scoped autorelease pools. First pass one
non-square direct smoke cell. Then exercise direct, left-transposed, right-transposed, and both-
transposed topology forms with at least eight Shape/data cells per form: non-square ordinary,
`M=1`, `K=1`, `N=1`, exact integers, all-term sentinels, cancellation/tree-sensitive values, and
subnormal/normal/special-value boundaries. This is at least 384 executables and 3,072 runs, in
addition to the smoke cell and run; report exact totals.

For every bounded output dot product, an independent raw-bit oracle must enumerate the complete
Model-permitted set: every declared `k` contribution exactly once; every reassociation/binary tree;
`FLOAT32` rounding; only corresponding multiply-add contraction by FMA; row-scoped DAZ of declared
subnormal arithmetic inputs; and optional FTZ of finite subnormal arithmetic results. Include both
unfused and every legal fused placement. An FMA cannot cross nodes or erase an observable value.
No contribution may be dropped, duplicated, replaced, or treated as an implicit reduced-precision
term. Mantissa-sensitive cells must distinguish full FLOAT32 products/accumulation from reduced
precision. Term-sentinel cells must make omission of each individual `k` term observable.

Cover signed zeros; minimum/ordinary/maximum subnormals; minimum normals and neighbors; ordinary
and maximum finite values; overflow; infinities; signed quiet/signaling NaNs; `K=1` multiplication
by one; cancellation and tree-sensitive permutations. Require raw-bit membership when the allowed
set is finite and exact result-class membership only where current Model NaN representation is
unconstrained. Epsilon, CPU-output comparison, tolerance, term-loss acceptance, reciprocal or
algebraic substitution, undocumented fast math, and special-value repair are forbidden. Any
selector, Shape, binding, canary, input-preservation, transpose, full-term, precision, or oracle
failure blocks 0018 without production/schema/capability edits.

## Native schema, identity, lifecycle, and cache

- Keep ABI v4, statuses, create/run signatures, and exactly thirteen exports. Evolve the fixed
  160-byte node schema `7 -> 8`; retain wires `1..14`; append only `MATMUL=15` with two ordered
  inputs, `NONE` attributes, required sentinels, zero count/reserved/payload, and canonical output.
- Java and native schema-eight validation must agree on rank-two Shapes, exact contraction,
  positive checked geometry, profile, value states, and authenticated local transpose provenance.
  `MATMUL` produces canonical state. Unknown wire `16+`, malformed records, or profile-incompatible
  programs reject before persistent resource publication.
- Advance workload-signature, exact-policy, candidate, compatibility, route-policy, and codec
  identities `8 -> 9`. Identity includes profile/schema, ordered Shapes/descriptors/states,
  transpose provenance, gradient flags, topology/targets, ABI, and splats. Old, cross-profile,
  changed-Shape, corrupt, stale, or foreign data rejects or misses safely.
- The whole-partition MPSGraph route is the only MATMUL route. Analysis declares exact boundary
  buffers and address workspace; finalization transactionally creates one reusable Shape-
  specialized executable; cold binding validates representations; hot execution performs one
  synchronous submission. Preserve direct targets, leases, rollback, cleanup, reuse, concurrency,
  session/context isolation, and close rejection. Add no fallback, retry, late route choice, copy,
  packing, or hidden host GEMM/transpose.

## Forward and explicitly seeded backward evidence

Through public `Engine` with only the Metal integration and `ACCELERATOR`, execute and materialize:

1. a direct non-square rank-two forward MATMUL;
2. no-bias `linear` with its exact local right `PERMUTE [1,0] -> MATMUL` chain; and
3. `engine.compile(List.of(output), List.of(seed), List.of(left, right))` for canonical rank-two
   operands and a caller-supplied canonical `[M,N]` cotangent seed.

Inspect the compiled graph and prove the two requested gradients are exactly the current Compiler
formulas `seed @ transpose(right)` and `transpose(left) @ seed`, with only rank-two local
`PERMUTE [1,0]` and MATMUL nodes, no reduction/unbroadcast, hidden copy, CPU partition, or implicit
seed. Execute the real dylib, materialize forward plus both gradient publications, and compare each
contraction to the same permitted-set oracle. Cover repeated and independent sessions, direct
publication, input preservation, close rejection, and strict-profile ownership failure.

This is a seeded first-order execution checkpoint, not scalar-objective training. Do not claim
`TrainingSession`, optimizer execution, convergence, implicit scalar seeds, batch unbroadcasting,
or general Metal backward support.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle) and
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)

If the local-transpose rule requires a shared representation/capability change, or implementation
requires reduced precision, term loss, a new export, wider transfer, host fallback, Runtime/Trace
profile state, or a Model/Compiler semantic change, stop and report the conflict.

## Dependencies and integration

- Depends on: Metal 0020; Model 0027; Config 0006; Engine 0018; not Metal 0009/0016/0017
- Conflicts with: any Metal 0016 restart and every Metal capability/preparation/native schema/
  candidate/codec/materialization/public Engine scope
- Parallel group: None
- Common base revision: `5b77c74239f379e424f8ac0c0c22d65e9a21bab0` plus this planning-only authorization
- Integration order: Serial after Metal 0020; fresh oracle before production edits; schema/identity,
  capability/topology, lifecycle, real forward/backward Engine proof, docs, independent review
- Integration validation: accelerator full-contraction rank-two MATMUL Class C checkpoint
- Shared-document integration owner: task implementer, finalized by independent Class C review

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalNegPreparationPlan` — exact
  profile/rank/Shape/state/topology domain and lowering.
- `MetalMpsGraphProgram`, `MetalNativeApi`, `MetalDeviceContext`, native foundation — schema eight,
  MATMUL wire 15, selector, validation, ABI-v4 stability, and real execution.
- Metal candidate/tuning/codec/decision types — version-nine deterministic identity and stale
  rejection; prepared resource/finalizer/executable owners retain the current lifecycle.
- Focused capability/schema/raw-native/prepared/candidate tests, Metal conformance,
  `GradientRulesTest`, and `EngineExplicitCompositionMetalIntegrationTest` — fail-closed boundaries,
  current formulas, and real seeded forward/backward evidence.
- Affected Javadocs, Metal/native guides, API/user/capability/preparer status, glossary, scoped
  backend-execution current-capability text, and synchronized planning. Update only the Metal
  current-set statement in the owning scoped contract after executable behavior passes; no Model
  semantics, architecture root/rule, ADR, Runtime, Trace, dependency, or public API signature
  changes are authorized.

## Acceptance criteria

1. The fresh real-M3 probe passes the full direct/transpose/Shape/numerical/control matrix and
   complete permitted-set oracle before edits; exact totals and artifact removal are recorded.
2. Strict MATMUL stays false. Accelerator admits exactly positive static same-type FLOAT32 rank-two
   MATMUL and authenticated local `[1,0]` transpose inputs; every excluded form fails closed.
3. Every output contains every declared contraction term exactly once and belongs to the Model
   DAZ/FTZ/reassociation/FMA set; no reduced precision, tolerance, term loss, cross-node
   contraction, fallback, or repair exists.
4. ABI v4 retains thirteen exports. Schema eight appends only wire 15; version-nine identity rejects
   old, cross-profile, malformed, corrupt, stale, changed-Shape, and foreign data.
5. Native/Java lifecycle proof covers direct and local-transpose forms, targets, input preservation,
   reuse, concurrency, sessions/contexts, rollback, cleanup, and close rejection.
6. A real CPU-free Engine executes oracle-checked direct/linear forward and explicitly seeded
   backward publications for both operands. Graph inspection proves only the planned rank-two
   PERMUTE/MATMUL formulas; strict ownership fails.
7. Focused/native/conformance/Compiler/Engine/architecture/full-build and documentation/diff checks
   pass; independent Class C review has no unresolved finding.

## Validation

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

Record probe totals/removal, oracle cardinalities and term masks, exports, schema/identity versions,
test counts/skips, strict/profile negatives, changed paths, and status/frontier checks. The worker
owns one full validation checkpoint; review reruns executable checks only for changed code or
concretely stale evidence.

## Documentation and review impact

Update affected Javadocs, Metal/native guides, API/user/capability/preparer status, targeted
glossary, the owning backend-execution contract's current Metal capability set, and synchronized
planning after executable behavior passes. No architecture rule or ADR changes. Independent Class
C review inspects the fresh oracle, all-term/reduced-precision controls, strict-false matrix, local
transpose authentication, schema/cache identity, lifecycle and cleanup, real seeded two-operand
Engine proof, Runtime/Trace absence, changed scope, and probe removal.

## Result

Blocked by the mandatory fresh Apple M3 oracle before production edits. The non-square direct
smoke passed. In the first full direct context-zero/optimization-zero `K=1` cell,
`+0.0f * -1.0f` published positive zero, outside this task's unchanged permitted set containing
only negative zero. The reduced-precision-fast-math-none set/read, selector, direct target,
bindings, and controls were valid; the failure is neither subnormal handling nor a K=1
reassociation/FMA choice. The disposable probe was removed, the tree is clean, and no production,
test, native, schema, capability, identity, or documentation implementation change remains.
Model 0029 later completed the separate semantic decision at `30826783`. Reverified Metal 0021,
not this task, is now the Ready implementation successor and requires a completely fresh full
oracle from the beginning. This historical task remains Blocked and is not reopened.

Status: Incomplete
Follow-up required: execute Ready Metal 0021 under the refined final-publication contract and its
completely fresh full-oracle gate; do not reopen this historical task.
