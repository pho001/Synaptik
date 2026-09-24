# Task 0021: ACCELERATOR FLOAT32 Rank-Two MATMUL After Exact-Zero Sign Refinement

## Status

Draft

This task cannot become Ready until Model 0029 is Complete with its proof, validation, synchronized
documentation, and independent Class C approval. Historical Metal 0018 remains Blocked under its
unchanged contract after the mandatory probe stopped before production edits. Promotion must set a
clean post-Model-0029 base and reverify every dependency, conflict, owner path, and oracle control.
No part of the Metal 0018 probe or its partial result may satisfy this task's full fresh oracle.

## Change class

Class C — this adds profile-qualified contraction capability, local affine-view topology, native
schema/cache identity, prepared execution, and public Engine forward/backward behavior across the
native boundary. A full fresh real-device oracle and independent final review are mandatory.

## Goal

After Model 0029 completes, add only positive, static, same-type `FLOAT32` rank-two `MATMUL` under
`NumericalProfile.ACCELERATOR`, including exact local rank-two transpose composition and
Compiler-generated explicitly seeded first-order gradients for both operands. Admit only the
refined Model set: existing DAZ/FTZ, reassociation, and corresponding FMA choices plus final-
publication either-zero-sign freedom for an exact-zero nonempty MATMUL result. Strict Metal MATMUL
stays false.

## Scope

- Admit exactly `MatmulKind.MATMUL` with `NoOperationAttrs.INSTANCE`, Shapes `[M,K]`, `[K,N]`, and
  `[M,N]`, positive static checked `M`, `K`, `N`, same-type FLOAT32, and canonical output.
- Accept each input only as canonical dense zero-offset non-view or the authenticated exact local
  rank-two `PERMUTE [1,0]` view of a canonical source. Reject foreign, feed, constant, arbitrary,
  malformed, or hidden-materialization view forms.
- Support direct, left-, right-, and both-transposed forms, including visible no-bias
  `PERMUTE -> MATMUL` linear composition. Add no `LINEAR` operation or cross-node contraction.
- Preserve ABI v4 and exactly thirteen exports. Evolve fixed node schema `7 -> 8`, append only
  `MATMUL=15`, and advance workload/candidate/compatibility/route/codec identities `8 -> 9`.
- Use the whole-partition MPSGraph route, exact shared-resource declaration, transactional reusable
  preparation, cold authenticated binding, and one synchronous hot submission with no fallback,
  retry, copy, packing, host GEMM, or late route choice.
- Prove direct and linear forward plus explicitly seeded gradients `seed @ transpose(right)` and
  `transpose(left) @ seed` through the real CPU-free Engine and real dylib.

## Numerical contract and full fresh oracle

Before any production/schema/capability edit, create a new disposable Objective-C probe under
`/tmp`, compile against Foundation/Metal/MPSGraph, run on the real M3, and remove source/binary.
Use only `matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:`. Require macOS 26's
`reducedPrecisionFastMath`, set/read `MPSGraphReducedPrecisionFastMathNone` at optimization levels
zero and one, and block if the exact control is unavailable.

Use two fresh contexts, three independently created executables per cell per context, eight runs
per executable, guarded caller-supplied direct targets, permuted feed/target bindings, retained
native names/Shapes, and cell-scoped autorelease pools. Pass a non-square direct smoke, then run the
complete direct/left-/right-/both-transposed matrix with at least eight Shape/data cells per form:
non-square ordinary, `M=1`, `K=1`, `N=1`, exact integers, every-term sentinels, cancellation/tree-
sensitive values, and subnormal/normal/special boundaries. Require at least 384 newly created
executables and 3,072 runs after the smoke and report exact totals. Start from the beginning; do not
resume after Metal 0018's first failed full cell.

The independent oracle must enumerate each declared `k` product exactly once, every permitted
binary tree, FLOAT32 rounding, every corresponding legal fused placement, row-scoped DAZ, optional
FTZ of finite subnormal arithmetic results, and Model 0029's final-publication rule. Product and
intermediate exact-zero signs remain fixed by the selected evaluation. Only a complete nonempty
MATMUL result that is exact zero may publish either sign. No implicit accumulator identity exists,
including for `K=1`. Cover the prior `+0.0f * -1.0f` case explicitly and prove its underlying
product is negative zero even though either final sign is accepted.

Cover signed zeros, minimum/ordinary/maximum subnormals, minimum normals and neighbors, ordinary
and maximum finite values, overflow, infinities, signed quiet/signaling NaNs, multiplication by
one, cancellation, and tree-sensitive permutations. Mantissa and per-term sentinels must reject
reduced precision or any omitted/duplicated term. Require raw-bit membership for finite exact sets
and only current Model classification freedom for NaN. Any control, binding, Shape, transpose,
full-term, precision, nonzero/classification, or oracle failure blocks without production edits.

## Non-goals

No strict MATMUL; vector, batched/broadcast, zero/dynamic-extent, other-type, promoted, arbitrary-
view, convolution, broader backward/training, packaging/discovery, performance, or mixed-owner
claim. No per-product/intermediate zero-sign choice, term loss, identity insertion, tolerance,
reduced precision, algebraic substitution, special-value repair, cross-node FMA, fallback, Runtime/
Trace profile state, public API signature, dependency, or ABI-export change.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle) and
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)

If implementation needs wider Model semantics, reduced precision, term loss, a shared
representation change, new export, wider transfer, host fallback, or Runtime/Trace profile state,
stop and report the conflict.

## Dependencies and integration

- Depends on: Model 0029 after Complete; Metal 0020; Model 0027; Config 0006; Engine 0018; not Metal
  0009/0016/0017/0018
- Conflicts with: any Metal 0016 restart and every Metal capability/preparation/native schema/
  candidate/codec/materialization/public Engine scope
- Parallel group: None
- Common base revision: N/A; set to the clean integrated Model 0029 revision before promotion
- Integration order: Serial after Model 0029; full fresh oracle before production edits; then
  schema/identity, capability/topology, lifecycle, Engine proof, documentation, independent review
- Integration validation: refined accelerator full-contraction rank-two MATMUL Class C checkpoint
- Shared-document integration owner: task implementer, finalized by independent Class C review

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalNegPreparationPlan` — exact profile,
  Shape/state/topology domain and lowering.
- `MetalMpsGraphProgram`, `MetalNativeApi`, `MetalDeviceContext`, native foundation — schema eight,
  wire 15, validation, ABI stability, and real execution.
- Metal candidate/tuning/codec/decision and prepared lifecycle types — version-nine identity,
  authentication, stale rejection, resources, reuse, and cleanup.
- Focused capability/schema/raw-native/prepared/candidate tests; Metal conformance;
  `GradientRulesTest`; `EngineExplicitCompositionMetalIntegrationTest`.
- Affected Javadocs, Metal/native/backend/API/user guides, targeted glossary, backend-execution
  current capability, and synchronized planning after executable behavior passes.

## Acceptance criteria

1. The full fresh oracle passes every numerical/control/topology cell before edits, reports totals,
   proves the prior K=1 case only through Model 0029 final-publication freedom, and is removed.
2. Strict stays false; accelerator admits exactly positive static same-type FLOAT32 rank-two MATMUL
   with authenticated local transpose inputs; all excluded forms fail closed.
3. Every output uses every term exactly once and belongs to the refined Model set; products and
   intermediates receive no new sign freedom, reduced precision, tolerance, or repair.
4. ABI v4 retains thirteen exports; schema eight appends only wire 15; version-nine identities
   reject old, cross-profile, malformed, corrupt, stale, changed-Shape, and foreign data.
5. Native/Java lifecycle proof covers targets, inputs, transpose forms, reuse, concurrency,
   sessions/contexts, rollback, cleanup, and close rejection.
6. A real CPU-free Engine executes oracle-checked direct/linear forward and seeded gradients for
   both operands; graph inspection proves only planned rank-two PERMUTE/MATMUL formulas.
7. Focused/native/conformance/Compiler/Engine/architecture/full-build/documentation/diff checks pass
   and independent Class C review has no unresolved finding.

## Validation

Worker validation:

```bash
xcrun clang -fobjc-arc -framework Foundation -framework Metal -framework MetalPerformanceShadersGraph /tmp/synaptik-metal-0021-matmul-probe.m -o /tmp/synaptik-metal-0021-matmul-probe
/tmp/synaptik-metal-0021-matmul-probe
rm -f /tmp/synaptik-metal-0021-matmul-probe.m /tmp/synaptik-metal-0021-matmul-probe
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

Record oracle totals/cardinalities/term masks, probe removal, exports, schema/identity versions,
test counts/skips, strict/profile negatives, changed paths, and status/frontier checks. The worker
owns one full checkpoint; review reruns executable checks only after relevant executable changes or
when evidence is concretely stale.

## Documentation and review impact

Finalize affected Javadocs and guides only after executable behavior passes. Independent Class C
review inspects the entire fresh oracle, exact-zero publication boundary, all-term/reduced-precision
controls, strict-false matrix, transpose authentication, schema/cache identity, lifecycle/cleanup,
real seeded Engine proof, Runtime/Trace absence, changed scope, and probe removal.

## Result

Not started.
