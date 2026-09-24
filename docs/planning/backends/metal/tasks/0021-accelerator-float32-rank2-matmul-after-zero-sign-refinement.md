# Task 0021: ACCELERATOR FLOAT32 Rank-Two MATMUL After Exact-Zero Sign Refinement

## Status

Ready

Implementation checkpoint: the completely fresh disposable Apple M3 Max oracle passed before
production edits and was removed. Metal remains ABI v4 with exactly thirteen exports; the
implementation advances the fixed node schema to 8 with `MATMUL=15` and advances
workload/exact-policy/candidate/compatibility/route/codec identities to version nine. Historical
Metal 0018 remains Blocked without active write scope; Metal 0016 has no restart. Model 0029 is
Complete at `308267837a17c2fdbeb8b90ad381d1f899980786`, and Metal 0020, Model 0027, Config
0006, and Engine 0018 remain Complete. This task remains Ready only because mandatory independent
Class C review has not yet run; no earlier 0018 probe artifact or partial cell was reused.

## Change class

Class C — this adds profile-qualified contraction capability, local affine-view topology, native
schema/cache identity, prepared execution, and public Engine forward/backward behavior across the
native boundary. A full fresh real-device oracle and independent final review are mandatory.

## Goal

Add only positive, static, same-type `FLOAT32` rank-two `MATMUL` under
`NumericalProfile.ACCELERATOR`, including exact local rank-two transpose composition and
Compiler-generated explicitly seeded first-order gradients for both operands. Admit only the
refined Model set: existing DAZ/FTZ, reassociation, and corresponding FMA choices plus final-
publication either-zero-sign freedom for an exact-zero complete nonempty MATMUL result. Strict
Metal MATMUL stays false.

## Scope

- Admit exactly `MatmulKind.MATMUL` with `NoOperationAttrs.INSTANCE`, Shapes `[M,K]`, `[K,N]`, and
  `[M,N]`, positive static checked `M`, `K`, `N`, same-type FLOAT32, and canonical output.
- Accept each input only as canonical dense zero-offset non-view or the authenticated exact local
  rank-two `PERMUTE [1,0]` view of a canonical source. Reject foreign, feed, constant, arbitrary,
  malformed, or hidden-materialization view forms.
- Support direct, left-, right-, and both-transposed forms, including visible no-bias
  `PERMUTE -> MATMUL` linear composition. Add no `LINEAR` operation or cross-node contraction.
- Preserve ABI v4 and exactly thirteen exports. Evolve the fixed 160-byte node schema `7 -> 8`,
  retain wires `1..14`, append only `MATMUL=15`, and advance workload-signature, exact-policy,
  candidate, compatibility, route-policy, and codec identities `8 -> 9`.
- Use the whole-partition MPSGraph route, exact shared-resource declaration, transactional reusable
  preparation, cold authenticated binding, and one synchronous hot submission with no fallback,
  retry, copy, packing, host GEMM, or late route choice.
- Through the real CPU-free Engine and real dylib, prove direct non-square forward, no-bias
  `linear` with its visible right transpose, and
  `engine.compile(List.of(output), List.of(seed), List.of(left, right))` with a caller-supplied
  canonical `[M,N]` cotangent seed. Inspect and execute exactly `seed @ transpose(right)` and
  `transpose(left) @ seed`, with only local rank-two `PERMUTE [1,0]` and MATMUL nodes.

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
binary reassociation tree, FLOAT32 rounding, every corresponding legal unfused/FMA placement,
row-scoped DAZ, optional FTZ of finite subnormal arithmetic results, and the completed Model
final-publication rule. Product and intermediate exact-zero signs remain fixed by the selected
evaluation; DAZ/FTZ and each legal FMA/reassociation choice must be modeled explicitly, never
inferred through a tolerance. Only after the complete nonempty otherwise-permitted arithmetic
result is exact zero may publication choose either sign. No implicit accumulator identity exists,
including for `K=1`. Cover the prior `+0.0f * -1.0f` witness explicitly: the oracle must retain its
negative-zero product and admit positive zero only at publication, never as a product or
intermediate result.

Cover signed zeros, minimum/ordinary/maximum subnormals, minimum normals and neighbors, ordinary
and maximum finite values, overflow, infinities, signed quiet/signaling NaNs, multiplication by
one, cancellation, and tree-sensitive permutations. Mantissa and per-term sentinels must reject
reduced precision or any omitted/duplicated term. Require raw-bit membership for finite exact sets
and only current Model representation freedom within an unchanged NaN classification. Any
reduced-precision-none, DAZ/FTZ, FMA/reassociation, control, binding, Shape, transpose, full-term,
precision, nonzero/classification, or oracle failure blocks without production edits.

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
- Common base revision: `308267837a17c2fdbeb8b90ad381d1f899980786` plus this planning-only
  authorization
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

1. A completely fresh full real-M3 oracle starts from the beginning, passes every numerical/
   control/topology cell before edits, reports exact totals, proves the prior K=1 witness only
   through final-publication freedom, and is removed.
2. Strict stays false; accelerator admits exactly positive static same-type FLOAT32 rank-two MATMUL
   with authenticated local transpose inputs; all excluded forms fail closed.
3. Every output uses every term exactly once and belongs to the refined Model set; products and
   intermediates receive no new sign freedom, reduced precision, tolerance, or repair.
4. ABI v4 retains thirteen exports; schema eight appends only wire 15; version-nine identities
   reject old, cross-profile, malformed, corrupt, stale, changed-Shape, and foreign data.
5. Native/Java lifecycle proof covers targets, inputs, transpose forms, reuse, concurrency,
   sessions/contexts, rollback, cleanup, and close rejection.
6. A real CPU-free Engine executes oracle-checked direct/linear forward and seeded gradients for
   both operands; graph inspection proves the planned rank-two PERMUTE/MATMUL formulas beneath the
   existing shape-restoration boundaries.
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

Implementation and worker validation are complete; status remains `Ready` pending one independent
Class C review. Before production edits, a fresh disposable Objective-C probe passed:

```text
PASS device=Apple M3 Max smoke_executables=1 smoke_runs=1 full_executables=384
full_runs=3072 oracle_cells=18441 oracle_normalizations=30729
oracle_raw_operations=4334198 oracle_members=31498 term_mask=0x1f
k1_product=-0 k1_publication=both opt=0,1 contexts=2 forms=4
reduced_precision=none
```

The matrix covered direct, left-transposed, right-transposed, and both-transposed forms; two
optimization levels and two contexts; three independently compiled executables per full cell and
eight runs per executable; Shape/data, stable/permuted binding, guarded-target, DAZ/FTZ,
reassociation/FMA, exact-zero publication, all-term, and reduced-precision controls. The prior
`K=1` witness retained product `-0` and passed only through Model 0029's final-publication
either-zero-sign freedom. The source and binary were removed before production edits.

The implementation adds the exact accelerator-only capability, local transpose authentication,
schema-eight wire `15`, Java/native fail-closed geometry and topology validation, direct MPSGraph
matrix multiplication with reduced-precision-none readback, version-nine identities, reusable
prepared execution, and CPU-free direct/linear/seeded-gradient Engine coverage. The seeded graph
inspection proves `seed @ transpose(right)` and `transpose(left) @ seed` beneath the existing
shape-restoration boundaries; real runs publish raw-bit-checked forward and both operand gradients
with no CPU owner. Strict MATMUL remains false. No Runtime, Trace, public API, ABI export,
packaging, discovery, scalar-loss training, batched MATMUL, or broader backward scope was added.
