# Task 0021: [ACCELERATOR](../../../../glossary.md#term-accelerator) [FLOAT32](../../../../glossary.md#term-float32) [Rank-Two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) After Exact-Zero Sign Refinement

## Status

Complete

Completed at implementation `ef2c6a1a`, worker-evidence remediation `be5543f9`, and final
evidence-count remediation `5631d51f`. The completely fresh disposable Apple M3 Max [oracle](../../../../glossary.md#term-test-oracle) passed
before production edits and was removed. [Metal](../../../../glossary.md#term-metal) remains [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4 with exactly thirteen exports; the
implementation advances the fixed [node](../../../../glossary.md#node) [schema](../../../../glossary.md#term-schema) to 8 with `MATMUL=15` and advances
workload/exact-policy/candidate/compatibility/route/[codec](../../../../glossary.md#term-codec) [identities](../../../../glossary.md#term-identity-meanings) to version nine. Historical
Metal 0018 remains Blocked without active write [scope](../../../../glossary.md#term-scope); Metal 0016 has no restart. [Model](../../../../glossary.md#term-model) 0029 is
Complete at `308267837a17c2fdbeb8b90ad381d1f899980786`, and Metal 0020, Model 0027, Config
0006, and [Engine](../../../../glossary.md#term-engine) 0018 remain Complete. Final independent Class C review returned `APPROVE` with
zero findings after verifying the corrected 15-of-15 Markdown evidence and retained [checkpoint](../../../../glossary.md#term-checkpoint),
ABI/export, probe-removal, path-count, and status evidence without unnecessary broad reruns.

## Change class

Class C — this adds profile-qualified [contraction](../../../../glossary.md#contraction-dimension) capability, local affine-[view](../../../../glossary.md#term-view-meanings) [topology](../../../../glossary.md#term-topology), native
[schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings), [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution), and public [Engine](../../../../glossary.md#term-engine) forward/backward behavior across the
native boundary. A full fresh real-[device](../../../../glossary.md#term-device) [oracle](../../../../glossary.md#term-test-oracle) and independent final review are mandatory.

## Goal

Add only positive, static, same-type [`FLOAT32`](../../../../glossary.md#term-float32) [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) under
`NumericalProfile.ACCELERATOR`, including exact local rank-two [transpose](../../../../glossary.md#term-transpose) composition and
[Compiler](../../../../glossary.md#term-compiler)-generated explicitly seeded first-order [gradients](../../../../glossary.md#term-gradient) for both operands. Admit only the
refined [Model](../../../../glossary.md#term-model) set: existing [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz), [reassociation](../../../../glossary.md#term-reassociation), and corresponding [FMA](../../../../glossary.md#term-fused-multiply-add-fma) choices plus final-
[publication](../../../../glossary.md#term-publication) either-zero-sign freedom for an exact-zero complete nonempty MATMUL result. Strict
[Metal](../../../../glossary.md#term-metal) MATMUL stays false.

## [Scope](../../../../glossary.md#term-scope)

- Admit exactly `MatmulKind.MATMUL` with `NoOperationAttrs.INSTANCE`, [Shapes](../../../../glossary.md#shape) `[M,K]`, `[K,N]`, and
  `[M,N]`, positive static checked `M`, `K`, `N`, same-type [FLOAT32](../../../../glossary.md#term-float32), and [canonical](../../../../glossary.md#term-canonical-form) output.
- Accept each input only as canonical dense zero-offset non-view or the authenticated exact local
  rank-two <span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings))</span>`PERMUTE [1,0]` view of a canonical source. Reject foreign, feed, constant, arbitrary,
  malformed, or hidden-materialization view forms.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant); [materialization glossary](../../../../glossary.md#materialization))</span>
- Support direct, left-, right-, and both-transposed forms, including visible no-[bias](../../../../glossary.md#term-bias)
  `PERMUTE -> MATMUL` [linear](../../../../glossary.md#term-linear-meanings) composition. Add no `LINEAR` [operation](../../../../glossary.md#operation) or cross-[node](../../../../glossary.md#node) [contraction](../../../../glossary.md#contraction-dimension).
- Preserve [ABI](../../../../glossary.md#term-application-binary-interface-abi) v4 and exactly thirteen exports. Evolve the fixed 160-byte node [schema](../../../../glossary.md#term-schema) `7 -> 8`,
  retain wires `1..14`, append only `MATMUL=15`, and advance workload-signature, exact-policy,
  candidate, compatibility, route-policy, and codec identities <span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [codec glossary](../../../../glossary.md#term-codec))</span>`8 -> 9`.
- Use the whole-partition MPSGraph route, exact shared-resource declaration, transactional reusable
  preparation, cold authenticated binding, and one synchronous hot submission with no fallback,
  retry, copy, packing, host GEMM, or late route choice.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [MPSGraph glossary](../../../../glossary.md#term-mpsgraph); [binding glossary](../../../../glossary.md#term-binding); [synchronous glossary](../../../../glossary.md#term-synchronous); [host glossary](../../../../glossary.md#term-host); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- Through the real [CPU](../../../../glossary.md#term-cpu)-free [Engine](../../../../glossary.md#term-engine) and real [dylib](../../../../glossary.md#term-dylib), prove direct non-square forward, no-bias
  `linear` with its visible right [transpose](../../../../glossary.md#term-transpose), and
  `engine.compile(List.of(output), List.of(seed), List.of(left, right))` with a caller-supplied
  canonical `[M,N]` [cotangent seed](../../../../glossary.md#primal-adjoint-and-differentiation-seed). Inspect and execute exactly `seed @ transpose(right)` and
  `transpose(left) @ seed`, with only local [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `PERMUTE [1,0]` and [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) nodes.

## Numerical contract and full fresh [oracle](../../../../glossary.md#term-test-oracle)

Before any production/[schema](../../../../glossary.md#term-schema)/capability edit, create a new disposable [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) probe under
`/tmp`, [compile](../../../../glossary.md#compile) against Foundation/[Metal](../../../../glossary.md#term-metal)/[MPSGraph](../../../../glossary.md#term-mpsgraph), [run](../../../../glossary.md#term-run) on the real M3, and remove source/[binary](../../../../glossary.md#term-binary-descriptor-and-metadata).
Use only `matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:`. Require macOS 26's
`reducedPrecisionFastMath`, set/read `MPSGraphReducedPrecisionFastMathNone` at optimization levels
zero and one, and block if the exact control is unavailable.

Use two fresh contexts, three independently created executables per cell per context, eight runs
per executable, guarded caller-supplied direct [targets](../../../../glossary.md#term-target), permuted feed/target [bindings](../../../../glossary.md#term-binding), retained
native names/[Shapes](../../../../glossary.md#shape), and cell-scoped autorelease pools. Pass a non-square direct [smoke](../../../../glossary.md#term-smoke-test), then run the
complete direct/left-/right-/both-transposed [matrix](../../../../glossary.md#term-matrix-meanings) with at least eight Shape/data cells per form:
non-square ordinary, `M=1`, `K=1`, `N=1`, exact integers, every-term sentinels, [cancellation](../../../../glossary.md#term-cancellation)/tree-
sensitive values, and [subnormal](../../../../glossary.md#term-subnormal)/normal/special boundaries. Require at least 384 newly created
executables and 3,072 runs after the smoke and report exact totals. Start from the beginning; do not
[resume](../../../../glossary.md#schema-serialization-and-checkpoint) after Metal 0018's first failed full cell.

The independent oracle must enumerate each declared `k` product exactly once, every permitted
binary [reassociation](../../../../glossary.md#term-reassociation) tree, [FLOAT32](../../../../glossary.md#term-float32) [rounding](../../../../glossary.md#term-rounding), every corresponding legal unfused/[FMA](../../../../glossary.md#term-fused-multiply-add-fma) placement,
row-scoped [DAZ](../../../../glossary.md#term-daz), optional [FTZ](../../../../glossary.md#term-ftz) of [finite](../../../../glossary.md#term-finite) subnormal arithmetic results, and the completed [Model](../../../../glossary.md#term-model)
final-[publication](../../../../glossary.md#term-publication) rule. Product and intermediate exact-zero signs remain fixed by the selected
[evaluation](../../../../glossary.md#term-evaluation); DAZ/FTZ and each legal FMA/reassociation choice must be modeled explicitly, never
inferred through a [tolerance](../../../../glossary.md#term-tolerance). Only after the complete nonempty otherwise-permitted arithmetic
result is exact zero may publication choose either sign. No implicit [accumulator](../../../../glossary.md#accumulator-type) [identity](../../../../glossary.md#term-identity-meanings) exists,
including for `K=1`. Cover the prior `+0.0f * -1.0f` witness explicitly: the oracle must retain its
negative-zero product and admit [positive zero](../../../../glossary.md#nan-infinity-and-signed-zero) only at publication, never as a product or
intermediate result.

Cover signed zeros, minimum/ordinary/maximum subnormals, minimum normals and neighbors, ordinary
and maximum finite values, [overflow](../../../../glossary.md#term-overflow), [infinities](../../../../glossary.md#nan-infinity-and-signed-zero), signed quiet/signaling NaNs, multiplication by
one, cancellation, and tree-sensitive permutations. [Mantissa](../../../../glossary.md#floating-point-precision-and-rounding) and per-term sentinels must reject
reduced [precision](../../../../glossary.md#term-precision) or any omitted/duplicated term. Require raw-bit membership for finite exact sets
and only current Model representation freedom within an unchanged [NaN](../../../../glossary.md#term-nan) classification. Any
reduced-precision-none, DAZ/FTZ, FMA/reassociation, control, binding, Shape, [transpose](../../../../glossary.md#term-transpose), full-term,
precision, nonzero/classification, or oracle failure blocks without production edits.

## Non-goals

No strict [MATMUL](../../../../glossary.md#matrix-multiplication--matmul); [vector](../../../../glossary.md#term-vector), batched/broadcast, zero/dynamic-[extent](../../../../glossary.md#term-extent), other-type, promoted, arbitrary-
[view](../../../../glossary.md#term-view-meanings), [convolution](../../../../glossary.md#term-channels-first-convolution-layers), broader backward/training, packaging/discovery, performance, or mixed-[owner](../../../../glossary.md#term-owner)
claim. No per-product/intermediate zero-sign choice, term [loss](../../../../glossary.md#loss), [identity](../../../../glossary.md#term-identity-meanings) insertion, [tolerance](../../../../glossary.md#term-tolerance),
reduced [precision](../../../../glossary.md#term-precision), algebraic substitution, special-value repair, cross-[node](../../../../glossary.md#node) [FMA](../../../../glossary.md#term-fused-multiply-add-fma), fallback, [Runtime](../../../../glossary.md#term-runtime)/
[Trace](../../../../glossary.md#trace) profile state, public [API](../../../../glossary.md#term-application-programming-interface-api) [signature](../../../../glossary.md#term-signature), [dependency](../../../../glossary.md#term-dependency), or [ABI](../../../../glossary.md#term-application-binary-interface-abi)-export change.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants)
- [Foundational modules — Numerical profiles](../../../../architecture/contracts/foundational-modules.md#numerical-profiles)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [automatic differentiation glossary](../../../../glossary.md#autograd))</span>
- [Backend execution — Metal backend](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal))</span>
- [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [lifecycle glossary](../../../../glossary.md#lifecycle))</span> and
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>

If implementation needs wider [Model](../../../../glossary.md#term-model) semantics, reduced [precision](../../../../glossary.md#term-precision), term [loss](../../../../glossary.md#loss), a shared
representation change, new export, wider transfer, [host](../../../../glossary.md#term-host) fallback, or [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) profile state,
stop and report the conflict.

## Dependencies and integration

- Depends on: Model 0029 after Complete; Metal 0020; Model 0027; Config 0006; Engine 0018; not Metal
  0009/0016/0017/0018<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Metal glossary](../../../../glossary.md#term-metal); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Conflicts with: any Metal 0016 restart and every Metal capability/preparation/native schema/
  candidate/codec/materialization/public Engine scope<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [codec glossary](../../../../glossary.md#term-codec); [materialization glossary](../../../../glossary.md#materialization); [scope glossary](../../../../glossary.md#term-scope))</span>
- Parallel group: None
- Common base revision: `308267837a17c2fdbeb8b90ad381d1f899980786` plus this planning-only
  authorization<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Integration order: Serial after Model 0029; full fresh oracle before production edits; then
  schema/identity, capability/topology, lifecycle, Engine proof, documentation, independent review<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [identity glossary](../../../../glossary.md#term-identity-meanings); [topology glossary](../../../../glossary.md#term-topology); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Integration [validation](../../../../glossary.md#term-validation): refined [accelerator](../../../../glossary.md#term-accelerator) full-[contraction](../../../../glossary.md#contraction-dimension) [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) Class C [checkpoint](../../../../glossary.md#term-checkpoint)
- Shared-document integration [owner](../../../../glossary.md#term-owner): task implementer, finalized by independent Class C review

## Files and symbols

- `MetalCapabilityProvider`, `MetalNegPartitionPreparer`, `MetalNegPreparationPlan` — exact profile,
  Shape/state/topology domain and lowering.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [topology glossary](../../../../glossary.md#term-topology); [lowering glossary](../../../../glossary.md#lowering))</span>
- `MetalMpsGraphProgram`, `MetalNativeApi`, `MetalDeviceContext`, native foundation — schema eight,
  wire 15, validation, ABI stability, and real execution.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Metal candidate/tuning/codec/decision and prepared lifecycle types — version-nine identity,
  authentication, stale rejection, resources, reuse, and cleanup.<span data-glossary-navigation="true"> ([codec glossary](../../../../glossary.md#term-codec); [lifecycle glossary](../../../../glossary.md#lifecycle); [identity glossary](../../../../glossary.md#term-identity-meanings); [authentication glossary](../../../../glossary.md#term-authentication))</span>
- Focused capability/[schema](../../../../glossary.md#term-schema)/raw-native/prepared/candidate tests; [Metal](../../../../glossary.md#term-metal) [conformance](../../../../glossary.md#term-conformance);
  `GradientRulesTest`; `EngineExplicitCompositionMetalIntegrationTest`.
- Affected Javadocs, Metal/native/backend/API/user guides, targeted glossary, backend-execution
  current capability, and synchronized planning after executable behavior passes.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [planning glossary](../../../../glossary.md#term-planning))</span>

## Acceptance criteria

1. A completely fresh full real-M3 oracle starts from the beginning, passes every numerical/
   control/topology cell before edits, reports exact totals, proves the prior K=1 witness only
   through final-publication freedom, and is removed.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [topology glossary](../../../../glossary.md#term-topology); [publication glossary](../../../../glossary.md#term-publication))</span>
2. Strict stays false; accelerator admits exactly positive static same-type FLOAT32 rank-two MATMUL
   with authenticated local transpose inputs; all excluded forms fail closed.<span data-glossary-navigation="true"> ([accelerator glossary](../../../../glossary.md#term-accelerator); [FLOAT32 glossary](../../../../glossary.md#term-float32); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [transpose glossary](../../../../glossary.md#term-transpose); [fail closed glossary](../../../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
3. Every output uses every term exactly once and belongs to the refined Model set; products and
   intermediates receive no new sign freedom, reduced precision, tolerance, or repair.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [precision glossary](../../../../glossary.md#term-precision); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
4. ABI v4 retains thirteen exports; schema eight appends only wire 15; version-nine identities
   reject old, cross-profile, malformed, corrupt, stale, changed-Shape, and foreign data.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [schema glossary](../../../../glossary.md#term-schema); [Shape glossary](../../../../glossary.md#shape))</span>
5. Native/Java lifecycle proof covers targets, inputs, transpose forms, reuse, concurrency,
   sessions/contexts, rollback, cleanup, and close rejection.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [targets glossary](../../../../glossary.md#term-target); [concurrency glossary](../../../../glossary.md#term-concurrency); [rollback glossary](../../../../glossary.md#term-rollback))</span>
6. A real CPU-free Engine executes oracle-checked direct/linear forward and seeded gradients for
   both operands; graph inspection proves the planned rank-two PERMUTE/MATMUL formulas beneath the
   existing shape-restoration boundaries.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine); [linear glossary](../../../../glossary.md#term-linear-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [graph glossary](../../../../glossary.md#graph))</span>
7. Focused/native/conformance/Compiler/Engine/architecture/full-build/documentation/diff checks pass
   and independent Class C review has no unresolved finding.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [Compiler glossary](../../../../glossary.md#term-compiler))</span>

## [Validation](../../../../glossary.md#term-validation)

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

Record [oracle](../../../../glossary.md#term-test-oracle) totals/cardinalities/term [masks](../../../../glossary.md#term-boolean-mask), probe removal, exports, [schema](../../../../glossary.md#term-schema)/[identity](../../../../glossary.md#term-identity-meanings) versions,
test counts/skips, strict/profile negatives, changed paths, and status/frontier checks. The worker
owns one full [checkpoint](../../../../glossary.md#term-checkpoint); review reruns executable checks only after relevant executable changes or
when evidence is concretely stale.

## Documentation and review impact

Finalize affected Javadocs and guides only after executable behavior passes. Independent Class C
review inspects the entire fresh [oracle](../../../../glossary.md#term-test-oracle), exact-zero [publication](../../../../glossary.md#term-publication) boundary, all-term/reduced-[precision](../../../../glossary.md#term-precision)
controls, strict-false [matrix](../../../../glossary.md#term-matrix-meanings), [transpose](../../../../glossary.md#term-transpose) [authentication](../../../../glossary.md#term-authentication), [schema](../../../../glossary.md#term-schema)/[cache](../../../../glossary.md#term-cache) [identity](../../../../glossary.md#term-identity-meanings), [lifecycle](../../../../glossary.md#lifecycle)/cleanup,
real seeded [Engine](../../../../glossary.md#term-engine) proof, [Runtime](../../../../glossary.md#term-runtime)/[Trace](../../../../glossary.md#trace) absence, changed [scope](../../../../glossary.md#term-scope), and probe removal.

## Result

Implementation, worker [validation](../../../../glossary.md#term-validation), evidence remediation, and independent Class C review are
complete. Before production edits, a fresh disposable [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) probe passed:

```text
PASS device=Apple M3 Max smoke_executables=1 smoke_runs=1 full_executables=384
full_runs=3072 oracle_cells=18441 oracle_normalizations=30729
oracle_raw_operations=4334198 oracle_members=31498 term_mask=0x1f
k1_product=-0 k1_publication=both opt=0,1 contexts=2 forms=4
reduced_precision=none
```

The [matrix](../../../../glossary.md#term-matrix-meanings) covered direct, left-transposed, right-transposed, and both-transposed forms; two
optimization levels and two contexts; three independently compiled executables per full cell and
eight runs per executable; [Shape](../../../../glossary.md#shape)/data, stable/permuted [binding](../../../../glossary.md#term-binding), guarded-[target](../../../../glossary.md#term-target), [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz),
[reassociation](../../../../glossary.md#term-reassociation)/[FMA](../../../../glossary.md#term-fused-multiply-add-fma), exact-zero [publication](../../../../glossary.md#term-publication), all-term, and reduced-[precision](../../../../glossary.md#term-precision) controls. The prior
`K=1` witness retained product `-0` and passed only through [Model](../../../../glossary.md#term-model) 0029's final-publication
either-zero-sign freedom. The source and [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) were removed before production edits.

The implementation [checkpoint](../../../../glossary.md#term-checkpoint) evidence later audited and accepted by independent review was:

```text
Metal focused tests: 98/98 passed, 0 skipped
GradientRules pattern: 41/41 passed, 0 skipped
Metal conformance: 5/5 passed, 0 skipped
Environment-enabled real Metal integration: 10/10 passed, 0 skipped
Architecture checks: 9/9 passed, 0 skipped
Full ./gradlew build: 3,496 total; 3,455 passed, 41 skipped, 0 failed/errors
  ordinary full-build skips: CPU 28 + integration 13
Aggregate executable checkpoint: 3,659 cases; 3,618 passed, 41 skipped, 0 failures/errors
Native build: passed
nm: exactly ABI v4 with 13 exports
Javadocs and full ./gradlew build: passed
```

The 10/10 real [Metal](../../../../glossary.md#term-metal) integration result above is the task-scoped,
environment-enabled checkpoint. It is distinct from the optional suites skipped by the
ordinary full build, whose 41 skips were [CPU](../../../../glossary.md#term-cpu) 28 plus integration 13. Fourteen of the 15 changed
Markdown files passed the ordinary validator. The 15th, `docs/api/tensor-api.md`, passed a
baseline-aware comparison with exactly its identical 172 pre-existing duplicate-heading
diagnostics and zero new or nonduplicate diagnostics; this accounts for all 15 changed Markdown
files. `git diff --check` and whitespace checks passed. Exactly 38 task-owned paths changed; the
disposable probe source and binary were removed. After the evidence-remediation [commit](../../../../glossary.md#term-commit), the
worktree was clean.

The implementation adds the exact [accelerator](../../../../glossary.md#term-accelerator)-only capability, local [transpose](../../../../glossary.md#term-transpose) [authentication](../../../../glossary.md#term-authentication),
[schema](../../../../glossary.md#term-schema)-eight wire `15`, Java/native [fail-closed](../../../../glossary.md#term-fail-closed) geometry and [topology](../../../../glossary.md#term-topology) validation, direct [MPSGraph](../../../../glossary.md#term-mpsgraph)
[matrix multiplication](../../../../glossary.md#matrix-multiplication--matmul) with reduced-precision-none readback, version-nine [identities](../../../../glossary.md#term-identity-meanings), reusable
[prepared execution](../../../../glossary.md#prepared-execution--preparedexecution), and CPU-free direct/[linear](../../../../glossary.md#term-linear-meanings)/seeded-[gradient](../../../../glossary.md#term-gradient) [Engine](../../../../glossary.md#term-engine) coverage. The seeded [graph](../../../../glossary.md#graph)
inspection proves `seed @ transpose(right)` and `transpose(left) @ seed` beneath the existing
shape-restoration boundaries; real runs publish raw-bit-checked forward and both operand gradients
with no CPU [owner](../../../../glossary.md#term-owner). Strict [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) remains false. No [Runtime](../../../../glossary.md#term-runtime), [Trace](../../../../glossary.md#trace), public [API](../../../../glossary.md#term-application-programming-interface-api), [ABI](../../../../glossary.md#term-application-binary-interface-abi) export,
packaging, discovery, [scalar](../../../../glossary.md#term-scalar)-[loss](../../../../glossary.md#loss) training, batched MATMUL, or broader backward [scope](../../../../glossary.md#term-scope) was added.

The final independent Class C review first identified missing worker-validation evidence, accepted
its task-brief-only correction at `be5543f9`, then identified the Markdown count wording defect.
Commit `5631d51f` corrected only that Task 0021 evidence: 14 changed Markdown files passed the
ordinary validator and `docs/api/tensor-api.md` passed the baseline-aware comparison with the same
172 pre-existing duplicate-heading diagnostics and no new or nonduplicate diagnostics. Re-review
returned `APPROVE` with zero findings. The complete chain is implementation `ef2c6a1a`, evidence
remediation `be5543f9`, final evidence correction `5631d51f`, and independent Class C approval.

Status: Complete
